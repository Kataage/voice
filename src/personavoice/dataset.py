from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 1


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def initialize(path: Path) -> None:
    with connect(path) as con:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS utterances (
              id TEXT PRIMARY KEY,
              source_id TEXT NOT NULL,
              start REAL NOT NULL,
              end REAL NOT NULL,
              speaker TEXT,
              target INTEGER NOT NULL,
              speaker_similarity REAL,
              speaker_coverage REAL,
              overlap_ratio REAL NOT NULL,
              text TEXT NOT NULL,
              text_annotated TEXT NOT NULL,
              emotion TEXT,
              events_json TEXT NOT NULL,
              caption TEXT NOT NULL,
              audio_path TEXT,
              quality REAL NOT NULL,
              raw_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_utterances_source_start ON utterances(source_id, start);
            CREATE INDEX IF NOT EXISTS idx_utterances_target ON utterances(target, quality DESC);
            """
        )
        con.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )


def replace_utterances(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    initialize(path)
    with connect(path) as con:
        con.execute("DELETE FROM utterances")
        for row in rows:
            con.execute(
                """
                INSERT INTO utterances(
                  id, source_id, start, end, speaker, target, speaker_similarity,
                  speaker_coverage, overlap_ratio, text, text_annotated, emotion,
                  events_json, caption, audio_path, quality, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    row["id"],
                    row["source_id"],
                    row["start"],
                    row["end"],
                    row.get("speaker"),
                    1 if row.get("target") else 0,
                    row.get("speaker_similarity"),
                    row.get("speaker_coverage"),
                    row.get("overlap_ratio", 0.0),
                    row.get("text", ""),
                    row.get("text_annotated", row.get("text", "")),
                    row.get("emotion"),
                    json.dumps(row.get("events", []), ensure_ascii=False),
                    row.get("caption", ""),
                    row.get("audio_path"),
                    row.get("quality", 0.0),
                    json.dumps(row, ensure_ascii=False),
                ),
            )


def load_utterances(path: Path, *, target_only: bool = False) -> list[dict[str, Any]]:
    query = "SELECT raw_json FROM utterances"
    if target_only:
        query += " WHERE target = 1"
    query += " ORDER BY source_id, start"
    with connect(path) as con:
        return [json.loads(row[0]) for row in con.execute(query)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    """Publish complete JSONL exports atomically.

    A killed prepare process must not leave a half-written training dataset at
    the canonical output path. The temporary file lives beside the destination
    so os.replace/Path.replace remains atomic on the same filesystem.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    count = 0
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        return count
    finally:
        temp.unlink(missing_ok=True)


def export_irodori(master_db: Path, output: Path, persona_name: str) -> int:
    rows = []
    for item in load_utterances(master_db, target_only=True):
        if not item.get("audio_path") or not item.get("text_annotated"):
            continue
        if item.get("quality", 0.0) < 0.45:
            continue
        rows.append(
            {
                "audio": str(Path(item["audio_path"]).resolve()),
                "text": item["text_annotated"],
                "caption": item.get("caption") or "自然に話している。",
                "speaker": persona_name,
                "utterance_id": item["id"],
            }
        )
    return write_jsonl(output, rows)


def export_seed_vc(master_db: Path, output_dir: Path) -> int:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    for existing in audio_dir.glob("*"):
        if existing.is_file():
            existing.unlink()
    manifest = []
    for item in load_utterances(master_db, target_only=True):
        if not item.get("audio_path") or item.get("quality", 0.0) < 0.55:
            continue
        source = Path(item["audio_path"])
        destination = audio_dir / f"{item['id']}{source.suffix}"
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        manifest.append({"audio": str(destination.resolve()), "speaker": "target"})
    return write_jsonl(output_dir / "manifest.jsonl", manifest)


def _conversation_blocks(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse consecutive segments from the same conversational side.

    Prepare may split one long turn into several clips. Treating each target
    clip as a new assistant response teaches the persona model to answer its own
    immediately preceding speech. Collapsing same-side segments reconstructs a
    cleaner dialogue-turn view while retaining the original acoustic rows for
    TTS training.
    """

    blocks: list[dict[str, Any]] = []
    for row in source_rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        target = bool(row.get("target"))
        speaker = str(row.get("speaker") or "")
        same_side = bool(
            blocks
            and blocks[-1]["target"] == target
            and (target or blocks[-1]["speaker"] == speaker)
        )
        if same_side:
            block = blocks[-1]
            block["text"] += text
            block["end"] = row.get("end", block["end"])
            block["rows"].append(row)
            if target and float(row.get("quality", 0.0)) > float(block["voice_row"].get("quality", 0.0)):
                block["voice_row"] = row
            continue
        blocks.append(
            {
                "target": target,
                "speaker": speaker,
                "text": text,
                "start": row.get("start"),
                "end": row.get("end"),
                "rows": [row],
                "voice_row": row,
            }
        )
    return blocks


def export_lfm(master_db: Path, output: Path, persona_name: str) -> int:
    all_rows = load_utterances(master_db)
    examples: list[dict[str, Any]] = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        by_source.setdefault(row["source_id"], []).append(row)
    system = (
        f"あなたは{persona_name}の会話スタイルを再現するローカル会話モデルです。"
        "返答は必ずJSONのみで、textとvoice.caption、voice.emotion、voice.eventsを返してください。"
    )
    for source_rows in by_source.values():
        blocks = _conversation_blocks(source_rows)
        for index, block in enumerate(blocks):
            if not block["target"] or index == 0:
                continue
            # After collapsing same-side segments, a valid persona response must
            # begin directly after another speaker. This excludes monologues and
            # avoids manufacturing a user prompt from the persona's own speech.
            if blocks[index - 1]["target"]:
                continue
            context = blocks[max(0, index - 4) : index]
            lines = []
            for context_block in context:
                speaker = persona_name if context_block["target"] else "相手"
                lines.append(f"{speaker}: {context_block['text']}")
            if not lines:
                continue
            user = "直前の会話:\n" + "\n".join(lines) + "\nこの続きとして自然に返答してください。"
            voice_row = block["voice_row"]
            answer = {
                "text": block["text"],
                "voice": {
                    "caption": voice_row.get("caption") or "自然に話している。",
                    "emotion": voice_row.get("emotion") or "NEUTRAL",
                    "events": voice_row.get("events") or [],
                },
            }
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                        {
                            "role": "assistant",
                            "content": json.dumps(answer, ensure_ascii=False),
                        },
                    ]
                }
            )
    return write_jsonl(output, examples)
