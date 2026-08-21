from __future__ import annotations

import json
import os
import shutil
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


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
        for index, row in enumerate(source_rows):
            if not row.get("target") or not row.get("text"):
                continue
            context = source_rows[max(0, index - 4) : index]
            if not context:
                continue
            lines = []
            for context_row in context:
                speaker = (
                    persona_name
                    if context_row.get("target")
                    else (context_row.get("speaker") or "相手")
                )
                if context_row.get("text"):
                    lines.append(f"{speaker}: {context_row['text']}")
            if not lines:
                continue
            user = "直前の会話:\n" + "\n".join(lines) + "\nこの続きとして自然に返答してください。"
            answer = {
                "text": row["text"],
                "voice": {
                    "caption": row.get("caption") or "自然に話している。",
                    "emotion": row.get("emotion") or "NEUTRAL",
                    "events": row.get("events") or [],
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
