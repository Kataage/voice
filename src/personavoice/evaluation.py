from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from personavoice.config import PersonaConfig
from personavoice.inference import synthesize
from personavoice.project import PersonaPaths
from personavoice.speaker import cosine_similarity, mean_embedding
from personavoice.workers import worker

CASES = [
    {"id": "neutral", "text": "今日はいい天気だね。", "emotion": "NEUTRAL"},
    {"id": "happy", "text": "やった、すごく嬉しい！", "emotion": "HAPPY"},
    {"id": "sad", "text": "そっか、それはちょっと悲しいね。", "emotion": "SAD"},
    {"id": "surprised", "text": "えっ、本当に？びっくりした。", "emotion": "SURPRISED"},
    {"id": "angry", "text": "もう、いい加減にしてよ。", "emotion": "ANGRY"},
]


def _normalize(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


def _successful(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output = {}
    errors = []
    for row in rows:
        item_id = str(row.get("id"))
        if row.get("ok"):
            output[item_id] = row.get("result") or {}
        else:
            errors.append(f"{item_id}: {row.get('error') or 'unknown error'}")
    if errors:
        raise RuntimeError(f"{label} failed:\n" + "\n".join(errors))
    return output


def _identity(repo_root: Path, paths: PersonaPaths) -> list[float] | None:
    refs = [
        path
        for path in paths.identity.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    ][:5]
    if not refs:
        return None
    response = worker(repo_root, "diarization").call(
        repo_root,
        "batch",
        {
            "embeddings": [
                {"id": str(index), "audio": str(path.resolve())}
                for index, path in enumerate(refs)
            ],
            "diarizations": [],
        },
    )
    results = _successful(response.get("embeddings") or [], label="identity embedding")
    embeddings = [
        result["embedding"]
        for result in results.values()
        if result.get("embedding")
    ]
    return mean_embedding(embeddings) if embeddings else None


def evaluate(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> dict:
    report_dir = paths.outputs / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    target_embedding = _identity(repo_root, paths)

    generated_by_id: dict[str, Path] = {}
    for case in CASES:
        output = report_dir / f"{case['id']}.wav"
        generated_by_id[case["id"]] = synthesize(
            repo_root,
            paths,
            cfg,
            case["text"],
            emotion=case["emotion"],
            candidates=1,
            output=output,
        )[0]

    audio_items = [
        {"id": case["id"], "audio": str(generated_by_id[case["id"]].resolve())}
        for case in CASES
    ]
    asr_response = worker(repo_root, "asr").call(
        repo_root,
        "batch_transcribe",
        {
            "items": audio_items,
            "model": cfg.prepare.asr_model,
            "compute_type": cfg.prepare.asr_compute_type,
            "language": cfg.language,
        },
    )
    asr_by_id = _successful(asr_response.get("results") or [], label="evaluation ASR")

    sense_response = worker(repo_root, "sense").call(
        repo_root,
        "batch_analyze",
        {"items": audio_items, "language": cfg.language},
    )
    sense_by_id = _successful(
        sense_response.get("results") or [],
        label="evaluation SenseVoice",
    )

    diar_response = worker(repo_root, "diarization").call(
        repo_root,
        "batch",
        {"embeddings": audio_items, "diarizations": []},
    )
    embedding_by_id = _successful(
        diar_response.get("embeddings") or [],
        label="evaluation speaker embedding",
    )

    rows = []
    for case in CASES:
        case_id = case["id"]
        transcript = asr_by_id[case_id]
        actual = "".join(
            str(segment.get("text") or "")
            for segment in transcript.get("segments", [])
        ).strip()
        text_score = SequenceMatcher(
            None,
            _normalize(case["text"]),
            _normalize(actual),
        ).ratio()
        acoustic = sense_by_id[case_id]
        speaker_score = None
        if target_embedding is not None:
            embedding = embedding_by_id[case_id].get("embedding")
            if embedding:
                speaker_score = cosine_similarity(target_embedding, embedding)
        rows.append(
            {
                **case,
                "output": str(generated_by_id[case_id]),
                "transcript": actual,
                "text_similarity": round(text_score, 4),
                "speaker_similarity": (
                    None if speaker_score is None else round(speaker_score, 4)
                ),
                "detected_emotion": acoustic.get("emotion"),
                "detected_events": acoustic.get("events", []),
            }
        )

    summary = {
        "text_similarity_mean": round(
            sum(row["text_similarity"] for row in rows) / len(rows),
            4,
        ),
        "speaker_similarity_mean": None,
        "emotion_accuracy": round(
            sum(row["detected_emotion"] == row["emotion"] for row in rows) / len(rows),
            4,
        ),
    }
    speaker_values = [
        row["speaker_similarity"]
        for row in rows
        if row["speaker_similarity"] is not None
    ]
    if speaker_values:
        summary["speaker_similarity_mean"] = round(
            sum(speaker_values) / len(speaker_values),
            4,
        )
    report = {"summary": summary, "cases": rows}
    (report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
