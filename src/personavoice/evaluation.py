from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

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


def _identity(repo_root: Path, paths: PersonaPaths) -> list[float] | None:
    refs = [path for path in paths.identity.rglob("*") if path.is_file() and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}]
    if not refs:
        return None
    embeddings = []
    diar = worker(repo_root, "diarization")
    for ref in refs[:5]:
        result = diar.call(repo_root, "embed", {"audio": str(ref.resolve())})
        if result.get("embedding"):
            embeddings.append(result["embedding"])
    return mean_embedding(embeddings) if embeddings else None


def evaluate(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> dict:
    report_dir = paths.outputs / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    target_embedding = _identity(repo_root, paths)
    asr = worker(repo_root, "asr")
    sense = worker(repo_root, "sense")
    diar = worker(repo_root, "diarization")
    rows = []
    for case in CASES:
        output = report_dir / f"{case['id']}.wav"
        generated = synthesize(
            repo_root, paths, cfg, case["text"], emotion=case["emotion"], candidates=1, output=output
        )[0]
        transcript = asr.call(
            repo_root,
            "transcribe",
            {"audio": str(generated.resolve()), "model": cfg.prepare.asr_model, "language": cfg.language},
        )
        actual = "".join(seg.get("text", "") for seg in transcript.get("segments", []))
        text_score = SequenceMatcher(None, _normalize(case["text"]), _normalize(actual)).ratio()
        acoustic = sense.call(repo_root, "analyze", {"audio": str(generated.resolve()), "language": cfg.language})
        speaker_score = None
        if target_embedding is not None:
            emb = diar.call(repo_root, "embed", {"audio": str(generated.resolve())}).get("embedding")
            if emb:
                speaker_score = cosine_similarity(target_embedding, emb)
        rows.append(
            {
                **case,
                "output": str(generated),
                "transcript": actual,
                "text_similarity": round(text_score, 4),
                "speaker_similarity": None if speaker_score is None else round(speaker_score, 4),
                "detected_emotion": acoustic.get("emotion"),
                "detected_events": acoustic.get("events", []),
            }
        )
    summary = {
        "text_similarity_mean": round(sum(r["text_similarity"] for r in rows) / len(rows), 4),
        "speaker_similarity_mean": None,
        "emotion_accuracy": round(sum(r["detected_emotion"] == r["emotion"] for r in rows) / len(rows), 4),
    }
    speaker_values = [r["speaker_similarity"] for r in rows if r["speaker_similarity"] is not None]
    if speaker_values:
        summary["speaker_similarity_mean"] = round(sum(speaker_values) / len(speaker_values), 4)
    report = {"summary": summary, "cases": rows}
    (report_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
