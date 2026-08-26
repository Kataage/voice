from __future__ import annotations

import json
import math
import re
import wave
from pathlib import Path
from typing import Any

from personavoice.artifacts import PublicationItem, publish_training_candidates
from personavoice.atomic import atomic_write_json
from personavoice.config import PersonaConfig
from personavoice.evaluation_metrics import (
    EvaluationSample,
    aggregate_evaluation_metrics,
    character_error_rate,
    duration_ratio_error,
    normalize_text,
    unseen_pronunciation_score,
    word_error_rate,
)
from personavoice.inference import synthesize
from personavoice.pipeline import _prepare_fingerprint
from personavoice.project import PersonaPaths
from personavoice.quality import evaluate_quality_gate
from personavoice.speaker import cosine_similarity, mean_embedding
from personavoice.state import StateStore
from personavoice.training import _fingerprint
from personavoice.training_inputs import ensure_irodori_manifest
from personavoice.training_plan import TrainingPlan, build_training_plan
from personavoice.workers import worker

# These prompts are versioned implementation inputs. Every publication checks
# them against the prepared corpus so an overlap cannot become a training score.
CASES = (
    {
        "id": "neutral-unseen",
        "text": "量子暗号の鍵配送について、ゆっくり説明するね。",
        "emotion": "NEUTRAL",
    },
    {
        "id": "happy-unseen",
        "text": "新しい星座を見つけたみたいで、胸がわくわくする！",
        "emotion": "HAPPY",
    },
    {
        "id": "sad-unseen",
        "text": "遠い港の汽笛を聞くと、少しだけ寂しくなるね。",
        "emotion": "SAD",
    },
    {
        "id": "surprised-unseen",
        "text": "えっ、氷の下にそんな大きな湖があるの？",
        "emotion": "SURPRISED",
    },
    {
        "id": "angry-unseen",
        "text": "約束した手順を勝手に変えるのは、絶対にだめだよ。",
        "emotion": "ANGRY",
    },
)

LFM_CASES = (
    {
        "id": "lfm-explain",
        "prompt": "7と5を足した答えを、数字を含む短い一文で教えて。",
        "expected_completion": "7と5を足すと12だよ。",
        "required_phrases": ("12",),
    },
    {
        "id": "lfm-empathy",
        "prompt": "今日は少し疲れたよ。「ゆっくり休んで」を含む短い一文で返して。",
        "expected_completion": "無理せず、今日はゆっくり休んでね。",
        "required_phrases": ("ゆっくり休んで",),
    },
)
_LFM_DIALOGUE_HEADER = "直前の会話:"
_LFM_DIALOGUE_SUFFIX = "この続きとして自然に返答してください。"


def _successful(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    errors = []
    for row in rows:
        item_id = str(row.get("id"))
        if row.get("ok"):
            result = row.get("result")
            output[item_id] = result if isinstance(result, dict) else {}
        else:
            errors.append(f"{item_id}: {row.get('error') or 'unknown error'}")
    if errors:
        raise RuntimeError(f"{label} failed:\n" + "\n".join(errors))
    return output


def _best_effort_embeddings(
    repo_root: Path,
    items: list[dict[str, str]],
    *,
    label: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Collect independent pyannote embeddings while failing quality closed.

    Invalid/non-finite worker outputs stay rejected by the worker
    contract, but one bad sample must not erase unrelated ASR,
    pronunciation, duration or emotion metrics. Missing speaker
    metrics remain None so publication still fails closed.
    """

    diarization = worker(repo_root, "diarization")
    results: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for item in items:
        item_id = str(item.get("id") or "")
        audio = item.get("audio")
        if not item_id or not isinstance(audio, str) or not audio:
            if item_id:
                errors[item_id] = f"{label}: invalid-request"
            continue
        try:
            value = diarization.call(repo_root, "embed", {"audio": audio})
        except Exception as exc:
            reason = (
                "invalid-response-schema"
                if "invalid response schema" in str(exc).casefold()
                else type(exc).__name__
            )
            errors[item_id] = f"{label}: {reason}"
            continue
        embedding = value.get("embedding") if isinstance(value, dict) else None
        if not isinstance(embedding, list) or not embedding:
            errors[item_id] = f"{label}: empty-embedding"
            continue
        results[item_id] = value
    return results, errors


def _identity(
    repo_root: Path,
    paths: PersonaPaths,
) -> tuple[list[float] | None, dict[str, str]]:
    refs = [
        path
        for path in paths.identity.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}
    ][:5]
    if not refs:
        return None, {}
    results, errors = _best_effort_embeddings(
        repo_root,
        [
            {"id": str(index), "audio": str(path.resolve())}
            for index, path in enumerate(refs)
        ],
        label="identity",
    )
    embeddings = [
        result["embedding"] for result in results.values() if result.get("embedding")
    ]
    return (mean_embedding(embeddings) if embeddings else None), errors

def _transcript(value: dict[str, Any]) -> str:
    return "".join(
        str(segment.get("text") or "")
        for segment in value.get("segments", [])
        if isinstance(segment, dict)
    ).strip()


def _wav_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            rate = handle.getframerate()
            frames = handle.getnframes()
    except (OSError, EOFError, wave.Error) as exc:
        raise RuntimeError(f"Evaluation output is not a readable WAV: {path}") from exc
    if rate <= 0 or frames <= 0:
        raise RuntimeError(f"Evaluation output has no measurable duration: {path}")
    return frames / rate


def _persona_relative(paths: PersonaPaths, path: Path) -> str:
    try:
        return path.resolve().relative_to(paths.root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Evaluation path escaped the persona root: {path}") from exc


def _training_texts(paths: PersonaPaths) -> set[str]:
    values: set[str] = set()
    source = paths.dataset / "irodori_source.jsonl"
    if not source.is_file():
        return values
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Prepared Irodori source line {line_number} is unreadable"
                ) from exc
            text = row.get("text") if isinstance(row, dict) else None
            if isinstance(text, str) and text.strip():
                values.add(normalize_text(text, remove_whitespace=True))
    return values


def _lfm_user_utterances(content: str, *, line_number: int) -> tuple[str, ...]:
    """Extract the actual dialogue turns from the exporter-owned user wrapper."""

    lines = [line.strip() for line in content.replace("\r\n", "\n").split("\n") if line.strip()]
    if not lines or lines[0] != _LFM_DIALOGUE_HEADER:
        if _LFM_DIALOGUE_HEADER in content or _LFM_DIALOGUE_SUFFIX in content:
            raise RuntimeError(
                f"Prepared LFM dataset line {line_number} has a malformed dialogue wrapper"
            )
        return (content,)
    if len(lines) < 3 or lines[-1] != _LFM_DIALOGUE_SUFFIX:
        raise RuntimeError(
            f"Prepared LFM dataset line {line_number} has a malformed dialogue wrapper"
        )
    utterances: list[str] = []
    for dialogue_line in lines[1:-1]:
        speaker, separator, utterance = dialogue_line.partition(":")
        if not separator or not speaker.strip() or not utterance.strip():
            raise RuntimeError(
                f"Prepared LFM dataset line {line_number} has a malformed dialogue turn"
            )
        utterances.append(utterance.strip())
    if not utterances:
        raise RuntimeError(f"Prepared LFM dataset line {line_number} has an empty dialogue wrapper")
    return tuple(utterances)


def _lfm_assistant_utterance(content: str, *, line_number: int) -> str:
    """Extract the exported JSON answer while retaining plain legacy completions."""

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        if content.lstrip().startswith(("{", "[")):
            raise RuntimeError(
                f"Prepared LFM dataset line {line_number} has an unreadable assistant completion"
            ) from exc
        return content
    if isinstance(parsed, str) and parsed.strip():
        return parsed
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Prepared LFM dataset line {line_number} has an invalid assistant completion"
        )
    text = parsed.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(
            f"Prepared LFM dataset line {line_number} has no assistant completion text"
        )
    return text


def _lfm_training_utterances(paths: PersonaPaths) -> set[str]:
    values: set[str] = set()
    dataset = paths.dataset / "lfm_train.jsonl"
    if not dataset.is_file():
        return values
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Prepared LFM dataset line {line_number} is unreadable"
                ) from exc
            prompt = row.get("prompt") if isinstance(row, dict) else None
            if not isinstance(prompt, list) or not prompt:
                raise RuntimeError(
                    f"Prepared LFM dataset line {line_number} has no prompt messages"
                )
            user_found = False
            for message in prompt:
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"Prepared LFM dataset line {line_number} has an invalid prompt"
                    )
                role = message.get("role")
                content = message.get("content")
                if not isinstance(role, str) or not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        f"Prepared LFM dataset line {line_number} has an invalid prompt"
                    )
                if role == "user":
                    user_found = True
                    for utterance in _lfm_user_utterances(
                        content,
                        line_number=line_number,
                    ):
                        normalized = normalize_text(utterance, remove_whitespace=True)
                        if not normalized:
                            raise RuntimeError(
                                f"Prepared LFM dataset line {line_number} has an empty user utterance"
                            )
                        values.add(normalized)
            if not user_found:
                raise RuntimeError(f"Prepared LFM dataset line {line_number} has no user prompt")
            completion = row.get("completion") if isinstance(row, dict) else None
            if not isinstance(completion, list) or not completion:
                raise RuntimeError(
                    f"Prepared LFM dataset line {line_number} has no completion messages"
                )
            for message in completion:
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"Prepared LFM dataset line {line_number} has an invalid completion"
                    )
                role = message.get("role")
                content = message.get("content")
                if role != "assistant" or not isinstance(content, str) or not content.strip():
                    raise RuntimeError(
                        f"Prepared LFM dataset line {line_number} has an invalid completion"
                    )
                normalized = normalize_text(
                    _lfm_assistant_utterance(content, line_number=line_number),
                    remove_whitespace=True,
                )
                if not normalized:
                    raise RuntimeError(
                        f"Prepared LFM dataset line {line_number} has an empty assistant utterance"
                    )
                values.add(normalized)
    return values


def _assert_held_out(paths: PersonaPaths, *, include_lfm: bool = True) -> None:
    training = _training_texts(paths)
    collisions = [
        f"irodori:{case['id']}"
        for case in CASES
        if normalize_text(case["text"], remove_whitespace=True) in training
    ]
    if include_lfm:
        lfm_utterances = _lfm_training_utterances(paths)
        for case in LFM_CASES:
            if normalize_text(case["prompt"], remove_whitespace=True) in lfm_utterances:
                collisions.append(f"lfm:{case['id']}:prompt")
            if (
                normalize_text(case["expected_completion"], remove_whitespace=True)
                in lfm_utterances
            ):
                collisions.append(f"lfm:{case['id']}:answer")
    if collisions:
        raise RuntimeError(
            "Evaluation prompts overlap the prepared training set; refusing to publish: "
            + ", ".join(collisions)
        )


def _plan_for_evaluation(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
) -> TrainingPlan:
    manifest = (
        ensure_irodori_manifest(
            repo_root,
            paths,
            conditioning=cfg.training.irodori.conditioning,
        )
        if cfg.training.irodori.enabled
        else paths.dataset / "irodori_manifest.jsonl"
    )
    return build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)


def _family_artifact(paths: PersonaPaths, family: dict[str, Any]) -> Path:
    raw = family.get("artifact")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("Training result contains no candidate artifact")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("Training result candidate path is not portable")
    candidate = (paths.root / relative).resolve()
    try:
        candidate.relative_to(paths.root.resolve())
    except ValueError as exc:
        raise RuntimeError("Training result candidate escaped the persona root") from exc
    return candidate


def _generate_voice_sets(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    artifact: Path,
    method: str,
    report_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, dict[str, Path]]]:
    candidate: dict[str, Path] = {}
    baseline: dict[str, Path] = {}
    for case in CASES:
        candidate_reference_mode = (
            "speaker-embed"
            if method == "speaker-inversion"
            else "none"
            if method == "full"
            else None
        )
        candidate[case["id"]] = synthesize(
            repo_root,
            paths,
            cfg,
            case["text"],
            emotion=case["emotion"],
            candidates=1,
            output=report_dir / "candidate" / f"{case['id']}.wav",
            irodori_artifact=artifact,
            irodori_method=method,
            reference_mode=candidate_reference_mode,
        )[0]
        baseline[case["id"]] = synthesize(
            repo_root,
            paths,
            cfg,
            case["text"],
            emotion=case["emotion"],
            candidates=1,
            output=report_dir / "base" / f"{case['id']}.wav",
            base_only=True,
            reference_mode="none" if method == "full" else None,
        )[0]

    mode_options = {
        "speaker-conditioned": {"reference_mode": "auto", "caption": False},
        "no-reference": {"reference_mode": "none", "caption": False},
        "caption-conditioned": {"reference_mode": "none", "caption": True},
    }
    modes: dict[str, dict[str, Path]] = {}
    if method == "full":
        for label, options in mode_options.items():
            mode_outputs: dict[str, Path] = {}
            for case in CASES:
                # The publication-gated full-model sample is already the
                # no-reference + caption path, so reuse those exact bytes in
                # the comparison instead of synthesizing a duplicate.
                if label == "caption-conditioned":
                    mode_outputs[case["id"]] = candidate[case["id"]]
                    continue
                mode_outputs[case["id"]] = synthesize(
                    repo_root,
                    paths,
                    cfg,
                    case["text"],
                    emotion=case["emotion"],
                    candidates=1,
                    output=report_dir / "modes" / label / f"{case['id']}.wav",
                    irodori_artifact=artifact,
                    irodori_method=method,
                    reference_mode=str(options["reference_mode"]),
                    caption_conditioning=bool(options["caption"]),
                )[0]
            modes[label] = mode_outputs
    return candidate, baseline, modes


def _analyze_voice_sets(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    candidate: dict[str, Path],
    baseline: dict[str, Path],
    probes: dict[str, dict[str, Path]],
) -> dict[str, Any]:
    target_embedding, identity_embedding_errors = _identity(repo_root, paths)
    all_audio: dict[str, Path] = {}
    all_audio.update({f"candidate:{key}": value for key, value in candidate.items()})
    all_audio.update({f"base:{key}": value for key, value in baseline.items()})
    all_audio.update(
        {
            f"mode:{mode}:{case_id}": value
            for mode, outputs in probes.items()
            for case_id, value in outputs.items()
        }
    )
    audio_items = [
        {"id": item_id, "audio": str(path.resolve())} for item_id, path in all_audio.items()
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
    asr = _successful(asr_response.get("results") or [], label="evaluation ASR")
    candidate_items = [
        {"id": f"candidate:{key}", "audio": str(value.resolve())}
        for key, value in candidate.items()
    ]
    mode_items = [
        {"id": f"mode:{mode}:{key}", "audio": str(value.resolve())}
        for mode, outputs in probes.items()
        for key, value in outputs.items()
    ]
    analyzed_items = [*candidate_items, *mode_items]
    sense_response = worker(repo_root, "sense").call(
        repo_root,
        "batch_analyze",
        {"items": analyzed_items, "language": cfg.language},
    )
    sense = _successful(
        sense_response.get("results") or [],
        label="evaluation SenseVoice",
    )
    embeddings, speaker_embedding_errors = _best_effort_embeddings(
        repo_root,
        analyzed_items,
        label="evaluation",
    )

    samples: list[EvaluationSample] = []
    rows: list[dict[str, Any]] = []
    baseline_cers: list[float] = []
    candidate_cers: list[float] = []
    for case in CASES:
        case_id = case["id"]
        candidate_id = f"candidate:{case_id}"
        baseline_id = f"base:{case_id}"
        candidate_text = _transcript(asr[candidate_id])
        baseline_text = _transcript(asr[baseline_id])
        acoustic = sense[candidate_id]
        speaker_score = None
        if target_embedding is not None:
            embedding = embeddings.get(candidate_id, {}).get("embedding")
            if embedding:
                speaker_score = cosine_similarity(target_embedding, embedding)
        baseline_duration = _wav_duration(baseline[case_id])
        candidate_duration = _wav_duration(candidate[case_id])
        samples.append(
            EvaluationSample(
                reference_text=case["text"],
                hypothesis_text=candidate_text,
                speaker_similarity=speaker_score,
                reference_duration_seconds=baseline_duration,
                generated_duration_seconds=candidate_duration,
                expected_emotion=case["emotion"],
                detected_emotion=acoustic.get("emotion"),
                unseen=True,
            )
        )
        baseline_cer = character_error_rate(case["text"], baseline_text)
        candidate_cer = character_error_rate(case["text"], candidate_text)
        baseline_cers.append(baseline_cer)
        candidate_cers.append(candidate_cer)
        rows.append(
            {
                **case,
                "output": _persona_relative(paths, candidate[case_id]),
                "base_output": _persona_relative(paths, baseline[case_id]),
                "transcript": candidate_text,
                "base_transcript": baseline_text,
                "cer": candidate_cer,
                "wer": word_error_rate(case["text"], candidate_text),
                "unseen_text_similarity": unseen_pronunciation_score(case["text"], candidate_text),
                "duration_ratio_error": duration_ratio_error(
                    baseline_duration,
                    candidate_duration,
                ),
                "speaker_similarity": speaker_score,
                "speaker_embedding_error": speaker_embedding_errors.get(candidate_id),
                "detected_emotion": acoustic.get("emotion"),
                "detected_events": acoustic.get("events", []),
            }
        )
    baseline_mean = math.fsum(baseline_cers) / len(baseline_cers)
    candidate_mean = math.fsum(candidate_cers) / len(candidate_cers)
    metrics = aggregate_evaluation_metrics(
        samples,
        baseline_base_cer=baseline_mean,
        candidate_base_cer=candidate_mean,
    )
    mode_rows: list[dict[str, Any]] = []
    for label, outputs in probes.items():
        mode_samples: list[EvaluationSample] = []
        mode_cases: list[dict[str, Any]] = []
        mode_candidate_cers: list[float] = []
        for case in CASES:
            case_id = case["id"]
            item_id = f"mode:{label}:{case_id}"
            transcript = _transcript(asr[item_id])
            acoustic = sense[item_id]
            speaker_score = None
            if target_embedding is not None:
                embedding = embeddings.get(item_id, {}).get("embedding")
                if embedding:
                    speaker_score = cosine_similarity(target_embedding, embedding)
            baseline_duration = _wav_duration(baseline[case_id])
            generated_duration = _wav_duration(outputs[case_id])
            mode_samples.append(
                EvaluationSample(
                    reference_text=case["text"],
                    hypothesis_text=transcript,
                    speaker_similarity=speaker_score,
                    reference_duration_seconds=baseline_duration,
                    generated_duration_seconds=generated_duration,
                    expected_emotion=case["emotion"],
                    detected_emotion=acoustic.get("emotion"),
                    unseen=True,
                )
            )
            cer = character_error_rate(case["text"], transcript)
            mode_candidate_cers.append(cer)
            mode_cases.append(
                {
                    **case,
                    "output": _persona_relative(paths, outputs[case_id]),
                    "transcript": transcript,
                    "cer": cer,
                    "wer": word_error_rate(case["text"], transcript),
                    "unseen_text_similarity": unseen_pronunciation_score(case["text"], transcript),
                    "duration_ratio_error": duration_ratio_error(
                        baseline_duration,
                        generated_duration,
                    ),
                    "speaker_similarity": speaker_score,
                    "speaker_embedding_error": speaker_embedding_errors.get(item_id),
                    "detected_emotion": acoustic.get("emotion"),
                    "detected_events": acoustic.get("events", []),
                }
            )
        mode_candidate_mean = math.fsum(mode_candidate_cers) / len(mode_candidate_cers)
        mode_metrics = aggregate_evaluation_metrics(
            mode_samples,
            baseline_base_cer=baseline_mean,
            candidate_base_cer=mode_candidate_mean,
        )
        mode_rows.append(
            {
                "mode": label,
                **mode_metrics.as_dict(),
                "base_cer_mean": baseline_mean,
                "candidate_cer_mean": mode_candidate_mean,
                "cases": mode_cases,
            }
        )
    return {
        **metrics.as_dict(),
        "cases": rows,
        "mode_comparison": mode_rows,
        "base_cer_mean": baseline_mean,
        "candidate_cer_mean": candidate_mean,
        "speaker_embedding_diagnostics": {
            "identity": identity_embedding_errors,
            "generated": speaker_embedding_errors,
        },
    }


def _parse_lfm_output(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    text = parsed.get("text")
    voice = parsed.get("voice")
    if not isinstance(text, str) or not text.strip() or not isinstance(voice, dict):
        return None
    caption = voice.get("caption")
    emotion = voice.get("emotion")
    events = voice.get("events")
    if (
        not isinstance(caption, str)
        or not caption.strip()
        or not isinstance(emotion, str)
        or not emotion.strip()
        or not isinstance(events, list)
        or any(not isinstance(event, str) for event in events)
    ):
        return None
    return {"text": text.strip(), "voice": voice}


def _lfm_contract_output(value: Any) -> bool:
    return _parse_lfm_output(value) is not None


def _complete_mean(values: list[float | None], *, expected_count: int) -> float | None:
    if len(values) != expected_count or any(
        value is None or not math.isfinite(value) for value in values
    ):
        return None
    return math.fsum(value for value in values if value is not None) / expected_count


def _required_phrase_coverage(text: str, phrases: tuple[str, ...]) -> float:
    if not phrases:
        raise ValueError("LFM held-out case has no required phrases")
    normalized = normalize_text(text, remove_whitespace=True)

    def present(phrase: str) -> bool:
        required = normalize_text(phrase, remove_whitespace=True)
        if not required:
            raise ValueError("LFM held-out required phrase is empty after normalization")
        # Numeric/Latin answers must be complete tokens: the required answer
        # ``12`` must not accept a semantically different ``120``.  Japanese
        # phrases intentionally retain containment semantics so natural suffixes
        # such as 「ゆっくり休んでね」 still satisfy 「ゆっくり休んで」.
        if re.fullmatch(r"[a-z0-9]+", required):
            return (
                re.search(rf"(?<![a-z0-9]){re.escape(required)}(?![a-z0-9])", normalized)
                is not None
            )
        return required in normalized

    matched = sum(present(phrase) for phrase in phrases)
    return matched / len(phrases)


def _evaluate_lfm(
    repo_root: Path,
    cfg: PersonaConfig,
    family: dict[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    method = family.get("method")
    if method not in {"full", "lora"}:
        raise RuntimeError("LFM training result contains an unsupported method")
    system = (
        f"あなたは{cfg.name}として自然に会話します。返答はJSONのみ。"
        '{"text":"...","voice":{"caption":"...","emotion":"NEUTRAL","events":[]}}。'
    )
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    candidate_contracts: list[bool] = []
    baseline_contracts: list[bool] = []
    candidate_similarities: list[float | None] = []
    candidate_cers: list[float | None] = []
    candidate_wers: list[float | None] = []
    baseline_similarities: list[float | None] = []
    baseline_cers: list[float | None] = []
    baseline_wers: list[float | None] = []
    phrase_coverages: list[float | None] = []
    similarity_regressions: list[float | None] = []
    for case in LFM_CASES:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": case["prompt"]},
        ]
        candidate = worker(repo_root, "lfm").call(
            repo_root,
            "infer",
            {
                "messages": messages,
                "full_model": str(artifact) if method == "full" else None,
                "adapter": str(artifact) if method == "lora" else None,
                "temperature": 0.0,
                "max_new_tokens": 192,
            },
        )
        baseline = worker(repo_root, "lfm").call(
            repo_root,
            "infer",
            {
                "messages": messages,
                "full_model": None,
                "adapter": None,
                "temperature": 0.0,
                "max_new_tokens": 192,
            },
        )
        candidate_raw = candidate.get("text") if isinstance(candidate, dict) else None
        baseline_raw = baseline.get("text") if isinstance(baseline, dict) else None
        candidate_output = _parse_lfm_output(candidate_raw)
        baseline_output = _parse_lfm_output(baseline_raw)
        candidate_contract = candidate_output is not None
        baseline_contract = baseline_output is not None
        candidate_contracts.append(candidate_contract)
        baseline_contracts.append(baseline_contract)
        expected = case["expected_completion"]
        required_phrases = tuple(case["required_phrases"])
        candidate_similarity: float | None = None
        candidate_cer: float | None = None
        candidate_wer: float | None = None
        baseline_similarity: float | None = None
        baseline_cer: float | None = None
        baseline_wer: float | None = None
        phrase_coverage: float | None = None
        similarity_regression: float | None = None
        if candidate_output is None:
            errors.append(f"{case['id']}: candidate output contract failed")
        if baseline_output is None:
            errors.append(f"{case['id']}: baseline output contract failed")
        if candidate_output is not None:
            candidate_text = candidate_output["text"]
            try:
                candidate_similarity = unseen_pronunciation_score(expected, candidate_text)
                candidate_cer = character_error_rate(expected, candidate_text)
                candidate_wer = word_error_rate(expected, candidate_text)
                phrase_coverage = _required_phrase_coverage(
                    candidate_text,
                    required_phrases,
                )
            except (TypeError, ValueError, OverflowError):
                errors.append(f"{case['id']}: candidate quality metrics are invalid")
        if baseline_output is not None:
            baseline_text = baseline_output["text"]
            try:
                baseline_similarity = unseen_pronunciation_score(expected, baseline_text)
                baseline_cer = character_error_rate(expected, baseline_text)
                baseline_wer = word_error_rate(expected, baseline_text)
            except (TypeError, ValueError, OverflowError):
                errors.append(f"{case['id']}: baseline quality metrics are invalid")
        if candidate_similarity is not None and baseline_similarity is not None:
            similarity_regression = max(0.0, baseline_similarity - candidate_similarity)
        candidate_similarities.append(candidate_similarity)
        candidate_cers.append(candidate_cer)
        candidate_wers.append(candidate_wer)
        baseline_similarities.append(baseline_similarity)
        baseline_cers.append(baseline_cer)
        baseline_wers.append(baseline_wer)
        phrase_coverages.append(phrase_coverage)
        similarity_regressions.append(similarity_regression)
        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_completion": expected,
                "required_phrases": list(required_phrases),
                "candidate_contract_passed": candidate_contract,
                "baseline_contract_passed": baseline_contract,
                "candidate": candidate_raw,
                "baseline": baseline_raw,
                "candidate_expected_similarity": candidate_similarity,
                "candidate_expected_cer": candidate_cer,
                "candidate_expected_wer": candidate_wer,
                "baseline_expected_similarity": baseline_similarity,
                "baseline_expected_cer": baseline_cer,
                "baseline_expected_wer": baseline_wer,
                "required_phrase_coverage": phrase_coverage,
                "base_similarity_regression": similarity_regression,
            }
        )
    expected_count = len(LFM_CASES)
    candidate_similarity_mean = _complete_mean(
        candidate_similarities,
        expected_count=expected_count,
    )
    candidate_cer_mean = _complete_mean(candidate_cers, expected_count=expected_count)
    candidate_wer_mean = _complete_mean(candidate_wers, expected_count=expected_count)
    baseline_similarity_mean = _complete_mean(
        baseline_similarities,
        expected_count=expected_count,
    )
    baseline_cer_mean = _complete_mean(baseline_cers, expected_count=expected_count)
    baseline_wer_mean = _complete_mean(baseline_wers, expected_count=expected_count)
    phrase_coverage_mean = _complete_mean(
        phrase_coverages,
        expected_count=expected_count,
    )
    regression_mean = _complete_mean(
        similarity_regressions,
        expected_count=expected_count,
    )
    regression_max = (
        max(value for value in similarity_regressions if value is not None)
        if regression_mean is not None
        else None
    )
    contract_passed = len(candidate_contracts) == expected_count and all(candidate_contracts)
    baseline_contract_passed = len(baseline_contracts) == expected_count and all(baseline_contracts)
    metrics = (
        candidate_similarity_mean,
        candidate_cer_mean,
        candidate_wer_mean,
        baseline_similarity_mean,
        baseline_cer_mean,
        baseline_wer_mean,
        phrase_coverage_mean,
        regression_max,
    )
    complete = (
        contract_passed
        and baseline_contract_passed
        and all(value is not None and math.isfinite(value) for value in metrics)
    )
    return {
        "enabled": True,
        "complete": complete,
        "contract_passed": contract_passed,
        "baseline_contract_passed": baseline_contract_passed,
        "candidate_expected_similarity_mean": candidate_similarity_mean,
        "candidate_expected_cer_mean": candidate_cer_mean,
        "candidate_expected_wer_mean": candidate_wer_mean,
        "baseline_expected_similarity_mean": baseline_similarity_mean,
        "baseline_expected_cer_mean": baseline_cer_mean,
        "baseline_expected_wer_mean": baseline_wer_mean,
        "required_phrase_coverage_mean": phrase_coverage_mean,
        "base_similarity_regression_max": regression_max,
        "errors": errors,
        "cases": rows,
    }


def _validation_summary(families: dict[str, Any]) -> dict[str, Any]:
    required: dict[str, float | None] = {}
    for name in ("irodori", "lfm"):
        family = families.get(name)
        if not isinstance(family, dict) or family.get("enabled") is not True:
            continue
        if family.get("method") not in {"full", "lora", "speaker-inversion"}:
            continue
        validation = family.get("validation")
        value = validation.get("loss") if isinstance(validation, dict) else None
        required[name] = (
            float(value)
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            else None
        )
    complete = bool(required) and all(value is not None for value in required.values())
    return {
        "loss": max(value for value in required.values() if value is not None)
        if complete
        else None,
        "complete": complete,
        "families": required,
    }


def _publication_destination(paths: PersonaPaths, family: str, method: str) -> Path:
    destinations = {
        ("irodori", "full"): paths.models / "irodori" / "full",
        ("irodori", "lora"): paths.models / "irodori" / "lora" / "checkpoint_final",
        ("irodori", "speaker-inversion"): (
            paths.models / "irodori" / "speaker" / "checkpoint_final.speaker.safetensors"
        ),
        ("lfm", "full"): paths.models / "lfm" / "full",
        ("lfm", "lora"): paths.models / "lfm" / "adapter",
        ("seed-vc", "finetune"): paths.models / "seed_vc" / "cfm.pth",
    }
    try:
        return destinations[(family, method)]
    except KeyError as exc:
        raise RuntimeError(f"No publication destination for {family}/{method}") from exc


def _publication_items(
    plan: TrainingPlan,
    paths: PersonaPaths,
    families: dict[str, Any],
) -> list[PublicationItem]:
    items: list[PublicationItem] = []
    for family_name, family in families.items():
        if not isinstance(family, dict) or family.get("enabled") is not True:
            continue
        method = family.get("method")
        if not isinstance(method, str):
            raise RuntimeError("Training result method is invalid")
        items.append(
            PublicationItem(
                family=family_name,
                method=method,
                family_fingerprint=plan.family(family_name).fingerprint,
                candidate=_family_artifact(paths, family),
                destination=_publication_destination(paths, family_name, method),
            )
        )
        auxiliary = family.get("auxiliary_speaker_embedding")
        if family_name == "irodori" and isinstance(auxiliary, str) and auxiliary:
            auxiliary_fingerprint = family.get("auxiliary_family_fingerprint")
            expected_auxiliary = plan.family("irodori").auxiliary_fingerprint
            if (
                expected_auxiliary is None
                or auxiliary_fingerprint != expected_auxiliary
            ):
                raise RuntimeError(
                    "Auxiliary speaker candidate changed its independent family contract"
                )
            items.append(
                PublicationItem(
                    family="irodori",
                    component="auxiliary-speaker",
                    method="speaker-inversion",
                    family_fingerprint=auxiliary_fingerprint,
                    candidate=_family_artifact(paths, {"artifact": auxiliary}),
                    destination=_publication_destination(
                        paths,
                        "irodori",
                        "speaker-inversion",
                    ),
                )
            )
    return items


def evaluate(repo_root: Path, paths: PersonaPaths, cfg: PersonaConfig) -> dict[str, Any]:
    """Evaluate locally and publish the complete candidate set only after all gates pass."""

    store = StateStore(paths.state)
    if not store.is_complete("prepare", _prepare_fingerprint(paths, cfg)):
        raise RuntimeError("Current prepared data is required before evaluation")
    train_fingerprint = _fingerprint(paths, cfg)
    if not store.is_trained(train_fingerprint):
        raise RuntimeError("No verified candidate set exists for the current training request")
    train_result = store.stage("train").get("result")
    if not isinstance(train_result, dict):
        raise RuntimeError("Training result is missing")
    plan = _plan_for_evaluation(repo_root, paths, cfg)
    if train_result.get("plan_fingerprint") != plan.fingerprint:
        raise RuntimeError("Candidate set belongs to a different immutable TrainingPlan")
    families = train_result.get("families")
    if not isinstance(families, dict):
        raise RuntimeError("Training result family contract is missing")

    _assert_held_out(paths, include_lfm=cfg.training.lfm.enabled)
    report_dir = paths.outputs / "evaluation" / plan.fingerprint
    report_dir.mkdir(parents=True, exist_ok=True)
    irodori = families.get("irodori")
    if not isinstance(irodori, dict) or irodori.get("enabled") is not True:
        raise RuntimeError("The voice quality gate requires an enabled Irodori candidate")
    irodori_method = irodori.get("method")
    if not isinstance(irodori_method, str):
        raise RuntimeError("Irodori candidate method is invalid")
    candidate, baseline, probes = _generate_voice_sets(
        repo_root,
        paths,
        cfg,
        artifact=_family_artifact(paths, irodori),
        method=irodori_method,
        report_dir=report_dir,
    )
    voice_report = _analyze_voice_sets(
        repo_root,
        paths,
        cfg,
        candidate=candidate,
        baseline=baseline,
        probes=probes,
    )
    lfm = families.get("lfm")
    lfm_report: dict[str, Any] = {"enabled": False, "contract_passed": True, "cases": []}
    if isinstance(lfm, dict) and lfm.get("enabled") is True:
        lfm_report = _evaluate_lfm(repo_root, cfg, lfm, _family_artifact(paths, lfm))
    validation = _validation_summary(families)
    gate = evaluate_quality_gate(
        {"summary": voice_report["summary"], "lfm": lfm_report},
        cfg.training.quality_gate.model_dump(mode="json"),
        validation=validation,
    )
    report: dict[str, Any] = {
        "schema_version": 2,
        "plan_fingerprint": plan.fingerprint,
        "summary": voice_report["summary"],
        "complete": voice_report["complete"],
        "errors": voice_report["errors"],
        "cases": voice_report["cases"],
        "mode_comparison": voice_report["mode_comparison"],
        "base_cer_mean": voice_report["base_cer_mean"],
        "candidate_cer_mean": voice_report["candidate_cer_mean"],
        "speaker_embedding_diagnostics": voice_report.get("speaker_embedding_diagnostics", {}),
        "lfm": lfm_report,
        "validation": validation,
        "quality_gate": gate,
        "published": False,
    }
    report_path = report_dir / "report.json"
    atomic_write_json(report_path, report)
    if gate.get("passed") is not True:
        train_result["quality_gate"] = {
            **gate,
            "pending_local_evaluation": False,
            "report": _persona_relative(paths, report_path),
        }
        store.set_result("train", train_result)
        return report

    items = _publication_items(plan, paths, families)
    publication = publish_training_candidates(
        paths.models,
        plan=plan,
        items=items,
        quality=gate,
    )
    primary_destinations = {
        item.family: item.destination for item in items if item.component == "primary"
    }
    auxiliary_destinations = {
        item.family: item.destination for item in items if item.component != "primary"
    }
    for family_name, destination in primary_destinations.items():
        family = families[family_name]
        family["artifact"] = _persona_relative(paths, destination)
        if family_name in auxiliary_destinations:
            family["auxiliary_speaker_embedding"] = _persona_relative(
                paths,
                auxiliary_destinations[family_name],
            )
    train_result["quality_gate"] = {
        **gate,
        "pending_local_evaluation": False,
        "report": _persona_relative(paths, report_path),
        "publication": _persona_relative(paths, paths.models / "publication.json"),
    }
    store.set_result("train", train_result)
    store.set_status("train", "complete")
    report["published"] = True
    report["publication"] = publication
    atomic_write_json(report_path, report)
    if not store.is_complete("train", train_fingerprint):
        raise RuntimeError("Published candidates failed final state verification")
    return report
