from __future__ import annotations

import math
from collections.abc import Iterable


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    av = [float(value) for value in a]
    bv = [float(value) for value in b]
    if len(av) != len(bv) or not av:
        return -1.0
    dot = sum(x * y for x, y in zip(av, bv, strict=True))
    na = math.sqrt(sum(x * x for x in av))
    nb = math.sqrt(sum(y * y for y in bv))
    if na <= 1e-12 or nb <= 1e-12:
        return -1.0
    return dot / (na * nb)


def mean_embedding(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("no embeddings")
    width = len(embeddings[0])
    if any(len(row) != width for row in embeddings):
        raise ValueError("embedding dimensions differ")
    return [sum(row[index] for row in embeddings) / len(embeddings) for index in range(width)]


def select_target_speaker(
    speaker_embeddings: dict[str, list[float]],
    identity_embeddings: list[list[float]],
    *,
    threshold: float,
) -> tuple[str, float]:
    if not speaker_embeddings:
        raise ValueError("diarization returned no speaker embeddings")
    if not identity_embeddings:
        if len(speaker_embeddings) == 1:
            label = next(iter(speaker_embeddings))
            return label, 1.0
        raise ValueError(
            "Multiple speakers were detected but identity/ has no usable reference audio. "
            "Add 1-3 clean clips of the authorized target speaker to identity/."
        )
    reference = mean_embedding(identity_embeddings)
    ranked = sorted(
        (
            (label, cosine_similarity(embedding, reference))
            for label, embedding in speaker_embeddings.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    label, score = ranked[0]
    if score < threshold:
        raise ValueError(
            f"Target-speaker match is uncertain: best={label} "
            f"similarity={score:.3f} < {threshold:.3f}. "
            "Add cleaner identity reference audio or lower "
            "prepare.min_identity_similarity deliberately."
        )
    return label, score


def interval_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def dominant_speaker(start: float, end: float, turns: list[dict]) -> tuple[str | None, float]:
    duration = max(1e-9, end - start)
    scores: dict[str, float] = {}
    for turn in turns:
        overlap = interval_overlap(start, end, float(turn["start"]), float(turn["end"]))
        if overlap > 0:
            speaker = str(turn["speaker"])
            scores[speaker] = scores.get(speaker, 0.0) + overlap
    if not scores:
        return None, 0.0
    speaker, overlap = max(scores.items(), key=lambda item: item[1])
    return speaker, min(1.0, overlap / duration)


def overlap_ratio(start: float, end: float, regular_turns: list[dict]) -> float:
    duration = max(1e-9, end - start)
    points = {start, end}
    for turn in regular_turns:
        turn_start = max(start, float(turn["start"]))
        turn_end = min(end, float(turn["end"]))
        if turn_end > turn_start:
            points.add(turn_start)
            points.add(turn_end)
    ordered = sorted(points)
    overlap = 0.0
    for left, right in zip(ordered, ordered[1:], strict=False):
        midpoint = (left + right) / 2
        active = sum(
            1
            for turn in regular_turns
            if float(turn["start"]) <= midpoint < float(turn["end"])
        )
        if active >= 2:
            overlap += right - left
    return min(1.0, overlap / duration)
