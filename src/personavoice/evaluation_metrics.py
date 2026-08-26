from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any


class MetricInputError(ValueError):
    """Raised when a standalone metric cannot be computed safely."""


def normalize_text(text: str, *, remove_whitespace: bool = False) -> str:
    """Return deterministic Unicode text for ASR comparison.

    NFKC joins canonically equivalent Japanese forms and folds full-width Latin
    text. Case-folding makes Latin comparisons case-insensitive. Unicode
    punctuation is treated as a boundary, whitespace is collapsed, and callers
    computing Japanese CER can remove all whitespace.
    """

    if not isinstance(text, str):
        raise MetricInputError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    collapsed = " ".join(without_punctuation.split())
    return "".join(collapsed.split()) if remove_whitespace else collapsed


def levenshtein_distance[T](reference: Sequence[T], hypothesis: Sequence[T]) -> int:
    """Compute exact edit distance with O(min(n, m)) auxiliary memory."""

    if len(reference) < len(hypothesis):
        shorter, longer = reference, hypothesis
    else:
        shorter, longer = hypothesis, reference
    previous = list(range(len(shorter) + 1))
    for longer_index, longer_item in enumerate(longer, start=1):
        current = [longer_index]
        for shorter_index, shorter_item in enumerate(shorter, start=1):
            insertion = current[-1] + 1
            deletion = previous[shorter_index] + 1
            substitution = previous[shorter_index - 1] + (longer_item != shorter_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _normalized_characters(text: str | None, *, label: str) -> str:
    if text is None:
        raise MetricInputError(f"{label} text is missing")
    normalized = normalize_text(text, remove_whitespace=True)
    if not normalized:
        raise MetricInputError(f"{label} text is empty after normalization")
    return normalized


def _word_script(character: str) -> str:
    codepoint = ord(character)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
    ):
        return "han"
    if 0x3040 <= codepoint <= 0x309F:
        return "hiragana"
    if 0x30A0 <= codepoint <= 0x30FF:
        return "katakana"
    category = unicodedata.category(character)
    if category[0] in {"L", "N"}:
        return "alphanumeric"
    if category[0] == "M":
        return "mark"
    return f"other-{category}"


def tokenize_words(text: str) -> tuple[str, ...]:
    """Tokenize Latin and unsegmented Japanese deterministically for WER.

    Whitespace remains the boundary for ordinary alphabetic text.  Japanese
    does not normally contain spaces, so contiguous Han, Hiragana and Katakana
    runs form separate word-like units; script changes form boundaries.  This
    dependency-free contract avoids treating an entire Japanese sentence as
    one binary token while remaining identical across local and CI platforms.
    It is intentionally a deterministic evaluation segmentation, not a claim
    to reproduce a particular morphological dictionary.
    """

    normalized = normalize_text(text)
    tokens: list[str] = []
    current: list[str] = []
    current_script: str | None = None

    def flush() -> None:
        nonlocal current, current_script
        if current:
            tokens.append("".join(current))
        current = []
        current_script = None

    for character in normalized:
        if character.isspace():
            flush()
            continue
        script = _word_script(character)
        if script == "mark" and current:
            current.append(character)
            continue
        # The Japanese prolonged sound mark belongs to the preceding kana run.
        if character == "ー" and current_script in {"hiragana", "katakana"}:
            current.append(character)
            continue
        if current and script != current_script:
            flush()
        current.append(character)
        current_script = script
    flush()
    return tuple(tokens)


def _normalized_words(text: str | None, *, label: str) -> tuple[str, ...]:
    if text is None:
        raise MetricInputError(f"{label} text is missing")
    words = tokenize_words(text)
    if not words:
        raise MetricInputError(f"{label} text has no words after normalization")
    return words


def character_error_rate(reference: str | None, hypothesis: str | None) -> float:
    """Calculate Japanese-friendly character Levenshtein error rate.

    The denominator is the normalized reference length, so insertion-heavy
    hypotheses may correctly produce a value greater than one.
    """

    reference_characters = _normalized_characters(reference, label="reference")
    if hypothesis is None:
        raise MetricInputError("hypothesis text is missing")
    hypothesis_characters = normalize_text(hypothesis, remove_whitespace=True)
    return levenshtein_distance(reference_characters, hypothesis_characters) / len(
        reference_characters
    )


def word_error_rate(reference: str | None, hypothesis: str | None) -> float:
    """Calculate WER with deterministic Japanese-aware script segmentation."""

    reference_words = _normalized_words(reference, label="reference")
    if hypothesis is None:
        raise MetricInputError("hypothesis text is missing")
    hypothesis_words = tokenize_words(hypothesis)
    return levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def unseen_pronunciation_score(reference: str | None, hypothesis: str | None) -> float:
    """Return a bounded character-edit similarity for held-out pronunciation."""

    reference_characters = _normalized_characters(reference, label="reference")
    if hypothesis is None:
        raise MetricInputError("hypothesis text is missing")
    hypothesis_characters = normalize_text(hypothesis, remove_whitespace=True)
    denominator = max(len(reference_characters), len(hypothesis_characters))
    distance = levenshtein_distance(reference_characters, hypothesis_characters)
    return max(0.0, 1.0 - distance / denominator)


def _finite_number(value: float | int | None, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MetricInputError(f"{label} is missing or is not numeric")
    try:
        converted = float(value)
    except (OverflowError, ValueError) as exc:
        raise MetricInputError(f"{label} cannot be represented safely") from exc
    if not math.isfinite(converted):
        raise MetricInputError(f"{label} is not finite")
    return converted


def duration_ratio_error(
    reference_seconds: float | int | None,
    generated_seconds: float | int | None,
) -> float:
    """Return ``abs(generated / reference - 1)`` for positive durations."""

    reference = _finite_number(reference_seconds, label="reference duration")
    generated = _finite_number(generated_seconds, label="generated duration")
    if reference <= 0:
        raise MetricInputError("reference duration must be greater than zero")
    if generated < 0:
        raise MetricInputError("generated duration must not be negative")
    return abs(generated / reference - 1.0)


def _emotion_label(value: str | None, *, label: str) -> str:
    if value is None or not isinstance(value, str):
        raise MetricInputError(f"{label} emotion is missing")
    normalized = normalize_text(value)
    if not normalized:
        raise MetricInputError(f"{label} emotion is empty")
    return normalized


def emotion_accuracy(
    expected: Sequence[str | None],
    detected: Sequence[str | None],
) -> float:
    """Return exact normalized-label accuracy without dropping missing labels."""

    if not expected or len(expected) != len(detected):
        raise MetricInputError("emotion sequences must be non-empty and have equal length")
    matches = 0
    for expected_value, detected_value in zip(expected, detected, strict=True):
        expected_label = _emotion_label(expected_value, label="expected")
        detected_label = _emotion_label(detected_value, label="detected")
        matches += expected_label == detected_label
    return matches / len(expected)


def mean_speaker_similarity(values: Sequence[float | int | None]) -> float:
    """Average cosine similarities, rejecting missing, non-finite, or out-of-range input."""

    if not values:
        raise MetricInputError("speaker similarities are missing")
    converted: list[float] = []
    for value in values:
        similarity = _finite_number(value, label="speaker similarity")
        if not -1.0 <= similarity <= 1.0:
            raise MetricInputError("speaker similarity must be between -1 and 1")
        converted.append(similarity)
    return math.fsum(converted) / len(converted)


def base_cer_regression(
    baseline_cer: float | int | None,
    candidate_cer: float | int | None,
) -> float:
    """Return only degradation from the frozen base model; improvement is zero."""

    baseline = _finite_number(baseline_cer, label="baseline base CER")
    candidate = _finite_number(candidate_cer, label="candidate base CER")
    if baseline < 0 or candidate < 0:
        raise MetricInputError("base CER values must not be negative")
    return max(0.0, candidate - baseline)


@dataclass(frozen=True, slots=True)
class EvaluationSample:
    """One generated sample; optional fields let aggregation fail closed with diagnostics."""

    reference_text: str | None = None
    hypothesis_text: str | None = None
    speaker_similarity: float | None = None
    reference_duration_seconds: float | None = None
    generated_duration_seconds: float | None = None
    expected_emotion: str | None = None
    detected_emotion: str | None = None
    unseen: bool = False


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    cer_mean: float | None
    wer_mean: float | None
    unseen_text_similarity_mean: float | None
    duration_ratio_error_mean: float | None
    emotion_accuracy: float | None
    speaker_similarity_mean: float | None
    base_cer_regression: float | None
    complete: bool
    errors: tuple[str, ...]

    def summary(self) -> dict[str, float | None]:
        """Return exactly the metric names consumed by the publication quality gate."""

        return {
            "cer_mean": self.cer_mean,
            "wer_mean": self.wer_mean,
            "unseen_text_similarity_mean": self.unseen_text_similarity_mean,
            "duration_ratio_error_mean": self.duration_ratio_error_mean,
            "emotion_accuracy": self.emotion_accuracy,
            "speaker_similarity_mean": self.speaker_similarity_mean,
            "base_cer_regression": self.base_cer_regression,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "complete": self.complete,
            "errors": list(self.errors),
        }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise MetricInputError("metric has no eligible samples")
    if not all(math.isfinite(value) for value in values):
        raise MetricInputError("metric contains a non-finite result")
    return math.fsum(values) / len(values)


def aggregate_evaluation_metrics(
    samples: Iterable[EvaluationSample],
    *,
    baseline_base_cer: float | int | None,
    candidate_base_cer: float | int | None,
) -> EvaluationMetrics:
    """Aggregate the quality report without ever averaging around invalid data.

    Each metric is all-or-nothing: if any eligible sample is missing or invalid,
    that metric becomes ``None`` and ``complete`` is false. Other valid metrics
    remain available for diagnosis. Error messages identify only the metric and
    sample index; raw text and model values are never copied into diagnostics.
    """

    rows = tuple(samples)
    errors: list[str] = []

    if any(not isinstance(row, EvaluationSample) for row in rows):
        errors.append("samples: every item must be an EvaluationSample")
        rows = ()

    def capture(name: str, operation: Callable[[], float]) -> float | None:
        try:
            value = operation()
        except (MetricInputError, TypeError, ValueError, OverflowError):
            errors.append(f"{name}: missing or invalid input")
            return None
        if not math.isfinite(value):
            errors.append(f"{name}: non-finite result")
            return None
        return value

    def per_sample(
        name: str,
        operation: Callable[[EvaluationSample], float],
        selected: Sequence[EvaluationSample] = rows,
    ) -> float | None:
        values: list[float] = []
        if not selected:
            errors.append(f"{name}: no eligible samples")
            return None
        for index, row in enumerate(selected):
            try:
                value = operation(row)
            except (MetricInputError, TypeError, ValueError, OverflowError):
                errors.append(f"{name}: sample {index} is missing or invalid")
                return None
            if not math.isfinite(value):
                errors.append(f"{name}: sample {index} is non-finite")
                return None
            values.append(value)
        return _mean(values)

    cer_mean = per_sample(
        "cer_mean",
        lambda row: character_error_rate(row.reference_text, row.hypothesis_text),
    )
    wer_mean = per_sample(
        "wer_mean",
        lambda row: word_error_rate(row.reference_text, row.hypothesis_text),
    )
    unseen_rows = tuple(row for row in rows if row.unseen is True)
    unseen_mean = per_sample(
        "unseen_text_similarity_mean",
        lambda row: unseen_pronunciation_score(row.reference_text, row.hypothesis_text),
        unseen_rows,
    )
    duration_mean = per_sample(
        "duration_ratio_error_mean",
        lambda row: duration_ratio_error(
            row.reference_duration_seconds,
            row.generated_duration_seconds,
        ),
    )
    emotion_mean = capture(
        "emotion_accuracy",
        lambda: emotion_accuracy(
            [row.expected_emotion for row in rows],
            [row.detected_emotion for row in rows],
        ),
    )
    speaker_mean = capture(
        "speaker_similarity_mean",
        lambda: mean_speaker_similarity([row.speaker_similarity for row in rows]),
    )
    regression = capture(
        "base_cer_regression",
        lambda: base_cer_regression(baseline_base_cer, candidate_base_cer),
    )
    summary_values = (
        cer_mean,
        wer_mean,
        unseen_mean,
        duration_mean,
        emotion_mean,
        speaker_mean,
        regression,
    )
    return EvaluationMetrics(
        cer_mean=cer_mean,
        wer_mean=wer_mean,
        unseen_text_similarity_mean=unseen_mean,
        duration_ratio_error_mean=duration_mean,
        emotion_accuracy=emotion_mean,
        speaker_similarity_mean=speaker_mean,
        base_cer_regression=regression,
        complete=not errors and all(value is not None for value in summary_values),
        errors=tuple(errors),
    )


__all__ = [
    "EvaluationMetrics",
    "EvaluationSample",
    "MetricInputError",
    "aggregate_evaluation_metrics",
    "base_cer_regression",
    "character_error_rate",
    "duration_ratio_error",
    "emotion_accuracy",
    "levenshtein_distance",
    "mean_speaker_similarity",
    "normalize_text",
    "tokenize_words",
    "unseen_pronunciation_score",
    "word_error_rate",
]
