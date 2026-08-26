from __future__ import annotations

import math

import pytest

from personavoice.evaluation_metrics import (
    EvaluationSample,
    MetricInputError,
    aggregate_evaluation_metrics,
    base_cer_regression,
    character_error_rate,
    duration_ratio_error,
    emotion_accuracy,
    levenshtein_distance,
    mean_speaker_similarity,
    normalize_text,
    tokenize_words,
    unseen_pronunciation_score,
    word_error_rate,
)


def test_unicode_normalization_folds_width_case_combining_marks_and_punctuation() -> None:
    assert normalize_text("  ＣＡＦＥ\u0301、 WORLD！\t") == "café world"
    assert normalize_text("今日は、いい天気。", remove_whitespace=True) == "今日はいい天気"


def test_levenshtein_distance_is_exact_for_characters_and_tokens() -> None:
    assert levenshtein_distance("キタテン", "キツネ") == 3
    assert levenshtein_distance(("one", "two"), ("one", "new", "two")) == 1
    assert levenshtein_distance("", "abc") == 3


def test_japanese_cer_uses_normalized_characters() -> None:
    assert character_error_rate("今日は、晴れ", "今日は晴") == pytest.approx(1 / 5)
    assert character_error_rate("ガッツ", "ｶﾞｯﾂ") == 0.0
    assert character_error_rate("一", "一二三") == 2.0


def test_wer_uses_normalized_whitespace_words() -> None:
    assert word_error_rate("The quick, brown fox", "the quick fox") == pytest.approx(0.25)
    assert word_error_rate("one two", "one new two") == pytest.approx(0.5)


def test_japanese_wer_uses_stable_script_units_instead_of_one_sentence_token() -> None:
    assert tokenize_words("今日は晴れ") == ("今日", "は", "晴", "れ")
    assert tokenize_words("AIについて、ゆっくり") == ("ai", "について", "ゆっくり")
    assert word_error_rate("今日は晴れ", "今日は雨れ") == pytest.approx(0.25)
    assert word_error_rate("量子暗号の鍵配送", "量子暗号の配送") == pytest.approx(1 / 3)


def test_unseen_pronunciation_is_bounded_edit_similarity() -> None:
    assert unseen_pronunciation_score("かな", "かなた") == pytest.approx(2 / 3)
    assert unseen_pronunciation_score("未知。", "未知") == 1.0
    assert unseen_pronunciation_score("未知", "") == 0.0


def test_duration_emotion_speaker_and_base_regression_metrics() -> None:
    assert duration_ratio_error(2.0, 3.0) == pytest.approx(0.5)
    assert emotion_accuracy(["HAPPY", "sad"], ["ｈａｐｐｙ", "NEUTRAL"]) == 0.5
    assert mean_speaker_similarity([0.5, 0.7, 0.9]) == pytest.approx(0.7)
    assert base_cer_regression(0.10, 0.16) == pytest.approx(0.06)
    assert base_cer_regression(0.16, 0.10) == 0.0


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (character_error_rate, ("", "text")),
        (word_error_rate, (None, "text")),
        (duration_ratio_error, (0.0, 1.0)),
        (duration_ratio_error, (1.0, math.nan)),
        (emotion_accuracy, (["HAPPY"], [None])),
        (mean_speaker_similarity, ([0.8, None],)),
        (mean_speaker_similarity, ([1.1],)),
        (base_cer_regression, (0.1, math.inf)),
        (base_cer_regression, (0.1, 10**1000)),
    ],
)
def test_standalone_metrics_reject_missing_nonfinite_and_invalid_input(
    operation,
    arguments,
) -> None:
    with pytest.raises(MetricInputError):
        operation(*arguments)


def test_complete_aggregation_produces_quality_gate_summary() -> None:
    samples = [
        EvaluationSample(
            reference_text="今日は晴れ",
            hypothesis_text="今日は晴れ",
            speaker_similarity=0.8,
            reference_duration_seconds=2.0,
            generated_duration_seconds=2.0,
            expected_emotion="HAPPY",
            detected_emotion="happy",
            unseen=True,
        ),
        EvaluationSample(
            reference_text="hello world",
            hypothesis_text="hello brave world",
            speaker_similarity=0.6,
            reference_duration_seconds=4.0,
            generated_duration_seconds=5.0,
            expected_emotion="SAD",
            detected_emotion="NEUTRAL",
        ),
    ]

    metrics = aggregate_evaluation_metrics(
        samples,
        baseline_base_cer=0.10,
        candidate_base_cer=0.15,
    )

    assert metrics.complete is True
    assert metrics.errors == ()
    assert metrics.cer_mean == pytest.approx(
        (
            character_error_rate("今日は晴れ", "今日は晴れ")
            + character_error_rate("hello world", "hello brave world")
        )
        / 2
    )
    assert metrics.wer_mean == pytest.approx(0.25)
    assert metrics.unseen_text_similarity_mean == 1.0
    assert metrics.duration_ratio_error_mean == pytest.approx(0.125)
    assert metrics.emotion_accuracy == 0.5
    assert metrics.speaker_similarity_mean == pytest.approx(0.7)
    assert metrics.base_cer_regression == pytest.approx(0.05)
    assert set(metrics.summary()) == {
        "cer_mean",
        "wer_mean",
        "unseen_text_similarity_mean",
        "duration_ratio_error_mean",
        "emotion_accuracy",
        "speaker_similarity_mean",
        "base_cer_regression",
    }
    assert metrics.as_dict()["summary"] == metrics.summary()


def test_aggregation_fails_each_metric_closed_instead_of_skipping_bad_samples() -> None:
    samples = [
        EvaluationSample(
            reference_text="valid unseen text",
            hypothesis_text="valid unseen text",
            speaker_similarity=0.9,
            reference_duration_seconds=1.0,
            generated_duration_seconds=1.0,
            expected_emotion="HAPPY",
            detected_emotion="HAPPY",
            unseen=True,
        ),
        EvaluationSample(
            reference_text="second text",
            hypothesis_text=None,
            speaker_similarity=None,
            reference_duration_seconds=1.0,
            generated_duration_seconds=math.nan,
            expected_emotion="SAD",
            detected_emotion=None,
        ),
    ]

    metrics = aggregate_evaluation_metrics(
        samples,
        baseline_base_cer=0.1,
        candidate_base_cer=math.inf,
    )

    assert metrics.complete is False
    assert metrics.cer_mean is None
    assert metrics.wer_mean is None
    assert metrics.unseen_text_similarity_mean == 1.0
    assert metrics.duration_ratio_error_mean is None
    assert metrics.emotion_accuracy is None
    assert metrics.speaker_similarity_mean is None
    assert metrics.base_cer_regression is None
    assert len(metrics.errors) == 6
    assert "valid unseen text" not in repr(metrics.errors)


def test_empty_aggregation_is_an_explicit_incomplete_report() -> None:
    metrics = aggregate_evaluation_metrics(
        [],
        baseline_base_cer=None,
        candidate_base_cer=None,
    )

    assert metrics.complete is False
    assert all(value is None for value in metrics.summary().values())
    assert metrics.errors
