from __future__ import annotations

from personavoice.config import QualityGateConfig
from personavoice.quality import evaluate_quality_gate


def _policy() -> dict:
    return QualityGateConfig().model_dump(mode="json")


def _passing_summary() -> dict:
    return {
        "summary": {
            "speaker_similarity_mean": 0.9,
            "cer_mean": 0.05,
            "wer_mean": 0.1,
            "emotion_accuracy": 0.8,
            "unseen_text_similarity_mean": 0.9,
            "duration_ratio_error_mean": 0.1,
            "base_cer_regression": 0.01,
        }
    }


def _passing_lfm() -> dict:
    return {
        "enabled": True,
        "complete": True,
        "contract_passed": True,
        "baseline_contract_passed": True,
        "candidate_expected_similarity_mean": 0.8,
        "candidate_expected_cer_mean": 0.2,
        "candidate_expected_wer_mean": 0.4,
        "required_phrase_coverage_mean": 1.0,
        "base_similarity_regression_max": 0.0,
    }


def test_quality_gate_passes_only_complete_finite_report() -> None:
    gate = evaluate_quality_gate(_passing_summary(), _policy(), validation={"loss": 1.2})
    assert gate["passed"] is True


def test_quality_gate_fails_closed_for_missing_metric() -> None:
    report = _passing_summary()
    del report["summary"]["speaker_similarity_mean"]
    gate = evaluate_quality_gate(report, _policy(), validation={"loss": 1.2})
    assert gate["passed"] is False
    assert (
        next(item for item in gate["checks"] if item["metric"] == "speaker_similarity_mean")[
            "passed"
        ]
        is False
    )


def test_quality_gate_requires_validation_loss_by_default() -> None:
    gate = evaluate_quality_gate(_passing_summary(), _policy(), validation=None)
    assert gate["passed"] is False


def test_quality_gate_requires_measured_lfm_quality_and_regression() -> None:
    report = _passing_summary()
    report["lfm"] = _passing_lfm()
    gate = evaluate_quality_gate(report, _policy(), validation={"loss": 0.5})
    assert gate["passed"] is True

    report["lfm"]["candidate_expected_similarity_mean"] = None
    report["lfm"]["base_similarity_regression_max"] = 0.2
    gate = evaluate_quality_gate(report, _policy(), validation={"loss": 0.5})
    assert gate["passed"] is False
    failed = {check["metric"] for check in gate["checks"] if not check["passed"]}
    assert "lfm_expected_similarity_mean" in failed
    assert "lfm_base_similarity_regression_max" in failed
