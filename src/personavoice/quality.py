from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

QUALITY_GATE_SCHEMA_VERSION = 1


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class GateCheck:
    metric: str
    value: float | None
    operator: str
    threshold: float | None
    passed: bool
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "operator": self.operator,
            "threshold": self.threshold,
            "required": self.required,
            "passed": self.passed,
        }


def _minimum(metric: str, value: Any, threshold: Any) -> GateCheck:
    actual = _finite(value)
    expected = _finite(threshold)
    return GateCheck(
        metric=metric,
        value=actual,
        operator=">=",
        threshold=expected,
        passed=actual is not None and expected is not None and actual >= expected,
    )


def _maximum(metric: str, value: Any, threshold: Any) -> GateCheck:
    actual = _finite(value)
    expected = _finite(threshold)
    return GateCheck(
        metric=metric,
        value=actual,
        operator="<=",
        threshold=expected,
        passed=actual is not None and expected is not None and actual <= expected,
    )


def evaluate_quality_gate(
    report: dict[str, Any],
    policy: dict[str, Any],
    *,
    validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate the versioned local publication gate; missing/NaN metrics fail closed."""

    summary = report.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    checks = [
        _minimum(
            "speaker_similarity_mean",
            summary.get("speaker_similarity_mean"),
            policy.get("min_speaker_similarity"),
        ),
        _maximum("cer_mean", summary.get("cer_mean"), policy.get("max_cer")),
        _maximum("wer_mean", summary.get("wer_mean"), policy.get("max_wer")),
        _minimum(
            "emotion_accuracy",
            summary.get("emotion_accuracy"),
            policy.get("min_emotion_accuracy"),
        ),
        _minimum(
            "unseen_text_similarity_mean",
            summary.get("unseen_text_similarity_mean"),
            policy.get("min_unseen_text_similarity"),
        ),
        _maximum(
            "duration_ratio_error_mean",
            summary.get("duration_ratio_error_mean"),
            policy.get("max_duration_ratio_error"),
        ),
        _maximum(
            "base_cer_regression",
            summary.get("base_cer_regression"),
            policy.get("max_base_cer_regression"),
        ),
    ]
    validation_loss = _finite((validation or {}).get("loss"))
    require_validation = policy.get("require_validation") is True
    checks.append(
        GateCheck(
            metric="validation_loss",
            value=validation_loss,
            operator="finite",
            threshold=None,
            required=require_validation,
            passed=validation_loss is not None if require_validation else True,
        )
    )
    lfm = report.get("lfm")
    if isinstance(lfm, dict) and lfm.get("enabled") is True:
        checks.extend(
            [
                GateCheck(
                    metric="lfm_held_out_complete",
                    value=1.0 if lfm.get("complete") is True else 0.0,
                    operator="==",
                    threshold=1.0,
                    required=True,
                    passed=lfm.get("complete") is True,
                ),
                GateCheck(
                    metric="lfm_held_out_contract",
                    value=1.0 if lfm.get("contract_passed") is True else 0.0,
                    operator="==",
                    threshold=1.0,
                    required=True,
                    passed=lfm.get("contract_passed") is True,
                ),
                GateCheck(
                    metric="lfm_baseline_contract",
                    value=1.0 if lfm.get("baseline_contract_passed") is True else 0.0,
                    operator="==",
                    threshold=1.0,
                    required=True,
                    passed=lfm.get("baseline_contract_passed") is True,
                ),
                _minimum(
                    "lfm_expected_similarity_mean",
                    lfm.get("candidate_expected_similarity_mean"),
                    policy.get("min_lfm_expected_similarity"),
                ),
                _maximum(
                    "lfm_expected_cer_mean",
                    lfm.get("candidate_expected_cer_mean"),
                    policy.get("max_lfm_expected_cer"),
                ),
                _maximum(
                    "lfm_expected_wer_mean",
                    lfm.get("candidate_expected_wer_mean"),
                    policy.get("max_lfm_expected_wer"),
                ),
                _minimum(
                    "lfm_required_phrase_coverage_mean",
                    lfm.get("required_phrase_coverage_mean"),
                    policy.get("min_lfm_required_phrase_coverage"),
                ),
                _maximum(
                    "lfm_base_similarity_regression_max",
                    lfm.get("base_similarity_regression_max"),
                    policy.get("max_lfm_base_similarity_regression"),
                ),
            ]
        )
    enabled = policy.get("enabled") is not False
    passed = bool(enabled and all(check.passed for check in checks if check.required))
    return {
        "schema_version": QUALITY_GATE_SCHEMA_VERSION,
        "enabled": enabled,
        "passed": passed,
        "checks": [check.as_dict() for check in checks],
    }
