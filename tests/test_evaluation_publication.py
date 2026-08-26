from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pytest

import personavoice.evaluation as evaluation
from personavoice.config import PersonaConfig
from personavoice.project import PersonaPaths
from personavoice.training_plan import FamilyPlan, TrainingPlan


def _family(name: str, method: str, enabled: bool = True) -> FamilyPlan:
    return FamilyPlan(
        family=name,
        enabled=enabled,
        method=method,
        dataset_fingerprint="d" * 64,
        training={},
        model_contract={},
        implementation_contract={},
        checkpoint_policy={},
        evaluation_policy={},
    )


def _plan() -> TrainingPlan:
    return TrainingPlan(
        persona="alice",
        files=(),
        families=(
            _family("irodori", "full"),
            _family("lfm", "full"),
            _family("seed-vc", "finetune", enabled=False),
        ),
    )


def _training_result(plan: TrainingPlan) -> dict[str, Any]:
    return {
        "train_schema": 9,
        "fingerprint": "train-fingerprint",
        "plan_fingerprint": plan.fingerprint,
        "families": {
            "irodori": {
                "enabled": True,
                "method": "full",
                "family_fingerprint": plan.family("irodori").fingerprint,
                "artifact": "models/.candidates/irodori/full",
                "validation": {"loss": 1.0},
            },
            "lfm": {
                "enabled": True,
                "method": "full",
                "family_fingerprint": plan.family("lfm").fingerprint,
                "artifact": "models/.candidates/lfm/full",
                "validation": {"loss": 0.8},
            },
            "seed-vc": {
                "enabled": False,
                "method": "finetune",
                "family_fingerprint": plan.family("seed-vc").fingerprint,
                "artifact": None,
                "validation": {"loss": None},
            },
        },
        "quality_gate": {"passed": False},
    }


def _passing_voice() -> dict[str, Any]:
    return {
        "summary": {
            "speaker_similarity_mean": 0.9,
            "cer_mean": 0.01,
            "wer_mean": 0.01,
            "emotion_accuracy": 1.0,
            "unseen_text_similarity_mean": 0.99,
            "duration_ratio_error_mean": 0.05,
            "base_cer_regression": 0.0,
        },
        "complete": True,
        "errors": [],
        "cases": [],
        "mode_comparison": [],
        "base_cer_mean": 0.05,
        "candidate_cer_mean": 0.01,
    }


def _passing_lfm() -> dict[str, Any]:
    return {
        "enabled": True,
        "complete": True,
        "contract_passed": True,
        "baseline_contract_passed": True,
        "candidate_expected_similarity_mean": 1.0,
        "candidate_expected_cer_mean": 0.0,
        "candidate_expected_wer_mean": 0.0,
        "baseline_expected_similarity_mean": 0.2,
        "baseline_expected_cer_mean": 0.8,
        "baseline_expected_wer_mean": 1.0,
        "required_phrase_coverage_mean": 1.0,
        "base_similarity_regression_max": 0.0,
        "errors": [],
        "cases": [],
    }


class FakeStore:
    instance: FakeStore | None = None
    result: dict[str, Any]

    def __init__(self, path: Path) -> None:
        del path
        FakeStore.instance = self
        self.result = FakeStore.result
        self.status: str | None = None

    def is_complete(self, name: str, fingerprint: str) -> bool:
        del fingerprint
        return name == "prepare" or self.status == "complete"

    def is_trained(self, fingerprint: str) -> bool:
        del fingerprint
        return True

    def stage(self, name: str) -> dict[str, Any]:
        assert name == "train"
        return {"result": self.result}

    def set_result(self, name: str, result: dict[str, Any]) -> None:
        assert name == "train"
        self.result = result

    def set_status(self, name: str, status: str) -> None:
        assert name == "train"
        self.status = status


@pytest.fixture
def wired_evaluation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan = _plan()
    paths = PersonaPaths(tmp_path / "persona")
    for directory in (paths.dataset, paths.models, paths.outputs):
        directory.mkdir(parents=True, exist_ok=True)
    for relative in (
        "models/.candidates/irodori/full",
        "models/.candidates/lfm/full",
    ):
        (paths.root / relative).mkdir(parents=True)
    FakeStore.result = _training_result(plan)
    monkeypatch.setattr(evaluation, "StateStore", FakeStore)
    monkeypatch.setattr(evaluation, "_prepare_fingerprint", lambda paths, cfg: "prepare")
    monkeypatch.setattr(evaluation, "_fingerprint", lambda paths, cfg: "train-fingerprint")
    monkeypatch.setattr(evaluation, "_plan_for_evaluation", lambda *args: plan)
    monkeypatch.setattr(evaluation, "_assert_held_out", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        evaluation,
        "_generate_voice_sets",
        lambda *args, **kwargs: ({}, {}, {}),
    )
    monkeypatch.setattr(
        evaluation,
        "_evaluate_lfm",
        lambda *args, **kwargs: _passing_lfm(),
    )
    return paths, plan


def test_passing_gate_publishes_then_marks_training_complete(
    wired_evaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, plan = wired_evaluation
    monkeypatch.setattr(evaluation, "_analyze_voice_sets", lambda *args, **kwargs: _passing_voice())
    published: list[list[evaluation.PublicationItem]] = []

    def publish(models_root, *, plan, items, quality):
        assert models_root == paths.models
        assert quality["passed"] is True
        published.append(items)
        for item in items:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
        (paths.models / "publication.json").write_text("{}", encoding="utf-8")
        return {"plan_fingerprint": plan.fingerprint}

    monkeypatch.setattr(evaluation, "publish_training_candidates", publish)

    report = evaluation.evaluate(Path.cwd(), paths, PersonaConfig(name="alice"))

    assert report["published"] is True
    assert published and {item.family for item in published[0]} == {"irodori", "lfm"}
    assert FakeStore.instance is not None
    assert FakeStore.instance.status == "complete"
    assert FakeStore.instance.result["families"]["irodori"]["artifact"] == ("models/irodori/full")
    assert report["plan_fingerprint"] == plan.fingerprint


def test_failed_gate_retains_candidates_and_never_calls_publication(
    wired_evaluation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths, _ = wired_evaluation
    failed = _passing_voice()
    failed["summary"]["speaker_similarity_mean"] = None
    monkeypatch.setattr(evaluation, "_analyze_voice_sets", lambda *args, **kwargs: failed)

    def must_not_publish(*args, **kwargs):
        del args, kwargs
        raise AssertionError("publication must not run")

    monkeypatch.setattr(evaluation, "publish_training_candidates", must_not_publish)

    report = evaluation.evaluate(Path.cwd(), paths, PersonaConfig(name="alice"))

    assert report["published"] is False
    assert report["quality_gate"]["passed"] is False
    assert FakeStore.instance is not None
    assert FakeStore.instance.status is None
    assert FakeStore.instance.result["families"]["irodori"]["artifact"].startswith(
        "models/.candidates/"
    )


def test_full_conditioning_comparison_covers_every_held_out_metric_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    report_dir = paths.outputs / "evaluation"
    artifact = paths.models / ".candidates" / "irodori" / "full"
    calls: list[dict[str, Any]] = []

    def fake_synthesize(*args, **kwargs):
        del args
        calls.append(dict(kwargs))
        return [Path(kwargs["output"])]

    monkeypatch.setattr(evaluation, "synthesize", fake_synthesize)

    candidate, baseline, modes = evaluation._generate_voice_sets(
        tmp_path,
        paths,
        PersonaConfig(name="alice"),
        artifact=artifact,
        method="full",
        report_dir=report_dir,
    )

    case_ids = {case["id"] for case in evaluation.CASES}
    assert set(candidate) == case_ids
    assert set(baseline) == case_ids
    assert set(modes) == {
        "speaker-conditioned",
        "no-reference",
        "caption-conditioned",
    }
    assert all(set(outputs) == case_ids for outputs in modes.values())
    assert modes["caption-conditioned"] == candidate
    assert len(calls) == len(evaluation.CASES) * 4

    candidate_calls = [
        call
        for call in calls
        if "irodori_artifact" in call and Path(call["output"]).parent.name == "candidate"
    ]
    baseline_calls = [call for call in calls if call.get("base_only") is True]
    speaker_calls = [call for call in calls if "speaker-conditioned" in str(call["output"])]
    no_ref_calls = [call for call in calls if "no-reference" in str(call["output"])]
    assert len(candidate_calls) == len(evaluation.CASES)
    assert all(call["reference_mode"] == "none" for call in candidate_calls)
    assert len(baseline_calls) == len(evaluation.CASES)
    assert all(call["reference_mode"] == "none" for call in baseline_calls)
    assert len(speaker_calls) == len(evaluation.CASES)
    assert all(
        call["reference_mode"] == "auto" and call["caption_conditioning"] is False
        for call in speaker_calls
    )
    assert len(no_ref_calls) == len(evaluation.CASES)
    assert all(
        call["reference_mode"] == "none" and call["caption_conditioning"] is False
        for call in no_ref_calls
    )


def test_held_out_gate_rejects_lfm_prompt_overlap_and_accepts_distinct_prompt(
    tmp_path: Path,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"text":"まったく別の音声学習文です。"}\n', encoding="utf-8"
    )
    lfm_dataset = paths.dataset / "lfm_train.jsonl"
    lfm_dataset.write_text(
        json.dumps(
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": evaluation.LFM_CASES[0]["prompt"],
                    }
                ],
                "completion": [{"role": "assistant", "content": "回答"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lfm:lfm-explain"):
        evaluation._assert_held_out(paths)

    lfm_dataset.write_text(
        json.dumps(
            {
                "prompt": [{"role": "user", "content": "別の質問です。"}],
                "completion": [{"role": "assistant", "content": "回答"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    evaluation._assert_held_out(paths)


def test_held_out_gate_extracts_exported_dialogue_and_assistant_answer(
    tmp_path: Path,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"text":"まったく別の音声学習文です。"}\n',
        encoding="utf-8",
    )
    lfm_dataset = paths.dataset / "lfm_train.jsonl"

    def write_example(user_content: str, answer: str) -> None:
        lfm_dataset.write_text(
            json.dumps(
                {
                    "prompt": [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": user_content},
                    ],
                    "completion": [
                        {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "text": answer,
                                    "voice": {
                                        "caption": "自然",
                                        "emotion": "NEUTRAL",
                                        "events": [],
                                    },
                                },
                                ensure_ascii=False,
                            ),
                        }
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    prompt_case = evaluation.LFM_CASES[0]
    wrapped_prompt = (
        f"{evaluation._LFM_DIALOGUE_HEADER}\n"
        f"相手: {prompt_case['prompt']}\n"
        f"{evaluation._LFM_DIALOGUE_SUFFIX}"
    )
    write_example(wrapped_prompt, "別の回答です。")
    with pytest.raises(RuntimeError, match="lfm:lfm-explain:prompt"):
        evaluation._assert_held_out(paths)

    wrapped_distinct = (
        f"{evaluation._LFM_DIALOGUE_HEADER}\n"
        "相手: これは別の会話です。\n"
        f"{evaluation._LFM_DIALOGUE_SUFFIX}"
    )
    write_example(wrapped_distinct, str(prompt_case["expected_completion"]))
    with pytest.raises(RuntimeError, match="lfm:lfm-explain:answer"):
        evaluation._assert_held_out(paths)

    # Exact normalized utterances collide; merely containing the fixed text in
    # a longer genuine utterance does not create a substring false positive.
    write_example(
        (
            f"{evaluation._LFM_DIALOGUE_HEADER}\n"
            f"相手: 前置きです。{prompt_case['prompt']}これは別の発話です。\n"
            f"{evaluation._LFM_DIALOGUE_SUFFIX}"
        ),
        str(prompt_case["expected_completion"]) + "これは別の発話です。",
    )
    evaluation._assert_held_out(paths)


@pytest.mark.parametrize(
    ("user_content", "answer", "message"),
    [
        (
            (
                f"prefix {evaluation._LFM_DIALOGUE_HEADER}\n"
                "相手: 別の会話です。\n"
                f"{evaluation._LFM_DIALOGUE_SUFFIX}"
            ),
            "別の回答です。",
            "malformed dialogue wrapper",
        ),
        (
            (
                f"{evaluation._LFM_DIALOGUE_HEADER}\n"
                "相手: 別の会話です。\n"
                f"{evaluation._LFM_DIALOGUE_SUFFIX}"
            ),
            '{"text":',
            "unreadable assistant completion",
        ),
    ],
)
def test_held_out_gate_rejects_ambiguous_exporter_payloads(
    tmp_path: Path,
    user_content: str,
    answer: str,
    message: str,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"text":"まったく別の音声学習文です。"}\n',
        encoding="utf-8",
    )
    (paths.dataset / "lfm_train.jsonl").write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": user_content},
                ],
                "completion": [{"role": "assistant", "content": answer}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        evaluation._assert_held_out(paths)


def test_held_out_gate_normalizes_full_width_spacing_and_punctuation(
    tmp_path: Path,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"text":"まったく別の音声学習文です。"}\n',
        encoding="utf-8",
    )
    prompt_case = evaluation.LFM_CASES[0]
    equivalent = str(prompt_case["prompt"]).replace("7", "７").replace("、", " 、 ")
    wrapper = (
        f"{evaluation._LFM_DIALOGUE_HEADER}\n相手: {equivalent}\n{evaluation._LFM_DIALOGUE_SUFFIX}"
    )
    (paths.dataset / "lfm_train.jsonl").write_text(
        json.dumps(
            {
                "prompt": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": wrapper},
                ],
                "completion": [{"role": "assistant", "content": "別の回答です。"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="lfm:lfm-explain:prompt"):
        evaluation._assert_held_out(paths)


def test_held_out_gate_ignores_stale_lfm_file_when_family_is_disabled(tmp_path: Path) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    paths.dataset.mkdir(parents=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"text":"別の音声学習文です。"}\n', encoding="utf-8"
    )
    (paths.dataset / "lfm_train.jsonl").write_text("not-json\n", encoding="utf-8")

    evaluation._assert_held_out(paths, include_lfm=False)


def test_lfm_held_out_evaluation_measures_expected_and_baseline_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "lfm-full"
    calls: list[dict[str, Any]] = []
    cases_by_prompt = {case["prompt"]: case for case in evaluation.LFM_CASES}

    class FakeLFMWorker:
        def call(self, repo_root, command, payload):
            del repo_root
            assert command == "infer"
            calls.append(dict(payload))
            prompt = payload["messages"][-1]["content"]
            case = cases_by_prompt[prompt]
            candidate = payload.get("full_model") or payload.get("adapter")
            answer = case["expected_completion"] if candidate else "まだうまく答えられないよ。"
            return {
                "text": json.dumps(
                    {
                        "text": answer,
                        "voice": {
                            "caption": "自然に話す",
                            "emotion": "NEUTRAL",
                            "events": [],
                        },
                    },
                    ensure_ascii=False,
                )
            }

    monkeypatch.setattr(evaluation, "worker", lambda repo_root, name: FakeLFMWorker())
    report = evaluation._evaluate_lfm(
        tmp_path,
        PersonaConfig(name="alice"),
        {"method": "full"},
        artifact,
    )

    assert report["complete"] is True
    assert report["contract_passed"] is True
    assert report["baseline_contract_passed"] is True
    assert report["candidate_expected_similarity_mean"] == pytest.approx(1.0)
    assert report["candidate_expected_cer_mean"] == pytest.approx(0.0)
    assert report["candidate_expected_wer_mean"] == pytest.approx(0.0)
    assert report["required_phrase_coverage_mean"] == pytest.approx(1.0)
    assert report["base_similarity_regression_max"] == pytest.approx(0.0)
    assert len(calls) == len(evaluation.LFM_CASES) * 2
    candidate_calls = [call for call in calls if call["full_model"] is not None]
    baseline_calls = [
        call for call in calls if call["full_model"] is None and call["adapter"] is None
    ]
    assert len(candidate_calls) == len(evaluation.LFM_CASES)
    assert len(baseline_calls) == len(evaluation.LFM_CASES)
    assert all(call["temperature"] == 0.0 for call in calls)
    assert all(call["max_new_tokens"] == 192 for call in calls)


@pytest.mark.parametrize(
    "generated",
    [
        "plain text",
        "```json\n{}\n```",
        "{}",
        '{"text":"ok","voice":{"emotion":"NEUTRAL","events":[]}}',
        '{"text":"ok","voice":{"caption":"natural","events":[]}}',
        '{"text":"ok","voice":{"caption":"natural","emotion":"NEUTRAL"}}',
        ('{"text":"ok","voice":{"caption":"natural","emotion":"NEUTRAL","events":[1]}}'),
    ],
)
def test_lfm_held_out_output_contract_rejects_incomplete_or_wrapped_json(
    generated: str,
) -> None:
    assert evaluation._parse_lfm_output(generated) is None


def test_lfm_required_numeric_phrase_does_not_accept_a_longer_number() -> None:
    assert evaluation._required_phrase_coverage("答えは12だよ。", ("12",)) == 1.0
    assert evaluation._required_phrase_coverage("答えは120だよ。", ("12",)) == 0.0
    assert (
        evaluation._required_phrase_coverage(
            "今日はゆっくり休んでね。",
            ("ゆっくり休んで",),
        )
        == 1.0
    )


def test_lfm_held_out_baseline_contract_failure_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_by_prompt = {case["prompt"]: case for case in evaluation.LFM_CASES}

    class MalformedBaselineWorker:
        def call(self, repo_root, command, payload):
            del repo_root
            assert command == "infer"
            case = cases_by_prompt[payload["messages"][-1]["content"]]
            if payload.get("full_model") is None and payload.get("adapter") is None:
                return {"text": "not-json"}
            return {
                "text": json.dumps(
                    {
                        "text": case["expected_completion"],
                        "voice": {
                            "caption": "自然に話す",
                            "emotion": "NEUTRAL",
                            "events": [],
                        },
                    },
                    ensure_ascii=False,
                )
            }

    monkeypatch.setattr(
        evaluation,
        "worker",
        lambda repo_root, name: MalformedBaselineWorker(),
    )
    report = evaluation._evaluate_lfm(
        tmp_path,
        PersonaConfig(name="alice"),
        {"method": "full"},
        tmp_path / "lfm-full",
    )

    assert report["contract_passed"] is True
    assert report["baseline_contract_passed"] is False
    assert report["complete"] is False
    assert report["baseline_expected_similarity_mean"] is None
    assert report["base_similarity_regression_max"] is None


def test_lfm_held_out_quality_regression_fails_publication_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases_by_prompt = {case["prompt"]: case for case in evaluation.LFM_CASES}
    calls: list[dict[str, Any]] = []

    class RegressedLFMWorker:
        def call(self, repo_root, command, payload):
            del repo_root, command
            calls.append(dict(payload))
            case = cases_by_prompt[payload["messages"][-1]["content"]]
            candidate = payload.get("full_model") or payload.get("adapter")
            answer = "関係のない回答です。" if candidate else case["expected_completion"]
            return {
                "text": json.dumps(
                    {
                        "text": answer,
                        "voice": {
                            "caption": "自然に話す",
                            "emotion": "NEUTRAL",
                            "events": [],
                        },
                    },
                    ensure_ascii=False,
                )
            }

    monkeypatch.setattr(evaluation, "worker", lambda repo_root, name: RegressedLFMWorker())
    lfm = evaluation._evaluate_lfm(
        tmp_path,
        PersonaConfig(name="alice"),
        {"method": "lora"},
        tmp_path / "lfm-adapter",
    )
    gate = evaluation.evaluate_quality_gate(
        {"summary": _passing_voice()["summary"], "lfm": lfm},
        PersonaConfig(name="alice").training.quality_gate.model_dump(mode="json"),
        validation={"loss": 0.5},
    )

    assert lfm["complete"] is True
    assert lfm["base_similarity_regression_max"] > 0.1
    assert lfm["required_phrase_coverage_mean"] == 0.0
    assert gate["passed"] is False
    failed = {check["metric"] for check in gate["checks"] if not check["passed"]}
    assert "lfm_required_phrase_coverage_mean" in failed
    assert "lfm_base_similarity_regression_max" in failed
    candidate_calls = [call for call in calls if call["adapter"] is not None]
    assert len(candidate_calls) == len(evaluation.LFM_CASES)
    assert all(call["full_model"] is None for call in candidate_calls)


def test_full_conditioning_comparison_reports_all_required_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = PersonaPaths(tmp_path / "persona")
    case_by_id = {case["id"]: case for case in evaluation.CASES}

    def make_wav(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(b"\0\0" * 1_600)
        return path

    candidate = {
        case_id: make_wav(paths.outputs / "candidate" / f"{case_id}.wav") for case_id in case_by_id
    }
    baseline = {
        case_id: make_wav(paths.outputs / "base" / f"{case_id}.wav") for case_id in case_by_id
    }
    modes = {
        mode: {
            case_id: make_wav(paths.outputs / "modes" / mode / f"{case_id}.wav")
            for case_id in case_by_id
        }
        for mode in ("speaker-conditioned", "no-reference", "caption-conditioned")
    }

    def case_for_item(item_id: str) -> dict[str, str]:
        return case_by_id[item_id.rsplit(":", 1)[-1]]

    class FakeWorker:
        def __init__(self, name: str) -> None:
            self.name = name

        def call(self, repo_root, command, payload):
            del repo_root
            if self.name == "diarization":
                assert command == "embed"
                assert set(payload) == {"audio"}
                return {"embedding": [1.0, 0.0]}

            items = payload.get("items")
            assert isinstance(items, list)
            if self.name == "asr":
                return {
                    "results": [
                        {
                            "id": item["id"],
                            "ok": True,
                            "result": {
                                "segments": [
                                    {"text": case_for_item(item["id"])["text"]}
                                ]
                            },
                        }
                        for item in items
                    ]
                }
            assert self.name == "sense"
            return {
                "results": [
                    {
                        "id": item["id"],
                        "ok": True,
                        "result": {
                            "emotion": case_for_item(item["id"])["emotion"],
                            "events": [],
                        },
                    }
                    for item in items
                ]
            }
    monkeypatch.setattr(evaluation, "worker", lambda repo_root, name: FakeWorker(name))
    monkeypatch.setattr(evaluation, "_identity", lambda repo_root, paths: ([1.0, 0.0], {}))

    report = evaluation._analyze_voice_sets(
        tmp_path,
        paths,
        PersonaConfig(name="alice"),
        candidate=candidate,
        baseline=baseline,
        probes=modes,
    )

    assert report["complete"] is True
    assert report["summary"] == {
        "cer_mean": 0.0,
        "wer_mean": 0.0,
        "unseen_text_similarity_mean": 1.0,
        "duration_ratio_error_mean": 0.0,
        "emotion_accuracy": 1.0,
        "speaker_similarity_mean": 1.0,
        "base_cer_regression": 0.0,
    }
    assert {row["mode"] for row in report["mode_comparison"]} == set(modes)
    for row in report["mode_comparison"]:
        assert row["complete"] is True
        assert row["summary"] == report["summary"]
        assert len(row["cases"]) == len(evaluation.CASES)
