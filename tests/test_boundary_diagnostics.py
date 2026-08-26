from __future__ import annotations

import struct
import subprocess
import wave
from pathlib import Path

import pytest

from personavoice import boundary_diagnostics as diagnostics
from personavoice import inference
from personavoice.config import PersonaConfig
from personavoice.pipeline import _prepare_fingerprint
from personavoice.project import init_persona
from personavoice.training_plan import build_training_plan


def _wav(path: Path, *, leading: float = 0.05, tone: float = 0.30, trailing: float = 0.10) -> None:
    rate = 16000
    samples: list[int] = []
    samples.extend([0] * round(rate * leading))
    for index in range(round(rate * tone)):
        samples.append(int(12000 * (index % 40) / 40))
    samples.extend([0] * round(rate * trailing))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def test_inference_diagnostic_matrix_is_stable_and_explicit() -> None:
    assert diagnostics.build_inference_diagnostic_matrix() == (
        {"id": "A", "duration_scale": 1.0, "trim_tail": True},
        {"id": "B", "duration_scale": 1.0, "trim_tail": False},
        {"id": "C", "duration_scale": 1.1, "trim_tail": True},
        {"id": "D", "duration_scale": 1.1, "trim_tail": False},
    )
    with pytest.raises(ValueError, match="differ from"):
        diagnostics.build_inference_diagnostic_matrix(margin_scale=1.0)


def test_audio_boundary_metrics_find_voiced_region_and_final_envelope(tmp_path: Path) -> None:
    output = tmp_path / "sample.wav"
    _wav(output)

    metrics = diagnostics.analyze_wav_boundary(output)

    assert metrics["first_voiced_frame"] is not None
    assert metrics["last_voiced_frame"] is not None
    assert metrics["first_voiced_seconds"] >= 0
    assert metrics["last_voiced_seconds"] > metrics["first_voiced_seconds"]
    assert metrics["final_energy_envelope"]
    assert metrics["duration_seconds"] == pytest.approx(0.45, abs=0.01)


def test_inference_boundary_settings_are_not_training_or_prepare_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = init_persona(tmp_path, "alice", authorized=True)
    (paths.dataset / "irodori_source.jsonl").write_text(
        '{"audio":"clip.flac","text":"test"}\n', encoding="utf-8"
    )
    (paths.dataset / "lfm_train.jsonl").write_text(
        '{"messages":[{"role":"assistant","content":"test"}]}\n', encoding="utf-8"
    )
    manifest = paths.dataset / "irodori_manifest.jsonl"
    manifest.write_text('{"latent_path":"latent.pt","text":"test"}\n', encoding="utf-8")
    cfg = PersonaConfig.load(paths.config)
    first_prepare = _prepare_fingerprint(paths, cfg)
    first_plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    cfg.inference.duration_scale = 1.35
    cfg.inference.trim_tail = False
    cfg.inference.tail_window_size = 31
    second_prepare = _prepare_fingerprint(paths, cfg)
    second_plan = build_training_plan(repo_root, paths, cfg, irodori_manifest=manifest)

    assert second_prepare == first_prepare
    assert second_plan.as_dict() == first_plan.as_dict()


def test_synthesize_forwards_explicit_boundary_contract_and_true_caption_off(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    cfg.inference.default_candidates = 1
    vendor = tmp_path / "vendor" / "Irodori-TTS"
    vendor.mkdir(parents=True)
    base = tmp_path / "base.safetensors"
    codec = tmp_path / "codec.pth"
    base.write_bytes(b"base")
    codec.write_bytes(b"codec")
    captured: dict[str, object] = {}

    monkeypatch.setattr(inference, "vendor_dir", lambda _root: vendor)
    monkeypatch.setattr(inference, "base_checkpoint", lambda _root: base)
    monkeypatch.setattr(inference, "codec_checkpoint", lambda _root: codec)
    monkeypatch.setattr(inference, "configured_backend", lambda _root: "cpu")
    monkeypatch.setattr(inference, "backend_device", lambda _backend: "cpu")
    monkeypatch.setattr(inference, "local_model_env", lambda *_args, **_kwargs: {})

    def fake_run(args, **kwargs):
        captured["argv"] = [str(value) for value in args]
        captured["kwargs"] = kwargs
        argv = captured["argv"]
        output = Path(argv[argv.index("--output-wav") + 1])
        _wav(output)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="info: predicted duration frames=42.0, scale=1.350, using_frames=57 (1.140s).",
            stderr="",
        )

    monkeypatch.setattr(inference, "run", fake_run)
    metadata: dict[str, object] = {}
    outputs = inference.synthesize(
        tmp_path,
        paths,
        cfg,
        "テスト",
        candidates=1,
        seed=7,
        caption_conditioning=False,
        duration_scale=1.35,
        trim_tail=False,
        metadata=metadata,
        capture_logs=True,
    )

    argv = captured["argv"]
    assert outputs == [Path(argv[argv.index("--output-wav") + 1])]
    assert argv[argv.index("--duration-scale") + 1] == "1.35"
    assert "--no-trim-tail" in argv
    assert argv[argv.index("--tail-window-size") + 1] == "20"
    assert "--caption" not in argv
    assert metadata["duration_scale"] == 1.35
    assert metadata["trim_tail"] is False
    assert metadata["seed"] == 7


def test_boundary_report_is_inference_only_and_keeps_generation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    case = (diagnostics.DEFAULT_BOUNDARY_EVALUATION_CASES[0],)
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base")

    def fake_synthesize(*_args, **kwargs):
        output = Path(kwargs["output"])
        _wav(output)
        metadata = kwargs["metadata"]
        metadata.update(
            {
                "seed": kwargs["seed"],
                "checkpoint": str(checkpoint),
                "method": "base-only" if kwargs.get("base_only") else "base",
                "reference_mode": kwargs.get("reference_mode"),
                "reference_fingerprint": "r" * 64,
                "duration_scale": kwargs.get("duration_scale"),
                "trim_tail": kwargs.get("trim_tail"),
                "tail_window_size": 20,
                "tail_std_threshold": 0.05,
                "tail_mean_threshold": 0.1,
                "stdout": (
                    "info: predicted duration frames=42.0, scale=1.000, "
                    "using_frames=42 (0.840s); final_loss=0.125"
                ),
            }
        )
        return [output]

    monkeypatch.setattr(inference, "synthesize", fake_synthesize)
    report = diagnostics.run_boundary_diagnostics(
        tmp_path,
        paths,
        cfg,
        seed=99,
        cases=case,
        output_dir=paths.outputs / "diagnostic-test",
        include_asr=False,
        include_sense=False,
    )

    assert Path(report["report"]).is_file()
    assert len(report["duration_tail_records"]) == 4
    assert len(report["leading_isolation_records"]) == 12
    assert report["policy"]["training_data_modified"] is False
    assert report["policy"]["prepare_rerun_required"] is False
    assert report["policy"]["training_rerun_required"] is False
    assert report["policy_recommendation"]["status"] == "insufficient-evidence"
    assert report["provenance"]["irodori_source_revision"]
    assert report["duration_tail_records"][0]["final_loss_evidence"]["value"] == pytest.approx(0.125)


def test_boundary_report_batches_asr_and_sense_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    case = (diagnostics.DEFAULT_BOUNDARY_EVALUATION_CASES[0],)
    checkpoint = tmp_path / "base.safetensors"
    checkpoint.write_bytes(b"base")
    calls: list[tuple[str, str, int]] = []

    def fake_synthesize(*_args, **kwargs):
        output = Path(kwargs["output"])
        _wav(output)
        kwargs["metadata"].update(
            {
                "seed": kwargs["seed"],
                "checkpoint": str(checkpoint),
                "method": "base",
                "reference_mode": "none",
                "reference_fingerprint": "r" * 64,
                "duration_scale": 1.0,
                "trim_tail": True,
                "tail_window_size": 20,
                "tail_std_threshold": 0.05,
                "tail_mean_threshold": 0.1,
                "stdout": "",
            }
        )
        return [output]

    class FakeWorker:
        def __init__(self, name: str):
            self.name = name

        def call(self, _root, command, payload):
            calls.append((self.name, command, len(payload["items"])))
            rows = []
            for item in payload["items"]:
                if self.name == "asr":
                    text = str(case[0]["text"])
                    rows.append(
                        {
                            "id": item["id"],
                            "ok": True,
                            "result": {
                                "language": "ja",
                                "duration": 0.45,
                                "language_probability": 1.0,
                                "segments": [
                                    {
                                        "start": 0.08,
                                        "end": 0.4,
                                        "text": text,
                                        "words": [
                                            {"start": 0.08, "end": 0.4, "word": text, "probability": 1.0}
                                        ],
                                    }
                                ],
                            },
                        }
                    )
                else:
                    rows.append(
                        {
                            "id": item["id"],
                            "ok": True,
                            "result": {"raw": "<|NEUTRAL|><|Speech|>", "emotion": "NEUTRAL", "events": ["Speech"]},
                        }
                    )
            return {"results": rows}

    monkeypatch.setattr(inference, "synthesize", fake_synthesize)
    monkeypatch.setattr(diagnostics, "worker", lambda _root, name: FakeWorker(name))
    report = diagnostics.run_boundary_diagnostics(
        tmp_path,
        paths,
        cfg,
        seed=99,
        cases=case,
        output_dir=paths.outputs / "diagnostic-batch-test",
        include_asr=True,
        include_sense=True,
    )

    assert calls == [("asr", "batch_transcribe", 16), ("sense", "batch_analyze", 16)]
    assert report["duration_tail_records"][0]["asr"]["cer"] == 0.0
    assert report["duration_tail_records"][0]["asr"]["final_token_preserved"] is True
    assert report["duration_tail_records"][0]["sense"]["emotion"] == "NEUTRAL"
