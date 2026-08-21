from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from personavoice import dataset


def _utterance(
    *,
    utterance_id: str,
    source_id: str,
    start: float,
    text: str,
    speaker: str,
    target: bool,
    caption: str = "自然に話している。",
    emotion: str = "NEUTRAL",
) -> dict:
    return {
        "id": utterance_id,
        "source_id": source_id,
        "source_path": "conversation.wav",
        "start": start,
        "end": start + 1.0,
        "speaker": speaker,
        "target": target,
        "speaker_similarity": 0.9 if target else None,
        "speaker_coverage": 1.0,
        "overlap_ratio": 0.0,
        "text": text,
        "text_annotated": text,
        "emotion": emotion,
        "events": [],
        "caption": caption,
        "audio_path": None,
        "quality": 0.9,
    }


def test_lfm_export_uses_conversational_prompt_completion(tmp_path: Path):
    master = tmp_path / "master.sqlite3"
    output = tmp_path / "lfm_train.jsonl"
    dataset.replace_utterances(
        master,
        [
            _utterance(
                utterance_id="a",
                source_id="source",
                start=0.0,
                text="今日はどうだった？",
                speaker="other",
                target=False,
            ),
            _utterance(
                utterance_id="b",
                source_id="source",
                start=1.0,
                text="めっちゃ楽しかったよ！",
                speaker="target",
                target=True,
                caption="明るく楽しそうに話している。",
                emotion="HAPPY",
            ),
        ],
    )

    assert dataset.export_lfm(master, output, "alice") == 1
    example = json.loads(output.read_text(encoding="utf-8").strip())

    assert "messages" not in example
    assert [message["role"] for message in example["prompt"]] == ["system", "user"]
    assert [message["role"] for message in example["completion"]] == ["assistant"]
    answer = json.loads(example["completion"][0]["content"])
    assert answer["text"] == "めっちゃ楽しかったよ！"
    assert answer["voice"]["emotion"] == "HAPPY"
    assert answer["voice"]["caption"] == "明るく楽しそうに話している。"


def _load_model_contract():
    path = Path(__file__).resolve().parents[1] / "workers" / "lfm" / "model_contract.py"
    spec = importlib.util.spec_from_file_location("personavoice_test_lfm_contract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lfm_lora_contract_targets_attention_only():
    contract = _load_model_contract()

    class FakeModel:
        def named_modules(self):
            names = [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.self_attn.k_proj",
                "model.layers.0.self_attn.v_proj",
                "model.layers.0.self_attn.out_proj",
                "model.layers.1.conv.in_proj",
                "model.layers.1.conv.out_proj",
                "model.layers.1.feed_forward.w1",
                "model.layers.1.feed_forward.w2",
                "model.layers.1.feed_forward.w3",
            ]
            return [(name, object()) for name in names]

    targets = contract.audited_attention_lora_targets(FakeModel())
    assert targets == [
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.out_proj",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.v_proj",
    ]
    assert all(".conv." not in target for target in targets)
    assert all("feed_forward" not in target for target in targets)


def test_lfm_lora_contract_fails_if_projection_disappears():
    contract = _load_model_contract()

    class BrokenModel:
        def named_modules(self):
            return [
                ("model.layers.0.self_attn.q_proj", object()),
                ("model.layers.0.self_attn.k_proj", object()),
                ("model.layers.0.self_attn.v_proj", object()),
            ]

    with pytest.raises(RuntimeError, match="missing=.*out_proj"):
        contract.audited_attention_lora_targets(BrokenModel())


def test_lfm_worker_and_trainer_are_pinned_to_jp_202606_contract():
    root = Path(__file__).resolve().parents[1]
    worker = (root / "workers" / "lfm" / "worker.py").read_text(encoding="utf-8")
    trainer = (root / "workers" / "lfm" / "train.py").read_text(encoding="utf-8")

    assert 'MODEL_ID = "LiquidAI/LFM2.5-1.2B-JP-202606"' in worker
    assert "b31023f2d69b95fbd7876898f8de9fae90e8afbd" in worker
    assert "temperature=float(payload.get(\"temperature\", 0.1))" in worker
    assert "top_k=50" in worker
    assert "repetition_penalty=1.05" in worker
    assert "top_p=" not in worker
    assert "completion_only_loss=True" in trainer
    assert "audited_attention_lora_targets(model)" in trainer
