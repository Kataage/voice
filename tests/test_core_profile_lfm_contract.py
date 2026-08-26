from __future__ import annotations

import json
from pathlib import Path

import pytest

from personavoice import inference
from personavoice.config import PersonaConfig
from personavoice.lfm_contract import (
    LFM_CONTRACT_FINGERPRINT,
    LFMContractError,
    build_lfm_messages,
    build_lfm_system_prompt,
    normalize_lfm_output,
)
from personavoice.pipeline import _prepare_fingerprint
from personavoice.profile import (
    CoreProfile,
    ProfileExpression,
    ProfileIdentity,
    ProfileRelationship,
    StableFact,
    load_core_profile,
)
from personavoice.project import PersonaPaths, init_persona


def _profile() -> CoreProfile:
    return CoreProfile(
        identity=ProfileIdentity(
            display_name="alice",
            self_concept="海辺の町で暮らす案内役",
            role="相手の話を聞き、一緒に考える相棒",
            background=("海の近くで暮らしている",),
            first_person="ぼく",
            preferred_address="かたあげさん",
        ),
        stable_facts=(
            StableFact(key="home", value="海辺の町", immutable=True),
        ),
        relationships=(
            ProfileRelationship(
                name="かたあげさん",
                relation="大切な会話相手",
                address="かたあげさん",
            ),
        ),
        conversation_rules=("分からないことは分からないと伝える",),
        expression=ProfileExpression(
            baseline_emotion="NEUTRAL",
            tendencies=("必要なときだけ明るさや真剣さを強める",),
        ),
    )


def test_core_profile_is_strict_versioned_and_old_personas_get_default(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    assert paths.core_profile.is_file()
    profile = load_core_profile(paths.core_profile, persona_name="alice")
    assert profile.schema_version == 1
    assert profile.identity.display_name == "alice"

    custom = _profile()
    custom.save(paths.core_profile)
    loaded = load_core_profile(paths.core_profile, persona_name="alice")
    assert loaded == custom
    assert loaded.fingerprint == custom.fingerprint
    assert loaded.canonical_json() == custom.canonical_json()

    old_persona = PersonaPaths(tmp_path / "old")
    old_persona.root.mkdir()
    default = load_core_profile(old_persona.core_profile, persona_name="alice")
    assert default.identity.display_name == "alice"
    with pytest.raises(ValueError):
        CoreProfile.model_validate(
            {"identity": {"display_name": "alice"}, "unexpected": True}
        )


def test_profile_edit_is_runtime_only_and_does_not_change_prepare_fingerprint(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    before = _prepare_fingerprint(paths, cfg)
    _profile().save(paths.core_profile)
    after = _prepare_fingerprint(paths, cfg)
    assert before == after


def test_system_prompt_separates_profile_style_history_prompt_and_output_contract():
    profile = _profile()
    prompt = build_lfm_system_prompt(profile)
    assert "Core Profile" in prompt
    assert "learned persona behavior" in prompt
    assert "conversation history" in prompt
    assert "current user prompt" in prompt
    assert "海辺の町" in prompt
    assert "ぼく" in prompt
    assert "voice.caption" in prompt
    assert len(prompt) < 12_000

    messages, diagnostics = build_lfm_messages(
        profile,
        [{"role": "user", "content": "前の話"}],
        "今の質問",
    )
    assert diagnostics == ()
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "今の質問"}


def test_shared_cross_version_fixture_is_loadable():
    fixture = Path(__file__).parent / "fixtures" / "persona_runtime_contract.json"
    value = json.loads(fixture.read_text(encoding="utf-8"))
    profile = CoreProfile.model_validate(value["profile"])
    messages, _ = build_lfm_messages(
        profile,
        value["history"],
        value["cases"][0]["prompt"],
    )
    assert {case["id"] for case in value["cases"]} == {
        "identity",
        "immutable_fact",
        "first_person_addressing",
        "multi_turn_continuity",
        "learned_style_preservation",
        "neutral",
        "happy_excited",
        "sad_serious",
        "surprise_gasp",
        "laughter",
        "breath_sigh",
        "mixed_speech_event",
    }
    assert "Core Profile" in messages[0]["content"]


@pytest.mark.parametrize(
    ("raw", "text", "emotion", "events", "dropped"),
    [
        (
            '{"text":"うれしい！","voice":{"caption":"弾む","emotion":"happy","events":[]}}',
            "うれしい！",
            "HAPPY",
            (),
            False,
        ),
        (
            '{"text":"えっ？","voice":{"caption":"息を呑む","emotion":"SURPRISED","events":["gasp"]}}',
            "えっ？",
            "SURPRISED",
            ("Breath",),
            False,
        ),
        (
            '{"text":"ふふ","voice":{"caption":"小さく","emotion":"NEUTRAL","events":["laughter","unsupported"]}}',
            "ふふ",
            "NEUTRAL",
            ("Laughter",),
            True,
        ),
        (
            '{"text":"大丈夫だよ。","voice":{"caption":"落ち着いて","emotion":"serious","events":["sigh"]}}',
            "大丈夫だよ。",
            "NEUTRAL",
            ("Breath",),
            False,
        ),
        (
            '{"text":"少し寂しいね。","voice":{"caption":"静かに","emotion":"sad","events":[]}}',
            "少し寂しいね。",
            "SAD",
            (),
            False,
        ),
    ],
)
def test_normalizer_uses_only_canonical_voice_vocabulary(raw, text, emotion, events, dropped):
    plan = normalize_lfm_output(raw)
    assert plan.text == text
    assert plan.emotion == emotion
    assert plan.events == events
    assert ("unsupported_event_dropped" in plan.diagnostics) is dropped


def test_normalizer_preserves_explicit_non_verbal_only_plan():
    plan = normalize_lfm_output(
        '{"text":"","voice":{"caption":"短く息を呑む","emotion":"SURPRISED","events":["gasp"]}}'
    )
    assert plan.non_verbal_only is True
    assert plan.text == ""
    assert plan.events == ("Breath",)
    with pytest.raises(LFMContractError, match="explicit non-verbal-only"):
        normalize_lfm_output(
            '{"text":"   ","voice":{"caption":"短く","emotion":"SURPRISED","events":["gasp"]}}'
        )


def test_normalizer_recovers_extra_fields_but_strict_evaluation_rejects_them():
    raw = '{"text":"了解したよ。","voice":{"caption":"自然に","emotion":"NEUTRAL","events":[],"extra":true},"trace":"debug"}'
    plan = normalize_lfm_output(raw)
    assert "extra_root_fields_dropped" in plan.diagnostics
    assert "extra_voice_fields_dropped" in plan.diagnostics
    with pytest.raises(LFMContractError, match="unsupported fields"):
        normalize_lfm_output(raw, allow_field_recovery=False)


@pytest.mark.parametrize(
    "raw",
    ["", "   ", "???", "????????????????", '{"voice":{"events":[]}}'],
)
def test_normalizer_rejects_empty_missing_or_degenerate_spoken_text(raw):
    with pytest.raises(LFMContractError):
        normalize_lfm_output(raw)


def test_runtime_retries_bounded_and_returns_exact_normalized_irodori_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = PersonaPaths(tmp_path / "persona")
    paths.root.mkdir()
    profile = _profile()
    profile.save(paths.core_profile)
    cfg = PersonaConfig(name="alice", consent={"authorized": True})
    calls: list[dict] = []

    class FakeWorker:
        def call(self, repo_root, command, payload):
            del repo_root
            assert command == "infer"
            calls.append(payload)
            if len(calls) == 1:
                return {"text": "???"}
            return {
                "text": json.dumps(
                    {
                        "text": "うれしいよ！",
                        "voice": {
                            "caption": "少し弾むように",
                            "emotion": "happy",
                            "events": ["unsupported"],
                        },
                    },
                    ensure_ascii=False,
                )
            }

    audio = tmp_path / "reply.wav"
    monkeypatch.setattr(inference, "worker", lambda *_args: FakeWorker())
    monkeypatch.setattr(inference, "synthesize", lambda *args, **kwargs: [audio])

    result = inference.chat_turn(tmp_path, paths, cfg, "どう思う？")
    assert len(calls) == 2
    assert calls[0]["max_new_tokens"] == 384
    assert calls[1]["max_new_tokens"] == 256
    assert "海辺の町" in calls[0]["messages"][0]["content"]
    assert result["voice"] == {
        "caption": "少し弾むように",
        "emotion": "HAPPY",
        "events": [],
    }
    assert result["provenance"]["lfm"]["attempts"] == 2
    assert result["provenance"]["lfm"]["contract_fingerprint"] == LFM_CONTRACT_FINGERPRINT
    assert result["provenance"]["irodori_handoff"] == {
        "text": "うれしいよ！",
        "voice": result["voice"],
        "non_verbal_only": False,
    }


def test_runtime_never_sends_failed_empty_output_to_irodori(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    paths = PersonaPaths(tmp_path / "persona")
    paths.root.mkdir()
    cfg = PersonaConfig(name="alice", consent={"authorized": True})
    syntheses: list[tuple] = []

    class EmptyWorker:
        def call(self, repo_root, command, payload):
            del repo_root, command, payload
            return {"text": ""}

    monkeypatch.setattr(inference, "worker", lambda *_args: EmptyWorker())
    monkeypatch.setattr(
        inference,
        "synthesize",
        lambda *args, **kwargs: syntheses.append((args, kwargs)),
    )
    with pytest.raises(RuntimeError, match="bounded attempts"):
        inference.chat_turn(tmp_path, paths, cfg, "返事して")
    assert syntheses == []
