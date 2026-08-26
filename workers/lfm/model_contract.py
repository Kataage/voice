from __future__ import annotations

import re

ATTENTION_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj")
_EXPLICIT_PATH_FIELDS = frozenset(
    {
        "_name_or_path",
        "name_or_path",
        "path",
        "file",
        "directory",
        "cache_dir",
        "tokenizer_file",
        "vocab_file",
        "merges_file",
        "added_tokens_file",
        "special_tokens_map_file",
        "tokenizer_config_file",
        "chat_template_file",
        "model_file",
    }
)


def _path_field(name: str | None) -> bool:
    if name is None:
        return False
    normalized = name.strip().lower()
    return normalized in _EXPLICIT_PATH_FIELDS or normalized.endswith(
        ("_path", "_file", "_dir", "_directory")
    )


def json_contains_absolute_local_path(value: object, *, field_name: str | None = None) -> bool:
    """Detect serialized machine paths without mistaking tokenizer tokens for paths.

    Tokenizer JSON legitimately contains BPE merge pieces such as ``/``, ``//``
    and ``/Ċ``. A context-free scan therefore rejects the exact audited LFM
    tokenizer. Windows drive paths are unambiguous and rejected in every string;
    POSIX/UNC/file-URI values are rejected in fields whose names declare path
    semantics. Lists inherit the containing field name.
    """

    if isinstance(value, dict):
        return any(
            json_contains_absolute_local_path(item, field_name=str(key))
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(
            json_contains_absolute_local_path(item, field_name=field_name) for item in value
        )
    if not isinstance(value, str):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", value) is not None:
        return True
    return _path_field(field_name) and (
        value.startswith(("/", "\\\\")) or value.lower().startswith("file:")
    )


def audited_attention_lora_targets(model) -> list[str]:
    """Return exact LFM2 attention projection module names for LoRA.

    Liquid's published TRL recipe targets q/k/v/output attention projections.
    The actual Hugging Face LFM2 implementation names the output projection
    ``out_proj`` (not the stale ``o_proj`` spelling shown in the recipe), while
    ShortConv blocks also contain an ``out_proj``. Resolve exact ``self_attn``
    module paths from the loaded audited model so PEFT cannot accidentally adapt
    the convolutional projection with an endswith match.
    """

    layer_types = getattr(getattr(model, "config", None), "layer_types", None)
    if not isinstance(layer_types, (list, tuple)):
        raise RuntimeError(
            "Pinned LFM config no longer exposes layer_types. Update/re-audit PersonaVoice "
            "before fine-tuning."
        )
    expected_attention_layers = sum(layer_type == "full_attention" for layer_type in layer_types)
    if expected_attention_layers <= 0:
        raise RuntimeError("Pinned LFM config exposes no full_attention layers.")

    found: dict[str, list[str]] = {suffix: [] for suffix in ATTENTION_LORA_SUFFIXES}
    for name, _module in model.named_modules():
        if ".self_attn." not in name:
            continue
        suffix = name.rsplit(".", 1)[-1]
        if suffix in found:
            found[suffix].append(name)

    missing = [suffix for suffix, names in found.items() if not names]
    if missing:
        raise RuntimeError(
            "Pinned LFM architecture no longer exposes the audited attention LoRA targets: "
            f"missing={missing}. Update/re-audit PersonaVoice before fine-tuning."
        )

    counts = {suffix: len(names) for suffix, names in found.items()}
    if any(count != expected_attention_layers for count in counts.values()):
        raise RuntimeError(
            "Pinned LFM attention projection counts do not match config.layer_types: "
            f"expected_each={expected_attention_layers}, actual={counts}. "
            "Update/re-audit PersonaVoice before fine-tuning."
        )

    return sorted(name for names in found.values() for name in names)
