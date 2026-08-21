from __future__ import annotations

ATTENTION_LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "out_proj")


def audited_attention_lora_targets(model) -> list[str]:
    """Return exact LFM2 attention projection module names for LoRA.

    Liquid's published TRL recipe targets q/k/v/output attention projections.
    The actual Hugging Face LFM2 implementation names the output projection
    ``out_proj`` (not the stale ``o_proj`` spelling shown in the recipe), while
    ShortConv blocks also contain an ``out_proj``. Resolve exact ``self_attn``
    module paths from the loaded audited model so PEFT cannot accidentally adapt
    the convolutional projection with an endswith match.
    """

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
    if len(set(counts.values())) != 1:
        raise RuntimeError(
            "Pinned LFM attention projection counts are inconsistent: "
            f"{counts}. Update/re-audit PersonaVoice before fine-tuning."
        )

    return sorted(name for names in found.values() for name in names)
