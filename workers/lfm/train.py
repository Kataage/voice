from __future__ import annotations

import argparse
from pathlib import Path

import torch
from checkpoint_contract import latest_complete_checkpoint, prune_incomplete_checkpoints
from datasets import load_dataset
from model_contract import audited_attention_lora_targets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
REVISION_MARKER = ".personavoice-revision"
ADAPTER_REVISION_MARKER = ".personavoice-base-revision"


def _model_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _verify_base(base: Path) -> None:
    if not (base / "config.json").is_file():
        raise FileNotFoundError(f"Pinned LFM base model is missing: {base}")
    marker = base / REVISION_MARKER
    actual = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if actual != MODEL_REVISION:
        raise RuntimeError(
            "LFM fine-tuning base does not match the audited revision: "
            f"expected {MODEL_REVISION}, got {actual!r}. Run `persona setup --download-models`."
        )


def _adapter_weight(output: Path) -> Path | None:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = output / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    base = Path(args.base).resolve()
    output = Path(args.output).resolve()
    _verify_base(base)
    output.mkdir(parents=True, exist_ok=True)
    # A power loss can leave a numerically newer Trainer directory only partly
    # written. Remove only those derived incomplete checkpoints before asking
    # Trainer to resume, then select the highest verified checkpoint explicitly.
    prune_incomplete_checkpoints(output)
    resume = latest_complete_checkpoint(output)

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    if len(dataset) < 2:
        raise RuntimeError("LFM fine-tuning needs at least two conversational examples")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=_model_dtype(),
        local_files_only=True,
    )
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=audited_attention_lora_targets(model),
        task_type="CAUSAL_LM",
    )
    has_cuda = torch.cuda.is_available()
    batch = (
        2
        if has_cuda and torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3
        else 1
    )
    grad_accum = 8 if batch == 1 else 4
    config = SFTConfig(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        use_cpu=not has_cuda,
        bf16=bool(has_cuda and torch.cuda.is_bf16_supported()),
        fp16=bool(has_cuda and not torch.cuda.is_bf16_supported()),
        logging_steps=5,
        # Periodic step checkpoints limit loss from hard crashes/OOMs while the
        # JIT checkpoint below covers graceful SIGTERM shutdowns.
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        save_only_model=False,
        enable_jit_checkpoint=True,
        max_length=2048,
        completion_only_loss=True,
        report_to="none",
        gradient_checkpointing=True,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    if not (output / "adapter_config.json").is_file() or _adapter_weight(output) is None:
        raise RuntimeError("LFM fine-tuning completed without a complete PEFT adapter")
    (output / ADAPTER_REVISION_MARKER).write_text(MODEL_REVISION + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
