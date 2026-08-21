from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_REVISION = "b31023f2d69b95fbd7876898f8de9fae90e8afbd"
REVISION_MARKER = ".personavoice-revision"


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
    _verify_base(base)
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    batch = (
        2
        if torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory >= 16 * 1024**3
        else 1
    )
    grad_accum = 8 if batch == 1 else 4
    config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.learning_rate,
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        max_length=2048,
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
    resume = True if list(Path(args.output).glob("checkpoint-*")) else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
