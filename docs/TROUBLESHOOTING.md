# Troubleshooting

Run `uv run persona doctor --deep` after setup. It loads the local ASR, diarization, SenseVoice, LFM, and Seed-VC stacks with Hugging Face offline mode enabled and reports the exact failing worker under `worker_health`.

## HF_TOKEN / pyannote

The first download of `pyannote/speaker-diarization-community-1` is gated. Accept the Hugging Face usage terms and set `HF_TOKEN` in the current shell. PersonaVoice does not store the token. Once the local Community-1 model is present, later setup runs reuse it.

## Irodori offline model lookup

Re-run `persona setup` if Irodori cannot find DACVAE or ModernBERT. Setup places `Aratako/Semantic-DACVAE-Japanese-32dim` and `sbintuitions/modernbert-ja-310m` in the same `HUGGINGFACE_HUB_CACHE` used during offline runtime.

## Target speaker is uncertain

Put cleaner authorized target-only clips in `personas/<name>/identity/`. A change to `identity/` is included in the prepare fingerprint, so the speaker-identification-dependent dataset is rebuilt automatically.

## Large dataset processing

ASR, pyannote, and SenseVoice run in batch workers, loading the model once per preparation batch. Existing source/clip caches are reused after interruption.

## Training inputs changed

Training has its own fingerprint. If the derived dataset or training configuration changes, stale Irodori latents/checkpoints, LFM adapters, and optional Seed-VC persona checkpoints are invalidated before retraining. Use `--force` to explicitly rebuild.

## Irodori out of memory

PersonaVoice patches the official Irodori training config with a conservative NVIDIA VRAM-based batch profile. Irodori backend can be selected during setup:

```bash
uv run persona setup --backend cu128
uv run persona setup --backend cpu
uv run persona setup --backend rocm
uv run persona setup --backend xpu
```

If a run is interrupted with the same input fingerprint, rerun the command and PersonaVoice resumes from available upstream checkpoints rather than invalidating them.

## Seed-VC problems

Seed-VC is an archived upstream with an older dependency stack. It is deliberately isolated under `workers/seed_vc`. Delete only `workers/seed_vc/.venv` and rerun `persona setup` to rebuild that environment without touching the root, Irodori, or LFM environments.

## API refuses non-loopback binding

PersonaVoice has no network authentication and refuses non-loopback binding by default. For a trusted network only:

```bash
uv run persona serve --host 0.0.0.0 --allow-remote
```

Use firewall or reverse-proxy authentication before wider exposure.
