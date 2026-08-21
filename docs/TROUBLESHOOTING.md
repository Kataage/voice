# Troubleshooting

Run `uv run persona doctor --deep` after setup. It verifies the local source revisions, lockfiles, model asset pins/checksums, then loads the local ASR, diarization, SenseVoice, LFM, Seed-VC, and Irodori stacks with network access disabled for normal runtime. The exact failing component is reported under `model_asset_integrity`, `vendor_integrity`, or `worker_health`.

## HF_TOKEN / pyannote

The first download of `pyannote/speaker-diarization-community-1` is gated. Accept the Hugging Face usage terms and set `HF_TOKEN` in the current shell. PersonaVoice does not store the token. Once the local Community-1 model is present, later setup runs reuse it.

If `doctor --deep` says pyannote cannot load offline, set `HF_TOKEN` again and rerun `uv run persona setup`. This re-materializes any gated/transitive files needed by the local pipeline; normal prepare/inference does not require the token afterward.

## Pinned model revision mismatch

LFM and faster-whisper snapshots contain `.personavoice-revision`. If setup finds a legacy directory without that marker, or a marker for a different revision, it removes only PersonaVoice's materialized model directory and reconstructs it from the audited revision. Hugging Face's shared cache under `models/hf-cache` is retained, so unchanged blobs can be reused.

Do not manually edit `.personavoice-revision`. If `model_asset_integrity` reports a mismatch, rerun:

```bash
uv run persona setup
uv run persona doctor --deep
```

## Irodori checksum mismatch

The Irodori v4.1 Small checkpoint and DACVAE `weights.pth` are SHA256-verified during setup and again by `doctor --deep`. A checksum mismatch means the local file is damaged or no longer matches the audited asset.

Remove only the path reported by doctor, then rerun `uv run persona setup`. Do not replace model files from an arbitrary mirror: the expected hashes are defined in `src/personavoice/model_assets.py`.

## Irodori offline model lookup

PersonaVoice passes the local DACVAE file directly to upstream Irodori with `--codec-repo`, and setup materializes the exact ModernBERT revision required by the pinned v4 configuration in the same `HUGGINGFACE_HUB_CACHE` used by offline runtime.

If Irodori still reports a missing text encoder, rerun `uv run persona setup` while online, then `uv run persona doctor --deep` before disconnecting the machine.

## Target speaker is uncertain

Put cleaner authorized target-only clips in `personas/<name>/identity/`. A change to `identity/` is included in the prepare fingerprint, so the speaker-identification-dependent dataset is rebuilt automatically.

## Large dataset processing

ASR, pyannote, and SenseVoice run in batch workers, loading the model once per preparation batch. Existing source/clip caches are reused after interruption. `persona prepare --force` explicitly invalidates dependent ASR/diarization/identity/SenseVoice/clip caches even when the previous run failed; a normal rerun with the same fingerprint preserves those expensive caches for resume.

## Training inputs changed

Training has its own fingerprint. If the derived dataset or training configuration changes, stale Irodori latents/checkpoints, LFM adapters, and optional Seed-VC persona checkpoints are invalidated before retraining. Use `--force` to explicitly rebuild.

## Irodori backend / out of memory

Irodori backend is selected during setup and recorded in `.runtime/setup.json`:

```bash
uv run persona setup --backend cu128
uv run persona setup --backend cpu
uv run persona setup --backend rocm
uv run persona setup --backend xpu
```

That recorded backend is used consistently for manifest creation, Irodori training, synthesis, and deep doctor. An explicit CPU backend does not switch itself back to CUDA merely because an NVIDIA GPU is visible.

PersonaVoice patches the official Irodori training config with a conservative batch profile. NVIDIA VRAM-based sizing is used only for `cu128`; CPU/ROCm/XPU use the conservative profile unless a future backend-specific profile is explicitly implemented.

If a run is interrupted with the same input fingerprint, rerun the command and PersonaVoice resumes from available upstream checkpoints rather than invalidating them.

## Reference mode errors

`inference.reference_mode` is intentional rather than advisory:

- `auto`: prefer Speaker Inversion embedding, then prepared audio references, then unconditioned synthesis.
- `speaker-embed`: require `checkpoint_final.speaker.safetensors`; missing embedding is an error.
- `audio`: require the prepared reference bank; an empty bank is an error.
- An explicit CLI/API reference overrides the default mode and is always treated as an audio reference.

This prevents a requested conditioning method from silently falling back to a different voice identity path.

## Seed-VC problems

Seed-VC is an archived upstream with an older dependency stack. It is deliberately isolated under `workers/seed_vc`. Delete only `workers/seed_vc/.venv` and rerun `persona setup` to rebuild that environment without touching the root, Irodori, or LFM environments.

## API refuses non-loopback binding

PersonaVoice has no network authentication and refuses non-loopback binding by default. For a trusted network only:

```bash
uv run persona serve --host 0.0.0.0 --allow-remote
```

Use firewall or reverse-proxy authentication before wider exposure.
