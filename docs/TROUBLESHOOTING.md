# Troubleshooting

Run `uv run persona doctor --deep` after setup. It verifies the local source revisions, lockfiles, model asset pins/checksums, current GPU/backend compatibility, then loads the local ASR, diarization, SenseVoice, LFM, Seed-VC, and Irodori stacks with network access disabled for normal runtime. The exact failing component is reported under `runtime_hardware`, `model_asset_integrity`, `vendor_integrity`, or `worker_health`.

## Windows FFmpeg / TorchCodec

PersonaVoice itself requires `ffmpeg`/`ffprobe`, and pyannote 4.x brings TorchCodec. On Windows, TorchCodec 0.10 needs FFmpeg 4 through 8 with the shared `avutil`, `avcodec`, `avformat`, `swresample`, and `swscale` DLLs beside the executables. A standalone/static `ffmpeg.exe` is not sufficient.

On Windows, `persona setup` obtains the exact Gyan shared FFmpeg 8.1.1 ZIP from its versioned official GitHub release, verifies the SHA256 independently published in Microsoft WinGet, extracts only its runtime `bin` tree into gitignored `.runtime/tools`, validates the required executable/DLL hashes, and atomically publishes the verified runtime. Normal build/inference/doctor paths never download FFmpeg.

```powershell
uv run --locked persona setup --backend auto
```

A compatible explicit override remains supported through `PERSONAVOICE_FFMPEG_BIN`; an invalid explicit override fails closed rather than being silently ignored. Existing compatible PATH/WinGet installations remain discoverable at runtime, but the default Windows setup is repository-local and reproducible.

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

## GPU backend / GPU交換 / CUDA_VISIBLE_DEVICES

通常はGPU型番を意識せず、次を使ってください。

```bash
uv run persona setup --backend auto
```

`auto`は、実際にlogical CUDA device 0として使われるNVIDIA GPUのcompute capabilityを監査済みwheel matrixと照合します。`CUDA_VISIBLE_DEVICES`で数値indexやGPU UUIDを指定している場合もそのdevice 0を基準にします。x86_64/Windowsの現在の固定stackでは、Pascal 6.x・Volta 7.0は`cu126`、Turing 7.5以降の監査済み世代は`cu128`を選択します。未知/未監査世代や安全に識別できないGPUはCUDAを推測せずCPUへfail-closedします。

明示指定も可能ですが、現在GPUと非互換なら環境を変更する前に拒否します。

```bash
uv run persona setup --backend cu126
uv run persona setup --backend cu128
uv run persona setup --backend cpu
uv run persona setup --backend rocm
uv run persona setup --backend xpu
```

CUDA setupでは、logical CUDA device 0のGPU UUID・compute capability・NVIDIA driver versionと、各独立CUDA環境で実kernelが成功したpreflight結果を`.runtime/setup.json`へ記録します。GPUを交換・取り外した場合、`CUDA_VISIBLE_DEVICES`変更でdevice 0の物理GPUが変わった場合、またはNVIDIA driverを更新した場合は、互換wheelであってもmodel processを起動せず次を要求します。

```bash
uv run persona setup --backend auto
uv run persona doctor --deep
```

これは同じwheelで動くGPU交換でも、driverや3rd-party CUDA extensionまで含めて実機kernelをもう一度検証するためです。同じbackend/lockが引き続き適合する場合、既存`.venv`と検証済みmodel assetは再利用されるので、再setupは全モデルの再ダウンロードや再学習を意味しません。`CUDA_VISIBLE_DEVICES`の文字列だけが変わってもlogical device 0が同じ物理GPUを指し続け、driverも同じならruntime契約は維持されます。CPU setupはGPU交換に依存しません。

Seed-VCは別の監査済みPyTorch 2.4/cu124 stackを使うため、main workerとはGPU対応範囲が異なります。Pascal〜Hopperではcu124を使用し、BlackwellではIrodori/diarization/Sense/LFMをcu128のまま維持しつつSeed-VCだけCPUへfallbackします。GPU交換やdriver変更後はまず再setup/preflightを行い、その結果としてSeed-VCだけCPUへ切り替わる場合でも、他の対応済みworkerを不必要にCPU化しません。

Irodoriの学習batch profileもphysical GPUの最大VRAMではなく、実際のlogical CUDA device 0のVRAMを使います。CPU/ROCm/XPU backendではNVIDIA VRAMを参照しません。

## Target speaker is uncertain

Put cleaner authorized target-only clips in `personas/<name>/identity/`. A change to `identity/` is included in the prepare fingerprint, so the speaker-identification-dependent dataset is rebuilt automatically.

## Large dataset processing

ASR, pyannote, and SenseVoice run in batch workers, loading the model once per preparation batch. Existing source/clip caches are reused after interruption. `persona prepare --force` explicitly invalidates dependent ASR/diarization/identity/SenseVoice/clip caches even when the previous run failed; a normal rerun with the same fingerprint preserves those expensive caches for resume.

## Training inputs changed

Training has its own fingerprint. If the derived dataset or training configuration changes, stale Irodori latents/checkpoints, LFM adapters, and optional Seed-VC persona checkpoints are invalidated before retraining. Use `--force` to explicitly rebuild.

## Irodori backend / out of memory

Irodori backend is selected during setup and recorded in `.runtime/setup.json`. That recorded backend is used consistently for manifest creation, Irodori training, synthesis, and deep doctor. An explicit CPU backend does not switch itself back to CUDA merely because an NVIDIA GPU is visible.

PersonaVoice patches the official Irodori training config with a conservative batch profile. NVIDIA VRAM-based sizing is used only for audited CUDA backends (`cu126`/`cu128`); CPU/ROCm/XPU use the conservative profile unless a future backend-specific profile is explicitly implemented.

If a run is interrupted with the same input fingerprint, rerun the command and PersonaVoice resumes from available upstream checkpoints rather than invalidating them.

## Reference mode errors

`inference.reference_mode` is intentional rather than advisory:

- `auto`: prefer Speaker Inversion embedding, then prepared audio references, then unconditioned synthesis.
- `speaker-embed`: require `checkpoint_final.speaker.safetensors`; missing embedding is an error.
- `audio`: require the prepared reference bank; an empty bank is an error.
- An explicit CLI/API reference overrides the default mode and is always treated as an audio reference.

This prevents a requested conditioning method from silently falling back to a different voice identity path.

## Seed-VC problems

Seed-VC is an archived upstream with an older dependency stack. It is deliberately isolated under `workers/seed_vc`. Delete only `workers/seed_vc/.venv` and rerun `persona setup --backend auto` to rebuild that environment without touching persona datasets or model assets.

## API refuses non-loopback binding

PersonaVoice has no network authentication and refuses non-loopback binding by default. For a trusted network only:

```bash
uv run persona serve --host 0.0.0.0 --allow-remote
```

Use firewall or reverse-proxy authentication before wider exposure.
