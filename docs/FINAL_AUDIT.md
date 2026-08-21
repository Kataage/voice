# Final audit notes

This document records the additional correctness contracts verified after the deep-audit hardening merge.

## Audited upstream contracts

- Irodori-TTS is pinned to `8224dafb46d0aba89209a8f905f1cb7e3299d9c1`, which is the upstream v4.1-Small revision audited by PersonaVoice.
- Irodori v4.1 Small reference-bank preparation is capped at the checkpoint's 120-second combined-reference limit; the default remains 40 seconds and shorter clean same-speaker clips are preferred.
- ASR is intentionally fixed to `Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478`. Runtime configuration cannot silently select a different remote ASR model.
- `pyannote/speaker-diarization-community-1` is loaded from the audited local snapshot and uses both regular and exclusive diarization plus speaker embeddings.
- LFM runtime and fine-tuning both require `LiquidAI/LFM2.5-1.2B-JP-202606@b31023f2d69b95fbd7876898f8de9fae90e8afbd`; the local revision marker is verified before model loading.
- SenseVoice normal inference and deep health are local-only. Only the explicit setup/download operation may contact ModelScope, and inference-critical assets are hash-verified before use.
- Seed-VC remains isolated on its archived, pinned upstream source and compatibility environment. Reenact output is isolated per request to avoid deterministic upstream filename collisions.

## Data integrity contracts

- Identical raw recordings placed under multiple filenames are represented once in the training pipeline; duplicate logical paths are retained in source inventory provenance.
- Prepare fingerprints include the absolute materialization root because exported downstream manifests contain local absolute paths. Moving the repository/persona therefore invalidates prepared exports and regenerates them at the new location.
- `persona.yaml`'s embedded persona name must match its containing persona directory.
- Training and runtime workers fail closed on missing/wrong audited model revisions rather than falling back to a network model.

## Remaining validation boundary

Static checks, unit tests, locked dependency resolution, and offline model-loading contracts can be automated in the repository. End-to-end acoustic quality and CUDA-driver/runtime compatibility still require `persona setup` and `persona doctor --deep` on the target Windows/NVIDIA machine with the actual model assets, followed by a real `persona build` and inference smoke test.
