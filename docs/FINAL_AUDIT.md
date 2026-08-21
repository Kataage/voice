# Final audit notes

This document records the additional correctness contracts verified after the deep-audit hardening merge. The upstream re-check described below was performed on 2026-08-22.

## Audited upstream contracts

- Irodori-TTS is pinned to `8224dafb46d0aba89209a8f905f1cb7e3299d9c1`, which was still the upstream `main` commit during this audit and is the v4.1-Small revision PersonaVoice targets.
- Irodori v4.1 Small reference-bank preparation is capped at the checkpoint's 120-second combined-reference limit; the default remains 40 seconds and shorter clean same-speaker clips are preferred.
- ASR is intentionally fixed to `Systran/faster-whisper-large-v3@edaa852ec7e145841d8ffdb056a99866b5f0a478`. Runtime configuration cannot silently select a different remote ASR model.
- `pyannote/speaker-diarization-community-1@3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` is loaded from the audited local snapshot and uses both regular and exclusive diarization plus speaker embeddings. The official model documentation supports loading the pipeline from a local directory for offline use.
- The conversation brain is exactly `LiquidAI/LFM2.5-1.2B-JP-202606@b31023f2d69b95fbd7876898f8de9fae90e8afbd`, using the native Transformers checkpoint rather than GGUF/ONNX. The current model card identifies it as the Japanese-capable text-only chat checkpoint intended for native fine-tuning/inference and recommends `temperature=0.1`, `top_k=50`, and `repetition_penalty=1.05`; the worker follows those values without injecting an extra `top_p` cutoff.
- The pinned LFM2 implementation uses `self_attn.out_proj` for the attention output projection even though the current Liquid TRL documentation shows the stale spelling `o_proj`. PersonaVoice resolves exact loaded `.self_attn.{q_proj,k_proj,v_proj,out_proj}` module paths instead of blindly suffix-matching, so ShortConv's separate `out_proj` is not adapted by accident.
- SenseVoice normal inference and deep health are local-only. Only the explicit setup/download operation may contact ModelScope, and inference-critical assets are hash-verified before use.
- Seed-VC remains isolated on `51383efd921027683c89e5348211d93ff12ac2a8`; this was still upstream `main` during this audit. Reenact output is isolated per request to avoid deterministic upstream filename collisions.

## Persona-learning contracts

- LFM SFT data is exported as conversational `prompt` + `completion` examples rather than a single full-conversation `messages` field. The system/user context is conditioning input while the authorized persona JSON reply is the completion.
- `SFTConfig(completion_only_loss=True)` is explicit, and CI instantiates that option inside the committed LFM worker environment on both Ubuntu and Windows. This prevents system prompts and the other speaker's text from receiving persona-imitation loss.
- Deep LFM health loads the pinned model and verifies all four audited attention projections exist for every `full_attention` layer declared by `config.layer_types` before training is attempted.
- A finalized LFM persona adapter requires `adapter_config.json`, a non-empty PEFT adapter weight, and `.personavoice-base-revision` matching the pinned JP-202606 revision. Training and inference both reject partial adapters or adapters finalized against a different base revision.
- LFM training provenance includes the base revision, worker lockfile, training implementation, and the exact LFM model-contract helper source. Changing dependency or targeting semantics invalidates an old adapter rather than silently reusing it.
- Seed-VC fine-tune checkpoint ordering uses numeric step values. Because the pinned upstream trainer can initialize weights but resets its local iteration counter, PersonaVoice records cumulative progress in staged run-directory offsets and resumes from the highest recoverable cumulative checkpoint.

## Data integrity and reproducibility contracts

- Identical raw recordings placed under multiple filenames are represented once in the training pipeline; duplicate logical paths are retained in source inventory provenance.
- The compact 16-hex source ID is derived from SHA256, and a theoretical truncated-prefix collision between different full hashes is rejected explicitly rather than corrupting utterance IDs.
- With explicit identity references, a source whose best diarized speaker falls below `prepare.min_identity_similarity` is treated as a valid source that does not contain the authorized target speaker. Its non-target transcript rows remain available for provenance/context, it contributes no target training clips, and it is reported in `dataset/skipped_sources.json`.
- Structural diarization failures remain errors. If completed preparation contains zero usable authorized-speaker utterances, the prepare stage fails rather than publishing a false-success state.
- Prepare fingerprints include the absolute materialization root because exported downstream manifests contain local absolute paths. Moving the repository/persona therefore invalidates prepared exports and regenerates them at the new location.
- Prepare cache policy includes the audited ASR/diarization/Sense model contract, isolated worker lockfile hashes, and relevant preprocessing source hashes. Training fingerprints include Irodori/LFM/Seed-VC source/model contracts, lockfile hashes, and relevant training implementation hashes. Dependency or implementation changes therefore invalidate stale derived artifacts even when user input is unchanged.
- `persona.yaml`'s embedded persona name must match its containing persona directory. In the public repository configuration, real persona YAML files are ignored by default together with raw media, generated datasets, models, outputs, and runtime state.
- Training and runtime workers fail closed on missing/wrong audited model revisions rather than falling back to a network model.
- A training component that is explicitly enabled is not silently skipped for data shortage: LFM LoRA requires at least two valid conversational examples, and optional Seed-VC fine-tuning requires at least two target-speaker clips.

## Runtime robustness contracts

- API generation endpoints use a single-flight lock by default so TTS/VC/LLM workloads do not concurrently contend for the same local accelerator and trigger avoidable OOM failures.
- Each Seed-VC reenact request owns a unique output directory, preventing concurrent or repeated conversions from confusing upstream deterministic filenames.
- LFM structured voice plans are normalized before synthesis so malformed `caption`, `emotion`, or `events` field types do not leak into the TTS command contract.
- A stale Seed-VC readiness marker is removed when an offline deep load proves the materialized assets are incomplete; setup may then explicitly repair that model state.
- Setup can discard and rematerialize only model views that fail deep offline verification while preserving shared download caches. CUDA visibility/runtime errors are not misclassified as corrupt model files.

## Automated validation

`core-ci` exercises Ubuntu and Windows independently. It performs locked root sync, Ruff, pytest, compileall, CLI smoke tests, and locked environment sync plus worker-entrypoint compilation for ASR, diarization, SenseVoice, LFM, and Seed-VC. GPU extras are also resolved with locked dry runs (`cu128` for modern Torch workers and `cu124` for the archived Seed-VC stack) without downloading/initializing model weights. GitHub Actions themselves are pinned to audited commit SHAs rather than mutable major-version tags.

## Remaining validation boundary

Static checks, unit tests, locked dependency resolution, and offline model-loading contracts can be automated in the repository. End-to-end acoustic quality and CUDA-driver/runtime compatibility still require `persona setup` and `persona doctor --deep` on the target Windows/NVIDIA machine with the actual model assets, followed by a real `persona build` and inference smoke test. No repository-only audit can prove subjective voice quality on hardware and data it cannot execute against; this boundary is intentionally explicit rather than hidden.
