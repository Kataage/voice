# Pinned model/upstream contracts

PersonaVoice treats source revisions, dependency lockfiles, and model assets as one reproducibility contract. The canonical machine-readable asset pins live in `src/personavoice/model_assets.py`; setup materializes them and `persona doctor` verifies that the local environment matches the audited contract.

## Irodori-TTS

- Upstream: `Aratako/Irodori-TTS`
- Pinned source commit: `8224dafb46d0aba89209a8f905f1cb7e3299d9c1`
- Base checkpoint: `Aratako/Irodori-TTS-v4.1-Small/model.safetensors`
- Base checkpoint SHA256: `c85de88c01700cb53538e706f128ebcb1b8513ad21d7d0e75f58bc82cdbf89f6`
- DACVAE: `Aratako/Semantic-DACVAE-Japanese-32dim/weights.pth`
- DACVAE SHA256: `db120339c5ee7eca1912cdf29bc612b947a0808e69c3cebfb4936b45a762c1d5`
- Text/caption encoder repo: `sbintuitions/modernbert-ja-310m`
- Text encoder revision required by the pinned v4 configs: `77675fc96a7e445e982e2ba90246b816efc74ec6`
- Training configs: `train_v4_small_lora.yaml`, `train_v4_small_speaker_inversion.yaml`
- Conditioning: text + reference/speaker embedding + caption
- Long reference: combined references supported by upstream (checkpoint limit 120s)
- Inference: `--num-candidates`, `--num-steps`, `--seed`, dynamic `--lora-adapter`

The DACVAE weight is materialized at `models/irodori/dacvae/weights.pth` and passed to upstream with `--codec-repo` for manifest creation, doctor smoke inference, and normal synthesis. This avoids a floating Hub lookup during offline operation. PersonaVoice also validates that the pinned upstream training config still names the audited ModernBERT repo/revision before starting training.

PersonaVoice intentionally calls upstream scripts instead of copying their model implementation.

## LFM

- Base: `LiquidAI/LFM2.5-1.2B-JP-202606`
- Pinned snapshot revision: `b31023f2d69b95fbd7876898f8de9fae90e8afbd`
- Purpose: conversation style / response planning
- Fine-tune: TRL SFT + PEFT LoRA, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Output contract: JSON containing text plus voice caption/emotion/events

The materialized local snapshot contains `.personavoice-revision`. Setup does not accept a legacy floating-revision directory merely because `config.json` exists; it rematerializes the pinned snapshot when the marker is absent or mismatched.

## Diarization

- `pyannote/speaker-diarization-community-1`
- Uses regular + exclusive diarization and speaker embeddings.
- Gated download: user must accept Hugging Face conditions and provide `HF_TOKEN` for initial setup.
- The pipeline is copied into `models/pyannote/community-1` and subsequently loaded from that local directory in offline mode.

Because this is a gated multi-component pipeline, PersonaVoice currently relies on the locally materialized pipeline plus `doctor --deep` model loading rather than claiming a checksum pin for assets it cannot independently fetch without the user's accepted access token.

## ASR

- `Systran/faster-whisper-large-v3`
- Pinned snapshot revision: `edaa852ec7e145841d8ffdb056a99866b5f0a478`
- Word timestamps enabled; built-in Silero VAD filter enabled.
- The local snapshot carries the same `.personavoice-revision` contract as LFM.

## Acoustic emotion / events

- `iic/SenseVoiceSmall`
- Japanese supported.
- Emotion/event tags are parsed before rich post-processing so the labels are not discarded.
- ModelScope/FunASR materialization is validated by loading the same worker in `persona doctor --deep` with network access disabled where applicable.

## Voice conversion

- Upstream: `Plachtaa/seed-vc`
- Pinned source commit: `51383efd921027683c89e5348211d93ff12ac2a8`
- Upstream was archived in 2025, so it is isolated in a Python 3.10 environment.
- V2 style conversion is used for `reenact`.
- Fine-tuning is disabled by default and can be enabled in `persona.yaml`.

## Backend contract

`persona setup --backend ...` writes the selected Irodori backend to `.runtime/setup.json`. Manifest preprocessing, Irodori training, normal synthesis, and deep doctor all consume that recorded value rather than independently choosing a visible accelerator. An explicit CPU setup therefore remains CPU even on a machine where `nvidia-smi` and CUDA are available.
