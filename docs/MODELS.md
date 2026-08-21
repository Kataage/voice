# Pinned model/upstream contracts

## Irodori-TTS

- Upstream: `Aratako/Irodori-TTS`
- Pinned commit: `8224dafb46d0aba89209a8f905f1cb7e3299d9c1`
- Base checkpoint: `Aratako/Irodori-TTS-v4.1-Small`
- Training configs: `train_v4_small_lora.yaml`, `train_v4_small_speaker_inversion.yaml`
- Conditioning: text + reference/speaker embedding + caption
- Long reference: combined references supported by upstream (checkpoint limit 120s)
- Inference: `--num-candidates`, `--num-steps`, `--seed`, dynamic `--lora-adapter`

PersonaVoice intentionally calls upstream scripts instead of copying its model implementation.

## LFM

- Base: `LiquidAI/LFM2.5-1.2B-JP-202606`
- Purpose: conversation style / response planning
- Fine-tune: TRL SFT + PEFT LoRA, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Output contract: JSON containing text plus voice caption/emotion/events

## Diarization

- `pyannote/speaker-diarization-community-1`
- Uses regular + exclusive diarization and speaker embeddings.
- Gated download: user must accept Hugging Face conditions and provide `HF_TOKEN` for initial setup.

## ASR

- `Systran/faster-whisper-large-v3`
- Word timestamps enabled; built-in Silero VAD filter enabled.

## Acoustic emotion / events

- `iic/SenseVoiceSmall`
- Japanese supported.
- Emotion/event tags are parsed before rich post-processing so the labels are not discarded.

## Voice conversion

- Upstream: `Plachtaa/seed-vc`
- Pinned commit: `51383efd921027683c89e5348211d93ff12ac2a8`
- Upstream was archived in 2025, so it is isolated in a Python 3.10 environment.
- V2 style conversion is used for `reenact`.
- Fine-tuning is disabled by default and can be enabled in `persona.yaml`.
