# Pinned model/upstream contracts

PersonaVoice treats source revisions, dependency lockfiles, model assets, and the implementation code that interprets them as one reproducibility contract. The canonical machine-readable asset pins live in `src/personavoice/model_assets.py`; setup materializes them and `persona doctor` verifies that the local environment matches the audited contract.

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
- `model.safetensors` SHA256: `abf38960d3f37c2be7c946a9b6b06d23ed04a1afb8ac192aa3b491e3dcdcf325`
- Purpose: Japanese conversation style / response planning and structured delivery planning
- Architecture contract: 16 layers, including 6 `full_attention` blocks and 10 convolution blocks in the pinned model configuration
- Fine-tune: TRL SFT + PEFT LoRA
- SFT dataset: conversational `prompt` + `completion`; loss is explicitly restricted to the authorized persona completion
- LoRA intent: attention q/k/v/output projections only
- Actual Transformers LFM2 output-projection name: `out_proj` (the Liquid TRL documentation currently shows the stale spelling `o_proj`)
- Target resolution: exact loaded module paths under `.self_attn.` for `q_proj`, `k_proj`, `v_proj`, and `out_proj`; ShortConv `out_proj` is deliberately excluded
- Deep health: verifies the number of each projection matches the number of `full_attention` layers in the loaded pinned model
- Generation defaults: `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`; no additional `top_p` restriction is injected
- Output contract: JSON containing text plus voice caption/emotion/events

The materialized local snapshot contains `.personavoice-revision`. Setup reuses it only when all required files are non-empty, the revision marker matches the audited revision, and `model.safetensors` matches the audited SHA256. The isolated LFM worker repeats the weight checksum before loading the model, so post-setup replacement or corruption fails closed. A failed explicit download never publishes the revision marker.

A finalized persona LoRA adapter must contain `adapter_config.json`, a non-empty PEFT adapter weight (`adapter_model.safetensors` or `adapter_model.bin`), and `.personavoice-base-revision` equal to the pinned JP-202606 revision. Both training orchestration and inference reject partial adapters or adapters finalized against another base revision.

## Diarization

- Model: `pyannote/speaker-diarization-community-1`
- Pinned snapshot revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
- `config.yaml` SHA256: `5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be`
- `embedding/pytorch_model.bin` SHA256: `5d9b8f3c197eae5da64a677d81a9d45f3bb5fc7c47814a22fda2511cf8bdeaae`
- `segmentation/pytorch_model.bin` SHA256: `8d77689c7f22bc0c88d95ec499113f2f4aeffb7c85c87572579d3458f3f33d7f`
- `plda/plda.npz` SHA256: `325f3b766ec4f95db36e1935f37d9f323ef46ab466e74f87e78d36b0a1b5965c`
- `plda/xvec_transform.npz` SHA256: `0fe345409f1722cb9f1e116dd1cfbd3a6ce6cad72bd56e63868b00cd48be726a`
- Uses regular + exclusive diarization and speaker embeddings.
- Gated download: user must accept Hugging Face conditions and provide `HF_TOKEN` when the audited snapshot must be materialized.
- The pipeline is copied into `models/pyannote/community-1`, tagged with `.personavoice-revision`, and loaded from that local directory in offline mode.

Setup reuses the local pyannote snapshot only when the required files, exact revision marker, and all audited asset hashes agree. The isolated diarization worker re-hashes the config, embedding, segmentation, and PLDA assets before pipeline load; a legacy, modified, or corrupted local view is therefore never silently used for dataset preparation.

## ASR

- Model: `Systran/faster-whisper-large-v3`
- Pinned snapshot revision: `edaa852ec7e145841d8ffdb056a99866b5f0a478`
- `model.bin` SHA256: `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`
- Word timestamps enabled; built-in Silero VAD filter enabled.
- The local snapshot carries the same `.personavoice-revision` contract as LFM.
- Setup reuses the snapshot only when required files, the exact revision marker, and the audited model-weight hash agree.
- The isolated ASR worker re-hashes `model.bin` before loading faster-whisper, so post-setup replacement or corruption is rejected before transcription.

## Acoustic emotion / events

- Model: `iic/SenseVoiceSmall`
- Japanese supported.
- `model.pt` SHA256: `833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea`
- `am.mvn` SHA256: `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5`
- `chn_jpn_yue_eng_ko_spectok.bpe.model` SHA256: `aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8`
- Emotion/event tags are parsed before rich post-processing so the labels are not discarded.

SenseVoice is currently materialized through ModelScope, whose `master` label is not treated as a reproducible revision. Instead, the isolated worker verifies the inference-critical weight, CMVN, and SentencePiece assets before reuse and before model loading. It also uses `trust_remote_code=False`, so model Python comes from the uv-locked FunASR environment rather than executing mutable repository code. Setup writes `sense-model-ready=verified` only after those checks succeed; `persona doctor --deep` re-enters the same verified load path with network access disabled.

## Voice conversion

- Upstream: `Plachtaa/seed-vc`
- Pinned source commit: `51383efd921027683c89e5348211d93ff12ac2a8`
- Upstream was archived in 2025, so it is isolated in a Python 3.10 environment.
- V2 style conversion is used for `reenact`.
- Fine-tuning is disabled by default and can be enabled in `persona.yaml`.
- Fine-tune checkpoints are ordered by numeric step, not filename string order.
- PersonaVoice records cumulative progress in staged run-directory offsets because the pinned upstream trainer can load prior weights but resets its local iteration counter.

## Cache/training reproducibility

Prepare cache validity is bound to the audited ASR revision and weight contract, pyannote revision and asset-hash contract, SenseVoice asset hashes, relevant worker `uv.lock` hashes, and preprocessing implementation hashes in addition to raw/identity/config fingerprints. Updating any of those contracts invalidates derived ASR/diarization/identity/Sense artifacts before rebuilding the dataset. The materialization root is part of the prepare fingerprint because downstream manifests intentionally contain local absolute paths.

Training fingerprints include the pinned Irodori source revision, Irodori base/DACVAE hashes, ModernBERT revision, LFM base revision, Seed-VC source revision, relevant worker/managed lockfile hashes, and the training/Irodori/LFM-contract/Seed worker implementation hashes. A base/source/dependency/implementation update therefore invalidates old persona adapters/checkpoints even if exported dataset bytes are unchanged. If training artifacts exist without a recorded train-stage fingerprint, PersonaVoice treats them as untracked and rebuilds them instead of guessing their provenance.

## Backend contract

`persona setup --backend ...` writes the selected Irodori backend to `.runtime/setup.json`. Manifest preprocessing, Irodori training, normal synthesis, and deep doctor all consume that recorded value rather than independently choosing a visible accelerator. An explicit CPU setup therefore remains CPU even on a machine where `nvidia-smi` and CUDA are available.
