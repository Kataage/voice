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
- Full training config: pinned upstream `train_v4_small.yaml`（新規personaの既定）
- Optional methods: `train_v4_small_lora.yaml`, `train_v4_small_speaker_inversion.yaml`
- Conditioning: text + reference/speaker embedding + caption
- Long reference: combined references supported by upstream (checkpoint limit 120s)
- Inference: `--num-candidates`, `--num-steps`, `--seed`, dynamic `--lora-adapter`

The DACVAE weight is materialized at `models/irodori/dacvae/weights.pth` and passed to upstream with `--codec-repo` for manifest creation, doctor smoke inference, and normal synthesis. This avoids a floating Hub lookup during offline operation. PersonaVoice also validates that the pinned upstream training config still names the audited ModernBERT repo/revision before starting training.

PersonaVoice intentionally calls upstream scripts instead of copying their model implementation.

`training.irodori.method`ごとの成果物契約は次の通りです。

- `full`: held-out validation lossで選んだ完全checkpointを、`model.safetensors`、`tokenizer/`、`manifest.json`、`provenance.json`を持つmachine-path非依存artifactへatomic変換する。optimizerを含む完全なnumeric checkpointだけをpreemption resume対象にする。公開前は全held-out caseでspeaker-conditioned / no-reference / caption-conditionedを同じspeaker similarity・CER/WER・pronunciation・duration・emotion指標で比較し、外付けidentityに依存しないno-reference + caption/emotion経路を主gateにする。公開後の`auto`推論もfullだけは`--no-ref`が既定。
- `lora`: validation-loss best checkpointから`adapter_config.json`と単一の非空adapter weightだけを`selected/`へatomic copyするPEFT artifact。trainer/optimizer stateはresume用native checkpoint側にだけ残し、portable candidateには含めない。adapter configのmachine-local base pathは除去し、選択元step/lossと全file SHA-256をprovenanceへ固定する。
- `speaker-inversion`: `checkpoint_final.speaker.safetensors`。`auxiliary_speaker_inversion: true`ではLoRA/fullの主artifactとは別に保持する。

method変更はoptimization/family fingerprintを変えますが、DACVAE latentの生成contractはmethodを含まないため、同一dataset/model/conditioningの既存latentを再利用します。executorとremote consentはどのmethodのfingerprintにも含まれません。

## LFM

- Base: `LiquidAI/LFM2.5-1.2B-JP-202606`
- Pinned snapshot revision: `b31023f2d69b95fbd7876898f8de9fae90e8afbd`
- `chat_template.jinja` SHA256: `89e790f027916b5a2bca145a6a8454e06ffc7a5043bf3b6d97829aff86bb543f`
- `config.json` SHA256: `df8dac1ebef28c06a010be6353e7dd2d0a3ff9c2ca23591bb8ced252d74510a1`
- `model.safetensors` SHA256: `abf38960d3f37c2be7c946a9b6b06d23ed04a1afb8ac192aa3b491e3dcdcf325`
- `special_tokens_map.json` SHA256: `742aefe2b7dec496e8caffdba03a75d0c1a9925d53bd3f3e0d388c96b591b6f4`
- `tokenizer.json` SHA256: `d7a0ab0fc22e41ec8c6d7450a9ff9ce40e196ec5e5a2fa6a2105e064e0514ed7`
- `tokenizer_config.json` SHA256: `8cba5b0c7acab23a0d4cc9ac587346c9220a1b6d288fc5346fe118202fd6f43e`
- Purpose: Japanese conversation style / response planning and structured delivery planning
- Architecture contract: 16 layers, including 6 `full_attention` blocks and 10 convolution blocks in the pinned model configuration
- Fine-tune: TRL SFT full model（新規personaの既定）またはPEFT LoRA
- SFT dataset: conversational `prompt` (`system`/`user`) + `completion` (`assistant` only), with non-empty string content; loss is explicitly restricted to the authorized persona completion. Bundle creation and the isolated trainer both reject role drift before tokenization
- Completion-mask audit: exact tokenizer/chat templateでraw prompt/full token列を再構成し、prompt prefix、非空completion、2048-token上限を学習前に検証する。TRLが処理した`completion_mask`もraw長のmultisetと照合し、truncationやmask driftを黙って学習しない
- LoRA intent: attention q/k/v/output projections only
- Actual Transformers LFM2 output-projection name: `out_proj` (the Liquid TRL documentation currently shows the stale spelling `o_proj`)
- Target resolution: exact loaded module paths under `.self_attn.` for `q_proj`, `k_proj`, `v_proj`, and `out_proj`; ShortConv `out_proj` is deliberately excluded
- Deep health: verifies the number of each projection matches the number of `full_attention` layers in the loaded pinned model
- Generation defaults: `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05`; no additional `top_p` restriction is injected
- Output contract: JSON containing text plus voice caption/emotion/events

The materialized local snapshot contains `.personavoice-revision`. Setup reuses it only when all required files are non-empty, the revision marker matches the audited revision, and every architecture/tokenizer/chat-template/weight file above matches its audited SHA256. This exact inventory is part of the LFM family fingerprint and is independently enforced by local setup, the isolated inference/training workers, and Modal asset materialization, so local and Modal cannot silently tokenize or construct different models under one training contract. A failed explicit download never publishes the revision marker.

A finalized persona LoRA adapter must contain `adapter_config.json`, a non-empty PEFT adapter weight (`adapter_model.safetensors` or `adapter_model.bin`), and `.personavoice-base-revision` equal to the pinned JP-202606 revision. Both training orchestration and inference reject partial adapters or adapters finalized against another base revision.

`training.lfm.method: full`はPEFTを挿入せず全parameterを学習し、validationで選んだcheckpointから`config.json`、`model.safetensors`、tokenizer一式、`manifest.json`、`provenance.json`を持つportable artifactを作ります。Transformers 5系が保存時に`special_tokens_map.json`を省略した場合も、固定revision/hash検証済みbase snapshotの同ファイルだけを補い、別revisionやmachine pathを混入させません。`method: lora`は上記projectionだけのadapterを作ります。両方ともworkerがsafetensors/config、Trainer step、optimizer、scheduler、CPU/CUDA RNG、precisionとFP16時のscalerをsafe-load検証し、全native fileのSHA-256をatomic attestationへ結び付けたcheckpointだけをresumeします。partial・欠損・改ざん・検証不能checkpointは候補から外しますが、自動削除やnative payloadの書換えはしません。

## Diarization

- Model: `pyannote/speaker-diarization-community-1`
- Pinned snapshot revision: `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`
- `config.yaml` SHA256: `5ce2bfa9a938dc132cec1172592d65173cbb8f444ea1e4133f10f9391de155be`
- `embedding/pytorch_model.bin` SHA256: `6f10ff60898a1d185fa22e1d11e0bfa8a92efec811f11bca48cb8cafebefd929`
- `segmentation/pytorch_model.bin` SHA256: `7ad24338d844fb95985486eb1a464e32d229f6d7a03c9abe60f978bacf3f816e`
- `plda/plda.npz` SHA256: `9b77bcd840692710dd3496f62ecfeed8d8e5f002fd991b785079b244eab7d255`
- `plda/xvec_transform.npz` SHA256: `325f1ce8e48f7e55e9c8aa47e05d2766b7c48c4b25b8de8dd751e7a4cc5fbe8f`
- Uses regular + exclusive diarization and speaker embeddings.
- Gated download: user must accept Hugging Face conditions and provide `HF_TOKEN` when the audited snapshot must be materialized.
- The pipeline is copied into `models/pyannote/community-1`, tagged with `.personavoice-revision`, and loaded from that local directory in offline mode.

Setup reuses the local pyannote snapshot only when the required files, exact revision marker, and all audited asset hashes agree. The isolated diarization worker re-hashes the config, embedding, segmentation, and PLDA assets before pipeline load; a legacy, modified, or corrupted local view is therefore never silently used for dataset preparation.

## ASR

- Legacy/reference: `openai/whisper-large-v3`, materialized through the audited
  `Systran/faster-whisper-large-v3` snapshot at
  `edaa852ec7e145841d8ffdb056a99866b5f0a478`.
  `model.bin` SHA256: `69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1`
  It remains
  available for explicit legacy configurations and is not the new-persona default.
- General modern backend: `Qwen/Qwen3-ASR-1.7B` at
  `7278e1e70fe206f11671096ffdd38061171dd6e5` (Apache-2.0). Setup records the complete
  required-file inventory and Hugging Face LFS object IDs in `integrity_ids.json`; those
  IDs are provenance evidence, not mislabeled local SHA256 values.
- General alignment: `Qwen/Qwen3-ForcedAligner-0.6B` at
  `c7cbfc2048c462b0d63a45797104fc9db3ad62b7`, an independent versioned `alignment-v1`
  contract. Whisper native word timestamps and Qwen forced alignment cannot be swapped
  across backends.
- Domain backend: `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` at
  `5a6a789ceb2f22d2b8606743b13a8159af218362` is disabled. Its Apache page badge does not
  override the GPL-3.0 `litagin/Galgame_Speech_ASR_16kHz` provenance, commercial-use
  prohibition, open-source-model requirement, or uncleared `ctc_aligner.pt` terms. The
  exact reason and coupled encoder are returned by `persona doctor`.
- BGM-aware analysis: `audio-separator==0.44.2` at source revision
  `fca0cf76d52b545cedbc75e1d3aea626d513c036`, with `UVR_MDXNET_KARA_2.onnx`. The MIT
  wrapper license and model-specific terms are tracked separately. A user must register
  the local weight and its source/terms before offline use; the model is never bundled or
  redistributed. Output is a derived analysis stem only, never a replacement for the
  canonical original audio.

See [`ASR_LINEAGE.md`](ASR_LINEAGE.md) for the immutable generation layout, exact setup
sequence, quality reports, and GTX 1080 Ti/Pascal limitations.

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

### Vevo2 FM-only (v0.4)

- Upstream source: `open-mmlab/Amphion`
- Pinned source commit: `26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`
- Source license: MIT ([upstream LICENSE](https://github.com/open-mmlab/Amphion/blob/main/LICENSE))
- Released model: `RMSnow/Vevo2`
- Pinned model revision: `2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8`
- Model license: CC BY-NC-ND 4.0 ([model card](https://huggingface.co/RMSnow/Vevo2), [license](https://creativecommons.org/licenses/by-nc-nd/4.0/))
- Initial supported route: upstream FM-only style-preserved VC, source audio plus target reference
- Out of scope for this integration: AR+FM, TTS, SVS, editing, and Vevo2 fine-tuning
- Isolated environment: `workers/vevo2`, Python 3.10, Torch 2.4.0 CPU/cu124 extras
- Model view: `models/vevo2/assets/vevo2`
- Readiness marker: `.runtime/vevo2-models-ready`, containing the SHA256 of `config/vevo2_assets.json`
- Whisper dependency: official OpenAI Whisper `medium.pt`, SHA256-pinned in the same contract

The eight FM model files and their SHA256 values are intentionally declared in
[`config/vevo2_assets.json`](../config/vevo2_assets.json) rather than inferred from a
directory listing. `persona setup` is the only materialization path. The root worker
launcher verifies the pinned Amphion checkout, the local model revision marker, every
required file, and the separate Whisper checksum before it starts the pipeline. Normal
inference and deep doctor set Hugging Face offline flags; a missing or damaged file is an
error, not a remote retry.

The source license and model-weight license are tracked separately. MIT source code does
not grant commercial rights to CC BY-NC-ND model weights or to any component terms. Review
all upstream terms before redistribution or commercial use. See [`docs/VEVO2.md`](VEVO2.md)
for setup, backend selection, A/B metrics, and target-machine validation.

## Cache/training reproducibility

Prepare cache validity is bound to the audited ASR revision and weight contract, pyannote revision and asset-hash contract, SenseVoice asset hashes, relevant worker `uv.lock` hashes, and preprocessing implementation hashes in addition to raw/identity/config fingerprints. Updating any of those contracts invalidates derived ASR/diarization/identity/Sense artifacts before rebuilding the dataset. The materialization root is part of the prepare fingerprint because downstream manifests intentionally contain local absolute paths.

Training family fingerprints include the pinned Irodori source revision, Irodori base/DACVAE hashes, ModernBERT revision, LFM base revision and complete architecture/tokenizer/chat-template/weight inventory hashes, Seed-VC source revision, relevant worker/managed lockfile hashes, and family trainer implementation hashes. A base/source/dependency/family-algorithm update therefore invalidates the affected persona artifact/checkpoint even if exported dataset bytes are unchanged。共有runner/bundle/Modal appはplan-level `executor_contract`でlocal deploymentとremote imageの一致を検証しますが、family optimization fingerprintからは除外するため、安全なtransport/orchestration変更だけでcheckpointを捨てません。Executor、remote authorization、Modal resource名、credential、local hardware、quality thresholdもoptimization semanticsではないためfamily fingerprintから除外されます。Training artifacts without a recorded train-stage/family fingerprint are treated as untracked and rebuilt instead of having their provenance guessed.

Full/LoRA training output is first a candidate. Portable manifest/provenance and every file checksum are verified before it can enter state. Candidateはheld-out quality gate合格後にのみpublishedへatomic promotionされ、inferenceは未公開candidateをproduction modelとして扱いません。

## Backend contract

`persona setup --backend ...` writes the selected Irodori backend to `.runtime/setup.json`. Manifest preprocessing, Irodori training, normal synthesis, and deep doctor all consume that recorded value rather than independently choosing a visible accelerator. An explicit CPU setup therefore remains CPU even on a machine where `nvidia-smi` and CUDA are available.
