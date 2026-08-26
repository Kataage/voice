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

Training has a plan fingerprint and separate family fingerprints. If the derived dataset, method, optimizer semantics, base revision, lock, or training implementation changes, only incompatible family checkpoints/artifacts are rejected. Executor、remote consent、Modal resource名の変更はoptimization planを変えず、quality threshold変更も既存checkpointを捨てず再評価します。Irodori methodだけを変えた場合も、dataset/model/conditioning contractが同じ事前計算latentは再利用されます。通常の中断復帰に`--force`は不要です。`train --force`もprepare cacheを消さず、family固有の完全artifact/checkpointはその検証契約に従って再利用できます。

## Irodori backend / out of memory

Irodori backend is selected during setup and recorded in `.runtime/setup.json`. That recorded backend is used consistently for manifest creation, Irodori training, synthesis, and deep doctor. An explicit CPU backend does not switch itself back to CUDA merely because an NVIDIA GPU is visible.

PersonaVoice patches the official Irodori training config with a conservative batch profile. NVIDIA VRAM-based sizing is used only for audited CUDA backends (`cu126`/`cu128`); CPU/ROCm/XPU use the conservative profile unless a future backend-specific profile is explicitly implemented.

If a run is interrupted with the same input fingerprint, rerun the command and PersonaVoice resumes from available upstream checkpoints rather than invalidating them.

## LFM completion mask / maximum sequence length

LFM SFTはsystem/user promptではなく、許可済みpersonaのassistant completionだけへlossを掛けます。prompt roleは`system`/`user`、completion roleは`assistant`だけを許可し、空contentやrole driftはbundle作成時とworker側の両方で拒否します。workerは固定tokenizer/chat templateで各raw例を再構成し、promptがfull sequenceの厳密なprefixであること、completion tokenが1個以上あること、full sequenceが2048 token以下であることを、model学習開始前に検査します。さらにTRLが生成した`completion_mask`の長さ・連続性・件数をraw側と照合します。

`exceeds the audited maximum sequence length`、`zero completion tokens`、`completion mask ... drifted`等で停止した場合は、該当する会話を意味を保った複数例へ分割するか、過長なcontext/completionを短くしてからprepare/trainを再実行してください。上限を上げたりtruncationを許可して先へ進めると、completion全体が切れてpersona lossが消える可能性があるためfail-closedです。失敗したnative checkpointやprepare cacheを手動削除する必要はありません。training inputが変われば新しいfamily fingerprintになり、旧checkpointは証跡として保持されたまま再開対象外になります。

## Local full-training preflight failed

新規personaのIrodori/LFM methodは`full`です。`auto`または`local`は、監査済みCUDA setup、総/free VRAM、available RAM、workspace diskを学習開始前に確認します。不足時にLoRAへ自動変更しないため、次のいずれかを明示してください。

1. `persona doctor`の`local_training_preflight.failures`を解消して`--executor local`を再実行する。
2. remote processingを許可しModal SDK/authを設定して、同じplanを`--executor modal`で実行する。
3. 本当に別methodを望む場合だけ`persona.yaml`の`method: lora`を編集する。この変更は別family planです。

`--executor`は1回限りのrouting overrideでconfigを保存しません。`status`のplan fingerprintが同じなら、local/Modal切替だけでprepare/latent/checkpointを削除しないでください。

## Modal executor is unavailable or unauthorized

`persona doctor`の`modal`には`SDK installed`相当のbool、auth configured、credentialの非secret sourceだけが表示され、token ID/secret値や`.env`内容は表示されません。doctorはModalへnetwork probeを行いません。

Production machineではlock済みoptional extraを使い、deployを先に完了させます。既定Function名は`train`です。

```bash
uv sync --locked --extra modal
uv run --locked --extra modal modal setup
uv run --locked --extra modal modal deploy -m personavoice.modal_app
uv run --locked --extra modal persona doctor
```

app/Volume名を変更する場合は、`PERSONAVOICE_MODAL_APP`と`PERSONAVOICE_MODAL_VOLUME`をdeployとtraining clientで同じ値にします。bundled appのtraining Function名は`train`、terminal cleanup Function名は`recover_terminal_claim`で、persistent claim Dictは`<app名>-claims`としてdeploy時に作られます。このDictはplan/family fingerprintとcall IDだけを持ちます。実行中のcallがある間は手動でclear/deleteしないでください。HF tokenはroot `.env`をuploadしたりshell argvへ含めたりせず、Modal dashboardで`PERSONAVOICE_MODAL_HF_SECRET`（既定`personavoice-huggingface`）と同名のSecretを作り、`HF_TOKEN` keyだけを登録します。`PERSONAVOICE_MODAL_FUNCTION`はclient lookup overrideなので、bundled appでは変更せず、互換な別deploymentを利用する場合だけその公開Function名へ合わせます。GPU既定は`A100-40GB`です。通常のlocal-only環境は`modal` extra不要で、CIもreal SDKのimport/spec構築とfake backendだけを使い実接続しません。

- `Remote training is disabled`: persona全体の`consent.authorized: true`に加えて、remote転送専用の`training.remote_data_authorized: true`を本人同意の範囲に沿って明示する。汎用consent scopeやModal credentialはこのbooleanを代替しない。
- `Modal authentication is not configured`: 上のlock済みextraを同期し、`modal setup`または`modal token set`、親processの`MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`、もしくはroot `.env`のallowlisted keyを使う。
- `Modal support is optional`: SDK未導入。`uv sync --locked --extra modal`をproductionでModalを選ぶmachineだけに実行する。
- `Modal Function lookup failed`: `modal deploy -m personavoice.modal_app`を完了し、deployとclientのapp/Function名および`MODAL_ENVIRONMENT`を一致させる。
- 保存済みcallがあるのにlocalを選べない: 同じplanを`--executor modal`で再実行してcallをpollする。重複submitを避けるための保護です。
- `failed-recoverable`: hard timeout等でModalのretryを使い切ったold callをclientが検出し、`recover_terminal_claim`がそのterminal状態をModal側で独立に再確認して、old callだけが所有するfamily claim、plan claimの順に解放済みです。同じ`persona train ... --executor modal`を再実行してください。新callはVolume上のchecksum済みmethod-native checkpointから再開します。状態がrunning、complete、expired/不明、またはclaim ownerが別callの場合は自動解放しません。

Remote consentの検査はbundle作成・auth probe・uploadより前です。bundleにはcanonical plan、Irodori latent、許可済みmetadataのlossless等価性をchecksumで証明するsource manifest、content-addressed latent相対pathのsanitized runtime manifest、LFM会話dataだけが入ります。二種類のIrodori manifestは監査契約と実行契約であり、どちらにもraw audioや絶対local pathは含みません。raw/identity/reference audio、SQLite、`.env`、secret、絶対pathは入りません。Seed-VC FTはaudio非転送contractのためModalでは拒否されます。

## Candidate exists but is not published

`persona train`が正常終了しても、stateが`trained`で`published_complete: false`なら、checksum検証済みcandidateはありますがlocal held-out quality gateがまだ完了または合格していません。speaker similarity、CER/WER、unseen pronunciation、validation loss、duration ratio、emotion/style、base CER regressionの欠損・`NaN`・無限値はfail-closedで、良い値だけを平均して通しません。Irodori fullでは主gateがno-reference + caption/emotion経路であること、`report.json`の`mode_comparison`に3 conditioning経路それぞれの全指標と全caseがあることも確認してください。

LFMが有効なら`report.json`の`lfm`を確認してください。`complete`、candidate/base両方の`contract_passed`、期待completionに対するsimilarity/CER/WER、`required_phrase_coverage_mean`、`base_similarity_regression_max`が全case分そろう必要があります。既定thresholdは順に`>= 0.35`、`<= 0.85`、`<= 1.0`、`>= 1.0`、`<= 0.10`です。`temperature: 0`はworker側でsamplingを無効化したgreedy generationになり、0温度のsampling errorへfallbackしません。training JSONLが壊れたwrapper/completionを含む場合、またはwrapperの実dialogue行やassistant JSONの`text`が固定eval prompt/answerと正規化後に完全一致する場合も、評価は公開前に停止します。

`persona status <name>`で`quality_gate`とfamily candidate状態を確認し、評価環境/held-out inputを直して同じplanを再評価してください。thresholdを下げるために学習checkpointを削除する必要はありません。gateを迂回して`.candidates`をpublished locationへ手作業で移動しないでください。

## v0.3 persona.yaml migration

legacy flat training fieldsは読み込み時にmemory上だけでschema v2へ変換されます。通常の`status`、`prepare`、`train`は`persona.yaml`を自動書換えしません。

```bash
uv run --locked persona migrate-config alice --dry-run
uv run --locked persona migrate-config alice
```

dry-runのnotesで意味保存結果を確認してください。旧IrodoriのLoRA+Speaker Inversionは`method: lora` + `auxiliary_speaker_inversion: true`、両方falseは`enabled: false`です。旧LFM LoRA learning rate等も保持されます。legacy fieldと新しいnested fieldの混在は曖昧なので拒否され、手動解決が必要です。移行自体はdataset/prepare/latent/checkpointのsemantic inputを変更しないため、有効なcacheを一律invalidateしません。

v0.3で作られたLFM optimizer checkpointは削除・上書きされません。ただしcompletion-only mask、固定base inventory、exact native attestationを持たない旧training contractを新しいfull/LoRA runへ推測でresumeすることはありません。互換性のあるIrodori latent、prepare output、完成artifactは各validatorを通ったものだけ個別に再利用されます。

## Root .env and offline/secret handling

`.env.example`からroot `.env`を作成できますが、PersonaVoiceが読むのは明示allowlistだけです。親processの値が常に優先され、unknown key、`PYTHONPATH`等は注入されません。secret値はloader report、doctor、status、training state、bundle、Modal payloadへ返しません。`.env`の`${...}`展開やcommand substitutionは行わずliteralとして扱います。

通常のprepare/inference/deep doctorはmaterialize済みmodelをofflineで使います。`HF_TOKEN`はgated assetを明示的な`persona setup`で初回取得する場合だけ必要です。`--offline`相当の通常経路で不足modelをremoteから黙って取得することはありません。

## Reference mode errors

`inference.reference_mode` is intentional rather than advisory:

- `auto`: LoRA/Speaker Inversion/baseではSpeaker Inversion embedding、prepared audio、unconditionedの順。公開済みfull persona checkpointではweight-owned identityを既定にするため`--no-ref`。
- `none`: 明示的に`--no-ref`を使い、reference audio/embeddingを付加しない。
- `speaker-embed`: require `checkpoint_final.speaker.safetensors`; missing embedding is an error.
- `audio`: require the prepared reference bank; an empty bank is an error.
- An explicit CLI/API reference overrides the default mode and is always treated as an audio reference.

This prevents a requested conditioning method from silently falling back to a different voice identity path.

## Seed-VC problems

Seed-VC is an archived upstream with an older dependency stack. It is deliberately isolated under `workers/seed_vc`. Delete only `workers/seed_vc/.venv` and rerun `persona setup --backend auto` to rebuild that environment without touching persona datasets or model assets.

## Vevo2 setup, offline runtime, and license

Vevo2 is the v0.4 selectable FM-only backend. Its source checkout is
`open-mmlab/Amphion@26f6883110181f1dbfe95c70a7c7dbaf4de5f42a`; its released model is
`RMSnow/Vevo2@2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8`. The source is MIT, while the
released weights are CC BY-NC-ND 4.0. These are separate terms; MIT source licensing is
not a commercial-use grant for the weights.

Only the explicit online setup path may materialize Vevo2:

```bash
uv run --locked persona setup --backend auto
uv run --locked persona doctor --deep
```

The first command downloads and hashes the contract-declared FM files and official Whisper
medium weights. The second command is offline and must not download. If it reports a
missing/empty file, revision mismatch, checksum mismatch, dirty Amphion checkout, or
missing `.runtime/vevo2-models-ready`, rerun explicit setup. Do not manually create the
ready marker and do not allow the worker to use an unpinned mirror. To prepare a machine
without the heavy Vevo2 view, use `persona setup --skip-vevo2-models`; that machine cannot
run Vevo2 until setup is run again without the skip flag.

Vevo2 uses its own Python 3.10/Torch 2.4 worker environment. A CUDA setup recorded as
`cu124` with no visible CUDA device is an error and never falls back to CPU. A requested
`fp16` CPU run is also an error. `fp32` is the default; do not infer Pascal VRAM fit or
performance from dependency lock resolution alone.

## Vevo2 vs Seed-VC evaluation is pending

The canonical evaluator needs an authorized prepared persona, a target reference bank,
and the target machine's materialized model/worker environments. Generate its immutable
manifest and run both backends as follows:

```bash
uv run --locked persona eval-vc-manifest alice --limit 200 --seed 20260827
uv run --locked persona eval-vc alice \
  --manifest personas/alice/dataset/vc_evaluation_manifest.jsonl
```

If `dataset/master.json`, target clips, or references are absent, the command stops and
does not fabricate rows. If a backend or metric fails, the per-sample error stays in
`report.json`. `report.md` and `human_review.json` are written below
`outputs/vc-evaluation/<run>/`. The report must contain Japanese CER, speaker similarity,
duration/prosody/timing, voiced/unvoiced, pause, non-verbal event bucket metrics, and
human review before a default decision is possible. Fewer than 100 clips, missing
normal/mixed/nonverbal buckets, or an incomplete human review remains
`pending target-machine validation`. Until Issue #30 records a completed gate, keep
`vc_backend: seed-vc-v2`.

## GTX 1080 Ti / Pascal smoke test

Hosted CI does not prove this target-machine validation. On the target machine, from the
fresh GitHub checkout:

```bash
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv
echo "$CUDA_VISIBLE_DEVICES"
uv run --locked persona setup --backend auto
uv run --locked persona doctor --deep
```

Confirm `.runtime/setup.json` has the intended GPU UUID and `worker_backends.vevo2` value,
then run one short authorized FP32 conversion and record peak VRAM, wall time, output
validity, reported device, and dtype:

```bash
uv run --locked persona reenact alice \
  personas/alice/dataset/clips/<short-clip>.flac \
  --ref personas/alice/references/<target-reference>.flac \
  --backend vevo2-fm
```

Repeat for speech, laughter/breath or mixed speech, and nonverbal-only samples before
running the full A/B manifest. Only after measured FP32 success should `fp16` be tried;
an OOM, unsupported kernel, or dtype error is recorded as failed validation. No silent
dtype or device fallback is permitted. GPU replacement, driver update, or a
`CUDA_VISIBLE_DEVICES` change selecting another physical device requires setup/preflight
again.

## API refuses non-loopback binding

PersonaVoice has no network authentication and refuses non-loopback binding by default. For a trusted network only:

```bash
uv run persona serve --host 0.0.0.0 --allow-remote
```

Use firewall or reverse-proxy authentication before wider exposure.
