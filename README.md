# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **専用音声 + 会話スタイル + Voice Conversion** を構築・実行するためのオーケストレーターです。

## Quick Start

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."  # pyannote Community-1初回取得時のみ
uv run --locked persona setup --backend auto
uv run --locked persona init alice --authorized
# personas/alice/raw/ と personas/alice/identity/ に素材を配置
uv run --locked persona doctor
uv run --locked persona build alice --executor auto
uv run --locked persona ui
```

## 機能

- faster-whisper `large-v3`: 日本語ASR + word timestamps
- pyannote Community-1: regular/exclusive diarization + speaker embeddings
- `identity/`本人参照による対象話者自動選択
- SenseVoiceSmall: 感情 + 笑い・泣き・息・咳・くしゃみ等のイベント解析
- Irodori-TTS v4.1 Small: full fine-tuning（既定）+ LoRA + Speaker Inversion
- LFM2.5-1.2B-JP-202606: full fine-tuning（既定）+ LoRA + 発話スタイル計画
- Seed-VC v2: 入力音声の間・抑揚・演技を使ったVoice Conversion
- `say`, `reenact`, `repeat`, `chat`, Web UI, localhost REST API
- Irodori境界診断（duration / tail A/B、leading artifactのseed・reference・caption比較）
- canonical SQLite dataset、content cache、途中再開、入力変更時の自動invalidation
- rootと各ML stackを独立した`uv`環境に隔離
- root / worker / Irodoriを監査済み`uv.lock`へ固定
- 推論・deep doctorは監査済みローカルmodelだけを使用し、欠損時にremote modelへ黙ってfallbackしない

## セットアップ

必要: `uv`, Git。NVIDIA GPU推奨。WindowsではTorchCodec用の監査済みshared FFmpeg 8.1.1を`persona setup`が`.runtime/tools`へ自動materializeするため、システム全体へのFFmpeg/WinGetインストールは不要です。Linux/macOSではFFmpeg 4〜8の実行ファイルとshared librariesを用意してください。

Windows:

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."
uv run --locked persona setup --backend auto
```

Linux/macOS:

```bash
./scripts/bootstrap.sh
export HF_TOKEN=hf_...
uv run --locked persona setup --backend auto
```

`--backend auto`を推奨します。PersonaVoiceは`CUDA_DEVICE_ORDER=PCI_BUS_ID`と`CUDA_VISIBLE_DEVICES`を考慮して実際のlogical CUDA device 0を特定し、そのGPUのcompute capabilityを監査済みwheel matrixへ照合します。x86_64の現在の固定stackではPascal 6.x / Volta 7.0は`cu126`、Turing以降の監査済み世代は`cu128`、未知・未監査世代や安全に識別できないGPUはCPUへfail-closedします。環境sync後にはIrodori・diarization・Sense・LFM・必要ならSeed-VCの各独立PyTorch環境で**実CUDA tensor/kernel**を実行し、`nvidia-smi`で選択したGPUとworkerが見るdevice 0のcompute capabilityが一致することを、大容量モデル取得前に検証します。ASRはCTranslate2が実際に返すsupported compute typesからCUDA型を選び、実行不能ならauto時だけCPUへ安全にfallbackします。

CUDA setupでは、setup時にlogical CUDA device 0のGPU UUID・compute capability・NVIDIA driver versionを記録します。GPU交換、`CUDA_VISIBLE_DEVICES`変更によるdevice 0の物理GPU変更、またはdriver更新を検出した場合は、互換wheelであってもdirect runtimeを開始せず`persona setup --backend auto`による**実CUDA kernel preflightの再実行**を要求します。同じlock/backendが引き続き適合する場合は既存環境と検証済みmodel assetを再利用するため、再setupは全モデルの再取得を意味しません。Seed-VCは古いPyTorch 2.4/cu124 stackを隔離しているため、Blackwellなどmain cu128 stackは使えるがSeed-VC cu124だけ非互換な場合は、次回setupでSeed-VCだけCPUへ安全にfallbackします。詳細は`docs/TROUBLESHOOTING.md`を参照してください。

`HF_TOKEN`は`pyannote/speaker-diarization-community-1`の初回取得時のみ必要です。PersonaVoiceはtokenを保存しません。必要なら`.env.example`をリポジトリrootの`.env`へコピーできます。読み込むのは`HF_TOKEN`、Modal credential、`PERSONAVOICE_MODAL_*`の明示allowlistだけで、起動元processの環境変数を上書きしません。secret値はdoctor/status/reportへ返さず、`.env`の変数展開やcommand実行も行いません。通常のoffline prepare/inferenceではtokenを不要とし、モデルをremoteへ黙って問い合わせません。`persona setup`は固定upstream revision取得、コミット済みlockを使った各独立`uv`環境のsync、GPU runtime preflight、モデル取得、offline model load検証まで実行します。

```bash
uv run --locked persona doctor
uv run --locked persona doctor --deep
```

`doctor --deep`はworkerのモデルロードだけでなく、Irodoriのoffline smoke synthesis、選択したGPU backend、現在GPUとのruntime compatibility、FFmpeg shared runtime、lockfile、Irodori/Seed-VC vendorの固定revisionとclean状態も検証します。通常の推論・`doctor --deep`はローカルにmaterialize済みの固定assetだけを使用します。モデル取得を許可する経路は明示的な`persona setup`です。

## 人物作成 / 学習

```bash
uv run --locked persona init alice --authorized
```

```text
personas/alice/raw/       # 動画・音声素材
personas/alice/identity/  # 本人だけが話す綺麗な参照音声を1〜3本以上推奨
```

```bash
uv run --locked persona build alice
```

処理概要:

```text
raw media
 -> SHA / ffprobe
 -> identical-content deduplication
 -> 48kHz mono FLAC
 -> faster-whisper word timestamps
 -> pyannote regular/exclusive diarization + embeddings
 -> identity照合
 -> 本人不在sourceは学習対象から除外して記録
 -> word-boundary発話分割
 -> overlap / speaker / ASR quality filtering
 -> target clips
 -> SenseVoice batch emotion/non-verbal analysis
 -> canonical SQLite/JSON
 -> Irodori/LFM/Seed-VC datasets + reference banks
 -> executor非依存TrainingPlan
 -> Irodori full / LoRA / Speaker Inversion
 -> LFM full / LoRA
 -> optional Seed-VC FT
 -> held-out evaluation + quality gate
 -> quality合格candidateだけをpublish
```

同じ音源を別名で`raw/`へ複数置いてもfull SHA256で1素材として扱い、重複パスはprovenanceとして記録します。`identity/`参照がある状態で、ある動画のbest speaker similarityが`prepare.min_identity_similarity`を下回った場合、その動画は「本人不在の可能性が高い正常な入力」としてTTS/VC学習対象から外し、`dataset/skipped_sources.json`へ記録して他の動画処理を継続します。diarizationがspeaker embedding自体を返せない等の処理異常はskipせずエラーにします。全素材から利用可能な本人発話が1件も得られなければprepare自体を失敗させます。

ASR・diarization・SenseVoiceはbatch workerとしてモデルを一度だけロードします。各batchは1項目の成功ごとにatomic checkpointを残し、worker crash・PC再起動・強制終了後も、同じfingerprintで`--force`を付けずに再実行すれば中央semantic validatorを通過した完了項目を正式cacheへ昇格して再利用します。不正・truncated・wrong-id checkpointは捨てて再計算します。長時間処理中は別ターミナルから`uv run --locked persona status alice`を実行すると、`audit.prepare.batch_progress`でworker/phase、`completed / total`、現在のitem ID、失敗数、checkpoint済み成功数を確認できます。stage OS lockが実行中判定のsource-of-truthで、progressファイル自体をliveness判定には使いません。

同じfingerprintで失敗/中断した通常の再実行は安全なcache/checkpointから再開します。`raw/`, `identity/`, prepare設定、固定model revision、前処理worker lockが変わった場合は依存するprepare artifactだけを、training dataset、family method/optimizer、base/implementation/worker lockが変わった場合は互換性を失ったfamily artifactだけを無効化します。executor、remote consent、quality thresholdだけの変更では、正常なprepare/latent/family checkpointを捨てません。リポジトリ/人物フォルダを移動した場合も、absolute-path exportを古い場所のまま再利用しません。`--force`は完了済みstage resultを明示的に迂回してstageへ再進入します。`prepare --force`はprepare派生cacheを破棄しますが、`train --force`はprepare cacheを消さず、family固有の完全artifact/checkpoint再利用契約も維持します。

```bash
uv run --locked persona prepare alice
uv run --locked persona train alice --executor auto
uv run --locked persona eval alice
uv run --locked persona status alice
uv run --locked persona build alice --force
```

新規personaの既定はIrodori/LFMともにfullです。`auto`は同じ手法・同じTrainingPlanを保ったまま保守的なlocal full-training preflightを行い、通ればlocal、通らなければremote consentとModal SDK/authを確認してModalへ送ります。VRAM不足を理由にfullをLoRAへ黙って変更しません。`--executor local|modal|auto`はその1回だけのoverrideで、`persona.yaml`を書き換えず、executorを変えても同じplan/checkpoint fingerprintを維持します。

```yaml
training:
  schema_version: 2
  executor: auto              # auto | local | modal
  remote_data_authorized: false
  irodori:
    enabled: true
    method: full              # full | lora | speaker-inversion
    auxiliary_speaker_inversion: false
    max_steps: 4000
  lfm:
    enabled: true
    method: full              # full | lora
    epochs: 3
    learning_rate: 0.00002
  seed_vc:
    finetune: false
    max_steps: 1000
  quality_gate:
    min_lfm_expected_similarity: 0.35
    max_lfm_expected_cer: 0.85
    max_lfm_expected_wer: 1.0
    min_lfm_required_phrase_coverage: 1.0
    max_lfm_base_similarity_regression: 0.10
```

明示的に有効化された学習が最低dataset条件を満たさない場合は黙って無効化せずエラーにします。不要なfamilyは対応する`enabled`または`finetune`を明示的に`false`へ変更してください。fullのlocal preflightに失敗した場合も設定したmethodは保持されます。

### Modalとremote consent

Remote処理には、persona全体の`consent.authorized: true`に加えて、remote転送専用の`training.remote_data_authorized: true`が必須です。汎用のconsent scopeやModal認証だけでは代替できません。remote専用許可の確認はModal認証、bundle作成、uploadより先に行います。送信bundleはcanonical TrainingPlan、LFMの会話JSONL、Irodoriの事前計算済みlatentと二種類のmanifest、およびchecksum/completion markerだけです。Irodoriのsource manifestは許可済みのtext/caption等metadataがsanitized runtime manifestへlosslessに保存されたことをchecksum付きで検証する監査証跡で、sanitized manifestはcontent hash化したlatent相対pathをremote runtimeで使います。両者ともraw audioや絶対local pathを含まず、実データと照合して相違を拒否します。`raw/`、`identity/`、reference audio、SQLite、`.env`、token、絶対local pathは拒否されます。localとModalはbyte-identicalなTrainingPlanを使います。Modalからはplan/family fingerprint、checksum、remote candidate-selection contractが一致するcandidateだけを取得し、その後のlocal held-out quality gateに合格した場合だけ推論用にpublishします。Seed-VC fine-tuningはaudioをbundleへ含めないprivacy契約のためlocal-onlyです。

```bash
# Modal SDKはlock済みoptional extraとしてproduction machineだけに導入
uv sync --locked --extra modal
uv run --locked --extra modal modal setup
uv run --locked --extra modal modal deploy -m personavoice.modal_app

# .env、Modal profile、または親processにcredentialを設定した後
uv run --locked --extra modal persona doctor
uv run --locked --extra modal persona train alice --executor modal
uv run --locked --extra modal persona status alice
```

既定deploymentはapp `personavoice-training`、training Function `train`、terminal recovery Function `recover_terminal_claim`、Volume `personavoice-training`、plan claim Dict `personavoice-training-claims`、HF用Secret `personavoice-huggingface`、GPU `A100-40GB`です。deploy前にModal dashboardで同名Secretを作成し、`HF_TOKEN` keyだけを登録してください。tokenをshellのargvやroot `.env` uploadへ載せず、remote asset materializationにだけ注入します。claim Dict名は常に`<app名>-claims`から決まり、secretや学習dataではなくplan/family fingerprintとModal call IDだけを保持します。root `.env`または親processの`PERSONAVOICE_MODAL_APP`、`PERSONAVOICE_MODAL_VOLUME`、`PERSONAVOICE_MODAL_GPU`はbundled appのdeploy設定とtraining clientへ共通に反映されます。`PERSONAVOICE_MODAL_FUNCTION`はclient側のtraining Function lookup overrideで、bundled appを使う場合は既定の`train`のままにします。互換な別deploymentを利用する場合だけ、その公開Function名へ合わせてください。Modal SDKは通常のlocal-only setupには不要で、`modal` extraもlockに含まれるためproductionで未固定の`pip install modal`や`uv run --with modal`を使う必要はありません。公式の認証とdeploy仕様は[Modal getting started](https://modal.com/docs/guide)、[Modal Secrets](https://modal.com/docs/guide/secrets)、[`modal deploy`](https://modal.com/docs/cli/latest/deploy)を参照してください。

`status`はsecret値を表示せず、local preflight、plan fingerprint、Modal call ID、remote state、step/checkpoint、familyごとのcandidate/published状態を報告します。最初のremote writeより前に、送信する全fileの相対path/role/SHA256/bytes、件数、総bytes、bundle/transfer fingerprintもatomic stateへ保存します。preemptionやCLI中断後は同じplanで同じcommandを再実行すると保存済みcall IDをpollし、remote checkpointから継続します。Modalが`spawn()`を受理してからlocal call ID保存までの瞬間にprocessが停止して再送されても、plan claimが一つのcanonical FunctionCallだけをtrainerへ通し、重複callはそのcallへredirectされます。quality/executor policyだけが異なり同じfamily checkpoint namespaceを再利用する別plan同士も、sorted family claimで書込みを直列化するため競合しません。通常のtrainer失敗時はclaimを解放し、同じcheckpointから新しいcallで回復できます。hard timeout等でModal自身のretryを使い切った場合は、clientがserialized recovery Functionへ問い合わせます。このFunctionがold callのterminal失敗をModal側で再確認したときだけ、そのcall所有のfamily/plan claimを順番に解放し、local stateを`failed-recoverable`にします。表示された同じtrain commandを再実行すると新callが完全検証済みのmethod-native checkpointだけから再開します。running/完了済み/状態不明のcallや別ownerのclaimは解放しないため、claim Dictを手動clearしないでください。

### Quality gateと公開

学習完了直後の成果物は`.candidates`配下のcandidateであり、推論用published artifactではありません。held-out speaker similarity、CER/WER、unseen pronunciation、validation loss、duration ratio、emotion/style、base Japanese CER regressionをすべて有限値として評価し、設定したquality gateに合格したcandidateだけをatomicにpublishします。Irodori fullは外付けidentityへ依存しない`--no-ref` + caption/emotionをpublication gateの標準経路とし、全held-out promptについてspeaker-conditioned / no-reference / caption-conditionedの3経路を同じ指標一式で比較してreportへ残します。公開済みfull modelの`reference_mode: auto`も既定では`--no-ref`になり、明示したaudio/speaker conditioningだけがそれを上書きします。

LFM full/LoRAもJSON shapeだけでは公開しません。固定held-out promptごとにcandidateと固定base modelをgreedy推論し、期待completionに対するnormalized similarity、CER、Japanese-aware WER、必須句coverage、baseに対する最大similarity regressionを測定します。上記の既定thresholdを全件・全指標で満たし、candidate/baseの両方が非空の`text`、`voice.caption`、`voice.emotion`、文字列配列`voice.events`を含むJSONを返した場合だけ合格します。欠損・壊れたJSON・`NaN`・無限値は良いcaseだけの平均へ落とさずfail-closedです。

評価前にはIrodori本文に加え、LFM exporterのuser wrapperを実dialogue行へ分解し、assistant completion JSONの`text`も抽出します。固定promptと期待answerはNFKC、case folding、句読点、空白を正規化した完全発話一致で照合します。wrapper内へ一行として埋め込まれたheld-out発話は拒否しますが、より長い別発話に同じ文字列が含まれるだけではsubstring collisionにしません。評価prompt・metric・LFM deterministic workerの実装hashはTrainingPlanのpublication policyへ記録されるため変更時は再評価されますが、family optimization fingerprintには入らず、互換なprepare cache・latent・checkpointは維持されます。`persona status`の`candidate_complete`と`published_complete`を区別してください。

### v0.3設定の明示migration

v0.3のflat training設定は読み込み時にmemory上で無損失変換されますが、ファイルは自動保存しません。特に旧`irodori_lora: true`かつ`irodori_speaker_inversion: true`は`method: lora` + `auxiliary_speaker_inversion: true`として保持され、new defaultのfullへ置き換えません。確認後にだけ明示commandで保存してください。`consent`のようにconfigを書き換える別commandもlegacy migrationへの便乗保存を拒否します。

```bash
uv run --locked persona migrate-config alice --dry-run
uv run --locked persona migrate-config alice
```

## 生成

Irodoriのduration predictorとlatent tail trimmingはupstreamの既定値へ暗黙依存せず、
`persona.yaml`の`inference.duration_scale`、`inference.trim_tail`、`inference.tail_*`
で明示的に制御されます。既存personaを再学習せずにIssue #33の同一seed A/Bを確認するには、
次を実行してください。

```bash
uv run --locked persona diagnose-boundaries alice
```

診断レポート、ASR/CER、最終token保持、energy envelope、reference fingerprint、
target-machine listening手順は[`docs/BOUNDARY_DIAGNOSTICS.md`](docs/BOUNDARY_DIAGNOSTICS.md)
にまとめています。診断や推論設定の変更はPrepare/LFM/学習checkpointを無効化しません。

```bash
uv run --locked persona say alice "おはよう"
uv run --locked persona say alice "えっ、本当に？" --style surprised
uv run --locked persona say alice "やった！" --emotion happy
uv run --locked persona say alice "ふぅ……疲れた" --event sigh
uv run --locked persona say alice "こんにちは" --ref happy
uv run --locked persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

Issue #26のschema-v2 trainingでは、Irodori LoRAもvalidation-loss best checkpointを必須とし、選択したadapterだけをchecksum/provenance付きportable `selected/` candidateへ変換します。`checkpoint_final`とtrainer stateはv0.3互換・resume証跡として元のrun directoryに保持され、best candidateへ混入しません。legacy推論だけは従来どおりbest checkpointを優先し、存在しない場合に`checkpoint_final`へフォールバックします。生成後はWAVが実際に作成されたことも検証します。Irodori v4.1 Smallのcombined referenceは設定段階でも120秒以下へ制限します。

## Audio → Persona

```bash
uv run --locked persona reenact alice acting.wav
uv run --locked persona reenact alice acting.wav --timbre-only
uv run --locked persona repeat alice input.wav
```

`reenact`はsourceの演技/間/抑揚をVoice Conversionで維持し、`repeat`は内容・感情を解析してIrodoriで本人として再演します。文字起こし不能な非言語音だけの場合、`repeat`は`reenact`へフォールバックします。Seed-VC出力はrequestごとの独立directoryに保存され、再実行や同時requestでupstreamの決定的な出力名が衝突しないようにしています。

## 会話

```bash
uv run --locked persona chat alice
uv run --locked persona chat alice "今日は何してた？"
```

LFMが本文と`voice.caption`, `voice.emotion`, `voice.events`を計画しIrodoriへ渡します。構造化JSONが崩れた場合もplain-text fallbackを行い、型の壊れたcaption/emotion/eventsは正規化してからTTSへ渡します。

## Web UI / API

```bash
uv run --locked persona ui
uv run --locked persona serve --host 127.0.0.1 --port 8848
```

UI: Talk/Voice Design, emotion/non-verbal/reference, Reenact, Repeat, Chat, WAV再生。

API:
- `GET /health`
- `GET /v1/personas`
- `GET /v1/output/{persona}/{path}`
- `POST /v1/tts`
- `POST /v1/voice-convert`
- `POST /v1/repeat`
- `POST /v1/chat`

認証を持たないため非loopback bindは`--allow-remote`なしでは拒否します。TTS/VC/repeat/chatの重量処理はサーバ内でsingle-flight実行し、1台のローカルGPUへ複数の巨大model処理が同時投入されてOOMする経路を抑えます。

## uv環境 / lock運用

```text
root .venv                 orchestration / CLI / API
workers/asr/.venv          faster-whisper
workers/diarization/.venv  pyannote.audio
workers/sense/.venv        SenseVoiceSmall
workers/lfm/.venv          Transformers / TRL / PEFT
workers/seed_vc/.venv      Seed-VC compatible Python 3.10
vendor/Irodori-TTS/.venv   pinned official Irodori environment
```

rootと全workerの`uv.lock`はリポジトリへコミットされています。Irodoriは固定upstream checkoutを直接変更しないため、監査済みproject overlayとlockを`locks/Irodori-TTS.pyproject.toml` / `locks/Irodori-TTS.uv.lock`として管理し、setup時だけ原子的に一時適用してvendor checkoutを元のclean状態へ戻します。

通常利用ではbootstrap/setupが`--locked`で同期し、依存定義とlockがずれていれば失敗させます。environment contractにはlockだけでなくGPU/backend選択・FFmpeg runtime・worker起動policyの実装SHA256も含めるため、これらの安全性ロジックが更新されたcheckoutで古いsetup環境をcurrent扱いしません。依存を意図的に更新した時だけlock更新scriptを実行し、生成されたlock差分をレビューしてください。prepare/training cache fingerprintにも関連worker lockのSHA256を含めているため、依存更新後に古い解析結果やadapterを黙って再利用しません。
