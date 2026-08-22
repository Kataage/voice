# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **専用音声 + 会話スタイル + Voice Conversion** を構築・実行するためのオーケストレーターです。

## Quick Start

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."  # pyannote Community-1初回取得時のみ
uv run --locked persona setup
uv run --locked persona init alice --authorized
# personas/alice/raw/ と personas/alice/identity/ に素材を配置
uv run --locked persona build alice
uv run --locked persona ui
```

## 機能

- faster-whisper `large-v3`: 日本語ASR + word timestamps
- pyannote Community-1: regular/exclusive diarization + speaker embeddings
- `identity/`本人参照による対象話者自動選択
- SenseVoiceSmall: 感情 + 笑い・泣き・息・咳・くしゃみ等のイベント解析
- Irodori-TTS v4.1 Small: reference voice + VoiceDesign + Speaker Inversion + LoRA
- LFM2.5-1.2B-JP-202606: 会話/口調LoRA + 発話スタイル計画
- Seed-VC v2: 入力音声の間・抑揚・演技を使ったVoice Conversion
- `say`, `reenact`, `repeat`, `chat`, Web UI, localhost REST API
- canonical SQLite dataset、content cache、途中再開、入力変更時の自動invalidation
- rootと各ML stackを独立した`uv`環境に隔離
- root / worker / Irodoriを監査済み`uv.lock`へ固定
- 推論・deep doctorは監査済みローカルmodelだけを使用し、欠損時にremote modelへ黙ってfallbackしない

## セットアップ

必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。WindowsではTorchCodec用にshared DLL付きFFmpeg 4〜8が必要で、bootstrapはWinGetのFFmpeg Shared 8.1.1を使用します。

Windows:

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."
uv run --locked persona setup
```

Linux/macOS:

```bash
./scripts/bootstrap.sh
export HF_TOKEN=hf_...
uv run --locked persona setup
```

`HF_TOKEN`は`pyannote/speaker-diarization-community-1`の初回取得時のみ必要です。PersonaVoiceはtokenを保存しません。`persona setup`は固定upstream revision取得、コミット済みlockを使った各独立`uv`環境のsync、モデル取得、offline model load検証まで実行します。

```bash
uv run --locked persona doctor
uv run --locked persona doctor --deep
```

`doctor --deep`はworkerのモデルロードだけでなく、Irodoriのoffline smoke synthesis、選択したGPU backend、lockfile、Irodori/Seed-VC vendorの固定revisionとclean状態も検証します。通常の推論・`doctor --deep`はローカルにmaterialize済みの固定assetだけを使用します。モデル取得を許可する経路は明示的な`persona setup`です。

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
 -> Irodori Speaker Inversion + LoRA
 -> LFM LoRA
 -> optional Seed-VC FT
 -> evaluation
```

同じ音源を別名で`raw/`へ複数置いてもfull SHA256で1素材として扱い、重複パスはprovenanceとして記録します。`identity/`参照がある状態で、ある動画のbest speaker similarityが`prepare.min_identity_similarity`を下回った場合、その動画は「本人不在の可能性が高い正常な入力」としてTTS/VC学習対象から外し、`dataset/skipped_sources.json`へ記録して他の動画処理を継続します。diarizationがspeaker embedding自体を返せない等の処理異常はskipせずエラーにします。全素材から利用可能な本人発話が1件も得られなければprepare自体を失敗させます。

ASR・diarization・SenseVoiceはbatch workerとしてモデルを一度だけロードします。同じfingerprintで失敗/中断した通常の再実行は安全なcache/checkpointから再開します。`raw/`, `identity/`, prepare設定、固定model revision、前処理worker lock、training dataset/設定、training worker lockが変わった場合は依存artifactを自動で無効化します。リポジトリ/人物フォルダを移動した場合も、absolute-path exportを古い場所のまま再利用しません。`--force`を指定した場合は、同じfingerprintの失敗後であってもprepare由来cacheを破棄して完全に作り直します。

```bash
uv run --locked persona prepare alice
uv run --locked persona train alice
uv run --locked persona eval alice
uv run --locked persona status alice
uv run --locked persona build alice --force
```

`training.lfm_lora: true`では有効な会話教師例が2件以上必要です。`training.seed_vc_finetune: true`では対象話者audio clipが2件以上必要です。明示的に有効化された学習が最低条件を満たさない場合は黙って無効化せずエラーにし、不要な機能なら`persona.yaml`で明示的に`false`へ変更します。

## 生成

```bash
uv run --locked persona say alice "おはよう"
uv run --locked persona say alice "えっ、本当に？" --style surprised
uv run --locked persona say alice "やった！" --emotion happy
uv run --locked persona say alice "ふぅ……疲れた" --event sigh
uv run --locked persona say alice "こんにちは" --ref happy
uv run --locked persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

Irodori LoRAにvalidation-loss best checkpointがある場合は推論時に最良checkpointを優先し、なければ`checkpoint_final`へフォールバックします。生成後はWAVが実際に作成されたことも検証します。Irodori v4.1 Smallのcombined referenceは設定段階でも120秒以下へ制限します。

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

rootと全workerの`uv.lock`はリポジトリへコミットされています。Irodoriは固定upstream checkoutを直接変更しないため、監査済みlockを`locks/Irodori-TTS.uv.lock`として管理し、setup時だけ一時適用してvendor checkoutを元のclean状態へ戻します。

通常利用ではbootstrap/setupが`--locked`で同期し、依存定義とlockがずれていれば失敗させます。依存を意図的に更新した時だけ次を実行し、生成されたlock差分をレビューしてください。prepare/training cache fingerprintにも関連worker lockのSHA256を含めているため、依存更新後に古い解析結果やadapterを黙って再利用しません。

```powershell
.\scripts\lock_all.ps1
```

```bash
./scripts/lock_all.sh
```

Irodori backendは`persona setup --backend auto|cu126|cu128|cpu|rocm|xpu`で選択できます。`auto`はNVIDIA device 0のcompute capabilityを見てPascal 6.xをCUDA 12.6、7.0以上をCUDA 12.8へ分けます。互換性のためTorch 2.4に固定しているSeed-VCはCUDA 12.4系へ明示的に解決します。

## テスト / 実機検証

GitHub Actions `core-ci` はLinux/Windowsの両方でrootのlocked sync、Ruff、pytest、compileall、CLI smokeを実行し、さらにASR / diarization / SenseVoice / LFM / Seed-VCの全worker環境を各OSで`uv sync --locked`して依存解決とPython compileを検証します。数GB級weight/GPUはCIに持ち込まず、対象実機上の`persona setup` + `persona doctor --deep`でoffline model loadとIrodori smoke synthesisを検証します。

## ローカルデータ / 同意

素材、dataset、model、output、vendor、runtime request、personaの実行stateはgitignore対象です。`consent.authorized: true`でないpersonaではprepare/train/voice generationを拒否します。`persona.yaml`の名前は格納directory名と一致している必要があります。ローカル利用、配布、公開、商用利用の許可範囲は別々に管理してください。

詳細: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) / [`docs/MODELS.md`](docs/MODELS.md) / [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
