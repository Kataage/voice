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

- faster-whisper: 日本語ASR + word timestamps
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

## セットアップ

必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。

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

`doctor --deep`はworkerのモデルロードだけでなく、Irodoriのoffline smoke synthesis、選択したGPU backend、lockfile、Irodori/Seed-VC vendorの固定revisionとclean状態も検証します。

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
 -> 48kHz mono FLAC
 -> faster-whisper word timestamps
 -> pyannote regular/exclusive diarization + embeddings
 -> identity照合
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

ASR・diarization・SenseVoiceはbatch workerとしてモデルを一度だけロードします。同じfingerprintで失敗/中断した通常の再実行は安全なcache/checkpointから再開します。`raw/`, `identity/`, prepare設定、training dataset/設定が変わった場合は依存artifactを自動で無効化します。`--force`を指定した場合は、同じfingerprintの失敗後であってもprepare由来cacheを破棄して完全に作り直します。

```bash
uv run --locked persona prepare alice
uv run --locked persona train alice
uv run --locked persona eval alice
uv run --locked persona status alice
uv run --locked persona build alice --force
```

## 生成

```bash
uv run --locked persona say alice "おはよう"
uv run --locked persona say alice "えっ、本当に？" --style surprised
uv run --locked persona say alice "やった！" --emotion happy
uv run --locked persona say alice "ふぅ……疲れた" --event sigh
uv run --locked persona say alice "こんにちは" --ref happy
uv run --locked persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

Irodori LoRAにvalidation-loss best checkpointがある場合は推論時に最良checkpointを優先し、なければ`checkpoint_final`へフォールバックします。生成後はWAVが実際に作成されたことも検証します。

## Audio → Persona

```bash
uv run --locked persona reenact alice acting.wav
uv run --locked persona reenact alice acting.wav --timbre-only
uv run --locked persona repeat alice input.wav
```

`reenact`はsourceの演技/間/抑揚をVoice Conversionで維持し、`repeat`は内容・感情を解析してIrodoriで本人として再演します。文字起こし不能な非言語音だけの場合、`repeat`は`reenact`へフォールバックします。

## 会話

```bash
uv run --locked persona chat alice
uv run --locked persona chat alice "今日は何してた？"
```

LFMが本文と`voice.caption`, `voice.emotion`, `voice.events`を計画しIrodoriへ渡します。構造化JSONが崩れた場合もplain-text fallbackで安全に処理します。

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

認証を持たないため非loopback bindは`--allow-remote`なしでは拒否します。

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

通常利用ではbootstrap/setupが`--locked`で同期し、依存定義とlockがずれていれば失敗させます。依存を意図的に更新した時だけ次を実行し、生成されたlock差分をレビューしてください。

```powershell
.\scripts\lock_all.ps1
```

```bash
./scripts/lock_all.sh
```

Irodori backendは`persona setup --backend auto|cu128|cpu|rocm|xpu`で選択できます。NVIDIA時はmodern Torch workerをCUDA 12.8系、互換性のためTorch 2.4に固定しているSeed-VCをCUDA 12.4系へ明示的に解決します。

## テスト / 実機検証

GitHub Actions `core-ci` はLinux/Windowsの両方でrootのlocked sync、Ruff、pytest、compileall、CLI smokeを実行し、さらにASR / diarization / SenseVoice / LFM / Seed-VCの全worker環境を各OSで`uv sync --locked`して依存解決とPython compileを検証します。数GB級weight/GPUはCIに持ち込まず、対象実機上の`persona setup` + `persona doctor --deep`でoffline model loadとIrodori smoke synthesisを検証します。

## ローカルデータ / 同意

素材、dataset、model、output、vendor、runtime request、personaの実行stateはgitignore対象です。`consent.authorized: true`でないpersonaではprepare/train/voice generationを拒否します。ローカル利用、配布、公開、商用利用の許可範囲は別々に管理してください。

詳細: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) / [`docs/MODELS.md`](docs/MODELS.md) / [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
