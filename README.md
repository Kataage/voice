# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **専用音声 + 会話スタイル + Voice Conversion** を構築・実行するためのオーケストレーターです。

詳細なセットアップと操作方法はこのREADMEに集約しています。基本フローは次の通りです。

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."  # pyannote Community-1初回取得時のみ
uv run persona setup
uv run persona init alice --authorized
# personas/alice/raw/ と personas/alice/identity/ に素材を配置
uv run persona build alice
uv run persona ui
```

## 主要機能

- faster-whisper: 日本語ASR + word timestamps
- pyannote Community-1: regular/exclusive diarization + speaker embeddings
- `identity/`本人参照による対象話者自動選択
- SenseVoiceSmall: 感情 + 笑い・泣き・息・咳・くしゃみ等の音響イベント解析
- Irodori-TTS v4.1 Small: reference voice + VoiceDesign + Speaker Inversion + LoRA
- LFM2.5-1.2B-JP-202606: 会話/口調LoRA + 発話スタイル計画
- Seed-VC v2: 入力音声の間・抑揚・演技を使ったVoice Conversion
- `say`, `reenact`, `repeat`, `chat`, Web UI, localhost REST API
- canonical SQLite dataset、content cache、途中再開、入力変更時の自動invalidaton
- rootと各ML stackを独立した`uv`環境に隔離

## セットアップ

必要なシステムツールは`uv`, Git, FFmpeg/ffprobeです。NVIDIA GPUを推奨します。

Windows:

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."
uv run persona setup
```

Linux/macOS:

```bash
./scripts/bootstrap.sh
export HF_TOKEN=hf_...
uv run persona setup
```

`HF_TOKEN`は`pyannote/speaker-diarization-community-1`の初回取得時のみ必要です。Hugging Face上で利用条件に同意してから設定してください。tokenはプロジェクトへ保存しません。

`persona setup`は、固定したIrodori-TTS/Seed-VC revisionの取得、各独立`uv`環境のsync、モデル取得、offline model load検証まで行います。取得済みモデルは再利用します。

```bash
uv run persona doctor
uv run persona doctor --deep
```

`--deep`はASR / pyannote / SenseVoice / LFM / Seed-VCを実際にoffline loadします。

## 人物作成と一発ビルド

```bash
uv run persona init alice --authorized
```

```text
personas/alice/raw/       # 動画・音声素材
personas/alice/identity/  # 本人だけが話す綺麗な参照音声を1〜3本以上推奨
```

```bash
uv run persona build alice
```

内部処理:

```text
raw media
  -> SHA / ffprobe inventory
  -> 48kHz mono lossless FLAC cache
  -> faster-whisper word timestamps
  -> pyannote regular + exclusive diarization + embeddings
  -> identity embedding照合
  -> word boundary発話分割
  -> overlap / speaker / ASR quality filtering
  -> target clips
  -> SenseVoice emotion / non-verbal event batch analysis
  -> master.sqlite3 + master.json
  -> Irodori / LFM / Seed-VC datasets
  -> default / emotion reference banks
  -> Irodori Speaker Inversion + LoRA
  -> LFM LoRA
  -> optional Seed-VC FT
  -> evaluation report
```

ASR・diarization・SenseVoiceはbatch workerとしてモデルを1回ロードして処理します。同じfingerprintで中断した場合はcache/checkpointから再開し、`raw/`, `identity/`, prepare設定、training dataset/設定が変化した場合は関連artifactを自動的に無効化します。

個別実行:

```bash
uv run persona prepare alice
uv run persona train alice
uv run persona eval alice
uv run persona status alice
uv run persona build alice --force
```

## 生成

```bash
uv run persona say alice "おはよう"
uv run persona say alice "えっ、本当に？" --style surprised
uv run persona say alice "やった！" --emotion happy
uv run persona say alice "ふぅ……疲れた" --event sigh
uv run persona say alice "こんにちは" --ref happy
uv run persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

## Audio → Persona

演技・間・抑揚を維持するVoice Conversion:

```bash
uv run persona reenact alice acting.wav
uv run persona reenact alice acting.wav --timbre-only
```

内容・感情を解析してIrodoriで本人として再演:

```bash
uv run persona repeat alice input.wav
```

文字起こしできない非言語音だけが検出された場合はVoice Conversionへフォールバックします。

## 会話

```bash
uv run persona chat alice
uv run persona chat alice "今日は何してた？"
```

LFMは本文と`voice.caption`, `voice.emotion`, `voice.events`をJSONで計画し、その条件をIrodoriへ渡します。

## Web UI / API

```bash
uv run persona ui
uv run persona serve --host 127.0.0.1 --port 8848
```

UIからTalk/Voice Design、emotion/non-verbal/reference、Reenact、Repeat、Chat、生成WAV再生を操作できます。

API endpoint:

- `GET /health`
- `GET /v1/personas`
- `GET /v1/output/{persona}/{path}`
- `POST /v1/tts`
- `POST /v1/voice-convert`
- `POST /v1/repeat`
- `POST /v1/chat`

認証を持たないため非loopback bindは`--allow-remote`を明示しない限り拒否します。

## uv環境

```text
root .venv                 orchestration / CLI / API
workers/asr/.venv          faster-whisper
workers/diarization/.venv  pyannote.audio
workers/sense/.venv        SenseVoiceSmall
workers/lfm/.venv          Transformers / TRL / PEFT
workers/seed_vc/.venv      Seed-VC compatible Python 3.10
vendor/Irodori-TTS/.venv   pinned official Irodori environment
```

各`uv sync`は対象プロジェクトの`uv.lock`をローカル生成/更新します。まとめてlockを更新する場合:

```powershell
.\scripts\lock_all.ps1
```

または:

```bash
./scripts/lock_all.sh
```

Irodori backendは`persona setup --backend auto|cu128|cpu|rocm|xpu`で選択できます。

## テスト

GitHub Actions `core-ci` はLinux/Windowsの双方で`uv sync`, Ruff, pytest, compileall, CLI smokeを実行します。重いモデルは対象実機上の`persona setup` + `persona doctor --deep`でoffline load検証します。

## ローカルデータ / 同意

素材、学習データ、モデル、出力、vendor checkout、runtime requestはgitignore対象です。`consent.authorized: true`でないpersonaではprepare/train/voice generationを拒否します。ローカル利用、配布、公開、商用利用の許可範囲は別々に管理してください。

詳細:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/MODELS.md`](docs/MODELS.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
