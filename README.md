# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **専用音声 + 会話スタイル + Voice Conversion** を構築・実行するためのオーケストレーターです。

## Quick Start

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."  # pyannote Community-1初回取得時のみ
uv run persona setup
uv run persona init alice --authorized
# personas/alice/raw/ と personas/alice/identity/ に素材を配置
uv run persona build alice
uv run persona ui
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
- canonical SQLite dataset、content cache、途中再開、入力変更時の自動invalidaton
- rootと各ML stackを独立した`uv`環境に隔離

## セットアップ

必要: `uv`, Git, FFmpeg/ffprobe。NVIDIA GPU推奨。

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

`HF_TOKEN`は`pyannote/speaker-diarization-community-1`の初回取得時のみ必要です。PersonaVoiceはtokenを保存しません。`persona setup`は固定upstream revision取得、各独立`uv`環境のsync、モデル取得、offline model load検証まで実行します。

```bash
uv run persona doctor
uv run persona doctor --deep
```

## 人物作成 / 学習

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

ASR・diarization・SenseVoiceはbatch workerとしてモデルを一度だけロードします。同じfingerprintでの中断はcache/checkpointから再開し、`raw/`, `identity/`, prepare設定、training dataset/設定の変更時は依存artifactを無効化します。

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

```bash
uv run persona reenact alice acting.wav
uv run persona reenact alice acting.wav --timbre-only
uv run persona repeat alice input.wav
```

`reenact`はsourceの演技/間/抑揚をVoice Conversionで維持し、`repeat`は内容・感情を解析してIrodoriで本人として再演します。文字起こし不能な非言語音だけの場合、`repeat`は`reenact`へフォールバックします。

## 会話

```bash
uv run persona chat alice
uv run persona chat alice "今日は何してた？"
```

LFMが本文と`voice.caption`, `voice.emotion`, `voice.events`を計画しIrodoriへ渡します。

## Web UI / API

```bash
uv run persona ui
uv run persona serve --host 127.0.0.1 --port 8848
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

各`uv sync`は対象プロジェクトの`uv.lock`をローカル生成/更新します。lock更新用に`scripts/lock_all.ps1` / `scripts/lock_all.sh`があります。Irodori backendは`persona setup --backend auto|cu128|cpu|rocm|xpu`で選択できます。

## テスト / 実機検証

GitHub Actions `core-ci` はLinux/Windowsで`uv sync`, Ruff, pytest, compileall, CLI smokeを実行します。数GB級weight/GPUはCIに持ち込まず、対象実機上の`persona setup` + `persona doctor --deep`でoffline model loadを検証します。

## ローカルデータ / 同意

素材、dataset、model、output、vendor、runtime requestはgitignore対象です。`consent.authorized: true`でないpersonaではprepare/train/voice generationを拒否します。ローカル利用、配布、公開、商用利用の許可範囲は別々に管理してください。

詳細: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) / [`docs/MODELS.md`](docs/MODELS.md) / [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
