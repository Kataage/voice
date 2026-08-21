# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **専用音声 + 会話スタイル + Voice Conversion** を構築・実行するためのオーケストレーターです。

## できること

- 動画/音声を`raw/`へ置き、`persona build NAME`でprepare → train → eval
- faster-whisper: 日本語ASR + word timestamps
- pyannote Community-1: regular/exclusive diarization + speaker embeddings
- `identity/`の本人音声から対象話者を自動選択
- SenseVoiceSmall: 感情 + 笑い・泣き・息・咳・くしゃみ等のイベント解析
- Irodori-TTS v4.1 Small: reference voice + VoiceDesign + Speaker Inversion + LoRA
- LFM2.5-1.2B-JP-202606: 会話/口調LoRA + 発話スタイル計画
- Seed-VC v2: 入力音声の間・抑揚・演技を使ったVoice Conversion
- `say`, `reenact`, `repeat`, `chat`, Web UI, localhost REST API
- canonical SQLite dataset、content cache、途中再開、入力変更時の自動invalidaton
- rootと各ML stackを独立した`uv`環境に隔離

## 1. 初回セットアップ

必要なシステムツール:

- `uv`
- Git
- FFmpeg / ffprobe
- NVIDIA GPU推奨

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

`HF_TOKEN`は`pyannote/speaker-diarization-community-1`の**初回取得時だけ**必要です。Hugging Face上で利用条件に同意してから設定してください。PersonaVoiceはtokenを設定ファイルへ保存しません。

`persona setup`は次を実行します。

1. Irodori-TTS / Seed-VCを固定revisionで`vendor/`へ取得
2. rootとは別に各workerを`uv sync`
3. 必要モデルをローカルへ取得
4. 各workerでモデルをoffline loadして検証

再実行時は取得済みモデルを再利用します。

環境だけ先に作る場合:

```bash
uv run persona setup --skip-models --no-verify
```

診断:

```bash
uv run persona doctor
uv run persona doctor --deep
```

`--deep`はASR / pyannote / SenseVoice / LFM / Seed-VCを実際にoffline loadします。

## 2. 人物作成

```bash
uv run persona init alice --authorized
```

配置するのは基本的に次の2箇所です。

```text
personas/alice/raw/
  video01.mp4
  stream02.mkv
  audio03.wav

personas/alice/identity/
  clean_target_01.wav
  clean_target_02.wav
```

`identity/`には本人だけが話している綺麗な短い音声を1〜3本以上置くことを推奨します。複数話者素材から本人を自動選択する基準になります。

## 3. 一発ビルド

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
  -> identity embeddingとの照合
  -> word boundaryに沿った発話分割
  -> overlap / speaker / ASR quality filtering
  -> target clip生成
  -> SenseVoice emotion / non-verbal event batch analysis
  -> master.sqlite3 + master.json
  -> Irodori / LFM / Seed-VC dataset export
  -> default / emotion別 reference bank
  -> Irodori Speaker Inversion + LoRA
  -> LFM LoRA
  -> Seed-VC FT（設定で有効化した場合）
  -> evaluation report
```

ASR・diarization・SenseVoiceは素材/clipごとにモデルを再ロードせず、batch workerとして処理します。

個別実行:

```bash
uv run persona prepare alice
uv run persona train alice
uv run persona eval alice
uv run persona status alice
```

中断した場合は同じコマンドを再実行してください。同じfingerprintならcache/checkpointから再開します。

`raw/`, `identity/`, prepare設定、training dataset/設定が変化した場合は、古いspeaker判定・Irodori latent・adapter等を完成済みと誤認せず、関連artifactを自動的に作り直します。

明示的に再構築:

```bash
uv run persona build alice --force
```

## 4. 音声生成

通常:

```bash
uv run persona say alice "おはよう"
```

VoiceDesign:

```bash
uv run persona say alice "えっ、本当に？" --style surprised
uv run persona say alice "大丈夫？" --style "かなり心配しながら優しく"
```

感情:

```bash
uv run persona say alice "やった！" --emotion happy
```

非言語cue:

```bash
uv run persona say alice "ふぅ……疲れた" --event sigh
uv run persona say alice "" --event laugh
```

prepareが作った感情別reference:

```bash
uv run persona say alice "こんにちは" --ref happy
```

任意reference:

```bash
uv run persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

## 5. Audio → Persona

入力音声の演技・間・抑揚をできる限り維持:

```bash
uv run persona reenact alice acting.wav
```

音色寄りに変換:

```bash
uv run persona reenact alice acting.wav --timbre-only
```

同じ内容を解析し、本人としてIrodoriで再演:

```bash
uv run persona repeat alice input.wav
```

文字起こしできない非言語音だけが検出された場合はVoice Conversionへフォールバックします。

## 6. 会話

```bash
uv run persona chat alice
```

1ターン:

```bash
uv run persona chat alice "今日は何してた？"
```

LFMは本文だけでなく`voice.caption`, `voice.emotion`, `voice.events`をJSONで計画し、その条件をIrodoriへ渡します。

## 7. Web UI / API

```bash
uv run persona ui
```

ブラウザから以下を操作できます。

- Talk / Voice Design
- emotion / non-verbal events / reference
- Reenact
- Repeat
- Chat
- 生成WAV再生

API:

```bash
uv run persona serve --host 127.0.0.1 --port 8848
```

主なendpoint:

- `GET /health`
- `GET /v1/personas`
- `GET /v1/output/{persona}/{path}`
- `POST /v1/tts`
- `POST /v1/voice-convert`
- `POST /v1/repeat`
- `POST /v1/chat`

認証を持たないためデフォルトはloopback限定です。非loopback bindは`--allow-remote`を明示しない限り拒否します。

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

lock生成:

```powershell
.\scripts\lock_all.ps1
```

または:

```bash
./scripts/lock_all.sh
```

IrodoriのPyTorch backendは`persona setup --backend auto|cu128|cpu|rocm|xpu`で選べます。

## テスト

GitHub Actions `core-ci` はLinux/Windowsの両方で以下を検証します。

```text
uv sync
ruff
pytest
compileall
persona --help
```

数GB級weight/GPUをCIへ持ち込む代わりに、実機では`persona setup`が最後に`persona doctor --deep`相当のoffline model loadを行います。

## ローカルデータ / 同意

素材、学習データ、モデル、出力、vendor checkout、runtime requestはgitignore対象です。

`consent.authorized: true`でないpersonaではprepare/train/voice generationを拒否します。ローカル利用、第三者配布、公開、商用利用などの許可範囲は別々に管理してください。

詳細:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/MODELS.md`](docs/MODELS.md)
- [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
