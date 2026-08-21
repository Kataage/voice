# PersonaVoice

許可を得た話者の動画・音声素材から、ローカルだけで **音声モデル + 会話スタイルモデル + Voice Conversion** を作るためのオーケストレーターです。

## できること

- 動画/音声を `raw/` に置くだけで音声抽出、ASR、話者分離、本人照合、感情/非言語イベント解析、学習データ生成
- Irodori-TTS v4.1 Small: Speaker Inversion + VoiceDesign LoRA
- LFM2.5-1.2B-JP-202606: 会話・口調 LoRA
- Seed-VC v2: 参照音声の演技/間/抑揚を保った Voice Conversion（FTは任意）
- `say`, `reenact`, `repeat`, `chat`, Web UI, localhost API
- SHA/cache + stage fingerprintで途中再開
- root と各ML workerを別々の `uv` 環境に隔離

## 1. 初回セットアップ

Python環境は `uv` で管理します。FFmpegとGitはシステム側に必要です。

Windows:

```powershell
.\scripts\bootstrap.ps1
$env:HF_TOKEN="hf_..."   # pyannote Community-1 の利用条件をHF上で承諾したtoken
uv run persona setup
```

Linux/macOS:

```bash
./scripts/bootstrap.sh
export HF_TOKEN=hf_...
uv run persona setup
```

`persona setup` は、固定したupstream revisionを `vendor/` にcloneし、各workerの `.venv` を `uv sync` し、必要モデルを `models/` に保存します。セットアップ後の通常処理は `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` で実行します。

> `HF_TOKEN` は設定ファイルに保存しません。`pyannote/speaker-diarization-community-1` の初回取得にだけ必要です。

## 2. 人物作成

```bash
uv run persona init alice --authorized
```

生成された次の2箇所だけに素材を入れます。

```text
personas/alice/raw/       # 動画、配信、音声など素材全部
personas/alice/identity/  # 本人だけが話している綺麗な短い音声を1〜3本推奨
```

複数話者素材では `identity/` が本人判定の基準になります。

## 3. 一発ビルド

```bash
uv run persona build alice
```

内部で `prepare -> train -> eval` を連続実行します。中断後に同じコマンドを再実行すると、完成済みキャッシュ/チェックポイントを再利用します。

個別実行も可能です。

```bash
uv run persona prepare alice
uv run persona train alice
uv run persona eval alice
uv run persona status alice
```

## 4. 生成

```bash
# 普通に発話
uv run persona say alice "おはよう"

# VoiceDesign caption
uv run persona say alice "えっ、本当に？" --style surprised
uv run persona say alice "大丈夫？" --style "かなり心配しながら優しく"

# 感情
uv run persona say alice "やった！" --emotion happy

# 非言語（Irodori emoji条件へ変換）
uv run persona say alice "ふぅ……疲れた" --event sigh
uv run persona say alice "" --event laugh

# prepare が作った感情別参照bank
uv run persona say alice "こんにちは" --ref happy

# 任意の参照音声
uv run persona say alice "こんにちは" --ref C:\path\to\reference.wav
```

### 音声 → 本人声（演技を維持）

```bash
uv run persona reenact alice acting.wav
```

Seed-VC v2 のstyle conversionを使い、sourceのタイミング・抑揚・感情表現をできる限り維持したまま対象話者の音色へ変換します。

```bash
uv run persona reenact alice acting.wav --timbre-only
```

ならstyle transferを切れます。

### 同じ内容を本人として再演

```bash
uv run persona repeat alice input.wav
```

ASR + SenseVoiceで内容/感情/イベントを取り、Irodoriで本人として言い直します。文字起こしできない非言語音だけなら自動的に `reenact` にフォールバックします。

### 会話

```bash
uv run persona chat alice
# 1ターンだけ
uv run persona chat alice "今日は何してた？"
```

LFM LoRAが `{text, voice.caption, voice.emotion, voice.events}` を生成し、その発話計画をIrodoriへ渡します。

## 5. UI / API

```bash
uv run persona ui
uv run persona serve --host 127.0.0.1 --port 8848
```

API:

- `GET /health`
- `GET /v1/personas`
- `POST /v1/tts`
- `POST /v1/voice-convert`
- `POST /v1/repeat`
- `POST /v1/chat`

デフォルトは localhost のみにbindします。

## 自動処理の概要

```text
raw media
  -> lossless mono FLAC cache
  -> faster-whisper (word timestamps)
  -> pyannote Community-1 (regular + exclusive diarization + speaker embeddings)
  -> identity/ embedding と照合して本人話者選択
  -> overlap/quality scoring
  -> target clips
  -> SenseVoiceSmall (emotion + laughter/cry/breath/cough/...)
  -> canonical SQLite + JSON
  -> Irodori / LFM / Seed-VC 各dataset
  -> references (default + by_emotion)
  -> Speaker Inversion + Irodori LoRA + LFM LoRA
  -> evaluation report
```

詳細は [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) と [`docs/MODELS.md`](docs/MODELS.md) を参照してください。

## プライバシー / 同意

`consent.authorized: true` でない人物について `prepare`, `train`, voice generation は実行しません。素材・モデル・出力はgitignoreされ、ローカルの `personas/`, `models/`, `vendor/` にのみ置きます。第三者公開・配布・商用利用などは、本人から得た許可範囲を別途確認してください。
