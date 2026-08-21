# Architecture

## Design goals

1. 素材を置いて `persona build NAME` だけで学習まで到達する。
2. root環境を軽く保ち、Torch/Transformers競合をworkerごとのuv環境へ隔離する。
3. 一度作ったcanonical datasetはモデル交換後も再利用する。
4. 初回download後は通常処理をoffline modeで行う。
5. 途中停止を前提に、素材fingerprint・中間cache・upstream checkpointで再開する。
6. consent gate、local bind、secret非保存をデフォルトにする。

## Runtime layout

```text
root (.venv, Python 3.12)
  orchestration / CLI / API / SQLite

workers/asr          faster-whisper
workers/diarization  pyannote.audio Community-1
workers/sense        SenseVoiceSmall
workers/lfm          Transformers + TRL + PEFT
workers/seed_vc      Seed-VC compatible Python 3.10 stack

vendor/Irodori-TTS   pinned upstream, own uv env
vendor/seed-vc       pinned archived upstream source
```

Worker calls use JSON files in `.runtime/requests/`. Root launches them via `uv run --project ... --no-sync`, so a worker can pin a different Torch version without affecting the others.

## Canonical data

`dataset/master.sqlite3` is the source of truth. Every utterance stores source/time span, speaker, target flag, identity similarity, overlap, transcript, annotated text, emotion, events, caption, audio path and quality score. Export adapters create:

- `irodori_source.jsonl`
- `irodori_manifest.jsonl` + DACVAE latents
- `lfm_train.jsonl`
- `seed_vc/audio/`
- `references/`

## Speaker resolution

Community-1 returns regular diarization (overlap retained), exclusive diarization (one speaker at a time), and one embedding per speaker. `identity/` clips are forced through `num_speakers=1`, averaged, and compared by cosine similarity against every source speaker. Multi-speaker material without identity references intentionally fails rather than guessing.

## Expressive labels

SenseVoiceSmall is used acoustically, not from transcript sentiment. It provides emotion labels and events such as laughter, cry, breath, cough, sneeze and applause. These are stored model-neutrally in the master dataset. The Irodori adapter maps them to caption language and supported emoji cues.

## Training

- Irodori: upstream `prepare_manifest.py`, v4 Small LoRA config, v4 Small Speaker Inversion config.
- LFM: TRL SFT + PEFT LoRA on q/k/v/o projections.
- Seed-VC: zero-shot V2 is default; CFM fine-tuning is opt-in because upstream is archived and FT can trade WER for similarity.

## Inference modes

- `say`: text -> Irodori
- `reenact`: source audio -> Seed-VC style conversion -> target voice
- `repeat`: source audio -> ASR/SenseVoice -> Irodori
- `chat`: user text -> LFM structured voice plan -> Irodori

## Offline behavior

After `persona setup`, root passes `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to workers. Model snapshots are materialized under `models/`. Seed-VC keeps its upstream-downloaded checkpoints inside ignored local vendor storage.
