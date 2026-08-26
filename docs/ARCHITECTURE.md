# Architecture

## Design goals

1. 素材を置いて `persona build NAME` だけで学習まで到達する。
2. root環境を軽く保ち、Torch/Transformers競合をworkerごとのuv環境へ隔離する。
3. 一度作ったcanonical datasetはモデル交換後も再利用する。
4. 初回download後は通常処理をoffline modeで行う。
5. 途中停止を前提に、素材fingerprint・中間cache・upstream checkpointで再開する。
6. consent gate、local bind、secret非保存をデフォルトにする。
7. upstream revisionと全Python依存をlockし、同じcheckoutから同じ環境を再現できるようにする。

v0.3のCore Profile/LFM expressive-plan runtimeはv0.4と同じlogical
contractを使う。ただしこのbranchのtraining architectureは既存のLoRA/
Speaker Inversion構成のままで、v0.4のfull-model実行基盤は持ち込まない。

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

Worker calls use JSON files in `.runtime/requests/`. Root launches them via `uv run --project ... --no-sync`, so a worker can pin a different Torch version without affecting the others. Setup時に選択したIrodori/worker backendは`.runtime/setup.json`へ記録され、`doctor --deep`が期待backendと実際のCUDA可視性を照合します。

## Dependency / upstream integrity

rootと全workerの`uv.lock`はリポジトリで管理します。Irodoriは固定upstream checkoutそのものを変更しないため、対応lockを`locks/Irodori-TTS.uv.lock`として保持し、setup時だけvendorの`uv.lock`へ一時適用して`uv sync --locked`した後に元のcheckout状態へ復元します。

Irodori/Seed-VCのgit revisionはコード上で固定されます。`doctor`はrevisionだけでなくvendor checkoutがcleanであることも検証し、意図しないupstream変更やローカルpatchをready扱いしません。

## Canonical data

`dataset/master.sqlite3` is the source of truth. Every utterance stores source/time span, speaker, target flag, identity similarity, overlap, transcript, annotated text, emotion, events, caption, audio path and quality score. Export adapters create:

- `irodori_source.jsonl`
- `irodori_manifest.jsonl` + DACVAE latents
- `lfm_train.jsonl`
- `seed_vc/audio/`
- `references/`

`core_profile.yaml`はpersona directoryに属するversioned runtime contractで、
master datasetやPrepare cacheの入力ではない。

## Prepare cache policy

Lossless source audioはsource SHAから作るため`cache/audio/`を再利用できます。一方、次のartifactはASRモデル・言語・diarization・segmentation・identity条件などprepare semanticsに依存するため、prepare fingerprintまたはcache policyが変わった場合に破棄します。

- `cache/asr/`
- `cache/diarization/`
- `cache/identity/`
- `cache/sense/`
- `dataset/clips/`

同一fingerprintの失敗/中断を通常再実行する場合は上記cacheを保持して再開します。明示的な`--force`は前回のstatusに関係なくprepare由来cacheを破棄します。これにより、新metadataと古いclip/ASR結果の混在を防ぎながら中断復帰性能を維持します。

## Speaker resolution

Community-1 returns regular diarization (overlap retained), exclusive diarization (one speaker at a time), and one embedding per speaker. `identity/` clips are forced through `num_speakers=1`, averaged, and compared by cosine similarity against every source speaker. Multi-speaker material without identity references intentionally fails rather than guessing.

## Expressive labels

SenseVoiceSmall is used acoustically, not from transcript sentiment. It provides emotion labels and events such as laughter, cry, breath, cough, sneeze and applause. These are stored model-neutrally in the master dataset. The Irodori adapter maps them to caption language and supported emoji cues.

## Training

- Irodori: upstream `prepare_manifest.py`, v4 Small LoRA config, v4 Small Speaker Inversion config. Interrupted training resumes from upstream checkpoint; inference prefers the lowest validation-loss LoRA checkpoint when available.
- LFM: TRL SFT + PEFT LoRA on q/k/v/o projections. CPUはfp32、CUDAはbf16対応時bf16/それ以外fp16でロードする。`export_lfm()`はCore Profileとcanonical output schemaを使い、`lfm_contract` markerを付ける。LFM export/設定だけが変わった場合は、明示的な再学習時にLFM artifactだけを無効化し、Irodori/Seed-VCを保持する。
- Seed-VC: zero-shot V2 is default; CFM fine-tuning is opt-in because upstream is archived and FT can trade WER for similarity. Fine-tuning is blocked when the isolated Seed-VC worker cannot see CUDA.

非LFMのtraining dataset/config fingerprintが変わるか`train --force`された場合はIrodori
latents/checkpoints、LFM adapter、optional Seed-VC persona checkpointを無効化して
再学習します。LFM export/config/contractだけが変わった場合は、上記の通りLFM
adapterだけを再生成し、既存のIrodori/Seed-VC artifactを再利用します。

## Inference modes

- `say`: text -> Irodori
- `reenact`: source audio -> Seed-VC style conversion -> target voice
- `repeat`: source audio -> ASR/SenseVoice -> Irodori
- `chat`: Core Profile + bounded history + user text -> LFM structured voice plan -> LFM-boundary normalization/recovery -> Irodori

Irodori/Seed-VCはsubprocess終了コードだけでなく生成WAVの実在と最低限のサイズも検証します。LFM structured outputはIrodoriへ渡す前にcanonical normalizationされ、malformed/empty/wrong-schema/degenerate outputは最大1回のbounded retry後にLFM境界で失敗します。fallbackが非言語イベントを発明することはありません。詳細は`docs/CORE_PROFILE_LFM_CONTRACT.md`を参照してください。

## Offline behavior

After `persona setup`, root passes `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to workers. Model snapshots are materialized under `models/`. Seed-VC keeps its upstream-downloaded checkpoints inside ignored local vendor storage.

`persona doctor --deep`はASR、diarization、SenseVoice、LFM、Seed-VCのlocal model loadに加え、Irodoriをofflineで短時間synthesisして実際のdecode pathまで検証します。さらにlockfile、setup state、vendor revision/cleanlinessをready条件に含めます。
