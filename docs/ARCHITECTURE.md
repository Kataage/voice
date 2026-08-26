# Architecture

## Design goals

1. 素材を置いて `persona build NAME` だけで学習まで到達する。
2. root環境を軽く保ち、Torch/Transformers競合をworkerごとのuv環境へ隔離する。
3. 一度作ったcanonical datasetはモデル交換後も再利用する。
4. 初回download後は通常処理をoffline modeで行う。
5. 途中停止を前提に、素材fingerprint・中間cache・upstream checkpointで再開する。
6. consent gate、local bind、secret非保存をデフォルトにする。
7. upstream revisionと全Python依存をlockし、同じcheckoutから同じ環境を再現できるようにする。

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

- Irodori: 新規personaは`method: full`。portable full artifactはmodel、tokenizer、configとprovenance/checksumを含む。`lora`はPEFT adapter、`speaker-inversion`はspeaker embeddingを生成し、LoRA/fullへ`auxiliary_speaker_inversion`を併用できる。
- LFM: 新規personaは`method: full`。TRL SFTのfull artifactはmodel/tokenizer/config、`lora`はq/k/v/output attention projectionのPEFT adapterを生成する。CPUはfp32、CUDAはbf16対応時bf16/それ以外fp16でロードする。
- Seed-VC: zero-shot V2 is default; CFM fine-tuning is opt-in because upstream is archived and FT can trade WER for similarity. Fine-tuning is blocked when the isolated Seed-VC worker cannot see CUDA.

学習はまずexecutor非依存・machine path非依存のimmutable `TrainingPlan`を作ります。dataset/model revision/family trainer hash、family methodとoptimization設定、checkpoint policyに加え、local/Modalが共有するrunner/bundle/Modal appの`executor_contract` hashを含みます。remote deploymentが古いshared runnerなら実行前に拒否します。一方、`executor_contract`、`executor`、remote consent、credential、local hardware、quality thresholdはfamily optimization fingerprintへ含めないため、runnerの安全修正やrouting/公開policyだけの変更では、family markerを新planへ再attestして実checkpointを再利用できます。同じplanのcanonical JSON bytesをlocal/Modalへ渡すため、routing変更でprepare、Irodori latent、または互換checkpointを無効化しません。quality threshold変更もoptimizer checkpointを捨てず、candidateの再評価・公開判断だけを更新します。

`executor: auto`はfull methodを要求するfamilyについて保守的なVRAM/RAM/disk/setup preflightを実行し、合格時だけlocalを選びます。不合格時はmethodをLoRAへ変更せず、persona全体の`consent.authorized: true`、remote専用の`training.remote_data_authorized: true`、Modal SDK/authがすべて揃う場合だけ同じplanをModalへroutingします。汎用consent scopeはremote専用booleanを代替しません。`local`はpreflight失敗を明示エラーにし、`modal`はremote専用許可をauth probe・bundle作成・network accessより前に確認します。

Remote bundleはallowlisted training artifactだけの最小copyです。Irodoriは事前計算済みlatentをcontent hash名へ変換した上で、許可済みmetadataのsource manifestとremote runtime用のsanitized manifestの二つを結合します。source側はtext/caption等がsanitizationでlosslessに保存されたことをchecksum付きで証明するための監査契約、sanitized側はcontent-addressedなlatent相対pathだけを指す実行契約です。両方ともraw audioと絶対local pathを含まず、bundle検証でmetadataの完全一致を照合します。LFM JSONLは`system`/`user` promptと`assistant` completionの非空message契約を再検証し、role drift、path-like/audio/secret field、Windows/UNC/POSIX absolute path、現在設定されているcredential値のメッセージへの埋め込みを拒否します。raw media、identity/reference audio、canonical SQLite、`.env`、credential、routing config、absolute path、symlink/junction traversalはbundle validationで拒否します。送信前監査はcompletion markerを含む全fileの相対path/role/SHA256/size、件数、総bytes、bundle/transfer fingerprintを最初の外部writeより前にstateへatomic保存し、completion markerを最後にremoteへpublishします。remote resultもplan/family/checkpoint/checksum/remote validation contract一致後にだけlocal candidateへ取得します。このremote検証はlocal held-out quality gateを代替せず、candidateの推論用publishは後者の合格まで行いません。Seed-VC FT audioはremote allowlist外なのでlocal-onlyです。

各familyはperiodicなcomplete checkpointだけをresume対象にし、preemption後は同じfamily fingerprintで最新の完全checkpointから続けます。Volumeのcompletion markerは合成した進捗JSONではなく、trainerが検証したmethod-native model/adapter、optimizer、scheduler、trainer/dataloader/RNG state（該当methodで必要なもの）そのものとfamily/method/dataset fingerprintを全ファイルSHA-256へ結び付け、最後にatomicで書きます。partial markerやpayload欠損はresume候補になりません。Irodori/LFM fullはvalidation lossのbest checkpointをportable candidateへ変換します。remote call ID、plan fingerprint、model、step、checkpointはsecret-free stateへatomic保存され、同じcommandの再実行は新規submitせず保存済みcallをpollします。call ID保存直前の停止はpersistent plan claimとcanonical-call redirectで重複trainerを防ぎ、checkpointを共有する異なるplanはsorted family claimで一writerに直列化します。同じFunctionCall IDのplatform retryだけはclaim所有者として通過し、通常の失敗ではplan/family claimを解放してnative checkpointから新callを再開できます。

学習成功は直ちに推論公開を意味しません。成果物はまず`models/.candidates/<family>/<family-fingerprint>/...`へ置かれ、held-out speaker similarity、CER/WER、unseen pronunciation、validation loss、duration、emotion/style、base Japanese forgettingのquality gateが完全・有限な入力で合格した場合だけpublished locationへatomic promotionされます。Irodori fullの主gateはweight-owned identityを検証するno-reference + caption/emotion経路で、speaker-conditioned / no-reference / caption-conditionedを全held-out case・同じ指標で併記します。

LFM publication gateは固定promptをcandidate full/LoRAと監査済みbase modelの両方へ`temperature: 0`のgreedy generationとして投入します。candidate/baseのJSON contractを各caseで検証したうえで、固定expected completionに対するnormalized edit similarity、CER、Japanese-aware WER、candidateのrequired-phrase coverage、baseからの最大similarity degradationを集計します。既定値はsimilarity `>= 0.35`、CER `<= 0.85`、WER `<= 1.0`、required-phrase coverage `>= 1.0`、base similarity regression `<= 0.10`です。case不足、JSON field不足、測定不能値、`NaN`、無限値はすべて公開拒否になります。学習corpus側はLFM exporter固有のuser wrapperを実dialogue行へ分解し、assistant JSONの`text`も取り出して、固定prompt/answerとのnormalized exact-utterance collisionを拒否します。これはwrapper全文比較のすり抜けを閉じつつ、より長い別発話への単なるsubstring一致をheld-out漏洩とは扱いません。

`state.json`の`trained`はverified candidate、`complete`はquality合格後のpublished状態です。固定case、metric、quality gate、deterministic LFM generationのsource digestはevaluation policyの一部なので変更後はcandidateを再評価します。ただし`FamilyPlan.fingerprint`はevaluation policyを除外するため、prepare output、Irodori latent、optimizer checkpointをこの公開判定変更だけで破棄しません。

### Invalidation boundaries

- executor、Modal設定、remote consentの変更: plan/family fingerprint、prepare cache、latent、checkpointを変更しない。
- quality thresholdの変更: family optimization fingerprint/checkpointを変更しない。gateは新しいthresholdで再評価する。
- held-out prompt、metric、publication worker contractの変更: overall TrainingPlanを変更してcandidateを再評価するが、family optimization fingerprint/checkpointは変更しない。
- Irodori `full|lora|speaker-inversion`、LFM `full|lora`、optimizer/max-step等の変更: 対応family plan/checkpointだけを変える。prepare datasetとmethod非依存Irodori latent contractは再利用する。
- raw/identity/prepare semantic/model/worker contractの変更: prepare fingerprintに依存するASR/diarization/identity/Sense/clip/datasetを再構築し、結果としてtraining input fingerprintが変わればfamily checkpointも変わる。
- base model/source revision、training implementation、training lockの変更: 対応family checkpoint/artifactを無効化するが、意味論が変わらないprepare outputを一律削除しない。
- `--force`: 完了済みstage resultを迂回してstageへ再進入する明示操作。`prepare --force`はprepare派生cacheを破棄するが、`train --force`はfamily固有の完全artifact/checkpoint再利用契約を無効化しない。通常の中断復帰には付けない。

## Inference modes

- `say`: text -> Irodori
- `reenact`: source audio -> Seed-VC style conversion -> target voice
- `repeat`: source audio -> ASR/SenseVoice -> Irodori
- `chat`: user text -> LFM structured voice plan -> Irodori

公開済みIrodori full checkpointでは、`reference_mode: auto`は外部referenceを自動付加せず`--no-ref`を選びます。明示的なaudio referenceまたは`reference_mode: audio|speaker-embed`は引き続き利用でき、LoRA/Speaker Inversionの既存conditioning規則も維持します。

Irodori/Seed-VCはsubprocess終了コードだけでなく生成WAVの実在と最低限のサイズも検証します。LFM structured outputがJSONとして壊れた場合はplain-text + neutral voice metadataへ安全にフォールバックします。

## Offline behavior

After `persona setup`, root passes `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to workers. Model snapshots are materialized under `models/`. Seed-VC keeps its upstream-downloaded checkpoints inside ignored local vendor storage.

`persona doctor --deep`はASR、diarization、SenseVoice、LFM、Seed-VCのlocal model loadに加え、Irodoriをofflineで短時間synthesisして実際のdecode pathまで検証します。さらにlockfile、setup state、vendor revision/cleanlinessをready条件に含めます。
