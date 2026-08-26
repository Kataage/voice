# Prepare crash-resume contract

`persona prepare` / `persona build` の ASR・話者分離・identity embedding・SenseVoice batch は、モデルを1回だけロードしたまま複数項目を順番に処理します。

各項目の成功結果は `personas/<name>/cache/<kind>/.checkpoints/` へ atomic JSON として保存されます。workerが異常終了した場合、次回の同一fingerprint実行ではroot側の中央semantic validatorを通過したcheckpointだけを正式cacheへ昇格し、完了済み項目の再計算を避けます。truncated JSON、ID不一致、schema/result不正は再利用せず削除します。

`uv run --locked persona status <name>` の `audit.prepare.batch_progress` には、実行中batchのworker/phase、`completed` / `total`、現在item、失敗数、checkpoint済み成功数が表示されます。ASRではさらにruntime選択・model load・transcribeのphase、実際に選ばれた`device` / `compute_type`、現在source内で処理済みの音声秒数とsegment数を表示します。progress metadataは観測用途だけで、実際にstageが生きているかどうかはOS stage lockがsource-of-truthです。

ASRはfaster-whisperがsegmentを生成している間もatomic progress heartbeatを更新します。root supervisorはPersonaVoice自身が生成したASR checkpoint directoryだけを監視し、runtime選択・model load・transcribeを含めて20分間heartbeatが一度も前進しない場合、native CUDA/CTranslate2等の停止とみなしてworker process treeを終了します。Windowsでは`uv -> python worker`の子孫を含めて終了し、POSIXでは専用process group全体を終了するため、親CLIだけ終了してGPU workerが孤児化する状態を避けます。

watchdogによる終了はprepareの失敗として明示されますが、すでに成功してatomic保存されたitem checkpointは残ります。同じ`persona prepare <name>`または`persona build <name>`を**`--force`なし**で再実行すると、それらを中央semantic validatorで再検証して再利用します。watchdogはsegment-level heartbeatを持つASR batchだけに適用し、1 item内部の前進を安全に観測できない他workerへ時間ベースの推測を広げません。

`raw/`・`identity/`・prepare設定・固定model/worker contractが変わった場合、または`--force`を指定した場合はcheckpointを含むprepare派生cacheを無効化します。

Prepare cache policyのソース契約はUTF-8テキストの改行をLFへ正規化してからhashするため、同一checkoutのLF/CRLF差だけでWindowsとLinuxのcache policyが変わりません。互換migrationは認識結果の意味論を変えない運用・復旧変更についても、CIで実測した正確なcanonical policy世代だけを明示的な前身として許可します。未知の旧policyや将来の別実装世代を動的に許可せず、意味論が変わったcacheは引き続きfail-closedです。

## Training / Modal preemption resume

Trainingはprepare stageとは独立した`TrainingPlan`とfamily fingerprintを持ちます。Irodori/LFM full、LoRA、Speaker Inversion、optional Seed-VCは各family固有のrun directoryへperiodic checkpointを保存し、model/optimizer/scheduler/stepとcompletion markerが揃ったcheckpointだけを再開候補にします。Irodori full/LoRAはfilenameまたはdirectoryのstepをnative trainer step、WSD/cosine schedulerの`last_step`、single-process dataloader/runtime stateへ結び付けます。Speaker Inversionはpinned Irodori runtimeの`safetensors.safe_open`でupstream固有のempty metadata、単一の`speaker_embedding` float32 tensor、非空2次元shapeを確認し、壊れた新しいembeddingがあれば古い検証済みembeddingへfallbackします。LFMはworker内のrestricted load後にmethod/step/precisionと全native file checksumをattestし、FP16時だけ`scaler.pt`を必須にします。partial、truncated、改ざん、別plan、検証不能checkpointは使わず、選択から外すだけで既存checkpointを自動削除・書換えしません。

通常のPC再起動・preemption・CLI中断では、同じ`persona train <name>`または`persona build <name>`を`--force`なしで再実行してください。localは最新の完全checkpointから続けます。Modalは`state.json`へ保存済みのsecret-free call ID、plan fingerprint、family contractsを読み、新しいjob/bundleを作らず同じFunctionCallをpollします。`spawn()`受理後かつcall ID保存前に停止した場合も、再送callはpersistent plan claimが選んだcanonical call IDへredirectされ、二つのtrainerは走りません。異なるplanが同じfamily fingerprint/checkpointを再利用する場合はfamily claimで一writerに直列化されます。remote側もfamily fingerprint namespaceのcomplete checkpointだけをresumeし、result completion markerと全checksumが揃うまでlocal candidateとして確定しません。推論用artifactへのpublishは、その後のlocal held-out quality gate合格時だけです。

`persona status <name>`では次を区別できます。

- `audit.prepare.batch_progress`: ASR/diarization/Sense等のitem checkpoint進捗。
- `audit.train.operation.remote_call_id`, `remote_state`, `step`, `checkpoint`: local/Modal training進捗。
- `plan_fingerprint`: executor間で共通のsemantic plan。
- `candidate_complete`: 学習成果物とchecksumは揃っているが、まだ公開前の場合を含む。
- `published_complete`: local held-out quality gate合格後に推論用artifactへ昇格済み。

## Invalidation matrix

| Change | Prepare cache / dataset | Irodori latent | Family checkpoint |
|---|---|---|---|
| `executor` local/Modal/auto、Modal app/GPU、remote consent | reuse | reuse | reuse（同じplan） |
| quality threshold | reuse | reuse | reuseして再評価 |
| Irodori/LFM methodまたはoptimizer設定 | reuse | method非依存contractが同じならreuse | 対応familyだけ新fingerprint |
| raw/identity/prepare semantics、ASR/diarization/Sense contract | dependent cacheをinvalidate | 新datasetなら再生成 | 新training inputなら新fingerprint |
| training base/source/lock/implementation | reuse | latent contract変更時だけ再生成 | 対応familyをinvalidate |
| `prepare --force` | prepare派生cache/checkpointを明示破棄 | dataset再生成後に必要なら再生成 | input fingerprint次第 |
| `train --force` | reuse | valid contractはreuse | stageへ再進入するが、family固有の完全artifact/checkpoint契約はreuse可 |

Routingだけの変更やv0.3→schema v2の意味保存migrationを理由に、正常なprepare/cache/latent/checkpointを破棄してはいけません。逆にmethod、base revision、dataset bytesが違うcheckpointを名前だけでresumeしてはいけません。
