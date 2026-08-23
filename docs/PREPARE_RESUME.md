# Prepare crash-resume contract

`persona prepare` / `persona build` の ASR・話者分離・identity embedding・SenseVoice batch は、モデルを1回だけロードしたまま複数項目を順番に処理します。

各項目の成功結果は `personas/<name>/cache/<kind>/.checkpoints/` へ atomic JSON として保存されます。workerが異常終了した場合、次回の同一fingerprint実行ではroot側の中央semantic validatorを通過したcheckpointだけを正式cacheへ昇格し、完了済み項目の再計算を避けます。truncated JSON、ID不一致、schema/result不正は再利用せず削除します。

`uv run --locked persona status <name>` の `audit.prepare.batch_progress` には、実行中batchのworker/phase、`completed` / `total`、現在item、失敗数、checkpoint済み成功数が表示されます。ASRではさらにruntime選択・model load・transcribeのphase、実際に選ばれた`device` / `compute_type`、現在source内で処理済みの音声秒数とsegment数を表示します。progress metadataは観測用途だけで、実際にstageが生きているかどうかはOS stage lockがsource-of-truthです。

ASRはfaster-whisperがsegmentを生成している間もatomic progress heartbeatを更新します。root supervisorはPersonaVoice自身が生成したASR checkpoint directoryだけを監視し、runtime選択・model load・transcribeを含めて20分間heartbeatが一度も前進しない場合、native CUDA/CTranslate2等の停止とみなしてworker process treeを終了します。Windowsでは`uv -> python worker`の子孫を含めて終了し、POSIXでは専用process group全体を終了するため、親CLIだけ終了してGPU workerが孤児化する状態を避けます。

watchdogによる終了はprepareの失敗として明示されますが、すでに成功してatomic保存されたitem checkpointは残ります。同じ`persona prepare <name>`または`persona build <name>`を**`--force`なし**で再実行すると、それらを中央semantic validatorで再検証して再利用します。watchdogはsegment-level heartbeatを持つASR batchだけに適用し、1 item内部の前進を安全に観測できない他workerへ時間ベースの推測を広げません。

`raw/`・`identity/`・prepare設定・固定model/worker contractが変わった場合、または`--force`を指定した場合はcheckpointを含むprepare派生cacheを無効化します。

Prepare cache policyのソース契約はUTF-8テキストの改行をLFへ正規化してからhashするため、同一checkoutのLF/CRLF差だけでWindowsとLinuxのcache policyが変わりません。互換migrationは認識結果の意味論を変えない運用・復旧変更についても、CIで実測した正確なcanonical policy世代だけを明示的な前身として許可します。未知の旧policyや将来の別実装世代を動的に許可せず、意味論が変わったcacheは引き続きfail-closedです。
