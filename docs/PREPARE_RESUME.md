# Prepare crash-resume contract

`persona prepare` / `persona build` の ASR・話者分離・identity embedding・SenseVoice batch は、モデルを1回だけロードしたまま複数項目を順番に処理します。

各項目の成功結果は `personas/<name>/cache/<kind>/.checkpoints/` へ atomic JSON として保存されます。workerが異常終了した場合、次回の同一fingerprint実行ではroot側の中央semantic validatorを通過したcheckpointだけを正式cacheへ昇格し、完了済み項目の再計算を避けます。truncated JSON、ID不一致、schema/result不正は再利用せず削除します。

`uv run --locked persona status <name>` の `audit.prepare.batch_progress` には、実行中batchのworker/phase、`completed` / `total`、現在item、失敗数、checkpoint済み成功数が表示されます。progress metadataは観測用途だけで、実際にstageが生きているかどうかはOS stage lockがsource-of-truthです。

`raw/`・`identity/`・prepare設定・固定model/worker contractが変わった場合、または`--force`を指定した場合はcheckpointを含むprepare派生cacheを無効化します。

Prepare cache policyのソース契約はUTF-8テキストの改行をLFへ正規化してからhashするため、同一checkoutのLF/CRLF差だけでWindowsとLinuxのcache policyが変わりません。今回のmigrationは直前mainで実測したWindows/Linuxの2つのpolicy値だけを新しいcanonical policyの前身として許可し、それ以前・別実装世代のcacheは引き続きfail-closedです。
