# v0.3 target-machine procedure

これはrelease/0.3を実persona audio/GPUへ適用するための手順です。GitHubのrelease/0.3とこの文書を唯一の実装根拠にし、過去Work session、別clone、既存local persona artifactを入力にしません。

Hosted CIはfixture/fake/lock/dry-runだけを実行します。実personaのASR精度、再学習成功、Irodori音響品質、Seed-VC試聴品質、VRAM余裕はtarget machineでこの手順を完了するまで未検証です。

## 1. Setup

```bash
git fetch origin
git switch release/0.3
git pull --ff-only origin release/0.3
uv sync --locked
uv run --locked persona setup --backend auto --asr-backend large-v3
uv run --locked persona doctor --deep
```

`setup`だけが監査済みmodel assetをmaterializeします。通常のPrepare/inferenceはoffline local assetを使い、production-scale modelをCIでdownloadしません。Hugging Face gated assetは、同じaccountで条件を受諾し、read tokenを現shellの`HF_TOKEN`へ設定してからsetupします。tokenはstate/logへ保存しません。

Qwenを使う場合は、setupでQwenとQwen ForcedAlignerを同じ監査revisionでmaterializeします。

```bash
uv run --locked persona setup --backend auto --asr-backend qwen3-asr-1.7b
```

その後、対象personaの`persona.yaml`で次を明示してからPrepareします。

```yaml
prepare:
  asr_model: qwen3-asr-1.7b
  asr_device: cpu       # target GPUを実測して許可するまでの安全な例
  asr_dtype: fp32
```

`jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`とdomain CTCはlicense/provenance policyによりfail-closedです。Apache表示だけでenableせず、コードやconfigを編集してpolicyを弱めないでください。

GTX 1080 Ti/PascalではBF16、FlashAttention、Qwenのimplicit CUDA fallbackを前提にしません。`asr_device`/`asr_dtype`は`cpu`/`fp32`などtargetで確認できる値を明示し、CUDAを要求するなら実際に`doctor --deep`とworker healthが通ることを確認します。unsupported dtype/deviceをautoで黙って置き換えることはありません。Irodori/worker backendも`setup`の実kernel preflight結果と一致させます。

## 2. Create a candidate Prepare lineage

```bash
uv run --locked persona init alice --authorized
# authorized source mediaを personas/alice/raw/ へ配置
# clean target-speaker identity clipsを personas/alice/identity/ へ配置
uv run --locked persona consent alice --authorized
uv run --locked persona prepare alice
uv run --locked persona status alice --verify
```

Prepare resultに `lineage_id: pl-...`、ASR/alignment/separation contract、master fingerprint、quality reportが出ることを確認します。separationを使う場合は、review済みの正確なUVR weightだけを登録します。

```bash
uv run --locked persona register-separator-model UVR_MDXNET_KARA_2.onnx \
  --source-url <reviewed source URL> \
  --model-terms <model-specific terms recorded by review>
uv run --locked persona prepare alice
```

canonical/raw audioはそのまま保持し、separated stemはanalysis-onlyであることを`dataset/analysis_audio.json`とquality reportで確認します。Prepare途中失敗時は同じcommandを`--force`なしで再実行し、valid checkpointを再利用します。force rerunでもfresh candidate lineageが選ばれ、active generationへ書き込みません。

## 3. Rebuild the dependent v0.3 family

新ASR lineageや新masterからの移行では、LFMだけを再学習して完了扱いにしません。

```bash
uv run --locked persona train alice
```

この順序でcandidate generationを作ります。

1. Irodori source/pair manifest、DACVAE latent、Speaker Inversion/LoRAを新Prepare lineageから構築。
2. LFM conversational exportを実際のpinned chat-template token budgetとquality gateで確認し、LFM LoRAを構築。
3. Seed-VC fine-tuneを明示的に有効化している場合は、新しいtarget clipsから依存checkpointを構築。zero-shot/reference-onlyの場合はreference/manifest markerだけを新lineageへ作成。
4. `gen-.../generation.json`へfamily marker、artifact digest、lineage/master provenanceを保存。

同じPrepare/master lineageでLFM export/filter policyだけを変えた場合だけ、次を使えます。

```bash
uv run --locked persona export-lfm alice
uv run --locked persona train alice
```

この狭い経路ではIrodori/Seed-VCを再学習せずLFMだけをcandidateへ再生成します。ASR/alignment/separation/masterが変わった場合はこの経路を使わず、Prepareからやり直します。

## 4. Validate, then activate

```bash
uv run --locked persona validate alice
uv run --locked persona status alice --verify
uv run --locked persona activate alice
uv run --locked persona status alice
```

validationはaccepted/rejected countsとreason/pathology distribution、target speaker evidence、duration/overlap/coverage、backend confidence、token count source、transcript/alignment provenanceを確認します。CIがgreenでもtarget-machineの音響品質をverifiedとは記録しません。実音声を使ったlistening/ASR/VRAM確認は別のtarget acceptanceとして記録します。

activation後だけ `say`、`reenact`、`repeat`、`chat`、APIがそのgenerationをruntimeへ解決します。activation前のcandidateは評価・再検証対象であり、runtimeへ自動公開されません。

## 5. Rollback

```bash
uv run --locked persona status alice
# active_lineage_id / active_generation_idを確認
uv run --locked persona activate alice \
  --lineage-id pl-<known-good lineage> \
  --generation-id gen-<known-good generation>
uv run --locked persona status alice --verify
```

rollback先は削除せず残っているvalidated candidateでなければなりません。activationはmanifest、Prepare lineage、全family artifactの存在・size・SHA256・validation passedを再検証し、失敗した場合は旧active pointerを維持します。前のpointerは`generations/activation-history/`に保存されます。

## Target acceptance boundary

target machineで次を実行して初めて、実GPU/device/dtype/model-loadが確認済みになります。

```bash
uv run --locked persona setup --backend auto --asr-backend <selected backend>
uv run --locked persona doctor --deep
uv run --locked persona prepare alice
uv run --locked persona train alice
uv run --locked persona validate alice
uv run --locked persona activate alice
uv run --locked persona say alice "短い日本語応答"
uv run --locked persona reenact alice authorized_input.wav
```

Work/hosted CIだけで上記のASR accuracy、Irodori/Seed-VC acoustic quality、VRAM、試聴結果を検証済みと主張しないでください。