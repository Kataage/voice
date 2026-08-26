# Issue #42 — v0.3 ASR / Prepare lineage contract

この文書はGitHub Issue #40、Issue #42本文、最新のexecution plan、release/0.3へmerge済みの#41/#37/#34 contractをv0.3へ適用した正式仕様です。対象はrelease/0.3だけです。

## Scope and boundary

- v0.3はpre-full-fine-tuning baselineであり、LFM LoRA、Irodori Speaker Inversion/LoRA、Seed-VC、既存のflat v0.3 training entrypointを維持する。
- Core Profile、canonical `{text, voice: {caption, emotion, events}}`、LFM malformed/empty/degenerate output boundary、Irodori boundary diagnostics/runtime controlsは既存contractを再利用する。
- full-model TrainingPlan/Executor/publication architectureやVevo2 backportはこのreleaseへ入れない。
- `personas/<name>/dataset`, `models`, `outputs`, `cache`の既存root artifactはhistorical baseline/rollback generationとして保持する。

## ASR backend registry

| logical backend | pinned identity | v0.3 policy |
| --- | --- | --- |
| legacy/reference | `openai/whisper-large-v3` | enabled; current worker uses the audited faster-whisper large-v3 snapshot and word timestamps |
| general modern | `Qwen/Qwen3-ASR-1.7B` | enabled; exact revision, local files, explicit device/dtype, no fabricated Whisper confidence |
| anime/character domain | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` | disabled/fail-closed pending license and provenance approval |

legacyは論理参照名と実行artifact identityを分けて記録します。Qwenはworkerが出したconfidenceだけを保存し、Qwen結果からWhisperの`avg_logprob`やprobabilityを推定しません。

domain backendを無効化する理由は、HFページにApache-2.0と表示されることではなく、固定した学習datasetとその制限を含むeffective provenanceです。現在のauditはdataset `litagin/Galgame_Speech_ASR_16kHz` / GPL-3.0、commercial-use prohibition、models-trained-on-dataset-must-be-open-sourcedを記録します。したがってconfig/setup/worker/Prepareの全入口でdomainを拒否します。

## Alignment contract

`alignment-v1`はASRから独立したversioned contractです。

- legacy Whisper: worker native word timestamps (`whisper-native-words`)。
- general Qwen: `Qwen/Qwen3-ForcedAligner-0.6B` の固定revision。
- domain: `domain-ctc-aligner` としてdomain encoder ID/revisionとのcouplingを表現するが、domain ASRがdisabledなので現在は実行不可。

domain CTC head (`ctc_aligner.pt`) はdomain encoder専用です。general Qwen encoder、Whisper、または別revisionへ流用する経路を作らず、ASR backend・model ID・model revision・alignment revision・transcript hashが一致しないcacheは拒否します。

## BGM-aware separation

separationは `audio-separator==0.44.2`、固定source revision、固定UVR lineage、human-reviewed local model manifestを要求します。terms/source URL/model digestがない場合、policy `auto`/`always`であってもPrepareを失敗させます。`off`ならcanonical audioだけを使います。

出力stemはlineage-local cacheのanalysis-only artifactです。`raw/`、lossless canonical extraction、masterのcanonical audio pathを破壊的に置換せず、ASR/alignment provenanceに`canonical_audio_path`、`analysis_audio_path`、separator decision/model digestを保存します。

## Prepare lineage and failure boundary

Prepare fingerprintはraw/identity inventory、config、ASR/alignment/separation contracts、worker lock、前処理コードを含みます。semantic changeは新しい `pl-<32 lowercase hex>` candidate lineageになります。`--force`もactive lineageを再利用せずfresh candidate nonceを付けます。

candidate lineageのファイルは次の下に作ります。

```text
personas/<name>/generations/prepare/pl-.../
  dataset/master.sqlite3
  dataset/{irodori_source,lfm_train,seed_vc}.*
  dataset/*_quality_report.json
  references/
  cache/
  lineage.json
```

失敗時はcandidate lineageの途中ファイルだけが残り得ますが、`generations/active.json`は変更しません。旧lineage、root historical artifact、旧active generationは削除・上書きしません。

## Quality gates and provenance

IrodoriとLFMの両方にmachine-readable reportを出します。最低文字数だけで採否を決めません。

- target speaker evidence: identity similarity、speaker coverage、target flag。
- overlap / fragmentation / pathology: overlap ratio、boundary、duration、text/audio fragmentation、replacement character、empty-text pathology。
- audio/duration: canonical/analysis path、positive duration、non-empty target clip。
- backend confidence: ASRが実際に出したsegment/word/language confidenceをbackend-specific fieldとして保存し、意味がないbackendには値を作らない。
- transcript/alignment provenance: backend、model/revision、transcript hash、alignment hash、boundary evidence。
- token budget: pinned LFM tokenizerの`apply_chat_template`でprompt+completionを実測し、fallback時は`conservative_character_estimate`をreportする。
- short Japanese answers and event-only nonverbal responses: 正常な短文・イベントを最低文字数だけで捨てず、audio/evidence/provenanceが有効なら保持する。

reportにはcandidate/accepted/rejected count、reason distribution、pathology counters、text/token distribution、token-count source、lineage/master fingerprintを残します。Irodori training-pair manifestとcontractにも同じPrepare lineage、master、source/transcript/alignment provenanceを保存します。

## Dependent-family migration

ASR/alignment/separation semantics、master、references、derived datasetが変わったときは次の順序を守ります。

```text
new Prepare lineage
  -> new Irodori Speaker Inversion/LoRA generation
  -> new LFM LoRA generation
  -> Seed-VC fine-tune or new reference/manifest dependency
  -> validation (quality/provenance/digest)
  -> explicit atomic activation
```

zero-shot/reference-only Seed-VCは無意味なSeed-VC model retrainingをせず、new lineageのreference bankとmanifest/markerを再生成します。これはIrodori/LFMを新しいASR-derived teacherへ移行する必要性を取り消しません。

同一master/Prepare lineageのままLFM export/filter/chat-template policyだけが変わる場合のみ、Irodori/Seed-VCを再学習せずLFM-only regeneration/retrainingを許可します。この例外を新ASR lineage migrationへ適用しません。

## Validation, activation, rollback

`persona train`はcandidateを作りますがruntimeを切り替えません。`persona validate`はfamily artifact、quality gate、Irodori pair provenance、Seed-VC dependency、manifest digestを確認します。

```bash
uv run --locked persona train alice
uv run --locked persona validate alice
uv run --locked persona activate alice
```

activationは次を再検証してから、唯一のmutable pointer `generations/active.json`をatomic replaceします。

1. Prepare lineage recordとcandidate manifestのlineage/master identity。
2. generation ID/fingerprintとv0.3-pre-full-fine-tuning architecture。
3. Irodori/LFM/Seed-VC各familyのcomplete/not_requested statusと全artifact size/SHA256。
4. validation passed。

旧pointerは`generations/activation-history/`へ保存されます。rollbackも同じactivation verifierを通す明示操作です。

```bash
uv run --locked persona activate alice \
  --lineage-id pl-<old lineage> \
  --generation-id gen-<old generation>
```

candidate artifactの改変、manifest mismatch、validation未完了、lineage mismatchはactivationを拒否し、現在のactive pointerを維持します。