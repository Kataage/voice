# ASR and Prepare lineage (v0.4)

This is the operational contract for Issue [#41](https://github.com/Kataage/voice/issues/41)
and its parent [#40](https://github.com/Kataage/voice/issues/40). GitHub `main` is the
source of truth for the implementation and these pins; a local clone or an old persona
directory is never an implicit input.

## Backend and license decisions

| Contract | Pin | Status | Notes |
|---|---|---|---|
| Legacy/reference ASR | `openai/whisper-large-v3`; runtime snapshot `Systran/faster-whisper-large-v3` at `edaa852ec7e145841d8ffdb056a99866b5f0a478` | enabled | Kept for reference and explicit legacy configurations; it is not the new-persona default. |
| General modern ASR | [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) at `7278e1e70fe206f11671096ffdd38061171dd6e5` | enabled | Apache-2.0 model pin; the CLI setup default. |
| General alignment | [`Qwen/Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) at `c7cbfc2048c462b0d63a45797104fc9db3ad62b7` | enabled | Separate `alignment-v1` contract; never a transcription backend. |
| Anime/domain ASR | `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf` at `5a6a789ceb2f22d2b8606743b13a8159af218362` | disabled | The page badge is not treated as the effective license. Provenance points to `litagin/Galgame_Speech_ASR_16kHz` at `3fb86654222b3f0af0f7c332ae6a0ef9752a9451` (GPL-3.0) with commercial-use and open-source-model restrictions; no rights-holder clearance is recorded. |
| Domain CTC head | `ctc_aligner.pt` from the domain repository | disabled | Its recorded coupling is only the exact domain encoder/revision. It is not attached to the general Qwen encoder or reused as a generic aligner. |
| BGM-aware analysis | `audio-separator==0.44.2`, source `fca0cf76d52b545cedbc75e1d3aea626d513c036`, model `UVR_MDXNET_KARA_2.onnx` | optional | Wrapper and model terms are recorded separately. The model is locally registered and audited; it is not redistributed by PersonaVoice. |

The domain backend is therefore fail-closed. `persona setup --asr-backend
qwen3-asr-1.7b-ja-anime-galgame-hf` refuses before downloading or mutating a model
directory and `persona doctor` reports the reason. There is no hidden A/B arbitration.

## Immutable Prepare generations

Each new Prepare result is stored below:

```text
personas/<name>/
  generations/
    prepare/
      pl-<32 hex>/
        dataset/       # master.sqlite3, master.json, exports, quality reports
        references/    # reference bank for this lineage
        cache/         # ASR, alignment, diarization, SenseVoice, derived stems
        models/        # training candidates and published artifacts for this lineage
        outputs/
        lineage.json
    active.json
    activation-history/
```

The lineage fingerprint binds raw and identity inventories, ASR model/revision, the
independent alignment model/revision, separator policy/model audit, Prepare schema, and
the implementation/cache contract. The master fingerprint is the canonical `master.json`
content hash. A change to ASR, alignment, separation policy/model, timestamp semantics, or
clip boundaries creates a different lineage; existing generations are not overwritten.

The newest Prepare candidate is usable for validation and training, but it is not runtime
active. `persona activate` verifies the lineage record, TrainingPlan/publication identity,
quality gate, and every published artifact checksum, then replaces `generations/active.json`
under the activation lock with an atomic write. The previous pointer is copied to
`activation-history/`. Explicitly passing an older `pl-...` is the rollback operation.

## Exact target-machine workflow

Run this on the target Windows or Linux machine after checking out the desired GitHub
`main` revision:

```bash
uv sync --locked
uv run --locked persona setup --backend auto --asr-backend qwen3-asr-1.7b
uv run --locked persona init alice --authorized
# Put authorized source media in personas/alice/raw and identity references in identity/.

# Optional BGM-heavy analysis. First audit the model's actual terms and register the
# local file; registration is atomic and refuses a different already-registered model.
uv run --locked persona register-separator-model /path/UVR_MDXNET_KARA_2.onnx \
  --source-url https://audited-source.example/model \
  --model-terms 'record the exact terms accepted for this local copy'

uv run --locked persona prepare alice
uv run --locked persona train alice --executor auto
uv run --locked persona eval alice
uv run --locked persona eval-vc-manifest alice
uv run --locked persona status alice

# Inspect the machine-readable reports and candidate artifacts before this explicit step.
uv run --locked persona activate alice --lineage pl-<32-hex>
uv run --locked persona status alice
```

`prepare`, `train`, and `eval` use the candidate lineage recorded in `state.json`; they do
not change the active runtime pointer. The full family dependency is intentional: changed
transcripts, timestamps, or clip boundaries regenerate the Irodori source/latents, LFM
examples, reference bank, and VC manifests. A zero-shot VC backend does not receive
meaningless weight retraining, but its lineage-bound references and manifests are
regenerated. Failed candidate preparation/training/evaluation leaves the prior active
generation untouched.

Set `prepare.separation_policy` to `off`, `auto`, or `always`. `auto` records the
deterministic evidence used (explicit `music_heavy` metadata or a transparent filename
hint); no opaque signal is used to choose an ASR result. A selected separator creates an
analysis-only stem under the candidate cache. The canonical lossless extraction and raw
source are never replaced, and the report records both paths and model digest.

## Quality and provenance reports

`dataset/lfm_quality_report.json` records accepted/rejected counts, deterministic rejection
reasons, pathology counters, text/token distributions, the actual pinned tokenizer
chat-template token count (or an explicitly labelled test fallback), and lineage metadata.
The gate uses target-speaker evidence, audio duration, overlap/coverage, ASR evidence where
the backend emits it, fragment/pathology checks, and the real token budget. Valid short
Japanese answers and supported non-verbal-only replies remain eligible.

`dataset/irodori_quality_report.json` and every Irodori row record the UTF-8 text hash,
duration, target evidence, overlap/coverage, ASR/alignment provenance, and clip-boundary
evidence. Missing provenance, broken text/audio pairs, invalid boundaries, and excessive
overlap are rejected rather than silently sent to training. VC evaluation manifests carry
the same text hash, duration, transcript/alignment provenance, boundary evidence, and
lineage identity.

On Pascal/GTX 1080 Ti, `--backend auto` is required. Qwen runtime selection reports the
actual device, dtype, capability, fallback reason, and eager-attention choice; it does not
assume BF16 or FlashAttention. Hosted CI uses fixtures, lock resolution, and dry-runs only.
It cannot establish real ASR accuracy, Irodori acoustic quality, VRAM capacity, or VC
quality on a user's machine, so those remain explicit target-machine validation steps.
