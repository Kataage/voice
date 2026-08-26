# Irodori boundary diagnostics (v0.3 backport)

PersonaVoice v0.3 includes the inference-only part of the Issue #33 contract
needed by Issue #34. It is a compatibility backport into the existing v0.3
architecture; it does not add the v0.4 TrainingPlan/Executor or publication
architecture.

## Runtime controls

`persona.yaml` may explicitly set:

```yaml
inference:
  duration_scale: 1.0
  trim_tail: true
  tail_window_size: 20
  tail_std_threshold: 0.05
  tail_mean_threshold: 0.1
```

The values are passed to the pinned Irodori inference command instead of being
left to accidental upstream defaults. `persona say` and `POST /v1/tts` accept
one-shot `duration_scale` and `trim_tail` overrides. These settings affect only
generation: they are excluded from Prepare and training fingerprints, so an
existing v0.3.0/v0.3.1 persona can use them without retraining.

`reference_mode` also accepts `none` for the leading-artifact isolation test.
The default `auto` behavior remains unchanged: a trained speaker embedding is
preferred, then the prepared audio reference bank, then no reference.

## Diagnostic matrix

Run this on a machine that has the persona, pinned Irodori checkout, models and
configured local environments:

```bash
uv run --locked persona diagnose-boundaries alice --no-asr --no-sense
```

For the four duration/tail rows, text, checkpoint, reference selection and
seed are held constant:

| Variant | Duration scale | Tail trim |
| --- | ---: | --- |
| A | 1.00 | on |
| B | 1.00 | off |
| C | candidate margin (default 1.10) | on |
| D | candidate margin (default 1.10) | off |

The margin is an evaluated candidate only. The command never edits
`persona.yaml` or selects a new default automatically. Use
`--margin-scale <value>` to test another positive candidate.

The same report also compares a fixed onset sentence across three seeds and
the conditions available on that persona:

- persona vs base-only;
- reference vs no reference;
- caption conditioning on vs off;
- audio reference and speaker embedding only when the artifacts exist.

Each successful row records the requested text, seed, method, checkpoint,
reference mode and content fingerprint, effective duration/tail settings,
upstream duration evidence when printed, output duration, output SHA-256,
first/last voiced frame, final energy envelope, and optional ASR/SenseVoice
evidence. A final-token/character suffix check is used for speech preservation;
adding silence after a missing phoneme is not considered a fix.

Reports are written below:

```text
personas/<name>/outputs/boundary_diagnostics/<timestamp>/report.json
```

The report includes the pinned Irodori source revision, diagnostic/runtime
contract, and a read-only snapshot of the existing Prepare/train stage
fingerprints. The diagnostic output is outside the training pipeline and does
not rewrite raw media, canonical SQLite/JSON, prepared clips, references,
Irodori latents, LFM artifacts, Seed-VC artifacts, or existing checkpoints.

ASR and SenseVoice are optional report enrichments. Use
`--no-asr --no-sense` when only the deterministic generation and WAV boundary
evidence is wanted. Hosted CI does not download models or claim an acoustic
quality result; this Work run did not execute the real-persona command.

## Interpretation policy

The report is evidence, not an automatic training decision:

- B better than A with the same duration suggests tail trimming is implicated;
- C/D better than A/B with retained final-token evidence suggests duration
  under-prediction is implicated;
- no safe A-D candidate means keep the configured policy and investigate the
  upstream/model behavior further;
- a stable onset difference across seeds and conditioning points toward the
  checkpoint or training boundary;
- a seed-dependent difference points toward sampling;
- a reference- or caption-dependent difference points toward conditioning.

No Irodori training-audio sanitation is included in this backport because
#33's evidence does not establish that v0.3 leading artifacts are caused by
transcript-unmatched training clips. If a future real-persona evaluation does
establish that cause, sanitation must be a deterministic, versioned, explicit
Irodori-only derived view. Canonical Prepare outputs remain the fallback and
must not be blindly fixed-trimmed; Japanese initial consonants, breaths and
intentional non-verbal events must be preserved.

## Target-machine smoke/listening gate

On Windows PowerShell or Linux bash, from the repository root:

```text
uv run --locked persona status alice --verify
uv run --locked persona diagnose-boundaries alice --no-asr --no-sense
```

Listen to A-D for short, long, sentence-final, fast/emotional, punctuation,
breath and laughter samples. Keep playback and loudness identical. Inspect the
first 250 ms and final 500 ms separately, and record:

- whether any pre-speech vocalization is intelligible, intentional or unwanted;
- whether the final mora/phoneme is actually spoken, not merely followed by
  silence;
- Japanese intelligibility/final-token retention;
- speaker identity, emotion, prosody, breath and laughter retention;
- whether a duration margin introduces repetition or unstable tail material.

If ASR/CER and SenseVoice evidence are required, repeat explicitly with
`--asr --sense`; these are model-worker operations and were not run in hosted
CI. A passing CI result therefore verifies contracts, deterministic metadata,
cross-platform path/CLI behavior and artifact-preservation rules, not the
subjective quality of a missing real persona.
