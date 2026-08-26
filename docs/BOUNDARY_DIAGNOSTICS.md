# Irodori boundary diagnostics

PersonaVoice v0.4 exposes an inference-only diagnosis for Issue #33:

```bash
uv run --locked persona diagnose-boundaries alice
```

The command evaluates the same text, checkpoint, reference selection, and
explicit seed under the following deterministic matrix:

| Variant | Duration scale | Tail trim |
| --- | ---: | --- |
| A | 1.00 | on |
| B | 1.00 | off |
| C | 1.10 | on |
| D | 1.10 | off |

The 1.10 value is an evaluated margin candidate only. It is never written back
to `persona.yaml`, and the report cannot silently select a new runtime policy.
Use `--margin-scale` to evaluate another calibrated candidate.

Each generated row records the requested text, seed, selected checkpoint and
method, effective reference fingerprint, duration settings, upstream duration
message when available, output duration, output SHA-256, final energy envelope,
first/last voiced frame, ASR transcript/CER/WER, final-token preservation, and
SenseVoice event evidence. ASR and SenseVoice failures are retained in the
report instead of deleting the successful synthesis evidence.

The command also runs a leading-artifact isolation matrix on one fixed onset
probe with three fixed seeds. It compares persona/base-only, reference/no-
reference, and caption-on/off paths. Optional audio-reference and speaker-
embedding rows are added only when those artifacts exist. A seed-dependent
onset difference is evidence for sampling; a reference- or caption-dependent
difference is evidence for conditioning; a stable difference is evidence that
the checkpoint or its training boundary deserves investigation. These are
interpretation hints in the report, not automatic training-data decisions.

Reports are written to:

```text
personas/<name>/outputs/boundary_diagnostics/<timestamp>/report.json
```

This path is outside the prepare and training stages. Running the diagnostic
does not rerun ASR/diarization Prepare, modify raw media, modify canonical
SQLite/JSON, invalidate Irodori latents, invalidate LFM data/checkpoints, or
delete existing personas. The explicit inference settings are likewise not
part of prepare or training fingerprints, so a runtime policy experiment can
be performed against already-trained personas.

## Target-machine listening gate

After the deterministic report is complete, listen to A-D for every category
in the report's evaluation set, with loudness matched and the same playback
device. Check the first 250 ms and the final 500 ms separately. Record:

- whether any pre-speech vocalization is intelligible, intentional, or unwanted;
- whether the final mora/phoneme is actually spoken, not merely followed by
  silence;
- speaker identity, emotion, breath/laughter retention, and prosody;
- whether a longer duration introduces unstable or repeated tail material.

Only after representative listening and the CER/final-token evidence agree
should `inference.duration_scale` or `inference.trim_tail` be changed. If the
leading artifact remains stable across the isolation matrix, investigate the
checkpoint/data boundary separately. Do not apply a fixed trim or
`silenceremove`; valid Japanese initial consonants, breaths, and intentional
non-verbal events must remain available for that investigation.
