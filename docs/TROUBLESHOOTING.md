# Troubleshooting

## `persona setup` fails on pyannote

Accept the conditions for `pyannote/speaker-diarization-community-1` on Hugging Face, create a read token, set it only in your shell (`HF_TOKEN`), then rerun. Token values are never written by PersonaVoice.

## OOM during Irodori training

The generated training config uses a conservative VRAM profile and gradient checkpointing on smaller GPUs. If a previous run failed, rerun the same command; upstream checkpoint resume is used. You can lower `training.irodori_max_steps` for test runs, but batch size should normally remain auto-managed.

## Multiple speakers detected and target cannot be selected

Place clean, target-only speech in `personas/NAME/identity/`. Do not lower the identity threshold first; improve the identity clips first.

## Seed-VC problems

Seed-VC is an archived upstream with an older dependency stack. It is deliberately isolated under `workers/seed_vc`. Delete only `workers/seed_vc/.venv` and rerun `persona setup` to rebuild that environment without touching the rest of PersonaVoice.

## Regenerate everything

`persona build NAME --force` recomputes stages. Raw source files are never deleted.
