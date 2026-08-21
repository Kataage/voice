# PersonaVoice

Local-first toolkit for building consented, dedicated Japanese voice personas from video/audio material.

## Goals

- Put source media in one folder and keep manual preprocessing to a minimum.
- Keep Python environments local to the repository and managed with `uv`.
- Separate orchestration from model workers so Torch/Transformers dependency conflicts do not poison the whole project.
- Build one canonical dataset, then derive Irodori, LFM, VC, ASR/evaluation data from it.
- Support resumable preparation/training and explicit provenance for every derived clip/model.
- Keep inference simple: text-to-speech, style/emotion control, reference-guided synthesis, speech reenactment, repeat/re-synthesis, chat, API, and UI.

## Current status

Foundation only. Heavy model workers are intentionally not wired yet. The CLI and project contracts are being established first so later integrations remain replaceable.

## Requirements

- Windows 10/11 or Linux
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- FFmpeg on `PATH` (media preparation will require it)
- NVIDIA GPU is recommended for model workers

## Bootstrap

```powershell
uv sync
uv run persona doctor
uv run persona init alice
```

The core environment lives in the repository-local `.venv` created by `uv`. Model workers will each have their own `pyproject.toml`, `.venv`, and `uv.lock` under `workers/` when added.

## Persona layout

```text
personas/
  alice/
    raw/          # videos/audio supplied by the user
    identity/     # clean target-speaker references
    dataset/      # generated canonical/derived datasets
    references/   # generated reference banks
    models/       # local fine-tunes/embeddings
    outputs/      # generated audio and reports
    cache/        # resumable intermediate artifacts
    logs/         # persona-specific logs
    state.json    # stage state and provenance
    persona.yaml  # persona configuration
```

Only `raw/`, `identity/`, and `persona.yaml` are intended for routine manual editing.

## Planned command surface

```text
persona doctor
persona init <name>
persona status <name>
persona prepare <name>
persona train <name>
persona say <name> <text>
persona reenact <name> <audio>
persona repeat <name> <audio>
persona chat <name>
persona ui <name>
persona serve [name]
```

See `docs/ARCHITECTURE.md` and `docs/ROADMAP.md` for the implementation contract.

## Safety and consent

This project is intended for voices whose use has been authorized by the speaker. Keep consent scope and source provenance with each persona. Do not publish or deploy a trained voice outside the granted scope.
