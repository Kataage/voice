# Worker environments

Heavy model integrations live here as independent uv projects.

Each worker must own its own:

```text
workers/<name>/
  pyproject.toml
  uv.lock
  .python-version
  worker.py
  README.md
```

The worker `.venv/` stays local and is never committed.

Do **not** add Irodori, pyannote, Whisper, Seed-VC, LFM, CUDA-specific Torch, or similar heavyweight dependencies to the repository root `pyproject.toml`. The root environment is intentionally lightweight and only orchestrates local worker processes.

Initial communication uses newline-delimited JSON over stdin/stdout. stdout is protocol-only; diagnostics belong on stderr or under the persona `logs/` directory.

Planned worker names are documented in `docs/ARCHITECTURE.md`.
