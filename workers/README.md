# Worker environments

Each worker is an independent uv project with its own `.venv` and `uv.lock`.
Do not merge heavy ML dependencies into the root PersonaVoice environment.

The root orchestrator communicates with workers through JSON request/response files.
Use `persona setup` to synchronize all environments and materialize model files.
