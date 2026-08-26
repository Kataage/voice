# PersonaVoice Vevo2 worker

This project is an isolated Python 3.10 environment for the v0.4 Vevo2 FM-only
voice-conversion path. It is launched only through the root worker launcher:

```text
uv run --project workers/vevo2 --no-sync python workers/vevo2/worker.py health --request request.json
```

The worker accepts `health`, `download`, and `convert`. `download` is reachable only
from the explicit online setup path; normal `health`/`convert` calls run with the
offline environment. Before loading any model, it verifies the exact Amphion checkout,
the Vevo2 model revision marker, every declared model SHA256, the official Whisper
medium SHA256, and the setup-recorded CPU/CUDA backend.

`convert` is intentionally limited to:

```json
{
  "source": "...",
  "target": "...",
  "output_dir": "...",
  "flow_matching_steps": 32,
  "use_pitch_shift": false,
  "dtype": "fp32"
}
```

The source and target paths must be non-empty local files, and the output directory must
remain inside `PERSONAVOICE_ROOT`. A requested `fp16` CPU run fails; a CUDA setup with no
CUDA runtime also fails. The worker does not retry with another dtype/device and never
turns a failed model load into a successful-looking output.
