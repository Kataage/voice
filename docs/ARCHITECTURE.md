# Architecture

## Design principles

1. **Local first.** Persona data, derived clips, checkpoints, caches, logs, and model outputs stay local by default.
2. **One canonical dataset.** Source media is parsed once into a model-neutral master dataset. Backend-specific manifests are generated from it.
3. **Replaceable workers.** Orchestration code does not import heavy model stacks directly. Each model family gets its own isolated uv project and process boundary.
4. **Resumable stages.** Every expensive stage is content-addressed where practical and records status/provenance in persona state.
5. **Explicit consent state.** Training/inference entry points will refuse destructive or publish-like flows when authorization metadata is incomplete.
6. **No silent quality loss.** Raw sources are never discarded. Rejected/low-confidence segments remain traceable through metadata.

## Repository layers

```text
src/personavoice/        lightweight orchestration and CLI
workers/                 isolated model environments
personas/<name>/         local persona data and artifacts
schemas/                  canonical interchange schemas (planned)
docs/                     architecture and operating contracts
```

## Worker boundary

A worker is an independent uv project:

```text
workers/irodori/
  pyproject.toml
  uv.lock
  .venv/
  worker.py
```

The orchestrator launches workers as subprocesses and communicates over newline-delimited JSON on stdin/stdout initially. This keeps setup simple and avoids managing ports. Long-running UI/API mode may later promote hot workers to local HTTP/IPC while preserving the same request/response schema.

### Common request envelope

```json
{
  "request_id": "uuid",
  "operation": "synthesize",
  "persona": "alice",
  "payload": {},
  "options": {}
}
```

### Common response envelope

```json
{
  "request_id": "uuid",
  "ok": true,
  "result": {},
  "error": null,
  "metrics": {}
}
```

Workers must write machine-readable protocol messages to stdout and human logs to stderr/files.

## Planned workers

- `media`: ffmpeg/ffprobe normalization and media inventory
- `vad`: speech/non-speech segmentation
- `diarization`: pyannote-based speaker segmentation
- `speaker`: target-speaker verification against `identity/`
- `asr`: Japanese transcription and timestamps
- `separation`: optional vocal/BGM/source separation
- `prosody`: pitch/energy/rate/acoustic descriptors
- `caption`: emotion/style/non-verbal annotations
- `irodori`: TTS, reference-guided synthesis, speaker inversion, LoRA training
- `seed_vc`: speech-to-speech reenactment / voice conversion
- `lfm`: persona-response and delivery-plan LoRA/inference
- `evaluation`: speaker similarity, intelligibility, quality and regression suite

Workers are enabled by capability discovery; missing optional workers must not make unrelated features unusable.

## Canonical preparation pipeline

```text
source inventory
  -> media probe/hash
  -> audio extraction/normalization
  -> optional source separation
  -> VAD
  -> diarization
  -> target-speaker verification
  -> overlap/music/clipping/quality analysis
  -> ASR + timestamps
  -> segment reconstruction
  -> prosody extraction
  -> emotion/style/non-verbal captioning
  -> canonical dataset
  -> train/validation/test split
  -> reference-bank selection
  -> backend manifest adapters
```

Every derived segment keeps source file hash, original time range, processing versions, and confidence values.

## Canonical dataset fields

The exact schema will be versioned, but it must cover:

- stable segment ID
- source path + content hash
- start/end timestamps
- target-speaker probability
- raw and normalized transcript
- ASR confidence
- overlap/music/noise/clipping flags
- signal-quality metrics
- structured emotion/style/prosody values
- free-form natural-language delivery caption
- non-verbal events with temporal positions
- split assignment
- derived audio path + derivation metadata

Backend-specific tokens/emoji must **not** be stored as canonical labels. Adapters translate neutral events into backend-specific controls.

## Inference routing

- text -> Irodori TTS
- text + style/caption -> Irodori VoiceDesign-style conditioning
- source audio + preserve performance -> Seed-VC reenactment
- source audio + re-perform as persona -> ASR/caption -> Irodori
- mixed verbal/non-verbal source -> hybrid router (planned)
- conversation -> LFM persona planner -> TTS

## Training plan

`persona train <name>` will eventually generate an automatic plan from usable data statistics. Initial target order:

1. Irodori speaker inversion
2. Irodori persona/style LoRA
3. LFM persona/delivery-plan LoRA
4. optional Seed-VC speaker fine-tune
5. regression evaluation
6. best-checkpoint selection
7. voicepack manifest generation

Each phase is resumable and independently rerunnable.

## Local storage policy

Large/raw/generated artifacts are gitignored. Git should contain code, schemas, configs, tests, and small deterministic fixtures only. Model downloads should use explicit local cache directories under the repository or a user-configured local path; worker code must not scatter caches across the OS by default.
