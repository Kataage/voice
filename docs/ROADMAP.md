# Roadmap

The project is built in narrow vertical slices. Each phase must leave the CLI usable and keep previous artifacts forward-migratable.

## Phase 0 — foundation

- [x] repository bootstrap
- [x] uv-managed lightweight core project
- [x] stable CLI command names
- [x] persona directory/config/state contract
- [x] worker isolation architecture
- [ ] generated `uv.lock`
- [ ] local bootstrap scripts for Windows/Linux
- [ ] CI for core-only lint/tests

## Phase 1 — media inventory and preparation core

- [ ] ffmpeg/ffprobe doctor details
- [ ] recursive source inventory
- [ ] content hashing and media metadata cache
- [ ] deterministic audio extraction/normalization
- [ ] resumable stage engine
- [ ] canonical SQLite/Parquet metadata store
- [ ] preparation report

## Phase 2 — target speaker extraction

- [ ] VAD worker
- [ ] pyannote diarization worker
- [ ] speaker embedding/verification worker
- [ ] identity reference enrollment
- [ ] overlap and contamination detection
- [ ] automatic quality grades A/B/C/D

## Phase 3 — transcription and expressive annotations

- [ ] Japanese ASR worker
- [ ] timestamp-aware transcript normalization
- [ ] prosody metrics
- [ ] emotion/style caption worker
- [ ] non-verbal event taxonomy + temporal annotations
- [ ] model-neutral canonical segment schema v1

## Phase 4 — Irodori voice backend

- [ ] v4.1 inference worker
- [ ] automatic reference bank selection
- [ ] reference-conditioned TTS
- [ ] style/caption controls
- [ ] speaker inversion training
- [ ] LoRA training
- [ ] checkpoint resume
- [ ] candidate generation/ranking

## Phase 5 — speech reenactment / VC

- [ ] Seed-VC worker
- [ ] zero-shot target reference mode
- [ ] optional speaker fine-tune
- [ ] `persona reenact`
- [ ] `persona repeat`
- [ ] hybrid verbal/non-verbal router
- [ ] real-time worker prototype

## Phase 6 — persona brain

- [ ] LFM2.5-JP worker
- [ ] conversation-pair extraction
- [ ] persona SFT dataset adapter
- [ ] delivery-plan JSON output schema
- [ ] LoRA training and evaluation
- [ ] knowledge/profile separation from style LoRA

## Phase 7 — product surface

- [ ] local REST API
- [ ] browser UI
- [ ] batch generation
- [ ] chat mode
- [ ] microphone / virtual-device integration hooks
- [ ] voicepack export/import

## Phase 8 — automated evaluation and hardening

- [ ] held-out evaluation suite
- [ ] speaker similarity metrics
- [ ] ASR intelligibility/CER
- [ ] acoustic-quality metrics
- [ ] expressive coverage matrix
- [ ] automatic best-checkpoint selection
- [ ] deterministic regression fixtures
- [ ] migration/versioning policy
- [ ] backup/export integrity checks

## Non-goals for early phases

- Cloud-required orchestration
- Uploading source voice data by default
- Hiding backend failures behind fabricated success
- Coupling canonical data to one TTS vendor/model
