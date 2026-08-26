# Vevo2 FM-only VC

PersonaVoice v0.4 adds Vevo2 as an explicit, selectable voice-conversion backend while
keeping Seed-VC v2 as the default. The initial contract is deliberately narrow:

- source audio + one target reference audio → Vevo2 flow-matching VC;
- style/prosody and non-verbal audio are supplied by the source audio;
- the upstream FM-only path is used; AR+FM, TTS, SVS, editing, and Vevo2 fine-tuning are
  outside this v0.4 integration;
- ordinary inference and `doctor --deep` are offline; only explicit setup/materialization
  may use the network.

## Pinned provenance

The source and released weights are separate contracts. The machine-readable contract is
[`config/vevo2_assets.json`](../config/vevo2_assets.json), and the constants used by setup,
doctor, and the isolated worker are in [`src/personavoice/model_assets.py`](../src/personavoice/model_assets.py).

| Item | Pin | License |
|---|---|---|
| Amphion/Vevo2 source | `open-mmlab/Amphion` commit `26f6883110181f1dbfe95c70a7c7dbaf4de5f42a` | MIT ([source license](https://github.com/open-mmlab/Amphion/blob/main/LICENSE)) |
| Vevo2 model snapshot | `RMSnow/Vevo2` revision `2674843cbaa50aa89ee7ccaf5bb15d6ccf46c6c8` | CC BY-NC-ND 4.0 ([model card](https://huggingface.co/RMSnow/Vevo2), [license](https://creativecommons.org/licenses/by-nc-nd/4.0/)) |
| Whisper dependency | OpenAI Whisper `medium.pt`, official SHA256-pinned URL | MIT ([source license](https://github.com/openai/whisper/blob/main/LICENSE)) |

The Vevo2 model license is not a commercial-use grant. Users must review the model card,
license, and any upstream component terms for their intended use. The source-code MIT
license is never used to infer rights for the released model weights.

The FM-only model snapshot is restricted to these inference-critical files, each with an
exact SHA256 in the JSON contract:

- content/style tokenizer: `tokenizer/contentstyle_fvq16384_12.5hz/model.safetensors`;
- flow-matching acoustic model, config, and Whisper statistics;
- vocoder config and its three safetensors files;
- official OpenAI Whisper medium weights.

No floating branch, mutable `main`, unverified mirror, or implicit remote model lookup is
accepted by the worker. Missing, empty, stale, or checksum-mismatched files fail closed.

## Setup and offline behavior

`persona setup` is the only normal PersonaVoice command that materializes Vevo2 weights.
It downloads the exact Hugging Face snapshot and official Whisper file, verifies every
checksum, syncs `workers/vevo2` from its committed `uv.lock`, verifies the pinned Amphion
checkout, then loads the local FM graph with network access disabled before publishing
`.runtime/vevo2-models-ready`.

```bash
# Online, explicit materialization. HF_TOKEN is needed only if another gated setup asset
# requires it; it is never stored in the Vevo2 contract or setup report.
uv run --locked persona setup --backend auto

# Deliberately skip the heavy Vevo2 model view while preparing another machine.
uv run --locked persona setup --backend auto --skip-vevo2-models

# No network is allowed by this verification path.
uv run --locked persona doctor --deep
```

The root environment and every worker are independent `uv` projects. Vevo2 uses Python
3.10 and its own Torch 2.4/cu124-or-CPU dependency set, separate from the main Torch
workers and from Seed-VC. The setup record says which backend was selected; the worker
does not switch CPU/CUDA or dtype implicitly. `fp32` is the explicit default. `fp16` is
accepted only when the setup record selects CUDA and the target machine has been tested;
unsupported layer/device errors propagate rather than triggering an fp32 retry.

Adding the Vevo2 worker does not delete or rewrite `Prepare`, ASR, diarization, SenseVoice,
Irodori, LFM, Seed-VC, optimizer, or training checkpoint artifacts. Re-running setup
reuses verified existing model views and caches. It may make an old environment contract
require a setup resync so that the new worker is audited, but it does not invalidate the
expensive data or training artifacts by itself.

## Selecting the backend

The persona config remains conservative:

```yaml
vc_backend: seed-vc-v2
inference:
  vevo2_flow_matching_steps: 32
  vevo2_use_pitch_shift: false
  vevo2_dtype: fp32
```

Use an explicit CLI/API override for an experiment, or change `vc_backend` only after a
completed decision gate has been reviewed:

```bash
uv run --locked persona reenact alice acting.wav --backend seed-vc-v2
uv run --locked persona reenact alice acting.wav --backend vevo2-fm
uv run --locked persona repeat alice acting.wav --backend vevo2-fm
```

`POST /v1/voice-convert` and `POST /v1/repeat` accept the same optional `backend` values:
`seed-vc-v2` and `vevo2-fm`. The Web UI exposes the same selector. `reenact()` is the
runtime abstraction used by CLI, API, UI, and the evaluation runner; Seed-VC's existing
path remains intact.

## Canonical Japanese/non-verbal A/B evaluation

The runner consumes the existing prepared `dataset/master.json`; it does not run Prepare
again and does not mutate source clips. It creates one immutable JSONL manifest containing
the same source path, target reference path, source/reference SHA256, Japanese text,
event labels, and one of these buckets:

- `normal_speech`
- `mixed_speech_event`
- `nonverbal_only`

Rare event buckets are allocated first with a deterministic seed. The requested range is
100–300 clips; a smaller prepared corpus is allowed for a dry run but is explicitly
underpowered and cannot pass the default gate. If `master.json`, target clips, or a
reference bank is absent, manifest creation stops with a pending-validation message; no
synthetic sample is created.

```bash
uv run --locked persona eval-vc-manifest alice \
  --output personas/alice/dataset/vc_evaluation_manifest.jsonl \
  --limit 200 --seed 20260827

uv run --locked persona eval-vc alice \
  --manifest personas/alice/dataset/vc_evaluation_manifest.jsonl
```

For every manifest row the runner invokes both backends with the identical source and
reference. Outputs are placed in separate report directories and are hashed. A backend
failure, missing ASR, missing speaker embedding, unreadable output, or missing event
metric remains visible in the per-sample JSON rather than disappearing from an average.

The machine-readable `report.json`, human-readable `report.md`, and generated
`human_review.json` are written below `personas/<name>/outputs/vc-evaluation/<run>/`.
The report includes manifest/output hashes, backend/runtime metadata, counts by bucket,
per-sample errors, and the exact commands used to reproduce the run. Human review can be
completed by copying the generated form, setting its `status` to `complete`, recording a
reviewer, and rerunning `eval-vc --human-review ...`; the report never assumes listening
was performed merely because audio files were generated.

The gate compares Vevo2 with Seed-VC on:

- Japanese CER (WER is secondary);
- target speaker similarity;
- duration ratio and F0/prosody correlation;
- voiced/unvoiced F1, pause ratio, and speech-rate/timing;
- overall non-verbal preservation, laughter, breath, mixed speech-event preservation,
  and non-verbal-only success;
- matched human listening of intelligibility, identity, prosody/timing, pauses, and events.

The gate requires full metric coverage, all three buckets, no failed samples, at least 100
samples, and human review when configured. It permits only the explicit tolerances in
`vc_evaluation` and requires at least one clear improvement. A pending or failed gate
always recommends `seed-vc-v2` and sets `default_changed: false`; the runner never edits
`persona.yaml` automatically.

## GTX 1080 Ti / Pascal 11 GB validation

This is a target-machine procedure, not a claim that hosted CI or the current Work
environment has this GPU. Do not record a pass until the commands below have actually
completed on the target machine.

1. Install the supported NVIDIA driver and confirm the physical device selected as logical
   CUDA device 0:

   ```bash
   nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version --format=csv
   echo "$CUDA_VISIBLE_DEVICES"
   ```

2. From the GitHub checkout, run setup with the device visible and let `--backend auto`
   select the audited Pascal-compatible main stack. Setup records UUID, compute
   capability, driver, and real FP32/FP16 CUDA kernel preflight results:

   ```bash
   uv run --locked persona setup --backend auto
   uv run --locked persona doctor --deep
   ```

3. Confirm `.runtime/setup.json` contains `worker_backends.vevo2` as `cu124`, the selected
   GPU is the intended UUID, and deep health loaded the local worker. Do not infer Vevo2
   VRAM fit from the dependency preflight; model loading is the first real memory test.

4. Use an authorized prepared persona and run one short FM-only sample in explicit FP32:

   ```bash
   uv run --locked persona reenact alice \
     personas/alice/dataset/clips/<short-clip>.flac \
     --ref personas/alice/references/<target-reference>.flac \
     --backend vevo2-fm
   ```

   Record peak VRAM, wall time, output validity, dtype, and whether CUDA was used. Repeat
   with a representative speech clip and each required non-verbal bucket before any A/B
   conclusion. Only after an explicit measured test should `vevo2_dtype: fp16` be tried;
   an OOM, unsupported kernel, or dtype error is a failed validation, not permission to
   silently retry with another dtype/device.

5. After the single-sample smoke succeeds, build the canonical manifest and run the full
   A/B procedure. Copy the generated report and completed human review into the persona's
   local evidence area as appropriate; do not commit raw audio or private reports to the
   public repository.

GPU replacement, driver change, or a `CUDA_VISIBLE_DEVICES` change that selects another
physical device requires setup/preflight again. No silent CPU fallback is permitted for a
setup explicitly recorded as `cu124`.
