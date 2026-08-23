# GPU provenance guard audit

PersonaVoice treats CUDA readiness as an executable runtime contract, not merely a successful package import or a reported compute capability.

- A setup records GPU UUID, compute capability, and NVIDIA driver version only after real CUDA preflight.
- Direct runtime must revalidate all three pieces of provenance before authorizing recorded CUDA environments.
- PyTorch-backed CUDA workers execute real tensor/matrix kernels during setup preflight, which catches missing or unusable CUDA native libraries before model work.
- ASR is a separate CTranslate2 runtime. CTranslate2 requires CUDA 12 cuBLAS and cuDNN for GPU speech recognition, but its isolated wheel does not provide the complete Windows runtime. PersonaVoice therefore exposes the cuBLAS/cuDNN libraries from the locked diarization PyTorch environment created in the same setup transaction, rather than depending on an arbitrary system CUDA installation.
- ASR setup preflight loads every required Windows cuBLAS/cuDNN DLL (or the corresponding Linux shared libraries) and creates/destroys real cuBLAS and cuDNN handles before a CUDA compute type is accepted. On Windows the probe does not rely on `PATH` for `ctypes`: Python 3.8+ uses a secure DLL loader policy, so the setup-provided runtime directories are registered with `os.add_dll_directory` and every audited DLL is opened by absolute path. A missing or non-loadable native runtime fails setup before model downloads or persona preparation.
- ASR worker launch fails closed when CUDA is authorized but that audited runtime provider is incomplete. CPU/ROCm/XPU setups continue to hide NVIDIA devices from ASR and do not require CUDA libraries.
- Captured Python subprocesses are forced to UTF-8 output because the parent process decodes worker JSON as UTF-8. This prevents Windows code-page settings from corrupting Japanese ASR/model responses.
- Environment-contract schema 6 fingerprints both `cuda_preflight.py` and `process.py`, so changes to native-runtime validation, DLL search policy, or subprocess encoding invalidate stale setup state before model work.
- Model revisions, recognition parameters, and persona preparation semantics are unchanged. Existing valid prepare caches/checkpoints remain semantically reusable.
