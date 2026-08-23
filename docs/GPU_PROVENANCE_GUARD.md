# GPU provenance guard audit

This follow-up audit hardens direct CUDA runtime authorization after the architecture-safe GPU work.

- A setup records GPU UUID, compute capability, and NVIDIA driver version only after real CUDA preflight.
- Direct runtime must revalidate all three pieces of provenance before authorizing recorded CUDA environments.
- `environment_contract.py`, which implements the authorization guard itself, is fingerprinted by the environment contract so future guard changes invalidate older setup state.
- Dependency versions and model assets are unchanged by this follow-up.
