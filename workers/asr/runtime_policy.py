from __future__ import annotations

# Modern CUDA devices that genuinely support float16 keep the fast path. When
# float16 is unavailable (for example Pascal / sm_61), prefer native float32
# over quantized int8_float32. The GTX 1080 Ti acceptance run showed that
# CTranslate2 4.8.1 could advertise int8_float32 as supported yet make no
# forward progress during large-v3 transcription. Explicit int8_float32 remains
# available for users who intentionally request it, but auto mode must favor the
# execution path that has the strongest architecture-level correctness margin.
CUDA_AUTO_PREFERENCE = ("float16", "float32", "int8_float32")
CPU_AUTO_PREFERENCE = ("int8", "int8_float32", "float32")


def choose_compute_type(
    device: str,
    supported: set[str],
    requested: str = "auto",
) -> str:
    """Choose a compute type that the detected CTranslate2 backend actually supports.

    CUDA visibility alone is insufficient: older architectures can expose a
    device while lacking a reliable float16 execution path. Auto mode therefore
    intersects the audited preference order with CTranslate2's runtime-reported
    capabilities. On CUDA devices without float16, float32 is intentionally
    preferred over int8_float32 because the Pascal acceptance path demonstrated
    a zero-progress native stall with the latter despite capability reporting.
    """

    normalized = {str(value) for value in supported}
    if requested != "auto":
        if requested not in normalized:
            available = ", ".join(sorted(normalized)) or "none"
            raise ValueError(
                f"Requested CTranslate2 compute type {requested!r} is not supported on "
                f"{device}; available: {available}"
            )
        return requested

    preferences = CUDA_AUTO_PREFERENCE if device == "cuda" else CPU_AUTO_PREFERENCE
    for value in preferences:
        if value in normalized:
            return value
    available = ", ".join(sorted(normalized)) or "none"
    raise RuntimeError(
        f"No audited automatic CTranslate2 compute type is supported on {device}; "
        f"available: {available}"
    )
