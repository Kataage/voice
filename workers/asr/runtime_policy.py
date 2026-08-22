from __future__ import annotations

CUDA_AUTO_PREFERENCE = ("float16", "int8_float32", "float32")
CPU_AUTO_PREFERENCE = ("int8", "int8_float32", "float32")


def choose_compute_type(
    device: str,
    supported: set[str],
    requested: str = "auto",
) -> str:
    """Choose a compute type that the detected CTranslate2 backend actually supports.

    In particular, Pascal GPUs such as the GTX 1080 Ti can expose a CUDA device
    while not supporting efficient float16 computation. Blindly selecting
    float16 from CUDA visibility alone makes faster-whisper fail during model
    construction. Auto mode therefore consults CTranslate2's runtime capability
    set and picks the fastest audited preference that is genuinely available.
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
