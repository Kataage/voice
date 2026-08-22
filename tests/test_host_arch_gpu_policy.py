from __future__ import annotations

from personavoice import hardware


def _gpu(capability: str) -> hardware.GpuInfo:
    return hardware.GpuInfo(
        index=0,
        name="NVIDIA test GPU",
        total_mib=16384,
        free_mib=12000,
        compute_capability=capability,
    )


def test_unknown_host_architecture_does_not_reuse_x86_cuda_matrix(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "ppc64le")
    assert hardware.cuda_backend_for_gpu(_gpu("8.6")) == "cpu"
    assert not hardware.backend_supports_gpu("cu126", _gpu("8.6"))
    assert not hardware.backend_supports_gpu("cu128", _gpu("8.6"))


def test_aarch64_cuda_fails_closed_until_full_stack_is_ci_audited(monkeypatch):
    monkeypatch.setattr(hardware.platform, "machine", lambda: "aarch64")
    assert hardware.cuda_backend_for_gpu(_gpu("9.0")) == "cpu"
    assert not hardware.backend_supports_gpu("cu126", _gpu("9.0"))
    assert not hardware.backend_supports_gpu("cu128", _gpu("9.0"))
    assert not hardware.seed_vc_cuda_supported(_gpu("9.0"))
