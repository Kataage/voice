from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src/personavoice/cli.py"
text = path.read_text(encoding="utf-8")

import_anchor = "from personavoice.setup_env import download_models, install_environments\n"
if text.count(import_anchor) != 1:
    raise RuntimeError("setup_env import anchor not found exactly once")
text = text.replace(
    import_anchor,
    import_anchor + "from personavoice.setup_lock import SetupLockError, setup_lock\n",
    1,
)

start = text.find("@app.command()\ndef setup(\n")
end = text.find("\n\n@app.command(\"init\")", start)
if start < 0 or end < 0:
    raise RuntimeError("setup command boundaries not found")

replacement = '''@app.command()\ndef setup(\n    backend: str = typer.Option("auto", help="Irodori backend: auto/cu126/cu128/cpu/rocm/xpu"),\n    download: bool = typer.Option(True, "--download-models/--skip-models"),\n    skip_seed_vc_models: bool = typer.Option(False),\n    verify: bool = typer.Option(True, "--verify/--no-verify"),\n) -> None:\n    """Install pinned local uv environments and model snapshots."""\n    backend = backend.strip().lower()\n    if backend not in SETUP_BACKENDS:\n        raise typer.BadParameter(\n            f"Unsupported Irodori backend {backend!r}; choose one of "\n            f"{', '.join(sorted(SETUP_BACKENDS))}."\n        )\n    root = find_repo_root()\n    try:\n        # Serialize the complete setup transaction: environment sync, model\n        # materialization/repair, verification, and final state publication.\n        # The OS lock is crash-safe and is released automatically on process exit.\n        with setup_lock(root):\n            result = install_environments(root, backend=None if backend == "auto" else backend)\n            if download:\n                result["models"] = _download_models_or_explain(\n                    root,\n                    include_seed_vc=not skip_seed_vc_models,\n                )\n            if verify:\n                verification = doctor_report(\n                    root,\n                    deep=True,\n                    require_seed_vc=not skip_seed_vc_models,\n                )\n                if download and not verification["ready_offline"]:\n                    repaired = repair_failed_model_materializations(\n                        root,\n                        verification,\n                        include_seed_vc=not skip_seed_vc_models,\n                    )\n                    if repaired:\n                        result["model_recovery"] = {\n                            "discarded_materializations": repaired,\n                            "download": _download_models_or_explain(\n                                root,\n                                include_seed_vc=not skip_seed_vc_models,\n                            ),\n                        }\n                        verification = doctor_report(\n                            root,\n                            deep=True,\n                            require_seed_vc=not skip_seed_vc_models,\n                        )\n                result["verification"] = verification\n                if not verification["ready_offline"]:\n                    _print(result)\n                    raise typer.Exit(1)\n            _print(result)\n    except SetupLockError as exc:\n        console.print(f"[bold red]{exc}[/bold red]")\n        raise typer.Exit(2) from None\n'''

path.write_text(text[:start] + replacement + text[end:], encoding="utf-8", newline="\n")
print("setup-session locking applied")
