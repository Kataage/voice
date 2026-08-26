from __future__ import annotations

import ipaddress
import json
from pathlib import Path

import typer
from huggingface_hub.errors import GatedRepoError
from rich.console import Console

from personavoice.boundary_diagnostics import run_boundary_diagnostics
from personavoice.config import PersonaConfig
from personavoice.dataset import export_lfm, load_lfm_tokenizer
from personavoice.doctor import report as doctor_report
from personavoice.environment import load_root_environment
from personavoice.evaluation import evaluate
from personavoice.inference import VC_BACKENDS, chat_turn, synthesize
from personavoice.inference import reenact as reenact_audio
from personavoice.inference import repeat as repeat_audio
from personavoice.lfm_contract import LFM_CONTRACT_FINGERPRINT, LFM_CONTRACT_SCHEMA_VERSION
from personavoice.lineage import (
    DomainBackendDisabledError,
    activate_generation,
    active_lineage_id,
    resolve_backend,
)
from personavoice.pipeline import prepare_persona
from personavoice.profile import load_core_profile
from personavoice.project import find_repo_root, get_persona, init_persona
from personavoice.repair import repair_failed_model_materializations
from personavoice.separation import register_separator_model
from personavoice.setup_env import download_models, install_environments
from personavoice.setup_lock import SetupLockError, setup_lock
from personavoice.stage_lock import StageLockError
from personavoice.state import StateStore
from personavoice.status import persona_status
from personavoice.training import train_persona
from personavoice.vc_evaluation import (
    build_vc_evaluation_manifest,
    evaluate_vc,
)

app = typer.Typer(no_args_is_help=True, help="PersonaVoice local-first voice persona toolkit")
console = Console()
SETUP_BACKENDS = {"auto", "cu126", "cu128", "cpu", "rocm", "xpu"}
TRAINING_EXECUTORS = {"auto", "local", "modal"}


def _repo_root() -> Path:
    root = find_repo_root()
    # The loader has an explicit allowlist, never replaces inherited values,
    # and returns only key names. CLI commands deliberately discard that report
    # so credentials cannot enter normal command output.
    load_root_environment(root)
    return root


def _load(name: str):
    root = _repo_root()
    paths = get_persona(root, name)
    return root, paths, PersonaConfig.load(paths.config)


def _print(value) -> None:
    console.print_json(data=value)


def _existing_file(path: Path) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise typer.BadParameter(f"File does not exist: {value}")
    return value


def _executor_override(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized not in TRAINING_EXECUTORS:
        raise typer.BadParameter(
            f"Unsupported training executor {value!r}; choose one of "
            f"{', '.join(sorted(TRAINING_EXECUTORS))}."
        )
    return normalized


def _is_loopback_host(host: str) -> bool:
    value = host.strip().lower()
    if value in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _download_models_or_explain(
    root: Path,
    *,
    include_seed_vc: bool,
    include_vevo2: bool = True,
    asr_backend: str | None = None,
) -> dict:
    try:
        kwargs = {"include_seed_vc": include_seed_vc}
        if asr_backend is not None:
            kwargs["asr_backends"] = (asr_backend,)
        # Keep the helper compatible with integrations that monkeypatch the
        # pre-v0.4 two-argument downloader; only an explicit skip needs the
        # new keyword because the production default is Vevo2-inclusive.
        if include_vevo2:
            return download_models(root, **kwargs)
        return download_models(root, include_vevo2=False, **kwargs)
    except GatedRepoError as exc:
        repo_id = getattr(exc, "repo_id", None) or "pyannote/speaker-diarization-community-1"
        console.print(f"[bold red]Hugging Face access was denied for {repo_id}.[/bold red]")
        console.print(
            "Open the model page while signed in to the same Hugging Face account, "
            "accept/request its access conditions, then use a token with read access."
        )
        console.print(f"Model page: https://huggingface.co/{repo_id}")
        console.print("Token page: https://huggingface.co/settings/tokens")
        console.print(
            "Set HF_TOKEN in the current shell and rerun `uv run --locked persona setup`. "
            "PersonaVoice never prints or stores the token."
        )
        raise typer.Exit(2) from None


@app.command()
def doctor(
    deep: bool = typer.Option(False, help="Load local models and verify offline readiness."),
) -> None:
    root = _repo_root()
    if deep:
        with console.status(
            "[bold cyan]Deep offline verification is running...[/bold cyan] "
            "[dim](model loading can take several minutes)[/dim]",
            spinner="dots",
        ):
            result = doctor_report(root, deep=True, require_vevo2=True)
    else:
        result = doctor_report(root, deep=False, require_vevo2=True)
    _print(result)
    if not result["commands_ok"] or (deep and not result["ready_offline"]):
        raise typer.Exit(1)


@app.command()
def setup(
    backend: str = typer.Option("auto", help="Irodori backend: auto/cu126/cu128/cpu/rocm/xpu"),
    asr_backend: str = typer.Option(
        "qwen3-asr-1.7b",
        "--asr-backend",
        help="ASR backend: qwen3-asr-1.7b or legacy whisper-large-v3.",
    ),
    download: bool = typer.Option(True, "--download-models/--skip-models"),
    skip_seed_vc_models: bool = typer.Option(False),
    skip_vevo2_models: bool = typer.Option(False),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Install pinned local uv environments and model snapshots."""
    backend = backend.strip().lower()
    if backend not in SETUP_BACKENDS:
        raise typer.BadParameter(
            f"Unsupported Irodori backend {backend!r}; choose one of "
            f"{', '.join(sorted(SETUP_BACKENDS))}."
        )
    try:
        asr_backend = resolve_backend(asr_backend).key
    except (ValueError, DomainBackendDisabledError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--asr-backend") from None
    root = _repo_root()
    console.print("[bold]PersonaVoice setup[/bold]")
    console.print(
        "[dim]Environment sync, downloads, and SHA256 verification are mostly network/disk/CPU "
        "work. Low GPU usage during those phases is normal. Animated status means the current "
        "phase is still running.[/dim]"
    )
    try:
        # Serialize the complete setup transaction: environment sync, model
        # materialization/repair, verification, and final state publication.
        # The OS lock is crash-safe and is released automatically on process exit.
        with setup_lock(root):
            console.print("[bold cyan]1/3[/bold cyan] Synchronizing audited environments...")
            result = install_environments(
                root,
                backend=None if backend == "auto" else backend,
                asr_backend=asr_backend,
            )
            selected_backend = result.get("irodori_backend", backend)
            console.print(
                f"[green]✓[/green] Environment synchronization complete "
                f"(Irodori backend: [bold]{selected_backend}[/bold])."
            )
            if download:
                with console.status(
                    "[bold cyan]2/3 Downloading/verifying pinned model assets...[/bold cyan] "
                    "[dim](low GPU usage is normal)[/dim]",
                    spinner="dots",
                ):
                    result["models"] = _download_models_or_explain(
                        root,
                        include_seed_vc=not skip_seed_vc_models,
                        include_vevo2=not skip_vevo2_models,
                        asr_backend=asr_backend,
                    )
                console.print("[green]✓[/green] Pinned model assets are materialized and verified.")
            else:
                console.print("[yellow]2/3[/yellow] Model download skipped by request.")
            if verify:
                with console.status(
                    "[bold cyan]3/3 Loading models for deep offline verification...[/bold cyan] "
                    "[dim](GPU usage rises only for workers executing CUDA kernels)[/dim]",
                    spinner="dots",
                ):
                    verification = doctor_report(
                        root,
                        deep=True,
                        require_seed_vc=not skip_seed_vc_models,
                        require_vevo2=not skip_vevo2_models,
                    )
                if download and not verification["ready_offline"]:
                    repaired = repair_failed_model_materializations(
                        root,
                        verification,
                        include_seed_vc=not skip_seed_vc_models,
                        include_vevo2=not skip_vevo2_models,
                    )
                    if repaired:
                        console.print(
                            "[yellow]Deep verification found repairable model materializations; "
                            "rebuilding only the affected assets.[/yellow]"
                        )
                        with console.status(
                            "[bold cyan]Repairing and re-verifying affected model assets...[/bold cyan]",
                            spinner="dots",
                        ):
                            result["model_recovery"] = {
                                "discarded_materializations": repaired,
                                "download": _download_models_or_explain(
                                    root,
                                    include_seed_vc=not skip_seed_vc_models,
                                    include_vevo2=not skip_vevo2_models,
                                    asr_backend=asr_backend,
                                ),
                            }
                            verification = doctor_report(
                                root,
                                deep=True,
                                require_seed_vc=not skip_seed_vc_models,
                                require_vevo2=not skip_vevo2_models,
                            )
                result["verification"] = verification
                if not verification["ready_offline"]:
                    _print(result)
                    raise typer.Exit(1)
                console.print("[green]✓[/green] Deep offline verification complete.")
            else:
                console.print("[yellow]3/3[/yellow] Deep verification skipped by request.")
            _print(result)
    except SetupLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None


@app.command("init")
def init_command(
    name: str,
    authorized: bool = typer.Option(
        False,
        "--authorized",
        help="Record that local voice use is authorized.",
    ),
) -> None:
    paths = init_persona(_repo_root(), name, authorized=authorized)
    console.print(f"Created: [bold]{paths.root}[/bold]")
    console.print(
        f"Put videos/audio in {paths.raw} and clean target-speaker clips in {paths.identity}."
    )


@app.command()
def consent(
    name: str,
    authorized: bool = typer.Option(True, "--authorized/--not-authorized"),
) -> None:
    _, paths, cfg = _load(name)
    if cfg.was_migrated:
        raise typer.BadParameter(
            "This persona still uses the v0.3 training schema. Run `persona migrate-config "
            f"{name} --dry-run` and then `persona migrate-config {name}` before changing consent; "
            "PersonaVoice will not save a schema migration implicitly."
        )
    cfg.consent.authorized = authorized
    cfg.save(paths.config)
    console.print(f"consent.authorized = {authorized}")


@app.command("migrate-config")
def migrate_config(
    name: str,
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and report the migration without writing persona.yaml.",
    ),
) -> None:
    """Explicitly rewrite one legacy v0.3 training config as schema version 2."""

    _, paths, cfg = _load(name)
    migrated = cfg.was_migrated
    written = False
    if migrated and not dry_run:
        cfg.save_migrated(paths.config)
        written = True
    _print(
        {
            "persona": name,
            "source_schema_version": cfg.training.migrated_from_schema_version,
            "target_schema_version": cfg.training.schema_version,
            "migration_required": migrated,
            "written": written,
            "dry_run": dry_run,
            "notes": list(cfg.migration_notes),
        }
    )


@app.command()
def status(
    name: str,
    verify: bool = typer.Option(
        False,
        "--verify",
        help="Re-hash current inputs and datasets to detect stale prepare/train fingerprints.",
    ),
) -> None:
    root, paths, cfg = _load(name)
    _print(persona_status(root, paths, cfg, verify_inputs=verify))


@app.command()
def activate(
    name: str,
    lineage: str | None = typer.Option(
        None,
        "--lineage",
        help="Prepare lineage id to activate; omit to use the current trained candidate.",
    ),
) -> None:
    """Atomically make one fully validated Prepare/model generation active."""

    _, paths, _ = _load(name)
    store = StateStore(paths.state)
    train_stage = store.stage("train")
    train_result = train_stage.get("result") if isinstance(train_stage, dict) else None
    selected = lineage
    if selected is None and isinstance(train_result, dict):
        raw_lineage = train_result.get("lineage_id")
        selected = raw_lineage if isinstance(raw_lineage, str) else None
    if selected is None:
        raise typer.BadParameter(
            "No lineage-bound trained candidate is available; run prepare, train, and eval first, "
            "then pass --lineage pl-... explicitly."
        )
    plan_fingerprint = (
        train_result.get("plan_fingerprint")
        if isinstance(train_result, dict)
        else None
    )
    # An explicit older lineage is a rollback/reference operation.  Do not
    # compare it with the currently recorded train plan; activate_generation
    # will independently verify that lineage's own publication contract.
    trained_lineage = (
        train_result.get("lineage_id") if isinstance(train_result, dict) else None
    )
    if selected != trained_lineage:
        plan_fingerprint = None
    previous = active_lineage_id(paths)
    try:
        pointer = activate_generation(
            paths,
            selected,
            plan_fingerprint=plan_fingerprint if isinstance(plan_fingerprint, str) else None,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        console.print(f"[bold red]Activation refused:[/bold red] {exc}")
        raise typer.Exit(1) from None
    _print({"activated": pointer, "previous_active_lineage_id": previous})


@app.command("register-separator-model")
def register_separator_model_command(
    model: Path = typer.Argument(..., help="Local UVR_MDXNET_KARA_2.onnx file."),
    source_url: str = typer.Option(
        ...,
        "--source-url",
        help="Audited upstream/download URL recorded in the local manifest.",
    ),
    model_terms: str = typer.Option(
        ...,
        "--model-terms",
        help="Model-weight license/usage terms accepted for this local copy.",
    ),
) -> None:
    """Register a locally obtained separator weight for offline analysis only."""

    root = _repo_root()
    try:
        result = register_separator_model(
            root,
            model,
            source_url=source_url,
            model_terms=model_terms,
        )
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    _print(result)


@app.command()
def prepare(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    try:
        with console.status(
            f"[bold cyan]Preparing persona {name}...[/bold cyan] "
            "[dim](ASR/diarization/analysis may have long CPU/GPU phases)[/dim]",
            spinner="dots",
        ):
            result = prepare_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command("export-lfm")
def export_lfm_command(name: str) -> None:
    """Regenerate only the LFM SFT export from an existing Prepare master."""

    root, paths, cfg = _load(name)
    prepare_result = StateStore(paths.state).stage("prepare").get("result")
    lineage_id = prepare_result.get("lineage_id") if isinstance(prepare_result, dict) else None
    if isinstance(lineage_id, str) and lineage_id:
        paths = paths.for_lineage(lineage_id)
    master = paths.dataset / "master.sqlite3"
    if not master.is_file():
        raise typer.BadParameter("Prepare master.sqlite3 is missing; run persona prepare first")
    output = paths.dataset / "lfm_train.jsonl"
    tokenizer = load_lfm_tokenizer(root / "models" / "lfm" / "base")
    count = export_lfm(
        master,
        output,
        cfg.name,
        profile=load_core_profile(paths.core_profile, persona_name=cfg.name),
        report_path=paths.dataset / "lfm_quality_report.json",
        lineage_metadata=(
            {
                "lineage_id": prepare_result.get("lineage_id"),
                "lineage_fingerprint": prepare_result.get("lineage_fingerprint"),
                "master_fingerprint": prepare_result.get("master_fingerprint"),
            }
            if isinstance(prepare_result, dict) and prepare_result.get("lineage_id")
            else None
        ),
        tokenizer=tokenizer,
    )
    _print(
        {
            "lfm_examples": count,
            "path": str(output.resolve()),
            "contract": {
                "schema_version": LFM_CONTRACT_SCHEMA_VERSION,
                "fingerprint": LFM_CONTRACT_FINGERPRINT,
            },
            "prepare_irodori_vc_reused": True,
        }
    )


@app.command()
def train(
    name: str,
    force: bool = False,
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Override training.executor for this run: auto, local, or modal.",
    ),
) -> None:
    executor = _executor_override(executor)
    root, paths, cfg = _load(name)
    try:
        with console.status(
            f"[bold cyan]Training persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            result = train_persona(root, paths, cfg, force=force, executor=executor)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command()
def build(
    name: str,
    force: bool = False,
    evaluate_after: bool = typer.Option(True, "--eval/--no-eval"),
    executor: str | None = typer.Option(
        None,
        "--executor",
        help="Override training.executor for this run: auto, local, or modal.",
    ),
) -> None:
    """One-command prepare + train + evaluation."""
    executor = _executor_override(executor)
    root, paths, cfg = _load(name)
    try:
        with console.status(
            f"[bold cyan]1/3 Preparing persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            result = {"prepare": prepare_persona(root, paths, cfg, force=force)}
        with console.status(
            f"[bold cyan]2/3 Training persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            result["train"] = train_persona(
                root,
                paths,
                cfg,
                force=force,
                executor=executor,
            )
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    if evaluate_after:
        with console.status(
            f"[bold cyan]3/3 Evaluating persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            result["evaluation"] = evaluate(root, paths, cfg)["summary"]
    else:
        console.print("[yellow]3/3[/yellow] Evaluation skipped by request.")
    _print(result)


@app.command()
def say(
    name: str,
    text: str,
    style: str | None = None,
    emotion: str | None = None,
    event: list[str] | None = typer.Option(None, "--event"),
    ref: str | None = None,
    candidates: int | None = None,
    seed: int | None = None,
) -> None:
    root, paths, cfg = _load(name)
    outputs = synthesize(
        root,
        paths,
        cfg,
        text,
        style=style,
        emotion=emotion,
        events=event or [],
        ref=ref,
        candidates=candidates,
        seed=seed,
    )
    _print({"outputs": [str(path) for path in outputs]})


@app.command("diagnose-boundaries")
def diagnose_boundaries(
    name: str,
    seed: int = typer.Option(20260826, help="Fixed seed used for every comparable condition."),
    margin_scale: float = typer.Option(
        1.10,
        help="Duration margin to evaluate as C/D; this never changes persona.yaml automatically.",
    ),
    asr: bool = typer.Option(True, "--asr/--no-asr", help="Run pinned ASR for CER/onset evidence."),
    sense: bool = typer.Option(
        True,
        "--sense/--no-sense",
        help="Run SenseVoice for non-verbal event evidence.",
    ),
) -> None:
    """Run Issue #33's inference-only duration/tail and onset diagnosis."""

    root, paths, cfg = _load(name)
    result = run_boundary_diagnostics(
        root,
        paths,
        cfg,
        seed=seed,
        margin_scale=margin_scale,
        include_asr=asr,
        include_sense=sense,
    )
    _print(result)


@app.command()
def reenact(
    name: str,
    source: Path,
    ref: str | None = None,
    transfer_style: bool = typer.Option(True, "--transfer-style/--timbre-only"),
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="VC backend override: seed-vc-v2 or vevo2-fm.",
    ),
) -> None:
    if backend is not None and backend not in VC_BACKENDS:
        raise typer.BadParameter(
            f"Unsupported VC backend {backend!r}; choose one of {', '.join(VC_BACKENDS)}"
        )
    root, paths, cfg = _load(name)
    source = _existing_file(source)
    _print(
        {
            "output": str(
                reenact_audio(
                    root,
                    paths,
                    cfg,
                    source,
                    ref=ref,
                    transfer_style=transfer_style,
                    backend=backend,
                )
            )
        }
    )


@app.command()
def repeat(
    name: str,
    source: Path,
    backend: str | None = typer.Option(
        None,
        "--backend",
        help="VC backend for non-verbal repeat fallback: seed-vc-v2 or vevo2-fm.",
    ),
) -> None:
    if backend is not None and backend not in VC_BACKENDS:
        raise typer.BadParameter(
            f"Unsupported VC backend {backend!r}; choose one of {', '.join(VC_BACKENDS)}"
        )
    root, paths, cfg = _load(name)
    source = _existing_file(source)
    _print(
        {
            "outputs": [
                str(path) for path in repeat_audio(root, paths, cfg, source, backend=backend)
            ]
        }
    )


@app.command()
def chat(name: str, prompt: str | None = None) -> None:
    root, paths, cfg = _load(name)
    history: list[dict[str, str]] = []
    if prompt is not None:
        _print(chat_turn(root, paths, cfg, prompt, history))
        return
    console.print("Interactive chat. Ctrl+C to exit.")
    while True:
        try:
            message = console.input("[bold cyan]You> [/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not message:
            continue
        result = chat_turn(root, paths, cfg, message, history)
        console.print(f"[bold green]{name}>[/bold green] {result.get('text', '')}")
        console.print(f"audio: {result.get('audio')}")
        history.extend(
            [
                {"role": "user", "content": message},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"text": result.get("text"), "voice": result.get("voice", {})},
                        ensure_ascii=False,
                    ),
                },
            ]
        )
        history = history[-12:]


@app.command("eval")
def eval_command(name: str) -> None:
    root, paths, cfg = _load(name)
    _print(evaluate(root, paths, cfg))


@app.command("eval-vc-manifest")
def eval_vc_manifest_command(
    name: str,
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Manifest path; defaults to dataset/vc_evaluation_manifest.jsonl.",
    ),
    reference: str | None = typer.Option(
        None,
        "--reference",
        help="Persona-relative reference audio path; defaults to the first prepared reference.",
    ),
    limit: int | None = typer.Option(None, min=1, max=300),
    seed: int | None = typer.Option(None),
) -> None:
    """Build a deterministic same-input manifest for Seed-VC vs Vevo2 FM."""

    _, paths, cfg = _load(name)
    result = build_vc_evaluation_manifest(
        paths,
        cfg,
        output=output,
        reference=reference,
        limit=limit,
        seed=seed,
    )
    _print(result)


@app.command("eval-vc")
def eval_vc_command(
    name: str,
    manifest: Path = typer.Option(..., "--manifest", help="Canonical VC evaluation JSONL."),
    human_review: Path | None = typer.Option(
        None,
        "--human-review",
        help="Completed human_review.json; omit to keep the gate pending.",
    ),
) -> None:
    """Run canonical same-source/same-reference Seed-VC and Vevo2 A/B evaluation."""

    root, paths, cfg = _load(name)
    result = evaluate_vc(
        root,
        paths,
        cfg,
        _existing_file(manifest),
        human_review=human_review,
    )
    _print(result)


@app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8848,
    allow_remote: bool = typer.Option(
        False,
        help="Allow non-loopback binding without authentication.",
    ),
) -> None:
    if not _is_loopback_host(host) and not allow_remote:
        raise typer.BadParameter(
            "Refusing non-loopback bind. PersonaVoice has no network authentication. "
            "Use --allow-remote only on a trusted network and with deliberate firewall rules."
        )
    _repo_root()
    import uvicorn

    uvicorn.run("personavoice.api:app", host=host, port=port, reload=False)


@app.command()
def ui(port: int = 8848) -> None:
    _repo_root()
    import threading
    import time
    import webbrowser

    import uvicorn

    url = f"http://127.0.0.1:{port}/"
    threading.Thread(
        target=lambda: (time.sleep(1.0), webbrowser.open(url)),
        daemon=True,
    ).start()
    uvicorn.run("personavoice.api:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    app()
