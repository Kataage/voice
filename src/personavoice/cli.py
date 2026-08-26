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
from personavoice.evaluation import evaluate
from personavoice.inference import chat_turn, synthesize
from personavoice.inference import reenact as reenact_audio
from personavoice.inference import repeat as repeat_audio
from personavoice.lfm_contract import LFM_CONTRACT_FINGERPRINT, LFM_CONTRACT_SCHEMA_VERSION
from personavoice.lineage import (
    activate_generation,
    load_lineage,
    prepared_paths,
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
from personavoice.training import train_persona, validate_generation

app = typer.Typer(no_args_is_help=True, help="PersonaVoice local-first voice persona toolkit")
console = Console()
SETUP_BACKENDS = {"auto", "cu126", "cu128", "cpu", "rocm", "xpu"}


def _load(name: str):
    root = find_repo_root()
    paths = get_persona(root, name)
    return root, paths, PersonaConfig.load(paths.config)


def _print(value) -> None:
    console.print_json(data=value)


def _existing_file(path: Path) -> Path:
    value = path.expanduser().resolve()
    if not value.is_file():
        raise typer.BadParameter(f"File does not exist: {value}")
    return value


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
    asr_backend: str = "large-v3",
) -> dict:
    try:
        return download_models(
            root,
            include_seed_vc=include_seed_vc,
            asr_backends=(asr_backend,),
        )
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
            "Set HF_TOKEN in the current shell and rerun PersonaVoice setup. "
            "PersonaVoice never prints or stores the token."
        )
        raise typer.Exit(2) from None


@app.command()
def doctor(
    deep: bool = typer.Option(False, help="Load local models and verify offline readiness."),
) -> None:
    root = find_repo_root()
    if deep:
        with console.status(
            "[bold cyan]Deep offline verification is running...[/bold cyan] "
            "[dim](model loading can take several minutes)[/dim]",
            spinner="dots",
        ):
            result = doctor_report(root, deep=True)
    else:
        result = doctor_report(root, deep=False)
    _print(result)
    if not result["commands_ok"] or (deep and not result["ready_offline"]):
        raise typer.Exit(1)


@app.command()
def setup(
    backend: str = typer.Option("auto", help="Irodori backend: auto/cu126/cu128/cpu/rocm/xpu"),
    download: bool = typer.Option(True, "--download-models/--skip-models"),
    skip_seed_vc_models: bool = typer.Option(False),
    asr_backend: str = typer.Option(
        "large-v3",
        "--asr-backend",
        help="Prepare ASR: large-v3 or qwen3-asr-1.7b; restricted domain backend is disabled.",
    ),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Install pinned local uv environments and model snapshots."""
    backend = backend.strip().lower()
    asr_backend = asr_backend.strip()
    if backend not in SETUP_BACKENDS:
        raise typer.BadParameter(
            f"Unsupported Irodori backend {backend!r}; choose one of "
            f"{', '.join(sorted(SETUP_BACKENDS))}."
        )
    root = find_repo_root()
    try:
        from personavoice.lineage import resolve_backend

        selected_asr = resolve_backend(asr_backend)
    except Exception as exc:
        raise typer.BadParameter(str(exc)) from exc
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
                asr_backend=selected_asr.key,
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
                        asr_backend=selected_asr.key,
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
                    )
                if download and not verification["ready_offline"]:
                    repaired = repair_failed_model_materializations(
                        root,
                        verification,
                        include_seed_vc=not skip_seed_vc_models,
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
                                    asr_backend=selected_asr.key,
                                ),
                            }
                            verification = doctor_report(
                                root,
                                deep=True,
                                require_seed_vc=not skip_seed_vc_models,
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
    paths = init_persona(find_repo_root(), name, authorized=authorized)
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
    cfg.consent.authorized = authorized
    cfg.save(paths.config)
    console.print(f"consent.authorized = {authorized}")


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
    """Regenerate only the LFM export from an existing Prepare lineage."""

    root, paths, cfg = _load(name)
    prepared = prepared_paths(paths)
    if prepared.lineage_id is None:
        raise typer.BadParameter("No immutable Prepare lineage exists; run persona prepare first")
    master = prepared.dataset / "master.sqlite3"
    if not master.is_file():
        raise typer.BadParameter("Prepare master.sqlite3 is missing; run persona prepare first")
    record = load_lineage(prepared, prepared.lineage_id)
    if not isinstance(record, dict):
        raise typer.BadParameter("Prepare lineage record is missing or invalid")
    output = prepared.dataset / "lfm_train.jsonl"
    report = prepared.dataset / "lfm_quality_report.json"
    tokenizer = load_lfm_tokenizer(root / "models" / "lfm" / "base")
    metadata = {
        "lineage_id": record.get("lineage_id"),
        "lineage_fingerprint": record.get("lineage_fingerprint"),
        "master_fingerprint": record.get("master_fingerprint"),
    }
    count = export_lfm(
        master,
        output,
        cfg.name,
        profile=load_core_profile(paths.core_profile, persona_name=cfg.name),
        report_path=report,
        lineage_metadata=metadata,
        tokenizer=tokenizer,
        max_tokens=cfg.training.lfm_max_tokens,
    )
    _print(
        {
            "lfm_examples": count,
            "path": str(output.resolve()),
            "quality_report": str(report.resolve()),
            "contract": {
                "schema_version": LFM_CONTRACT_SCHEMA_VERSION,
                "fingerprint": LFM_CONTRACT_FINGERPRINT,
            },
            "prepare_irodori_vc_reused": True,
            "lfm_only_regeneration": True,
            "lineage_id": prepared.lineage_id,
        }
    )


@app.command()
def validate(
    name: str,
    lineage_id: str | None = typer.Option(None, "--lineage-id"),
    generation_id: str | None = typer.Option(None, "--generation-id"),
) -> None:
    """Validate a trained v0.3 candidate without activating it."""

    if (lineage_id is None) != (generation_id is None):
        raise typer.BadParameter("--lineage-id and --generation-id must be supplied together")
    root, paths, cfg = _load(name)
    try:
        result = validate_generation(
            root,
            paths,
            cfg,
            generation_id=generation_id,
            lineage_id=lineage_id,
        )
    except (StageLockError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from None
    _print(result)
    if result.get("passed") is not True:
        raise typer.Exit(1)


@app.command()
def activate(
    name: str,
    lineage_id: str | None = typer.Option(
        None,
        "--lineage-id",
        help="Prepare lineage to activate; use with --generation-id for rollback.",
    ),
    generation_id: str | None = typer.Option(
        None,
        "--generation-id",
        help="Validated v0.3 generation to activate or restore.",
    ),
) -> None:
    """Atomically activate a validated v0.3 candidate generation."""

    _, paths, _ = _load(name)
    if (lineage_id is None) != (generation_id is None):
        raise typer.BadParameter("--lineage-id and --generation-id must be supplied together")

    selected_lineage: str
    selected_generation: str
    selected_fingerprint: str
    if lineage_id is not None and generation_id is not None:
        try:
            candidate = paths.for_generation(lineage_id, generation_id)
            manifest = json.loads(candidate.generation_manifest.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise typer.BadParameter(
                "The requested candidate generation manifest is missing or unreadable"
            ) from exc
        if not isinstance(manifest, dict) or not isinstance(
            manifest.get("generation_fingerprint"), str
        ):
            raise typer.BadParameter("The requested candidate has no valid generation fingerprint")
        selected_lineage = lineage_id
        selected_generation = generation_id
        selected_fingerprint = manifest["generation_fingerprint"]
    else:
        result = StateStore(paths.state).stage("train").get("result")
        if not isinstance(result, dict):
            raise typer.BadParameter("No trained candidate exists; run persona train first")
        selected_lineage = result.get("lineage_id")
        selected_generation = result.get("generation_id")
        selected_fingerprint = result.get("generation_fingerprint")
        if not all(
            isinstance(value, str)
            for value in (selected_lineage, selected_generation, selected_fingerprint)
        ):
            raise typer.BadParameter("Training result has no complete lineage/generation identity")
        validation = result.get("validation")
        if not isinstance(validation, dict) or validation.get("passed") is not True:
            raise typer.BadParameter("Candidate has not passed validation; run persona validate first")

    try:
        pointer = activate_generation(
            paths,
            selected_lineage,
            generation_id=selected_generation,
            generation_fingerprint=selected_fingerprint,
        )
    except (ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from None
    _print({"activation": pointer})


@app.command("register-separator-model")
def register_separator_model_command(
    source: Path,
    source_url: str = typer.Option(..., "--source-url"),
    model_terms: str = typer.Option(..., "--model-terms"),
) -> None:
    """Register a locally reviewed UVR separator weight; no implicit download occurs."""

    root = find_repo_root()
    try:
        result = register_separator_model(
            root,
            _existing_file(source),
            source_url=source_url,
            model_terms=model_terms,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(1) from None
    _print(result)


@app.command()
def train(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    try:
        with console.status(
            f"[bold cyan]Training persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            result = train_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command()
def build(
    name: str,
    force: bool = False,
    evaluate_after: bool = typer.Option(True, "--eval/--no-eval"),
) -> None:
    """One-command prepare + train + evaluation."""
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
            result["train"] = train_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    if evaluate_after:
        with console.status(
            f"[bold cyan]3/3 Evaluating persona {name}...[/bold cyan]",
            spinner="dots",
        ):
            evaluation_paths = paths
            train_result = result.get("train")
            if isinstance(train_result, dict):
                candidate_lineage = train_result.get("lineage_id")
                candidate_generation = train_result.get("generation_id")
                if isinstance(candidate_lineage, str) and isinstance(candidate_generation, str):
                    evaluation_paths = paths.for_generation(candidate_lineage, candidate_generation)
            result["evaluation"] = evaluate(root, evaluation_paths, cfg)["summary"]
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
    duration_scale: float | None = typer.Option(
        None,
        help="Override inference.duration_scale for this utterance only.",
    ),
    trim_tail: bool | None = typer.Option(
        None,
        "--trim-tail/--no-trim-tail",
        help="Override latent tail trimming for this utterance only.",
    ),
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
        duration_scale=duration_scale,
        trim_tail=trim_tail,
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
    asr: bool = typer.Option(
        True,
        "--asr/--no-asr",
        help="Run pinned ASR for CER/onset evidence.",
    ),
    sense: bool = typer.Option(
        True,
        "--sense/--no-sense",
        help="Run SenseVoice for non-verbal event evidence.",
    ),
) -> None:
    """Run the Issue #33 inference-only duration/tail and onset diagnosis."""

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
) -> None:
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
                )
            )
        }
    )


@app.command()
def repeat(name: str, source: Path) -> None:
    root, paths, cfg = _load(name)
    source = _existing_file(source)
    _print({"outputs": [str(path) for path in repeat_audio(root, paths, cfg, source)]})


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
    import uvicorn

    uvicorn.run("personavoice.api:app", host=host, port=port, reload=False)


@app.command()
def ui(port: int = 8848) -> None:
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
