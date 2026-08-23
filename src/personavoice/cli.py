from __future__ import annotations

import ipaddress
import json
from pathlib import Path

import typer
from huggingface_hub.errors import GatedRepoError
from rich.console import Console

from personavoice.config import PersonaConfig
from personavoice.doctor import report as doctor_report
from personavoice.evaluation import evaluate
from personavoice.inference import chat_turn, synthesize
from personavoice.inference import reenact as reenact_audio
from personavoice.inference import repeat as repeat_audio
from personavoice.pipeline import prepare_persona
from personavoice.project import find_repo_root, get_persona, init_persona
from personavoice.repair import repair_failed_model_materializations
from personavoice.setup_env import download_models, install_environments
from personavoice.setup_lock import SetupLockError, setup_lock
from personavoice.stage_lock import StageLockError
from personavoice.status import persona_status
from personavoice.training import train_persona

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


def _download_models_or_explain(root: Path, *, include_seed_vc: bool) -> dict:
    try:
        return download_models(root, include_seed_vc=include_seed_vc)
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
    result = doctor_report(find_repo_root(), deep=deep)
    _print(result)
    if not result["commands_ok"] or (deep and not result["ready_offline"]):
        raise typer.Exit(1)


@app.command()
def setup(
    backend: str = typer.Option("auto", help="Irodori backend: auto/cu126/cu128/cpu/rocm/xpu"),
    download: bool = typer.Option(True, "--download-models/--skip-models"),
    skip_seed_vc_models: bool = typer.Option(False),
    verify: bool = typer.Option(True, "--verify/--no-verify"),
) -> None:
    """Install pinned local uv environments and model snapshots."""
    backend = backend.strip().lower()
    if backend not in SETUP_BACKENDS:
        raise typer.BadParameter(
            f"Unsupported Irodori backend {backend!r}; choose one of "
            f"{', '.join(sorted(SETUP_BACKENDS))}."
        )
    root = find_repo_root()
    try:
        # Serialize the complete setup transaction: environment sync, model
        # materialization/repair, verification, and final state publication.
        # The OS lock is crash-safe and is released automatically on process exit.
        with setup_lock(root):
            result = install_environments(root, backend=None if backend == "auto" else backend)
            if download:
                result["models"] = _download_models_or_explain(
                    root,
                    include_seed_vc=not skip_seed_vc_models,
                )
            if verify:
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
                        result["model_recovery"] = {
                            "discarded_materializations": repaired,
                            "download": _download_models_or_explain(
                                root,
                                include_seed_vc=not skip_seed_vc_models,
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
        result = prepare_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    _print(result)


@app.command()
def train(name: str, force: bool = False) -> None:
    root, paths, cfg = _load(name)
    try:
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
        result = {"prepare": prepare_persona(root, paths, cfg, force=force)}
        result["train"] = train_persona(root, paths, cfg, force=force)
    except StageLockError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        raise typer.Exit(2) from None
    if evaluate_after:
        result["evaluation"] = evaluate(root, paths, cfg)["summary"]
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
