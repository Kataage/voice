from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from personavoice.config import PersonaConfig
from personavoice.media import inventory, inventory_fingerprint
from personavoice.project import find_repo_root, init_persona
from personavoice.state import StateStore

app = typer.Typer(no_args_is_help=True, help="PersonaVoice local orchestration CLI")
console = Console()


def _persona_root(name: str) -> Path:
    root = find_repo_root() / "personas" / name
    if not root.exists():
        raise typer.BadParameter(f"Unknown persona: {name!r}. Run `persona init {name}` first.")
    return root


@app.command()
def doctor() -> None:
    """Check local prerequisites without downloading anything."""
    checks = {
        "python": sys.version.split()[0],
        "uv": shutil.which("uv") or "NOT FOUND",
        "ffmpeg": shutil.which("ffmpeg") or "NOT FOUND",
        "ffprobe": shutil.which("ffprobe") or "NOT FOUND",
        "nvidia-smi": shutil.which("nvidia-smi") or "not found (optional)",
    }
    table = Table(title="PersonaVoice doctor")
    table.add_column("Check")
    table.add_column("Result")
    for key, value in checks.items():
        table.add_row(key, value)
    console.print(table)

    missing_required = [key for key in ("uv", "ffmpeg", "ffprobe") if checks[key] == "NOT FOUND"]
    if missing_required:
        raise typer.Exit(code=1)


@app.command("init")
def init_command(
    name: str,
    authorized: bool = typer.Option(False, "--authorized", help="Record that voice use is authorized."),
) -> None:
    """Create a local persona workspace."""
    paths = init_persona(find_repo_root(), name, authorized=authorized)
    console.print(f"Created persona workspace: [bold]{paths.root}[/bold]")
    if not authorized:
        console.print("[yellow]Consent is not marked authorized yet in persona.yaml.[/yellow]")


@app.command()
def status(name: str) -> None:
    """Show persona configuration and resumable stage state."""
    root = _persona_root(name)
    config = PersonaConfig.load(root / "persona.yaml")
    state = json.loads((root / "state.json").read_text(encoding="utf-8"))
    console.print_json(
        data={
            "config": config.model_dump(mode="json"),
            "state": state,
            "raw_files": sum(1 for path in (root / "raw").rglob("*") if path.is_file() and path.name != ".gitkeep"),
            "identity_files": sum(1 for path in (root / "identity").rglob("*") if path.is_file() and path.name != ".gitkeep"),
        }
    )


def _reserved(command: str) -> None:
    console.print(
        f"[yellow]{command} is part of the stable CLI contract but its model worker is not wired yet.[/yellow]"
    )
    raise typer.Exit(code=2)


@app.command()
def prepare(name: str, force: bool = typer.Option(False, "--force")) -> None:
    """Inventory raw media and record reproducible source metadata."""
    root = _persona_root(name)
    config = PersonaConfig.load(root / "persona.yaml")
    if not config.consent.authorized:
        console.print("[red]Training preparation is blocked until consent.authorized=true.[/red]")
        raise typer.Exit(code=3)

    raw_dir = root / "raw"
    fingerprint = inventory_fingerprint(raw_dir)
    state = StateStore(root / "state.json")
    if not force and state.is_complete("media_inventory", fingerprint):
        console.print("Media inventory is unchanged; skipping.")
        return

    with state.running("media_inventory", fingerprint):
        rows = inventory(raw_dir)
        destination = root / "dataset" / "source_inventory.json"
        destination.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    console.print(f"Inventoried [bold]{len(rows)}[/bold] media files -> {destination}")


@app.command()
def train(name: str) -> None:
    _persona_root(name)
    _reserved("train")


@app.command()
def say(name: str, text: str) -> None:
    _persona_root(name)
    _reserved("say")


@app.command()
def reenact(name: str, audio: Path) -> None:
    _persona_root(name)
    _reserved("reenact")


@app.command()
def repeat(name: str, audio: Path) -> None:
    _persona_root(name)
    _reserved("repeat")


@app.command()
def chat(name: str) -> None:
    _persona_root(name)
    _reserved("chat")


@app.command()
def ui(name: str) -> None:
    _persona_root(name)
    _reserved("ui")


@app.command()
def serve(name: str | None = None) -> None:
    if name is not None:
        _persona_root(name)
    _reserved("serve")


if __name__ == "__main__":
    app()
