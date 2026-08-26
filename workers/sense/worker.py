from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from uuid import uuid4

import torch
from funasr import AutoModel
from modelscope import snapshot_download

MODEL_ID = "iic/SenseVoiceSmall"
MODEL_WEIGHT_SHA256 = "833ca2dcfdf8ec91bd4f31cfac36d6124e0c459074d5e909aec9cabe6204a3ea"
MODEL_CMVN_SHA256 = "29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5"
MODEL_TOKENIZER_SHA256 = "aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8"
MODEL_ASSETS = {
    "model.pt": MODEL_WEIGHT_SHA256,
    "am.mvn": MODEL_CMVN_SHA256,
    "chn_jpn_yue_eng_ko_spectok.bpe.model": MODEL_TOKENIZER_SHA256,
}
EMOTIONS = {"HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"}
EVENTS = {
    "BGM",
    "Speech",
    "Applause",
    "Laughter",
    "Cry",
    "Sneeze",
    "Breath",
    "Cough",
    "Sing",
    "Speech_Noise",
}
TAG_RE = re.compile(r"<\|([^|]+)\|>")


def read_request(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_CHECKPOINT_ALLOWED = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


def _checkpoint_item_id(value: object) -> str:
    item_id = str(value)
    if (
        not item_id
        or len(item_id) > 128
        or item_id == "progress"
        or not item_id[0].isalnum()
        or any(char not in _CHECKPOINT_ALLOWED for char in item_id)
    ):
        raise ValueError(f"Unsafe prepare checkpoint item id: {item_id!r}")
    return item_id


def _checkpoint_directory(payload: dict) -> Path | None:
    raw = payload.get("checkpoint_dir")
    if raw is None:
        return None
    target = Path(str(raw))
    if not target.is_absolute():
        raise ValueError("Prepare checkpoint directory must be an absolute path")
    root = Path(os.environ["PERSONAVOICE_ROOT"]).resolve()
    resolved = target.resolve(strict=False)
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("Prepare checkpoint directory escapes PERSONAVOICE_ROOT") from exc
    parts = relative.parts
    base_layout = len(parts) >= 5 and parts[0] == "personas" and parts[2] == "cache"
    lineage_layout = (
        len(parts) >= 8
        and parts[0] == "personas"
        and parts[2:4] == ("generations", "prepare")
        and re.fullmatch(r"pl-[0-9a-f]{32}", parts[4] or "") is not None
        and parts[5] == "cache"
    )
    if (not base_layout and not lineage_layout) or parts[-1] != ".checkpoints":
        raise ValueError(
            "Prepare checkpoint directory must be personas/<name>/cache/<kind>/.checkpoints "
            "or personas/<name>/generations/prepare/<lineage>/cache/<kind>/.checkpoints"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    verified = resolved.resolve()
    try:
        verified.relative_to(root)
    except ValueError as exc:
        raise ValueError("Prepare checkpoint directory resolves outside PERSONAVOICE_ROOT") from exc
    return verified


def _atomic_checkpoint_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_item_checkpoint(directory: Path | None, item_id: str, result: dict) -> None:
    if directory is None:
        return
    safe_id = _checkpoint_item_id(item_id)
    _atomic_checkpoint_json(
        directory / f"{safe_id}.json",
        {"schema": 1, "id": safe_id, "result": result},
    )


def _write_batch_progress(
    directory: Path | None,
    *,
    worker_name: str,
    command: str,
    phase: str,
    completed: int,
    total: int,
    failed: int,
    current_id: str | None,
    state: str,
) -> None:
    if directory is None:
        return
    _atomic_checkpoint_json(
        directory / "progress.json",
        {
            "schema": 1,
            "worker": worker_name,
            "command": command,
            "phase": phase,
            "completed": completed,
            "total": total,
            "failed": failed,
            "current_id": current_id,
            "state": state,
        },
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root() -> Path:
    return Path(os.environ["PERSONAVOICE_ROOT"])


def _local_dir() -> Path:
    return _root() / "models" / "sense" / "SenseVoiceSmall"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _mark_verified() -> None:
    _atomic_write_text(_root() / ".runtime" / "sense-model-ready", "verified\n")


def verify_local_assets() -> dict:
    local = _local_dir()
    if not local.is_dir():
        raise FileNotFoundError(f"SenseVoice model directory is missing: {local}")
    verified: dict[str, str] = {}
    for relative, expected in MODEL_ASSETS.items():
        path = local / relative
        if not path.is_file():
            raise FileNotFoundError(f"SenseVoice required asset is missing: {path}")
        actual = _sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(
                f"SenseVoice checksum mismatch for {relative}: expected {expected}, got {actual}. "
                "Remove the local SenseVoice model and rerun `persona setup`."
            )
        verified[relative] = actual
    # Verification is the authority for this marker. This also migrates legacy
    # `ready` markers from older PersonaVoice versions without re-downloading a
    # byte when the already materialized model matches the audited hashes.
    _mark_verified()
    return {"model": MODEL_ID, "path": str(local), "verified": verified}


def local_model() -> str:
    """Return only the audited local model path.

    Normal inference and deep health are intentionally fail-closed. The sole
    operation allowed to contact ModelScope is the explicit `download` command.
    """

    local = _local_dir()
    if not local.is_dir():
        raise FileNotFoundError(
            f"Pinned SenseVoice model is missing: {local}. Run `persona setup`."
        )
    verify_local_assets()
    return str(local)


def load_model() -> AutoModel:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Use the uv-locked FunASR implementation rather than executing model-repo
    # Python code. The inference-critical model assets are hash-verified above.
    # FunASR's version check is explicitly disabled so deep doctor remains a
    # deterministic local-only operation instead of attempting an update lookup.
    return AutoModel(
        model=local_model(),
        trust_remote_code=False,
        disable_update=True,
        device=device,
    )


def parse_result(result) -> dict:
    raw = result[0].get("text", "") if result else ""
    tags = TAG_RE.findall(raw)
    emotion = next((tag for tag in tags if tag in EMOTIONS), "UNKNOWN")
    events = []
    for tag in tags:
        if tag in EVENTS and tag not in events:
            events.append(tag)
    return {"raw": raw, "emotion": emotion, "events": events, "tags": tags}


def analyze_with_model(model: AutoModel, audio: str, *, language: str) -> dict:
    result = model.generate(
        input=audio,
        cache={},
        language=language,
        use_itn=True,
        batch_size=1,
    )
    return parse_result(result)


def analyze(payload: dict) -> dict:
    model = load_model()
    return analyze_with_model(model, payload["audio"], language=payload.get("language", "ja"))


def batch_analyze(payload: dict) -> dict:
    model = load_model()
    checkpoint = _checkpoint_directory(payload)
    language = payload.get("language", "ja")
    items = payload.get("items") or []
    total = len(items)
    failed = 0
    results = []
    _write_batch_progress(
        checkpoint,
        worker_name="sense",
        command="batch_analyze",
        phase="analyze",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
    )
    for index, item in enumerate(items):
        item_id = _checkpoint_item_id(item["id"])
        _write_batch_progress(
            checkpoint,
            worker_name="sense",
            command="batch_analyze",
            phase="analyze",
            completed=index,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
        )
        try:
            value = analyze_with_model(model, str(item["audio"]), language=language)
            _write_item_checkpoint(checkpoint, item_id, value)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        _write_batch_progress(
            checkpoint,
            worker_name="sense",
            command="batch_analyze",
            phase="analyze",
            completed=index + 1,
            total=total,
            failed=failed,
            current_id=None,
            state="running",
        )
    _write_batch_progress(
        checkpoint,
        worker_name="sense",
        command="batch_analyze",
        phase="analyze",
        completed=total,
        total=total,
        failed=failed,
        current_id=None,
        state="finished",
    )
    return {"results": results}


def download_model(payload: dict) -> dict:
    local = _local_dir()
    local.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(MODEL_ID, local_dir=str(local))
    return verify_local_assets()


def health(payload: dict) -> dict:
    result = {"ok": True, "cuda": torch.cuda.is_available()}
    if payload.get("deep"):
        model = load_model()
        result["model_loaded"] = model is not None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PersonaVoice isolated SenseVoice worker"
    )
    parser.add_argument(
        "command",
        choices=["analyze", "batch_analyze", "download", "verify", "health"],
    )
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    payload = read_request(args.request)
    if args.command == "analyze":
        result = analyze(payload)
    elif args.command == "batch_analyze":
        result = batch_analyze(payload)
    elif args.command == "download":
        result = download_model(payload)
    elif args.command == "verify":
        result = verify_local_assets()
    else:
        result = health(payload)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
