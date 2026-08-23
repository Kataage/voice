from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD_POLICY = os.environ.get("PERSONAVOICE_OLD_PREPARE_POLICY")
if not OLD_POLICY:
    raise RuntimeError("PERSONAVOICE_OLD_PREPARE_POLICY is required")
NEW_POLICY_PLACEHOLDER = "__PERSONAVOICE_NEW_PREPARE_POLICY__"


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"Expected at least {count} patch anchor(s) in {path}, found {found}: {old[:160]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


WORKER_CHECKPOINT_HELPERS = r'''
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
    if (
        len(parts) < 5
        or parts[0] != "personas"
        or parts[2] != "cache"
        or parts[-1] != ".checkpoints"
    ):
        raise ValueError(
            "Prepare checkpoint directory must be personas/<name>/cache/<kind>/.checkpoints"
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

'''

# Central pipeline consumes worker-written candidates only after semantic validation.
patch(
    "src/personavoice/pipeline.py",
    "from personavoice.project import PersonaPaths\n",
    "from personavoice.prepare_checkpoints import (\n"
    "    checkpoint_dir,\n"
    "    cleanup_checkpoint_dir,\n"
    "    discard_checkpoint,\n"
    "    recover_checkpoint,\n"
    ")\n"
    "from personavoice.project import PersonaPaths\n",
)

old_batch_block = r'''def _identity_embeddings(repo_root: Path, paths: PersonaPaths) -> list[list[float]]:
    identity = media_files(paths.identity)
    if not identity:
        return []
    diarization = worker(repo_root, "diarization")
    values_by_key: dict[str, list[float]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    for source in identity:
        key = sha256_file(source)[:20]
        cache = paths.cache / "identity" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        embedding = cached.get("embedding") if cached is not None else None
        if isinstance(embedding, list) and embedding:
            values_by_key[key] = [float(value) for value in embedding]
        else:
            if cache.exists():
                cache.unlink(missing_ok=True)
            pending.append({"id": key, "audio": str(source.resolve())})
    if pending:
        response = diarization.call(
            repo_root,
            "batch",
            {"embeddings": pending, "diarizations": []},
        )
        results = _batch_results(response.get("embeddings") or [], operation="identity embedding")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            embedding = result.get("embedding")
            if embedding:
                values_by_key[key] = [float(value) for value in embedding]
    return [values_by_key[key] for key in sorted(values_by_key)]


def _batch_asr(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "asr" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            values[source_id] = cached
        else:
            pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "asr").call(
            repo_root,
            "batch_transcribe",
            {
                "items": pending,
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "language": cfg.prepare.language,
            },
        )
        results = _batch_results(response.get("results") or [], operation="ASR")
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            values[source_id] = result
    return values


def _batch_diarization(
    repo_root: Path,
    paths: PersonaPaths,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "diarization" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            values[source_id] = cached
        else:
            pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "diarization").call(
            repo_root,
            "batch",
            {"embeddings": [], "diarizations": pending},
        )
        results = _batch_results(
            response.get("diarizations") or [],
            operation="speaker diarization",
        )
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            values[source_id] = result
    return values


def _batch_sense(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not cfg.prepare.use_sensevoice:
        return {}
    values: dict[str, dict[str, Any]] = {}
    pending = []
    pending_keys: set[str] = set()
    cache_paths: dict[str, Path] = {}
    for row in rows:
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        audio = Path(audio_path)
        key = sha256_file(audio)[:20]
        row["sense_key"] = key
        cache = paths.cache / "sense" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            values[key] = cached
        elif key not in pending_keys:
            pending.append({"id": key, "audio": str(audio.resolve())})
            pending_keys.add(key)
    if pending:
        response = worker(repo_root, "sense").call(
            repo_root,
            "batch_analyze",
            {"items": pending, "language": cfg.prepare.language},
        )
        results = _batch_results(response.get("results") or [], operation="SenseVoice analysis")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            values[key] = result
    return values
'''

new_batch_block = r'''def _identity_embeddings(repo_root: Path, paths: PersonaPaths) -> list[list[float]]:
    identity = media_files(paths.identity)
    if not identity:
        return []
    diarization = worker(repo_root, "diarization")
    values_by_key: dict[str, list[float]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "identity")
    for source in identity:
        key = sha256_file(source)[:20]
        cache = paths.cache / "identity" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        embedding = cached.get("embedding") if cached is not None else None
        if isinstance(embedding, list) and embedding:
            discard_checkpoint(checkpoints, key)
            values_by_key[key] = [float(value) for value in embedding]
            continue
        if cache.exists():
            cache.unlink(missing_ok=True)
        recovered = recover_checkpoint(checkpoints, key, "identity")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, key)
            embedding = recovered.get("embedding")
            if isinstance(embedding, list) and embedding:
                values_by_key[key] = [float(value) for value in embedding]
                continue
        pending.append({"id": key, "audio": str(source.resolve())})
    if pending:
        response = diarization.call(
            repo_root,
            "batch",
            {
                "embeddings": pending,
                "diarizations": [],
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("embeddings") or [], operation="identity embedding")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            discard_checkpoint(checkpoints, key)
            embedding = result.get("embedding")
            if embedding:
                values_by_key[key] = [float(value) for value in embedding]
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return [values_by_key[key] for key in sorted(values_by_key)]


def _batch_asr(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "asr")
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "asr" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = cached
            continue
        recovered = recover_checkpoint(checkpoints, source_id, "asr")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = recovered
            continue
        pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "asr").call(
            repo_root,
            "batch_transcribe",
            {
                "items": pending,
                "model": cfg.prepare.asr_model,
                "compute_type": cfg.prepare.asr_compute_type,
                "language": cfg.prepare.language,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("results") or [], operation="ASR")
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values


def _batch_diarization(
    repo_root: Path,
    paths: PersonaPaths,
    sources: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    pending = []
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "diarization")
    for source in sources:
        source_id = str(source["source_id"])
        cache = paths.cache / "diarization" / f"{source_id}.json"
        cache_paths[source_id] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = cached
            continue
        recovered = recover_checkpoint(checkpoints, source_id, "diarization")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = recovered
            continue
        pending.append({"id": source_id, "audio": str(source["audio"].resolve())})
    if pending:
        response = worker(repo_root, "diarization").call(
            repo_root,
            "batch",
            {
                "embeddings": [],
                "diarizations": pending,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(
            response.get("diarizations") or [],
            operation="speaker diarization",
        )
        for source_id, result in results.items():
            _dump(cache_paths[source_id], result)
            discard_checkpoint(checkpoints, source_id)
            values[source_id] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values


def _batch_sense(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not cfg.prepare.use_sensevoice:
        return {}
    values: dict[str, dict[str, Any]] = {}
    pending = []
    pending_keys: set[str] = set()
    cache_paths: dict[str, Path] = {}
    checkpoints = checkpoint_dir(paths.cache / "sense")
    for row in rows:
        audio_path = row.get("audio_path")
        if not audio_path:
            continue
        audio = Path(audio_path)
        key = sha256_file(audio)[:20]
        row["sense_key"] = key
        cache = paths.cache / "sense" / f"{key}.json"
        cache_paths[key] = cache
        cached = _read_cache_json(cache) if cache.is_file() else None
        if cached is not None:
            discard_checkpoint(checkpoints, key)
            values[key] = cached
            continue
        recovered = recover_checkpoint(checkpoints, key, "sense")
        if recovered is not None:
            _dump(cache, recovered)
            discard_checkpoint(checkpoints, key)
            values[key] = recovered
            continue
        if key not in pending_keys:
            pending.append({"id": key, "audio": str(audio.resolve())})
            pending_keys.add(key)
    if pending:
        response = worker(repo_root, "sense").call(
            repo_root,
            "batch_analyze",
            {
                "items": pending,
                "language": cfg.prepare.language,
                "checkpoint_dir": str(checkpoints.resolve()),
            },
        )
        results = _batch_results(response.get("results") or [], operation="SenseVoice analysis")
        for key, result in results.items():
            _dump(cache_paths[key], result)
            discard_checkpoint(checkpoints, key)
            values[key] = result
        cleanup_checkpoint_dir(checkpoints)
    elif checkpoints.exists():
        cleanup_checkpoint_dir(checkpoints)
    return values
'''
patch("src/personavoice/pipeline.py", old_batch_block, new_batch_block)

# Status exposes advisory live progress; stage locks remain the liveness authority.
patch(
    "src/personavoice/status.py",
    "from personavoice.pipeline import _prepare_fingerprint\n",
    "from personavoice.pipeline import _prepare_fingerprint\n"
    "from personavoice.prepare_checkpoints import prepare_batch_progress\n",
)
patch(
    "src/personavoice/status.py",
    '    prepare = _stage_audit(store, state, "prepare")\n'
    '    train = _stage_audit(store, state, "train")\n',
    '    prepare = _stage_audit(store, state, "prepare")\n'
    '    prepare["batch_progress"] = prepare_batch_progress(paths.root)\n'
    '    train = _stage_audit(store, state, "train")\n',
)

# Cache policy changes because pipeline/worker implementations change. Preserve
# fully validated caches from exactly the immediately preceding main generation,
# but only while the current policy is this specific migration target.
patch("src/personavoice/state.py", '        "schema": 12,\n', '        "schema": 13,\n')
patch(
    "src/personavoice/state.py",
    '        "pipeline_code_sha256": _file_contract(repo / "src" / "personavoice" / "pipeline.py"),\n',
    '        "pipeline_code_sha256": _file_contract(repo / "src" / "personavoice" / "pipeline.py"),\n'
    '        "prepare_checkpoints_code_sha256": _file_contract(\n'
    '            repo / "src" / "personavoice" / "prepare_checkpoints.py"\n'
    '        ),\n',
)
patch(
    "src/personavoice/state.py",
    '    return f"12-{hashlib.sha256(encoded).hexdigest()[:20]}"\n',
    '    return f"13-{hashlib.sha256(encoded).hexdigest()[:20]}"\n',
)
patch(
    "src/personavoice/state.py",
    "PREPARE_CACHE_POLICY_VERSION = _prepare_cache_policy()\n\n\n",
    "PREPARE_CACHE_POLICY_VERSION = _prepare_cache_policy()\n"
    "PREPARE_CACHE_POLICY_COMPATIBILITY = {\n"
    f'    "{NEW_POLICY_PLACEHOLDER}": frozenset({{{OLD_POLICY!r}}}),\n'
    "}\n\n\n"
    "def _prepare_policy_compatible(recorded: Any) -> bool:\n"
    "    if recorded == PREPARE_CACHE_POLICY_VERSION:\n"
    "        return True\n"
    "    compatible = PREPARE_CACHE_POLICY_COMPATIBILITY.get(PREPARE_CACHE_POLICY_VERSION)\n"
    "    return bool(compatible and recorded in compatible)\n\n\n",
)
patch(
    "src/personavoice/state.py",
    '        if name == "prepare" and stage.get("cache_policy_version") != PREPARE_CACHE_POLICY_VERSION:\n'
    "            return False\n",
    '        if name == "prepare" and not _prepare_policy_compatible(\n'
    '            stage.get("cache_policy_version")\n'
    '        ):\n'
    "            return False\n",
)
patch(
    "src/personavoice/state.py",
    "                must_invalidate = (\n"
    "                    force\n"
    "                    or old_policy != PREPARE_CACHE_POLICY_VERSION\n"
    "                    or (old_fingerprint is not None and old_fingerprint != fingerprint)\n"
    "                )\n",
    "                must_invalidate = (\n"
    "                    force\n"
    "                    or not _prepare_policy_compatible(old_policy)\n"
    "                    or (old_fingerprint is not None and old_fingerprint != fingerprint)\n"
    "                )\n",
)

# Add the isolated worker checkpoint writer. It is intentionally duplicated in
# the three isolated projects so no PYTHONPATH/shared-code relaxation is needed.
for path, anchor in (
    (
        "workers/asr/worker.py",
        'def request(path: str) -> dict:\n    return json.loads(Path(path).read_text(encoding="utf-8"))\n\n',
    ),
    (
        "workers/diarization/worker.py",
        'def read_request(path: str) -> dict:\n    return json.loads(Path(path).read_text(encoding="utf-8"))\n\n',
    ),
    (
        "workers/sense/worker.py",
        'def read_request(path: str) -> dict:\n    return json.loads(Path(path).read_text(encoding="utf-8"))\n\n',
    ),
):
    patch(path, anchor, anchor + WORKER_CHECKPOINT_HELPERS)

old_asr_batch = r'''def batch_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    name = payload.get("model", PINNED_MODEL_NAME)
    model = make_model(name, payload.get("compute_type", "auto"))
    language = payload.get("language") or "ja"
    results = []
    for item in items:
        item_id = str(item["id"])
        try:
            value = transcribe_with_model(model, str(item["audio"]), language=language)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"results": results}
'''
new_asr_batch = r'''def batch_transcribe(payload: dict) -> dict:
    items = payload.get("items") or []
    checkpoint = _checkpoint_directory(payload)
    name = payload.get("model", PINNED_MODEL_NAME)
    model = make_model(name, payload.get("compute_type", "auto"))
    language = payload.get("language") or "ja"
    results = []
    failed = 0
    total = len(items)
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="transcribe",
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
            worker_name="asr",
            command="batch_transcribe",
            phase="transcribe",
            completed=index,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
        )
        try:
            value = transcribe_with_model(model, str(item["audio"]), language=language)
            _write_item_checkpoint(checkpoint, item_id, value)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        _write_batch_progress(
            checkpoint,
            worker_name="asr",
            command="batch_transcribe",
            phase="transcribe",
            completed=index + 1,
            total=total,
            failed=failed,
            current_id=None,
            state="running",
        )
    _write_batch_progress(
        checkpoint,
        worker_name="asr",
        command="batch_transcribe",
        phase="transcribe",
        completed=total,
        total=total,
        failed=failed,
        current_id=None,
        state="finished",
    )
    return {"results": results}
'''
patch("workers/asr/worker.py", old_asr_batch, new_asr_batch)

old_diar_batch = r'''def batch(payload: dict) -> dict:
    pipeline = load_pipeline()
    results: dict[str, list[dict]] = {"embeddings": [], "diarizations": []}
    for item in payload.get("embeddings") or []:
        item_id = str(item["id"])
        try:
            value = embed_with_pipeline(pipeline, str(item["audio"]))
            results["embeddings"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["embeddings"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    for item in payload.get("diarizations") or []:
        item_id = str(item["id"])
        try:
            value = diarize_with_pipeline(pipeline, str(item["audio"]))
            results["diarizations"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results["diarizations"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
    return results
'''
new_diar_batch = r'''def batch(payload: dict) -> dict:
    pipeline = load_pipeline()
    checkpoint = _checkpoint_directory(payload)
    embeddings = payload.get("embeddings") or []
    diarizations = payload.get("diarizations") or []
    total = len(embeddings) + len(diarizations)
    completed = 0
    failed = 0
    results: dict[str, list[dict]] = {"embeddings": [], "diarizations": []}
    _write_batch_progress(
        checkpoint,
        worker_name="diarization",
        command="batch",
        phase="embedding" if embeddings else "diarization",
        completed=0,
        total=total,
        failed=0,
        current_id=None,
        state="running",
    )
    for item in embeddings:
        item_id = _checkpoint_item_id(item["id"])
        _write_batch_progress(
            checkpoint,
            worker_name="diarization",
            command="batch",
            phase="embedding",
            completed=completed,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
        )
        try:
            value = embed_with_pipeline(pipeline, str(item["audio"]))
            _write_item_checkpoint(checkpoint, item_id, value)
            results["embeddings"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results["embeddings"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        completed += 1
        _write_batch_progress(
            checkpoint,
            worker_name="diarization",
            command="batch",
            phase="embedding",
            completed=completed,
            total=total,
            failed=failed,
            current_id=None,
            state="running",
        )
    for item in diarizations:
        item_id = _checkpoint_item_id(item["id"])
        _write_batch_progress(
            checkpoint,
            worker_name="diarization",
            command="batch",
            phase="diarization",
            completed=completed,
            total=total,
            failed=failed,
            current_id=item_id,
            state="running",
        )
        try:
            value = diarize_with_pipeline(pipeline, str(item["audio"]))
            _write_item_checkpoint(checkpoint, item_id, value)
            results["diarizations"].append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            failed += 1
            results["diarizations"].append(
                {"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        completed += 1
        _write_batch_progress(
            checkpoint,
            worker_name="diarization",
            command="batch",
            phase="diarization",
            completed=completed,
            total=total,
            failed=failed,
            current_id=None,
            state="running",
        )
    _write_batch_progress(
        checkpoint,
        worker_name="diarization",
        command="batch",
        phase="diarization" if diarizations else "embedding",
        completed=completed,
        total=total,
        failed=failed,
        current_id=None,
        state="finished",
    )
    return results
'''
patch("workers/diarization/worker.py", old_diar_batch, new_diar_batch)

old_sense_batch = r'''def batch_analyze(payload: dict) -> dict:
    model = load_model()
    language = payload.get("language", "ja")
    results = []
    for item in payload.get("items") or []:
        item_id = str(item["id"])
        try:
            value = analyze_with_model(model, str(item["audio"]), language=language)
            results.append({"id": item_id, "ok": True, "result": value})
        except Exception as exc:
            results.append({"id": item_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return {"results": results}
'''
new_sense_batch = r'''def batch_analyze(payload: dict) -> dict:
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
'''
patch("workers/sense/worker.py", old_sense_batch, new_sense_batch)

# Regression tests: semantic checkpoint recovery, crash resume, status visibility,
# path confinement, and the one-generation cache-policy compatibility migration.
test_path = ROOT / "tests" / "test_prepare_checkpoints.py"
test_path.write_text(
r'''from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from personavoice import pipeline
from personavoice.atomic import atomic_write_json
from personavoice.config import PersonaConfig
from personavoice.prepare_checkpoints import (
    checkpoint_dir,
    prepare_batch_progress,
    recover_checkpoint,
)
from personavoice.project import init_persona
from personavoice.state import (
    PREPARE_CACHE_POLICY_COMPATIBILITY,
    PREPARE_CACHE_POLICY_VERSION,
    _prepare_policy_compatible,
)
from personavoice.status import persona_status


def _asr_result() -> dict:
    return {
        "language": "ja",
        "language_probability": 1.0,
        "duration": 1.0,
        "segments": [],
    }


def _checkpoint(directory: Path, item_id: str, result: dict) -> Path:
    path = directory / f"{item_id}.json"
    atomic_write_json(path, {"schema": 1, "id": item_id, "result": result})
    return path


def test_checkpoint_requires_matching_id_and_semantic_result(tmp_path: Path):
    directory = tmp_path / ".checkpoints"
    item_id = "abc123"
    path = _checkpoint(directory, item_id, _asr_result())
    assert recover_checkpoint(directory, item_id, "asr") == _asr_result()
    assert path.exists()

    atomic_write_json(path, {"schema": 1, "id": "other", "result": _asr_result()})
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()

    path = _checkpoint(directory, item_id, {"language": "ja", "segments": []})
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated", encoding="utf-8")
    assert recover_checkpoint(directory, item_id, "asr") is None
    assert not path.exists()


def test_asr_batch_recovers_successful_item_after_worker_crash(tmp_path: Path, monkeypatch):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    source_id = "a" * 16
    audio = paths.cache / "audio" / f"{source_id}.flac"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"audio")
    sources = [{"source_id": source_id, "audio": audio}]

    class CrashingWorker:
        def call(self, _root, command, payload):
            assert command == "batch_transcribe"
            directory = Path(payload["checkpoint_dir"])
            _checkpoint(directory, source_id, _asr_result())
            atomic_write_json(
                directory / "progress.json",
                {
                    "schema": 1,
                    "worker": "asr",
                    "command": "batch_transcribe",
                    "phase": "transcribe",
                    "completed": 1,
                    "total": 1,
                    "failed": 0,
                    "current_id": None,
                    "state": "running",
                },
            )
            raise RuntimeError("simulated process crash")

    monkeypatch.setattr(pipeline, "worker", lambda _root, _name: CrashingWorker())
    with pytest.raises(RuntimeError, match="simulated process crash"):
        pipeline._batch_asr(tmp_path, paths, cfg, sources)

    partial = checkpoint_dir(paths.cache / "asr") / f"{source_id}.json"
    assert partial.is_file()

    class MustNotRun:
        def call(self, *_args, **_kwargs):
            raise AssertionError("recovered ASR item must not be recomputed")

    monkeypatch.setattr(pipeline, "worker", lambda _root, _name: MustNotRun())
    result = pipeline._batch_asr(tmp_path, paths, cfg, sources)
    assert result[source_id] == _asr_result()
    assert json.loads((paths.cache / "asr" / f"{source_id}.json").read_text(encoding="utf-8")) == _asr_result()
    assert not checkpoint_dir(paths.cache / "asr").exists()


def test_status_exposes_advisory_batch_progress(tmp_path: Path):
    paths = init_persona(tmp_path, "alice", authorized=True)
    cfg = PersonaConfig.load(paths.config)
    directory = checkpoint_dir(paths.cache / "asr")
    _checkpoint(directory, "a" * 16, _asr_result())
    atomic_write_json(
        directory / "progress.json",
        {
            "schema": 1,
            "worker": "asr",
            "command": "batch_transcribe",
            "phase": "transcribe",
            "completed": 3,
            "total": 10,
            "failed": 1,
            "current_id": "b" * 16,
            "state": "running",
        },
    )
    progress = prepare_batch_progress(paths.root)
    assert progress["asr"]["completed"] == 3
    assert progress["asr"]["checkpointed_successes"] == 1
    status = persona_status(tmp_path, paths, cfg)
    assert status["audit"]["prepare"]["batch_progress"]["asr"]["total"] == 10


def _load_worker(monkeypatch, name: str):
    if name == "asr":
        ctranslate2 = types.ModuleType("ctranslate2")
        ctranslate2.get_cuda_device_count = lambda: 0
        ctranslate2.get_supported_compute_types = lambda *_args, **_kwargs: {"float32"}
        monkeypatch.setitem(sys.modules, "ctranslate2", ctranslate2)
        faster = types.ModuleType("faster_whisper")
        faster.WhisperModel = object
        monkeypatch.setitem(sys.modules, "faster_whisper", faster)
        runtime_policy = types.ModuleType("runtime_policy")
        runtime_policy.choose_compute_type = lambda _device, _supported, _requested: "float32"
        monkeypatch.setitem(sys.modules, "runtime_policy", runtime_policy)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    elif name == "diarization":
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        torch.device = lambda value: value
        torch.float32 = object()
        monkeypatch.setitem(sys.modules, "torch", torch)
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        pyannote = types.ModuleType("pyannote")
        audio = types.ModuleType("pyannote.audio")
        audio.Pipeline = object
        pyannote.audio = audio
        monkeypatch.setitem(sys.modules, "pyannote", pyannote)
        monkeypatch.setitem(sys.modules, "pyannote.audio", audio)
    else:
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch)
        funasr = types.ModuleType("funasr")
        funasr.AutoModel = object
        monkeypatch.setitem(sys.modules, "funasr", funasr)
        modelscope = types.ModuleType("modelscope")
        modelscope.snapshot_download = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "modelscope", modelscope)

    worker_path = Path(__file__).parents[1] / "workers" / name / "worker.py"
    spec = importlib.util.spec_from_file_location(f"checkpoint_test_{name}", worker_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["asr", "diarization", "sense"])
def test_worker_checkpoint_path_is_confined_to_persona_cache(tmp_path: Path, monkeypatch, name: str):
    module = _load_worker(monkeypatch, name)
    monkeypatch.setenv("PERSONAVOICE_ROOT", str(tmp_path))
    valid = tmp_path / "personas" / "alice" / "cache" / name / ".checkpoints"
    resolved = module._checkpoint_directory({"checkpoint_dir": str(valid)})
    assert resolved == valid.resolve()

    with pytest.raises(ValueError, match="escapes PERSONAVOICE_ROOT"):
        module._checkpoint_directory({"checkpoint_dir": str(tmp_path.parent / "outside")})
    with pytest.raises(ValueError, match="Unsafe prepare checkpoint item id"):
        module._write_item_checkpoint(resolved, "../escape", {"ok": True})


def test_prepare_policy_migration_is_scoped_to_exact_new_generation():
    assert set(PREPARE_CACHE_POLICY_COMPATIBILITY) == {PREPARE_CACHE_POLICY_VERSION}
    previous = PREPARE_CACHE_POLICY_COMPATIBILITY[PREPARE_CACHE_POLICY_VERSION]
    assert previous
    assert PREPARE_CACHE_POLICY_VERSION not in previous
    assert all(_prepare_policy_compatible(value) for value in previous)
    assert not _prepare_policy_compatible("12-unrelated-old-policy")
''',
    encoding="utf-8",
    newline="\n",
)

print("Crash-resumable prepare batch patches applied")
