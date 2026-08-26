from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from personavoice.config import PersonaConfig
from personavoice.lfm_contract import LFM_CONTRACT_FINGERPRINT, LFM_CONTRACT_SCHEMA_VERSION
from personavoice.model_assets import (
    IRODORI_DACVAE_REVISION,
    IRODORI_DACVAE_SHA256,
    IRODORI_MODEL_REVISION,
    IRODORI_MODEL_SHA256,
    IRODORI_SOURCE_REVISION,
    IRODORI_TEXT_ENCODER_REVISION,
    LFM_MODEL_ASSET_SHA256,
    LFM_MODEL_REVISION,
    LFM_MODEL_WEIGHT_SHA256,
)
from personavoice.project import PersonaPaths
from personavoice.setup_env import SEED_VC_REVISION

TRAINING_PLAN_SCHEMA = 2
_HASH_CHUNK_BYTES = 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PERSONA_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
EXECUTOR_CONTRACT_FILES = (
    "src/personavoice/artifacts.py",
    "src/personavoice/config.py",
    "src/personavoice/environment.py",
    "src/personavoice/executors.py",
    "src/personavoice/modal_app.py",
    "src/personavoice/modal_transport.py",
    "src/personavoice/process.py",
    "src/personavoice/training.py",
    "src/personavoice/training_bundle.py",
    "src/personavoice/training_plan.py",
    "src/personavoice/workers.py",
)
EVALUATION_CONTRACT_FILES = (
    "src/personavoice/evaluation.py",
    "src/personavoice/evaluation_metrics.py",
    "src/personavoice/quality.py",
    "workers/lfm/worker.py",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-compatible values used by an immutable plan."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw(child) for child in value]
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_source_sha256(path: Path) -> str:
    """Hash source/lock contracts without checkout-specific line endings."""

    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        normalized = raw
    else:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")
    return _sha256_bytes(normalized)


def _portable_relative(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Training input escapes the persona root: {resolved}") from exc
    value = PurePosixPath(relative).as_posix()
    if not value or value.startswith("/") or ".." in PurePosixPath(value).parts:
        raise ValueError(f"Training input is not portable: {path}")
    return value


@dataclass(frozen=True)
class FileContract:
    path: str
    role: str
    sha256: str
    size: int
    transfer: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "sha256": self.sha256,
            "size": self.size,
            "transfer": self.transfer,
        }


@dataclass(frozen=True)
class FamilyPlan:
    family: str
    enabled: bool
    method: str
    dataset_fingerprint: str
    training: Mapping[str, Any]
    model_contract: Mapping[str, Any]
    implementation_contract: Mapping[str, str]
    checkpoint_policy: Mapping[str, Any]
    evaluation_policy: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "training", _freeze(self.training))
        object.__setattr__(self, "model_contract", _freeze(self.model_contract))
        object.__setattr__(self, "implementation_contract", _freeze(self.implementation_contract))
        object.__setattr__(self, "checkpoint_policy", _freeze(self.checkpoint_policy))
        object.__setattr__(self, "evaluation_policy", _freeze(self.evaluation_policy))

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "enabled": self.enabled,
            "method": self.method,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training": _thaw(self.training),
            "model_contract": _thaw(self.model_contract),
            "implementation_contract": _thaw(self.implementation_contract),
            "checkpoint_policy": _thaw(self.checkpoint_policy),
            "evaluation_policy": _thaw(self.evaluation_policy),
        }

    def _effective_training(self) -> dict[str, Any]:
        """Return only optimization knobs consumed by this family/method.

        ``as_dict()`` deliberately retains the complete requested configuration
        so the overall TrainingPlan changes whenever evaluation/publication must
        be reconsidered.  Native checkpoint compatibility is narrower: a LoRA
        rank cannot affect LFM full training, and an optional Irodori speaker
        embedding must not invalidate its primary full/LoRA optimization run.
        """

        training = _thaw(self.training)
        if self.family == "irodori":
            common = {
                name: training[name]
                for name in (
                    "conditioning",
                    "validation_ratio",
                    "validation_every",
                    "checkpoint_best_n",
                )
                if name in training
            }
            if self.method == "speaker-inversion":
                # The upstream Speaker Inversion trainer consumes its dedicated
                # budget, not the primary full/LoRA max_steps value.
                max_steps = training.get(
                    "speaker_inversion_max_steps",
                    training.get("max_steps"),
                )
            else:
                max_steps = training.get("max_steps")
            if max_steps is not None:
                common["max_steps"] = max_steps
            return common
        if self.family == "lfm":
            names = ["epochs", "learning_rate", "validation_ratio", "save_steps"]
            if self.method == "lora":
                names.extend(("lora_r", "lora_alpha"))
            return {name: training[name] for name in names if name in training}
        if self.family == "seed-vc":
            return {"max_steps": training["max_steps"]} if "max_steps" in training else {}
        # Strict plan parsing rejects unknown families.  Keeping this fallback
        # makes directly constructed test contracts fail conservatively.
        return training

    def _checkpoint_contract(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "enabled": self.enabled,
            "method": self.method,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training": self._effective_training(),
            "model_contract": _thaw(self.model_contract),
            "implementation_contract": _thaw(self.implementation_contract),
            "checkpoint_policy": _thaw(self.checkpoint_policy),
        }

    @property
    def fingerprint(self) -> str:
        # Publication thresholds do not change optimization semantics and must
        # not invalidate an otherwise resumable expensive checkpoint.
        return _sha256_bytes(_canonical_json(self._checkpoint_contract()).encode("utf-8"))

    @property
    def auxiliary_family(self) -> FamilyPlan | None:
        """Return the independently resumable auxiliary Speaker Inversion plan."""

        if not (
            self.family == "irodori"
            and self.enabled
            and self.method in {"full", "lora"}
            and self.training.get("auxiliary_speaker_inversion") is True
        ):
            return None
        training = _thaw(self.training)
        return FamilyPlan(
            family="irodori",
            enabled=True,
            method="speaker-inversion",
            dataset_fingerprint=self.dataset_fingerprint,
            training={
                "auxiliary_speaker_inversion": False,
                "max_steps": training["speaker_inversion_max_steps"],
                "speaker_inversion_max_steps": training["speaker_inversion_max_steps"],
                "conditioning": training["conditioning"],
                "validation_ratio": training["validation_ratio"],
                "validation_every": training["validation_every"],
                "checkpoint_best_n": training["checkpoint_best_n"],
            },
            model_contract=_thaw(self.model_contract),
            implementation_contract=_thaw(self.implementation_contract),
            checkpoint_policy=_thaw(self.checkpoint_policy),
            evaluation_policy=_thaw(self.evaluation_policy),
        )

    @property
    def auxiliary_fingerprint(self) -> str | None:
        auxiliary = self.auxiliary_family
        return auxiliary.fingerprint if auxiliary is not None else None


@dataclass(frozen=True)
class TrainingPlan:
    persona: str
    files: tuple[FileContract, ...]
    families: tuple[FamilyPlan, ...]
    executor_contract: Mapping[str, str] = field(default_factory=dict)
    schema_version: int = TRAINING_PLAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "executor_contract", _freeze(self.executor_contract))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "persona": self.persona,
            "files": [item.as_dict() for item in self.files],
            "families": [item.as_dict() for item in self.families],
            "executor_contract": _thaw(self.executor_contract),
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_bytes(_canonical_json(self.as_dict()).encode("utf-8"))

    @property
    def plan_id(self) -> str:
        return self.fingerprint[:24]

    def family(self, name: str) -> FamilyPlan:
        for family in self.families:
            if family.family == name:
                return family
        raise KeyError(name)

    @classmethod
    def from_dict(cls, value: Any) -> TrainingPlan:
        """Strictly reconstruct a plan received across an executor boundary."""

        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "persona",
            "files",
            "families",
            "executor_contract",
        }:
            raise ValueError("TrainingPlan root schema is invalid")
        if value.get("schema_version") != TRAINING_PLAN_SCHEMA:
            raise ValueError("TrainingPlan schema_version is unsupported")
        persona = value.get("persona")
        if not isinstance(persona, str) or _PERSONA_RE.fullmatch(persona) is None:
            raise ValueError("TrainingPlan persona is invalid")

        raw_files = value.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("TrainingPlan files must be a list")
        files: list[FileContract] = []
        seen_paths: set[str] = set()
        for raw in raw_files:
            if not isinstance(raw, Mapping) or set(raw) != {
                "path",
                "role",
                "sha256",
                "size",
                "transfer",
            }:
                raise ValueError("TrainingPlan file contract is invalid")
            path = raw.get("path")
            role = raw.get("role")
            digest = raw.get("sha256")
            size = raw.get("size")
            transfer = raw.get("transfer")
            if not isinstance(path, str):
                raise ValueError("TrainingPlan file path is invalid")
            relative = PurePosixPath(path)
            if (
                not path
                or "\\" in path
                or relative.is_absolute()
                or relative.as_posix() != path
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or path in seen_paths
            ):
                raise ValueError("TrainingPlan file path is unsafe or duplicated")
            if not isinstance(role, str) or not role:
                raise ValueError("TrainingPlan file role is invalid")
            if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise ValueError("TrainingPlan file checksum is invalid")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError("TrainingPlan file size is invalid")
            if not isinstance(transfer, bool):
                raise ValueError("TrainingPlan file transfer flag is invalid")
            seen_paths.add(path)
            files.append(
                FileContract(
                    path=path,
                    role=role,
                    sha256=digest,
                    size=size,
                    transfer=transfer,
                )
            )
        if files != sorted(files, key=lambda item: (item.role, item.path)):
            raise ValueError("TrainingPlan file contracts are not canonically ordered")

        raw_executor_contract = value.get("executor_contract")
        if not isinstance(raw_executor_contract, Mapping):
            raise ValueError("TrainingPlan executor contract is invalid")
        if set(raw_executor_contract) != set(EXECUTOR_CONTRACT_FILES):
            raise ValueError("TrainingPlan executor contract is incomplete")
        executor_contract: dict[str, str] = {}
        for raw_path, digest in raw_executor_contract.items():
            if not isinstance(raw_path, str) or not isinstance(digest, str):
                raise ValueError("TrainingPlan executor implementation entry is invalid")
            relative = PurePosixPath(raw_path)
            if (
                not raw_path
                or "\\" in raw_path
                or relative.is_absolute()
                or relative.as_posix() != raw_path
                or any(part in {"", ".", ".."} for part in raw_path.split("/"))
                or _SHA256_RE.fullmatch(digest) is None
            ):
                raise ValueError("TrainingPlan executor implementation contract is unsafe")
            executor_contract[raw_path] = digest

        raw_families = value.get("families")
        if not isinstance(raw_families, list) or not raw_families:
            raise ValueError("TrainingPlan families must be a non-empty list")
        families: list[FamilyPlan] = []
        seen_families: set[str] = set()
        allowed_methods = {
            "irodori": {"full", "lora", "speaker-inversion"},
            "lfm": {"full", "lora"},
            "seed-vc": {"finetune"},
        }
        family_keys = {
            "family",
            "enabled",
            "method",
            "dataset_fingerprint",
            "training",
            "model_contract",
            "implementation_contract",
            "checkpoint_policy",
            "evaluation_policy",
        }
        for raw in raw_families:
            if not isinstance(raw, Mapping) or set(raw) != family_keys:
                raise ValueError("TrainingPlan family contract is invalid")
            family = raw.get("family")
            method = raw.get("method")
            enabled = raw.get("enabled")
            dataset_fingerprint = raw.get("dataset_fingerprint")
            if (
                not isinstance(family, str)
                or family not in allowed_methods
                or family in seen_families
            ):
                raise ValueError("TrainingPlan family is unknown or duplicated")
            if not isinstance(method, str) or method not in allowed_methods[family]:
                raise ValueError(f"TrainingPlan method is invalid for {family}")
            if not isinstance(enabled, bool):
                raise ValueError(f"TrainingPlan enabled flag is invalid for {family}")
            if (
                not isinstance(dataset_fingerprint, str)
                or _SHA256_RE.fullmatch(dataset_fingerprint) is None
            ):
                raise ValueError(f"TrainingPlan dataset fingerprint is invalid for {family}")
            mappings = (
                "training",
                "model_contract",
                "implementation_contract",
                "checkpoint_policy",
                "evaluation_policy",
            )
            if any(not isinstance(raw.get(name), Mapping) for name in mappings):
                raise ValueError(f"TrainingPlan mappings are invalid for {family}")
            seen_families.add(family)
            families.append(
                FamilyPlan(
                    family=family,
                    enabled=enabled,
                    method=method,
                    dataset_fingerprint=dataset_fingerprint,
                    training=dict(raw["training"]),
                    model_contract=dict(raw["model_contract"]),
                    implementation_contract=dict(raw["implementation_contract"]),
                    checkpoint_policy=dict(raw["checkpoint_policy"]),
                    evaluation_policy=dict(raw["evaluation_policy"]),
                )
            )
        if seen_families != set(allowed_methods):
            raise ValueError("TrainingPlan must contain exactly irodori, lfm, and seed-vc")
        rebuilt = cls(
            persona=persona,
            files=tuple(files),
            families=tuple(families),
            executor_contract=executor_contract,
        )
        if rebuilt.as_dict() != value:
            raise ValueError("TrainingPlan is not in canonical form")
        return rebuilt

    @classmethod
    def from_bytes(cls, value: bytes) -> TrainingPlan:
        """Parse canonical UTF-8 JSON and reject semantically equivalent rewrites."""

        if not isinstance(value, bytes) or not value:
            raise ValueError("TrainingPlan bytes are missing")
        try:
            decoded = value.decode("utf-8")
            raw = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("TrainingPlan is not valid canonical UTF-8 JSON") from exc
        plan = cls.from_dict(raw)
        canonical = _canonical_json(plan.as_dict()).encode("utf-8")
        if value != canonical:
            raise ValueError("TrainingPlan JSON bytes are not canonical")
        return plan


def _file_contract(
    path: Path,
    *,
    root: Path,
    role: str,
    transfer: bool = True,
) -> FileContract:
    if not path.is_file():
        raise FileNotFoundError(f"Required training input is missing: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise ValueError(f"Required training input is empty: {path}")
    return FileContract(
        path=_portable_relative(path, root),
        role=role,
        sha256=sha256_file(path),
        size=size,
        transfer=transfer,
    )


def _optional_file_contract(
    path: Path,
    *,
    root: Path,
    role: str,
    transfer: bool = True,
) -> FileContract | None:
    return _file_contract(path, root=root, role=role, transfer=transfer) if path.is_file() else None


def _dataset_fingerprint(files: list[FileContract], roles: set[str]) -> str:
    payload = [item.as_dict() for item in files if item.role in roles]
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))


def _implementation_contract(repo_root: Path, relatives: tuple[str, ...]) -> dict[str, str]:
    contract: dict[str, str] = {}
    for relative in relatives:
        path = repo_root / relative
        contract[PurePosixPath(relative).as_posix()] = (
            normalized_source_sha256(path) if path.is_file() else "missing"
        )
    return contract


def build_training_plan(
    repo_root: Path,
    paths: PersonaPaths,
    cfg: PersonaConfig,
    *,
    irodori_manifest: Path,
) -> TrainingPlan:
    """Build the executor-independent, portable semantic training contract.

    Routing, credentials, local hardware and remote authorization are deliberately
    absent. Changing where an identical plan runs must never invalidate datasets or
    checkpoints.
    """

    file_items: list[FileContract] = []
    if cfg.training.irodori.enabled:
        file_items.extend(
            [
                _file_contract(
                    paths.dataset / "irodori_source.jsonl",
                    root=paths.root,
                    role="irodori-source",
                    transfer=False,
                ),
                _file_contract(
                    irodori_manifest,
                    root=paths.root,
                    role="irodori-latent-manifest",
                ),
            ]
        )
    enabled_inputs: list[tuple[Path, str]] = []
    if cfg.training.lfm.enabled:
        enabled_inputs.append((paths.dataset / "lfm_train.jsonl", "lfm-conversations"))
    if cfg.training.seed_vc.finetune:
        enabled_inputs.append((paths.dataset / "seed_vc" / "manifest.jsonl", "seed-vc-manifest"))
    for path, role in enabled_inputs:
        file_items.append(
            _file_contract(
                path,
                root=paths.root,
                role=role,
                transfer=role != "seed-vc-manifest",
            )
        )
    file_items.sort(key=lambda item: (item.role, item.path))

    quality_policy = cfg.training.quality_gate.model_dump(mode="json")
    quality_policy["implementation_contract"] = _implementation_contract(
        repo_root,
        EVALUATION_CONTRACT_FILES,
    )
    irodori_cfg = cfg.training.irodori.model_dump(mode="json")
    lfm_cfg = cfg.training.lfm.model_dump(mode="json")
    seed_cfg = cfg.training.seed_vc.model_dump(mode="json")

    irodori = FamilyPlan(
        family="irodori",
        enabled=bool(irodori_cfg.pop("enabled")),
        method=str(irodori_cfg.pop("method")),
        dataset_fingerprint=_dataset_fingerprint(
            file_items,
            {"irodori-source", "irodori-latent-manifest"},
        ),
        training=irodori_cfg,
        model_contract={
            "source_revision": IRODORI_SOURCE_REVISION,
            "base_revision": IRODORI_MODEL_REVISION,
            "base_sha256": IRODORI_MODEL_SHA256,
            "dacvae_revision": IRODORI_DACVAE_REVISION,
            "dacvae_sha256": IRODORI_DACVAE_SHA256,
            "text_encoder_revision": IRODORI_TEXT_ENCODER_REVISION,
        },
        implementation_contract=_implementation_contract(
            repo_root,
            (
                "src/personavoice/irodori.py",
                "locks/Irodori-TTS.pyproject.toml",
                "locks/Irodori-TTS.uv.lock",
            ),
        ),
        checkpoint_policy={
            "periodic": True,
            "resume_complete_only": True,
            "select": "best-validation-loss",
            "portable_format": "safetensors",
        },
        evaluation_policy=quality_policy,
    )
    lfm = FamilyPlan(
        family="lfm",
        enabled=bool(lfm_cfg.pop("enabled")),
        method=str(lfm_cfg.pop("method")),
        dataset_fingerprint=_dataset_fingerprint(file_items, {"lfm-conversations"}),
        training=lfm_cfg,
        model_contract={
            "base_revision": LFM_MODEL_REVISION,
            "base_sha256": LFM_MODEL_WEIGHT_SHA256,
            "base_assets_sha256": LFM_MODEL_ASSET_SHA256,
            "runtime_contract_schema_version": LFM_CONTRACT_SCHEMA_VERSION,
            "runtime_contract_fingerprint": LFM_CONTRACT_FINGERPRINT,
        },
        implementation_contract=_implementation_contract(
            repo_root,
            (
                "workers/lfm/train.py",
                "workers/lfm/checkpoint_contract.py",
                "workers/lfm/model_contract.py",
                "workers/lfm/pyproject.toml",
                "workers/lfm/uv.lock",
            ),
        ),
        checkpoint_policy={
            "periodic": True,
            "resume_complete_only": True,
            "select": "best-validation-loss",
            "portable_format": "safetensors",
        },
        evaluation_policy=quality_policy,
    )
    seed_vc = FamilyPlan(
        family="seed-vc",
        enabled=bool(seed_cfg.pop("finetune")),
        method="finetune",
        dataset_fingerprint=_dataset_fingerprint(file_items, {"seed-vc-manifest"}),
        training=seed_cfg,
        model_contract={"source_revision": SEED_VC_REVISION},
        implementation_contract=_implementation_contract(
            repo_root,
            (
                "workers/seed_vc/worker.py",
                "workers/seed_vc/pyproject.toml",
                "workers/seed_vc/uv.lock",
            ),
        ),
        checkpoint_policy={
            "periodic": True,
            "resume_complete_only": True,
            "select": "latest-step",
            "portable_format": "pth",
        },
        evaluation_policy={},
    )
    return TrainingPlan(
        persona=cfg.name,
        files=tuple(file_items),
        families=(irodori, lfm, seed_vc),
        executor_contract=_implementation_contract(
            repo_root,
            EXECUTOR_CONTRACT_FILES,
        ),
    )


def verify_plan_files(
    plan: TrainingPlan,
    persona_root: Path,
    *,
    transferred_only: bool = False,
) -> None:
    """Fail closed if any local/bundled dataset member differs from the plan."""

    root = persona_root.resolve()
    for item in plan.files:
        if transferred_only and not item.transfer:
            continue
        relative = PurePosixPath(item.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Training plan contains a non-portable path: {item.path}")
        path = root.joinpath(*relative.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Training plan path escapes the persona root: {item.path}") from exc
        if not path.is_file() or path.stat().st_size != item.size:
            raise RuntimeError(f"Training plan file size contract failed: {item.path}")
        actual = sha256_file(path)
        if actual != item.sha256:
            raise RuntimeError(
                f"Training plan file checksum failed: {item.path}; "
                f"expected {item.sha256}, got {actual}"
            )
