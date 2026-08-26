from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from personavoice.atomic import atomic_write_text


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ConsentConfig(StrictConfigModel):
    authorized: bool = False
    scope: str = "local-private"
    notes: str = ""


class PrepareConfig(StrictConfigModel):
    language: str = Field(default="ja", min_length=1)
    # PersonaVoice setup/cache contracts audit and materialize exactly this ASR model.
    # Allowing an arbitrary model name here could silently bypass those guarantees.
    asr_model: Literal["large-v3"] = "large-v3"
    asr_compute_type: str = Field(default="auto", min_length=1)
    min_clip_seconds: float = Field(default=1.0, gt=0)
    max_clip_seconds: float = Field(default=18.0, gt=0)
    merge_gap_seconds: float = Field(default=0.45, ge=0)
    max_overlap_ratio: float = Field(default=0.08, ge=0, le=1)
    min_identity_similarity: float = Field(default=0.45, ge=-1, le=1)
    # Pinned Irodori v4.1 Small supports a combined reference window up to 120 s.
    reference_seconds: float = Field(default=40.0, gt=0, le=120.0)
    reference_clip_max_seconds: float = Field(default=12.0, gt=0)
    keep_nonverbal_only: bool = True
    use_sensevoice: bool = True

    @model_validator(mode="after")
    def validate_clip_ranges(self) -> Self:
        if self.max_clip_seconds < self.min_clip_seconds:
            raise ValueError("prepare.max_clip_seconds must be >= prepare.min_clip_seconds")
        if self.reference_clip_max_seconds < self.min_clip_seconds:
            raise ValueError(
                "prepare.reference_clip_max_seconds must be >= prepare.min_clip_seconds"
            )
        return self


class IrodoriTrainingConfig(StrictConfigModel):
    enabled: bool = True
    method: Literal["full", "lora", "speaker-inversion"] = "full"
    auxiliary_speaker_inversion: bool = False
    max_steps: int = Field(default=4000, ge=1)
    speaker_inversion_max_steps: int = Field(default=2000, ge=1)
    conditioning: Literal["speaker", "none"] = "speaker"
    validation_ratio: float = Field(default=0.0005, gt=0, lt=1)
    validation_every: int = Field(default=1000, ge=1)
    checkpoint_best_n: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def validate_auxiliary_method(self) -> Self:
        if self.method == "speaker-inversion" and self.auxiliary_speaker_inversion:
            raise ValueError(
                "training.irodori.auxiliary_speaker_inversion must be false when "
                "training.irodori.method is speaker-inversion"
            )
        return self


class LFMTrainingConfig(StrictConfigModel):
    enabled: bool = True
    method: Literal["full", "lora"] = "full"
    epochs: float = Field(default=3.0, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    validation_ratio: float = Field(default=0.1, gt=0, lt=1)
    save_steps: int = Field(default=25, ge=1)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)


class SeedVCTrainingConfig(StrictConfigModel):
    finetune: bool = False
    max_steps: int = Field(default=1000, ge=1)


class QualityGateConfig(StrictConfigModel):
    enabled: bool = True
    require_validation: bool = True
    min_speaker_similarity: float = Field(default=0.45, ge=0, le=1)
    max_cer: float = Field(default=0.25, ge=0, le=1)
    max_wer: float = Field(default=0.50, ge=0, le=1)
    min_emotion_accuracy: float = Field(default=0.40, ge=0, le=1)
    min_unseen_text_similarity: float = Field(default=0.75, ge=0, le=1)
    max_duration_ratio_error: float = Field(default=0.50, ge=0, le=1)
    max_base_cer_regression: float = Field(default=0.10, ge=0, le=1)
    min_lfm_expected_similarity: float = Field(default=0.35, ge=0, le=1)
    max_lfm_expected_cer: float = Field(default=0.85, ge=0, le=2)
    max_lfm_expected_wer: float = Field(default=1.00, ge=0, le=2)
    min_lfm_required_phrase_coverage: float = Field(default=1.00, ge=0, le=1)
    max_lfm_base_similarity_regression: float = Field(default=0.10, ge=0, le=1)


class _LegacyTrainingConfig(StrictConfigModel):
    """The complete v0.3.0 training shape used only for lossless migration."""

    irodori_speaker_inversion: bool = True
    irodori_lora: bool = True
    lfm_lora: bool = True
    seed_vc_finetune: bool = False
    irodori_max_steps: int = Field(default=4000, ge=1)
    speaker_inversion_max_steps: int = Field(default=2000, ge=1)
    lfm_epochs: float = Field(default=3.0, gt=0)
    lfm_learning_rate: float = Field(default=2e-4, gt=0)
    lfm_lora_r: int = Field(default=16, ge=1)
    lfm_lora_alpha: int = Field(default=32, ge=1)
    seed_vc_max_steps: int = Field(default=1000, ge=1)


class TrainingConfig(StrictConfigModel):
    SCHEMA_VERSION: ClassVar[int] = 2
    LEGACY_SCHEMA_VERSION: ClassVar[int] = 1
    LEGACY_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "irodori_speaker_inversion",
            "irodori_lora",
            "lfm_lora",
            "seed_vc_finetune",
            "irodori_max_steps",
            "speaker_inversion_max_steps",
            "lfm_epochs",
            "lfm_learning_rate",
            "lfm_lora_r",
            "lfm_lora_alpha",
            "seed_vc_max_steps",
        }
    )
    NESTED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "executor",
            "remote_data_authorized",
            "irodori",
            "lfm",
            "seed_vc",
            "quality_gate",
        }
    )

    schema_version: Literal[2] = 2
    executor: Literal["auto", "local", "modal"] = "auto"
    remote_data_authorized: bool = False
    irodori: IrodoriTrainingConfig = Field(default_factory=IrodoriTrainingConfig)
    lfm: LFMTrainingConfig = Field(default_factory=LFMTrainingConfig)
    seed_vc: SeedVCTrainingConfig = Field(default_factory=SeedVCTrainingConfig)
    quality_gate: QualityGateConfig = Field(default_factory=QualityGateConfig)
    migration_notes: tuple[str, ...] = Field(
        default=(),
        exclude=True,
        frozen=True,
        repr=False,
    )
    migrated_from_schema_version: int | None = Field(
        default=None,
        exclude=True,
        frozen=True,
        repr=False,
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_shape(cls, value: Any, info: ValidationInfo) -> Any:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            return value

        raw = dict(value)
        # Pydantic re-runs model-level validation after field assignment and
        # supplies the complete internal field mapping in that path. Excluded
        # migration metadata is therefore legitimate only when ``field_name``
        # identifies assignment validation; callers may never inject it while
        # constructing a config.
        assignment_validation = info.field_name in cls.model_fields
        if not assignment_validation and (
            "migration_notes" in raw or "migrated_from_schema_version" in raw
        ):
            raise ValueError("training migration metadata is read-only")

        raw_version = raw.get("schema_version")
        if raw_version is not None and type(raw_version) is not int:
            raise ValueError(
                f"unsupported training schema_version {raw_version!r}; expected integer 1 or 2"
            )
        legacy_keys = cls.LEGACY_KEYS.intersection(raw)
        nested_keys = cls.NESTED_KEYS.intersection(raw)
        explicitly_legacy = raw_version == cls.LEGACY_SCHEMA_VERSION
        explicitly_current = raw_version == cls.SCHEMA_VERSION

        if legacy_keys and (nested_keys or explicitly_current):
            conflicts = sorted(legacy_keys | nested_keys | {"schema_version"})
            raise ValueError(
                "training config mixes legacy flat fields with schema v2 fields: "
                + ", ".join(conflicts)
            )
        if explicitly_legacy and nested_keys:
            raise ValueError(
                "training schema_version 1 cannot contain schema v2 fields: "
                + ", ".join(sorted(nested_keys))
            )
        if raw_version is not None and not explicitly_legacy and not explicitly_current:
            raise ValueError(
                f"unsupported training schema_version {raw_version!r}; expected 1 or 2"
            )

        if not legacy_keys and not explicitly_legacy:
            return raw

        # Pass every non-version key through the strict legacy model so typos
        # cannot disappear as a side effect of migration.
        legacy_payload = {key: item for key, item in raw.items() if key != "schema_version"}
        legacy = _LegacyTrainingConfig.model_validate(legacy_payload)

        if legacy.irodori_lora:
            irodori_enabled = True
            irodori_method = "lora"
            auxiliary_speaker_inversion = legacy.irodori_speaker_inversion
        elif legacy.irodori_speaker_inversion:
            irodori_enabled = True
            irodori_method = "speaker-inversion"
            auxiliary_speaker_inversion = False
        else:
            irodori_enabled = False
            irodori_method = "lora"
            auxiliary_speaker_inversion = False

        notes = [
            "Migrated the v0.3.0 flat training configuration to training schema_version 2.",
            (
                "Preserved Irodori behavior as "
                f"enabled={str(irodori_enabled).lower()}, method={irodori_method}, "
                "auxiliary_speaker_inversion="
                f"{str(auxiliary_speaker_inversion).lower()}."
            ),
            (f"Preserved LFM behavior as enabled={str(legacy.lfm_lora).lower()}, method=lora."),
            (f"Preserved Seed-VC behavior as finetune={str(legacy.seed_vc_finetune).lower()}."),
        ]
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "executor": "auto",
            "remote_data_authorized": False,
            "irodori": {
                "enabled": irodori_enabled,
                "method": irodori_method,
                "auxiliary_speaker_inversion": auxiliary_speaker_inversion,
                "max_steps": legacy.irodori_max_steps,
                "speaker_inversion_max_steps": legacy.speaker_inversion_max_steps,
            },
            "lfm": {
                "enabled": legacy.lfm_lora,
                "method": "lora",
                "epochs": legacy.lfm_epochs,
                "learning_rate": legacy.lfm_learning_rate,
                "lora_r": legacy.lfm_lora_r,
                "lora_alpha": legacy.lfm_lora_alpha,
            },
            "seed_vc": {
                "finetune": legacy.seed_vc_finetune,
                "max_steps": legacy.seed_vc_max_steps,
            },
            "quality_gate": QualityGateConfig().model_dump(mode="json"),
            "migration_notes": tuple(notes),
            "migrated_from_schema_version": cls.LEGACY_SCHEMA_VERSION,
        }

    def canonical_dict(self) -> dict[str, Any]:
        """Return the strict schema-v2 payload without transient migration metadata."""

        return self.model_dump(mode="json")

    @property
    def was_migrated(self) -> bool:
        return self.migrated_from_schema_version is not None

    # Read/write aliases keep the v0.3.0 Python API usable while callers move to
    # the nested schema. They are not accepted as schema-v2 input fields and are
    # never emitted by canonical serialization.
    @property
    def irodori_speaker_inversion(self) -> bool:
        return self.irodori.enabled and (
            self.irodori.method == "speaker-inversion" or self.irodori.auxiliary_speaker_inversion
        )

    @property
    def irodori_lora(self) -> bool:
        return self.irodori.enabled and self.irodori.method == "lora"

    @property
    def lfm_lora(self) -> bool:
        return self.lfm.enabled and self.lfm.method == "lora"

    @property
    def seed_vc_finetune(self) -> bool:
        return self.seed_vc.finetune

    @seed_vc_finetune.setter
    def seed_vc_finetune(self, value: bool) -> None:
        self.seed_vc.finetune = value

    @property
    def irodori_max_steps(self) -> int:
        return self.irodori.max_steps

    @property
    def speaker_inversion_max_steps(self) -> int:
        return self.irodori.speaker_inversion_max_steps

    @property
    def lfm_epochs(self) -> float:
        return self.lfm.epochs

    @property
    def lfm_learning_rate(self) -> float:
        return self.lfm.learning_rate

    @property
    def lfm_lora_r(self) -> int:
        return self.lfm.lora_r

    @property
    def lfm_lora_alpha(self) -> int:
        return self.lfm.lora_alpha

    @property
    def seed_vc_max_steps(self) -> int:
        return self.seed_vc.max_steps

    @seed_vc_max_steps.setter
    def seed_vc_max_steps(self, value: int) -> None:
        self.seed_vc.max_steps = value


class InferenceConfig(StrictConfigModel):
    default_candidates: int = Field(default=3, ge=1, le=16)
    default_num_steps: int = Field(default=24, ge=1, le=500)
    tts_cfg_scale: float = Field(default=3.0, ge=0, le=100)
    reference_mode: Literal["auto", "none", "speaker-embed", "audio"] = "auto"
    # These values are intentionally part of the inference contract rather
    # than being inherited from the pinned upstream CLI defaults.  Changing
    # them affects generated audio only; they are not training inputs and are
    # therefore deliberately excluded from prepare/training fingerprints.
    duration_scale: float = Field(default=1.0, gt=0, le=4)
    trim_tail: bool = True
    tail_window_size: int = Field(default=20, ge=1, le=4096)
    tail_std_threshold: float = Field(default=0.05, ge=0, le=10)
    tail_mean_threshold: float = Field(default=0.1, ge=0, le=10)
    seed_vc_diffusion_steps: int = Field(default=30, ge=1, le=500)
    seed_vc_similarity_cfg: float = Field(default=0.7, ge=0)
    seed_vc_intelligibility_cfg: float = Field(default=0.7, ge=0)
    # Vevo2's initial supported path is the upstream FM-only,
    # style-preserved VC pipeline.  The worker selects CPU/CUDA from the
    # audited setup state; it never silently changes the requested dtype.
    vevo2_flow_matching_steps: int = Field(default=32, ge=1, le=500)
    vevo2_use_pitch_shift: bool = False
    vevo2_dtype: Literal["fp32", "fp16"] = "fp32"


class VCEvaluationConfig(StrictConfigModel):
    """Canonical Seed-VC/Vevo2 comparison policy.

    ``sample_count`` is intentionally allowed below the recommended 100 clip
    target so a small prepared persona can produce a useful dry report. The
    report marks that run as underpowered; it never turns a small run into a
    quality acceptance.
    """

    schema_version: Literal[1] = 1
    sample_count: int = Field(default=200, ge=1, le=300)
    seed: int = 20260827
    max_cer_regression: float = Field(default=0.05, ge=0, le=1)
    max_speaker_similarity_regression: float = Field(default=0.03, ge=0, le=1)
    max_duration_ratio_error_regression: float = Field(default=0.05, ge=0, le=1)
    max_f0_correlation_regression: float = Field(default=0.10, ge=0, le=1)
    max_voiced_unvoiced_f1_regression: float = Field(default=0.05, ge=0, le=1)
    max_pause_ratio_error_regression: float = Field(default=0.05, ge=0, le=1)
    max_nonverbal_event_regression: float = Field(default=0.05, ge=0, le=1)
    require_human_review: bool = True


class PersonaConfig(StrictConfigModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    language: str = Field(default="ja", min_length=1)
    consent: ConsentConfig = Field(default_factory=ConsentConfig)
    prepare: PrepareConfig = Field(default_factory=PrepareConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    vc_evaluation: VCEvaluationConfig = Field(default_factory=VCEvaluationConfig)
    tts_backend: Literal["irodori-v4.1-small"] = "irodori-v4.1-small"
    # Seed-VC remains the default until a completed Japanese/non-verbal gate
    # proves that Vevo2 is at least as safe for this persona.
    vc_backend: Literal["seed-vc-v2", "vevo2-fm"] = "seed-vc-v2"
    brain_backend: Literal["lfm2.5-1.2b-jp-202606"] = "lfm2.5-1.2b-jp-202606"

    @model_validator(mode="after")
    def validate_language_consistency(self) -> Self:
        if self.prepare.language != self.language:
            raise ValueError(
                "persona language and prepare.language must match; use one consistent language"
            )
        return self

    @classmethod
    def load(cls, path: Path) -> PersonaConfig:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Persona config must contain a YAML mapping: {path}")
        config = cls.model_validate(value)
        # A persona.yaml is namespaced by its directory. Allowing the embedded
        # name to drift would change speaker labels/model run names while the
        # command still targets a different persona directory.
        if path.name == "persona.yaml" and path.parent.name != config.name:
            raise ValueError(
                f"Persona config name {config.name!r} does not match directory "
                f"{path.parent.name!r}: {path}"
            )
        return config

    @property
    def migration_notes(self) -> tuple[str, ...]:
        return self.training.migration_notes

    @property
    def was_migrated(self) -> bool:
        return self.training.was_migrated

    def migrated_yaml(self) -> str:
        """Serialize the current canonical schema without mutating the source file."""

        return yaml.safe_dump(
            self.model_dump(mode="json"),
            allow_unicode=True,
            sort_keys=False,
        )

    def save_migrated(self, path: Path) -> tuple[str, ...]:
        """Atomically publish canonical YAML and return value-free migration notes."""

        notes = self.migration_notes
        atomic_write_text(path, self.migrated_yaml())
        return notes

    def save(self, path: Path) -> None:
        atomic_write_text(path, self.migrated_yaml())
