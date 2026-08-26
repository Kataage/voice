from __future__ import annotations

import os
import re
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ENV_ALLOWLIST = frozenset(
    {
        "HF_TOKEN",
        "MODAL_TOKEN_ID",
        "MODAL_TOKEN_SECRET",
        "MODAL_ENVIRONMENT",
        "PERSONAVOICE_MODAL_APP",
        "PERSONAVOICE_MODAL_FUNCTION",
        "PERSONAVOICE_MODAL_VOLUME",
        "PERSONAVOICE_MODAL_GPU",
        "PERSONAVOICE_MODAL_HF_SECRET",
        "PERSONAVOICE_MODAL_TIMEOUT_SECONDS",
        "PERSONAVOICE_MODAL_RETRIES",
        "PERSONAVOICE_MODAL_POLL_SECONDS",
    }
)

SECRET_ENV_KEYS = frozenset({"HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"})

# These are operational names and limits, not credentials. Environment variables
# supplied by the parent process or root .env always take precedence.
NONSECRET_ENV_DEFAULTS = MappingProxyType(
    {
        "PERSONAVOICE_MODAL_APP": "personavoice-training",
        "PERSONAVOICE_MODAL_FUNCTION": "train",
        "PERSONAVOICE_MODAL_VOLUME": "personavoice-training",
        "PERSONAVOICE_MODAL_GPU": "A100-40GB",
        "PERSONAVOICE_MODAL_HF_SECRET": "personavoice-huggingface",
        "PERSONAVOICE_MODAL_TIMEOUT_SECONDS": "86400",
        "PERSONAVOICE_MODAL_RETRIES": "2",
        "PERSONAVOICE_MODAL_POLL_SECONDS": "10",
    }
)

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentFileError(ValueError):
    """Raised for a malformed allowlisted entry without exposing its value."""


@dataclass(frozen=True)
class EnvironmentLoadReport:
    """Value-free audit information for one explicit root .env load."""

    env_file: Path
    file_found: bool
    loaded_from_file: tuple[str, ...]
    defaults_applied: tuple[str, ...]
    existing_preserved: tuple[str, ...]
    ignored_keys: tuple[str, ...]
    empty_keys: tuple[str, ...]


def _double_quoted_value(value: str, *, key: str, line_number: int) -> tuple[str, str]:
    output: list[str] = []
    escaped = False
    escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"'}
    for index, character in enumerate(value[1:], start=1):
        if escaped:
            output.append(escapes.get(character, f"\\{character}"))
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            return "".join(output), value[index + 1 :]
        output.append(character)
    raise EnvironmentFileError(f"Unterminated quoted .env value for {key!r} on line {line_number}")


def _single_quoted_value(value: str, *, key: str, line_number: int) -> tuple[str, str]:
    closing = value.find("'", 1)
    if closing < 0:
        raise EnvironmentFileError(
            f"Unterminated quoted .env value for {key!r} on line {line_number}"
        )
    return value[1:closing], value[closing + 1 :]


def _parse_env_value(raw: str, *, key: str, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value.startswith('"'):
        parsed, trailing = _double_quoted_value(value, key=key, line_number=line_number)
    elif value.startswith("'"):
        parsed, trailing = _single_quoted_value(value, key=key, line_number=line_number)
    else:
        comment = next(
            (
                index
                for index, character in enumerate(value)
                if character == "#" and index > 0 and value[index - 1].isspace()
            ),
            None,
        )
        return value[:comment].rstrip() if comment is not None else value

    trailing = trailing.strip()
    if trailing and not trailing.startswith("#"):
        raise EnvironmentFileError(
            f"Unexpected text after quoted .env value for {key!r} on line {line_number}"
        )
    return parsed


def _read_allowlisted_env(path: Path) -> tuple[dict[str, str], set[str]]:
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise EnvironmentFileError(f"Unable to read root environment file: {path}") from exc

    values: dict[str, str] = {}
    ignored: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise EnvironmentFileError(f"Malformed .env assignment on line {line_number}")
        raw_key, raw_value = line.split("=", 1)
        key = raw_key.strip()
        if not _ENV_KEY.fullmatch(key):
            raise EnvironmentFileError(f"Invalid .env key on line {line_number}")
        if key not in ENV_ALLOWLIST:
            ignored.add(key)
            continue
        values[key] = _parse_env_value(raw_value, key=key, line_number=line_number)
    return values, ignored


def load_root_environment(
    repo_root: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    apply_defaults: bool = True,
) -> EnvironmentLoadReport:
    """Load only approved keys from ``repo_root/.env`` without overriding callers.

    Values are applied directly to the provided environment mapping and are
    intentionally absent from the return value, exceptions, and logging. There
    is no interpolation or command expansion. Pre-existing process values have
    highest priority, followed by .env, followed by non-secret defaults.
    """

    target_environment = os.environ if environ is None else environ
    env_file = repo_root.resolve() / ".env"
    file_found = env_file.is_file()
    values: dict[str, str] = {}
    ignored: set[str] = set()
    if file_found:
        values, ignored = _read_allowlisted_env(env_file)

    loaded: set[str] = set()
    preserved: set[str] = set()
    empty: set[str] = set()
    for key, value in values.items():
        if key in target_environment:
            preserved.add(key)
        elif value:
            target_environment[key] = value
            loaded.add(key)
        else:
            empty.add(key)

    defaulted: set[str] = set()
    if apply_defaults:
        for key, value in NONSECRET_ENV_DEFAULTS.items():
            if key in target_environment:
                if key not in loaded:
                    preserved.add(key)
                continue
            target_environment[key] = value
            defaulted.add(key)

    return EnvironmentLoadReport(
        env_file=env_file,
        file_found=file_found,
        loaded_from_file=tuple(sorted(loaded)),
        defaults_applied=tuple(sorted(defaulted)),
        existing_preserved=tuple(sorted(preserved)),
        ignored_keys=tuple(sorted(ignored)),
        empty_keys=tuple(sorted(empty)),
    )
