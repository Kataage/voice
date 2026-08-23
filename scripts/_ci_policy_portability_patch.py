from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET_PLACEHOLDER = "__PERSONAVOICE_CANONICAL_POLICY__"
PREVIOUS_POLICIES = (
    "12-6ef53c9f266fd6794c3e",  # previous main on LF checkouts
    "12-1d31ef1abd217bcf5c4f",  # previous main on Windows CRLF checkouts
)


def patch(path: str, old: str, new: str, *, count: int = 1) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    found = text.count(old)
    if found < count:
        raise RuntimeError(
            f"Expected at least {count} patch anchor(s) in {path}, found {found}: {old[:160]!r}"
        )
    target.write_text(text.replace(old, new, count), encoding="utf-8", newline="\n")


patch(
    "src/personavoice/state.py",
    '''def _file_contract(path: Path) -> str:\n    try:\n        if not path.is_file():\n            return "missing"\n        return hashlib.sha256(path.read_bytes()).hexdigest()\n    except OSError:\n        return "unreadable"\n''',
    '''def _file_contract(path: Path) -> str:\n    """Hash audited text contracts independently of checkout line endings."""\n\n    try:\n        if not path.is_file():\n            return "missing"\n        raw = path.read_bytes()\n    except OSError:\n        return "unreadable"\n    try:\n        text = raw.decode("utf-8")\n    except UnicodeDecodeError:\n        normalized = raw\n    else:\n        normalized = text.replace("\\r\\n", "\\n").replace("\\r", "\\n").encode("utf-8")\n    return hashlib.sha256(normalized).hexdigest()\n''',
)
patch("src/personavoice/state.py", '        "schema": 13,\n', '        "schema": 14,\n')
patch(
    "src/personavoice/state.py",
    '    return f"13-{hashlib.sha256(encoded).hexdigest()[:20]}"\n',
    '    return f"14-{hashlib.sha256(encoded).hexdigest()[:20]}"\n',
)
patch(
    "src/personavoice/state.py",
    '''PREPARE_CACHE_POLICY_COMPATIBILITY = {\n    "13-72c7cffa967913f59b99": frozenset({'12-6ef53c9f266fd6794c3e'}),\n}\n''',
    f'''PREPARE_CACHE_POLICY_COMPATIBILITY = {{\n    "{TARGET_PLACEHOLDER}": frozenset({PREVIOUS_POLICIES!r}),\n}}\n''',
)

path = ROOT / "tests" / "test_prepare_checkpoints.py"
text = path.read_text(encoding="utf-8")
text = text.replace(
    "    _prepare_policy_compatible,\n",
    "    _file_contract,\n    _prepare_policy_compatible,\n",
    1,
)
anchor = "\ndef test_prepare_policy_migration_is_scoped_to_exact_new_generation():\n"
insert = '''\ndef test_prepare_policy_text_contract_is_line_ending_independent(tmp_path: Path):\n    source = tmp_path / "contract.py"\n    source.write_bytes(b"alpha\\nbeta\\n")\n    lf = _file_contract(source)\n    source.write_bytes(b"alpha\\r\\nbeta\\r\\n")\n    crlf = _file_contract(source)\n    source.write_bytes(b"alpha\\rbeta\\r")\n    cr = _file_contract(source)\n    assert lf == crlf == cr\n\n\n'''
if anchor not in text:
    raise RuntimeError("prepare policy migration test anchor not found")
text = text.replace(anchor, "\n" + insert + anchor.lstrip("\n"), 1)
text = text.replace(
    "    assert previous\n",
    "    assert previous == frozenset({\n"
    "        \"12-6ef53c9f266fd6794c3e\",\n"
    "        \"12-1d31ef1abd217bcf5c4f\",\n"
    "    })\n",
    1,
)
path.write_text(text, encoding="utf-8", newline="\n")

print("Cross-platform prepare policy patch applied")
