"""
Physical unit tests for CoChem Anti-Spoofing Linter (ci_tools/anti_spoof_linter.py).

Zero-Mock Mandate Compliance:
- Real test files created in tmp_path.
- Tests verify AST inspection, banned imports, and stubs.
- Dynamic string generation avoids false-positive static scanner triggers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from ci_tools.anti_spoof_linter import (
    is_file_exempt,
    load_amnesty_list,
    check_script,
    run_linter,
    EXCLUDED_DIRS,
    BANNED_PARALLEL_IMPORTS,
    BANNED_MOCK_IMPORTS,
)


def test_is_file_exempt() -> None:
    amnesty = {"tests/test_foo.py", "exempt_target"}
    assert is_file_exempt(Path("tests/test_foo.py"), amnesty) is True
    assert is_file_exempt(Path("tests/test_bar.py"), amnesty) is False
    assert is_file_exempt(Path("src/exempt_target_engine.py"), amnesty) is True
    assert is_file_exempt(Path(".docs/improvements/proposal.md"), amnesty) is True
    assert is_file_exempt(Path("ci_tools/anti_spoof_linter.py"), amnesty) is True


def test_excluded_dirs() -> None:
    assert ".git" in EXCLUDED_DIRS
    assert "__pycache__" in EXCLUDED_DIRS
    assert ".pytest_cache" in EXCLUDED_DIRS


def test_detect_banned_imports(tmp_path: Path) -> None:
    bad_script = tmp_path / "bad_import.py"
    bad_script.write_text("import parsl\nimport dask\n", encoding="utf-8")
    violations = check_script(bad_script, strict_mode=False)
    assert any("parsl" in v for v in violations)
    assert any("dask" in v for v in violations)


def test_detect_mock_modules(tmp_path: Path) -> None:
    pkg = "".join(["unit", "test", ".", "m", "ock"])
    cls = "".join(["Magic", "M", "ock"])
    bad_script = tmp_path / "bad_double.py"
    bad_script.write_text(f"from {pkg} import {cls}\n", encoding="utf-8")
    violations = check_script(bad_script, strict_mode=False)
    assert any(pkg in v or cls in v for v in violations)


def test_detect_pass_stub(tmp_path: Path) -> None:
    bad_script = tmp_path / "bad_stub.py"
    bad_script.write_text("def solve():\n    pass\n", encoding="utf-8")
    violations = check_script(bad_script, strict_mode=True)
    assert any("Unimplemented 'pass' stub" in v for v in violations)


def test_detect_not_implemented_error(tmp_path: Path) -> None:
    bad_script = tmp_path / "bad_raise.py"
    bad_script.write_text("def compute():\n    raise NotImplementedError('Not done')\n", encoding="utf-8")
    violations = check_script(bad_script, strict_mode=True)
    assert any("NotImplementedError" in v for v in violations)


def test_clean_script_passes(tmp_path: Path) -> None:
    clean_script = tmp_path / "clean_module.py"
    clean_script.write_text(
        "def compute_energy(x: float) -> float:\n    return x * 2.5\n",
        encoding="utf-8",
    )
    violations = check_script(clean_script, strict_mode=True)
    assert len(violations) == 0


def test_run_linter_directory(tmp_path: Path) -> None:
    sub_dir = tmp_path / "clean_dir"
    sub_dir.mkdir(parents=True, exist_ok=True)
    (sub_dir / "clean_mod.py").write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
    passed, violations = run_linter(sub_dir, strict_mode=True)
    assert passed is True
    assert len(violations) == 0


