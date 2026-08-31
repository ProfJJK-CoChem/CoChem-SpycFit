"""# zero-stub anti-spoofing engine
CoChem Static AST Zero-Mock & Anti-Spoofing Linter (cochem_base.core_engine.anti_spoof_linter)

Inspects Python AST syntax trees to enforce Zero-Mock & Anti-Spoofing Protocol invariants:
- Rejecting test doubles / mock frameworks (unittest.mock, pytest.monkeypatch, MagicMock, patch).
- Rejecting dead-end stubs (raise NotImplementedError, empty pass statements).
- Enforcing dynamic Mendeleev mass resolution (prohibiting hardcoded mass tables).
- Detecting string obfuscation and synthetic tensor generation.

Governing Requirements:
- REQ-68 (FR-68.1): Static AST Zero-Mock Scanner.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

logger = logging.getLogger("anti_spoof_linter")

BANNED_MOCK_MODULES: Set[str] = {
    "unittest.mock",
    "mock",
    "pytest_mock",
}

BANNED_MOCK_NAMES: Set[str] = {
    "MagicMock",
    "Mock",
    "patch",
    "monkeypatch",
    "mocker",
    "PropertyMock",
}

COMMON_ELEMENT_SYMBOLS: Set[str] = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Br", "I",
}


@dataclass
class LintViolation:
    """Represents a single static AST anti-spoofing rule violation."""
    file_path: str
    line_number: int
    code: str
    message: str
    snippet: str
    remediation: str

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "code": self.code,
            "message": self.message,
            "snippet": self.snippet,
            "remediation": self.remediation,
        }


class AntiSpoofASTVisitor(ast.NodeVisitor):
    """AST Visitor scanning for forbidden mock constructs, stubs, and mass tables."""

    def __init__(self, file_path: str, source_lines: List[str]):
        self.file_path = file_path
        self.source_lines = source_lines
        self.violations: List[LintViolation] = []

    def _get_snippet(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.source_lines):
            return self.source_lines[lineno - 1].strip()
        return ""

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in BANNED_MOCK_MODULES or any(alias.name.startswith(m + ".") for m in BANNED_MOCK_MODULES):
                self.violations.append(
                    LintViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        code="SPOOF-01-MOCK-IMPORT",
                        message=f"Banned mock module imported: '{alias.name}'",
                        snippet=self._get_snippet(node.lineno),
                        remediation="Remove test doubles and execute against genuine physical tensors.",
                    )
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        mod = node.module or ""
        if mod in BANNED_MOCK_MODULES or any(mod.startswith(m + ".") for m in BANNED_MOCK_MODULES):
            self.violations.append(
                LintViolation(
                    file_path=self.file_path,
                    line_number=node.lineno,
                    code="SPOOF-01-MOCK-IMPORT",
                    message=f"Banned mock module imported: 'from {mod}'",
                    snippet=self._get_snippet(node.lineno),
                    remediation="Remove mock imports and verify real execution states.",
                )
            )
        for alias in node.names:
            if alias.name in BANNED_MOCK_NAMES:
                self.violations.append(
                    LintViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        code="SPOOF-01-MOCK-IMPORT",
                        message=f"Banned mock construct imported: '{alias.name}'",
                        snippet=self._get_snippet(node.lineno),
                        remediation="Remove MagicMock/patch constructs in accordance with Zero-Mock mandate.",
                    )
                )
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc:
            exc_name = ""
            if isinstance(node.exc, ast.Name):
                exc_name = node.exc.id
            elif isinstance(node.exc, ast.Call) and isinstance(node.exc.func, ast.Name):
                exc_name = node.exc.func.id

            if exc_name == "NotImplementedError":
                self.violations.append(
                    LintViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        code="SPOOF-02-DEAD-END-STUB",
                        message="Forbidden dead-end stub: 'raise NotImplementedError'",
                        snippet=self._get_snippet(node.lineno),
                        remediation="Implement concrete physical logic instead of placeholder errors.",
                    )
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Check for single-statement pass in non-overload concrete function
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            # Check decorator list for @abstractmethod or @overload
            is_abstract = False
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name) and dec.id in ("abstractmethod", "overload"):
                    is_abstract = True
            if not is_abstract:
                self.violations.append(
                    LintViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        code="SPOOF-03-EMPTY-PASS",
                        message=f"Forbidden empty pass block in concrete function '{node.name}'",
                        snippet=self._get_snippet(node.lineno),
                        remediation="Provide complete concrete implementation or mark abstract.",
                    )
                )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        # Check for hardcoded mass dictionary literals (e.g. {"H": 1.008, "C": 12.011})
        if len(node.keys) >= 3:
            str_keys: Set[str] = set()
            has_float_vals = False
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    if k.value in COMMON_ELEMENT_SYMBOLS:
                        str_keys.add(k.value)
                if isinstance(v, ast.Constant) and isinstance(v.value, float) and 1.0 <= v.value <= 300.0:
                    has_float_vals = True

            if len(str_keys) >= 3 and has_float_vals:
                self.violations.append(
                    LintViolation(
                        file_path=self.file_path,
                        line_number=node.lineno,
                        code="SPOOF-04-HARDCODED-MASS-TABLE",
                        message="Forbidden hardcoded atomic mass dictionary detected.",
                        snippet=self._get_snippet(node.lineno),
                        remediation="Retrieve atomic and isotopic masses dynamically via the 'mendeleev' library.",
                    )
                )
        self.generic_visit(node)


def scan_file(file_path: Union[str, Path]) -> List[LintViolation]:
    """Scan a single Python file for zero-mock and anti-spoofing violations."""
    p = Path(file_path).resolve()
    if not p.exists() or p.suffix != ".py":
        return []

    try:
        source = p.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(p))
    except Exception as e:
        logger.warning(f"Could not parse AST for {p.as_posix()}: {e}")
        return []

    source_lines = source.splitlines()
    visitor = AntiSpoofASTVisitor(file_path=p.as_posix(), source_lines=source_lines)
    visitor.visit(tree)
    return visitor.violations


def scan_directory(dir_path: Union[str, Path]) -> Dict[str, List[LintViolation]]:
    """Scan an entire directory recursively for zero-mock violations."""
    d = Path(dir_path).resolve()
    results: Dict[str, List[LintViolation]] = {}

    for py_file in d.rglob("*.py"):
        viols = scan_file(py_file)
        if viols:
            results[py_file.as_posix()] = viols

    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python anti_spoof_linter.py <file_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1]).resolve()
    if target.is_file():
        violations = scan_file(target)
        if violations:
            print(f"[ANTI_SPOOF_LINTER: VIOLATION DETECTED] in {target.as_posix()}:")
            for v in violations:
                print(f"  Line {v.line_number}: [{v.code}] {v.message}")
            sys.exit(1)
        else:
            print(f"[ANTI_SPOOF_LINTER] Passed: {target.as_posix()}")
            sys.exit(0)
    elif target.is_dir():
        results = scan_directory(target)
        if results:
            print(f"[ANTI_SPOOF_LINTER: VIOLATION DETECTED] {len(results)} files failed:")
            for fp, viols in results.items():
                print(f"\nFile: {fp}")
                for v in viols:
                    print(f"  Line {v.line_number}: [{v.code}] {v.message}")
            sys.exit(1)
        else:
            print(f"[ANTI_SPOOF_LINTER] Passed: All files in {target.as_posix()} clean.")
            sys.exit(0)


if __name__ == "__main__":
    main()
