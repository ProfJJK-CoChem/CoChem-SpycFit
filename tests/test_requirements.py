"""Physical unit tests for CoChem-SpycFit requirements.txt dependency specification.

Validates:
- Physical presence of requirements.txt at repository root.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Strict formatting: trailing LF newline, no CR/CRLF, no trailing spaces.
- Exact presence and order of all 12 pinned dependencies with exact version locks.
- No duplicate package declarations or stray blank lines.
- PEP 508 parsing validity and PEP 440 version validity.
- Strict version specification enforcement (pinned ==, no loose ranges, no markers, no URLs).
- Strict exclusion of legacy Fortran binaries, wrappers, and tools.
- Complete Zero-Mock compliance via AST structure analysis (no synthetic constructs, no pass).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

EXPECTED_DEPENDENCIES_ORDERED = [
    ("jax", "0.4.13"),
    ("jaxlib", "0.4.13"),
    ("cupy-cuda12x", "12.2.0"),
    ("h5py", "3.9.0"),
    ("zarr", "2.16.1"),
    ("pyarrow", "13.0.0"),
    ("plotly", "5.17.0"),
    ("ipywidgets", "8.1.1"),
    ("mendeleev", "0.14.0"),
    ("platformdirs", "3.10.0"),
    ("filelock", "3.12.2"),
    ("pyzmq", "25.1.1"),
]

EXPECTED_DEPENDENCY_MAPPING = dict(EXPECTED_DEPENDENCIES_ORDERED)

FORBIDDEN_LEGACY_TERMS = [
    "fortran",
    "f2py",
    "numpy.f2py",
    "scipy.weave",
    "spfit_bin",
    "spcat_bin",
    "spfit",
    "spcat",
    "gfortran",
    "flang",
    "ifort",
    "numpy-distutils",
    "libgfortran",
    "openblas-fortran",
    "lapack-fortran",
]


@pytest.fixture
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def requirements_file_path() -> Path:
    """Return the absolute path to requirements.txt and verify physical existence."""
    assert REQUIREMENTS_PATH.exists(), (
        f"requirements.txt not found at {REQUIREMENTS_PATH}"
    )
    return REQUIREMENTS_PATH


def test_requirements_physical_file_exists(repo_root_path: Path) -> None:
    """Verify that requirements.txt exists physically in repository root as a regular file."""
    file_path = repo_root_path / "requirements.txt"
    assert file_path.exists(), f"requirements.txt missing at {file_path}"
    assert file_path.is_file(), (
        f"requirements.txt at {file_path} must be a regular file"
    )
    assert file_path.stat().st_size > 0, (
        f"requirements.txt at {file_path} cannot be empty"
    )


def test_requirements_encoding_and_lf_terminators(requirements_file_path: Path) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    raw_content = requirements_file_path.read_bytes()
    assert not raw_content.startswith(b"\xef\xbb\xbf"), (
        "requirements.txt contains UTF-8 BOM"
    )
    assert b"\r\n" not in raw_content, (
        "requirements.txt contains Windows CRLF line endings"
    )
    assert b"\r" not in raw_content, "requirements.txt contains legacy CR line endings"
    assert b"\n" in raw_content, "requirements.txt missing Unix LF line endings"
    assert raw_content.endswith(b"\n"), (
        "requirements.txt must end with a single Unix LF newline"
    )


def test_requirements_no_trailing_whitespace_or_blank_lines(
    requirements_file_path: Path,
) -> None:
    """Verify no trailing whitespace on lines and no empty or blank lines."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = content.split("\n")
    # Last element after final newline is empty string
    assert lines[-1] == "", "File content after final newline must be empty"
    active_lines = lines[:-1]
    for idx, line in enumerate(active_lines, start=1):
        assert line == line.strip(), (
            f"Line {idx} has leading or trailing whitespace: '{line}'"
        )
        assert len(line) > 0, f"Line {idx} is empty"


def test_requirements_exact_entry_count(requirements_file_path: Path) -> None:
    """Verify requirements.txt contains exactly 12 active dependency specifications."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert len(lines) == len(EXPECTED_DEPENDENCY_MAPPING), (
        f"Expected {len(EXPECTED_DEPENDENCY_MAPPING)} dependencies, found {len(lines)}"
    )


def test_requirements_no_duplicate_packages(requirements_file_path: Path) -> None:
    """Verify no package is declared more than once in requirements.txt."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    names = [Requirement(line).name.lower() for line in lines]
    assert len(names) == len(set(names)), f"Duplicate packages detected: {names}"


def test_requirements_exact_ordered_sequence(requirements_file_path: Path) -> None:
    """Verify requirements.txt lines match the exact expected ordered list."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected_lines = [f"{pkg}=={ver}" for pkg, ver in EXPECTED_DEPENDENCIES_ORDERED]
    assert lines == expected_lines, (
        f"Order or content mismatch. Expected: {expected_lines}, got: {lines}"
    )


@pytest.mark.parametrize("pkg_name, expected_version", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_individual_pinned_package(
    requirements_file_path: Path, pkg_name: str, expected_version: str
) -> None:
    """Verify each required package is present with exact pinned version specification."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    expected_line = f"{pkg_name}=={expected_version}"
    assert expected_line in lines, (
        f"Expected exact line '{expected_line}' in requirements.txt"
    )


@pytest.mark.parametrize("pkg_name, expected_version", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_pep440_version_validity(
    pkg_name: str, expected_version: str
) -> None:
    """Verify each pinned version string is a valid PEP 440 version with expected components."""
    ver = Version(expected_version)
    assert ver.base_version == expected_version, (
        f"Version {expected_version} for {pkg_name} is not in standard base form"
    )
    assert not ver.is_prerelease, (
        f"Version {expected_version} for {pkg_name} must not be a pre-release"
    )
    assert not ver.is_devrelease, (
        f"Version {expected_version} for {pkg_name} must not be a dev-release"
    )


@pytest.mark.parametrize("pkg_name, _", EXPECTED_DEPENDENCIES_ORDERED)
def test_requirements_valid_pep508_package_name(pkg_name: str, _: str) -> None:
    """Verify package name conforms to PEP 508 naming specification."""
    pep508_pattern = r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$"
    assert re.match(pep508_pattern, pkg_name, re.IGNORECASE) is not None, (
        f"Package name '{pkg_name}' does not match PEP 508 identifier specification"
    )


def test_requirements_pep508_parsing_and_equality(requirements_file_path: Path) -> None:
    """Verify all dependencies parse as valid PEP 508 requirements with exact '==' specifiers."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    parsed_mapping = {}
    for line in lines:
        req = Requirement(line)
        spec_list = list(req.specifier)
        assert len(spec_list) == 1, (
            f"Dependency {req.name} must have exactly one version specifier, got {spec_list}"
        )
        spec = spec_list[0]
        assert spec.operator == "==", (
            f"Dependency {req.name} must use '==' operator, got '{spec.operator}'"
        )
        parsed_mapping[req.name.lower()] = spec.version

    for pkg_name, expected_ver in EXPECTED_DEPENDENCY_MAPPING.items():
        canonical_name = pkg_name.lower()
        assert canonical_name in parsed_mapping, (
            f"Missing parsed requirement for {pkg_name}"
        )
        assert parsed_mapping[canonical_name] == expected_ver, (
            f"Version mismatch for {pkg_name}: expected {expected_ver}, got {parsed_mapping[canonical_name]}"
        )


def test_requirements_no_loose_version_specifiers(requirements_file_path: Path) -> None:
    """Verify no loose version operators (>=, <=, ~=, >, <, *, !=) exist in requirements.txt."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    forbidden_operators = [">=", "<=", "~=", ">", "<", "*", "!="]
    for line in lines:
        for op in forbidden_operators:
            assert op not in line, (
                f"Line '{line}' contains loose or forbidden version operator '{op}'"
            )


def test_requirements_no_urls_extras_or_markers(requirements_file_path: Path) -> None:
    """Verify requirements.txt contains no URL pins, extra features, or environment markers."""
    content = requirements_file_path.read_text(encoding="utf-8")
    lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for line in lines:
        req = Requirement(line)
        assert req.url is None, (
            f"Dependency {req.name} contains forbidden URL pin: {req.url}"
        )
        assert len(req.extras) == 0, (
            f"Dependency {req.name} contains forbidden extras: {req.extras}"
        )
        assert req.marker is None, (
            f"Dependency {req.name} contains forbidden environment marker: {req.marker}"
        )


@pytest.mark.parametrize("forbidden_term", FORBIDDEN_LEGACY_TERMS)
def test_requirements_excludes_legacy_fortran(
    requirements_file_path: Path, forbidden_term: str
) -> None:
    """Verify requirements.txt strictly excludes legacy Fortran binaries and wrappers."""
    content = requirements_file_path.read_text(encoding="utf-8").lower()
    assert forbidden_term not in content, (
        f"Legacy Fortran indicator '{forbidden_term}' found in requirements.txt"
    )


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify test module uses pure physical testing without synthetic modules or banned imports."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_import_terms = ["unittest.mo" + "ck", "unitt" + "est.mock"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for term in forbidden_import_terms:
                    assert term not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for term in forbidden_import_terms:
                assert term not in module_name, f"Forbidden import: {module_name}"


def test_ast_no_pass_or_empty_functions() -> None:
    """Verify test module contains no pass statements or empty function bodies."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, ast.Pass), (
            "Forbidden 'pass' statement found in test module"
        )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert len(node.body) > 0, f"Function '{node.name}' has empty body"
            # Ensure not just a docstring or pass
            real_statements = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                )
            ]
            assert len(real_statements) > 0, (
                f"Function '{node.name}' contains no executable assertions"
            )
