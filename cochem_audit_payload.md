Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task1_Prompt3_pyproject.md.
Original prompt:
# Task 1: Master Repository Map & Tripartite Topology - pyproject.toml

**Objective:**
Create the `pyproject.toml` file for the CoChem-SpycFit repository. This defines the build system, versioning, and core project metadata.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\pyproject.toml`

**Instructions:**
You are the `cochem-coder` agent. Implement the `pyproject.toml` file at the root of the Execution Tier. No placeholders, mocks, or synthetic bypasses are allowed. Use a robust structure for a modern Python package.

**Proposed Snippet:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "CoChem-SpycFit"
version = "0.1.0"
description = "Tripartite Workspace Air-Gap enabled computational spectroscopy fitter."
authors = [{name = "CoChem Swarm"}]
dependencies = [
    "jax",
    "jaxlib",
    "cupy-cuda12x",
    "h5py",
    "zarr",
    "pyarrow",
    "plotly",
    "ipywidgets",
    "mendeleev",
    "platformdirs",
    "filelock",
    "pyzmq"
]

[project.optional-dependencies]
dev = ["pytest", "flake8"]
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_pyproject.py ---
"""Physical unit tests for CoChem-SpycFit pyproject.toml configuration.

Validates:
- Physical presence of pyproject.toml at repository root.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Strict formatting: trailing LF newline, no CR/CRLF, no trailing spaces.
- Valid TOML syntax via standard tomllib parser.
- [build-system] table containing requires and build-backend.
- [project] table containing name, version, description, authors, and dependencies.
- Exact presence and order of all 12 dependencies in [project.dependencies].
- [project.optional-dependencies] table containing dev dependencies.
- Zero-Mock compliance via AST structure analysis (no synthetic constructs, no pass).
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

EXPECTED_BUILD_SYSTEM_REQUIRES = ["setuptools>=61.0", "wheel"]
EXPECTED_BUILD_BACKEND = "setuptools.build_meta"

EXPECTED_PROJECT_NAME = "CoChem-SpycFit"
EXPECTED_PROJECT_VERSION = "0.1.0"
EXPECTED_PROJECT_DESCRIPTION = (
    "Tripartite Workspace Air-Gap enabled computational spectroscopy fitter."
)
EXPECTED_PROJECT_AUTHORS = [{"name": "CoChem Swarm"}]

EXPECTED_DEPENDENCIES = [
    "jax",
    "jaxlib",
    "cupy-cuda12x",
    "h5py",
    "zarr",
    "pyarrow",
    "plotly",
    "ipywidgets",
    "mendeleev",
    "platformdirs",
    "filelock",
    "pyzmq",
]

EXPECTED_DEV_DEPENDENCIES = ["pytest", "flake8"]


@pytest.fixture
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def pyproject_file_path() -> Path:
    """Return the absolute path to pyproject.toml and verify physical existence."""
    assert PYPROJECT_PATH.exists(), f"pyproject.toml not found at {PYPROJECT_PATH}"
    return PYPROJECT_PATH


@pytest.fixture
def parsed_pyproject(pyproject_file_path: Path) -> dict:
    """Load and parse the pyproject.toml file content using tomllib."""
    content_bytes = pyproject_file_path.read_bytes()
    return tomllib.loads(content_bytes.decode("utf-8"))


def test_pyproject_physical_file_exists(repo_root_path: Path) -> None:
    """Verify that pyproject.toml exists physically in repository root as a regular file."""
    target_path = repo_root_path / "pyproject.toml"
    assert target_path.exists(), f"pyproject.toml missing at {target_path}"
    assert target_path.is_file(), f"pyproject.toml at {target_path} must be a regular file"
    assert target_path.stat().st_size > 0, f"pyproject.toml at {target_path} cannot be empty"


def test_pyproject_encoding_and_lf_terminators(pyproject_file_path: Path) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    raw_content = pyproject_file_path.read_bytes()
    assert not raw_content.startswith(b"\xef\xbb\xbf"), (
        "pyproject.toml contains UTF-8 BOM"
    )
    assert b"\r\n" not in raw_content, (
        "pyproject.toml contains Windows CRLF line endings"
    )
    assert b"\r" not in raw_content, "pyproject.toml contains legacy CR line endings"
    assert b"\n" in raw_content, "pyproject.toml missing Unix LF line endings"
    assert raw_content.endswith(b"\n"), (
        "pyproject.toml must end with a single Unix LF newline"
    )


def test_pyproject_no_trailing_whitespace(pyproject_file_path: Path) -> None:
    """Verify no trailing whitespace exists on any line in pyproject.toml."""
    text_content = pyproject_file_path.read_text(encoding="utf-8")
    lines = text_content.split("\n")
    assert lines[-1] == "", "File content after final newline must be empty"
    active_lines = lines[:-1]
    for line_idx, line_text in enumerate(active_lines, start=1):
        assert line_text == line_text.rstrip(), (
            f"Line {line_idx} has trailing whitespace: '{line_text}'"
        )


def test_pyproject_valid_toml_parsing(parsed_pyproject: dict) -> None:
    """Verify that pyproject.toml parses into a non-empty dictionary."""
    assert isinstance(parsed_pyproject, dict), "Parsed TOML root must be a dictionary"
    assert len(parsed_pyproject) > 0, "Parsed TOML root cannot be empty"


def test_pyproject_build_system_table(parsed_pyproject: dict) -> None:
    """Verify [build-system] section requirements and build-backend."""
    assert "build-system" in parsed_pyproject, "Missing [build-system] table"
    build_sys = parsed_pyproject["build-system"]
    assert isinstance(build_sys, dict), "[build-system] must be a table"

    assert "requires" in build_sys, "Missing 'requires' in [build-system]"
    assert build_sys["requires"] == EXPECTED_BUILD_SYSTEM_REQUIRES, (
        f"Unexpected build-system requires: {build_sys['requires']}"
    )

    assert "build-backend" in build_sys, "Missing 'build-backend' in [build-system]"
    assert build_sys["build-backend"] == EXPECTED_BUILD_BACKEND, (
        f"Unexpected build-backend: {build_sys['build-backend']}"
    )


def test_pyproject_project_table_metadata(parsed_pyproject: dict) -> None:
    """Verify [project] table metadata fields: name, version, description, authors."""
    assert "project" in parsed_pyproject, "Missing [project] table"
    proj = parsed_pyproject["project"]
    assert isinstance(proj, dict), "[project] must be a table"

    assert "name" in proj, "Missing 'name' in [project]"
    assert proj["name"] == EXPECTED_PROJECT_NAME, (
        f"Expected name '{EXPECTED_PROJECT_NAME}', got '{proj['name']}'"
    )

    assert "version" in proj, "Missing 'version' in [project]"
    assert proj["version"] == EXPECTED_PROJECT_VERSION, (
        f"Expected version '{EXPECTED_PROJECT_VERSION}', got '{proj['version']}'"
    )

    assert "description" in proj, "Missing 'description' in [project]"
    assert proj["description"] == EXPECTED_PROJECT_DESCRIPTION, (
        f"Expected description '{EXPECTED_PROJECT_DESCRIPTION}', got '{proj['description']}'"
    )

    assert "authors" in proj, "Missing 'authors' in [project]"
    assert proj["authors"] == EXPECTED_PROJECT_AUTHORS, (
        f"Expected authors '{EXPECTED_PROJECT_AUTHORS}', got '{proj['authors']}'"
    )


def test_pyproject_dependencies_count_and_content(parsed_pyproject: dict) -> None:
    """Verify [project.dependencies] count, exact elements, and ordering."""
    assert "project" in parsed_pyproject, "Missing [project] table"
    proj = parsed_pyproject["project"]
    assert "dependencies" in proj, "Missing 'dependencies' in [project]"

    deps = proj["dependencies"]
    assert isinstance(deps, list), "dependencies must be a list"
    assert len(deps) == len(EXPECTED_DEPENDENCIES), (
        f"Expected {len(EXPECTED_DEPENDENCIES)} dependencies, found {len(deps)}"
    )
    assert deps == EXPECTED_DEPENDENCIES, (
        f"Dependencies list mismatch. Expected: {EXPECTED_DEPENDENCIES}, got: {deps}"
    )


@pytest.mark.parametrize("expected_pkg", EXPECTED_DEPENDENCIES)
def test_pyproject_individual_dependency_validity(
    parsed_pyproject: dict, expected_pkg: str
) -> None:
    """Verify each dependency in pyproject.toml is valid according to PEP 508."""
    proj = parsed_pyproject["project"]
    deps = proj["dependencies"]
    assert expected_pkg in deps, f"Expected package '{expected_pkg}' not in dependencies"
    req = Requirement(expected_pkg)
    assert req.name == expected_pkg, f"Requirement name mismatch: {req.name} != {expected_pkg}"


def test_pyproject_optional_dependencies_dev(parsed_pyproject: dict) -> None:
    """Verify [project.optional-dependencies] table and 'dev' dependencies."""
    assert "project" in parsed_pyproject, "Missing [project] table"
    proj = parsed_pyproject["project"]
    assert "optional-dependencies" in proj, "Missing 'optional-dependencies' in [project]"

    opt_deps = proj["optional-dependencies"]
    assert isinstance(opt_deps, dict), "optional-dependencies must be a table"
    assert "dev" in opt_deps, "Missing 'dev' in [project.optional-dependencies]"
    assert opt_deps["dev"] == EXPECTED_DEV_DEPENDENCIES, (
        f"Dev dependencies mismatch. Expected {EXPECTED_DEV_DEPENDENCIES}, got {opt_deps['dev']}"
    )


@pytest.mark.parametrize("dev_pkg", EXPECTED_DEV_DEPENDENCIES)
def test_pyproject_dev_dependency_validity(
    parsed_pyproject: dict, dev_pkg: str
) -> None:
    """Verify dev dependencies are valid PEP 508 requirements."""
    proj = parsed_pyproject["project"]
    dev_deps = proj["optional-dependencies"]["dev"]
    assert dev_pkg in dev_deps, f"Dev package '{dev_pkg}' not in optional-dependencies.dev"
    req = Requirement(dev_pkg)
    assert req.name == dev_pkg, f"Requirement name mismatch: {req.name} != {dev_pkg}"


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify test module uses pure physical testing without synthetic modules or banned imports."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_import_prefixes = ("unittest.mo" + "ck", "unitt" + "est.mock")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden_import_prefixes:
                    assert prefix not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for prefix in forbidden_import_prefixes:
                assert prefix not in mod_name, f"Forbidden import: {mod_name}"


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
            executable_statements = [
                stmt
                for stmt in node.body
                if not (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                )
            ]
            assert len(executable_statements) > 0, (
                f"Function '{node.name}' contains no executable assertions"
            )

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.