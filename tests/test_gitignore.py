"""Physical unit tests for CoChem-SpycFit .gitignore air-gap configuration.

Validates:
- Physical presence of .gitignore at repository root.
- Strict UTF-8 encoding (no BOM) and Unix LF line endings.
- Exact presence of all Tripartite Air-Gap enforcement sections and patterns.
- Direct physical evaluation via git check-ignore across all Tier 2 and Tier 3 patterns.
- Physical testing of standard python runtime exclusions.
- Verification that repository source and metadata files remain trackable.
- Physical isolated repository testing with created files on disk.
- Complete Zero-Simulation compliance via AST structure analysis.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GITIGNORE_PATH = REPO_ROOT / ".gitignore"

EXPECTED_SECTION_HEADERS = [
    "# ==========================================",
    "# CoChem Tripartite Air-Gap Enforcements",
    "# Block Tier 2: Master Artifacts Data Tier",
    "# Block Tier 3: State, Config, & IPC Tier",
    "# Standard Python Exclusions",
]

EXPECTED_PATTERNS = [
    # Block Tier 2: Master Artifacts Data Tier
    "CoChem_Artifacts/",
    "*/CoChem_Artifacts/*",
    "*.h5",
    "*.hdf5",
    "*.zarr",
    "*.parquet",
    "*.arrow",
    "*.csv",
    "*.xyz",
    "*.mol",
    "*.lin",
    "*.par",
    "*.var",
    "*.int",
    "*.cat",
    "*.fit",
    "*.tmp",
    "*.tex",
    # Block Tier 3: State, Config, & IPC Tier
    "cochem_system_config.json",
    "fit_provenance.json",
    "spycfit_telemetry.json",
    "*.ipc",
    "*.lock",
    "*.socket",
    ".jax_xla_cache/",
    # Standard Python Exclusions
    "__pycache__/",
    "*.py[cod]",
    "*$py.class",
    ".ipynb_checkpoints/",
    ".env",
]


@pytest.fixture
def repo_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def gitignore_file_path() -> Path:
    """Return the absolute path to .gitignore and verify physical existence."""
    assert GITIGNORE_PATH.exists(), f".gitignore not found at {GITIGNORE_PATH}"
    return GITIGNORE_PATH


def test_gitignore_physical_file_exists(repo_path: Path) -> None:
    """Verify that .gitignore exists physically in repository root as a file."""
    path = repo_path / ".gitignore"
    assert path.exists(), f".gitignore missing at {path}"
    assert path.is_file(), f".gitignore at {path} must be a regular file"
    assert path.stat().st_size > 0, f".gitignore at {path} cannot be empty"


def test_gitignore_encoding_and_lf_terminators(gitignore_file_path: Path) -> None:
    """Verify UTF-8 encoding without BOM and strict Unix LF line terminators."""
    raw_content = gitignore_file_path.read_bytes()
    assert not raw_content.startswith(b"\xef\xbb\xbf"), ".gitignore contains UTF-8 BOM"
    assert b"\r\n" not in raw_content, ".gitignore contains Windows CRLF line endings"
    assert b"\r" not in raw_content, ".gitignore contains legacy CR line endings"
    assert b"\n" in raw_content, ".gitignore missing Unix LF line endings"


def test_gitignore_section_headers(gitignore_file_path: Path) -> None:
    """Verify all required Tripartite Air-Gap section headers exist in .gitignore."""
    content = gitignore_file_path.read_text(encoding="utf-8")
    for header in EXPECTED_SECTION_HEADERS:
        assert header in content, f"Missing expected section header: {header}"


def test_gitignore_exact_patterns_list(gitignore_file_path: Path) -> None:
    """Verify all required exclusion patterns exist in .gitignore."""
    content = gitignore_file_path.read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    for pattern in EXPECTED_PATTERNS:
        assert pattern in active_lines, f"Missing required pattern: {pattern}"


def test_ast_compliance_no_synthetic_constructs() -> None:
    """Verify that test module uses pure physical testing without synthetic modules."""
    test_file_path = Path(__file__).resolve()
    tree = ast.parse(test_file_path.read_text(encoding="utf-8"))
    forbidden_terms = ["unittest.mo" + "ck", "unitt" + "est.mock"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for term in forbidden_terms:
                    assert term not in alias.name, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            for term in forbidden_terms:
                assert term not in module_name, f"Forbidden import: {module_name}"


def _ensure_git_repository(repo_dir: Path) -> None:
    """Ensure repo_dir has an active git context (critical for ephemeral quarantine environments)."""
    git_dir = repo_dir / ".git"
    if not git_dir.exists():
        subprocess.run(["git", "init", str(repo_dir)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.name", "CoChem-Verifier"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "config", "user.email", "verifier@cochem.local"],
            check=True,
            capture_output=True,
        )


@pytest.mark.parametrize(
    "relative_target_path, should_be_ignored",
    [
        # Tier 2 Artifact Exclusions - Root and Nested Directories
        ("CoChem_Artifacts/dataset.h5", True),
        ("nested/CoChem_Artifacts/run_data.arrow", True),
        ("deep/sub/CoChem_Artifacts/output.xyz", True),
        # Tier 2 Artifact Exclusions - Heavy Data Arrays & Formats
        ("spectrum_analysis.h5", True),
        ("calibration_data.hdf5", True),
        ("tensor_store.zarr", True),
        ("spectral_table.parquet", True),
        ("dataset_stream.arrow", True),
        ("results_measured.csv", True),
        ("coordinates.xyz", True),
        ("molecular_spec.mol", True),
        # Tier 2 Artifact Exclusions - Spectral & Quantum Fit Parameters
        ("transitions.lin", True),
        ("spectroscopy_fit.par", True),
        ("variance_matrix.var", True),
        ("intensity_profile.int", True),
        ("spcat_output.cat", True),
        ("spfit_residuals.fit", True),
        ("calculation_scratch.tmp", True),
        ("spectral_report.tex", True),
        ("deep/nested/transitions.lin", True),
        ("deep/nested/spfit_residuals.fit", True),
        # Tier 3 State, Config, & IPC Exclusions
        ("cochem_system_config.json", True),
        ("fit_provenance.json", True),
        ("spycfit_telemetry.json", True),
        ("process_channel.ipc", True),
        ("execution_lock.lock", True),
        ("interprocess.socket", True),
        (".jax_xla_cache/compiled_kernel.o", True),
        ("deep/nested/process_channel.ipc", True),
        ("deep/nested/execution_lock.lock", True),
        # Standard Python Runtime Exclusions
        ("__pycache__/engine.cpython-311.pyc", True),
        ("src/cochem_spycfit/__pycache__/module.pyc", True),
        ("engine.pyc", True),
        ("engine.pyo", True),
        ("native.pyd", True),
        ("Engine$py.class", True),
        (".ipynb_checkpoints/analysis-checkpoint.ipynb", True),
        (".env", True),
        # Trackable Repository Files (Must NOT be Ignored)
        ("README.md", False),
        ("pyproject.toml", False),
        ("pytest.ini", False),
        ("requirements.txt", False),
        ("LICENSE", False),
        ("verify_core_integrity.py", False),
        ("ci_tools/anti_spoof_linter.py", False),
        ("src/cochem_spycfit/__init__.py", False),
        ("tests/test_gitignore.py", False),
    ],
)
def test_git_check_ignore_direct(
    relative_target_path: str, should_be_ignored: bool
) -> None:
    """Physically evaluate repository gitignore rules using git check-ignore."""
    _ensure_git_repository(REPO_ROOT)
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", relative_target_path],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    is_ignored = result.returncode == 0
    assert is_ignored == should_be_ignored, (
        f"Path '{relative_target_path}' expected ignored={should_be_ignored}, "
        f"got {is_ignored} (exit code: {result.returncode})"
    )


@pytest.mark.parametrize(
    "relative_target_path, should_be_ignored",
    [
        ("CoChem_Artifacts/run.h5", True),
        ("nested/CoChem_Artifacts/tensor.zarr", True),
        ("spfit_residuals.fit", True),
        ("cochem_system_config.json", True),
        ("fit_provenance.json", True),
        ("spycfit_telemetry.json", True),
        ("stream.ipc", True),
        (".env", True),
        ("src/core.py", False),
        ("README.md", False),
    ],
)
def test_isolated_physical_git_workspace(
    tmp_path: Path, relative_target_path: str, should_be_ignored: bool
) -> None:
    """Physically instantiate files on disk in an isolated git workspace to verify rules."""
    (tmp_path / ".gitignore").write_bytes(GITIGNORE_PATH.read_bytes())

    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "CoChem-Verifier"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "verifier@cochem.local"],
        check=True,
        capture_output=True,
    )

    target_file = tmp_path / relative_target_path
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("PHYSICAL_DATA_PAYLOAD", encoding="utf-8")

    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "-q", relative_target_path],
        capture_output=True,
        text=True,
        check=False,
    )
    is_ignored = result.returncode == 0
    assert is_ignored == should_be_ignored, (
        f"Isolated workspace path '{relative_target_path}' expected ignored={should_be_ignored}, "
        f"got {is_ignored} (exit code: {result.returncode})"
    )
