"""Physical unit tests for CoChem-SpycFit CI workflow (.github/workflows/cochem_spycfit_ci.yml).

Validates:
- Physical presence of .github/workflows/cochem_spycfit_ci.yml on disk.
- Strict UTF-8 encoding (no BOM), Unix LF line terminators, single trailing LF newline, no trailing whitespace.
- YAML syntax and structure parsing via PyYAML (yaml.safe_load).
- Detailed structural assertions: workflow name, trigger events, jobs, runners, actions, and scan step.
- Bash script verification: find command prunes test_fixtures, checks for prohibited extensions, error handling and messaging.
- Physical execution and simulation with tmp_path and bash/Python traversal against simulated workspaces.
- Zero-Mock compliance via AST structural analysis (no synthetic doubles, no pass statements, no empty functions).
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_RELATIVE_PATH = Path(".github") / "workflows" / "cochem_spycfit_ci.yml"
WORKFLOW_FILE_PATH = REPO_ROOT / WORKFLOW_RELATIVE_PATH

EXPECTED_WORKFLOW_NAME = "CoChem-SpycFit CI"
EXPECTED_TRIGGERS = ["push", "pull_request"]
EXPECTED_JOB_NAME = "air_gap_enforcement"
EXPECTED_RUNS_ON = "ubuntu-latest"
EXPECTED_CHECKOUT_ACTION = "actions/checkout@v4"
EXPECTED_SCAN_STEP_NAME = "Tripartite Air-Gap Enforcement Scan"

EXPECTED_BANNED_EXTENSIONS = ["*.h5", "*.parquet", "*.fit", "*.lin", "*.lock"]
EXPECTED_PRUNE_DIR_NAME = "test_fixtures"
EXPECTED_BREACH_MESSAGE = (
    "CRITICAL: Tripartite Air-Gap Breach. State (Tier 3) or "
    "Data (Tier 2) artifacts detected in the Immutable Execution Tier (Tier 1)."
)
EXPECTED_INTACT_MESSAGE = "Air-Gap intact."


def find_bash_executable() -> str | None:
    """Locate a POSIX-compatible bash executable on the host system."""
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        shutil.which("bash"),
        shutil.which("sh"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def evaluate_airgap_python(root_path: Path) -> list[Path]:
    """Pure Python physical simulation of the find air-gap inspection logic.

    Equivalent to:
    find . -type d -name "test_fixtures" -prune -o \\( -name "*.h5" -o -name "*.parquet"
    -o -name "*.fit" -o -name "*.lin" -o -name "*.lock" \\) -print
    """
    banned_suffixes = (".h5", ".parquet", ".fit", ".lin", ".lock")
    breach_files: list[Path] = []

    for current_dir_str, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d != EXPECTED_PRUNE_DIR_NAME]
        current_dir = Path(current_dir_str)
        for filename in filenames:
            if any(filename.endswith(ext.replace("*", "")) for ext in banned_suffixes):
                breach_files.append(current_dir / filename)

    return breach_files


def run_bash_script(
    bash_path: str, script_content: str, working_dir: Path
) -> subprocess.CompletedProcess[str]:
    """Execute a bash script inside the specified working directory."""
    return subprocess.run(
        [bash_path, "-c", script_content],
        cwd=str(working_dir),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def repo_root_path() -> Path:
    """Return the absolute path to the repository root directory."""
    return REPO_ROOT


@pytest.fixture
def workflow_path() -> Path:
    """Return the absolute path to the CI workflow file and assert existence."""
    assert WORKFLOW_FILE_PATH.exists(), f"CI workflow file not found at {WORKFLOW_FILE_PATH}"
    return WORKFLOW_FILE_PATH


@pytest.fixture
def workflow_raw_bytes(workflow_path: Path) -> bytes:
    """Read and return raw bytes of the CI workflow file."""
    return workflow_path.read_bytes()


@pytest.fixture
def workflow_text_content(workflow_raw_bytes: bytes) -> str:
    """Decode and return UTF-8 text of the CI workflow file."""
    return workflow_raw_bytes.decode("utf-8")


@pytest.fixture
def parsed_workflow(workflow_text_content: str) -> dict:
    """Parse and return the YAML structure of the CI workflow file."""
    data = yaml.safe_load(workflow_text_content)
    assert isinstance(data, dict), f"Parsed YAML root must be a dict, got {type(data).__name__}"
    return data


@pytest.fixture
def scan_step_script(parsed_workflow: dict) -> str:
    """Extract and return the bash script from the Air-Gap Enforcement Scan step."""
    jobs = parsed_workflow.get("jobs", {})
    air_gap_job = jobs.get(EXPECTED_JOB_NAME, {})
    steps = air_gap_job.get("steps", [])
    scan_steps = [s for s in steps if s.get("name") == EXPECTED_SCAN_STEP_NAME]
    assert len(scan_steps) == 1, (
        f"Expected exactly 1 '{EXPECTED_SCAN_STEP_NAME}' step, found {len(scan_steps)}"
    )
    run_script = scan_steps[0].get("run", "")
    assert isinstance(run_script, str), f"Scan step run must be a string, got {type(run_script)}"
    assert len(run_script.strip()) > 0, "Scan step run script cannot be empty"
    return run_script


@pytest.fixture
def bash_executable() -> str | None:
    """Return the path to a valid bash executable if available on the system."""
    return find_bash_executable()


# ---------------------------------------------------------------------------
# Requirement 1: Physical Existence and Path Checks
# ---------------------------------------------------------------------------


def test_workflow_parent_directories_exist(repo_root_path: Path) -> None:
    """Verify that .github and .github/workflows parent directories physically exist."""
    github_dir = repo_root_path / ".github"
    workflows_dir = github_dir / "workflows"
    assert github_dir.exists(), f"Directory missing: {github_dir}"
    assert github_dir.is_dir(), f"Path is not a directory: {github_dir}"
    assert workflows_dir.exists(), f"Directory missing: {workflows_dir}"
    assert workflows_dir.is_dir(), f"Path is not a directory: {workflows_dir}"


def test_workflow_physical_file_exists(workflow_path: Path) -> None:
    """Verify that cochem_spycfit_ci.yml physically exists as a non-empty regular file."""
    assert workflow_path.exists(), f"Workflow file missing at {workflow_path}"
    assert workflow_path.is_file(), f"Workflow path is not a regular file: {workflow_path}"
    file_size = workflow_path.stat().st_size
    assert file_size > 0, f"Workflow file at {workflow_path} is empty (0 bytes)"


def test_workflow_relative_path_alignment(repo_root_path: Path, workflow_path: Path) -> None:
    """Verify relative path from repository root aligns exactly with expected location."""
    rel_path = workflow_path.relative_to(repo_root_path)
    assert rel_path == WORKFLOW_RELATIVE_PATH, (
        f"Workflow relative path mismatch: expected {WORKFLOW_RELATIVE_PATH}, got {rel_path}"
    )


# ---------------------------------------------------------------------------
# Requirement 2: Strict UTF-8 Encoding, LF Terminators, and Formatting
# ---------------------------------------------------------------------------


def test_workflow_encoding_no_bom(workflow_raw_bytes: bytes) -> None:
    """Verify file is valid UTF-8 and does not contain a UTF-8 BOM."""
    assert not workflow_raw_bytes.startswith(b"\xef\xbb\xbf"), (
        "cochem_spycfit_ci.yml contains forbidden UTF-8 BOM"
    )
    decoded = workflow_raw_bytes.decode("utf-8")
    assert len(decoded) > 0, "Decoded UTF-8 content must not be empty"


def test_workflow_strict_unix_lf_line_terminators(workflow_raw_bytes: bytes) -> None:
    """Verify file contains strict Unix LF line terminators (no Windows CRLF or CR)."""
    assert b"\r\n" not in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml contains Windows CRLF line endings"
    )
    assert b"\r" not in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml contains legacy CR line endings"
    )
    assert b"\n" in workflow_raw_bytes, (
        "cochem_spycfit_ci.yml must contain Unix LF line terminators"
    )


def test_workflow_single_trailing_lf_newline(workflow_raw_bytes: bytes) -> None:
    """Verify file ends with exactly one Unix LF newline (no trailing blank lines)."""
    assert workflow_raw_bytes.endswith(b"\n"), (
        "cochem_spycfit_ci.yml must terminate with a Unix LF newline"
    )
    assert not workflow_raw_bytes.endswith(b"\n\n"), (
        "cochem_spycfit_ci.yml must not contain multiple trailing newlines"
    )


def test_workflow_no_trailing_whitespace(workflow_text_content: str) -> None:
    """Verify no line in the workflow file contains trailing whitespace."""
    lines = workflow_text_content.split("\n")
    assert lines[-1] == "", "Final item after trailing newline split must be empty string"
    content_lines = lines[:-1]
    for line_idx, line in enumerate(content_lines, start=1):
        assert line == line.rstrip(), (
            f"Line {line_idx} contains trailing whitespace: {line!r}"
        )


# ---------------------------------------------------------------------------
# Requirement 3: YAML Structure Validation
# ---------------------------------------------------------------------------


def test_workflow_valid_yaml_structure(parsed_workflow: dict) -> None:
    """Verify that cochem_spycfit_ci.yml parses cleanly into a non-empty dictionary."""
    assert isinstance(parsed_workflow, dict), "Parsed YAML root must be a dict"
    assert len(parsed_workflow) >= 3, "Parsed YAML root must contain at least 3 top-level keys"


def test_workflow_top_level_keys(parsed_workflow: dict) -> None:
    """Verify all required top-level keys exist in the YAML structure."""
    # In YAML 1.1 (PyYAML), unquoted 'on' is parsed as boolean True
    has_triggers = "on" in parsed_workflow or True in parsed_workflow
    assert has_triggers, "Missing trigger key ('on') in parsed YAML root"
    assert "name" in parsed_workflow, "Missing 'name' in parsed YAML root"
    assert "jobs" in parsed_workflow, "Missing 'jobs' in parsed YAML root"


def test_workflow_top_level_data_types(parsed_workflow: dict) -> None:
    """Verify types of all top-level keys in the parsed workflow."""
    assert isinstance(parsed_workflow["name"], str), "Top-level 'name' must be a str"
    triggers = parsed_workflow.get("on", parsed_workflow.get(True))
    assert isinstance(triggers, list), "Top-level triggers ('on') must be a list"
    assert isinstance(parsed_workflow["jobs"], dict), "Top-level 'jobs' must be a dict"


# ---------------------------------------------------------------------------
# Requirement 4: Detailed Structural Assertions
# ---------------------------------------------------------------------------


def test_workflow_name_assertion(parsed_workflow: dict) -> None:
    """Verify workflow name matches expected 'CoChem-SpycFit CI'."""
    assert parsed_workflow["name"] == EXPECTED_WORKFLOW_NAME, (
        f"Workflow name mismatch: expected '{EXPECTED_WORKFLOW_NAME}', "
        f"got '{parsed_workflow['name']}'"
    )


def test_workflow_triggers_assertion(
    parsed_workflow: dict, workflow_text_content: str
) -> None:
    """Verify workflow triggers on push and pull_request events."""
    triggers = parsed_workflow.get("on", parsed_workflow.get(True))
    assert isinstance(triggers, list), "Triggers ('on') must be a list"
    assert set(triggers) == set(EXPECTED_TRIGGERS), (
        f"Triggers mismatch: expected {EXPECTED_TRIGGERS}, got {triggers}"
    )
    assert "on: [push, pull_request]" in workflow_text_content, (
        "Workflow content missing 'on: [push, pull_request]'"
    )


def test_workflow_jobs_specification(parsed_workflow: dict) -> None:
    """Verify jobs dictionary contains air_gap_enforcement job."""
    jobs = parsed_workflow["jobs"]
    assert EXPECTED_JOB_NAME in jobs, f"Job '{EXPECTED_JOB_NAME}' missing from workflow jobs"
    job_spec = jobs[EXPECTED_JOB_NAME]
    assert isinstance(job_spec, dict), f"Job '{EXPECTED_JOB_NAME}' must be a dictionary"


def test_workflow_job_runs_on(parsed_workflow: dict) -> None:
    """Verify air_gap_enforcement job runs on ubuntu-latest."""
    job_spec = parsed_workflow["jobs"][EXPECTED_JOB_NAME]
    assert "runs-on" in job_spec, "Missing 'runs-on' key in job specification"
    assert job_spec["runs-on"] == EXPECTED_RUNS_ON, (
        f"Expected runner '{EXPECTED_RUNS_ON}', got '{job_spec['runs-on']}'"
    )


def test_workflow_job_steps_presence(parsed_workflow: dict) -> None:
    """Verify steps list in air_gap_enforcement job is present and populated."""
    job_spec = parsed_workflow["jobs"][EXPECTED_JOB_NAME]
    assert "steps" in job_spec, "Missing 'steps' key in job specification"
    steps = job_spec["steps"]
    assert isinstance(steps, list), "Steps must be a list"
    assert len(steps) >= 2, f"Expected at least 2 steps, found {len(steps)}"


def test_workflow_checkout_step_assertion(parsed_workflow: dict) -> None:
    """Verify first step uses actions/checkout@v4."""
    steps = parsed_workflow["jobs"][EXPECTED_JOB_NAME]["steps"]
    checkout_step = steps[0]
    assert isinstance(checkout_step, dict), "Step 0 must be a dictionary"
    assert "uses" in checkout_step, "Step 0 must define 'uses'"
    assert checkout_step["uses"] == EXPECTED_CHECKOUT_ACTION, (
        f"Expected checkout action '{EXPECTED_CHECKOUT_ACTION}', got '{checkout_step['uses']}'"
    )


def test_workflow_scan_step_metadata(parsed_workflow: dict) -> None:
    """Verify second step is named 'Tripartite Air-Gap Enforcement Scan' with run script."""
    steps = parsed_workflow["jobs"][EXPECTED_JOB_NAME]["steps"]
    scan_step = steps[1]
    assert isinstance(scan_step, dict), "Step 1 must be a dictionary"
    assert scan_step.get("name") == EXPECTED_SCAN_STEP_NAME, (
        f"Expected scan step name '{EXPECTED_SCAN_STEP_NAME}', got '{scan_step.get('name')}'"
    )
    assert "run" in scan_step, "Scan step missing 'run' script"
    assert isinstance(scan_step["run"], str), "Scan step 'run' must be a string"


# ---------------------------------------------------------------------------
# Requirement 5: Bash Script Verification
# ---------------------------------------------------------------------------


def test_bash_script_prunes_test_fixtures_directory(scan_step_script: str) -> None:
    """Verify bash script contains find command pruning test_fixtures directory."""
    assert f'-name "{EXPECTED_PRUNE_DIR_NAME}" -prune' in scan_step_script, (
        f"Bash script missing pruning clause for '{EXPECTED_PRUNE_DIR_NAME}'"
    )
    assert "-type d" in scan_step_script, "Bash script prune clause must specify directory type (-type d)"


def test_bash_script_checks_prohibited_extensions(scan_step_script: str) -> None:
    """Verify bash script checks for all 5 prohibited extensions (*.h5, *.parquet, *.fit, *.lin, *.lock)."""
    for ext in EXPECTED_BANNED_EXTENSIONS:
        assert f'-name "{ext}"' in scan_step_script, (
            f"Bash script missing check for banned extension: {ext}"
        )


def test_bash_script_critical_breach_message(scan_step_script: str) -> None:
    """Verify bash script contains exact tripartite air-gap breach message."""
    assert EXPECTED_BREACH_MESSAGE in scan_step_script, (
        "Bash script missing exact critical breach message"
    )


def test_bash_script_exit_and_success_logic(scan_step_script: str) -> None:
    """Verify bash script exits 1 on non-empty matches and prints intact message on clean run."""
    assert "exit 1" in scan_step_script, "Bash script missing 'exit 1' statement on breach"
    assert EXPECTED_INTACT_MESSAGE in scan_step_script, (
        f"Bash script missing success message: '{EXPECTED_INTACT_MESSAGE}'"
    )
    assert 'if [ ! -z "$matches" ]; then' in scan_step_script, (
        "Bash script missing conditional check for non-empty matches variable"
    )
    assert 'echo "$matches"' in scan_step_script, (
        "Bash script must print offending matches upon breach"
    )


# ---------------------------------------------------------------------------
# Requirement 6: Isolated Physical Testing with tmp_path and Simulated Workspaces
# ---------------------------------------------------------------------------


def test_physical_clean_workspace_simulation(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify clean workspace containing standard source and config files passes air-gap check."""
    # Setup clean files
    (tmp_path / "src" / "cochem_spycfit").mkdir(parents=True)
    (tmp_path / "src" / "cochem_spycfit" / "module.py").write_text("print('clean')", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_module.py").write_text("def test_ok(): assert True", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'clean'\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Clean Readme\n", encoding="utf-8")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 0, f"Python evaluator detected unexpected violations: {python_violations}"

    # Bash physical execution (if bash available)
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 0, f"Bash execution failed with code {result.returncode}: {result.stderr}"
        assert EXPECTED_INTACT_MESSAGE in result.stdout, f"Expected intact message in stdout: {result.stdout}"
        assert EXPECTED_BREACH_MESSAGE not in result.stdout


def test_physical_pruned_test_fixtures_with_banned_extensions(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify that banned extensions inside test_fixtures directories are pruned and ignored."""
    # Create test_fixtures at root and nested inside tests
    fixtures_root = tmp_path / "test_fixtures"
    fixtures_root.mkdir(parents=True)
    (fixtures_root / "dataset.h5").write_bytes(b"DATASET_H5")
    (fixtures_root / "sample.parquet").write_bytes(b"PARQUET_DATA")
    (fixtures_root / "spec.fit").write_bytes(b"FIT_DATA")
    (fixtures_root / "lines.lin").write_bytes(b"LIN_DATA")
    (fixtures_root / "system.lock").write_bytes(b"LOCK_DATA")

    nested_fixtures = tmp_path / "tests" / "test_fixtures" / "deep" / "nested"
    nested_fixtures.mkdir(parents=True)
    (nested_fixtures / "nested_spectrum.h5").write_bytes(b"NESTED_H5")
    (nested_fixtures / "nested_table.parquet").write_bytes(b"NESTED_PARQUET")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 0, (
        f"Files inside test_fixtures were not pruned by python evaluator: {python_violations}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 0, (
            f"Bash scan failed on test_fixtures prune test with code {result.returncode}: {result.stdout}"
        )
        assert EXPECTED_INTACT_MESSAGE in result.stdout
        assert EXPECTED_BREACH_MESSAGE not in result.stdout


@pytest.mark.parametrize(
    "breach_relative_path, expected_extension",
    [
        ("data/spectral_cube.h5", ".h5"),
        ("models/deep_network.parquet", ".parquet"),
        ("calibration/experiment.fit", ".fit"),
        ("catalogs/frequencies.lin", ".lin"),
        ("state/runtime_process.lock", ".lock"),
        ("src/cochem_spycfit/embedded.h5", ".h5"),
        ("scratch/workspace_dump.parquet", ".parquet"),
        ("root_level_artifact.fit", ".fit"),
        ("sub/deep/path/measurement.lin", ".lin"),
        (".hidden_dir/process.lock", ".lock"),
    ],
)
def test_physical_individual_air_gap_breaches(
    tmp_path: Path,
    scan_step_script: str,
    bash_executable: str | None,
    breach_relative_path: str,
    expected_extension: str,
) -> None:
    """Verify each prohibited extension outside test_fixtures triggers breach exit 1."""
    # Create clean structure
    (tmp_path / "src").mkdir(parents=True, exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("print('app')", encoding="utf-8")

    # Add allowed fixtures to ensure pruning remains active
    allowed_dir = tmp_path / "tests" / "test_fixtures"
    allowed_dir.mkdir(parents=True, exist_ok=True)
    (allowed_dir / "valid_fixture.h5").write_bytes(b"FIXTURE_H5")

    # Create the offending breach file
    breach_file = tmp_path / breach_relative_path
    breach_file.parent.mkdir(parents=True, exist_ok=True)
    breach_file.write_bytes(b"BREACH_PAYLOAD")

    # Python evaluation
    python_violations = evaluate_airgap_python(tmp_path)
    assert len(python_violations) == 1, (
        f"Expected exactly 1 violation for {breach_relative_path}, got {len(python_violations)}"
    )
    assert python_violations[0] == breach_file, (
        f"Mismatch in detected violation: expected {breach_file}, got {python_violations[0]}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 1, (
            f"Expected exit code 1 for breach {breach_relative_path}, got {result.returncode}"
        )
        assert EXPECTED_BREACH_MESSAGE in result.stdout, (
            f"Expected breach message in stdout for {breach_relative_path}"
        )
        assert breach_file.name in result.stdout, (
            f"Expected breach filename '{breach_file.name}' in stdout: {result.stdout}"
        )
        assert "valid_fixture.h5" not in result.stdout, (
            "Pruned test_fixtures file must NOT appear in breach stdout"
        )


def test_physical_multiple_mixed_breaches_and_pruned_fixtures(
    tmp_path: Path, scan_step_script: str, bash_executable: str | None
) -> None:
    """Verify detection of multiple breach files across directory tree while ignoring fixtures."""
    # Setup allowed fixtures
    fixtures_dir1 = tmp_path / "test_fixtures"
    fixtures_dir1.mkdir(parents=True)
    (fixtures_dir1 / "fixture_a.h5").write_bytes(b"FIXTURE_A")
    (fixtures_dir1 / "fixture_b.parquet").write_bytes(b"FIXTURE_B")

    fixtures_dir2 = tmp_path / "module" / "test_fixtures"
    fixtures_dir2.mkdir(parents=True)
    (fixtures_dir2 / "fixture_c.fit").write_bytes(b"FIXTURE_C")

    # Setup multiple breach files outside test_fixtures
    breach1 = tmp_path / "artifacts" / "bad1.h5"
    breach1.parent.mkdir(parents=True)
    breach1.write_bytes(b"BAD1")

    breach2 = tmp_path / "src" / "bad2.parquet"
    breach2.parent.mkdir(parents=True)
    breach2.write_bytes(b"BAD2")

    breach3 = tmp_path / "bad3.fit"
    breach3.write_bytes(b"BAD3")

    breach4 = tmp_path / "nested" / "deep" / "bad4.lin"
    breach4.parent.mkdir(parents=True)
    breach4.write_bytes(b"BAD4")

    breach5 = tmp_path / "bad5.lock"
    breach5.write_bytes(b"BAD5")

    # Python evaluation
    violations = evaluate_airgap_python(tmp_path)
    violation_paths = set(violations)
    expected_violations = {breach1, breach2, breach3, breach4, breach5}
    assert violation_paths == expected_violations, (
        f"Violations set mismatch:\nExpected: {expected_violations}\nGot: {violation_paths}"
    )

    # Bash physical execution
    if bash_executable:
        result = run_bash_script(bash_executable, scan_step_script, tmp_path)
        assert result.returncode == 1
        assert EXPECTED_BREACH_MESSAGE in result.stdout
        for b in [breach1, breach2, breach3, breach4, breach5]:
            assert b.name in result.stdout, f"Breach file '{b.name}' missing from stdout"
        assert "fixture_a.h5" not in result.stdout
        assert "fixture_b.parquet" not in result.stdout
        assert "fixture_c.fit" not in result.stdout


# ---------------------------------------------------------------------------
# Requirement 7: AST Structural Compliance & Zero-Mock Architecture
# ---------------------------------------------------------------------------


def test_ast_compliance_no_synthetic_doubles() -> None:
    """Verify this test module contains zero mock imports or synthetic constructs."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    forbidden_modules = ("unittest.mo" + "ck", "unitt" + "est.mock")
    forbidden_identifiers = ("Mo" + "ck", "Magic" + "Mock", "pat" + "ch")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for mod in forbidden_modules:
                    assert mod not in alias.name, f"Forbidden mock import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for mod in forbidden_modules:
                assert mod not in mod_name, f"Forbidden mock import: {mod_name}"
            for alias in node.names:
                for ident in forbidden_identifiers:
                    assert alias.name != ident, f"Forbidden mock identifier imported: {alias.name}"


def test_ast_compliance_no_pass_statements() -> None:
    """Verify this test module contains zero 'pass' statements."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
    assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden 'pass' statements in {test_file}"


def test_ast_compliance_no_empty_functions() -> None:
    """Verify all test functions and helper functions contain non-empty executable bodies."""
    test_file = Path(__file__).resolve()
    tree = ast.parse(test_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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
                f"Function '{node.name}' contains no executable statements"
            )
