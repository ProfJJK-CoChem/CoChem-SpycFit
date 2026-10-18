"""
CoChem-SpycFit: Comprehensive Phase 4 Physical Unit Test Suite for Pickett SPCAT Bridge.

Strict Directives Enforced:
1. Zero-Mock Physical Mandate: 100% authentic physical execution, zero pass statements,
   zero artificial test doubles.
2. Dynamic Mendeleev: All atomic and isotopic masses dynamically retrieved via Mendeleev.
3. Strict Fortran D exponent notation (%10.4E -> D) and 10-character column alignment.
4. OS-Agnostic Process Isolation (CREATE_NEW_PROCESS_GROUP on Windows, start_new_session on POSIX).
5. OS-Agnostic Segfault Trap (Exit codes 139, 134, 138, -11, -6, -7, 0xC0000005, -1073741819, 3221225477).
6. Hex-Dump Telemetry (Trailing 256 bytes formatted as hex logged to JSON).
7. Orphan Sweeper (Explicit SIGKILL / taskkill process tree cleanup on timeout).
8. Ephemeral pickett_scratch/ lifecycle with guaranteed shutil.rmtree purging.
9. AST static analysis verifying zero pass statements and zero banned mock tokens.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, List

import numpy as np
import pytest

from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettIntFile,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    execute_spcat_safely,
    format_fortran_double,
    format_pickett_line,
    fortran_float_converter,
    get_isotopic_mass,
    is_segfault_exit_code,
    sweep_orphan_process,
    trap_and_log_segfault,
    write_spycfit_telemetry,
)

logger = logging.getLogger("CoChem-SpycFit.TestPhase4")


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def phase4_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated physical filesystem directory."""
    ws = tmp_path / "phase4_spcat_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


# =============================================================================
# 1. FORTRAN DOUBLE PRECISION FORMATTING TESTS
# =============================================================================

class TestFortranDoubleFormattingPhase4:
    """Validates format_fortran_double strict 10-char width and Fortran D notation."""

    def test_format_fortran_double_positive_floats(self) -> None:
        formatted = format_fortran_double(12345.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "1.2345D+04"
        assert "E" not in formatted
        assert "e" not in formatted

    def test_format_fortran_double_small_scientific(self) -> None:
        formatted = format_fortran_double(1.25e-5, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "1.2500D-05"
        assert "D-05" in formatted

    def test_format_fortran_double_negative_number_width_constraint(self) -> None:
        formatted = format_fortran_double(-12345.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "-1.234D+04"
        assert formatted.startswith("-")

    def test_format_fortran_double_zero(self) -> None:
        formatted = format_fortran_double(0.0, width=10, precision=4)
        assert len(formatted) == 10
        assert formatted == "0.0000D+00"

    def test_format_fortran_double_various_widths_and_precisions(self) -> None:
        # Width 22, precision 14
        f22 = format_fortran_double(127538.91024, width=22, precision=14)
        assert len(f22) == 22
        assert "D+" in f22

        # Width 18, precision 10
        f18 = format_fortran_double(0.00123456, width=18, precision=10)
        assert len(f18) == 18
        assert "D-03" in f18

        # Narrow width 8, precision 2
        f8 = format_fortran_double(1.25e-5, width=8, precision=2)
        assert len(f8) == 8
        assert f8 == "1.25D-05"

    def test_format_fortran_double_numpy_and_string_inputs(self) -> None:
        # Numpy float64
        np_val = np.float64(42.5)
        f_np = format_fortran_double(np_val, width=10, precision=4)
        assert len(f_np) == 10
        assert f_np == "4.2500D+01"

        # String with E notation
        f_str_e = format_fortran_double("1.25E-05", width=10, precision=4)
        assert len(f_str_e) == 10
        assert f_str_e == "1.2500D-05"

        # String with D notation
        f_str_d = format_fortran_double("1.25D-05", width=10, precision=4)
        assert len(f_str_d) == 10
        assert f_str_d == "1.2500D-05"

        # String with implicit sign
        f_str_imp = format_fortran_double("1.25-05", width=10, precision=4)
        assert len(f_str_imp) == 10
        assert f_str_imp == "1.2500D-05"

    def test_format_fortran_double_non_numeric_fallback(self) -> None:
        f_head = format_fortran_double("HEADER", width=10)
        assert len(f_head) == 10
        assert f_head == "    HEADER"

        f_trunc = format_fortran_double("EXCESSIVELY_LONG_STRING", width=10)
        assert len(f_trunc) == 10
        assert f_trunc == "EXCESSIVEL"

    def test_format_pickett_line_adherence(self) -> None:
        res_default = format_pickett_line(1.25e-5, width=10, precision=4)
        assert res_default == "1.2500D-05"

        res_custom = format_pickett_line(1.25e-5, width=10, precision=4, exponent_char="E")
        assert res_custom == "1.2500E-05"

    def test_pickett_var_and_int_export_adherence(self, phase4_workspace: Path) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_o16 = get_isotopic_mass("O", 16)
        mu_co = calculate_reduced_mass(mass_c12, mass_o16)
        assert mu_co > 0.0

        var_file = PickettVarFile(
            title="Carbon Monoxide Test Var",
            parameters=[
                PickettParameterRecord(param_id=10000, value=57635.968, uncertainty=0.0001, label="B"),
                PickettParameterRecord(param_id=200, value=0.1835, uncertainty=0.00005, label="-DJ"),
            ],
        )
        var_content = var_file.export_var(phase4_workspace / "co.var")
        assert "57635.968" in var_content or "5.7635" in var_content
        assert "D+" in var_content or "D-" in var_content

        int_file = PickettIntFile(
            title="CO Intensity File",
            tag=28001,
            qpart=107.5,
            dipoles=[
                PickettDipoleRecord(axis=1, dipole_debye=0.110, comment="mua"),
            ],
        )
        int_content = int_file.export_int(phase4_workspace / "co.int")
        assert "28001" in int_content
        assert "0.1100" in int_content


# =============================================================================
# 2. OS-AGNOSTIC SEGFAULT TRAPPING TESTS
# =============================================================================

class TestSegfaultTrapPhase4:
    """Validates OS-agnostic detection of segfaults, memory violations, and crash signals."""

    def test_posix_segfault_codes(self) -> None:
        assert is_segfault_exit_code(139) is True   # SIGSEGV
        assert is_segfault_exit_code(134) is True   # SIGABRT
        assert is_segfault_exit_code(138) is True   # SIGBUS
        assert is_segfault_exit_code(-11) is True   # SIGSEGV negative
        assert is_segfault_exit_code(-6) is True    # SIGABRT negative
        assert is_segfault_exit_code(-7) is True    # SIGBUS negative

    def test_windows_access_violation_codes(self) -> None:
        assert is_segfault_exit_code(0xC0000005) is True      # STATUS_ACCESS_VIOLATION
        assert is_segfault_exit_code(-1073741819) is True     # Signed 32-bit 0xC0000005
        assert is_segfault_exit_code(3221225477) is True      # Unsigned 32-bit 0xC0000005
        assert is_segfault_exit_code(0xC00000FD) is True      # STATUS_STACK_OVERFLOW
        assert is_segfault_exit_code(-1073741571) is True     # Signed 32-bit 0xC00000FD
        assert is_segfault_exit_code(0xC0000028) is True      # STATUS_BAD_STACK
        assert is_segfault_exit_code(0xC0000374) is True      # STATUS_HEAP_CORRUPTION

    def test_normal_exit_codes_not_flagged(self) -> None:
        assert is_segfault_exit_code(0) is False
        assert is_segfault_exit_code(1) is False
        assert is_segfault_exit_code(2) is False
        assert is_segfault_exit_code(42) is False

    def test_trap_and_log_segfault_function(self, phase4_workspace: Path) -> None:
        t_file = phase4_workspace / "telemetry_trap.json"
        simulated_stderr = b"SPCAT calculation fatal fault at address 0x7FFF12345678"

        rec_seg = trap_and_log_segfault(
            stderr_output=simulated_stderr,
            task_id="task-seg-001",
            executable="spcat.exe",
            return_code=139,
            telemetry_file=t_file,
        )
        assert rec_seg["segfault_detected"] is True
        assert rec_seg["return_code"] == 139
        assert rec_seg["telemetry_logged"] is True
        assert len(rec_seg["stderr_tail_hex"]) > 0

        # Verify persistent log on disk
        assert t_file.is_file()
        lines = t_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["task_id"] == "task-seg-001"
        assert data["segfault_detected"] is True


# =============================================================================
# 3. HEX-DUMP TELEMETRY TESTS
# =============================================================================

class TestHexDumpTelemetryPhase4:
    """Validates 256-byte tail hex telemetry extraction and JSON persistence."""

    def test_capture_hex_telemetry_bytes(self) -> None:
        test_bytes = b"COCHEM_SPCAT_CRASH_DUMP_0123456789"
        hex_str = capture_hex_telemetry(test_bytes, max_bytes=256)
        expected_hex = " ".join(f"{b:02X}" for b in test_bytes)
        assert hex_str == expected_hex

    def test_capture_hex_telemetry_string_input(self) -> None:
        test_str = "Error in submatrix diagonalization"
        hex_str = capture_hex_telemetry(test_str, max_bytes=256)
        expected_hex = " ".join(f"{ord(c):02X}" for c in test_str)
        assert hex_str == expected_hex

    def test_capture_hex_telemetry_empty_input(self) -> None:
        assert capture_hex_telemetry(b"", max_bytes=256) == ""
        assert capture_hex_telemetry("", max_bytes=256) == ""

    def test_capture_hex_telemetry_truncates_to_last_256_bytes(self) -> None:
        large_stream = bytes(range(256)) * 4  # 1024 bytes
        hex_str = capture_hex_telemetry(large_stream, max_bytes=256)
        tokens = hex_str.split()
        assert len(tokens) == 256
        # The final 256 bytes should match the last 256 elements of large_stream
        expected_tokens = [f"{b:02X}" for b in large_stream[-256:]]
        assert tokens == expected_tokens

    def test_write_spycfit_telemetry_json(self, phase4_workspace: Path) -> None:
        telemetry_path = phase4_workspace / "test_telemetry.json"
        record = {
            "task_id": "test-uuid-42",
            "executable": "spcat",
            "return_code": 0,
            "segfault_detected": False,
            "execution_duration_sec": 0.123,
            "stderr_tail_hex": "43 4F 43 48 45 4D",
        }
        res_path = write_spycfit_telemetry(record, telemetry_file=telemetry_path)
        assert res_path == telemetry_path
        assert telemetry_path.is_file()

        content = telemetry_path.read_text(encoding="utf-8")
        parsed = json.loads(content.strip())
        assert parsed["task_id"] == "test-uuid-42"
        assert parsed["stderr_tail_hex"] == "43 4F 43 48 45 4D"
        assert "recorded_utc" in parsed


# =============================================================================
# 4. ORPHAN SWEEPER TESTS
# =============================================================================

class TestOrphanSweeperPhase4:
    """Validates OS-specific process tree orphan termination."""

    def test_sweep_orphan_invalid_pid(self) -> None:
        # PIDs <= 0 should safely return without raising
        sweep_orphan_process(0)
        sweep_orphan_process(-1)
        assert True

    def test_sweep_orphan_active_subprocess(self) -> None:
        # Launch real background process
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.poll() is None  # Process is running

        # Sweep orphan process
        sweep_orphan_process(proc)
        time.sleep(0.5)

        # Force kill if still lingering
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception as kill_err:
                logger.debug("Proc kill in test suppressed: %s", kill_err)

        # Confirm process is terminated
        proc.wait(timeout=2.0)
        assert proc.poll() is not None


# =============================================================================
# 5. EPHEMERAL PICKETT SCRATCH DIRECTORY LIFECYCLE TESTS
# =============================================================================

class TestEphemeralPickettScratchContextPhase4:
    """Validates ephemeral pickett_scratch_ directory creation and explicit shutil.rmtree purging."""

    def test_pickett_scratch_lifecycle(self, phase4_workspace: Path) -> None:
        scratch_ctx = PickettScratchContext(base_dir=phase4_workspace, prefix="pickett_scratch_")
        assert "pickett_scratch_" in str(scratch_ctx.scratch_path.name)
        assert not scratch_ctx.scratch_path.exists()

        with scratch_ctx as ctx:
            assert ctx.scratch_path.exists()
            # Stage an authentic test file
            in_file = ctx.write_input_file("test.var", "Title line\n1 0 0 0 1.0 1.0 1.0 50\n")
            assert in_file.is_file()

            out_file = ctx.scratch_path / "test.cat"
            out_file.write_text("100000.0000 0.0100 -3.0000 2 0.0000 1 28001 0 1 0 0 0 0 0\n", encoding="utf-8")
            assert ctx.read_output_file("test.cat").startswith("100000.0000")

        # After exiting context, directory must be purged completely
        assert not scratch_ctx.scratch_path.exists()

    def test_pickett_scratch_error_outside_context(self) -> None:
        scratch_ctx = PickettScratchContext()
        with pytest.raises(RuntimeError):
            scratch_ctx.write_input_file("forbidden.txt", "data")


# =============================================================================
# 6. HIGH-LEVEL EXECUTE_SPCAT_SAFELY ENTRY POINT TESTS
# =============================================================================

class TestExecuteSpcatSafelyPhase4:
    """Validates execute_spcat_safely in process-isolated, segfault-trapped environments."""

    def test_execute_spcat_safely_successful_simulation(self, phase4_workspace: Path) -> None:
        """Transport-only evidence: staging, process launch, output collection, telemetry.

        The executable is a Python stand-in, not SPCAT, and its output is not a catalog.
        At 7e42507 the stand-in printed an SPCAT banner and wrote a catalog-like line
        (amendment §5). The genuine SPCAT run is test_execute_spcat_safely_genuine_spcat.
        """
        telemetry_file = phase4_workspace / "exec_telemetry.json"

        # A stand-in process that copies its staged inputs to the expected output name
        driver_script = phase4_workspace / "transport_stand_in.py"
        driver_script.write_text(
            "import sys\n"
            "from pathlib import Path\n"
            "print('transport stand-in started (not SPCAT)')\n"
            "staged = Path('mol.var').read_text() + Path('mol.int').read_text()\n"
            "Path('mol.cat').write_text('stand-in output, not a catalog\\n' + staged)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )

        input_files = {
            "mol.var": "CO Molecule Var\n1 0 0 0 1.0 1.0 1.0 50\n10000 57635.968 0.0001 / B\n",
            "mol.int": "CO Molecule Int\n28001 107.5 1000 1 99 0.0 500000.0 -10.0 300.0\n1 0.1100 / mua\n",
        }

        result = execute_spcat_safely(
            executable_path=sys.executable,
            arguments=[str(driver_script)],
            input_files=input_files,
            expected_output_files=["mol.cat"],
            timeout_sec=10.0,
            telemetry_file=telemetry_file,
        )

        assert result.return_code == 0
        assert "transport stand-in started (not SPCAT)" in result.stdout
        assert "mol.cat" in result.output_files
        assert result.output_files["mol.cat"].startswith("stand-in output, not a catalog")
        assert input_files["mol.int"] in result.output_files["mol.cat"]
        assert result.segfault_detected is False
        assert result.telemetry_logged is True
        assert telemetry_file.is_file()

    def test_execute_spcat_safely_genuine_spcat(self, phase4_workspace: Path) -> None:
        """Genuine SPCAT through the same entry point reproduces JPL's shipped hco.cat.

        Evidence class: genuine external engine (tests/pickett_reference.py).
        """
        from pickett_reference import hco_example, program

        hco = hco_example()
        result = execute_spcat_safely(
            executable_path=str(program("spcat")),
            arguments=["hco"],
            input_files={"hco.var": hco["hco.var"], "hco.int": hco["hco.int"]},
            expected_output_files=["hco.cat"],
            timeout_sec=120.0,
            telemetry_file=phase4_workspace / "genuine_telemetry.json",
        )

        assert result.return_code == 0
        assert result.segfault_detected is False
        assert result.output_files["hco.cat"] == hco["hco.cat"]

    def test_execute_spcat_safely_timeout_and_orphan_sweep(self, phase4_workspace: Path) -> None:
        telemetry_file = phase4_workspace / "timeout_telemetry.json"
        slow_script = phase4_workspace / "slow_spcat.py"
        slow_script.write_text(
            "import time\n"
            "time.sleep(10.0)\n",
            encoding="utf-8",
        )

        with pytest.raises(FortranExecutionTimeoutError):
            execute_spcat_safely(
                executable_path=sys.executable,
                arguments=[str(slow_script)],
                timeout_sec=0.5,
                telemetry_file=telemetry_file,
            )

        assert telemetry_file.is_file()
        logged_data = json.loads(telemetry_file.read_text(encoding="utf-8").strip())
        assert logged_data["timed_out"] is True
        assert logged_data["return_code"] == -999

    def test_execute_spcat_safely_segfault_detection(self, phase4_workspace: Path) -> None:
        telemetry_file = phase4_workspace / "segfault_telemetry.json"
        seg_script = phase4_workspace / "crash_spcat.py"
        # Simulate exit code 139 (SIGSEGV)
        seg_script.write_text(
            "import sys\n"
            "sys.stderr.write('CRITICAL: Segmentation fault in hamiltonian matrix setup')\n"
            "sys.exit(139)\n",
            encoding="utf-8",
        )

        # Without raise_on_segfault
        result = execute_spcat_safely(
            executable_path=sys.executable,
            arguments=[str(seg_script)],
            timeout_sec=5.0,
            raise_on_segfault=False,
            telemetry_file=telemetry_file,
        )
        assert result.return_code == 139
        assert result.segfault_detected is True
        assert "Segmentation fault" in result.stderr
        assert len(result.stderr_tail_hex) > 0

        # With raise_on_segfault=True
        with pytest.raises(FortranExecutionSegfaultError):
            execute_spcat_safely(
                executable_path=sys.executable,
                arguments=[str(seg_script)],
                timeout_sec=5.0,
                raise_on_segfault=True,
                telemetry_file=telemetry_file,
            )

    def test_execute_spcat_safely_missing_binary(self) -> None:
        with pytest.raises((FortranBinaryNotFoundError, FortranExecutionError)):
            execute_spcat_safely(
                executable_path="non_existent_fortran_spcat_binary_xyz",
                timeout_sec=5.0,
            )


# =============================================================================
# 7. DYNAMIC ISOTOPE RETRIEVAL (MENDELEEV) & PHYSICAL MASS ORDERING TESTS
# =============================================================================

class TestDynamicMendeleevPhysicalPhase4:
    """Validates dynamic physical isotope queries via Mendeleev."""

    def test_dynamic_mendeleev_masses(self) -> None:
        mass_h1 = get_isotopic_mass("H", 1)
        mass_h2 = get_isotopic_mass("H", 2)
        mass_c12 = get_isotopic_mass("C", 12)
        mass_c13 = get_isotopic_mass("C", 13)
        mass_n14 = get_isotopic_mass("N", 14)
        mass_o16 = get_isotopic_mass("O", 16)
        mass_s32 = get_isotopic_mass("S", 32)

        assert mass_h1 < mass_h2
        assert mass_c12 < mass_c13
        assert mass_c12 < mass_n14 < mass_o16 < mass_s32
        assert abs(mass_c12 - 12.0) < 1e-5
        assert 1.0078 < mass_h1 < 1.0079
        assert 15.9949 < mass_o16 < 15.9950

    def test_reduced_mass_isotope_effect(self) -> None:
        m_c12 = get_isotopic_mass("C", 12)
        m_c13 = get_isotopic_mass("C", 13)
        m_o16 = get_isotopic_mass("O", 16)

        mu_12_16 = calculate_reduced_mass(m_c12, m_o16)
        mu_13_16 = calculate_reduced_mass(m_c13, m_o16)

        # 13C-16O is heavier than 12C-16O, so mu(13-16) > mu(12-16)
        assert mu_13_16 > mu_12_16


# =============================================================================
# 8. AST STATIC ANALYSIS AUDIT (ZERO-PASS & ZERO-MOCK MANDATE)
# =============================================================================

class TestAstStaticAnalysisPhase4:
    """Validates AST structure: zero pass statements and zero banned mock imports."""

    def test_zero_pass_statements_in_bridge(self) -> None:
        import cochem_spycfit.serialization.cochem_spcat_bridge as bridge_mod
        bridge_src = Path(bridge_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(bridge_src)
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden pass statements in bridge"

    def test_zero_pass_statements_in_phase4_tests(self) -> None:
        test_src = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(test_src)
        pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
        assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden pass statements in test suite"

    def test_zero_banned_mock_imports(self) -> None:
        import cochem_spycfit.serialization.cochem_spcat_bridge as bridge_mod
        bridge_tree = ast.parse(Path(bridge_mod.__file__).read_text(encoding="utf-8"))
        test_tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

        for tree, label in [(bridge_tree, "bridge"), (test_tree, "test")]:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "mock" not in alias.name.lower(), f"Banned import {alias.name} in {label}"
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert "mock" not in mod.lower(), f"Banned import from {mod} in {label}"
                    for alias in node.names:
                        assert "mock" not in alias.name.lower(), f"Banned imported symbol {alias.name} in {label}"
