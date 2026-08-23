"""
CoChem-SpycFit: Comprehensive Physical Unit Test Suite for Legacy Fortran Translator & Pickett Bridge.

Strict Zero-Mock & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or synthetic mocks
- 100% authentic physical spectroscopy data, real Pickett formats, and actual OS subprocess execution
- Validation of strict fixed-width truncation and padding (%10.4E with 'D' exponent)
- Validation of 1980s Pickett format exporters and parsers (.var, .int, .lin, .cat, .egy, .fit)
- Validation of dynamic isotope mass retrieval via Mendeleev and reduced mass computations
- Validation of FortranTaskQueue, FortranBridgeDispatcher, process isolation, and async dispatch
- Validation of segfault trapping and hex-dump telemetry capture
- Validation of ephemeral PickettScratchContext lifecycle and executable deployment routing
- AST static analysis enforcing zero pass statements and zero banned tokens
"""

from __future__ import annotations

import ast
import asyncio
import io
import json
import os
import platform
import queue
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from cochem_spycfit.serialization.cochem_spcat_bridge import (
    FortranBinaryNotFoundError,
    FortranBridgeDispatcher,
    FortranBridgeError,
    FortranExecutionError,
    FortranExecutionResult,
    FortranExecutionSegfaultError,
    FortranExecutionTask,
    FortranExecutionTimeoutError,
    FortranTaskQueue,
    PickettCatFile,
    PickettCatRecord,
    PickettDipoleRecord,
    PickettEgyFile,
    PickettEgyRecord,
    PickettFitResidual,
    PickettFitResult,
    PickettFormatError,
    PickettIntFile,
    PickettLinFile,
    PickettLinRecord,
    PickettParameterRecord,
    PickettScratchContext,
    PickettVarFile,
    calculate_reduced_mass,
    capture_hex_telemetry,
    decode_pickett_quantum,
    dispatch_fortran_binary,
    encode_pickett_quantum,
    format_pickett_line,
    fortran_float_converter,
    get_fortran_deployment_dir,
    get_isotopic_mass,
    is_segfault_exit_code,
    parse_pickett_cat,
    parse_pickett_egy,
    parse_pickett_fit,
    parse_pickett_int,
    parse_pickett_lin,
    parse_pickett_var,
    resolve_fortran_binary,
    write_spycfit_telemetry,
)


# =============================================================================
# 1. FIXTURES & ISOLATED WORKSPACE
# =============================================================================

@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provides an isolated physical directory workspace on disk."""
    ws = tmp_path / "spcat_bridge_workspace"
    ws.mkdir(parents=True, exist_ok=True)
    yield ws


# =============================================================================
# 2. PHYSICAL CONSTANTS & DYNAMIC ISOTOPE RETRIEVAL TESTS (MENDELEEV)
# =============================================================================

class TestPhysicalConstantsAndIsotopes:
    """Validates dynamic retrieval of physical isotopic masses from Mendeleev."""

    def test_mendeleev_element_atomic_weights(self) -> None:
        mass_h = get_isotopic_mass("H")
        mass_c = get_isotopic_mass("C")
        mass_n = get_isotopic_mass("N")
        mass_o = get_isotopic_mass("O")

        assert 1.007 < mass_h < 1.009
        assert 12.010 < mass_c < 12.012
        assert 14.006 < mass_n < 14.008
        assert 15.998 < mass_o < 16.000

    def test_mendeleev_specific_isotopes(self) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_c13 = get_isotopic_mass("C", 13)
        mass_o16 = get_isotopic_mass("O", 16)
        mass_o18 = get_isotopic_mass("O", 18)

        assert abs(mass_c12 - 12.000000) < 1e-5
        assert 13.003 < mass_c13 < 13.004
        assert 15.994 < mass_o16 < 15.995
        assert 17.999 < mass_o18 < 18.000

    def test_calculate_reduced_mass(self) -> None:
        mass_c12 = get_isotopic_mass("C", 12)
        mass_o16 = get_isotopic_mass("O", 16)
        mu_co_12_16 = calculate_reduced_mass(mass_c12, mass_o16)

        expected = (mass_c12 * mass_o16) / (mass_c12 + mass_o16)
        assert abs(mu_co_12_16 - expected) < 1e-9
        assert 6.85 < mu_co_12_16 < 6.87

    def test_calculate_reduced_mass_invalid(self) -> None:
        with pytest.raises(ValueError):
            calculate_reduced_mass(-1.0, 10.0)

    def test_mendeleev_unknown_element_raises_error(self) -> None:
        with pytest.raises(ValueError):
            get_isotopic_mass("NonExistentElement")

    def test_mendeleev_unknown_isotope_mass_raises_error(self) -> None:
        with pytest.raises(ValueError):
            get_isotopic_mass("C", 999)


# =============================================================================
# 3. FORTRAN NUMBER CONVERTER & FIXED-WIDTH FORMATTER TESTS
# =============================================================================

class TestFortranFormatterAndConverter:
    """Validates Fortran number parser and strict fixed-width alignment logic."""

    def test_fortran_float_converter_variations(self) -> None:
        assert fortran_float_converter("1.2345D+04") == 12345.0
        assert fortran_float_converter("-2.5000d-02") == -0.025
        assert fortran_float_converter("3.14159E+00") == pytest.approx(3.14159)
        assert fortran_float_converter("1.2345-05") == pytest.approx(1.2345e-5)
        assert fortran_float_converter("1.2345+02") == pytest.approx(123.45)
        assert fortran_float_converter("  100.5  ") == 100.5
        assert fortran_float_converter("") == 0.0
        assert fortran_float_converter(42.5) == 42.5
        assert fortran_float_converter(100) == 100.0

    def test_fortran_float_converter_invalid(self) -> None:
        with pytest.raises(ValueError):
            fortran_float_converter("invalid_float_string")

    def test_format_pickett_line_exact_width_guarantee(self) -> None:
        # Standard %10.4E with 'D'
        res1 = format_pickett_line(12345.0, width=10, precision=4, exponent_char="D")
        assert len(res1) == 10
        assert res1 == "1.2345D+04"

        # Negative number
        res2 = format_pickett_line(-12345.0, width=10, precision=4, exponent_char="D")
        assert len(res2) == 10
        assert res2 == "-1.234D+04"

        # Zero
        res3 = format_pickett_line(0.0, width=10, precision=4, exponent_char="D")
        assert len(res3) == 10
        assert res3 == "0.0000D+00"

        # Width 22, precision 14
        res4 = format_pickett_line(12345.6789012345, width=22, precision=14, exponent_char="D")
        assert len(res4) == 22
        assert "D+" in res4

        # Non-numeric string input
        res5 = format_pickett_line("header", width=10)
        assert len(res5) == 10
        assert res5 == "    header"

        # Truncation of long string
        res6 = format_pickett_line("excessively_long_string", width=10)
        assert len(res6) == 10
        assert res6 == "excessivel"

    def test_format_pickett_line_very_large_and_small_numbers(self) -> None:
        res_small = format_pickett_line(1.2345e-120, width=10, precision=4, exponent_char="D")
        assert len(res_small) == 10

        res_large = format_pickett_line(-1.2345e120, width=10, precision=4, exponent_char="D")
        assert len(res_large) == 10


class TestPickettQuantumEncoding:
    """Validates JPL/CDMS alphanumeric quantum encoding and decoding."""

    def test_decode_standard_integers(self) -> None:
        assert decode_pickett_quantum(" 0") == 0
        assert decode_pickett_quantum(" 5") == 5
        assert decode_pickett_quantum("42") == 42
        assert decode_pickett_quantum("99") == 99

    def test_decode_alphanumeric_quanta(self) -> None:
        # a0 to z9: 10 to 269
        assert decode_pickett_quantum("a0") == 10
        assert decode_pickett_quantum("a9") == 19
        assert decode_pickett_quantum("b0") == 20
        assert decode_pickett_quantum("z9") == 269

        # A0 to Z9: 100 to 359
        assert decode_pickett_quantum("A0") == 100
        assert decode_pickett_quantum("A9") == 109
        assert decode_pickett_quantum("B0") == 110
        assert decode_pickett_quantum("Z9") == 359

    def test_decode_invalid_quantum_raises_error(self) -> None:
        with pytest.raises(PickettFormatError):
            decode_pickett_quantum("??")

    def test_encode_pickett_quantum(self) -> None:
        assert encode_pickett_quantum(0) == " 0"
        assert encode_pickett_quantum(5) == " 5"
        assert encode_pickett_quantum(42) == "42"
        assert encode_pickett_quantum(100) == "A0"
        assert encode_pickett_quantum(109) == "A9"
        assert encode_pickett_quantum(115) == "B5"


# =============================================================================
# 4. PICKETT .VAR / .PAR PARAMETER FILE TESTS
# =============================================================================

class TestPickettVarFile:
    """Validates creation, export, and parsing of Pickett .var / .par files."""

    def test_pickett_var_model_and_export(self, temp_workspace: Path) -> None:
        params = [
            PickettParameterRecord(param_id=10000, value=127538.91, uncertainty=0.0012, label="A"),
            PickettParameterRecord(param_id=20000, value=25228.14, uncertainty=0.00085, label="B"),
            PickettParameterRecord(param_id=30000, value=21384.02, uncertainty=0.00071, label="C"),
            PickettParameterRecord(param_id=200, value=0.03842, uncertainty=0.00015, label="DJ"),
        ]

        var_file = PickettVarFile(
            title="Methanol Ground State Fit",
            nline=150,
            nopt=0,
            nwarn=0,
            erpar=1.0,
            wtfac=1.0,
            scale=1.0,
            maxit=50,
            parameters=params,
        )

        out_path = temp_workspace / "methanol.var"
        var_text = var_file.export_var(out_path)

        assert out_path.is_file()
        assert "Methanol Ground State Fit" in var_text
        assert "10000" in var_text
        assert "20000" in var_text

        # Verify parsing from disk
        parsed_var = parse_pickett_var(out_path)
        assert parsed_var.title == "Methanol Ground State Fit"
        assert parsed_var.npar == 4
        assert parsed_var.nline == 150
        assert len(parsed_var.parameters) == 4
        assert parsed_var.parameters[0].param_id == 10000
        assert parsed_var.parameters[0].value == pytest.approx(127538.91)
        assert parsed_var.parameters[0].label == "A"
        assert parsed_var.parameters[3].param_id == 200
        assert parsed_var.parameters[3].value == pytest.approx(0.03842)

    def test_parse_pickett_var_string_io(self) -> None:
        raw_text = (
            "Acetonitrile Model\n"
            "    2    50     0     0    1.0000    1.0000    1.0000   50\n"
            "      10000  2.7500000000D+04  1.0000000000D-03 / A\n"
            "      20000  9.1980000000D+03  5.0000000000D-04 / B\n"
        )
        sio = io.StringIO(raw_text)
        parsed = parse_pickett_var(sio)
        assert parsed.title == "Acetonitrile Model"
        assert len(parsed.parameters) == 2
        assert parsed.parameters[1].value == pytest.approx(9198.0)

    def test_parse_pickett_var_empty_raises_error(self) -> None:
        with pytest.raises(PickettFormatError):
            parse_pickett_var("Short line\n")


# =============================================================================
# 5. PICKETT .INT INTENSITY FILE TESTS
# =============================================================================

class TestPickettIntFile:
    """Validates creation, export, and parsing of Pickett .int intensity files."""

    def test_pickett_int_model_and_export(self, temp_workspace: Path) -> None:
        dipoles = [
            PickettDipoleRecord(axis=1, dipole_debye=3.92, comment="mua"),
            PickettDipoleRecord(axis=2, dipole_debye=0.00, comment="mub"),
            PickettDipoleRecord(axis=3, dipole_debye=0.00, comment="muc"),
        ]

        int_file = PickettIntFile(
            title="Acetonitrile Intensities",
            tag=41001,
            qpart=1852.34,
            nline=5000,
            iq=1,
            max_j=99,
            min_freq_mhz=0.0,
            max_freq_mhz=500000.0,
            min_intensity=-10.0,
            temperature_k=300.0,
            dipoles=dipoles,
        )

        out_path = temp_workspace / "ch3cn.int"
        int_text = int_file.export_int(out_path)

        assert out_path.is_file()
        assert "Acetonitrile Intensities" in int_text
        assert "41001" in int_text
        assert "300.00" in int_text

        # Verify parsing from disk
        parsed_int = parse_pickett_int(out_path)
        assert parsed_int.title == "Acetonitrile Intensities"
        assert parsed_int.tag == 41001
        assert parsed_int.qpart == pytest.approx(1852.34)
        assert parsed_int.temperature_k == pytest.approx(300.0)
        assert len(parsed_int.dipoles) == 3
        assert parsed_int.dipoles[0].axis == 1
        assert parsed_int.dipoles[0].dipole_debye == pytest.approx(3.92)


# =============================================================================
# 6. PICKETT .LIN LINE LIST FILE TESTS
# =============================================================================

class TestPickettLinFile:
    """Validates creation, export, and parsing of Pickett .lin line list files."""

    def test_pickett_lin_model_and_export(self, temp_workspace: Path) -> None:
        records = [
            PickettLinRecord(quantum_upper=(1, 0, 1), quantum_lower=(0, 0, 0), frequency_mhz=18397.05, uncertainty_mhz=0.005, weight=1.0, description="1_01-0_00"),
            PickettLinRecord(quantum_upper=(2, 0, 2), quantum_lower=(1, 0, 1), frequency_mhz=36794.10, uncertainty_mhz=0.005, weight=1.0, description="2_02-1_01"),
            PickettLinRecord(quantum_upper=(3, 0, 3), quantum_lower=(2, 0, 2), frequency_mhz=55191.15, uncertainty_mhz=0.005, weight=1.0, description="3_03-2_02"),
        ]

        lin_file = PickettLinFile(
            title="CH3CN Line Measurements",
            records=records,
        )

        out_path = temp_workspace / "ch3cn.lin"
        lin_text = lin_file.export_lin(out_path)

        assert out_path.is_file()
        assert "18397.0500" in lin_text
        assert "36794.1000" in lin_text

        # Verify parsing from disk
        parsed_lin = parse_pickett_lin(out_path)
        assert len(parsed_lin.records) == 3
        assert parsed_lin.records[0].quantum_upper[:3] == (1, 0, 1)
        assert parsed_lin.records[0].quantum_lower[:3] == (0, 0, 0)
        assert parsed_lin.records[0].frequency_mhz == pytest.approx(18397.05)
        assert parsed_lin.records[1].frequency_mhz == pytest.approx(36794.10)


# =============================================================================
# 7. PICKETT OUTPUT PARSER TESTS (.CAT, .EGY, .FIT)
# =============================================================================

class TestPickettOutputParsers:
    """Validates parsing of calculation results (.cat, .egy, .fit)."""

    def test_parse_pickett_cat_file(self, temp_workspace: Path) -> None:
        cat_content = (
            "  18397.0500  0.0050 -4.5123 2   15.6789  3  41001 1404 1 0 1 0 0 0 0 0 0 0 0 0\n"
            "  36794.1000  0.0050 -3.8541 2   30.8912  5  41001 1404 2 0 2 0 0 0 1 0 1 0 0 0\n"
            "  55191.1500  0.0050 -3.4210 2   46.1234  7  41001 1404a0 0a0 0 0 0 9 0 9 0 0 0\n"
        )
        cat_path = temp_workspace / "test_prediction.cat"
        cat_path.write_text(cat_content, encoding="utf-8")

        parsed_cat = parse_pickett_cat(cat_path)
        assert len(parsed_cat.records) == 3

        # Line 1
        r0 = parsed_cat.records[0]
        assert r0.frequency_mhz == pytest.approx(18397.05)
        assert r0.uncertainty_mhz == pytest.approx(0.005)
        assert r0.log_intensity == pytest.approx(-4.5123)
        assert r0.species_tag == 41001
        assert r0.quantum_upper[0] == 1
        assert r0.quantum_lower[0] == 0

        # Line 3 (alphanumeric quantum 'a0' -> 10)
        r2 = parsed_cat.records[2]
        assert r2.frequency_mhz == pytest.approx(55191.15)
        assert r2.quantum_upper[0] == 10
        assert r2.quantum_lower[0] == 9

    def test_parse_pickett_egy_file(self, temp_workspace: Path) -> None:
        egy_content = (
            "   1      0.00000000      0.00000000    1.00000000   0   0   0  A1\n"
            "   2     12.34567890      0.00005000    0.99980000   1   0   1  E1\n"
            "   3     37.03703700      0.00008000    0.99950000   2   0   2  E2\n"
        )
        egy_path = temp_workspace / "test_states.egy"
        egy_path.write_text(egy_content, encoding="utf-8")

        parsed_egy = parse_pickett_egy(egy_path)
        assert len(parsed_egy.records) == 3
        assert parsed_egy.records[0].state_index == 1
        assert parsed_egy.records[0].energy_cm1 == 0.0
        assert parsed_egy.records[1].state_index == 2
        assert parsed_egy.records[1].energy_cm1 == pytest.approx(12.3456789)
        assert parsed_egy.records[1].quantum_numbers == (1, 0, 1)
        assert parsed_egy.records[1].label == "E1"

    def test_parse_pickett_fit_file(self, temp_workspace: Path) -> None:
        fit_content = (
            "METHANOL GROUND STATE FIT\n"
            "STANDARD ERROR OF FIT = 0.8542   VARIANCE = 0.7296\n"
            "LINES FITTED = 2\n"
            "PARAMETERS:\n"
            "    10000   1.2753891000D+05   1.2000000000D-03 / A\n"
            "    20000   2.5228140000D+04   8.5000000000D-04 / B\n"
            "RESIDUALS (OBS-CALC):\n"
            "   1   1  0  1   0  0  0   18397.0500   18397.0450   0.0050   0.0050   1.0000\n"
            "   2   2  0  2   1  0  1   36794.1000   36794.0980   0.0020   0.0050   1.0000\n"
        )
        fit_path = temp_workspace / "test_fit.fit"
        fit_path.write_text(fit_content, encoding="utf-8")

        parsed_fit = parse_pickett_fit(fit_path)
        assert parsed_fit.title == "METHANOL GROUND STATE FIT"
        assert parsed_fit.standard_error == pytest.approx(0.8542)
        assert parsed_fit.variance == pytest.approx(0.7296)
        assert parsed_fit.lines_fitted == 2
        assert len(parsed_fit.parameters) == 2
        assert len(parsed_fit.residuals) == 2
        assert parsed_fit.residuals[0].residual_mhz == pytest.approx(0.0050)
        assert parsed_fit.residuals[1].residual_mhz == pytest.approx(0.0020)


# =============================================================================
# 8. DEPLOYMENT ROUTING & EPHEMERAL SCRATCH CONTEXT TESTS
# =============================================================================

class TestDeploymentAndScratchContext:
    """Validates executable deployment directory routing and scratch context."""

    def test_get_fortran_deployment_dir(self, temp_workspace: Path) -> None:
        custom_bin = temp_workspace / "custom_bin"
        old_env = os.environ.get("COCHEM_BIN_DIR")
        try:
            os.environ["COCHEM_BIN_DIR"] = str(custom_bin)
            resolved = get_fortran_deployment_dir()
            assert resolved == custom_bin.resolve()
            assert resolved.is_dir()
        finally:
            if old_env is not None:
                os.environ["COCHEM_BIN_DIR"] = old_env
            elif "COCHEM_BIN_DIR" in os.environ:
                del os.environ["COCHEM_BIN_DIR"]

    def test_resolve_fortran_binary(self, temp_workspace: Path) -> None:
        custom_bin = temp_workspace / "test_bin_dir"
        custom_bin.mkdir(parents=True, exist_ok=True)
        old_env = os.environ.get("COCHEM_BIN_DIR")
        try:
            os.environ["COCHEM_BIN_DIR"] = str(custom_bin)

            # Create an executable script in deployment dir
            sample_bin = custom_bin / ("spcat.exe" if platform.system().lower() == "windows" else "spcat")
            sample_bin.write_text("#!/bin/sh\necho 'SPCAT'", encoding="utf-8")
            if platform.system().lower() != "windows":
                sample_bin.chmod(0o755)

            found = resolve_fortran_binary("spcat")
            assert found is not None
            assert found.name.startswith("spcat")
        finally:
            if old_env is not None:
                os.environ["COCHEM_BIN_DIR"] = old_env
            elif "COCHEM_BIN_DIR" in os.environ:
                del os.environ["COCHEM_BIN_DIR"]

    def test_pickett_scratch_context_lifecycle(self, temp_workspace: Path) -> None:
        scratch_base = temp_workspace / "scratch_pool"
        with PickettScratchContext(base_dir=scratch_base) as ctx:
            assert ctx.scratch_path.exists()
            assert ctx._active is True

            in_file = ctx.write_input_file("input.var", "Test Var Content")
            assert in_file.is_file()

            out_file = ctx.scratch_path / "output.cat"
            out_file.write_text("Test Output Content", encoding="utf-8")

            read_back = ctx.read_output_file("output.cat")
            assert read_back == "Test Output Content"

            scratch_target = ctx.scratch_path

        # After exiting, scratch directory must be completely purged
        assert not scratch_target.exists()


# =============================================================================
# 9. PROCESS ISOLATION, SEGFAULT TRAPPING & TELEMETRY TESTS
# =============================================================================

class TestProcessIsolationAndTelemetry:
    """Validates segfault identification, hex telemetry capture, and structured logging."""

    def test_is_segfault_exit_code(self) -> None:
        # POSIX segfault & abort
        assert is_segfault_exit_code(139) is True
        assert is_segfault_exit_code(134) is True
        assert is_segfault_exit_code(-11) is True

        # Windows STATUS_ACCESS_VIOLATION
        assert is_segfault_exit_code(0xC0000005) is True
        assert is_segfault_exit_code(-1073741819) is True
        assert is_segfault_exit_code(3221225477) is True

        # Normal exits
        assert is_segfault_exit_code(0) is False
        assert is_segfault_exit_code(1) is False
        assert is_segfault_exit_code(2) is False

    def test_capture_hex_telemetry(self) -> None:
        sample_bytes = b"SPCAT SEGFAULT TRACE DATA 2026"
        hex_trace = capture_hex_telemetry(sample_bytes, max_bytes=10)
        assert isinstance(hex_trace, str)
        assert len(hex_trace.split()) == 10
        assert capture_hex_telemetry(b"") == ""

    def test_write_spycfit_telemetry(self, temp_workspace: Path) -> None:
        log_file = temp_workspace / "telemetry_test.json"
        record = {
            "task_id": "test-task-1234",
            "executable": "spcat",
            "return_code": 0,
            "segfault_detected": False,
            "execution_duration_sec": 0.045,
            "stderr_tail_hex": "41 42 43",
        }

        t_path = write_spycfit_telemetry(record, telemetry_file=log_file)
        assert t_path.is_file()

        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["task_id"] == "test-task-1234"
        assert entry["executable"] == "spcat"
        assert entry["recorded_utc"] is not None


# =============================================================================
# 10. ASYNCHRONOUS TASK QUEUE & DISPATCHER TESTS
# =============================================================================

class TestFortranTaskQueueAndDispatcher:
    """Validates FortranTaskQueue, FortranBridgeDispatcher, and isolated execution."""

    def test_fortran_task_queue_operations(self) -> None:
        q = FortranTaskQueue()
        assert q.empty() is True

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", "print('Queue Task')"],
        )
        task_id = q.submit(task)
        assert task_id == task.task_id
        assert q.qsize() == 1
        assert q.empty() is False

        retrieved = q.get(block=False)
        assert retrieved.task_id == task.task_id
        q.task_done()
        assert q.empty() is True

    def test_dispatch_task_with_real_python_subprocess(self, temp_workspace: Path) -> None:
        telemetry_log = temp_workspace / "telemetry_run.json"
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=10.0, log_telemetry=True, telemetry_file=telemetry_log)

        # Create a Python script task that reads an input file and writes an output file
        worker_script = (
            "import sys\n"
            "from pathlib import Path\n"
            "var_content = Path('input.var').read_text()\n"
            "Path('output.cat').write_text('GENERATED CAT: ' + var_content)\n"
            "sys.stderr.write('CALCULATION COMPLETED CLEANLY\\n')\n"
        )

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", worker_script],
            input_files={"input.var": "Physical Parameter Data 12345"},
            expected_output_files=["output.cat"],
            timeout_sec=10.0,
        )

        result = dispatcher.dispatch_task(task)

        assert result.return_code == 0
        assert result.segfault_detected is False
        assert "output.cat" in result.output_files
        assert "GENERATED CAT: Physical Parameter Data 12345" in result.output_files["output.cat"]
        assert "CALCULATION COMPLETED CLEANLY" in result.stderr
        assert result.telemetry_logged is True
        assert telemetry_log.is_file()

    def test_dispatch_task_timeout(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=0.2, log_telemetry=False)
        hang_script = "import time\ntime.sleep(5.0)\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", hang_script],
            timeout_sec=0.2,
        )

        with pytest.raises(FortranExecutionTimeoutError):
            dispatcher.dispatch_task(task)

    def test_dispatch_task_segfault_trapping(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)

        # Simulate process exiting with segfault return code (139)
        crash_script = "import sys\nsys.stderr.write('FATAL MEMORY CORRUPTION')\nsys.exit(139)\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", crash_script],
            timeout_sec=5.0,
        )

        # Without raising: returns result with segfault_detected=True
        result = dispatcher.dispatch_task(task, raise_on_segfault=False)
        assert result.return_code == 139
        assert result.segfault_detected is True
        assert "FATAL MEMORY CORRUPTION" in result.stderr

        # With raise_on_segfault=True: raises FortranExecutionSegfaultError
        with pytest.raises(FortranExecutionSegfaultError):
            dispatcher.dispatch_task(task, raise_on_segfault=True)

    def test_async_dispatch_task(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)
        async_script = "import sys\nprint('ASYNC DISPATCH SUCCESS')\n"

        task = FortranExecutionTask(
            executable_path=sys.executable,
            arguments=["-c", async_script],
            timeout_sec=5.0,
        )

        result = asyncio.run(dispatcher.async_dispatch_task(task))
        assert result.return_code == 0
        assert "ASYNC DISPATCH SUCCESS" in result.stdout

    def test_convenience_dispatch_fortran_binary(self) -> None:
        script = "import sys\nsys.stdout.write('DIRECT BINARY DISPATCH')\n"
        result = dispatch_fortran_binary(
            executable_path=sys.executable,
            input_files={},
            arguments=["-c", script],
            timeout_sec=5.0,
        )
        assert result.return_code == 0
        assert "DIRECT BINARY DISPATCH" in result.stdout

    def test_process_queue_batch(self, temp_workspace: Path) -> None:
        dispatcher = FortranBridgeDispatcher(default_timeout_sec=5.0, log_telemetry=False)
        task_q = FortranTaskQueue()

        for i in range(3):
            t = FortranExecutionTask(
                executable_path=sys.executable,
                arguments=["-c", f"print('TASK {i}')"],
            )
            task_q.submit(t)

        assert task_q.qsize() == 3
        results = dispatcher.process_queue(task_q)
        assert len(results) == 3
        assert task_q.empty() is True
        for i, res in enumerate(results):
            assert res.return_code == 0
            assert f"TASK {i}" in res.stdout


# =============================================================================
# 11. AST & ZERO-MOCK COMPLIANCE STATIC ANALYSIS
# =============================================================================

class TestZeroMockASTCompliance:
    """Enforces zero pass statements and zero banned tokens across the codebase."""

    def test_cochem_spcat_bridge_ast_compliance(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_spcat_bridge.py"
        assert module_path.is_file()

        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_test_suite_ast_compliance(self) -> None:
        test_path = Path(__file__).resolve()
        assert test_path.is_file()

        source = test_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        pass_nodes: List[int] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Pass):
                pass_nodes.append(node.lineno)

        assert len(pass_nodes) == 0, f"Forbidden 'pass' statements detected at lines: {pass_nodes}"

    def test_no_banned_tokens_in_production_code(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        module_path = repo_root / "src" / "cochem_spycfit" / "serialization" / "cochem_spcat_bridge.py"
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        banned = ["mock", "dummy", "placeholder", "fake", "stub"]
        violations: List[str] = []

        class IdentifierScanner(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                for b in banned:
                    if b in node.name.lower():
                        violations.append(f"Function {node.name} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Name(self, node: ast.Name) -> None:
                for b in banned:
                    if b in node.id.lower():
                        violations.append(f"Variable {node.id} contains banned token '{b}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_Import(self, node: ast.Import) -> None:
                for alias in node.names:
                    for b in banned:
                        if b in alias.name.lower():
                            violations.append(f"Forbidden import '{alias.name}' at line {node.lineno}")
                self.generic_visit(node)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                if node.module:
                    for b in banned:
                        if b in node.module.lower():
                            violations.append(f"Forbidden import from '{node.module}' at line {node.lineno}")
                self.generic_visit(node)

        IdentifierScanner().visit(tree)
        assert len(violations) == 0, f"Banned tokens found: {violations}"
