"""
Unit and physical validation tests for SpfitSandbox and Pickett Fortran wrappers.
Adheres strictly to the physical validation mandate and Anti-Spoofing Directive v2.
Authentic physical constants and transitions for water (H2O).
"""

from __future__ import annotations

import asyncio
import os
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import pytest

from cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper import (
    PickettInputError,
    SpfitSandbox,
    SPFIT_WATSON_A_PARAM_MAP,
)

from pickett_reference import hco_example, program


@pytest.fixture
def h2o_physical_constants() -> Dict[str, float]:
    """Authentic literature rotational and centrifugal distortion constants for H2O (MHz)."""
    return {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "DJ": 37.594,
        "DJK": -172.91,
        "DK": 973.29,
        "dJ": 9.423,
        "dK": 41.25,
    }


@pytest.fixture
def h2o_uncertainties() -> Dict[str, float]:
    """Authentic initial uncertainties for rotational constants (MHz)."""
    return {
        "A": 0.050,
        "B": 0.030,
        "C": 0.020,
        "DJ": 0.005,
        "DJK": 0.020,
        "DK": 0.050,
        "dJ": 0.002,
        "dK": 0.010,
    }


@pytest.fixture
def h2o_five_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """
    Authentic low-J rotational transitions for H2O from literature line lists.
    Format: (J_upper, Ka_upper, Kc_upper, J_lower, Ka_lower, Kc_lower, nu_exp_MHz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985),
        (2, 1, 1, 2, 0, 2, 752033.227),
        (3, 1, 2, 3, 0, 3, 1097364.791),
        (4, 1, 3, 4, 0, 4, 1602219.182),
        (5, 1, 4, 5, 0, 5, 2264148.770),
    )


@pytest.fixture
def h2o_five_transitions_with_unc(h2o_five_transitions):
    """The five transitions with an uncertainty and weight column for formatting tests.

    0.1 MHz and 1.0 are formatting inputs, not measured uncertainties.
    """
    return tuple(t + (0.1, 1.0) for t in h2o_five_transitions)


def _check_spfit_installed() -> bool:
    """Helper to detect if spfit binary is present."""
    suffix = ".exe" if platform.system() == "Windows" else ""
    return shutil.which(f"spfit{suffix}") is not None or shutil.which("spfit") is not None


# =============================================================================
# 1. BINARY RESOLUTION TESTS
# =============================================================================

def test_resolve_binary_missing_raises():
    """Verify that attempting to resolve an nonexistent binary raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Cannot locate"):
        SpfitSandbox._resolve_binary("nonexistent_binary_exec_xyz")


# =============================================================================
# 2. FORTRAN FIXED-WIDTH FORMATTER TESTS (80-COLUMN CONSTRAINT)
# =============================================================================

def test_format_var_file_80_columns(h2o_physical_constants, h2o_uncertainties):
    """Verify format_var_file generates strict 80-column compliant Fortran lines."""
    var_text = SpfitSandbox.format_var_file(
        physical_constants=h2o_physical_constants,
        uncertainties=h2o_uncertainties,
        stem="water_test",
    )

    lines = var_text.splitlines()
    assert len(lines) >= 3, "Var file must contain title, control line, and parameter lines"

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"

    # Check title
    assert "water_test" in lines[0]

    # Check control line (npar is second line)
    ctrl_line = lines[1]
    assert len(ctrl_line) <= 80
    npar = int(ctrl_line[:5])
    assert npar == len(h2o_physical_constants)

    # Check parameter lines containing Fortran 'D' exponent
    param_text = "\n".join(lines[2:])
    assert "8.35840288" in param_text or "8.3584" in param_text
    assert "D+" in param_text or "D-" in param_text or "D0" in param_text or "D" in param_text


def test_format_int_file_80_columns():
    """Verify format_int_file generates strict 80-column compliant Fortran lines."""
    dipoles = {"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0}
    int_text = SpfitSandbox.format_int_file(
        dipole_components=dipoles,
        j_max=10,
        stem="water_int_test",
    )

    lines = int_text.splitlines()
    assert len(lines) >= 3, "Int file must contain title, control line, and dipole lines"

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"

    # Verify control line elements
    ctrl = lines[1]
    assert len(ctrl) <= 80
    # Verify dipole entries
    dipole_section = "\n".join(lines[2:])
    assert "1.854" in dipole_section
    assert "mua" in dipole_section or "mu_a" in dipole_section or "1" in dipole_section


def test_format_lin_file_80_columns_and_alignment(h2o_five_transitions_with_unc):
    """Verify format_lin_file produces strict 80-column fixed-format lines."""
    lin_text = SpfitSandbox.format_lin_file(
        assignments=h2o_five_transitions_with_unc,
        stem="water_lin_test",
    )

    lines = lin_text.splitlines()
    assert len(lines) >= len(h2o_five_transitions_with_unc)

    for idx, line in enumerate(lines, start=1):
        assert len(line) <= 80, f"Line {idx} exceeds 80 columns: '{line}' ({len(line)} chars)"
        if not line.startswith("!"):
            # Check 6I3 upper, 6I3 lower, F13.4, F8.4, F8.4 alignment
            qu_str = line[0:18]
            ql_str = line[18:36]
            freq_str = line[36:49]
            unc_str = line[49:57]
            wt_str = line[57:65]
            assert len(qu_str) == 18
            assert len(ql_str) == 18
            assert float(freq_str.strip()) > 0.0
            assert float(unc_str.strip()) > 0.0
            assert float(wt_str.strip()) > 0.0


def test_format_lin_file_refuses_missing_uncertainty(h2o_five_transitions):
    """Regression: at 7e42507 a missing uncertainty became 0.0100 MHz (amendment §5 class).
    SPFIT weights lines by 1/ERR^2, so the writer now refuses."""
    with pytest.raises(PickettInputError, match="no measurement uncertainty"):
        SpfitSandbox.format_lin_file(assignments=h2o_five_transitions, stem="no_unc")


def test_format_var_file_refuses_unknown_name_and_missing_error(h2o_physical_constants, h2o_uncertainties):
    """Regression: unknown names became code 10000 (A) and missing errors became 0.0,
    which in SPFIT constrains the parameter. Both are now refused."""
    with pytest.raises(PickettInputError, match="no Pickett parameter code"):
        SpfitSandbox.format_var_file({"rho": 1.0}, {"rho": 1e37})
    partial = {k: v for k, v in h2o_uncertainties.items() if k != "DK"}
    with pytest.raises(PickettInputError, match="No a priori error given for 'DK'"):
        SpfitSandbox.format_var_file(h2o_physical_constants, partial)


def test_format_var_file_writes_pickett_sign(h2o_physical_constants, h2o_uncertainties):
    """SPFIT stores -DJ at code 200; a Watson-sign DJ is written negated."""
    var_text = SpfitSandbox.format_var_file(h2o_physical_constants, h2o_uncertainties)
    line_200 = next(ln for ln in var_text.splitlines() if ln.split() and ln.split()[0] == "200")
    value = float(line_200.split()[1].replace("D", "E"))
    assert value == pytest.approx(-h2o_physical_constants["DJ"])


def test_format_int_file_refuses_missing_dipoles():
    """Regression: an empty dipole set became mu_a = 1.0 D; it is now refused."""
    with pytest.raises(PickettInputError, match="No dipole components"):
        SpfitSandbox.format_int_file({}, j_max=5)
    with pytest.raises(PickettInputError, match="Unknown dipole component"):
        SpfitSandbox.format_int_file({"mu_x": 1.0}, j_max=5)


def test_format_lin_file_overflow_protection():
    """Verify buffer overflow prevention when quantum numbers exceed column width."""
    overflow_assignments = ((1000, 1, 0, 1, 0, 1, 556935.985),)
    with pytest.raises(ValueError, match="exceeds 3-column field width"):
        SpfitSandbox.format_lin_file(assignments=overflow_assignments, stem="overflow_test")


# =============================================================================
# 3. .FIT FILE PARSING TESTS
# =============================================================================

def test_parse_fit_file_structure(tmp_path):
    """Parse a genuine SPFIT .fit/.var pair (JPL HCO example run through the pinned build).

    At 7e42507 this test parsed an invented .fit layout and expected parity_status "PASS".
    Genuine SPFIT output has a different layout, and a fit report carries no parity status.
    """
    hco = hco_example()
    sandbox = SpfitSandbox(
        scratch_root=tmp_path / "scratch",
        timeout_seconds=120.0,
        spfit_binary=program("spfit"),
        spcat_binary=program("spcat"),
    )
    run = sandbox.run_spfit_files(hco["hco.par"], hco["hco.lin"], stem="cochem_run")
    fit_path = tmp_path / "cochem_run.fit"
    fit_path.write_text(run.record.outputs["cochem_run.fit"], encoding="ascii")
    (tmp_path / "cochem_run.var").write_text(run.record.outputs["cochem_run.var"], encoding="ascii")

    parsed = SpfitSandbox._parse_fit_file_static(fit_path)

    assert "constants" in parsed
    assert "uncertainties" in parsed
    assert "rmsd_khz" in parsed
    assert "n_lines" in parsed
    assert "parity_status" not in parsed
    assert parsed["n_lines"] == len(run.report.final.residuals) > 0
    assert parsed["rmsd_khz"] == pytest.approx(run.report.final.microwave_rms_mhz * 1000.0)
    assert set(parsed["constants"]) == set(parsed["uncertainties"])
    assert {"A", "B", "C", "-DJ"} <= set(parsed["constants"])


# =============================================================================
# 4. SANDBOX CLEANUP & AIR-GAP ISOLATION TESTS
# =============================================================================

def test_sandbox_tempdir_cleanup_guarantee(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions_with_unc):
    """Verify sandbox directory is completely destroyed with zero file leakage."""
    custom_scratch = Path(tempfile.mkdtemp(prefix="cochem_scratch_root_"))
    try:
        sandbox_holder = []

        # Execute sandboxed workflow
        with tempfile.TemporaryDirectory(dir=custom_scratch) as sandbox:
            sandbox_path = Path(sandbox)
            sandbox_holder.append(sandbox_path)
            stem = "cochem_run"

            # Write files inside sandbox
            (sandbox_path / f"{stem}.var").write_text(
                SpfitSandbox.format_var_file(h2o_physical_constants, h2o_uncertainties, stem),
                encoding="ascii",
            )
            (sandbox_path / f"{stem}.int").write_text(
                SpfitSandbox.format_int_file({"mu_a": 1.854}, j_max=5, stem=stem),
                encoding="ascii",
            )
            (sandbox_path / f"{stem}.lin").write_text(
                SpfitSandbox.format_lin_file(h2o_five_transitions_with_unc, stem),
                encoding="ascii",
            )

            assert sandbox_path.exists()
            assert len(list(sandbox_path.iterdir())) == 3

        # After exiting context manager, sandbox MUST NOT exist
        assert not sandbox_holder[0].exists()
        assert len(list(custom_scratch.iterdir())) == 0
    finally:
        if custom_scratch.exists():
            shutil.rmtree(custom_scratch)


# =============================================================================
# 5. EXECUTION & ERROR TRAPPING TESTS
# =============================================================================

def test_spfit_live_execution(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions):
    """Verify live SPFIT execution against authentic water data when binary is installed."""
    # At 7e42507 any exception here, including a failed assertion, was replaced by an ASE/EMT
    # moment-of-inertia check (amendment §5). The SPFIT outcome is now the test outcome.
    sandbox = SpfitSandbox(timeout_seconds=10.0)
    result = sandbox.run_spfit(
        physical_constants=h2o_physical_constants,
        uncertainties=h2o_uncertainties,
        assignments=h2o_five_transitions,
        dipole_components={"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0},
    )
    assert result["exit_code"] == 0
    assert result["n_lines"] == len(h2o_five_transitions)
    assert result["constants"]["A"] == pytest.approx(h2o_physical_constants["A"], rel=1e-2)
    assert result["constants"]["B"] == pytest.approx(h2o_physical_constants["B"], rel=1e-2)
    assert result["constants"]["C"] == pytest.approx(h2o_physical_constants["C"], rel=1e-2)


def test_spfit_live_async_execution(h2o_physical_constants, h2o_uncertainties, h2o_five_transitions):
    """Verify non-blocking async execution against authentic water data."""
    async def _runner():
        # The ASE/EMT substitution that replaced any exception here was removed; see
        # test_spfit_live_execution.
        sandbox = SpfitSandbox(timeout_seconds=10.0)
        result = await sandbox.run_spfit_async(
            physical_constants=h2o_physical_constants,
            uncertainties=h2o_uncertainties,
            assignments=h2o_five_transitions,
            dipole_components={"mu_a": 1.854, "mu_b": 0.0, "mu_c": 0.0},
        )
        assert result["exit_code"] == 0
        assert result["n_lines"] == len(h2o_five_transitions)

    asyncio.run(_runner())


# =============================================================================
# 6. ERROR & EDGE CASE TRAPPING TESTS
# =============================================================================

def test_scratch_root_custom_and_fallback():
    """Verify custom scratch root resolution logic."""
    custom_dir = Path(tempfile.gettempdir()) / "custom_cochem_scratch_test"
    resolved = SpfitSandbox._resolve_scratch_root(custom_dir)
    assert resolved == custom_dir.resolve()
    assert resolved.is_dir()
    if resolved.exists():
        resolved.rmdir()


def test_format_lin_invalid_record():
    """Verify format_lin_file rejects malformed assignment records with < 7 fields."""
    malformed_assignments = ((1, 1, 0),)
    with pytest.raises(ValueError, match="requires at least 7 fields"):
        SpfitSandbox.format_lin_file(malformed_assignments, stem="bad_record")


def test_format_lin_frequency_overflow():
    """Verify format_lin_file rejects frequencies that overflow Fortran F13.4."""
    overflow_freq = ((1, 1, 0, 1, 0, 1, 1e12),)
    with pytest.raises(ValueError, match="exceeds F13.4 field width"):
        SpfitSandbox.format_lin_file(overflow_freq, stem="bad_freq")


# =============================================================================
# 7. AST ZERO-PASS MANDATE STATIC ANALYSIS
# =============================================================================


def test_zero_pass_mandate() -> None:
    """
    Verify that cochem_spycfit_spcat_wrapper contains zero pass statements.
    """
    import inspect
    import ast
    import cochem_spycfit.serialization.cochem_spycfit_spcat_wrapper as mod

    source_code = inspect.getsource(mod)
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Pass):
            pytest.fail(
                f"Forbidden 'pass' statement found in cochem_spycfit_spcat_wrapper.py at line {getattr(node, 'lineno', '?')}."
            )


# =============================================================================
# 8. DYNAMIC MENDELEEV MASS COMPLIANCE
# =============================================================================

def test_mendeleev_dynamic_mass_resolution() -> None:
    """Verify dynamic elemental and isotopic mass queries directly from Mendeleev without hardcoding."""
    import mendeleev

    h_elem = mendeleev.element("H")
    o_elem = mendeleev.element("O")

    assert float(h_elem.atomic_weight) == pytest.approx(1.008, rel=1e-2)
    assert float(o_elem.atomic_weight) == pytest.approx(15.999, rel=1e-2)

    # Check isotopes dynamically
    h1_iso = next(iso for iso in h_elem.isotopes if iso.mass_number == 1)
    o16_iso = next(iso for iso in o_elem.isotopes if iso.mass_number == 16)
    assert float(h1_iso.mass) == pytest.approx(1.007825, rel=1e-4)
    assert float(o16_iso.mass) == pytest.approx(15.994915, rel=1e-4)



