"""Physical unit tests for Mathematical Parity Auditor (JAX autodiff vs. Pickett SPFIT).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational and centrifugal distortion constants for H2O.
- Physical mathematical divergence checks against strict 0.1 kHz threshold.
- Direct validation of frozen dataclass immutability, boundary conditions, and uncertainty preservation.
- Static AST validation confirming zero synthetic mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from typing import Dict

import pytest

from cochem_spycfit.physics_core.cochem_spycfit_auditor import (
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    format_contrast_dataframe,
)


@pytest.fixture
def h2o_constants() -> Dict[str, float]:
    """Authentic literature Watson A-reduction parameters for H2O in MHz."""
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


def test_parity_verified_identical_constants(h2o_constants: Dict[str, float]) -> None:
    """Verify VERIFIED parity status when JAX and SPFIT constants are identical."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    assert isinstance(warning, ParityWarning)
    assert warning.status == ParityStatus.VERIFIED
    assert warning.status == "VERIFIED"
    assert warning.is_verified is True
    assert warning.has_divergence is False
    assert warning.max_delta_khz == 0.0
    assert warning.divergent_constants == ()
    assert len(warning.contrast_table) == len(h2o_constants)

    for const_name in h2o_constants:
        entry = warning.contrast_table[const_name]
        assert entry["status"] == "✅"
        assert entry["delta_khz"] == 0.0
        assert entry["is_divergent"] is False


def test_parity_warning_perturbed_constant(h2o_constants: Dict[str, float]) -> None:
    """Verify WARNING parity status when constant A is perturbed by +0.0002 MHz (0.2 kHz > 0.1 kHz)."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0002  # 0.2 kHz divergence

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert warning.status == "WARNING"
    assert warning.is_verified is False
    assert warning.has_divergence is True
    assert "A" in warning.divergent_constants
    assert len(warning.divergent_constants) == 1
    assert warning.max_delta_khz == pytest.approx(0.2, abs=0.001)

    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True
    assert warning.contrast_table["B"]["status"] == "✅"
    assert warning.contrast_table["B"]["is_divergent"] is False


def test_parity_exact_boundary_tolerance(h2o_constants: Dict[str, float]) -> None:
    """Verify boundary condition: perturbation of exactly 0.0001 MHz (0.1 kHz) remains VERIFIED."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001  # Exactly 0.1 kHz = tolerance threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.VERIFIED
    assert warning.divergent_constants == ()
    assert warning.max_delta_khz == pytest.approx(0.1, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "✅"
    assert warning.contrast_table["A"]["is_divergent"] is False


def test_parity_slight_threshold_breach(h2o_constants: Dict[str, float]) -> None:
    """Verify boundary breach: perturbation of 0.0001001 MHz (0.1001 kHz > 0.1 kHz) triggers WARNING."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001001  # 0.1001 kHz > 0.1 kHz threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert "A" in warning.divergent_constants
    assert warning.max_delta_khz == pytest.approx(0.1001, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True


def test_uncertainty_preservation(h2o_constants: Dict[str, float]) -> None:
    """Verify uncertainty preservation when jax_uncertainties and spfit_uncertainties are passed."""
    auditor = ParityAuditor()
    jax_unc = {
        "A": 0.0050,
        "B": 0.0030,
        "C": 0.0020,
        "DJ": 0.0001,
        "DJK": 0.0010,
        "DK": 0.0100,
        "dJ": 0.0001,
        "dK": 0.0005,
    }
    spfit_unc = {
        "A": 0.0048,
        "B": 0.0031,
        "C": 0.0019,
        "DJ": 0.0001,
        "DJK": 0.0011,
        "DK": 0.0098,
        "dJ": 0.0001,
        "dK": 0.00049,
    }

    warning = auditor.audit(
        jax_constants=h2o_constants,
        spfit_constants=h2o_constants,
        jax_uncertainties=jax_unc,
        spfit_uncertainties=spfit_unc,
    )

    assert warning.jax_uncertainties == jax_unc
    assert warning.spfit_uncertainties == spfit_unc
    assert warning.contrast_table["A"]["jax_unc"] == 0.0050
    assert warning.contrast_table["A"]["spfit_unc"] == 0.0048
    assert warning.contrast_table["DK"]["jax_unc"] == 0.0100
    assert warning.contrast_table["DK"]["spfit_unc"] == 0.0098


def test_disjoint_constant_sets() -> None:
    """Verify that when parameter sets are disjoint, only common keys are audited."""
    auditor = ParityAuditor()
    jax_constants = {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "HJ": 0.0001,
    }
    spfit_constants = {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "LK": 0.00005,
    }

    warning = auditor.audit(jax_constants=jax_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.VERIFIED
    assert warning.max_delta_khz == 0.0
    assert warning.divergent_constants == ()
    assert set(warning.contrast_table.keys()) == {"A", "B", "C"}
    assert "HJ" not in warning.contrast_table
    assert "LK" not in warning.contrast_table


def test_format_contrast_dataframe(h2o_constants: Dict[str, float]) -> None:
    """Verify format_contrast_dataframe returns valid pandas DataFrame with correct columns and emojis."""
    import pandas as pd  # type: ignore[import-untyped]

    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0003  # 0.3 kHz divergence (⚠️)
    spfit_constants["B"] += 0.00005  # 0.05 kHz within tolerance (✅)

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)
    df = auditor.format_contrast_dataframe(warning)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Constant", "JAX (MHz)", "SPFIT (MHz)", "Δ (kHz)", "Status"]
    assert len(df) == len(h2o_constants)

    a_row = df[df["Constant"] == "A"].iloc[0]
    assert a_row["Status"] == "⚠️"
    assert a_row["Δ (kHz)"] == pytest.approx(0.3, abs=0.001)

    b_row = df[df["Constant"] == "B"].iloc[0]
    assert b_row["Status"] == "✅"
    assert b_row["Δ (kHz)"] == pytest.approx(0.05, abs=0.001)

    # Test direct module-level function call
    df_direct = format_contrast_dataframe(warning)
    assert isinstance(df_direct, pd.DataFrame)
    assert list(df_direct.columns) == list(df.columns)
    assert len(df_direct) == len(df)

    # Test instance helper method parity
    df_helper = warning.to_dataframe()
    assert isinstance(df_helper, pd.DataFrame)
    assert list(df_helper.columns) == list(df.columns)
    assert len(df_helper) == len(df)


def test_dataclass_immutability(h2o_constants: Dict[str, float]) -> None:
    """Verify ParityWarning is a strictly frozen immutable dataclass."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    with pytest.raises(FrozenInstanceError):
        warning.status = ParityStatus.WARNING  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.max_delta_khz = 100.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.divergent_constants = ("A",)  # type: ignore[misc]


def test_spfit_failed_state(h2o_constants: Dict[str, float]) -> None:
    """Verify SPFIT_FAILED state when spfit_constants is None."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=None)

    assert warning.status == ParityStatus.SPFIT_FAILED
    assert warning.is_verified is False
    assert math.isinf(warning.max_delta_khz)
    assert warning.contrast_table == {}
    assert set(warning.divergent_constants) == set(h2o_constants.keys())


def test_custom_tolerance(h2o_constants: Dict[str, float]) -> None:
    """Verify custom tolerance thresholds via constructor and method argument overrides."""
    auditor_relaxed = ParityAuditor(default_tolerance_khz=0.5)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0003  # 0.3 kHz divergence

    # Relaxed tolerance of 0.5 kHz accepts 0.3 kHz delta
    warning_relaxed = auditor_relaxed.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)
    assert warning_relaxed.status == ParityStatus.VERIFIED

    # Overridden strict tolerance of 0.1 kHz flags 0.3 kHz delta
    warning_strict = auditor_relaxed.audit(
        jax_constants=h2o_constants,
        spfit_constants=spfit_constants,
        tolerance_khz=0.1,
    )
    assert warning_strict.status == ParityStatus.WARNING
    assert "A" in warning_strict.divergent_constants


def test_multiple_divergences(h2o_constants: Dict[str, float]) -> None:
    """Verify handling of multiple diverging parameters."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0005  # 0.5 kHz
    spfit_constants["DJK"] += 0.0008  # 0.8 kHz

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert set(warning.divergent_constants) == {"A", "DJK"}
    assert warning.max_delta_khz == pytest.approx(0.8, abs=0.001)


def test_dual_tree_parity_exports() -> None:
    """Verify dual-tree imports from src and proxy directories are properly exported."""
    import cochem_spycfit.physics_core as proxy_core
    import src.cochem_spycfit.physics_core as src_core

    assert hasattr(src_core, "ParityAuditor")
    assert hasattr(src_core, "ParityStatus")
    assert hasattr(src_core, "ParityWarning")
    assert hasattr(src_core, "format_contrast_dataframe")

    assert hasattr(proxy_core, "ParityAuditor")
    assert hasattr(proxy_core, "ParityStatus")
    assert hasattr(proxy_core, "ParityWarning")
    assert hasattr(proxy_core, "format_contrast_dataframe")

    assert "ParityAuditor" in src_core.__all__
    assert "ParityStatus" in src_core.__all__
    assert "ParityWarning" in src_core.__all__
    assert "format_contrast_dataframe" in src_core.__all__

    assert "ParityAuditor" in proxy_core.__all__
    assert "ParityStatus" in proxy_core.__all__
    assert "ParityWarning" in proxy_core.__all__
    assert "format_contrast_dataframe" in proxy_core.__all__


def test_anti_spoofing_compliance_ast() -> None:
    """Enforce strict Zero-Mock and Anti-Spoofing rules via AST inspection."""
    test_file = Path(__file__).resolve()
    impl_file = test_file.parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_auditor.py"

    for file_path in [test_file, impl_file]:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.Name):
                assert node.id not in ("MagicMock", "Mock", "AsyncMock"), f"Prohibited mock identifier in {file_path}"
