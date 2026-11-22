"""Unit tests for the constant-level comparison between native SpycFit and SPFIT fits.

These tests exercise the comparison logic only (SF-CMP-006, SF-CMP-008, SF-CMP-012). The
constants are input values for the comparator; no physical claim about water is tested.
The 0.1 kHz tolerance is a test parameter, not an acceptance tolerance (SF-CMP-008).
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

# Test parameter only. SF-CMP-008 forbids a universal acceptance tolerance.
TEST_TOLERANCE_KHZ = 0.1


@pytest.fixture
def h2o_constants() -> Dict[str, float]:
    """Input values for the comparator, shaped like Watson A-reduction constants (MHz)."""
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
    """Identical complete constant sets are WITHIN_TOLERANCE."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    assert isinstance(warning, ParityWarning)
    assert warning.status == ParityStatus.WITHIN_TOLERANCE
    assert warning.status == "WITHIN_TOLERANCE"
    assert warning.is_within_tolerance is True
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
    """A 0.2 kHz difference in A against a 0.1 kHz tolerance is DIVERGENT."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0002  # 0.2 kHz divergence

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.DIVERGENT
    assert warning.status == "DIVERGENT"
    assert warning.is_within_tolerance is False
    assert warning.has_divergence is True
    assert "A" in warning.divergent_constants
    assert len(warning.divergent_constants) == 1
    assert warning.max_delta_khz == pytest.approx(0.2, abs=0.001)

    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True
    assert warning.contrast_table["B"]["status"] == "✅"
    assert warning.contrast_table["B"]["is_divergent"] is False


def test_parity_exact_boundary_tolerance(h2o_constants: Dict[str, float]) -> None:
    """A difference equal to the tolerance (0.1 kHz) is WITHIN_TOLERANCE."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001  # Exactly 0.1 kHz = tolerance threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WITHIN_TOLERANCE
    assert warning.divergent_constants == ()
    assert warning.max_delta_khz == pytest.approx(0.1, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "✅"
    assert warning.contrast_table["A"]["is_divergent"] is False


def test_parity_slight_threshold_breach(h2o_constants: Dict[str, float]) -> None:
    """A 0.1001 kHz difference against a 0.1 kHz tolerance is DIVERGENT."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001001  # 0.1001 kHz > 0.1 kHz threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.DIVERGENT
    assert "A" in warning.divergent_constants
    assert warning.max_delta_khz == pytest.approx(0.1001, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True


def test_uncertainty_preservation(h2o_constants: Dict[str, float]) -> None:
    """Verify uncertainty preservation when jax_uncertainties and spfit_uncertainties are passed."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
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
    """Constants present in only one set make the comparison INCOMPLETE (SF-CMP-012)."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
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

    assert warning.status == ParityStatus.INCOMPLETE
    assert warning.is_within_tolerance is False
    assert warning.is_complete is False
    assert warning.max_delta_khz == 0.0
    assert warning.divergent_constants == ()
    assert set(warning.contrast_table.keys()) == {"A", "B", "C"}
    assert set(warning.compared_constants) == {"A", "B", "C"}
    assert warning.missing_from_native == ("LK",)
    assert warning.missing_from_spfit == ("HJ",)
    assert "HJ" not in warning.contrast_table
    assert "LK" not in warning.contrast_table


def test_format_contrast_dataframe(h2o_constants: Dict[str, float]) -> None:
    """Verify format_contrast_dataframe returns valid pandas DataFrame with correct columns and emojis."""
    import pandas as pd  # type: ignore[import-untyped]

    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
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
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    with pytest.raises(FrozenInstanceError):
        warning.status = ParityStatus.DIVERGENT  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.max_delta_khz = 100.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.divergent_constants = ("A",)  # type: ignore[misc]


def test_spfit_failed_state(h2o_constants: Dict[str, float]) -> None:
    """A failed SPFIT run (no constants) is FAILED, with nothing compared."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=None)

    assert warning.status == ParityStatus.FAILED
    assert warning.is_within_tolerance is False
    assert warning.max_delta_khz is None
    assert warning.contrast_table == {}
    assert warning.divergent_constants == ()
    assert warning.compared_constants == ()
    assert set(warning.missing_from_spfit) == set(h2o_constants.keys())


def test_custom_tolerance(h2o_constants: Dict[str, float]) -> None:
    """Verify custom tolerance thresholds via constructor and method argument overrides."""
    auditor_relaxed = ParityAuditor(default_tolerance_khz=0.5)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0003  # 0.3 kHz divergence

    # Relaxed tolerance of 0.5 kHz accepts 0.3 kHz delta
    warning_relaxed = auditor_relaxed.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)
    assert warning_relaxed.status == ParityStatus.WITHIN_TOLERANCE

    # Overridden strict tolerance of 0.1 kHz flags 0.3 kHz delta
    warning_strict = auditor_relaxed.audit(
        jax_constants=h2o_constants,
        spfit_constants=spfit_constants,
        tolerance_khz=0.1,
    )
    assert warning_strict.status == ParityStatus.DIVERGENT
    assert "A" in warning_strict.divergent_constants


def test_multiple_divergences(h2o_constants: Dict[str, float]) -> None:
    """Verify handling of multiple diverging parameters."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0005  # 0.5 kHz
    spfit_constants["DJK"] += 0.0008  # 0.8 kHz

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.DIVERGENT
    assert set(warning.divergent_constants) == {"A", "DJK"}
    assert warning.max_delta_khz == pytest.approx(0.8, abs=0.001)


def test_no_overlapping_constants_is_incomplete() -> None:
    """Regression: no shared constant names returned VERIFIED at 7e42507 (amendment §5)."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(
        jax_constants={"A": 835840.288, "B": 435351.717},
        spfit_constants={"10000": 835840.288, "20000": 435351.717},
    )

    assert warning.status == ParityStatus.INCOMPLETE
    assert warning.is_within_tolerance is False
    assert warning.compared_constants == ()
    assert warning.max_delta_khz is None
    assert warning.contrast_table == {}


def test_empty_constant_sets_are_incomplete() -> None:
    """Two empty sets give an empty comparison, which is INCOMPLETE (SF-CMP-012)."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(jax_constants={}, spfit_constants={})

    assert warning.status == ParityStatus.INCOMPLETE
    assert warning.max_delta_khz is None
    assert warning.is_complete is False


def test_non_finite_value_is_not_compared(h2o_constants: Dict[str, float]) -> None:
    """A NaN or infinite constant is reported missing, never compared as a number."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = dict(h2o_constants)
    spfit_constants["DK"] = float("nan")
    native_constants = dict(h2o_constants)
    native_constants["dK"] = float("inf")

    warning = auditor.audit(jax_constants=native_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.INCOMPLETE
    assert warning.missing_from_spfit == ("DK",)
    assert warning.missing_from_native == ("dK",)
    assert "DK" not in warning.contrast_table
    assert "dK" not in warning.contrast_table


def test_required_constants_absent_from_both_sets(h2o_constants: Dict[str, float]) -> None:
    """A required constant missing from both fits makes a matching comparison INCOMPLETE."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    warning = auditor.audit(
        jax_constants=h2o_constants,
        spfit_constants=h2o_constants,
        required_constants=list(h2o_constants) + ["HJ"],
    )

    assert warning.status == ParityStatus.INCOMPLETE
    assert warning.missing_from_native == ("HJ",)
    assert warning.missing_from_spfit == ("HJ",)
    assert warning.divergent_constants == ()


def test_divergence_reported_even_when_coverage_is_partial(h2o_constants: Dict[str, float]) -> None:
    """Divergence is never hidden by partial coverage; missing constants stay listed."""
    auditor = ParityAuditor(default_tolerance_khz=TEST_TOLERANCE_KHZ)
    spfit_constants = {k: v for k, v in h2o_constants.items() if k != "dK"}
    spfit_constants["A"] += 0.001  # 1 kHz

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.DIVERGENT
    assert warning.divergent_constants == ("A",)
    assert warning.missing_from_spfit == ("dK",)
    assert warning.is_complete is False


def test_tolerance_must_be_declared(h2o_constants: Dict[str, float]) -> None:
    """No built-in tolerance exists (SF-CMP-008): an undeclared tolerance is an error."""
    auditor = ParityAuditor()
    with pytest.raises(ValueError, match="No comparison tolerance was declared"):
        auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)


def test_per_constant_tolerances(h2o_constants: Dict[str, float]) -> None:
    """Tolerances can be declared per constant and are recorded in the result."""
    tolerances = {k: 0.1 for k in h2o_constants}
    tolerances["DK"] = 5.0
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["DK"] += 0.002  # 2 kHz, within the 5 kHz tolerance declared for DK
    spfit_constants["A"] += 0.0002  # 0.2 kHz, outside the 0.1 kHz tolerance for A

    warning = auditor.audit(
        jax_constants=h2o_constants,
        spfit_constants=spfit_constants,
        tolerance_khz=tolerances,
    )

    assert warning.status == ParityStatus.DIVERGENT
    assert warning.divergent_constants == ("A",)
    assert warning.tolerance_khz["DK"] == 5.0
    assert warning.contrast_table["DK"]["is_divergent"] is False
    assert warning.contrast_table["DK"]["tolerance_khz"] == 5.0


def test_per_constant_tolerance_missing_entry_raises(h2o_constants: Dict[str, float]) -> None:
    """A tolerance mapping that omits a compared constant is an error, not a default."""
    auditor = ParityAuditor()
    with pytest.raises(ValueError, match="No tolerance declared for constants: dK"):
        auditor.audit(
            jax_constants=h2o_constants,
            spfit_constants=h2o_constants,
            tolerance_khz={k: 0.1 for k in h2o_constants if k != "dK"},
        )


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
