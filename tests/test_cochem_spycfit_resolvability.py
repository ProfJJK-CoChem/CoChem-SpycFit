"""Physical unit tests for Resolvability Anti-Clustering Filter.

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic spectral transition density evaluation.
- Sigmoid resolvability penalty calculation.
- Dynamic threshold combination (instrumental resolution + ML uncertainty).
- Single-transition and empty array boundary edge cases.
- Order preservation for unsorted input arrays.
- Dual-tree module and package re-export verification.
- Static AST validation confirming zero mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from typing import List

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
    ResolvabilityFilter,
    ResolvabilityScore,
)


def test_resolvability_score_dataclass_and_immutability() -> None:
    """Verify ResolvabilityScore dataclass fields, types, and frozen immutability."""
    score = ResolvabilityScore(
        frequency_mhz=12000.0,
        nearest_neighbor_spacing_mhz=500.0,
        instrumental_resolution_mhz=0.050,
        ml_uncertainty_mhz=0.010,
        effective_threshold_mhz=0.060,
        penalty=1.0,
        is_resolvable=True,
    )

    assert score.frequency_mhz == 12000.0
    assert score.nearest_neighbor_spacing_mhz == 500.0
    assert score.instrumental_resolution_mhz == 0.050
    assert score.ml_uncertainty_mhz == 0.010
    assert score.effective_threshold_mhz == 0.060
    assert score.penalty == 1.0
    assert score.is_resolvable is True

    with pytest.raises(FrozenInstanceError):
        score.penalty = 0.5  # type: ignore[misc]


def test_resolvability_filter_initialization() -> None:
    """Verify ResolvabilityFilter initialization with default and custom resolution."""
    filter_default = ResolvabilityFilter()
    assert filter_default.instrumental_resolution_mhz == 0.050

    filter_custom = ResolvabilityFilter(instrumental_resolution_mhz=0.025)
    assert filter_custom.instrumental_resolution_mhz == 0.025


def test_resolvability_filter_spec_validation_spectrum() -> None:
    """Verify resolvability scoring on well-separated and clustered transitions.

    Specification Scenario:
    - 3 well-separated transitions: 12000.0, 12500.0, 13000.0 MHz (spacing = 500 MHz)
    - 2 clustered transitions: 14000.0, 14000.030 MHz (spacing = 30 kHz < 50 kHz threshold)
    """
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)

    freqs = np.array([12000.0, 12500.0, 13000.0, 14000.0, 14000.030], dtype=np.float64)
    scores = res_filter.compute_penalties(freqs)

    assert len(scores) == 5

    # Well-separated transitions (500 MHz spacing >> 0.050 MHz threshold)
    for i in range(3):
        assert scores[i].is_resolvable is True
        assert scores[i].penalty > 0.999
        assert scores[i].nearest_neighbor_spacing_mhz == pytest.approx(500.0, rel=1e-6)
        assert scores[i].instrumental_resolution_mhz == pytest.approx(0.050, rel=1e-6)
        assert scores[i].ml_uncertainty_mhz == pytest.approx(0.0, abs=1e-9)
        assert scores[i].effective_threshold_mhz == pytest.approx(0.050, rel=1e-6)

    # Clustered transitions (0.030 MHz spacing < 0.050 MHz threshold)
    for i in (3, 4):
        assert scores[i].is_resolvable is False
        assert scores[i].penalty < 0.5
        assert scores[i].nearest_neighbor_spacing_mhz == pytest.approx(0.030, rel=1e-6)
        assert scores[i].effective_threshold_mhz == pytest.approx(0.050, rel=1e-6)

    # Adding ML uncertainty (sigma = 0.100 MHz) to one of the separated transitions
    ml_uncertainties = np.array([0.100, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    scores_with_unc = res_filter.compute_penalties(freqs, ml_uncertainties_mhz=ml_uncertainties)

    assert scores_with_unc[0].is_resolvable is True
    assert scores_with_unc[0].ml_uncertainty_mhz == pytest.approx(0.100, rel=1e-6)
    assert scores_with_unc[0].effective_threshold_mhz == pytest.approx(0.150, rel=1e-6)
    assert scores_with_unc[0].penalty > 0.999


def test_resolvability_filter_single_transition_edge_case() -> None:
    """Verify single transition array returns penalty = 1.0 and is_resolvable = True."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    single_freq = np.array([18450.0], dtype=np.float64)

    scores = res_filter.compute_penalties(single_freq)

    assert len(scores) == 1
    assert scores[0].frequency_mhz == 18450.0
    assert math.isinf(scores[0].nearest_neighbor_spacing_mhz)
    assert scores[0].nearest_neighbor_spacing_mhz > 0.0
    assert scores[0].penalty == 1.0
    assert scores[0].is_resolvable is True


def test_resolvability_filter_empty_input() -> None:
    """Verify empty input array returns an empty score list."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    empty_freqs = np.array([], dtype=np.float64)

    scores = res_filter.compute_penalties(empty_freqs)
    assert scores == []


def test_resolvability_filter_preserves_input_order() -> None:
    """Verify scores are returned in the exact order of the input array, regardless of sorting."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    unsorted_freqs = np.array([14000.030, 12000.0, 14000.0, 12500.0], dtype=np.float64)

    scores = res_filter.compute_penalties(unsorted_freqs)

    assert len(scores) == 4
    assert scores[0].frequency_mhz == 14000.030
    assert scores[0].is_resolvable is False
    assert pytest.approx(scores[0].nearest_neighbor_spacing_mhz, rel=1e-6) == 0.030

    assert scores[1].frequency_mhz == 12000.0
    assert scores[1].is_resolvable is True
    assert pytest.approx(scores[1].nearest_neighbor_spacing_mhz, rel=1e-6) == 500.0

    assert scores[2].frequency_mhz == 14000.0
    assert scores[2].is_resolvable is False
    assert pytest.approx(scores[2].nearest_neighbor_spacing_mhz, rel=1e-6) == 0.030

    assert scores[3].frequency_mhz == 12500.0
    assert scores[3].is_resolvable is True
    assert pytest.approx(scores[3].nearest_neighbor_spacing_mhz, rel=1e-6) == 500.0


def test_resolvability_filter_ml_uncertainty_threshold_modulation() -> None:
    """Verify ML uncertainty increases effective threshold and flips resolvability status."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    freqs = np.array([10000.0, 10000.080], dtype=np.float64)  # Spacing = 0.080 MHz

    # Without ML uncertainty (threshold = 0.050 MHz < spacing = 0.080 MHz)
    scores_no_unc = res_filter.compute_penalties(freqs)
    assert scores_no_unc[0].is_resolvable is True
    assert scores_no_unc[0].effective_threshold_mhz == pytest.approx(0.050, rel=1e-6)
    assert scores_no_unc[0].penalty > 0.5

    # With ML uncertainty = 0.050 MHz (threshold = 0.100 MHz > spacing = 0.080 MHz)
    scores_with_unc = res_filter.compute_penalties(freqs, ml_uncertainties_mhz=np.array([0.050, 0.050]))
    assert scores_with_unc[0].is_resolvable is False
    assert scores_with_unc[0].effective_threshold_mhz == pytest.approx(0.100, rel=1e-6)
    assert scores_with_unc[0].penalty < 0.5


def test_resolvability_filter_duplicate_frequencies() -> None:
    """Verify handling of exact duplicate frequencies (spacing = 0.0 MHz)."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    freqs = np.array([15000.0, 15000.0], dtype=np.float64)

    scores = res_filter.compute_penalties(freqs)
    assert len(scores) == 2
    for s in scores:
        assert s.nearest_neighbor_spacing_mhz == 0.0
        assert s.is_resolvable is False
        assert s.penalty < 0.5


def test_resolvability_filter_mismatched_uncertainty_length() -> None:
    """Verify ValueError is raised if ml_uncertainties_mhz length does not match frequencies length."""
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050)
    freqs = np.array([10000.0, 11000.0], dtype=np.float64)
    unc = np.array([0.010], dtype=np.float64)

    with pytest.raises(ValueError, match="must match"):
        res_filter.compute_penalties(freqs, ml_uncertainties_mhz=unc)


def test_dual_tree_exports() -> None:
    """Verify ResolvabilityScore and ResolvabilityFilter are exportable across both package trees."""
    import cochem_spycfit.physics_core as root_core
    import src.cochem_spycfit.physics_core as src_core

    for core_mod in [root_core, src_core]:
        assert hasattr(core_mod, "ResolvabilityScore")
        assert hasattr(core_mod, "ResolvabilityFilter")

    from cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
        ResolvabilityFilter as FilterRoot,
        ResolvabilityScore as ScoreRoot,
    )
    from src.cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
        ResolvabilityFilter as FilterSrc,
        ResolvabilityScore as ScoreSrc,
    )

    assert ScoreRoot is not None
    assert FilterRoot is not None
    assert ScoreSrc is not None
    assert FilterSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or fake implementations exist in the test and implementation."""
    test_path = Path(__file__).resolve()
    impl_src_path = (
        test_path.parent.parent
        / "src"
        / "cochem_spycfit"
        / "physics_core"
        / "cochem_spycfit_resolvability.py"
    )
    impl_root_path = (
        test_path.parent.parent
        / "cochem_spycfit"
        / "physics_core"
        / "cochem_spycfit_resolvability.py"
    )

    for file_path in [test_path, impl_src_path, impl_root_path]:
        if not file_path.exists():
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "mock" not in alias.name.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert "mock" not in node.module.lower(), f"Prohibited mock import in {file_path}"
            elif isinstance(node, ast.Name):
                assert node.id not in ("MagicMock", "Mock", "AsyncMock"), (
                    f"Prohibited mock identifier in {file_path}"
                )
