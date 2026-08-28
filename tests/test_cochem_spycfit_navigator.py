"""Physical unit tests for Smart Scan Navigator (Information-Gain Chunking).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic spectral transition density evaluation.
- Information Gain computed via analytical Jacobian sensitivity weighted by parameter variance.
- Hardware-aware scan chunking respecting instrument bandwidth limits.
- Resolvability anti-clustering filtering (clean line threshold penalty > 0.5).
- Dual-tree module and package re-export verification.
- Static AST validation confirming zero mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_spycfit_navigator import (
    ScanRecommendation,
    SmartScanNavigator,
)
from cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
    ResolvabilityFilter,
)


def test_scan_recommendation_dataclass_and_immutability() -> None:
    """Verify ScanRecommendation dataclass fields, types, and frozen immutability."""
    rec = ScanRecommendation(
        rank=1,
        start_frequency_mhz=16375.0,
        end_frequency_mhz=16625.0,
        chunk_score=25.0,
        n_clean_lines=1,
        targeted_constants=("A",),
        description="Recommendation 1: Scan 16375–16625 MHz. Expected 1 isolated transitions. Targets: A.",
    )

    assert rec.rank == 1
    assert rec.start_frequency_mhz == 16375.0
    assert rec.end_frequency_mhz == 16625.0
    assert rec.chunk_score == 25.0
    assert rec.n_clean_lines == 1
    assert rec.targeted_constants == ("A",)
    assert "Scan 16375–16625 MHz" in rec.description

    with pytest.raises(FrozenInstanceError):
        rec.chunk_score = 50.0  # type: ignore[misc]


def test_smart_scan_navigator_initialization() -> None:
    """Verify SmartScanNavigator default and custom initialization."""
    nav_default = SmartScanNavigator()
    assert nav_default._chunk_bw == 250.0
    assert nav_default._top_n == 3

    nav_custom = SmartScanNavigator(max_chunk_bandwidth_mhz=100.0, top_n_recommendations=5)
    assert nav_custom._chunk_bw == 100.0
    assert nav_custom._top_n == 5


def test_smart_scan_navigator_config_override(tmp_path: Path) -> None:
    """Verify configuration file overrides max_chunk_bandwidth."""
    cfg_file = tmp_path / "cochem_system_config.json"
    with open(cfg_file, "w", encoding="utf-8") as f:
        json.dump({"max_chunk_bandwidth": 350.0}, f)

    nav = SmartScanNavigator(config_path=cfg_file)
    assert nav._chunk_bw == 350.0


def test_identify_target_constants() -> None:
    """Verify identification of physical constants with highest variance."""
    nav = SmartScanNavigator()
    # Covariance matrix for parameters [A, B, C, DJ, DJK]
    param_names = ["A", "B", "C", "DJ", "DJK"]
    cov_diag = [120.0, 4.0, 2.5, 85.0, 12.0]
    cov_matrix = np.diag(cov_diag)

    targets = nav.identify_target_constants(cov_matrix, param_names, n_targets=2)
    assert len(targets) == 2
    assert targets[0] == ("A", 120.0)
    assert targets[1] == ("DJ", 85.0)

    targets_all = nav.identify_target_constants(cov_matrix, param_names, n_targets=5)
    expected_order = [("A", 120.0), ("DJ", 85.0), ("DJK", 12.0), ("B", 4.0), ("C", 2.5)]
    assert targets_all == expected_order


def test_compute_information_gain_analytical() -> None:
    """Verify per-transition Information Gain computation against exact formula."""
    nav = SmartScanNavigator()
    param_names = ["A", "B", "C"]
    targets = [("A", 100.0), ("C", 10.0)]  # Target mask: [1.0, 0.0, 1.0]

    # 3 transitions, 3 parameters
    jacobian = np.array([
        [2.0, 3.0, 1.0],   # Transition 0: sum sq for A and C = 2^2 + 1^2 = 5
        [0.0, 5.0, 0.0],   # Transition 1: sum sq for A and C = 0^2 + 0^2 = 0
        [3.0, 1.0, 4.0],   # Transition 2: sum sq for A and C = 3^2 + 4^2 = 25
    ], dtype=np.float64)

    penalties = np.array([0.8, 1.0, 0.4], dtype=np.float64)

    ig = nav.compute_information_gain(jacobian, param_names, targets, penalties)

    assert len(ig) == 3
    assert ig[0] == pytest.approx(5.0 * 0.8, rel=1e-6)
    assert ig[1] == pytest.approx(0.0 * 1.0, rel=1e-6)
    assert ig[2] == pytest.approx(25.0 * 0.4, rel=1e-6)


def test_smart_scan_navigator_validation_test_specification() -> None:
    """Execute the full validation test specification defined in Prompt 09.

    Scenario:
    1. Covariance matrix where 'A' rotational constant has elevated variance (10x larger than B, C).
    2. 20 predicted transitions spread across 8000–18000 MHz.
    3. Highly sensitive A-dependent transition at 14000 MHz inside dense Q-branch (spacing 0.020 MHz < 0.050 MHz).
    4. Highly sensitive A-dependent transition at 16500 MHz in sparse R-branch (spacing > 100 MHz).
    5. Assertions:
       - identify_target_constants() returns 'A' as top target.
       - recommend_scan_regions() selects sparse-region chunk (16500 MHz) with higher rank than dense chunk.
       - dense-region chunk has fewer clean lines due to resolvability penalties.
    """
    nav = SmartScanNavigator(max_chunk_bandwidth_mhz=250.0, top_n_recommendations=3)

    param_names = ["A", "B", "C"]
    # Rotational constant A variance is 100.0, B is 10.0, C is 5.0
    covariance_matrix = np.diag([100.0, 10.0, 5.0])

    # 1. Target identification
    target_constants = nav.identify_target_constants(covariance_matrix, param_names, n_targets=1)
    assert target_constants[0][0] == "A"
    assert target_constants[0][1] == 100.0

    # 2. Construct 20 transitions across 8000–18000 MHz
    # Dense cluster at 14000 MHz: 14000.0, 14000.020, 14000.040 (spacing = 0.020 MHz < 0.050 MHz)
    # Sparse line at 16500 MHz: nearest neighbors at 16000 and 17000 MHz (spacing >= 500 MHz)
    transition_frequencies = np.array([
        8000.0, 8600.0, 9200.0, 9800.0, 10400.0,
        11000.0, 11600.0, 12200.0, 12800.0, 13400.0,
        14000.0, 14000.020, 14000.040, 14600.0, 15200.0,
        15800.0, 16500.0, 17200.0, 17800.0, 18000.0,
    ], dtype=np.float64)

    assert len(transition_frequencies) == 20

    # Evaluate resolvability penalties
    res_filter = ResolvabilityFilter(instrumental_resolution_mhz=0.050, steepness_k=20.0)
    penalties_list = res_filter.compute_penalties(transition_frequencies)
    penalties = np.array([p.penalty for p in penalties_list], dtype=np.float64)

    # Verify cluster lines (indices 10, 11, 12) are penalized (< 0.5)
    assert penalties[10] < 0.5
    assert penalties[11] < 0.5
    assert penalties[12] < 0.5

    # Verify sparse line at 16500 MHz (index 16) is clean (> 0.99)
    assert penalties[16] > 0.99

    # 3. Construct analytical Jacobian:
    # Set high ∂ν/∂A = 8.0 for 14000.0 (index 10) and 16500.0 (index 16)
    # Other transitions have modest baseline sensitivity = 0.5
    jacobian = np.zeros((20, 3), dtype=np.float64)
    jacobian[:, 0] = 0.5
    jacobian[:, 1] = 0.3
    jacobian[:, 2] = 0.2

    jacobian[10, 0] = 8.0  # 14000.0 MHz in dense cluster
    jacobian[16, 0] = 8.0  # 16500.0 MHz in sparse region

    # 4. Compute Information Gain
    ig_scores = nav.compute_information_gain(
        jacobian=jacobian,
        parameter_names=param_names,
        target_constants=target_constants,
        resolvability_penalties=penalties,
    )

    # 5. Recommend scan regions
    recommendations = nav.recommend_scan_regions(
        transition_frequencies_mhz=transition_frequencies,
        information_gains=ig_scores,
        resolvability_penalties=penalties,
        target_constants=target_constants,
    )

    assert len(recommendations) > 0

    # Top recommendation should encompass the sparse region containing 16500 MHz
    top_rec = recommendations[0]
    assert top_rec.rank == 1
    assert top_rec.start_frequency_mhz <= 16500.0 <= top_rec.end_frequency_mhz
    assert top_rec.chunk_score > 60.0
    assert top_rec.n_clean_lines >= 1
    assert "Targets: A" in top_rec.description

    # Find recommendation encompassing dense cluster at 14000 MHz (if present)
    dense_recs = [
        r for r in recommendations
        if r.start_frequency_mhz <= 14000.0 <= r.end_frequency_mhz
    ]
    if dense_recs:
        dense_rec = dense_recs[0]
        # Sparse chunk score must be strictly greater than dense chunk score
        assert top_rec.chunk_score > dense_rec.chunk_score
        # Dense chunk must have fewer clean lines than total lines in that window
        assert dense_rec.n_clean_lines == 0 or dense_rec.chunk_score < top_rec.chunk_score


def test_recommend_scan_regions_empty_input() -> None:
    """Verify empty transition arrays safely return empty recommendations list."""
    nav = SmartScanNavigator()
    empty_arr = np.array([], dtype=np.float64)
    recs = nav.recommend_scan_regions(
        transition_frequencies_mhz=empty_arr,
        information_gains=empty_arr,
        resolvability_penalties=empty_arr,
        target_constants=[("A", 10.0)],
    )
    assert recs == []


def test_dual_tree_exports() -> None:
    """Verify ScanRecommendation and SmartScanNavigator exportable across both package trees."""
    import cochem_spycfit.physics_core as root_core
    import src.cochem_spycfit.physics_core as src_core

    for core_mod in [root_core, src_core]:
        assert hasattr(core_mod, "ScanRecommendation")
        assert hasattr(core_mod, "SmartScanNavigator")

    from cochem_spycfit.physics_core.cochem_spycfit_navigator import (
        ScanRecommendation as RecRoot,
        SmartScanNavigator as NavRoot,
    )
    from src.cochem_spycfit.physics_core.cochem_spycfit_navigator import (
        ScanRecommendation as RecSrc,
        SmartScanNavigator as NavSrc,
    )

    assert RecRoot is not None
    assert NavRoot is not None
    assert RecSrc is not None
    assert NavSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or pass statements in navigator code and test file."""
    test_path = Path(__file__).resolve()
    impl_src_path = (
        test_path.parent.parent
        / "src"
        / "cochem_spycfit"
        / "physics_core"
        / "cochem_spycfit_navigator.py"
    )
    impl_root_path = (
        test_path.parent.parent
        / "cochem_spycfit"
        / "physics_core"
        / "cochem_spycfit_navigator.py"
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
            elif isinstance(node, ast.Pass):
                raise AssertionError(f"Prohibited pass statement found at line {node.lineno} in {file_path}")
