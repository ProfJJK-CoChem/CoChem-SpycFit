"""
CoChem-SpycFit: Unit Test Suite for AI Assigner & Active PES Refinement Core.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or mocks
- 100% authentic physical and mathematical calculations in JAX FP64
- Full mathematical validation of D-optimality (Shannon Information Gain), A-optimality (trace variance reduction),
  Sherman-Morrison rank-1 covariance updating, active PES refinement triggering (The Healing Loop),
  and pure PyTree functional contracts.
"""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_assign_ml import (
    ActivePESHealingConfig,
    AssignMLError,
    DLPNOForbiddenError,
    EmptyCandidateError,
    InformationCriterion,
    InformationGainConfig,
    InvalidCovarianceError,
    InvalidJacobianError,
    InvalidResidualError,
    back_calculate_driving_coordinates,
    calculate_information_gain,
    flag_unresolved_residuals,
    generate_pes_healing_payload,
    rank_candidate_transitions,
    update_covariance_after_measurement,
)

# Enable JAX FP64 precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. MATHEMATICAL VALIDATION OF INFORMATION GAIN & OPTIMALITY CRITERIA
# =============================================================================

def test_shannon_information_gain_analytical_match() -> None:
    """Verifies D-optimal Shannon Information Gain matches exact analytical log-determinant reduction."""
    # 2x2 Parameter Covariance Matrix
    # Sigma = [[4.0, 1.0], [1.0, 2.0]] -> det(Sigma) = 4*2 - 1 = 7.0
    cov = jnp.array([[4.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    # Sensitivity gradient vector j = [1.0, 2.0]
    # j^T Sigma j = 1*(4*1 + 1*2) + 2*(1*1 + 2*2) = 1*6 + 2*5 = 16.0
    jac = jnp.array([1.0, 2.0], dtype=jnp.float64)
    meas_var = 1.0

    # Calculate information gain via cochem_assign_ml
    gain = calculate_information_gain(
        covariance_matrix=cov,
        jacobian=jac,
        measurement_variance=meas_var,
        criterion=InformationCriterion.D_OPTIMAL,
    )

    # Analytical Shannon gain in bits: 0.5 * log2(1 + 16 / 1) = 0.5 * log2(17)
    expected_gain = 0.5 * math.log2(17.0)
    diff = abs(float(gain) - expected_gain)
    assert diff < 1e-12, f"Shannon Information Gain deviation {diff} exceeds FP64 limit."

    # Compare directly to full log-determinant reduction: 0.5 * (log2(det(Sigma)) - log2(det(Sigma_new)))
    cov_new = update_covariance_after_measurement(cov, jac, meas_var)
    det_prior = float(jnp.linalg.det(cov))
    det_post = float(jnp.linalg.det(cov_new))
    log_det_reduction = 0.5 * (math.log2(det_prior) - math.log2(det_post))
    diff_det = abs(float(gain) - log_det_reduction)
    assert diff_det < 1e-12, f"Log-det reduction mismatch: {diff_det}"


def test_a_optimal_trace_reduction_analytical_match() -> None:
    """Verifies A-optimal trace reduction matches exact Tr(Sigma) - Tr(Sigma_new)."""
    cov = jnp.array([[4.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    jac = jnp.array([1.0, 2.0], dtype=jnp.float64)
    meas_var = 1.0

    gain_a = calculate_information_gain(
        covariance_matrix=cov,
        jacobian=jac,
        measurement_variance=meas_var,
        criterion=InformationCriterion.A_OPTIMAL,
    )

    # Sigma^2 = [[17, 6], [6, 5]]
    # j^T Sigma^2 j = 1*17 + 2*6*1*2 + 4*5 = 17 + 24 + 20 = 61.0
    # j^T Sigma j = 16.0
    # Delta Tr = 61.0 / (1.0 + 16.0) = 61.0 / 17.0
    expected_trace_reduction = 61.0 / 17.0
    diff_a = abs(float(gain_a) - expected_trace_reduction)
    assert diff_a < 1e-12, f"A-optimal trace reduction mismatch: {diff_a}"

    # Verify directly against trace difference of covariance matrices
    cov_new = update_covariance_after_measurement(cov, jac, meas_var)
    tr_prior = float(jnp.trace(cov))
    tr_post = float(jnp.trace(cov_new))
    actual_tr_diff = tr_prior - tr_post
    diff_tr_direct = abs(float(gain_a) - actual_tr_diff)
    assert diff_tr_direct < 1e-12, f"Direct trace difference mismatch: {diff_tr_direct}"


def test_g_optimal_predictive_variance() -> None:
    """Verifies G-optimal criterion returns normalized predictive variance j^T Sigma j / sigma^2."""
    cov = jnp.array([[3.0, 0.5], [0.5, 1.5]], dtype=jnp.float64)
    jac = jnp.array([2.0, -1.0], dtype=jnp.float64)
    meas_var = 2.0

    # j^T Sigma j = 2*(3*2 + 0.5*-1) - 1*(0.5*2 + 1.5*-1) = 2*(5.5) - 1*(-0.5) = 11.0 + 0.5 = 11.5
    # G-optimality = 11.5 / 2.0 = 5.75
    gain_g = calculate_information_gain(
        covariance_matrix=cov,
        jacobian=jac,
        measurement_variance=meas_var,
        criterion=InformationCriterion.G_OPTIMAL,
    )

    expected_g = 11.5 / 2.0
    assert abs(float(gain_g) - expected_g) < 1e-12


def test_sherman_morrison_precision_matrix_inversion() -> None:
    """Verifies Sherman-Morrison rank-1 update satisfies (Sigma_new)^(-1) = Sigma^(-1) + w * j * j^T."""
    # 3x3 positive definite covariance matrix
    a_mat = np.array([[2.0, 0.5, 0.2], [0.5, 3.0, 0.8], [0.2, 0.8, 1.5]], dtype=np.float64)
    cov = jnp.array(a_mat, dtype=jnp.float64)
    jac = jnp.array([1.2, -0.4, 0.9], dtype=jnp.float64)
    meas_var = 0.5

    cov_new = update_covariance_after_measurement(cov, jac, meas_var)

    # Invert Sigma_prior and Sigma_post
    prec_prior = jnp.linalg.inv(cov)
    prec_post = jnp.linalg.inv(cov_new)

    # Rank-1 precision update: Prec_post = Prec_prior + (1/meas_var) * j * j^T
    w = 1.0 / meas_var
    expected_prec_post = prec_prior + w * jnp.outer(jac, jac)

    prec_diff = float(jnp.max(jnp.abs(prec_post - expected_prec_post)))
    assert prec_diff < 1e-10, f"Precision matrix update mismatch: {prec_diff}"


# =============================================================================
# 2. VECTORIZED CANDIDATE RANKING & INTENSITY WEIGHTING
# =============================================================================

def test_vectorized_candidate_ranking_order() -> None:
    """Verifies candidate transition ranking correctly sorts transitions by information gain."""
    # 3 parameters (e.g. rotational constants A, B, C)
    cov = jnp.diag(jnp.array([100.0, 25.0, 4.0], dtype=jnp.float64))

    # 4 Candidate transitions with distinct sensitivities:
    # Cand 0: sensitive only to C (lowest uncertainty parameter) -> small gain
    # Cand 1: sensitive only to A (highest uncertainty parameter) -> highest gain
    # Cand 2: sensitive to B -> medium gain
    # Cand 3: sensitive to A and B -> very high gain
    candidate_jacs = jnp.array([
        [0.0, 0.0, 2.0],   # Cand 0: j^T Sigma j = 4 * 4 = 16
        [2.0, 0.0, 0.0],   # Cand 1: j^T Sigma j = 4 * 100 = 400
        [0.0, 2.0, 0.0],   # Cand 2: j^T Sigma j = 4 * 25 = 100
        [2.0, 2.0, 0.0],   # Cand 3: j^T Sigma j = 400 + 100 = 500
    ], dtype=jnp.float64)

    ranking_result = rank_candidate_transitions(
        covariance_matrix=cov,
        candidate_jacobians=candidate_jacs,
        measurement_variances=1.0,
        criterion=InformationCriterion.D_OPTIMAL,
    )

    ranked_indices = [int(idx) for idx in ranking_result["ranked_indices"]]
    # Expected order: Cand 3 (500) > Cand 1 (400) > Cand 2 (100) > Cand 0 (16)
    assert ranked_indices == [3, 1, 2, 0]
    assert ranking_result["best_candidate_index"] == 3
    assert ranking_result["total_candidates"] == 4


def test_intensity_weighted_ranking() -> None:
    """Verifies that line intensity weighting promotes strong observable transitions."""
    cov = jnp.diag(jnp.array([100.0, 100.0], dtype=jnp.float64))

    # Cand 0: high information gain (j=[2,0] -> q=400), but virtually zero intensity (0.001)
    # Cand 1: moderate information gain (j=[1,0] -> q=100), but very high intensity (1.0)
    candidate_jacs = jnp.array([
        [2.0, 0.0],
        [1.0, 0.0],
    ], dtype=jnp.float64)

    intensities = jnp.array([0.001, 1.0], dtype=jnp.float64)

    # Without intensity weight: Cand 0 wins
    res_unweighted = rank_candidate_transitions(
        covariance_matrix=cov,
        candidate_jacobians=candidate_jacs,
        intensities=intensities,
        intensity_weight=0.0,
    )
    assert res_unweighted["best_candidate_index"] == 0

    # With intensity weight=1.0: Cand 1 wins because Cand 0 intensity is negligible
    res_weighted = rank_candidate_transitions(
        covariance_matrix=cov,
        candidate_jacobians=candidate_jacs,
        intensities=intensities,
        intensity_weight=1.0,
    )
    assert res_weighted["best_candidate_index"] == 1


# =============================================================================
# 3. ACTIVE PES REFINEMENT TRIGGER & THE HEALING LOOP
# =============================================================================

def test_flag_unresolved_residuals_viable_and_unviable() -> None:
    """Verifies that large (O-C) residuals with high confidence trigger Active PES Refinement."""
    # 5 experimental line assignments:
    # Line 0: O-C = 0.02 MHz (converged)
    # Line 1: O-C = -0.04 MHz (converged)
    # Line 2: O-C = 14.8 MHz (massive unresolvable residual, confidence 0.95) -> FLAG
    # Line 3: O-C = -8.2 MHz (massive unresolvable residual, confidence 0.85) -> FLAG
    # Line 4: O-C = 25.0 MHz (massive residual, but low confidence 0.20 -> spurious artifact, DO NOT FLAG)
    residuals = jnp.array([0.02, -0.04, 14.8, -8.2, 25.0], dtype=jnp.float64)
    confidences = jnp.array([0.99, 0.98, 0.95, 0.85, 0.20], dtype=jnp.float64)

    # Quantum state descriptors: (J, Ka, Kc) quantum numbers
    quantum_states = {
        "J": jnp.array([1, 2, 8, 9, 15], dtype=jnp.int32),
        "Ka": jnp.array([0, 1, 4, 5, 2], dtype=jnp.int32),
        "Kc": jnp.array([1, 2, 4, 5, 13], dtype=jnp.int32),
    }

    report = flag_unresolved_residuals(
        residuals=residuals,
        states=quantum_states,
        threshold=5.0,
        confidence_scores=confidences,
        min_confidence=0.70,
    )

    assert report["is_triggered"] is True
    assert report["flagged_count"] == 2
    flagged_idx = [int(i) for i in report["flagged_indices"]]
    assert flagged_idx == [2, 3]

    # Verify flagged quantum states PyTree structure
    flagged_states = report["flagged_states"]
    assert list(np.array(flagged_states["J"])) == [8, 9]
    assert list(np.array(flagged_states["Ka"])) == [4, 5]
    assert list(np.array(flagged_states["Kc"])) == [4, 5]

    assert report["max_residual"] == pytest.approx(14.8, abs=1e-5)
    assert report["mean_residual"] == pytest.approx(11.5, abs=1e-5)


def test_flag_unresolved_residuals_no_trigger() -> None:
    """Verifies that all converged residuals produce is_triggered=False and empty state PyTrees."""
    residuals = jnp.array([0.005, -0.012, 0.034], dtype=jnp.float64)
    states = {"J": jnp.array([1, 2, 3], dtype=jnp.int32)}

    report = flag_unresolved_residuals(
        residuals=residuals,
        states=states,
        threshold=5.0,
    )

    assert report["is_triggered"] is False
    assert report["flagged_count"] == 0
    assert len(report["flagged_indices"]) == 0
    assert report["max_residual"] == 0.0
    assert len(report["flagged_states"]["J"]) == 0


# =============================================================================
# 4. DRIVING COORDINATE BACK-CALCULATION TESTS
# =============================================================================

def test_back_calculate_driving_coordinates() -> None:
    """Verifies identification of internal PES coordinates driving large (O-C) residuals."""
    # 2 Flagged lines with massive residuals
    flagged_res = jnp.array([12.0, 18.0], dtype=jnp.float64)

    # 4 Hamiltonian parameters: ['A', 'B', 'C', 'V3_torsion']
    # Parameter 3 (V3_torsion) has enormous sensitivity gradient for these states
    flagged_jacs = jnp.array([
        [0.1, 0.05, 0.02, 10.0],
        [0.2, 0.10, 0.04, 15.0],
    ], dtype=jnp.float64)

    coord_names = ["A", "B", "C", "V3_torsion"]

    coord_report = back_calculate_driving_coordinates(
        flagged_residuals=flagged_res,
        state_jacobians=flagged_jacs,
        coordinate_names=coord_names,
    )

    assert coord_report["top_driver_name"] == "V3_torsion"
    assert coord_report["top_driver_index"] == 3
    # V3 sensitivity dominates > 95% of total
    assert float(coord_report["relative_proportions"][3]) > 0.95
    assert coord_report["driver_ranking"][0][0] == "V3_torsion"


# =============================================================================
# 5. PAYLOAD GENERATION & METHOD MATRIX COMPLIANCE
# =============================================================================

def test_generate_pes_healing_payload_canonical_ccsdt() -> None:
    """Verifies generation of structured, cryptographically hashed refinement payload."""
    flagged_report = {
        "flagged_count": 2,
        "max_residual": 14.8,
        "rmse_flagged": 12.0,
        "flagged_states": {"J": [8, 9]},
    }
    coord_report = {
        "top_driver_name": "tau_torsion",
        "top_driver_index": 2,
        "coordinate_sensitivities": [1.0, 2.0, 50.0],
        "relative_proportions": [0.019, 0.038, 0.943],
        "driver_ranking": [("tau_torsion", 0.943), ("B", 0.038), ("A", 0.019)],
    }

    payload = generate_pes_healing_payload(
        flagged_state_report=flagged_report,
        coordinate_report=coord_report,
        molecular_formula="CH3OH",
        atomic_symbols=["C", "H", "H", "H", "O", "H"],
        target_method="canonical-CCSD(T)",
        basis_set="def2-QZVPP",
    )

    assert payload["payload_type"] == "ACTIVE_PES_REFINEMENT_REQUEST"
    assert payload["molecular_formula"] == "CH3OH"
    assert payload["target_method"] == "canonical-CCSD(T)"
    assert payload["top_driving_coordinate"] == "tau_torsion"
    assert payload["flagged_unresolved_count"] == 2
    assert len(payload["sha256_checksum"]) == 64


def test_generate_pes_healing_payload_dlpno_forbidden() -> None:
    """Verifies that requesting DLPNO-CCSD(T) strictly raises DLPNOForbiddenError per Method Matrix."""
    flagged_report = {"flagged_count": 1, "max_residual": 10.0, "rmse_flagged": 10.0}
    coord_report = {"top_driver_name": "R_CO", "top_driver_index": 0}

    with pytest.raises(DLPNOForbiddenError, match="strictly forbidden"):
        generate_pes_healing_payload(
            flagged_state_report=flagged_report,
            coordinate_report=coord_report,
            molecular_formula="CO",
            target_method="DLPNO-CCSD(T)",
        )

    with pytest.raises(DLPNOForbiddenError, match="strictly forbidden"):
        ActivePESHealingConfig(target_method="DLPNO-CCSD(T)")


# =============================================================================
# 6. PURE-JAX FUNCTIONAL PURITY & ERROR HANDLING
# =============================================================================

def test_invalid_covariance_and_jacobian_dimensions() -> None:
    """Verifies typed exceptions for malformed covariance matrices and Jacobians."""
    # Non-square covariance matrix
    with pytest.raises(InvalidCovarianceError, match="must be square"):
        calculate_information_gain(
            covariance_matrix=np.ones((3, 2)),
            jacobian=np.ones((2, 3)),
        )

    # Incompatible parameter dimension between cov and jac
    with pytest.raises(InvalidJacobianError, match="does not match covariance dimension"):
        calculate_information_gain(
            covariance_matrix=np.eye(4),
            jacobian=np.ones((2, 3)),
        )

    # Empty candidate transitions
    with pytest.raises(EmptyCandidateError, match="zero candidate"):
        calculate_information_gain(
            covariance_matrix=np.eye(3),
            jacobian=np.empty((0, 3)),
        )


# =============================================================================
# 7. STRICT AST INSPECTION: ZERO PASS STATEMENTS
# =============================================================================

def test_cochem_assign_ml_zero_pass_statements() -> None:
    """Strict AST audit verifying zero pass statements in cochem_assign_ml.py and this test file."""
    import cochem_spycfit.physics_core.cochem_assign_ml as assign_mod

    # Audit source module
    source_mod = inspect.getsource(assign_mod)
    tree_mod = ast.parse(source_mod)
    passes_mod = [node.lineno for node in ast.walk(tree_mod) if isinstance(node, ast.Pass)]
    assert passes_mod == [], f"Forbidden pass statements in cochem_assign_ml.py at lines: {passes_mod}"

    # Audit test file itself
    test_source = Path(__file__).read_text(encoding="utf-8")
    tree_test = ast.parse(test_source)
    passes_test = [node.lineno for node in ast.walk(tree_test) if isinstance(node, ast.Pass)]
    assert passes_test == [], f"Forbidden pass statements in test_cochem_assign_ml.py at lines: {passes_test}"
