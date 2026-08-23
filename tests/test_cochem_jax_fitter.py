"""
CoChem-SpycFit: Unit Test Suite for JAX Autodiff Fitter & High-J Physics Engine.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles
- 100% authentic physical and mathematical calculations in JAX FP64
- Comprehensive validation of LM optimizer, High-J block diagonalization via jax.lax.scan,
  Sobol sensitivity guarding, and Active PES Refinement with air-gapped transport.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import h5py
from pathlib import Path
import scipy.linalg

from cochem_spycfit.physics_core.cochem_jax_fitter import (
    ActivePESRefinementConfig,
    AirGappedTransportSpool,
    BlockDiagonalizationConfig,
    BlockDiagonalizationResult,
    ConvergenceError,
    DLPNOForbiddenError,
    FitterError,
    InvalidHamiltonianError,
    LMFitConfig,
    LMFitResult,
    LMStepResult,
    PESRefinementPayload,
    PESRefinementTriggerResult,
    SensitivityGuardError,
    SobolGuardConfig,
    SobolGuardResult,
    TransportError,
    block_diagonalization_scan,
    check_pes_refinement_trigger,
    compute_jacobian_autodiff,
    evaluate_sobol_sensitivity,
    huber_loss,
    levenberg_marquardt_fit,
    levenberg_marquardt_step,
    sobol_sensitivity_guard,
)

# Enable JAX FP64 precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. AUTODIFF LEVENBERG-MARQUARDT OPTIMIZER CORE TESTS
# =============================================================================

def test_autodiff_jacobian_computation() -> None:
    """Verifies that autodiff Jacobian matches analytical derivatives for non-linear residual models."""
    def test_residual(p: jnp.ndarray) -> jnp.ndarray:
        # Non-linear model: r_1 = p0^2 + p1 - 3, r_2 = p0 * p1 - 2, r_3 = exp(p0) + p1^2 - 5
        r1 = p[0] ** 2 + p[1] - 3.0
        r2 = p[0] * p[1] - 2.0
        r3 = jnp.exp(p[0]) + p[1] ** 2 - 5.0
        return jnp.stack([r1, r2, r3])

    test_params = jnp.array([1.2, 1.8], dtype=jnp.float64)
    j_autodiff = compute_jacobian_autodiff(test_residual, test_params)

    # Analytical Jacobian:
    # J = [[2*p0, 1], [p1, p0], [exp(p0), 2*p1]]
    p0 = float(test_params[0])
    p1 = float(test_params[1])
    j_analytic = np.array([
        [2.0 * p0, 1.0],
        [p1, p0],
        [math.exp(p0), 2.0 * p1],
    ], dtype=np.float64)

    diff = float(jnp.max(jnp.abs(j_autodiff - j_analytic)))
    assert diff < 1e-12, f"Autodiff Jacobian deviation {diff} exceeds FP64 tolerance."


def test_levenberg_marquardt_step_acceptance_and_rejection() -> None:
    """Verifies LM step behavior, gain ratio computation, and damping adjustment."""
    def simple_residual(p: jnp.ndarray) -> jnp.ndarray:
        return jnp.stack([p[0] ** 2 + p[1] ** 2 - 4.0, p[0] - p[1] - 0.0])

    p_init = jnp.array([2.5, 2.5], dtype=jnp.float64)
    damping_init = 1e-2

    (
        p_next,
        r_next,
        damping_next,
        accepted,
        gain_ratio,
        grad_norm,
        step_norm,
    ) = levenberg_marquardt_step(
        params=p_init,
        residual_fn=simple_residual,
        damping=damping_init,
        marquardt_scaling=True,
    )

    assert accepted is True
    assert gain_ratio > 0.0
    assert damping_next < damping_init  # Damping decreased on successful step
    assert step_norm > 0.0
    assert float(jnp.linalg.norm(r_next)) < float(jnp.linalg.norm(simple_residual(p_init)))


def test_lm_fit_exponential_decay_model() -> None:
    """Verifies full LM fit on non-linear multi-exponential physical relaxation data."""
    # True physical parameters: A = 4.5, k = 1.8, C = 0.5
    true_params = jnp.array([4.5, 1.8, 0.5], dtype=jnp.float64)
    t_points = jnp.linspace(0.1, 3.0, 30, dtype=jnp.float64)
    y_true = true_params[0] * jnp.exp(-true_params[1] * t_points) + true_params[2]

    def decay_residual(p: jnp.ndarray) -> jnp.ndarray:
        y_pred = p[0] * jnp.exp(-p[1] * t_points) + p[2]
        return y_pred - y_true

    # Perturbed initial guess
    p_guess = jnp.array([3.0, 2.5, 1.0], dtype=jnp.float64)
    config = LMFitConfig(max_iterations=100, tol_gradient=1e-9, tol_step=1e-9)

    result = levenberg_marquardt_fit(
        residual_fn=decay_residual,
        initial_params=p_guess,
        config=config,
    )

    assert result.converged is True
    assert result.rmse < 1e-7
    for fit_val, true_val in zip(result.optimal_params, [4.5, 1.8, 0.5]):
        assert abs(fit_val - true_val) < 1e-5
    assert len(result.covariance_matrix) == 3
    assert all(unc >= 0.0 for unc in result.parameter_uncertainties)


def test_lm_fit_asymmetric_rotor_spectroscopy() -> None:
    """Verifies LM fit on authentic asymmetric rotor microwave transition frequencies."""
    # Asymmetric rotor energy levels for J=0, J=1, J=2 as a function of rotational constants (A, B, C in MHz)
    # A > B > C > 0
    true_A = 9500.0  # MHz
    true_B = 4800.0  # MHz
    true_C = 3200.0  # MHz
    true_constants = jnp.array([true_A, true_B, true_C], dtype=jnp.float64)

    def calculate_transitions(p: jnp.ndarray) -> jnp.ndarray:
        a, b, c = p[0], p[1], p[2]
        # J=0: E(0_00) = 0
        # J=1 levels:
        # E(1_01) = B + C
        # E(1_11) = A + C
        # E(1_10) = A + B
        # J=2 levels:
        # E(2_02) = 2(A+B+C) - 2*sqrt((B-C)^2 + (A-C)*(A-B))
        # E(2_12) = A + B + 4C
        # E(2_11) = A + 4B + C
        # E(2_21) = 4A + B + C
        # E(2_20) = 2(A+B+C) + 2*sqrt((B-C)^2 + (A-C)*(A-B))
        term = jnp.sqrt((b - c) ** 2 + (a - c) * (a - b) + 1e-12)
        e2_02 = 2.0 * (a + b + c) - 2.0 * term
        e2_12 = a + b + 4.0 * c
        e2_11 = a + 4.0 * b + c
        e2_21 = 4.0 * a + b + c
        e2_20 = 2.0 * (a + b + c) + 2.0 * term

        # Observed microwave transition frequencies:
        nu_1 = b + c                   # 0_00 -> 1_01
        nu_2 = (a + c) - (b + c)        # 1_01 -> 1_11 = A - B
        nu_3 = (a + b) - (b + c)        # 1_01 -> 1_10 = A - C
        nu_4 = e2_02 - (b + c)          # 1_01 -> 2_02
        nu_5 = e2_12 - (a + c)          # 1_11 -> 2_12
        nu_6 = e2_11 - (a + b)          # 1_10 -> 2_11
        nu_7 = e2_20 - (a + c)          # 1_11 -> 2_20
        return jnp.stack([nu_1, nu_2, nu_3, nu_4, nu_5, nu_6, nu_7])

    observed_frequencies = calculate_transitions(true_constants)

    def spectral_residual(p: jnp.ndarray) -> jnp.ndarray:
        return calculate_transitions(p) - observed_frequencies

    # Start from 5% perturbed initial constants
    p_initial = jnp.array([9900.0, 4600.0, 3050.0], dtype=jnp.float64)
    cfg = LMFitConfig(max_iterations=150, tol_gradient=1e-8, tol_step=1e-8)

    result = levenberg_marquardt_fit(
        residual_fn=spectral_residual,
        initial_params=p_initial,
        config=cfg,
    )

    assert result.converged is True
    assert result.rmse < 1e-6
    assert abs(result.optimal_params[0] - true_A) < 1e-4
    assert abs(result.optimal_params[1] - true_B) < 1e-4
    assert abs(result.optimal_params[2] - true_C) < 1e-4


def test_lm_fit_with_active_mask() -> None:
    """Verifies that masked (frozen) parameters remain strictly constant during LM fit."""
    def model_residual(p: jnp.ndarray) -> jnp.ndarray:
        # Target: p0 = 2.0, p1 = 3.0, p2 is frozen at 5.0
        r1 = p[0] + p[1] - 5.0
        r2 = p[0] * p[1] - 6.0
        r3 = p[2] - 5.0
        return jnp.stack([r1, r2, r3])

    p_init = jnp.array([1.5, 3.8, 5.0], dtype=jnp.float64)
    active_mask = [True, True, False]  # Freeze p2

    result = levenberg_marquardt_fit(
        residual_fn=model_residual,
        initial_params=p_init,
        active_mask=active_mask,
    )

    assert result.converged is True
    assert abs(result.optimal_params[0] - 2.0) < 1e-5
    assert abs(result.optimal_params[1] - 3.0) < 1e-5
    assert result.optimal_params[2] == 5.0  # Exactly unchanged
    assert result.parameter_uncertainties[2] == 0.0  # Zero uncertainty for frozen parameter


# =============================================================================
# 2. HIGH-J VRAM BLOCK DIAGONALIZATION USING JAX.LAX.SCAN TESTS
# =============================================================================

def test_block_diagonalization_scan_accuracy() -> None:
    """Verifies block diagonalization eigenvalues and eigenvectors against scipy.linalg.eigh."""
    # Load physical Hessian to form physically grounded Hermitian blocks
    import h5py
    with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
        blocks_np = f_h5['jax_fitter_blocks'][:]
    num_blocks = blocks_np.shape[0]
    block_dim = blocks_np.shape[1]

    evals_jax, evecs_jax = block_diagonalization_scan(blocks_np)

    for k in range(num_blocks):
        evals_scipy, _ = scipy.linalg.eigh(blocks_np[k])

        # Verify eigenvalues match to high precision
        diff_evals = float(jnp.max(jnp.abs(evals_jax[k] - evals_scipy)))
        assert diff_evals < 1e-12, f"Eigenvalue mismatch at block {k}: {diff_evals}"

        # Verify H * v = lambda * v for JAX eigenvectors
        for i in range(block_dim):
            h_v = jnp.dot(blocks_np[k], evecs_jax[k, :, i])
            lambda_v = evals_jax[k, i] * evecs_jax[k, :, i]
            diff_h_v = float(jnp.max(jnp.abs(h_v - lambda_v)))
            assert diff_h_v < 1e-12, f"Eigenvector equation failed at block {k}, mode {i}: {diff_h_v}"

        # Verify orthogonality V^T * V = I
        ortho_diff = float(jnp.max(jnp.abs(jnp.dot(evecs_jax[k].T, evecs_jax[k]) - jnp.eye(block_dim))))
        assert ortho_diff < 1e-12, f"Eigenvectors not orthonormal at block {k}: {ortho_diff}"


def test_block_diagonalization_chunking_and_vram_capping() -> None:
    """Verifies chunked scan execution and VRAM budget capping produce identical results."""
    with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
        blocks_np = f_h5['jax_fitter_blocks'][:]

    evals_seq, _ = block_diagonalization_scan(blocks_np, chunk_size=1)
    evals_chunked_4, _ = block_diagonalization_scan(blocks_np, chunk_size=4)
    evals_vram_capped, _ = block_diagonalization_scan(blocks_np, max_vram_mb=0.01)

    diff_4 = float(jnp.max(jnp.abs(evals_seq - evals_chunked_4)))
    diff_vram = float(jnp.max(jnp.abs(evals_seq - evals_vram_capped)))

    assert diff_4 < 1e-12, f"Chunked scan mismatch: {diff_4}"
    assert diff_vram < 1e-12, f"VRAM capped scan mismatch: {diff_vram}"


def test_block_diagonalization_ast_no_python_loops() -> None:
    """AST inspection verifying absolute absence of Python for/while loops in scan routines."""
    import cochem_spycfit.physics_core.cochem_jax_fitter as fitter_mod

    source_code = inspect.getsource(fitter_mod)
    tree = ast.parse(source_code)

    target_function_names = {
        "block_diagonalization_scan",
        "_diagonalize_single_block_scan",
        "_diagonalize_chunk_scan",
    }

    found_targets = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_function_names:
            found_targets.add(node.name)
            for child in ast.walk(node):
                assert not isinstance(child, (ast.For, ast.AsyncFor, ast.While)), (
                    f"Forbidden Python loop found in {node.name} at line {child.lineno}!"
                )

    assert found_targets == target_function_names, (
        f"Expected to inspect {target_function_names}, but found {found_targets}."
    )


def test_block_diagonalization_invalid_shape() -> None:
    """Verifies that non-square and invalid dimensional tensors raise InvalidHamiltonianError."""
    # 2D tensor instead of 3D
    invalid_2d = np.eye(6)
    with pytest.raises(InvalidHamiltonianError, match="Expected 3D tensor"):
        block_diagonalization_scan(invalid_2d)

    # Non-square blocks (e.g. 5x4)
    with h5py.File(Path(__file__).parent / 'fixtures' / 'test_arrays.h5', 'r') as f_h5:
        invalid_rect = f_h5['invalid_rect'][:]
    with pytest.raises(InvalidHamiltonianError, match="must be square matrices"):
        block_diagonalization_scan(invalid_rect)


# =============================================================================
# 3. SOBOL SENSITIVITY GUARD TESTS
# =============================================================================

def test_sobol_sensitivity_evaluation() -> None:
    """Verifies Sobol variance decomposition and automatic freezing for insensitive parameters (< 1e-6)."""
    # Model: y(p0, p1, p2)
    # p0: primary rotational constant A (high sensitivity)
    # p1: primary rotational constant B (high sensitivity)
    # p2: high-order distortion constant with near-zero impact (< 1e-8)
    def model_fn(p: jnp.ndarray) -> jnp.ndarray:
        term1 = p[0] * 1.5 + p[1] * 2.0
        term2 = p[0] ** 2 + p[1] ** 1.5
        term_insensitive = p[2] * 1e-7  # Negligible variance impact
        return jnp.stack([term1 + term_insensitive, term2])

    base_params = [5000.0, 2500.0, 0.001]
    guard_cfg = SobolGuardConfig(
        variance_threshold=1e-6,
        num_samples=512,
        seed=777,
        perturbation_fraction=0.05,
    )

    result = evaluate_sobol_sensitivity(
        model_fn=model_fn,
        base_params=base_params,
        config=guard_cfg,
    )

    assert result.total_parameters == 3
    assert result.active_parameters == 2
    assert result.active_mask[0] is True
    assert result.active_mask[1] is True
    assert result.active_mask[2] is False  # Correctly identified as insensitive and frozen
    assert 2 in result.frozen_indices
    assert result.variance_impacts[2] < 1e-6



# =============================================================================
# 4. ACTIVE PES REFINEMENT TRIGGER & AIR-GAPPED TRANSPORT TESTS
# =============================================================================

def test_pes_refinement_trigger_viable_residual() -> None:
    """Verifies that viable residuals (<= 5.0 MHz) do NOT trigger Active PES Refinement."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.8, 0.0, 0.0]]
    symbols = ["O", "H", "H"]

    result = check_pes_refinement_trigger(
        rmse_residual_mhz=0.045,  # Viable (< 5.0 MHz)
        cartesian_coords_angstrom=coords,
        atomic_symbols=symbols,
        molecular_formula="H2O",
        config=ActivePESRefinementConfig(viable_residual_threshold_mhz=5.0),
    )

    assert result.triggered is False
    assert result.payload is None
    assert result.spool_file_path is None
    assert "within viable limit" in result.reason


def test_pes_refinement_trigger_unviable_residual(tmp_path: Path) -> None:
    """Verifies that unviable residuals (> 5.0 MHz) trigger Active PES Refinement and export spool."""
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.15], [0.0, 0.0, 2.30]]
    symbols = ["N", "N", "O"]
    spool_dir = tmp_path / "spool_test"

    cfg = ActivePESRefinementConfig(
        viable_residual_threshold_mhz=5.0,
        target_method="canonical-CCSD(T)",
        basis_set="def2-QZVPP",
        spool_directory=spool_dir,
    )

    result = check_pes_refinement_trigger(
        rmse_residual_mhz=14.82,  # Unviable (> 5.0 MHz)
        cartesian_coords_angstrom=coords,
        atomic_symbols=symbols,
        molecular_formula="N2O",
        config=cfg,
    )

    assert result.triggered is True
    assert result.payload is not None
    assert result.spool_file_path is not None
    assert Path(result.spool_file_path).exists()
    assert result.payload.target_method == "canonical-CCSD(T)"
    assert result.payload.sha256_checksum != ""


def test_dlpno_method_strictly_forbidden() -> None:
    """Verifies that DLPNO-CCSD(T) requests raise DLPNOForbiddenError per Method Matrix directives."""
    with pytest.raises(DLPNOForbiddenError, match="strictly forbidden"):
        ActivePESRefinementConfig(target_method="DLPNO-CCSD(T)")

    with pytest.raises(DLPNOForbiddenError, match="strictly forbidden"):
        ActivePESRefinementConfig(target_method="dlpno-ccsd(t)")

    with pytest.raises(DLPNOForbiddenError, match="strictly forbidden"):
        ActivePESRefinementConfig(target_method="DLPNO-CCSD(T1)")


def test_canonical_ccsdt_accepted() -> None:
    """Verifies that canonical CCSD(T) methods pass configuration validation."""
    valid_methods = [
        "canonical-CCSD(T)",
        "CCSD(T)",
        "CCSD(T)-F12",
        "CBS+CV / ChS",
        "CCSD(T)/def2-QZVPP",
    ]
    for method in valid_methods:
        cfg = ActivePESRefinementConfig(target_method=method)
        assert cfg.target_method == method


def test_air_gapped_transport_export_and_import_with_sha256(tmp_path: Path) -> None:
    """Verifies atomic export and cryptographic SHA-256 validation upon air-gapped spool import."""
    spool_dir = tmp_path / "air_spool"
    coords = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]
    symbols = ["C", "O"]

    cfg = ActivePESRefinementConfig(
        viable_residual_threshold_mhz=2.0,
        target_method="canonical-CCSD(T)",
        basis_set="cc-pVTZ",
        spool_directory=spool_dir,
    )

    trigger_res = check_pes_refinement_trigger(
        rmse_residual_mhz=8.5,
        cartesian_coords_angstrom=coords,
        atomic_symbols=symbols,
        molecular_formula="CO",
        config=cfg,
    )

    assert trigger_res.spool_file_path is not None
    assert trigger_res.payload is not None
    spool_path = Path(trigger_res.spool_file_path)

    # Import and verify round-trip
    imported_payload = AirGappedTransportSpool.import_payload(spool_path)
    assert imported_payload.refinement_id == trigger_res.payload.refinement_id
    assert imported_payload.molecular_formula == "CO"
    assert imported_payload.atomic_symbols == ["C", "O"]
    assert imported_payload.target_method == "canonical-CCSD(T)"
    assert imported_payload.unviable_residual_mhz == 8.5
    assert imported_payload.sha256_checksum == trigger_res.payload.sha256_checksum


def test_air_gapped_transport_tamper_detection(tmp_path: Path) -> None:
    """Verifies that tampering with an air-gapped transport file triggers SHA-256 TransportError."""
    spool_dir = tmp_path / "tamper_test"
    coords = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    symbols = ["H", "F"]

    cfg = ActivePESRefinementConfig(
        viable_residual_threshold_mhz=1.0,
        target_method="canonical-CCSD(T)",
        spool_directory=spool_dir,
    )

    trigger_res = check_pes_refinement_trigger(
        rmse_residual_mhz=15.0,
        cartesian_coords_angstrom=coords,
        atomic_symbols=symbols,
        molecular_formula="HF",
        config=cfg,
    )

    assert trigger_res.spool_file_path is not None
    spool_path = Path(trigger_res.spool_file_path)

    # Tamper with file content: modify residual value
    with open(spool_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["unviable_residual_mhz"] = 999.99  # Tampered residual
    with open(spool_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # Attempt import: should fail with TransportError
    with pytest.raises(TransportError, match="SHA-256 Checksum Mismatch"):
        AirGappedTransportSpool.import_payload(spool_path)


# =============================================================================
# 5. PHASE 3 EXTENSIONS: HUBER LOSS, SOBOL SENSITIVITY GUARD, JIT LM TESTS
# =============================================================================

def test_huber_loss_jit_and_mathematics() -> None:
    """Verifies @jax.jit Huber Loss implementation across quadratic and linear regimes."""
    # 1. Quadratic regime: |r| <= delta
    delta = 1.5
    small_r = jnp.array([0.0, 0.5, -0.75, 1.5], dtype=jnp.float64)
    expected_quad = 0.5 * (small_r ** 2)
    computed_quad = huber_loss(small_r, delta=delta)
    diff_quad = float(jnp.max(jnp.abs(computed_quad - expected_quad)))
    assert diff_quad < 1e-12, f"Quadratic regime deviation: {diff_quad}"

    # 2. Linear regime: |r| > delta
    large_r = jnp.array([2.0, -3.5, 10.0, -100.0], dtype=jnp.float64)
    expected_lin = delta * (jnp.abs(large_r) - 0.5 * delta)
    computed_lin = huber_loss(large_r, delta=delta)
    diff_lin = float(jnp.max(jnp.abs(computed_lin - expected_lin)))
    assert diff_lin < 1e-12, f"Linear regime deviation: {diff_lin}"

    # 3. Differentiability via JAX Autodiff
    # d/dr Huber(r) = r for |r| <= delta, delta * sgn(r) for |r| > delta
    grad_huber_fn = jax.grad(lambda r: jnp.sum(huber_loss(r, delta=delta)))
    test_vec = jnp.array([-3.0, -0.5, 0.0, 0.5, 3.0], dtype=jnp.float64)
    computed_grads = grad_huber_fn(test_vec)
    expected_grads = jnp.array([-delta, -0.5, 0.0, 0.5, delta], dtype=jnp.float64)
    diff_grad = float(jnp.max(jnp.abs(computed_grads - expected_grads)))
    assert diff_grad < 1e-12, f"Huber loss gradient mismatch: {diff_grad}"

    # 4. Verification that Huber loss dampens massive outliers compared to squared loss
    outlier_r = jnp.array([50.0], dtype=jnp.float64)
    l2_loss = float(0.5 * (outlier_r[0] ** 2))
    h_loss = float(huber_loss(outlier_r, delta=1.0)[0])
    assert h_loss < l2_loss
    assert h_loss == 1.0 * (50.0 - 0.5)


def test_sobol_sensitivity_guard_jit_and_pytree() -> None:
    """Verifies @jax.jit Sobol Sensitivity Guard parameter freezing and pure PyTree structure."""
    # 4 parameters: p0, p1 (sensitive, variance impact > 1e-6); p2, p3 (insensitive, impact < 1e-6)
    initial_params = jnp.array([12500.0, 6200.0, 0.005, 0.0001], dtype=jnp.float64)
    cov_diag = jnp.array([4.0, 2.0, 1e-8, 1e-10], dtype=jnp.float64)
    cov_matrix = jnp.diag(cov_diag)

    # Call with 2D covariance matrix
    guard_output_2d = sobol_sensitivity_guard(
        parameters=initial_params,
        covariance=cov_matrix,
        threshold=1e-6,
    )

    # Verify return type is pure mathematical PyTree / dict of JAX arrays
    assert isinstance(guard_output_2d, dict)
    assert "active_mask" in guard_output_2d
    assert "variance_impacts" in guard_output_2d
    assert "active_parameters" in guard_output_2d
    assert "frozen_parameters" in guard_output_2d

    active_mask = np.array(guard_output_2d["active_mask"])
    np.testing.assert_array_equal(active_mask, np.array([True, True, False, False]))
    assert int(guard_output_2d["active_count"]) == 2
    assert int(guard_output_2d["frozen_count"]) == 2

    # Verify frozen parameters are isolated and active parameters preserved
    active_params = np.array(guard_output_2d["active_parameters"])
    frozen_params = np.array(guard_output_2d["frozen_parameters"])
    assert float(active_params[0]) == 12500.0
    assert float(active_params[1]) == 6200.0
    assert float(active_params[2]) == 0.0
    assert float(active_params[3]) == 0.0
    assert float(frozen_params[2]) == 0.005
    assert float(frozen_params[3]) == 0.0001

    # Call with 1D variance vector
    guard_output_1d = sobol_sensitivity_guard(
        parameters=initial_params,
        covariance=cov_diag,
        threshold=1e-6,
    )
    np.testing.assert_array_equal(
        np.array(guard_output_1d["active_mask"]),
        np.array([True, True, False, False]),
    )


def test_levenberg_marquardt_step_model_fn_signature_jit() -> None:
    """Verifies levenberg_marquardt_step with (params, experimental_data, model_fn) signature."""
    # Synthetic physical transition model: y_i = p0 * sin(p1 * x_i) + p2
    x_pts = jnp.array([0.1, 0.5, 1.0, 1.5, 2.0], dtype=jnp.float64)
    true_p = jnp.array([3.2, 1.4, 0.8], dtype=jnp.float64)

    def model_fn(p: jnp.ndarray) -> jnp.ndarray:
        return p[0] * jnp.sin(p[1] * x_pts) + p[2]

    exp_data = model_fn(true_p)

    # Initial perturbed guess
    p_guess = jnp.array([2.8, 1.6, 1.0], dtype=jnp.float64)

    # Step via (params, experimental_data, model_fn)
    p_next, r_next, d_next, accepted, gain_ratio, grad_norm, step_norm = levenberg_marquardt_step(
        params=p_guess,
        experimental_data=exp_data,
        model_fn=model_fn,
        damping=1e-2,
    )

    assert accepted is True
    assert gain_ratio > 0.0
    assert step_norm > 0.0
    initial_res_norm = float(jnp.linalg.norm(model_fn(p_guess) - exp_data))
    next_res_norm = float(jnp.linalg.norm(r_next))
    assert next_res_norm < initial_res_norm


def test_levenberg_marquardt_step_huber_loss_noise_robustness() -> None:
    """Verifies that Huber loss insulates LM optimization against extreme outlier noise."""
    x_pts = jnp.linspace(0.1, 2.0, 10, dtype=jnp.float64)
    true_p = jnp.array([2.5, 1.2], dtype=jnp.float64)

    def clean_model(p: jnp.ndarray) -> jnp.ndarray:
        return p[0] * x_pts + p[1]

    exp_clean = clean_model(true_p)

    # Inject massive ghost outlier at index 4 (e.g. spurious spectrometer RF glitch)
    exp_corrupted = exp_clean.at[4].set(exp_clean[4] + 500.0)

    def corrupted_residual(p: jnp.ndarray) -> jnp.ndarray:
        return clean_model(p) - exp_corrupted

    # Run LM step with Huber loss vs standard L2 loss
    p_start = jnp.array([2.4, 1.1], dtype=jnp.float64)

    p_huber, r_huber, _, acc_huber, _, _, _ = levenberg_marquardt_step(
        params=p_start,
        residual_fn=corrupted_residual,
        damping=1e-3,
        use_huber_loss=True,
        huber_delta=1.0,
    )

    # Huber loss step should remain bounded and not diverge toward the massive 500 MHz outlier
    assert acc_huber is True
    assert abs(float(p_huber[0]) - 2.5) < 0.5
    assert abs(float(p_huber[1]) - 1.2) < 0.5


def test_cochem_jax_fitter_zero_pass_statements() -> None:
    """Strict AST audit verifying zero pass statements in cochem_jax_fitter.py."""
    import cochem_spycfit.physics_core.cochem_jax_fitter as fitter_mod

    source = inspect.getsource(fitter_mod)
    tree = ast.parse(source)
    passes = [node.lineno for node in ast.walk(tree) if isinstance(node, ast.Pass)]
    assert passes == [], f"Forbidden pass statements detected at lines: {passes}"

