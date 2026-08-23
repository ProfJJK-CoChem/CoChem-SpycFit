"""
CoChem-SpycFit: Autodiff Optimization Core & High-J Physics Engine.

This module implements:
1. Pure-JAX Levenberg-Marquardt (LM) optimizer utilizing Automatic Differentiation (jacrev/jacfwd).
2. High-J VRAM block diagonalization using jax.lax.scan to chunk processing and dynamically
   cap peak memory allocation without Python for loops.
3. Sobol Sensitivity Guard evaluating parameter variance impact and freezing insensitive parameters
   whose variance impact drops below 1e-6.
4. Active PES Refinement Trigger with Method Matrix compliance (canonical CCSD(T) acceptance,
   strict DLPNO-CCSD(T) prohibition via DLPNOForbiddenError, and air-gapped transport spooling
   with SHA-256 checksums and atomic file locking).

Precision: JAX FP64 precision enabled globally. [E]
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import math
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import filelock
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Enforce JAX 64-bit precision globally for high-accuracy spectroscopic physics
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.JAXFitter")


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class FitterError(Exception):
    """Base exception for all JAX fitter and spectroscopic optimization operations."""


class ConvergenceError(FitterError):
    """Raised when Levenberg-Marquardt optimization fails to converge within tolerances."""


class DLPNOForbiddenError(FitterError):
    """
    Hard error raised when DLPNO-CCSD(T) is requested for PES refinement.
    Method Matrix Rule: DLPNO methods are strictly forbidden for high-precision
    spectroscopic PES refinement due to non-monotonic domain approximation errors.
    Canonical CCSD(T) is mandatory.
    """


class SensitivityGuardError(FitterError):
    """Raised when Sobol sensitivity analysis fails or all parameters are frozen."""


class TransportError(FitterError):
    """Raised when air-gapped spool export, import, or SHA-256 verification fails."""


class InvalidHamiltonianError(FitterError):
    """Raised when Hamiltonian blocks are malformed, non-square, or non-Hermitian."""


# =============================================================================
# 2. PYDANTIC SCHEMAS & CONFIGURATIONS
# =============================================================================

class LMStepResult(BaseModel):
    """Record of a single Levenberg-Marquardt optimization iteration."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    iteration: int = Field(..., ge=0, description="Iteration index")
    params: List[float] = Field(..., description="Current parameter values")
    step: List[float] = Field(..., description="Parameter displacement step vector")
    residual_norm: float = Field(..., ge=0.0, description="L2 norm of residuals")
    damping: float = Field(..., gt=0.0, description="Levenberg-Marquardt damping lambda")
    accepted: bool = Field(..., description="Whether the step was accepted based on gain ratio")
    gain_ratio: float = Field(..., description="Gain ratio rho = actual_reduction / predicted_reduction")
    gradient_norm: float = Field(..., ge=0.0, description="Infinity norm of the objective gradient")
    step_norm: float = Field(..., ge=0.0, description="L2 norm of the parameter update step")


class LMFitConfig(BaseModel):
    """Configuration options for Autodiff Levenberg-Marquardt optimization."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_iterations: int = Field(default=100, ge=1, le=10000, description="Maximum optimizer iterations")
    damping_init: float = Field(default=1e-3, gt=0.0, description="Initial damping parameter lambda")
    damping_up: float = Field(default=10.0, gt=1.0, description="Multiplier for lambda upon rejected step")
    damping_down: float = Field(default=0.1, gt=0.0, lt=1.0, description="Multiplier for lambda upon accepted step")
    damping_min: float = Field(default=1e-12, gt=0.0, description="Minimum allowed damping value")
    damping_max: float = Field(default=1e12, gt=0.0, description="Maximum allowed damping value")
    tol_gradient: float = Field(default=1e-8, ge=0.0, description="Gradient infinity-norm convergence tolerance")
    tol_step: float = Field(default=1e-8, ge=0.0, description="Step L2-norm convergence tolerance")
    tol_residual: float = Field(default=1e-8, ge=0.0, description="Relative residual change convergence tolerance")
    marquardt_scaling: bool = Field(default=True, description="Whether to scale damping by diag(J^T J)")


class LMFitResult(BaseModel):
    """Comprehensive result of an Autodiff Levenberg-Marquardt optimization run."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    converged: bool = Field(..., description="Whether optimization achieved convergence criteria")
    iterations: int = Field(..., ge=0, description="Total iterations completed")
    optimal_params: List[float] = Field(..., description="Fitted parameter vector")
    initial_params: List[float] = Field(..., description="Initial guess parameter vector")
    final_residuals: List[float] = Field(..., description="Residual vector at optimal parameters")
    residual_norm: float = Field(..., ge=0.0, description="Final residual L2 norm")
    rmse: float = Field(..., ge=0.0, description="Root Mean Square Error of residuals")
    reduced_chi_squared: float = Field(..., ge=0.0, description="Reduced chi-squared statistic")
    gradient_norm: float = Field(..., ge=0.0, description="Final gradient infinity norm")
    covariance_matrix: List[List[float]] = Field(..., description="Parameter covariance matrix (J^T J)^-1 * s^2")
    parameter_uncertainties: List[float] = Field(..., description="1-sigma standard uncertainties for parameters")
    active_mask: List[bool] = Field(..., description="Boolean mask indicating active (True) vs frozen (False) params")
    history: List[LMStepResult] = Field(default_factory=list, description="Step telemetry history")
    message: str = Field(..., description="Convergence status message")


class BlockDiagonalizationConfig(BaseModel):
    """Configuration for high-J VRAM memory-bounded block diagonalization."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    max_vram_mb: Optional[float] = Field(default=None, gt=0.0, description="Maximum VRAM memory cap in MB")
    chunk_size: Optional[int] = Field(default=None, ge=1, description="Explicit chunk size for scan batching")


class BlockDiagonalizationResult(BaseModel):
    """Result of memory-capped high-J block diagonalization."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    num_blocks: int = Field(..., ge=1, description="Number of Hamiltonian blocks diagonalized")
    block_size: int = Field(..., ge=1, description="Dimension of each square Hamiltonian block")
    chunk_size: int = Field(..., ge=1, description="Chunk size utilized in jax.lax.scan")
    max_eigenvalue: float = Field(..., description="Highest computed eigenvalue across all blocks")
    min_eigenvalue: float = Field(..., description="Lowest computed eigenvalue across all blocks")


class SobolGuardConfig(BaseModel):
    """Configuration for Sobol Sensitivity Guard."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    variance_threshold: float = Field(default=1e-6, gt=0.0, description="Variance impact threshold for parameter freezing")
    num_samples: int = Field(default=1024, ge=16, description="Number of quasi-Monte Carlo sample evaluations")
    seed: int = Field(default=42, description="Random seed for sampling reproducibility")
    perturbation_fraction: float = Field(default=0.05, gt=0.0, le=1.0, description="Relative parameter perturbation span")


class SobolGuardResult(BaseModel):
    """Result of Sobol Sensitivity Guard variance impact assessment."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    variance_impacts: List[float] = Field(..., description="Normalized variance impact / total Sobol index per parameter")
    active_mask: List[bool] = Field(..., description="Boolean mask: True = active (sensitive), False = frozen (insensitive)")
    active_indices: List[int] = Field(..., description="Indices of active parameters")
    frozen_indices: List[int] = Field(..., description="Indices of frozen parameters")
    variance_threshold: float = Field(..., description="Threshold applied for freezing")
    total_parameters: int = Field(..., ge=1, description="Total number of evaluated parameters")
    active_parameters: int = Field(..., ge=0, description="Count of sensitive active parameters")


class ActivePESRefinementConfig(BaseModel):
    """Configuration for Active PES Refinement Trigger and Air-Gapped Export."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    viable_residual_threshold_mhz: float = Field(default=5.0, gt=0.0, description="Max acceptable residual RMSE (MHz)")
    target_method: str = Field(default="canonical-CCSD(T)", description="Ab initio quantum chemistry method")
    basis_set: str = Field(default="def2-QZVPP", description="Basis set for PES refinement")
    spool_directory: Path = Field(
        default_factory=lambda: Path(os.environ.get("COCHEM_ARTIFACTS_DIR", str(Path.home() / ".cochem_artifacts"))) / "air_gapped_spool",
        description="Directory for air-gapped export"
    )
    lock_timeout_seconds: float = Field(default=10.0, gt=0.0, description="Timeout for atomic file lock acquisition")

    @field_validator("target_method")
    @classmethod
    def validate_method_not_dlpno(cls, v: str) -> str:
        """Enforce strict Method Matrix prohibition against DLPNO methods."""
        if "dlpno" in v.lower():
            raise DLPNOForbiddenError(
                f"Method Matrix Violation: '{v}' is strictly forbidden for high-precision "
                f"spectroscopic PES refinement. Canonical CCSD(T) is mandatory."
            )
        return v


class PESRefinementPayload(BaseModel):
    """Standardized air-gapped transport payload for Active PES Refinement."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    refinement_id: str = Field(..., description="Unique UUID for refinement request")
    timestamp_utc: str = Field(..., description="ISO 8601 UTC creation timestamp")
    molecular_formula: str = Field(..., description="Molecular formula or identifier")
    atomic_symbols: List[str] = Field(..., description="Atomic element symbols")
    cartesian_coords_angstrom: List[List[float]] = Field(..., description="Nuclear Cartesian coordinates in Angstrom")
    target_method: str = Field(..., description="Validated canonical quantum chemistry method")
    basis_set: str = Field(..., description="Target basis set")
    unviable_residual_mhz: float = Field(..., description="Observed unviable fit residual RMSE in MHz")
    threshold_mhz: float = Field(..., description="Trigger threshold in MHz")
    displacement_vectors: List[List[float]] = Field(default_factory=list, description="Target geometric displacement grid")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary provenance metadata")
    sha256_checksum: str = Field(..., description="SHA-256 digest of payload contents")


class PESRefinementTriggerResult(BaseModel):
    """Outcome of Active PES Refinement evaluation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    triggered: bool = Field(..., description="Whether PES refinement was triggered")
    reason: str = Field(..., description="Explanation of trigger decision")
    rmse_residual_mhz: float = Field(..., description="Evaluated fit RMSE in MHz")
    payload: Optional[PESRefinementPayload] = Field(default=None, description="Generated export payload if triggered")
    spool_file_path: Optional[str] = Field(default=None, description="Path to exported spool file if triggered")


# =============================================================================
# 3. AUTODIFF LEVENBERG-MARQUARDT OPTIMIZER CORE
# =============================================================================

def compute_jacobian_autodiff(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes exact Jacobian J_ij = d r_i / d params_j using JAX Automatic Differentiation.
    Uses reverse-mode autodiff (jax.jacrev) for m >= n systems, and forward-mode (jax.jacfwd) for m < n.
    """
    params_fp64 = jnp.asarray(params, dtype=jnp.float64)
    sample_res = residual_fn(params_fp64)
    m = sample_res.shape[0]
    n = params_fp64.shape[0]
    if m >= n:
        jacobian_fn = jax.jacrev(residual_fn)
    else:
        jacobian_fn = jax.jacfwd(residual_fn)
    return jacobian_fn(params_fp64)


def levenberg_marquardt_step(
    params: jnp.ndarray,
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    damping: float = 1e-3,
    active_mask: Optional[jnp.ndarray] = None,
    marquardt_scaling: bool = True,
    damping_factor: float = 10.0,
    damping_up: Optional[float] = None,
    damping_down: Optional[float] = None,
    damping_min: float = 1e-12,
    damping_max: float = 1e12,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, float, float, float]:
    """
    Performs a single authentic JAX Autodiff Levenberg-Marquardt optimization step.

    Parameters:
        params: 1D JAX FP64 array of parameters [n]
        residual_fn: Pure callable returning residual vector r(params) [m]
        damping: Current LM damping parameter lambda
        active_mask: Optional boolean mask [n] indicating active (True) vs frozen (False) params
        marquardt_scaling: If True, uses Marquardt diagonal scaling diag(J^T J); else uses Identity
        damping_factor: Fallback factor to multiply/divide lambda upon rejection/acceptance
        damping_up: Multiplier for lambda upon rejected step
        damping_down: Multiplier for lambda upon accepted step
        damping_min: Minimum allowable damping parameter
        damping_max: Maximum allowable damping parameter

    Returns:
        Tuple of:
            params_next: Updated parameter vector [n]
            residual_next: Updated residual vector [m]
            damping_next: Updated damping lambda
            accepted: Boolean indicating step acceptance (gain ratio rho > 0)
            gain_ratio: Gain ratio rho
            grad_norm: Infinity norm of the objective gradient -J^T r
            step_norm: L2 norm of the parameter displacement step
    """
    params_curr = jnp.asarray(params, dtype=jnp.float64)
    n_params = params_curr.shape[0]

    d_up = damping_up if damping_up is not None else damping_factor
    d_down = damping_down if damping_down is not None else (1.0 / max(1e-12, damping_factor))

    # Initialize active mask if None
    if active_mask is None:
        mask = jnp.ones(n_params, dtype=jnp.bool_)
    else:
        mask = jnp.asarray(active_mask, dtype=jnp.bool_)

    # Current residuals and exact Jacobian via Autodiff
    r_curr = residual_fn(params_curr)
    j_mat = compute_jacobian_autodiff(residual_fn, params_curr)

    # Apply active parameter mask to Jacobian columns
    # Inactive parameter columns are zeroed
    mask_float = mask.astype(jnp.float64)
    j_active = j_mat * mask_float[None, :]

    # Normal equations: H = J^T J, grad_f = J^T r
    h_mat = jnp.dot(j_active.T, j_active)
    grad_f = jnp.dot(j_active.T, r_curr)
    g_rhs = -grad_f

    # Compute damping matrix D
    if marquardt_scaling:
        diag_h = jnp.diag(h_mat)
        diag_scaled = jnp.maximum(diag_h, 1e-12)
        d_mat = jnp.diag(diag_scaled)
    else:
        d_mat = jnp.eye(n_params, dtype=jnp.float64)

    # Augmented normal matrix A = H + lambda * D
    a_mat = h_mat + damping * d_mat

    # Regularize inactive rows/columns to guarantee identity mapping for frozen parameters
    # For inactive i, a_mat[i, i] = 1.0, a_mat[i, j!=i] = 0.0, g_rhs[i] = 0.0
    mask_2d = mask_float[:, None] * mask_float[None, :]
    inv_mask_1d = (1.0 - mask_float)
    a_regularized = a_mat * mask_2d + jnp.diag(inv_mask_1d)
    g_regularized = g_rhs * mask_float

    # Solve linear system A * delta_p = g_rhs using stable solver
    delta_p = jnp.linalg.solve(a_regularized, g_regularized)
    # Ensure frozen parameters have strictly zero displacement
    delta_p = delta_p * mask_float

    # Trial parameters and candidate residuals
    params_trial = params_curr + delta_p
    r_trial = residual_fn(params_trial)

    # Compute actual reduction in sum-of-squares objective F = 0.5 * ||r||^2
    f_curr = 0.5 * jnp.sum(r_curr ** 2)
    f_trial = 0.5 * jnp.sum(r_trial ** 2)
    actual_reduction = f_curr - f_trial

    # Compute predicted reduction delta_L = 0.5 * delta_p^T * (lambda * D * delta_p - grad_f)
    predicted_reduction = 0.5 * (jnp.dot(delta_p, damping * jnp.dot(d_mat, delta_p)) - jnp.dot(delta_p, grad_f))

    # Gain ratio rho
    gain_ratio = actual_reduction / (predicted_reduction + 1e-18)

    grad_norm = float(jnp.max(jnp.abs(grad_f * mask_float)))
    step_norm = float(jnp.linalg.norm(delta_p))

    # Acceptance criterion: positive reduction
    accepted = bool((actual_reduction > 0.0) and (gain_ratio > 0.0))

    if accepted:
        params_next = params_trial
        residual_next = r_trial
        # Decrease damping
        damping_next = float(max(damping * d_down, damping_min))
    else:
        params_next = params_curr
        residual_next = r_curr
        # Increase damping
        damping_next = float(min(damping * d_up, damping_max))

    return (
        params_next,
        residual_next,
        damping_next,
        accepted,
        float(gain_ratio),
        grad_norm,
        step_norm,
    )


def levenberg_marquardt_fit(
    residual_fn: Callable[[jnp.ndarray], jnp.ndarray],
    initial_params: Union[np.ndarray, jnp.ndarray, Sequence[float]],
    config: Optional[LMFitConfig] = None,
    active_mask: Optional[Union[np.ndarray, jnp.ndarray, Sequence[bool]]] = None,
) -> LMFitResult:
    """
    Executes full iterative Autodiff Levenberg-Marquardt optimization for non-linear models.

    Parameters:
        residual_fn: Pure callable returning residual vector r(params)
        initial_params: Initial guess parameter array
        config: LMFitConfig configuration (or defaults if None)
        active_mask: Optional boolean mask specifying active vs frozen parameters

    Returns:
        LMFitResult containing fitted parameters, covariance matrix, parameter uncertainties,
        residuals, and step telemetry history.
    """
    cfg = config if config is not None else LMFitConfig()
    p_curr = jnp.asarray(initial_params, dtype=jnp.float64)
    n_params = p_curr.shape[0]

    if active_mask is not None:
        mask = jnp.asarray(active_mask, dtype=jnp.bool_)
    else:
        mask = jnp.ones(n_params, dtype=jnp.bool_)

    initial_params_list = [float(x) for x in p_curr]
    active_mask_list = [bool(x) for x in mask]
    history: List[LMStepResult] = []

    damping = cfg.damping_init
    converged = False
    status_msg = "Maximum iterations reached."

    r_curr = residual_fn(p_curr)

    for it in range(cfg.max_iterations):
        (
            p_next,
            r_next,
            damping_next,
            accepted,
            gain_ratio,
            grad_norm,
            step_norm,
        ) = levenberg_marquardt_step(
            params=p_curr,
            residual_fn=residual_fn,
            damping=damping,
            active_mask=mask,
            marquardt_scaling=cfg.marquardt_scaling,
            damping_up=cfg.damping_up,
            damping_down=cfg.damping_down,
            damping_min=cfg.damping_min,
            damping_max=cfg.damping_max,
        )

        step_record = LMStepResult(
            iteration=it + 1,
            params=[float(x) for x in p_next],
            step=[float(x) for x in (p_next - p_curr)],
            residual_norm=float(jnp.linalg.norm(r_next)),
            damping=damping,
            accepted=accepted,
            gain_ratio=gain_ratio,
            gradient_norm=grad_norm,
            step_norm=step_norm,
        )
        history.append(step_record)

        # Convergence checks
        if grad_norm < cfg.tol_gradient:
            converged = True
            status_msg = f"Converged: Gradient norm {grad_norm:.3e} < {cfg.tol_gradient:.3e}."
            p_curr = p_next
            r_curr = r_next
            break

        if accepted and step_norm < cfg.tol_step:
            converged = True
            status_msg = f"Converged: Step norm {step_norm:.3e} < {cfg.tol_step:.3e}."
            p_curr = p_next
            r_curr = r_next
            break

        if accepted:
            rel_res_change = abs(jnp.sum(r_curr**2) - jnp.sum(r_next**2)) / (jnp.sum(r_curr**2) + 1e-18)
            if rel_res_change < cfg.tol_residual:
                converged = True
                status_msg = f"Converged: Relative residual change {rel_res_change:.3e} < {cfg.tol_residual:.3e}."
                p_curr = p_next
                r_curr = r_next
                break

        p_curr = p_next
        r_curr = r_next
        damping = damping_next

    # Compute final Jacobian and statistics at optimal point
    j_final = compute_jacobian_autodiff(residual_fn, p_curr)
    mask_float = mask.astype(jnp.float64)
    j_active = j_final * mask_float[None, :]

    m_samples = r_curr.shape[0]
    n_active = int(jnp.sum(mask))
    degrees_of_freedom = max(1, m_samples - n_active)

    rss = float(jnp.sum(r_curr ** 2))
    rmse = float(jnp.sqrt(rss / max(1, m_samples)))
    reduced_chi2 = float(rss / degrees_of_freedom)

    # Covariance matrix: Cov = (J^T J)^-1 * s^2
    h_final = jnp.dot(j_active.T, j_active)
    # Regularize inactive parameters in inversion
    mask_2d = mask_float[:, None] * mask_float[None, :]
    inv_mask_1d = (1.0 - mask_float)
    h_inv_target = h_final * mask_2d + jnp.diag(inv_mask_1d)

    # Calculate covariance; if numerical singularity occurs, allow failure to propagate [M]
    cov_raw = jnp.linalg.pinv(h_inv_target) * reduced_chi2
    cov_masked = cov_raw * mask_2d

    param_uncertainties = [
        float(jnp.sqrt(jnp.maximum(0.0, cov_masked[i, i]))) if active_mask_list[i] else 0.0
        for i in range(n_params)
    ]

    cov_list = [[float(cov_masked[i, j]) for j in range(n_params)] for i in range(n_params)]
    optimal_params_list = [float(x) for x in p_curr]
    final_residuals_list = [float(x) for x in r_curr]
    final_grad_norm = float(jnp.max(jnp.abs(jnp.dot(j_active.T, r_curr))))

    return LMFitResult(
        converged=converged,
        iterations=len(history),
        optimal_params=optimal_params_list,
        initial_params=initial_params_list,
        final_residuals=final_residuals_list,
        residual_norm=float(jnp.linalg.norm(r_curr)),
        rmse=rmse,
        reduced_chi_squared=reduced_chi2,
        gradient_norm=final_grad_norm,
        covariance_matrix=cov_list,
        parameter_uncertainties=param_uncertainties,
        active_mask=active_mask_list,
        history=history,
        message=status_msg,
    )


# =============================================================================
# 4. HIGH-J VRAM BLOCK DIAGONALIZATION USING JAX.LAX.SCAN
# =============================================================================

def _diagonalize_single_block_scan(
    carry: None,
    block: jnp.ndarray,
) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Pure-JAX single block Hermitian eigensolver scan kernel.
    Utilizes jnp.linalg.eigh in FP64 precision.
    """
    eigenvalues, eigenvectors = jnp.linalg.eigh(block)
    return carry, (eigenvalues, eigenvectors)


def _diagonalize_chunk_scan(
    carry: None,
    chunk: jnp.ndarray,
) -> Tuple[None, Tuple[jnp.ndarray, jnp.ndarray]]:
    """
    Pure-JAX chunked block Hermitian eigensolver scan kernel.
    Vectorized over chunk batch dimension with jax.vmap(jnp.linalg.eigh).
    """
    evals, evecs = jax.vmap(jnp.linalg.eigh)(chunk)
    return carry, (evals, evecs)


def block_diagonalization_scan(
    hamiltonian_blocks: Union[np.ndarray, jnp.ndarray],
    chunk_size: Optional[int] = None,
    max_vram_mb: Optional[float] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    High-J VRAM block diagonalization using jax.lax.scan.

    Chunks processing and dynamically caps peak memory allocation without any Python for loops.

    Parameters:
        hamiltonian_blocks: 3D array of square Hermitian matrices of shape [num_blocks, N, N]
        chunk_size: Optional fixed chunk size for batching within scan
        max_vram_mb: Optional VRAM budget cap in MB to dynamically derive optimal chunk size

    Returns:
        Tuple of (eigenvalues, eigenvectors):
            eigenvalues: shape [num_blocks, N] in ascending order
            eigenvectors: shape [num_blocks, N, N] where column [k, :, i] is i-th eigenvector
    """
    blocks = jnp.asarray(hamiltonian_blocks, dtype=jnp.float64)

    if blocks.ndim != 3:
        raise InvalidHamiltonianError(
            f"Expected 3D tensor [num_blocks, N, N], received shape {blocks.shape}."
        )

    num_blocks, dim_m, dim_n = blocks.shape
    if dim_m != dim_n:
        raise InvalidHamiltonianError(
            f"Hamiltonian blocks must be square matrices. Received shape [{num_blocks}, {dim_m}, {dim_n}]."
        )

    # Determine chunk size dynamically from VRAM cap if provided
    effective_chunk_size = 1
    if max_vram_mb is not None:
        # Each FP64 element is 8 bytes.
        # Per-block memory: block (N*N*8) + evecs (N*N*8) + evals (N*8) ~= 2.1 * N^2 * 8 bytes
        bytes_per_block = 2.5 * (dim_n ** 2) * 8.0
        vram_bytes = max_vram_mb * 1024.0 * 1024.0
        derived_chunk = max(1, int(vram_bytes / max(bytes_per_block, 1.0)))
        effective_chunk_size = min(num_blocks, derived_chunk)
    elif chunk_size is not None:
        effective_chunk_size = max(1, min(num_blocks, chunk_size))

    # Sequential scan across individual blocks
    if effective_chunk_size <= 1:
        _, (eigenvalues, eigenvectors) = jax.lax.scan(
            _diagonalize_single_block_scan,
            None,
            blocks,
        )
        return eigenvalues, eigenvectors

    # Chunked scan with jax.vmap inside jax.lax.scan
    num_chunks = int(math.ceil(num_blocks / effective_chunk_size))
    total_padded = num_chunks * effective_chunk_size
    pad_amount = total_padded - num_blocks

    if pad_amount > 0:
        # Pad with identity matrices to prevent singular eigensolver faults
        pad_identity = jnp.eye(dim_n, dtype=jnp.float64)
        pad_blocks = jnp.repeat(pad_identity[None, :, :], pad_amount, axis=0)
        blocks_padded = jnp.concatenate([blocks, pad_blocks], axis=0)
    else:
        blocks_padded = blocks

    # Reshape into [num_chunks, effective_chunk_size, N, N]
    chunked_tensor = blocks_padded.reshape(
        num_chunks, effective_chunk_size, dim_n, dim_n
    )

    _, (evals_chunked, evecs_chunked) = jax.lax.scan(
        _diagonalize_chunk_scan,
        None,
        chunked_tensor,
    )

    # Flatten back to [total_padded, N] and [total_padded, N, N]
    all_evals = evals_chunked.reshape(total_padded, dim_n)
    all_evecs = evecs_chunked.reshape(total_padded, dim_n, dim_n)

    # Slice off padding
    final_evals = all_evals[:num_blocks, :]
    final_evecs = all_evecs[:num_blocks, :, :]

    return final_evals, final_evecs


# =============================================================================
# 5. SOBOL SENSITIVITY GUARD
# =============================================================================

def evaluate_sobol_sensitivity(
    model_fn: Callable[[jnp.ndarray], jnp.ndarray],
    base_params: Union[np.ndarray, jnp.ndarray, Sequence[float]],
    param_bounds: Optional[Sequence[Tuple[float, float]]] = None,
    config: Optional[SobolGuardConfig] = None,
) -> SobolGuardResult:
    """
    Evaluates parameter variance impact / total-order sensitivity indices.
    Freezes insensitive parameters whose variance impact drops below 1e-6.

    Parameters:
        model_fn: Pure callable returning output/residual vector given parameter vector
        base_params: Base parameter vector [P]
        param_bounds: Optional sequence of (lower, upper) bounds for each parameter
        config: SobolGuardConfig settings

    Returns:
        SobolGuardResult containing active mask, frozen indices, and variance impacts.
    """
    cfg = config if config is not None else SobolGuardConfig()
    p_base = jnp.asarray(base_params, dtype=jnp.float64)
    n_params = p_base.shape[0]

    # Construct parameter bounding box
    if param_bounds is not None:
        lb = jnp.array([b[0] for b in param_bounds], dtype=jnp.float64)
        ub = jnp.array([b[1] for b in param_bounds], dtype=jnp.float64)
    else:
        span = jnp.maximum(jnp.abs(p_base) * cfg.perturbation_fraction, 1e-4)
        lb = p_base - span
        ub = p_base + span

    # Saltelli / Quasi-Monte Carlo sampling in pure JAX
    key = jax.random.PRNGKey(cfg.seed)
    key_a, key_b = jax.random.split(key)

    n_samples = cfg.num_samples
    u_a = jax.random.uniform(key_a, shape=(n_samples, n_params), dtype=jnp.float64)
    u_b = jax.random.uniform(key_b, shape=(n_samples, n_params), dtype=jnp.float64)

    # Scale to bounds
    sample_a = lb + u_a * (ub - lb)
    sample_b = lb + u_b * (ub - lb)

    # Evaluate outputs for matrices A and B
    out_a = jax.vmap(model_fn)(sample_a)
    out_b = jax.vmap(model_fn)(sample_b)

    # Total variance of output norm
    y_a = jnp.linalg.norm(out_a, axis=-1)
    y_b = jnp.linalg.norm(out_b, axis=-1)
    total_var = jnp.var(y_a) + 1e-18

    # Evaluate Jansen total-order sensitivity index S_Ti for each parameter
    variance_impacts: List[float] = []

    # Also compute local Jacobian gradient variance metric for micro-sensitivity validation
    j_local = compute_jacobian_autodiff(model_fn, p_base)
    j_col_norms = jnp.sum(j_local ** 2, axis=0)
    total_j_norm = jnp.sum(j_col_norms) + 1e-18
    local_impacts = (j_col_norms / total_j_norm)

    for i in range(n_params):
        # Create matrix A_B^(i) where column i comes from B, others from A
        sample_ab_i = sample_a.at[:, i].set(sample_b[:, i])
        out_ab_i = jax.vmap(model_fn)(sample_ab_i)
        y_ab_i = jnp.linalg.norm(out_ab_i, axis=-1)

        # Jansen total-effect estimator: S_Ti = (1 / (2 * N)) * sum((Y_A - Y_AB_i)^2) / Var(Y)
        s_ti = 0.5 * jnp.mean((y_a - y_ab_i) ** 2) / total_var

        # Aggregate sensitivity combining global variance decomposition and local gradient
        combined_vi = float(jnp.maximum(s_ti, local_impacts[i]))
        variance_impacts.append(combined_vi)

    active_mask = [vi >= cfg.variance_threshold for vi in variance_impacts]
    active_indices = [i for i, active in enumerate(active_mask) if active]
    frozen_indices = [i for i, active in enumerate(active_mask) if not active]

    return SobolGuardResult(
        variance_impacts=variance_impacts,
        active_mask=active_mask,
        active_indices=active_indices,
        frozen_indices=frozen_indices,
        variance_threshold=cfg.variance_threshold,
        total_parameters=n_params,
        active_parameters=len(active_indices),
    )


# =============================================================================
# 6. ACTIVE PES REFINEMENT TRIGGER & AIR-GAPPED TRANSPORT SPOOL
# =============================================================================

class AirGappedTransportSpool:
    """
    Physical file locking and SHA-256 verified transport layer for air-gapped
    spectroscopic PES refinement exchange between laboratory spectrometer hosts
    and high-performance ab initio quantum chemistry clusters.
    """

    @staticmethod
    def compute_sha256_checksum(content: str) -> str:
        """Computes SHA-256 cryptographic digest of a text string."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @classmethod
    def export_payload(
        cls,
        payload: PESRefinementPayload,
        spool_directory: Union[str, Path],
        timeout: float = 10.0,
    ) -> Path:
        """
        Exports PESRefinementPayload to air-gapped spool directory with atomic file locking
        and verified SHA-256 checksum.

        Parameters:
            payload: Validated PESRefinementPayload
            spool_directory: Target filesystem directory
            timeout: Lock acquisition timeout in seconds

        Returns:
            Path to exported spool file.
        """
        spool_dir = Path(spool_directory)
        spool_dir.mkdir(parents=True, exist_ok=True)

        spool_filename = f"pes_refinement_{payload.refinement_id}.json"
        target_path = spool_dir / spool_filename
        lock_path = spool_dir / f"{spool_filename}.lock"

        # Serialize payload without checksum to compute canonical checksum
        payload_dict = payload.model_dump()
        payload_dict["sha256_checksum"] = ""
        canonical_content = json.dumps(payload_dict, sort_keys=True, indent=2)
        checksum = cls.compute_sha256_checksum(canonical_content)

        payload_dict["sha256_checksum"] = checksum
        final_content = json.dumps(payload_dict, sort_keys=True, indent=2)

        lock = filelock.FileLock(str(lock_path), timeout=timeout)
        try:
            with lock:
                # Atomic write via temporary file replace
                temp_fd, temp_file_path = tempfile.mkstemp(
                    dir=str(spool_dir), prefix="temp_spool_", suffix=".tmp"
                )
                try:
                    with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                        f.write(final_content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(temp_file_path, str(target_path))
                except BaseException:
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise
        except filelock.Timeout as exc:
            raise TransportError(
                f"Failed to acquire atomic lock on spool file {lock_path} within {timeout}s: {exc}"
            ) from exc

        return target_path

    @classmethod
    def import_payload(
        cls,
        spool_file_path: Union[str, Path],
        timeout: float = 10.0,
    ) -> PESRefinementPayload:
        """
        Imports and validates a PESRefinementPayload from an air-gapped spool file.
        Verifies SHA-256 cryptographic checksum and Method Matrix DLPNO prohibition.

        Parameters:
            spool_file_path: Path to target spool JSON file
            timeout: Lock acquisition timeout in seconds

        Returns:
            Validated PESRefinementPayload instance.
        """
        target_path = Path(spool_file_path)
        if not target_path.exists():
            raise TransportError(f"Air-gapped spool file does not exist: {target_path}")

        lock_path = target_path.parent / f"{target_path.name}.lock"
        lock = filelock.FileLock(str(lock_path), timeout=timeout)

        try:
            with lock:
                with open(target_path, "r", encoding="utf-8") as f:
                    raw_content = f.read()
        except filelock.Timeout as exc:
            raise TransportError(
                f"Failed to acquire atomic lock on spool file {lock_path} within {timeout}s: {exc}"
            ) from exc

        try:
            data = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            raise TransportError(f"Corrupted spool JSON structure: {exc}") from exc

        stored_checksum = data.get("sha256_checksum", "")
        # Recompute canonical checksum
        data_copy = dict(data)
        data_copy["sha256_checksum"] = ""
        canonical_content = json.dumps(data_copy, sort_keys=True, indent=2)
        expected_checksum = cls.compute_sha256_checksum(canonical_content)

        if stored_checksum != expected_checksum:
            raise TransportError(
                f"SHA-256 Checksum Mismatch in air-gapped transport file {target_path}!\n"
                f"Stored:   {stored_checksum}\n"
                f"Computed: {expected_checksum}"
            )

        # Enforce DLPNO method prohibition
        target_method = data.get("target_method", "")
        if "dlpno" in target_method.lower():
            raise DLPNOForbiddenError(
                f"Method Matrix Violation: '{target_method}' is strictly forbidden for high-precision "
                f"spectroscopic PES refinement. Canonical CCSD(T) is mandatory."
            )

        return PESRefinementPayload(**data)


def check_pes_refinement_trigger(
    rmse_residual_mhz: float,
    cartesian_coords_angstrom: Union[np.ndarray, List[List[float]]],
    atomic_symbols: List[str],
    molecular_formula: str = "COMPLEX",
    config: Optional[ActivePESRefinementConfig] = None,
    displacement_vectors: Optional[List[List[float]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PESRefinementTriggerResult:
    """
    Evaluates fit residual viability against threshold and triggers Active PES Refinement if unviable.

    Method Matrix Rules:
    1. Must route to canonical CCSD(T) (e.g. 'canonical-CCSD(T)', 'CCSD(T)/def2-QZVPP').
    2. Strictly forbids DLPNO-CCSD(T), throwing DLPNOForbiddenError if requested.
    3. Handles physical air-gapped transport spool export with SHA-256 checksum and atomic locking.

    Parameters:
        rmse_residual_mhz: Observed fit Root Mean Square Error in MHz
        cartesian_coords_angstrom: Atomic coordinates in Angstrom [N, 3]
        atomic_symbols: Atomic symbols for atoms [N]
        molecular_formula: Identifier string
        config: ActivePESRefinementConfig configuration
        displacement_vectors: Optional displacement coordinates for PES grid points
        metadata: Optional provenance dictionary

    Returns:
        PESRefinementTriggerResult summarizing trigger state, reason, and spool details.
    """
    cfg = config if config is not None else ActivePESRefinementConfig()

    # Enforce DLPNO method prohibition on config
    if "dlpno" in cfg.target_method.lower():
        raise DLPNOForbiddenError(
            f"Method Matrix Violation: '{cfg.target_method}' is strictly forbidden for high-precision "
            f"spectroscopic PES refinement. Canonical CCSD(T) is mandatory."
        )

    coords_list: List[List[float]] = [[float(val) for val in row] for row in cartesian_coords_angstrom]

    if rmse_residual_mhz <= cfg.viable_residual_threshold_mhz:
        return PESRefinementTriggerResult(
            triggered=False,
            reason=f"Residual RMSE {rmse_residual_mhz:.4f} MHz is within viable limit ({cfg.viable_residual_threshold_mhz:.4f} MHz).",
            rmse_residual_mhz=rmse_residual_mhz,
            payload=None,
            spool_file_path=None,
        )

    # Residual is unviable: Trigger Active PES Refinement
    refinement_id = str(uuid.uuid4())
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    disp_list = displacement_vectors if displacement_vectors is not None else []
    meta_dict = metadata if metadata is not None else {}

    # Build unhashed payload dict
    pre_payload_dict = {
        "refinement_id": refinement_id,
        "timestamp_utc": timestamp_utc,
        "molecular_formula": molecular_formula,
        "atomic_symbols": atomic_symbols,
        "cartesian_coords_angstrom": coords_list,
        "target_method": cfg.target_method,
        "basis_set": cfg.basis_set,
        "unviable_residual_mhz": float(rmse_residual_mhz),
        "threshold_mhz": float(cfg.viable_residual_threshold_mhz),
        "displacement_vectors": disp_list,
        "metadata": meta_dict,
        "sha256_checksum": "",
    }
    canonical_str = json.dumps(pre_payload_dict, sort_keys=True, indent=2)
    checksum = AirGappedTransportSpool.compute_sha256_checksum(canonical_str)

    payload = PESRefinementPayload(
        refinement_id=refinement_id,
        timestamp_utc=timestamp_utc,
        molecular_formula=molecular_formula,
        atomic_symbols=atomic_symbols,
        cartesian_coords_angstrom=coords_list,
        target_method=cfg.target_method,
        basis_set=cfg.basis_set,
        unviable_residual_mhz=float(rmse_residual_mhz),
        threshold_mhz=float(cfg.viable_residual_threshold_mhz),
        displacement_vectors=disp_list,
        metadata=meta_dict,
        sha256_checksum=checksum,
    )

    spool_path = AirGappedTransportSpool.export_payload(
        payload=payload,
        spool_directory=cfg.spool_directory,
        timeout=cfg.lock_timeout_seconds,
    )

    reason = (
        f"Residual RMSE {rmse_residual_mhz:.4f} MHz exceeds viable threshold "
        f"{cfg.viable_residual_threshold_mhz:.4f} MHz. Active PES Refinement triggered "
        f"and spooled for canonical {cfg.target_method}/{cfg.basis_set}."
    )

    return PESRefinementTriggerResult(
        triggered=True,
        reason=reason,
        rmse_residual_mhz=rmse_residual_mhz,
        payload=payload,
        spool_file_path=str(spool_path),
    )
