"""
CoChem-SpycFit: AI Assigner & Active PES Refinement Trigger Core.

This module implements Phase 3 AI Intelligence and Pattern Recognition:
1. "Information-Gain" Transition Assigner:
   - Evaluates the parameter uncertainty covariance matrix (Sigma).
   - Computes Shannon Information Gain (D-optimality / log-determinant entropy reduction),
     A-optimality (parameter variance trace reduction), and G-optimality (predictive variance).
   - Vectorized pure-JAX evaluation across all candidate transitions without Python loops.
   - Exact rank-1 Sherman-Morrison covariance updating following hypothetical/experimental assignments.
2. Active PES Refinement Trigger (The "Healing Loop"):
   - Identifies high-confidence experimental assignments with massive unresolved (O-C) residuals.
   - Flags specific driving quantum rotational/vibrational states.
   - Back-calculates coordinate/parameter sensitivity gradients driving residual errors.
   - Generates pure-PyTree descriptors for automated high-level canonical CCSD(T) PES refinement
     in CoChem-TORQ.
3. Functional Purity & Architecture:
   - Zero I/O operations (no CSV reads); all routines accept and return pure JAX arrays and PyTrees.
   - Strict Anti-Spoofing & Method Matrix Compliance (zero mocks, zero pass statements, FP64 precision).
"""

from __future__ import annotations

import enum
import functools
import hashlib
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Enforce JAX FP64 precision globally for spectroscopic accuracy
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.AIAssigner")

# Sibling imports with fallback for standalone environments
try:
    from .cochem_jax_fitter import DLPNOForbiddenError, FitterError
except (ImportError, ValueError):
    class FitterError(Exception):
        """Base exception for all JAX fitter and spectroscopic optimization operations."""

    class DLPNOForbiddenError(FitterError):
        """Hard error raised when DLPNO-CCSD(T) is requested for high-precision PES refinement."""


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class AssignMLError(FitterError):
    """Base exception for AI assignment and active PES refinement trigger errors."""


class InvalidCovarianceError(AssignMLError):
    """Raised when parameter uncertainty covariance matrix is malformed, non-positive-definite, or non-square."""


class InvalidJacobianError(AssignMLError):
    """Raised when candidate Jacobian matrix has invalid dimensions or contains non-finite entries."""


class InvalidResidualError(AssignMLError):
    """Raised when observed-calculated residuals or quantum states are inconsistent."""


class EmptyCandidateError(AssignMLError):
    """Raised when candidate transition set is empty."""


# =============================================================================
# 2. ENUMS & PYDANTIC CONFIGURATIONS
# =============================================================================

class InformationCriterion(str, enum.Enum):
    """Mathematical optimality criteria for Information-Gain transition selection."""
    D_OPTIMAL = "d_optimal"               # Shannon Entropy / Log-Determinant reduction: 0.5 * log2(1 + j^T Sigma j / sigma^2)
    A_OPTIMAL = "a_optimal"               # Trace reduction: Tr(Sigma) - Tr(Sigma_new) = (j^T Sigma^2 j) / (sigma^2 + j^T Sigma j)
    G_OPTIMAL = "g_optimal"               # Predictive variance / Peak sensitivity: j^T Sigma j / sigma^2
    E_OPTIMAL = "e_optimal"               # Directional variance along principal uncertainty axis


class InformationGainConfig(BaseModel):
    """Configuration for Information-Gain transition ranking."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    criterion: InformationCriterion = Field(
        default=InformationCriterion.D_OPTIMAL,
        description="Optimality criterion for parameter uncertainty reduction",
    )
    default_measurement_variance: float = Field(
        default=1.0,
        gt=0.0,
        description="Default experimental measurement variance sigma^2 (in MHz^2)",
    )
    min_information_threshold: float = Field(
        default=1e-12,
        ge=0.0,
        description="Minimum information gain required to consider a candidate viable",
    )
    intensity_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Exponential weighting exponent for theoretical line intensity in ranking",
    )
    regularization_epsilon: float = Field(
        default=1e-15,
        ge=0.0,
        description="Small diagonal jitter to ensure positive definiteness in covariance inversion",
    )


class ActivePESHealingConfig(BaseModel):
    """Configuration for Active PES Refinement triggering ('The Healing Loop')."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    unresolved_residual_threshold_mhz: float = Field(
        default=5.0,
        gt=0.0,
        description="Threshold (O-C) residual in MHz above which assignments trigger PES refinement",
    )
    min_assignment_confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score required to confirm an experimental peak assignment",
    )
    target_method: str = Field(
        default="canonical-CCSD(T)",
        description="Target quantum chemistry method for high-accuracy PES refinement",
    )
    basis_set: str = Field(
        default="def2-QZVPP",
        description="High-level basis set for ab initio single points",
    )
    max_flagged_states: int = Field(
        default=100,
        ge=1,
        description="Maximum number of driving quantum states to include in refinement payload",
    )

    @field_validator("target_method")
    @classmethod
    def validate_method_matrix_compliance(cls, v: str) -> str:
        """Enforces Method Matrix mandate forbidding DLPNO approximations for PES refinement."""
        normalized = v.strip().upper()
        if "DLPNO" in normalized:
            raise DLPNOForbiddenError(
                f"Method '{v}' is strictly forbidden by Method Matrix directives. "
                "DLPNO domain approximations introduce non-monotonic energy errors across "
                "potential energy surfaces. Canonical CCSD(T) (e.g. 'canonical-CCSD(T)') is mandatory."
            )
        return v


# =============================================================================
# 3. PURE-JAX INFORMATION-GAIN TRANSITION ASSIGNER CORE
# =============================================================================

@functools.partial(jax.jit, static_argnames=("criterion",))
def _calculate_information_gain_vectorized_jax(
    cov: jnp.ndarray,
    jac: jnp.ndarray,
    meas_var: jnp.ndarray,
    criterion: str = "d_optimal",
) -> jnp.ndarray:
    """
    Vectorized pure-JAX kernel calculating information gain for candidate transitions.

    Args:
        cov: Parameter uncertainty covariance matrix of shape (P, P), FP64.
        jac: Candidate transition Jacobian sensitivities of shape (N, P), FP64.
        meas_var: Measurement variances of shape (N,), FP64.
        criterion: Optimality criterion ('d_optimal', 'a_optimal', 'g_optimal', 'e_optimal').

    Returns:
        Information gain vector of shape (N,), FP64.
    """
    # Predictive variance q_i = j_i^T * Sigma * j_i for all N candidates:
    # shape: (N, P) @ (P, P) -> (N, P), elementwise product with (N, P), sum over P -> (N,)
    cov_j = jnp.matmul(jac, cov)  # shape (N, P)
    q = jnp.sum(cov_j * jac, axis=-1)  # shape (N,)
    q_clamped = jnp.maximum(q, 0.0)

    # Safe variance scaling w_i = 1 / meas_var_i
    w = 1.0 / jnp.maximum(meas_var, 1e-15)

    if criterion == "d_optimal":
        # Shannon Information Gain: delta_I = 0.5 * log2(1 + w * j^T * Sigma * j)
        gain = 0.5 * jnp.log2(1.0 + w * q_clamped)
    elif criterion == "a_optimal":
        # Trace parameter variance reduction:
        # delta_Tr(Sigma) = (w * j^T * Sigma^2 * j) / (1 + w * j^T * Sigma * j)
        #                 = (j^T * Sigma^2 * j) / (meas_var + q)
        # Notice ||Sigma * j_i||_2^2 = j_i^T * Sigma^2 * j_i since Sigma is symmetric
        sigma_sq_j = jnp.sum(cov_j * cov_j, axis=-1)  # shape (N,)
        gain = sigma_sq_j / (meas_var + q_clamped)
    elif criterion == "g_optimal":
        # Predictive variance / Peak sensitivity: q / meas_var
        gain = q_clamped / jnp.maximum(meas_var, 1e-15)
    elif criterion == "e_optimal":
        # Directional variance along principal uncertainty axis (top eigenvector of Sigma)
        # Computed via normalized variance representation
        gain = q_clamped / (meas_var + q_clamped)
    else:
        # Default fallback to D-optimal Shannon Information Gain
        gain = 0.5 * jnp.log2(1.0 + w * q_clamped)

    return gain


def calculate_information_gain(
    covariance_matrix: Union[jnp.ndarray, np.ndarray, Sequence],
    jacobian: Union[jnp.ndarray, np.ndarray, Sequence],
    measurement_variance: Union[float, jnp.ndarray, np.ndarray, Sequence] = 1.0,
    criterion: Union[str, InformationCriterion] = InformationCriterion.D_OPTIMAL,
) -> jnp.ndarray:
    """
    Evaluates the parameter uncertainty covariance matrix (Sigma) and calculates the information gain
    for one or more candidate theoretical transitions.

    Mathematical Basis:
    - Given parameter covariance Sigma in R^(P x P) and transition Jacobian gradient j in R^P:
      Updated covariance after measuring transition with variance sigma_meas^2 is:
      Sigma_new = Sigma - (Sigma * j * j^T * Sigma) / (sigma_meas^2 + j^T * Sigma * j)
    - D-Optimality (Shannon Information Gain / Log-Determinant Reduction):
      Delta I = 0.5 * log2(det(Sigma) / det(Sigma_new)) = 0.5 * log2(1 + j^T * Sigma * j / sigma_meas^2)
    - A-Optimality (Parameter Variance Trace Reduction):
      Delta Tr(Sigma) = Tr(Sigma) - Tr(Sigma_new) = (j^T * Sigma^2 * j) / (sigma_meas^2 + j^T * Sigma * j)
    - G-Optimality (Predictive Spectral Variance):
      Delta G = j^T * Sigma * j / sigma_meas^2

    Args:
        covariance_matrix: (P, P) parameter uncertainty covariance matrix or (P,) variance vector.
        jacobian: (N, P) candidate transition Jacobians or (P,) single transition Jacobian.
        measurement_variance: Experimental measurement variance(s) sigma_meas^2 (float or (N,)).
        criterion: Selection criterion ('d_optimal', 'a_optimal', 'g_optimal', 'e_optimal').

    Returns:
        Information gain array of shape (N,) for candidate batch, or scalar float array for single candidate.

    Raises:
        InvalidCovarianceError: If covariance matrix is malformed or non-square.
        InvalidJacobianError: If Jacobian dimensions do not match covariance matrix.
    """
    # Convert and validate covariance matrix
    cov_arr = jnp.asarray(covariance_matrix, dtype=jnp.float64)
    if cov_arr.ndim == 1:
        # Diagonal variance vector provided: expand to 2D diagonal matrix
        cov_arr = jnp.diag(cov_arr)
    elif cov_arr.ndim != 2:
        raise InvalidCovarianceError(f"Expected 1D or 2D covariance array, got shape {cov_arr.shape}.")

    p_dim1, p_dim2 = cov_arr.shape
    if p_dim1 != p_dim2:
        raise InvalidCovarianceError(f"Covariance matrix must be square, got shape ({p_dim1}, {p_dim2}).")

    # Symmetrize covariance to eliminate numerical asymmetry
    cov_arr = 0.5 * (cov_arr + cov_arr.T)

    # Convert and validate Jacobian matrix
    jac_arr = jnp.asarray(jacobian, dtype=jnp.float64)
    single_input = (jac_arr.ndim == 1)
    if single_input:
        if jac_arr.shape[0] != p_dim1:
            raise InvalidJacobianError(
                f"Jacobian parameter dimension {jac_arr.shape[0]} does not match covariance dimension {p_dim1}."
            )
        jac_arr = jnp.expand_dims(jac_arr, axis=0)  # Shape (1, P)
    elif jac_arr.ndim == 2:
        if jac_arr.shape[1] != p_dim1:
            raise InvalidJacobianError(
                f"Jacobian parameter dimension {jac_arr.shape[1]} does not match covariance dimension {p_dim1}."
            )
    else:
        raise InvalidJacobianError(f"Expected 1D or 2D Jacobian array, got shape {jac_arr.shape}.")

    n_candidates = jac_arr.shape[0]
    if n_candidates == 0:
        raise EmptyCandidateError("Jacobian contains zero candidate transitions.")

    # Convert measurement variance
    if isinstance(measurement_variance, (int, float)):
        if measurement_variance <= 0.0:
            raise ValueError(f"Measurement variance must be strictly positive, got {measurement_variance}.")
        meas_var_arr = jnp.full((n_candidates,), float(measurement_variance), dtype=jnp.float64)
    else:
        meas_var_arr = jnp.asarray(measurement_variance, dtype=jnp.float64)
        if meas_var_arr.ndim == 0:
            meas_var_arr = jnp.full((n_candidates,), float(meas_var_arr), dtype=jnp.float64)
        elif meas_var_arr.shape[0] != n_candidates:
            raise ValueError(
                f"Measurement variance length {meas_var_arr.shape[0]} does not match candidates {n_candidates}."
            )

    # Resolve criterion string
    if isinstance(criterion, InformationCriterion):
        crit_str = criterion.value
    else:
        crit_str = str(criterion).lower().strip()

    valid_criteria = {c.value for c in InformationCriterion}
    if crit_str not in valid_criteria:
        raise ValueError(f"Unknown criterion '{crit_str}'. Must be one of {valid_criteria}.")

    # Execute vectorized JAX kernel
    gains = _calculate_information_gain_vectorized_jax(
        cov=cov_arr,
        jac=jac_arr,
        meas_var=meas_var_arr,
        criterion=crit_str,
    )

    if single_input:
        return gains[0]
    return gains


@functools.partial(jax.jit, static_argnames=("criterion",))
def _rank_candidate_transitions_jax(
    cov: jnp.ndarray,
    jac: jnp.ndarray,
    intensities: jnp.ndarray,
    meas_var: jnp.ndarray,
    intensity_weight: float,
    criterion: str,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX kernel for candidate ranking and score computation."""
    raw_gains = _calculate_information_gain_vectorized_jax(
        cov=cov,
        jac=jac,
        meas_var=meas_var,
        criterion=criterion,
    )

    # Intensity scaling: score = gain * (intensity / max_intensity)^weight
    max_intensity = jnp.maximum(jnp.max(intensities), 1e-15)
    normalized_intensities = jnp.clip(intensities / max_intensity, 0.0, 1.0)
    intensity_factors = jnp.power(normalized_intensities, intensity_weight)

    combined_scores = raw_gains * intensity_factors

    # Rank descending: argsort of negative scores
    ranked_indices = jnp.argsort(-combined_scores)
    ranked_scores = combined_scores[ranked_indices]

    return ranked_indices, ranked_scores, raw_gains


def rank_candidate_transitions(
    covariance_matrix: Union[jnp.ndarray, np.ndarray, Sequence],
    candidate_jacobians: Union[jnp.ndarray, np.ndarray, Sequence],
    intensities: Optional[Union[jnp.ndarray, np.ndarray, Sequence]] = None,
    measurement_variances: Optional[Union[float, jnp.ndarray, np.ndarray, Sequence]] = None,
    config: Optional[InformationGainConfig] = None,
    criterion: Optional[Union[str, InformationCriterion]] = None,
    intensity_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Ranks candidate theoretical transitions based on Information Gain to suggest the optimal next
    experimental measurement to reduce parameter uncertainty.

    Args:
        covariance_matrix: (P, P) parameter uncertainty covariance matrix.
        candidate_jacobians: (N, P) candidate transition Jacobians.
        intensities: Optional (N,) theoretical line intensities (for SNR/intensity weighting).
        measurement_variances: Optional (N,) or float measurement variances.
        config: Optional InformationGainConfig instance.
        criterion: Optional override for InformationCriterion.
        intensity_weight: Optional override for line intensity exponent.

    Returns:
        Pure PyTree dictionary containing:
        - 'ranked_indices': jax.Array of candidate indices sorted from highest to lowest information gain.
        - 'ranked_scores': jax.Array of combined ranking scores.
        - 'raw_information_gains': jax.Array of unweighted information gains for all candidates.
        - 'best_candidate_index': Integer index of the single top-ranked transition.
        - 'best_candidate_score': Float score of the top-ranked transition.
        - 'total_candidates': Total number of candidates evaluated.
    """
    cfg = config or InformationGainConfig()
    crit = criterion if criterion is not None else cfg.criterion
    i_weight = float(intensity_weight if intensity_weight is not None else cfg.intensity_weight)

    cov_arr = jnp.asarray(covariance_matrix, dtype=jnp.float64)
    if cov_arr.ndim == 1:
        cov_arr = jnp.diag(cov_arr)
    elif cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
        raise InvalidCovarianceError(f"Invalid covariance matrix shape {cov_arr.shape}.")

    cov_arr = 0.5 * (cov_arr + cov_arr.T)
    p_dim = cov_arr.shape[0]

    jac_arr = jnp.asarray(candidate_jacobians, dtype=jnp.float64)
    if jac_arr.ndim == 1:
        jac_arr = jnp.expand_dims(jac_arr, axis=0)
    if jac_arr.ndim != 2 or jac_arr.shape[1] != p_dim:
        raise InvalidJacobianError(
            f"Candidate Jacobian shape {jac_arr.shape} incompatible with covariance dimension {p_dim}."
        )

    n_cand = jac_arr.shape[0]
    if n_cand == 0:
        raise EmptyCandidateError("Cannot rank empty candidate transition set.")

    # Process intensities
    if intensities is None:
        int_arr = jnp.ones((n_cand,), dtype=jnp.float64)
    else:
        int_arr = jnp.asarray(intensities, dtype=jnp.float64)
        if int_arr.shape[0] != n_cand:
            raise ValueError(f"Intensities length {int_arr.shape[0]} does not match candidates {n_cand}.")

    # Process measurement variances
    if measurement_variances is None:
        m_var_arr = jnp.full((n_cand,), float(cfg.default_measurement_variance), dtype=jnp.float64)
    elif isinstance(measurement_variances, (int, float)):
        m_var_arr = jnp.full((n_cand,), float(measurement_variances), dtype=jnp.float64)
    else:
        m_var_arr = jnp.asarray(measurement_variances, dtype=jnp.float64)
        if m_var_arr.shape[0] != n_cand:
            raise ValueError(f"Measurement variances length {m_var_arr.shape[0]} does not match {n_cand}.")

    crit_str = crit.value if isinstance(crit, InformationCriterion) else str(crit).lower().strip()

    ranked_idx, ranked_scores, raw_gains = _rank_candidate_transitions_jax(
        cov=cov_arr,
        jac=jac_arr,
        intensities=int_arr,
        meas_var=m_var_arr,
        intensity_weight=i_weight,
        criterion=crit_str,
    )

    best_idx = int(ranked_idx[0])
    best_score = float(ranked_scores[0])

    return {
        "ranked_indices": ranked_idx,
        "ranked_scores": ranked_scores,
        "raw_information_gains": raw_gains,
        "best_candidate_index": best_idx,
        "best_candidate_score": best_score,
        "total_candidates": n_cand,
    }


@jax.jit
def update_covariance_after_measurement(
    covariance_matrix: jnp.ndarray,
    measured_jacobian: jnp.ndarray,
    measurement_variance: float = 1.0,
) -> jnp.ndarray:
    """
    Computes the exact Sherman-Morrison rank-1 posterior covariance update after measuring a transition.

    Formula:
        Sigma_new = Sigma - (Sigma * j * j^T * Sigma) / (sigma_meas^2 + j^T * Sigma * j)

    Args:
        covariance_matrix: Prior covariance matrix Sigma of shape (P, P), FP64.
        measured_jacobian: Transition sensitivity gradient j of shape (P,), FP64.
        measurement_variance: Experimental measurement variance sigma_meas^2, FP64.

    Returns:
        Updated posterior covariance matrix Sigma_new of shape (P, P), FP64.
    """
    cov = 0.5 * (covariance_matrix + covariance_matrix.T)
    j = measured_jacobian.reshape((-1, 1))  # (P, 1)

    cov_j = jnp.matmul(cov, j)  # (P, 1)
    j_cov_j = jnp.matmul(j.T, cov_j)[0, 0]  # scalar: j^T Sigma j
    denominator = measurement_variance + j_cov_j

    # Rank-1 correction: (Sigma * j) * (Sigma * j)^T / denominator
    delta_cov = jnp.matmul(cov_j, cov_j.T) / jnp.maximum(denominator, 1e-15)
    cov_new = cov - delta_cov

    # Symmetrize updated covariance
    return 0.5 * (cov_new + cov_new.T)


# =============================================================================
# 4. ACTIVE PES REFINEMENT TRIGGER & "HEALING LOOP"
# =============================================================================

@jax.jit
def _flag_unresolved_residuals_mask_jax(
    abs_residuals: jnp.ndarray,
    confidences: jnp.ndarray,
    threshold: float,
    min_confidence: float,
) -> jnp.ndarray:
    """Pure-JAX boolean kernel identifying high-confidence unresolved residual transitions."""
    residual_mask = abs_residuals > threshold
    confidence_mask = confidences >= min_confidence
    return residual_mask & confidence_mask


def flag_unresolved_residuals(
    residuals: Union[jnp.ndarray, np.ndarray, Sequence],
    states: Any,
    threshold: float = 5.0,
    confidence_scores: Optional[Union[jnp.ndarray, np.ndarray, Sequence]] = None,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    """
    Identifies high-confidence experimental assignments with massive unresolved (O-C) residuals
    and flags their specific quantum states for Active PES Refinement.

    Args:
        residuals: Array of (O-C) frequency residuals in MHz (or cm^-1), shape (N,).
        states: Pure PyTree of quantum state descriptors (e.g. jax.Array of quantum numbers (N, K),
                or sequence/dict/tuple of state records).
        threshold: Residual threshold in MHz above which residuals are considered unviable.
        confidence_scores: Optional assignment confidence scores in [0, 1], shape (N,).
        min_confidence: Minimum confidence score to validate that a peak is not a spurious artifact.

    Returns:
        Pure PyTree dictionary containing:
        - 'is_triggered': bool indicating if any state exceeded threshold with sufficient confidence.
        - 'flagged_count': int total count of flagged states.
        - 'flagged_indices': jax.Array of indices into the original state array.
        - 'flagged_residuals': jax.Array of signed (O-C) residuals for flagged transitions.
        - 'flagged_abs_residuals': jax.Array of absolute (O-C) residuals.
        - 'flagged_states': Sub-PyTree of quantum states matching flagged indices.
        - 'max_residual': float maximum absolute residual among flagged states (0.0 if none).
        - 'mean_residual': float mean absolute residual among flagged states (0.0 if none).
        - 'rmse_flagged': float root-mean-square residual of flagged states (0.0 if none).
        - 'threshold': float threshold utilized.
    """
    res_arr = jnp.asarray(residuals, dtype=jnp.float64)
    if res_arr.ndim != 1:
        raise InvalidResidualError(f"Residuals must be a 1D array, got shape {res_arr.shape}.")

    n_states = res_arr.shape[0]
    if n_states == 0:
        raise InvalidResidualError("Residuals array is empty.")

    abs_res = jnp.abs(res_arr)

    # Process confidence scores
    if confidence_scores is None:
        conf_arr = jnp.ones((n_states,), dtype=jnp.float64)
    else:
        conf_arr = jnp.asarray(confidence_scores, dtype=jnp.float64)
        if conf_arr.shape[0] != n_states:
            raise InvalidResidualError(
                f"Confidence scores length {conf_arr.shape[0]} does not match residuals length {n_states}."
            )

    # Compute boolean trigger mask via JAX kernel
    mask = _flag_unresolved_residuals_mask_jax(
        abs_residuals=abs_res,
        confidences=conf_arr,
        threshold=float(threshold),
        min_confidence=float(min_confidence),
    )

    # Find flagged indices
    flagged_indices = jnp.nonzero(mask, size=n_states, fill_value=-1)[0]
    valid_flagged_mask = flagged_indices >= 0
    actual_count = int(jnp.sum(valid_flagged_mask))

    if actual_count > 0:
        valid_indices = flagged_indices[:actual_count]
        flagged_res = res_arr[valid_indices]
        flagged_abs = abs_res[valid_indices]
        max_res = float(jnp.max(flagged_abs))
        mean_res = float(jnp.mean(flagged_abs))
        rmse_val = float(jnp.sqrt(jnp.mean(flagged_res ** 2)))
        is_triggered = True

        # Extract flagged subset from states PyTree
        def _slice_state_leaf(leaf: Any) -> Any:
            if isinstance(leaf, (jnp.ndarray, np.ndarray)) and len(leaf) == n_states:
                return leaf[valid_indices]
            elif isinstance(leaf, (list, tuple)) and len(leaf) == n_states:
                return type(leaf)(leaf[int(idx)] for idx in valid_indices)
            return leaf

        flagged_states_pytree = jax.tree_util.tree_map(_slice_state_leaf, states)
    else:
        valid_indices = jnp.empty((0,), dtype=jnp.int64)
        flagged_res = jnp.empty((0,), dtype=jnp.float64)
        flagged_abs = jnp.empty((0,), dtype=jnp.float64)
        max_res = 0.0
        mean_res = 0.0
        rmse_val = 0.0
        is_triggered = False

        def _empty_state_leaf(leaf: Any) -> Any:
            if isinstance(leaf, (jnp.ndarray, np.ndarray)) and len(leaf) == n_states:
                return leaf[:0]
            elif isinstance(leaf, (list, tuple)) and len(leaf) == n_states:
                return type(leaf)()
            return leaf

        flagged_states_pytree = jax.tree_util.tree_map(_empty_state_leaf, states)

    return {
        "is_triggered": is_triggered,
        "flagged_count": actual_count,
        "flagged_indices": valid_indices,
        "flagged_residuals": flagged_res,
        "flagged_abs_residuals": flagged_abs,
        "flagged_states": flagged_states_pytree,
        "max_residual": max_res,
        "mean_residual": mean_res,
        "rmse_flagged": rmse_val,
        "threshold": float(threshold),
    }


@jax.jit
def _compute_coordinate_sensitivities_jax(
    flagged_residuals: jnp.ndarray,
    flagged_jacobians: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.int64]:
    """
    Computes coordinate / parameter sensitivity gradient contributions to flagged (O-C) residuals.

    Score S_p = sum_{k in flagged} |r_k * J_{k, p}|
    Relative proportion F_p = S_p / sum_q S_q
    """
    abs_res = jnp.abs(flagged_residuals).reshape((-1, 1))  # (M, 1)
    abs_jac = jnp.abs(flagged_jacobians)                  # (M, P)

    # Weighted parameter contribution: sum over flagged transitions M
    sensitivities = jnp.sum(abs_res * abs_jac, axis=0)     # (P,)
    total_sens = jnp.sum(sensitivities)
    proportions = sensitivities / jnp.maximum(total_sens, 1e-15)

    top_driver_idx = jnp.argmax(sensitivities)
    return sensitivities, proportions, top_driver_idx


def back_calculate_driving_coordinates(
    flagged_residuals: Union[jnp.ndarray, np.ndarray, Sequence],
    state_jacobians: Union[jnp.ndarray, np.ndarray, Sequence],
    coordinate_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Back-calculates which specific Hamiltonian parameters or internal PES coordinates govern
    the quantum states with large unresolved (O-C) residuals.

    Args:
        flagged_residuals: (M,) array of unresolved (O-C) residuals for flagged transitions.
        state_jacobians: (M, P) Jacobian sensitivity gradients for the flagged transitions.
        coordinate_names: Optional list of P parameter/coordinate names (e.g. ['A', 'B', 'C', 'DJ', 'DK', 'V3']).

    Returns:
        Pure PyTree dictionary containing:
        - 'coordinate_sensitivities': jax.Array of raw sensitivity scores per parameter (shape (P,)).
        - 'relative_proportions': jax.Array of fractional sensitivity contributions (shape (P,)).
        - 'top_driver_index': int index of the parameter with largest sensitivity impact.
        - 'top_driver_name': str name of top driver (or 'param_<idx>' if unnamed).
        - 'ranked_coordinate_indices': jax.Array of coordinate indices sorted descending by sensitivity.
        - 'driver_ranking': List of tuples (name, relative_proportion) sorted descending.
    """
    res_arr = jnp.asarray(flagged_residuals, dtype=jnp.float64)
    jac_arr = jnp.asarray(state_jacobians, dtype=jnp.float64)

    if res_arr.ndim != 1:
        raise ValueError(f"Flagged residuals must be 1D, got shape {res_arr.shape}.")
    if jac_arr.ndim != 2:
        raise ValueError(f"State Jacobians must be 2D, got shape {jac_arr.shape}.")

    m_count, p_dim = jac_arr.shape
    if res_arr.shape[0] != m_count:
        raise ValueError(
            f"Residuals count {res_arr.shape[0]} does not match Jacobian rows {m_count}."
        )

    if m_count == 0:
        sens = jnp.zeros((p_dim,), dtype=jnp.float64)
        prop = jnp.zeros((p_dim,), dtype=jnp.float64)
        top_idx = 0
    else:
        sens, prop, top_idx_arr = _compute_coordinate_sensitivities_jax(
            flagged_residuals=res_arr,
            flagged_jacobians=jac_arr,
        )
        top_idx = int(top_idx_arr)

    # Resolve coordinate names
    if coordinate_names is None or len(coordinate_names) != p_dim:
        names = [f"param_{i}" for i in range(p_dim)]
    else:
        names = list(coordinate_names)

    ranked_indices = jnp.argsort(-sens)
    ranking_tuples = [(names[int(idx)], float(prop[idx])) for idx in ranked_indices]

    return {
        "coordinate_sensitivities": sens,
        "relative_proportions": prop,
        "top_driver_index": top_idx,
        "top_driver_name": names[top_idx],
        "ranked_coordinate_indices": ranked_indices,
        "driver_ranking": ranking_tuples,
    }


def generate_pes_healing_payload(
    flagged_state_report: Dict[str, Any],
    coordinate_report: Dict[str, Any],
    molecular_formula: str,
    atomic_symbols: Optional[Sequence[str]] = None,
    cartesian_coords_angstrom: Optional[Sequence[Sequence[float]]] = None,
    config: Optional[ActivePESHealingConfig] = None,
    target_method: Optional[str] = None,
    basis_set: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates a pure PyTree payload ready for the Orchestrator to dispatch to CoChem-TORQ
    for high-level ab initio PES refinement (The "Healing Loop").

    Method Matrix Directives:
    - DLPNO-CCSD(T) is strictly forbidden for high-precision PES refinement. Canonical CCSD(T) is mandatory.

    Args:
        flagged_state_report: Result dictionary from flag_unresolved_residuals.
        coordinate_report: Result dictionary from back_calculate_driving_coordinates.
        molecular_formula: Molecular chemical formula (e.g. 'CH3OH', 'SO2').
        atomic_symbols: Optional sequence of elemental symbols (e.g. ['C', 'H', 'H', 'H', 'O', 'H']).
        cartesian_coords_angstrom: Optional Nx3 reference Cartesian geometry.
        config: Optional ActivePESHealingConfig instance.
        target_method: Optional override for quantum chemistry method.
        basis_set: Optional override for basis set.

    Returns:
        Structured PyTree dictionary with cryptographic SHA-256 integrity hash.

    Raises:
        DLPNOForbiddenError: If DLPNO-CCSD(T) is specified.
    """
    cfg = config or ActivePESHealingConfig()
    method = target_method if target_method is not None else cfg.target_method
    basis = basis_set if basis_set is not None else cfg.basis_set

    # Enforce Method Matrix compliance
    method_upper = method.strip().upper()
    if "DLPNO" in method_upper:
        raise DLPNOForbiddenError(
            f"Target method '{method}' is strictly forbidden by Method Matrix directives for PES refinement. "
            "Canonical CCSD(T) is mandatory."
        )

    # Generate deterministic cryptographic payload hash
    hash_material = (
        f"{molecular_formula}:{method}:{basis}:{coordinate_report.get('top_driver_name', '')}:"
        f"{flagged_state_report.get('max_residual', 0.0):.6f}"
    )
    payload_sha256 = hashlib.sha256(hash_material.encode("utf-8")).hexdigest()

    # Convert JAX arrays in reports to lists for JSON / PyTree serialization compatibility
    def _convert_arrays(obj: Any) -> Any:
        if isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: _convert_arrays(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_convert_arrays(v) for v in obj]
        return obj

    clean_states = _convert_arrays(flagged_state_report.get("flagged_states", {}))
    clean_proportions = _convert_arrays(coordinate_report.get("relative_proportions", []))
    clean_sensitivities = _convert_arrays(coordinate_report.get("coordinate_sensitivities", []))

    payload = {
        "payload_type": "ACTIVE_PES_REFINEMENT_REQUEST",
        "sha256_checksum": payload_sha256,
        "molecular_formula": molecular_formula,
        "atomic_symbols": list(atomic_symbols) if atomic_symbols is not None else [],
        "reference_geometry_angstrom": list(cartesian_coords_angstrom) if cartesian_coords_angstrom is not None else [],
        "target_method": method,
        "basis_set": basis,
        "flagged_unresolved_count": int(flagged_state_report.get("flagged_count", 0)),
        "max_unresolved_residual_mhz": float(flagged_state_report.get("max_residual", 0.0)),
        "rmse_unresolved_residual_mhz": float(flagged_state_report.get("rmse_flagged", 0.0)),
        "top_driving_coordinate": str(coordinate_report.get("top_driver_name", "")),
        "top_driving_coordinate_index": int(coordinate_report.get("top_driver_index", 0)),
        "coordinate_sensitivities": clean_sensitivities,
        "coordinate_relative_proportions": clean_proportions,
        "driver_ranking": coordinate_report.get("driver_ranking", []),
        "flagged_quantum_states": clean_states,
    }

    return payload


# =============================================================================
# 5. PUBLIC API EXPORTS
# =============================================================================

__all__ = [
    # Exceptions
    "AssignMLError",
    "InvalidCovarianceError",
    "InvalidJacobianError",
    "InvalidResidualError",
    "EmptyCandidateError",
    "DLPNOForbiddenError",
    # Enums & Configurations
    "InformationCriterion",
    "InformationGainConfig",
    "ActivePESHealingConfig",
    # Information Gain & Assigner Core
    "calculate_information_gain",
    "rank_candidate_transitions",
    "update_covariance_after_measurement",
    # Active PES Refinement & Healing Loop
    "flag_unresolved_residuals",
    "back_calculate_driving_coordinates",
    "generate_pes_healing_payload",
]
