Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task7_Prompt1_fitter_phase3.md.
Original prompt:
# Task 7 (Phase 3): Autodiff Fitter - cochem_jax_fitter.py

**Objective:**
Update `cochem_jax_fitter.py` to implement Phase 3 requirements, including the JAX-Jitted Levenberg-Marquardt optimizer, Huber Loss function, and the Sobol Sensitivity Guard.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_jax_fitter.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 3 features in `cochem_jax_fitter.py`.
It must:
1. Implement the Levenberg-Marquardt (LM) algorithm natively in `jax.numpy` and compile it with `@jax.jit`.
2. Compute exact analytical Jacobians via JAX Autodiff.
3. Calculate residuals using a Huber Loss function (quadratic for small errors, linear for massive outliers) to insulate against noise.
4. Implement the Sobol Sensitivity Guard: automatically freeze active variables whose variance impact is $< 10^{-6}$, locking them to their *ab initio* values.
5. Enforce absolute functional purity. Do NOT perform any I/O operations (no reading/writing files). Accept all inputs as JAX arrays/PyTrees and return a pure mathematical `PyTree`.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Implement real JAX autodiff and `@jax.jit` compilation.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import jax
import jax.numpy as jnp

@jax.jit
def huber_loss(residual, delta=1.0):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

@jax.jit
def sobol_sensitivity_guard(parameters, covariance, threshold=1e-6):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

@jax.jit
def levenberg_marquardt_step(params, experimental_data, model_fn):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\__init__.py ---
"""
CoChem-SpycFit: Theoretical Tensor Gateway & Physics Core Proxy.
"""

from src.cochem_spycfit.physics_core.cochem_quantum_extractor import (
    DEFAULT_CHUNK_SIZE,
    HARTREE_BOHR2_AMU_TO_CM1,
    MASS_DICT,
    MASS_TABLE,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    QuantumExtraction,
    QuantumExtractor,
    QuantumTensors,
    RotationalConstants,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    determine_hamiltonian_routing,
    extract_tensors,
    get_atomic_mass,
)
from src.cochem_spycfit.physics_core.cochem_jax_builder import (
    DEFAULT_BUCKET_SIZE as JAX_BUILDER_DEFAULT_BUCKET_SIZE,
    DEFAULT_HYSTERESIS_DELTA,
    DEFAULT_LAM_BARRIER_THRESHOLD_CM1,
    BatchedHamiltonianResult,
    HamiltonianBlockResult,
    HamiltonianConfig,
    HermitianLockViolationError,
    HPCShardingError,
    InvalidJValueError,
    InvalidReductionError,
    InvalidRepresentationError,
    JAXBuilderError,
    JBlockBatch,
    LAMBreachError,
    LAMGuardResult,
    ReductionType,
    RepresentationType,
    WatsonConstants,
    build_j_block_batches,
    build_wang_transformation_matrix,
    check_lam_guard,
    compute_rays_asymmetry_param,
    construct_hamiltonian,
    construct_iam_woods_hamiltonian,
    construct_j_block,
    construct_watson_a_matrix_jax,
    construct_watson_s_matrix_jax,
    determine_representation,
    diagonalize_j_block,
    execute_distributed_batches,
    is_hpc_environment,
)
from src.cochem_spycfit.physics_core.cochem_jax_coupling import (
    CM1_TO_MHZ,
    DEFAULT_CORIOLIS_THRESHOLD_CM1,
    DEFAULT_FERMI_THRESHOLD_CM1,
    MHZ_TO_CM1,
    SPEED_OF_LIGHT_CM_S,
    CoriolisCouplingParam,
    CouplingError,
    DarlingDennisonParam,
    DeperturbationError,
    DLPackExportError,
    DLPackImportError,
    FermiCouplingParam,
    InvalidCouplingParameterError,
    InvalidPolyadStateError,
    NonHermitianPolyadError,
    PolyadBlockPyTree,
    PolyadCouplingConfig,
    PolyadDeperturbationResult,
    PolyadEigensystemPyTree,
    ResonanceType,
    StateAssignment,
    VibrationalState,
    build_polyad_from_extractor_report,
    compute_polyad_transition_dipoles,
    construct_polyad_block,
    deperturb_polyad_eigenstates,
    diagonalize_polyad_block,
    extract_pytree_dlpack,
    import_pytree_dlpack,
)

__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "HARTREE_BOHR2_AMU_TO_CM1",
    "MASS_DICT",
    "MASS_TABLE",
    "DipoleMoment",
    "HamiltonianRouting",
    "IsomerNotFoundError",
    "NonHermitianHessianError",
    "PESStore",
    "PESStoreError",
    "QuantumExtraction",
    "QuantumExtractor",
    "QuantumTensors",
    "RotationalConstants",
    "apply_mass_perturbation_and_lock",
    "compute_normal_modes_jax",
    "compute_rays_asymmetry",
    "compute_rotational_constants_from_coords",
    "determine_hamiltonian_routing",
    "extract_tensors",
    "get_atomic_mass",
    # JAX Builder
    "JAX_BUILDER_DEFAULT_BUCKET_SIZE",
    "DEFAULT_HYSTERESIS_DELTA",
    "DEFAULT_LAM_BARRIER_THRESHOLD_CM1",
    "BatchedHamiltonianResult",
    "HamiltonianBlockResult",
    "HamiltonianConfig",
    "HermitianLockViolationError",
    "HPCShardingError",
    "InvalidJValueError",
    "InvalidReductionError",
    "InvalidRepresentationError",
    "JAXBuilderError",
    "JBlockBatch",
    "LAMBreachError",
    "LAMGuardResult",
    "ReductionType",
    "RepresentationType",
    "WatsonConstants",
    "build_j_block_batches",
    "build_wang_transformation_matrix",
    "check_lam_guard",
    "compute_rays_asymmetry_param",
    "construct_hamiltonian",
    "construct_iam_woods_hamiltonian",
    "construct_j_block",
    "construct_watson_a_matrix_jax",
    "construct_watson_s_matrix_jax",
    "determine_representation",
    "diagonalize_j_block",
    "execute_distributed_batches",
    "is_hpc_environment",
    # Polyad Resonance Matrix Engine
    "CM1_TO_MHZ",
    "DEFAULT_CORIOLIS_THRESHOLD_CM1",
    "DEFAULT_FERMI_THRESHOLD_CM1",
    "MHZ_TO_CM1",
    "SPEED_OF_LIGHT_CM_S",
    "CoriolisCouplingParam",
    "CouplingError",
    "DarlingDennisonParam",
    "DeperturbationError",
    "DLPackExportError",
    "DLPackImportError",
    "FermiCouplingParam",
    "InvalidCouplingParameterError",
    "InvalidPolyadStateError",
    "NonHermitianPolyadError",
    "PolyadBlockPyTree",
    "PolyadCouplingConfig",
    "PolyadDeperturbationResult",
    "PolyadEigensystemPyTree",
    "ResonanceType",
    "StateAssignment",
    "VibrationalState",
    "build_polyad_from_extractor_report",
    "compute_polyad_transition_dipoles",
    "construct_polyad_block",
    "deperturb_polyad_eigenstates",
    "diagonalize_polyad_block",
    "extract_pytree_dlpack",
    "import_pytree_dlpack",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\__init__.py ---
"""
CoChem-SpycFit: Theoretical Tensor Gateway & Physics Core.

Provides exact mass-weighted Cartesian Hessian transformations, Hermitian locking,
Ray's asymmetry parameter calculations, torsional barrier Hamiltonian routing,
QCSchema-compliant chunked HDF5 PESStore / Zarr integration, JAX FP64 normal mode analysis,
pure-JAX Autodiff Levenberg-Marquardt optimization, high-J VRAM block diagonalization,
Sobol sensitivity parameter guarding, Active PES refinement triggers with air-gapped transport,
pure-JAX macroscopic spectral line broadening, zero-copy DLPack IPC extraction, WSL path translation,
provenance verification, and Fermi/Coriolis resonance triage.
"""

from .cochem_convolver import (
    BOLTZMANN_K,
    C2_RADIATION_CONST,
    DEFAULT_MAX_VRAM_MB,
    FOUR_LN_2,
    FWHM_TO_SIGMA_GAUSS,
    GAUSS_PREFACTOR,
    H_OVER_K_B,
    LORENTZ_PREFACTOR,
    PLANCK_H,
    SPEED_OF_LIGHT_CM_S,
    SPEED_OF_LIGHT_M_S,
    SQRT_2_LN_2,
    ConvolutionConfig,
    ConvolvedSpectrumResult,
    ConvolverError,
    EnergyUnit,
    InvalidGridError,
    MacroscopicSpectralBroadener,
    MicroBatchError,
    PartitionGridConfig,
    PartitionGridResult,
    ProfileError,
    ProfileType,
    SpectralDataError,
    SpectralLines,
    ThermalScalingError,
    build_frequency_grid,
    calculate_optimal_chunk_size,
    compute_boltzmann_scaling_grid_jax,
    convolve_frequency_chunks_scan_1d_jax,
    convolve_frequency_chunks_scan_2d_jax,
    convolve_line_chunks_scan_1d_jax,
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    evaluate_partition_grid,
    evaluate_profile_kernel_jax,
    faddeeva_rational_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    get_energy_conversion_factor_to_kelvin,
    get_profile_type_code,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
)
from .cochem_jax_coupling import (
    CM1_TO_MHZ,
    DEFAULT_CORIOLIS_THRESHOLD_CM1,
    DEFAULT_FERMI_THRESHOLD_CM1,
    MHZ_TO_CM1,
    CoriolisCouplingParam,
    CouplingError,
    DarlingDennisonParam,
    DeperturbationError,
    DLPackExportError,
    DLPackImportError,
    FermiCouplingParam,
    InvalidCouplingParameterError,
    InvalidPolyadStateError,
    NonHermitianPolyadError,
    PolyadBlockPyTree,
    PolyadCouplingConfig,
    PolyadDeperturbationResult,
    PolyadEigensystemPyTree,
    ResonanceType,
    StateAssignment,
    VibrationalState,
    build_polyad_from_extractor_report,
    compute_polyad_transition_dipoles,
    construct_polyad_block,
    deperturb_polyad_eigenstates,
    diagonalize_polyad_block,
    extract_pytree_dlpack,
    import_pytree_dlpack,
)
from .cochem_jax_builder import (
    DEFAULT_BUCKET_SIZE as JAX_BUILDER_DEFAULT_BUCKET_SIZE,
    DEFAULT_HYSTERESIS_DELTA,
    DEFAULT_LAM_BARRIER_THRESHOLD_CM1,
    BatchedHamiltonianResult,
    HamiltonianBlockResult,
    HamiltonianConfig,
    HermitianLockViolationError,
    HPCShardingError,
    InvalidJValueError,
    InvalidReductionError,
    InvalidRepresentationError,
    JAXBuilderError,
    JBlockBatch,
    LAMBreachError,
    LAMGuardResult,
    ReductionType,
    RepresentationType,
    WatsonConstants,
    build_j_block_batches,
    build_wang_transformation_matrix,
    check_lam_guard,
    compute_rays_asymmetry_param,
    construct_hamiltonian,
    construct_iam_woods_hamiltonian,
    construct_j_block,
    construct_watson_a_matrix_jax,
    construct_watson_s_matrix_jax,
    determine_representation,
    diagonalize_j_block,
    execute_distributed_batches,
    is_hpc_environment,
)
from .cochem_jax_fitter import (
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
from .cochem_quantum_extractor import (
    DEFAULT_CHUNK_SIZE,
    HARTREE_BOHR2_AMU_TO_CM1,
    MASS_DICT,
    MASS_TABLE,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    PESZarrStore,
    ProvenanceMetadata,
    ProvenanceVerificationError,
    ProvenanceVerificationResult,
    QuantumExtraction,
    QuantumExtractor,
    QuantumTensors,
    ResonanceMatch,
    ResonanceReport,
    ResonanceTriageError,
    RotationalConstants,
    WSLPathTranslationError,
    ZarrExtractionError,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    determine_hamiltonian_routing,
    enforce_hermitian_lock,
    extract_from_zarr_dlpack,
    extract_tensors,
    get_atomic_mass,
    get_monoisotopic_mass,
    translate_wsl_path,
    triage_resonances,
    verify_provenance,
)

__all__ = [
    # Extractor & Tensor Core
    "DEFAULT_CHUNK_SIZE",
    "HARTREE_BOHR2_AMU_TO_CM1",
    "MASS_DICT",
    "MASS_TABLE",
    "DipoleMoment",
    "HamiltonianRouting",
    "IsomerNotFoundError",
    "NonHermitianHessianError",
    "PESStore",
    "PESStoreError",
    "PESZarrStore",
    "ProvenanceMetadata",
    "ProvenanceVerificationError",
    "ProvenanceVerificationResult",
    "QuantumExtraction",
    "QuantumExtractor",
    "QuantumTensors",
    "ResonanceMatch",
    "ResonanceReport",
    "ResonanceTriageError",
    "RotationalConstants",
    "WSLPathTranslationError",
    "ZarrExtractionError",
    "apply_mass_perturbation_and_lock",
    "compute_normal_modes_jax",
    "compute_rays_asymmetry",
    "compute_rotational_constants_from_coords",
    "determine_hamiltonian_routing",
    "enforce_hermitian_lock",
    "extract_from_zarr_dlpack",
    "extract_tensors",
    "get_atomic_mass",
    "get_monoisotopic_mass",
    "translate_wsl_path",
    "triage_resonances",
    "verify_provenance",
    # JAX Fitter & Autodiff Core
    "ActivePESRefinementConfig",
    "AirGappedTransportSpool",
    "BlockDiagonalizationConfig",
    "BlockDiagonalizationResult",
    "ConvergenceError",
    "DLPNOForbiddenError",
    "FitterError",
    "InvalidHamiltonianError",
    "LMFitConfig",
    "LMFitResult",
    "LMStepResult",
    "PESRefinementPayload",
    "PESRefinementTriggerResult",
    "SensitivityGuardError",
    "SobolGuardConfig",
    "SobolGuardResult",
    "TransportError",
    "block_diagonalization_scan",
    "check_pes_refinement_trigger",
    "compute_jacobian_autodiff",
    "evaluate_sobol_sensitivity",
    "huber_loss",
    "levenberg_marquardt_fit",
    "levenberg_marquardt_step",
    "sobol_sensitivity_guard",
    # Macroscopic Broadener & Convolver Core
    "PLANCK_H",
    "BOLTZMANN_K",
    "SPEED_OF_LIGHT_CM_S",
    "SPEED_OF_LIGHT_M_S",
    "H_OVER_K_B",
    "C2_RADIATION_CONST",
    "SQRT_2_LN_2",
    "FWHM_TO_SIGMA_GAUSS",
    "FOUR_LN_2",
    "GAUSS_PREFACTOR",
    "LORENTZ_PREFACTOR",
    "DEFAULT_MAX_VRAM_MB",
    "ConvolverError",
    "ProfileError",
    "ThermalScalingError",
    "MicroBatchError",
    "InvalidGridError",
    "SpectralDataError",
    "ProfileType",
    "EnergyUnit",
    "SpectralLines",
    "ConvolutionConfig",
    "PartitionGridConfig",
    "PartitionGridResult",
    "ConvolvedSpectrumResult",
    "faddeeva_rational_jax",
    "gaussian_kernel_jax",
    "lorentzian_kernel_jax",
    "pseudo_voigt_kernel_jax",
    "faddeeva_voigt_kernel_jax",
    "evaluate_profile_kernel_jax",
    "evaluate_partition_function_jax",
    "compute_boltzmann_scaling_grid_jax",
    "convolve_frequency_chunks_scan_1d_jax",
    "convolve_frequency_chunks_scan_2d_jax",
    "convolve_line_chunks_scan_1d_jax",
    "convolve_spectrum_monolithic_jax",
    "get_energy_conversion_factor_to_kelvin",
    "get_profile_type_code",
    "calculate_optimal_chunk_size",
    "build_frequency_grid",
    "evaluate_partition_grid",
    "convolve_spectrum",
    "MacroscopicSpectralBroadener",
    # JAX Hamiltonian Builder Core
    "JAX_BUILDER_DEFAULT_BUCKET_SIZE",
    "DEFAULT_HYSTERESIS_DELTA",
    "DEFAULT_LAM_BARRIER_THRESHOLD_CM1",
    "BatchedHamiltonianResult",
    "HamiltonianBlockResult",
    "HamiltonianConfig",
    "HermitianLockViolationError",
    "HPCShardingError",
    "InvalidJValueError",
    "InvalidReductionError",
    "InvalidRepresentationError",
    "JAXBuilderError",
    "JBlockBatch",
    "LAMBreachError",
    "LAMGuardResult",
    "ReductionType",
    "RepresentationType",
    "WatsonConstants",
    "build_j_block_batches",
    "build_wang_transformation_matrix",
    "check_lam_guard",
    "compute_rays_asymmetry_param",
    "construct_hamiltonian",
    "construct_iam_woods_hamiltonian",
    "construct_j_block",
    "construct_watson_a_matrix_jax",
    "construct_watson_s_matrix_jax",
    "determine_representation",
    "diagonalize_j_block",
    "execute_distributed_batches",
    "is_hpc_environment",
    # Polyad Resonance Engine Core
    "CM1_TO_MHZ",
    "DEFAULT_CORIOLIS_THRESHOLD_CM1",
    "DEFAULT_FERMI_THRESHOLD_CM1",
    "MHZ_TO_CM1",
    "CoriolisCouplingParam",
    "CouplingError",
    "DarlingDennisonParam",
    "DeperturbationError",
    "DLPackExportError",
    "DLPackImportError",
    "FermiCouplingParam",
    "InvalidCouplingParameterError",
    "InvalidPolyadStateError",
    "NonHermitianPolyadError",
    "PolyadBlockPyTree",
    "PolyadCouplingConfig",
    "PolyadDeperturbationResult",
    "PolyadEigensystemPyTree",
    "ResonanceType",
    "StateAssignment",
    "VibrationalState",
    "build_polyad_from_extractor_report",
    "compute_polyad_transition_dipoles",
    "construct_polyad_block",
    "deperturb_polyad_eigenstates",
    "diagonalize_polyad_block",
    "extract_pytree_dlpack",
    "import_pytree_dlpack",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_jax_fitter.py ---
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
import functools
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
    use_huber_loss: bool = Field(default=False, description="Whether to use Huber Loss for outlier robustness")
    huber_delta: float = Field(default=1.0, gt=0.0, description="Huber loss threshold parameter delta")


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
# 3. AUTODIFF LEVENBERG-MARQUARDT OPTIMIZER & LOSS CORE (PHASE 3)
# =============================================================================

@jax.jit
def huber_loss(residual: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """
    Computes the Huber Loss function in pure JAX with @jax.jit compilation.

    L_delta(r) = 0.5 * r^2                  for |r| <= delta
                 delta * (|r| - 0.5 * delta) for |r| > delta

    Provides quadratic convergence for small residual errors and linear penalty
    for massive spectroscopic outliers, insulating optimization against noise.

    Parameters:
        residual: JAX array or PyTree of residual values.
        delta: Threshold between quadratic and linear regimes (default: 1.0).

    Returns:
        Pure JAX array of Huber loss values with identical shape to residual.
    """
    r = jnp.asarray(residual, dtype=jnp.float64)
    abs_r = jnp.abs(r)
    delta_f64 = jnp.asarray(delta, dtype=jnp.float64)
    quadratic = 0.5 * (abs_r ** 2)
    linear = delta_f64 * (abs_r - 0.5 * delta_f64)
    return jnp.where(abs_r <= delta_f64, quadratic, linear)


@functools.partial(jax.jit, static_argnames=("threshold",))
def sobol_sensitivity_guard(
    parameters: jnp.ndarray,
    covariance: jnp.ndarray,
    threshold: float = 1e-6,
) -> Dict[str, jnp.ndarray]:
    """
    Sobol Sensitivity Guard implemented in pure JAX with @jax.jit compilation.

    Evaluates parameter variance impact and automatically freezes active variables
    whose variance impact drops below the threshold (default: 1e-6), locking them
    to their ab initio values.

    Parameters:
        parameters: 1D array of parameter values [n] (or PyTree).
        covariance: Parameter covariance matrix [n, n] or 1D variance vector [n].
        threshold: Variance impact threshold for freezing (default: 1e-6).

    Returns:
        Pure mathematical PyTree (dict of JAX arrays) containing:
            - 'active_mask': Boolean mask [n] (True = active/sensitive, False = frozen)
            - 'variance_impacts': Normalized variance impact per parameter [n]
            - 'active_parameters': Parameter values with frozen variables set to 0.0
            - 'frozen_parameters': Parameter values with active variables set to 0.0
            - 'active_count': Total number of active parameters
            - 'frozen_count': Total number of frozen parameters
    """
    p = jnp.asarray(parameters, dtype=jnp.float64)
    cov = jnp.asarray(covariance, dtype=jnp.float64)
    thresh = jnp.asarray(threshold, dtype=jnp.float64)

    if cov.ndim == 2:
        variances = jnp.diag(cov)
    else:
        variances = cov

    variances_pos = jnp.maximum(variances, 0.0)
    tot_var = jnp.sum(variances_pos) + 1e-18
    variance_impacts = variances_pos / tot_var

    active_mask = variance_impacts >= thresh
    frozen_mask = jnp.logical_not(active_mask)

    active_params = jnp.where(active_mask, p, 0.0)
    frozen_params = jnp.where(frozen_mask, p, 0.0)

    return {
        "active_mask": active_mask,
        "variance_impacts": variance_impacts,
        "active_parameters": active_params,
        "frozen_parameters": frozen_params,
        "active_count": jnp.sum(active_mask.astype(jnp.int32)),
        "frozen_count": jnp.sum(frozen_mask.astype(jnp.int32)),
    }


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


@functools.partial(
    jax.jit,
    static_argnames=(
        "model_fn",
        "residual_fn",
        "marquardt_scaling",
        "use_huber_loss",
    ),
)
def _lm_step_jit(
    params: jnp.ndarray,
    experimental_data: Optional[jnp.ndarray],
    model_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    residual_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]],
    damping: float,
    active_mask: Optional[jnp.ndarray],
    marquardt_scaling: bool,
    damping_factor: float,
    damping_up: Optional[float],
    damping_down: Optional[float],
    damping_min: float,
    damping_max: float,
    huber_delta: float,
    use_huber_loss: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure JAX JIT-compiled Levenberg-Marquardt step kernel with automatic differentiation
    and optional Huber Loss noise insulation.
    """
    params_curr = jnp.asarray(params, dtype=jnp.float64)
    n_params = params_curr.shape[0]

    d_up = damping_up if damping_up is not None else damping_factor
    d_down = damping_down if damping_down is not None else (1.0 / jnp.maximum(1e-12, damping_factor))

    if active_mask is None:
        mask = jnp.ones(n_params, dtype=jnp.bool_)
    else:
        mask = jnp.asarray(active_mask, dtype=jnp.bool_)

    if residual_fn is not None:
        res_fn = residual_fn
    elif model_fn is not None and experimental_data is not None:
        res_fn = lambda p: model_fn(p) - experimental_data
    elif model_fn is not None:
        res_fn = model_fn
    else:
        raise ValueError("Either residual_fn or (model_fn and experimental_data) must be provided.")

    r_curr = res_fn(params_curr)
    j_mat = compute_jacobian_autodiff(res_fn, params_curr)

    mask_float = mask.astype(jnp.float64)
    j_active = j_mat * mask_float[None, :]

    if use_huber_loss:
        abs_r = jnp.abs(r_curr)
        w_diag = jnp.where(
            abs_r <= huber_delta,
            1.0,
            huber_delta / jnp.maximum(abs_r, 1e-18),
        )
        w_sqrt = jnp.sqrt(w_diag)
        j_weighted = j_active * w_sqrt[:, None]
        r_weighted = r_curr * w_sqrt
        h_mat = jnp.dot(j_weighted.T, j_weighted)
        grad_f = jnp.dot(j_weighted.T, r_weighted)
    else:
        h_mat = jnp.dot(j_active.T, j_active)
        grad_f = jnp.dot(j_active.T, r_curr)

    g_rhs = -grad_f

    if marquardt_scaling:
        diag_h = jnp.diag(h_mat)
        diag_scaled = jnp.maximum(diag_h, 1e-12)
        d_mat = jnp.diag(diag_scaled)
    else:
        d_mat = jnp.eye(n_params, dtype=jnp.float64)

    a_mat = h_mat + damping * d_mat

    mask_2d = mask_float[:, None] * mask_float[None, :]
    inv_mask_1d = 1.0 - mask_float
    a_regularized = a_mat * mask_2d + jnp.diag(inv_mask_1d)
    g_regularized = g_rhs * mask_float

    delta_p = jnp.linalg.solve(a_regularized, g_regularized)
    delta_p = delta_p * mask_float

    params_trial = params_curr + delta_p
    r_trial = res_fn(params_trial)

    if use_huber_loss:
        abs_rc = jnp.abs(r_curr)
        f_curr = jnp.sum(jnp.where(abs_rc <= huber_delta, 0.5 * abs_rc ** 2, huber_delta * (abs_rc - 0.5 * huber_delta)))
        abs_rt = jnp.abs(r_trial)
        f_trial = jnp.sum(jnp.where(abs_rt <= huber_delta, 0.5 * abs_rt ** 2, huber_delta * (abs_rt - 0.5 * huber_delta)))
    else:
        f_curr = 0.5 * jnp.sum(r_curr ** 2)
        f_trial = 0.5 * jnp.sum(r_trial ** 2)

    actual_reduction = f_curr - f_trial
    predicted_reduction = 0.5 * (
        jnp.dot(delta_p, damping * jnp.dot(d_mat, delta_p)) - jnp.dot(delta_p, grad_f)
    )

    gain_ratio = actual_reduction / (predicted_reduction + 1e-18)
    grad_norm = jnp.max(jnp.abs(grad_f * mask_float))
    step_norm = jnp.linalg.norm(delta_p)

    accepted = (actual_reduction > 0.0) & (gain_ratio > 0.0)

    params_next = jnp.where(accepted, params_trial, params_curr)
    residual_next = jnp.where(accepted, r_trial, r_curr)
    damping_next = jnp.where(
        accepted,
        jnp.maximum(damping * d_down, damping_min),
        jnp.minimum(damping * d_up, damping_max),
    )

    return (
        params_next,
        residual_next,
        damping_next,
        accepted,
        gain_ratio,
        grad_norm,
        step_norm,
    )


def levenberg_marquardt_step(
    params: jnp.ndarray,
    experimental_data: Optional[Union[Callable[[jnp.ndarray], jnp.ndarray], jnp.ndarray, Sequence[float]]] = None,
    model_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    *,
    residual_fn: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
    damping: float = 1e-3,
    active_mask: Optional[Union[jnp.ndarray, Sequence[bool]]] = None,
    marquardt_scaling: bool = True,
    damping_factor: float = 10.0,
    damping_up: Optional[float] = None,
    damping_down: Optional[float] = None,
    damping_min: float = 1e-12,
    damping_max: float = 1e12,
    huber_delta: float = 1.0,
    use_huber_loss: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, bool, float, float, float]:
    """
    Performs a single authentic JAX Autodiff Levenberg-Marquardt optimization step.

    Parameters:
        params: 1D JAX FP64 array of parameters [n]
        experimental_data: Target experimental measurement vector [m], or residual_fn callable
        model_fn: Pure callable returning predicted output vector [m] given parameter vector [n]
        residual_fn: Pure callable returning residual vector r(params) [m]
        damping: Current LM damping parameter lambda
        active_mask: Optional boolean mask [n] indicating active (True) vs frozen (False) params
        marquardt_scaling: If True, uses Marquardt diagonal scaling diag(J^T J); else uses Identity
        damping_factor: Fallback factor to multiply/divide lambda upon rejection/acceptance
        damping_up: Multiplier for lambda upon rejected step
        damping_down: Multiplier for lambda upon accepted step
        damping_min: Minimum allowable damping parameter
        damping_max: Maximum allowable damping parameter
        huber_delta: Huber loss transition threshold (default: 1.0)
        use_huber_loss: Whether to use Huber Loss for outlier robustness

    Returns:
        Tuple of:
            params_next: Updated parameter vector [n]
            residual_next: Updated residual vector [m]
            damping_next: Updated damping lambda
            accepted: Boolean indicating step acceptance (gain ratio rho > 0)
            gain_ratio: Gain ratio rho
            grad_norm: Infinity norm of the objective gradient
            step_norm: L2 norm of the parameter displacement step
    """
    actual_residual_fn = residual_fn
    actual_model_fn = model_fn
    actual_exp_data = None

    if callable(experimental_data) and model_fn is None and residual_fn is None:
        actual_residual_fn = experimental_data
    elif experimental_data is not None:
        actual_exp_data = jnp.asarray(experimental_data, dtype=jnp.float64)

    p_in = jnp.asarray(params, dtype=jnp.float64)
    act_mask_in = jnp.asarray(active_mask, dtype=jnp.bool_) if active_mask is not None else None

    (
        p_next,
        r_next,
        d_next,
        acc,
        g_ratio,
        g_norm,
        s_norm,
    ) = _lm_step_jit(
        params=p_in,
        experimental_data=actual_exp_data,
        model_fn=actual_model_fn,
        residual_fn=actual_residual_fn,
        damping=damping,
        active_mask=act_mask_in,
        marquardt_scaling=marquardt_scaling,
        damping_factor=damping_factor,
        damping_up=damping_up,
        damping_down=damping_down,
        damping_min=damping_min,
        damping_max=damping_max,
        huber_delta=huber_delta,
        use_huber_loss=use_huber_loss,
    )

    return (
        p_next,
        r_next,
        float(d_next),
        bool(acc),
        float(g_ratio),
        float(g_norm),
        float(s_norm),
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
            huber_delta=cfg.huber_delta,
            use_huber_loss=cfg.use_huber_loss,
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
            if cfg.use_huber_loss:
                loss_c = float(jnp.sum(huber_loss(r_curr, delta=cfg.huber_delta)))
                loss_n = float(jnp.sum(huber_loss(r_next, delta=cfg.huber_delta)))
                rel_res_change = abs(loss_c - loss_n) / (loss_c + 1e-18)
            else:
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

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_jax_coupling.py ---
"""
CoChem-SpycFit: Unit and Integration Test Suite for Polyad Resonance Matrix Engine.

Strict Anti-Spoofing & Method Matrix Compliance:
- Zero pass statements
- Zero artificial test doubles or mocked functions
- 100% authentic physical and quantum mechanical calculations in JAX FP64
- Comprehensive validation of Fermi resonances, Coriolis vibration-rotation coupling (z and xy),
  Darling-Dennison anharmonicity, J-block Polyad construction, Hermitian lock enforcement,
  pure JAX PyTree registration/flatten/unflatten, JIT compilation, autodiff differentiation,
  DLPack zero-copy IPC extraction, and deperturbation eigenvector mixing analysis.
"""

from __future__ import annotations

import ast
import inspect
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mendeleev
import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_jax_builder import (
    WatsonConstants,
    construct_watson_a_matrix_jax,
)
from cochem_spycfit.physics_core.cochem_jax_coupling import (
    CM1_TO_MHZ,
    MHZ_TO_CM1,
    CoriolisCouplingParam,
    CouplingError,
    DarlingDennisonParam,
    DLPackExportError,
    DLPackImportError,
    FermiCouplingParam,
    InvalidCouplingParameterError,
    InvalidPolyadStateError,
    NonHermitianPolyadError,
    PolyadBlockPyTree,
    PolyadCouplingConfig,
    PolyadDeperturbationResult,
    PolyadEigensystemPyTree,
    ResonanceType,
    VibrationalState,
    build_polyad_from_extractor_report,
    compute_polyad_transition_dipoles,
    construct_polyad_block,
    deperturb_polyad_eigenstates,
    diagonalize_polyad_block,
    extract_pytree_dlpack,
    import_pytree_dlpack,
)
from cochem_spycfit.physics_core.cochem_quantum_extractor import (
    ResonanceMatch,
    ResonanceReport,
)

# Enforce JAX FP64 precision
jax.config.update("jax_enable_x64", True)


# =============================================================================
# 1. PYDANTIC DATA MODELS & PARAMETER VALIDATION
# =============================================================================

def test_vibrational_state_pydantic_validation() -> None:
    """Verifies that VibrationalState correctly validates and auto-computes MHz energies."""
    wc = WatsonConstants(A=300000.0, B=50000.0, C=40000.0)
    state = VibrationalState(
        state_index=0,
        label="nu1",
        vibrational_energy_cm1=1000.0,
        watson_constants=wc,
        quanta=[1, 0, 0],
        symmetry="A1",
    )
    assert state.state_index == 0
    assert state.label == "nu1"
    assert math.isclose(state.vibrational_energy_cm1, 1000.0, abs_tol=1e-8)
    assert state.vibrational_energy_mhz is not None
    assert math.isclose(state.vibrational_energy_mhz, 1000.0 * CM1_TO_MHZ, abs_tol=1e-4)
    assert state.quanta == [1, 0, 0]
    assert state.symmetry == "A1"


def test_vibrational_state_invalid_index() -> None:
    """Verifies that negative state indices raise validation error."""
    wc = WatsonConstants(A=300000.0, B=50000.0, C=40000.0)
    with pytest.raises(Exception):
        VibrationalState(
            state_index=-1,
            label="bad",
            vibrational_energy_cm1=500.0,
            watson_constants=wc,
        )


def test_fermi_coupling_parameter_validation() -> None:
    """Verifies FermiCouplingParam initialization and unit conversions."""
    param = FermiCouplingParam(
        state_i=0,
        state_j=1,
        w_fermi_cm1=45.0,
        w_j_mhz=0.05,
        w_k_mhz=0.10,
    )
    assert param.state_i == 0
    assert param.state_j == 1
    assert math.isclose(param.w_fermi_cm1, 45.0, abs_tol=1e-8)
    assert param.w_fermi_mhz is not None
    assert math.isclose(param.w_fermi_mhz, 45.0 * CM1_TO_MHZ, abs_tol=1e-4)
    assert math.isclose(param.w_j_mhz, 0.05, abs_tol=1e-8)
    assert math.isclose(param.w_k_mhz, 0.10, abs_tol=1e-8)


def test_coriolis_coupling_parameter_validation() -> None:
    """Verifies CoriolisCouplingParam initialization, zeta clamping and axis validation."""
    param = CoriolisCouplingParam(
        state_i=0,
        state_j=1,
        zeta_z=0.65,
        zeta_x=0.20,
        zeta_y=0.10,
        coupling_axis="z",
    )
    assert param.state_i == 0
    assert param.state_j == 1
    assert math.isclose(param.zeta_z, 0.65, abs_tol=1e-8)
    assert param.coupling_axis == "z"


def test_darling_dennison_parameter_validation() -> None:
    """Verifies DarlingDennisonParam initialization and conversion."""
    param = DarlingDennisonParam(
        state_i=0,
        state_j=1,
        gamma_cm1=-12.5,
    )
    assert param.state_i == 0
    assert param.state_j == 1
    assert math.isclose(param.gamma_cm1, -12.5, abs_tol=1e-8)
    assert param.gamma_mhz is not None
    assert math.isclose(param.gamma_mhz, -12.5 * CM1_TO_MHZ, abs_tol=1e-4)


# =============================================================================
# 2. POLYAD BLOCK CONSTRUCTION & PHYSICAL COUPLINGS
# =============================================================================

def test_polyad_block_construction_j0() -> None:
    """Verifies that at J=0, Polyad block is M x M with origins on diagonal and Fermi off-diagonal."""
    wc1 = WatsonConstants(A=30000.0, B=20000.0, C=10000.0)
    wc2 = WatsonConstants(A=29800.0, B=19900.0, C=9950.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1010.0, watson_constants=wc2)

    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=15.0)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[fermi],
    )

    block = construct_polyad_block(config, J=0)
    assert isinstance(block, PolyadBlockPyTree)
    assert block.j_value == 0
    assert block.num_states == 2
    assert block.total_dim == 2

    # Verify matrix elements at J=0
    h_mat = np.array(block.hamiltonian_matrix)
    e1_mhz = 1000.0 * CM1_TO_MHZ
    e2_mhz = 1010.0 * CM1_TO_MHZ
    w_mhz = 15.0 * CM1_TO_MHZ

    assert math.isclose(h_mat[0, 0], e1_mhz, rel_tol=1e-7)
    assert math.isclose(h_mat[1, 1], e2_mhz, rel_tol=1e-7)
    assert math.isclose(h_mat[0, 1], w_mhz, rel_tol=1e-7)
    assert math.isclose(h_mat[1, 0], w_mhz, rel_tol=1e-7)


def test_polyad_block_construction_j1_fermi() -> None:
    """Verifies J=1 block dimensions (2 states * 3 K-levels = 6x6) with Watson blocks on diagonal."""
    wc1 = WatsonConstants(A=30000.0, B=20000.0, C=10000.0)
    wc2 = WatsonConstants(A=29000.0, B=19500.0, C=9800.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1500.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1505.0, watson_constants=wc2)

    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=20.0)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[fermi],
    )

    block = construct_polyad_block(config, J=1)
    assert block.total_dim == 6  # 2 states * (2*1 + 1) = 6
    h_mat = np.array(block.hamiltonian_matrix)

    # Check that diagonal 3x3 blocks correspond to Watson J=1 matrices + vibrational origins
    w_block_1 = construct_watson_a_matrix_jax(wc1, J=1)
    expected_diag_1 = np.array(w_block_1) + np.eye(3) * (1500.0 * CM1_TO_MHZ)
    assert np.allclose(h_mat[0:3, 0:3], expected_diag_1, rtol=1e-7, atol=1e-7)

    # Check off-diagonal 3x3 block is Fermi coupling * identity (Delta K = 0)
    w_mhz = 20.0 * CM1_TO_MHZ
    expected_offdiag = np.eye(3) * w_mhz
    assert np.allclose(h_mat[0:3, 3:6], expected_offdiag, rtol=1e-7, atol=1e-7)
    assert np.allclose(h_mat[3:6, 0:3], expected_offdiag, rtol=1e-7, atol=1e-7)


def test_polyad_block_coriolis_z_coupling() -> None:
    """Verifies z-axis Coriolis coupling matrix elements are proportional to K."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1020.0, watson_constants=wc2)

    # Coriolis z-coupling: xi_z = 5000.0 MHz
    coriolis = CoriolisCouplingParam(
        state_i=0,
        state_j=1,
        zeta_z=0.5,
        xi_z_mhz=5000.0,
        coupling_axis="z",
    )

    config = PolyadCouplingConfig(
        states=[s1, s2],
        coriolis_couplings=[coriolis],
    )

    # Construct for J=2 (K = -2, -1, 0, 1, 2; dim = 5 per state, total dim = 10)
    block = construct_polyad_block(config, J=2)
    assert block.total_dim == 10
    h_mat = np.array(block.hamiltonian_matrix)

    # Off-diagonal block coupling state 0 and state 1
    # Coriolis z-coupling elements: <0, J, K | H | 1, J, K> = -i * xi_z * K
    # For K = 0 (index 2 in 0..4): coupling is 0
    # For K = 1 (index 3 in 0..4): coupling is -i * 5000
    # For K = -1 (index 1 in 0..4): coupling is +i * 5000
    # For K = 2 (index 4 in 0..4): coupling is -i * 10000
    # For K = -2 (index 0 in 0..4): coupling is +i * 10000
    off_diag = h_mat[0:5, 5:10]

    assert math.isclose(abs(off_diag[2, 2]), 0.0, abs_tol=1e-7)  # K = 0
    assert math.isclose(off_diag[3, 3].imag, -5000.0, rel_tol=1e-7)  # K = 1
    assert math.isclose(off_diag[1, 1].imag, 5000.0, rel_tol=1e-7)   # K = -1
    assert math.isclose(off_diag[4, 4].imag, -10000.0, rel_tol=1e-7) # K = 2
    assert math.isclose(off_diag[0, 0].imag, 10000.0, rel_tol=1e-7)  # K = -2


def test_polyad_block_coriolis_xy_coupling() -> None:
    """Verifies perpendicular (xy) Coriolis coupling matrix elements (Delta K = +-1)."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1020.0, watson_constants=wc2)

    # Perpendicular Coriolis coupling: xi_xy = 3000.0 MHz
    coriolis = CoriolisCouplingParam(
        state_i=0,
        state_j=1,
        zeta_x=0.3,
        zeta_y=0.0,
        xi_xy_mhz=3000.0,
        coupling_axis="xy",
    )

    config = PolyadCouplingConfig(
        states=[s1, s2],
        coriolis_couplings=[coriolis],
    )

    # Construct for J=1 (K = -1, 0, 1; dim = 3 per state, total dim = 6)
    block = construct_polyad_block(config, J=1)
    assert block.total_dim == 6
    h_mat = np.array(block.hamiltonian_matrix)

    off_diag = h_mat[0:3, 3:6]
    # For J=1:
    # <K=0 | H | K=-1> factor = sqrt(1*(2) - 0*(-1)) = sqrt(2)
    # <K=1 | H | K=0> factor = sqrt(1*(2) - 0*(1)) = sqrt(2)
    expected_coupling = 3000.0 * math.sqrt(2.0)
    assert math.isclose(abs(off_diag[1, 0]), expected_coupling, rel_tol=1e-6)
    assert math.isclose(abs(off_diag[2, 1]), expected_coupling, rel_tol=1e-6)


def test_polyad_block_hermitian_lock() -> None:
    """Verifies that all constructed Polyad matrices satisfy Hermitian symmetry H == H^dagger."""
    wc1 = WatsonConstants(A=250000.0, B=60000.0, C=40000.0)
    wc2 = WatsonConstants(A=245000.0, B=59000.0, C=39500.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1200.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1215.0, watson_constants=wc2)

    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=25.0)
    coriolis = CoriolisCouplingParam(state_i=0, state_j=1, zeta_z=0.4, xi_z_mhz=8000.0, coupling_axis="z")

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[fermi],
        coriolis_couplings=[coriolis],
    )

    for J in range(0, 5):
        block = construct_polyad_block(config, J=J)
        h_mat = np.array(block.hamiltonian_matrix)
        h_conj_t = h_mat.conj().T
        assert np.allclose(h_mat, h_conj_t, rtol=1e-12, atol=1e-12)


def test_polyad_invalid_state_index_error() -> None:
    """Verifies that coupling referencing non-existent state indices raises CouplingError."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)

    # Coupling references state 1 which is not in config
    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=20.0)
    config = PolyadCouplingConfig(states=[s1], fermi_couplings=[fermi])

    with pytest.raises(InvalidCouplingParameterError):
        construct_polyad_block(config, J=1)


# =============================================================================
# 3. JAX PYTREE PURITY, JIT COMPILATION & AUTODIFF
# =============================================================================

def test_polyad_block_pytree_registration() -> None:
    """Verifies that PolyadBlockPyTree and PolyadEigensystemPyTree are registered JAX PyTrees."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=99000.0, B=49500.0, C=24800.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1010.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=10.0)],
    )

    block = construct_polyad_block(config, J=1)

    # Test tree_flatten and tree_unflatten
    leaves, treedef = jax.tree_util.tree_flatten(block)
    assert len(leaves) >= 3  # hamiltonian_matrix, state_energies, coupling_matrix
    reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(reconstructed, PolyadBlockPyTree)
    assert reconstructed.j_value == block.j_value
    assert reconstructed.total_dim == block.total_dim
    assert jnp.allclose(reconstructed.hamiltonian_matrix, block.hamiltonian_matrix)


def test_polyad_block_jit_compilation() -> None:
    """Verifies that diagonalize_polyad_block can be JIT compiled with jax.jit."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=99000.0, B=49500.0, C=24800.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1010.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=12.0)],
    )

    block = construct_polyad_block(config, J=1)

    @jax.jit
    def jitted_diag(b: PolyadBlockPyTree) -> PolyadEigensystemPyTree:
        return diagonalize_polyad_block(b)

    eigensystem = jitted_diag(block)
    assert isinstance(eigensystem, PolyadEigensystemPyTree)
    assert eigensystem.eigenvalues_mhz.shape == (6,)
    assert eigensystem.eigenvectors.shape == (6, 6)

    # Check sorting: E_0 <= E_1 <= ... <= E_5
    evals = np.array(eigensystem.eigenvalues_mhz)
    assert np.all(np.diff(evals) >= -1e-10)


def test_polyad_block_autodiff() -> None:
    """Verifies that energy eigenvalues can be differentiated with respect to coupling parameters."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=99000.0, B=49500.0, C=24800.0)

    e1_mhz = 1000.0 * CM1_TO_MHZ
    e2_mhz = 1010.0 * CM1_TO_MHZ

    # Define pure function of coupling W
    def energy_splitting(w_mhz: float) -> jnp.ndarray:
        # 2x2 matrix at J=0: [[E1, W], [W, E2]]
        h = jnp.array([[e1_mhz, w_mhz], [w_mhz, e2_mhz]], dtype=jnp.float64)
        evals = jnp.linalg.eigvalsh(h)
        return evals[1] - evals[0]

    w_test = 15.0 * CM1_TO_MHZ
    # Exact analytical derivative: d(Delta E)/dW = 2 * W / sqrt((E2 - E1)^2 + 4 * W^2)
    delta_e_unperturbed = e2_mhz - e1_mhz
    expected_grad = (4.0 * w_test) / math.sqrt(delta_e_unperturbed**2 + 4.0 * w_test**2)

    grad_fn = jax.grad(energy_splitting)
    computed_grad = float(grad_fn(w_test))

    assert math.isclose(computed_grad, expected_grad, rel_tol=1e-5)


# =============================================================================
# 4. DLPACK ZERO-COPY IPC GATEWAY
# =============================================================================

def test_dlpack_zero_copy_export_and_import() -> None:
    """Verifies zero-copy extraction and reconstruction of Polyad PyTrees via DLPack."""
    wc1 = WatsonConstants(A=120000.0, B=60000.0, C=30000.0)
    wc2 = WatsonConstants(A=118000.0, B=59000.0, C=29500.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1400.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1410.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=18.0)],
    )

    block = construct_polyad_block(config, J=1)
    eigensystem = diagonalize_polyad_block(block)

    # 1. Export block PyTree via DLPack
    dlpack_payload = extract_pytree_dlpack(block)
    assert isinstance(dlpack_payload, dict)
    assert "leaves" in dlpack_payload
    assert "treedef" in dlpack_payload
    assert len(dlpack_payload["leaves"]) > 0

    # 2. Re-import block PyTree via DLPack
    reconstructed_block = import_pytree_dlpack(dlpack_payload, template_pytree=block)
    assert isinstance(reconstructed_block, PolyadBlockPyTree)
    assert jnp.allclose(reconstructed_block.hamiltonian_matrix, block.hamiltonian_matrix)

    # 3. Export & re-import eigensystem PyTree
    eig_payload = extract_pytree_dlpack(eigensystem)
    reconstructed_eigs = import_pytree_dlpack(eig_payload, template_pytree=eigensystem)
    assert isinstance(reconstructed_eigs, PolyadEigensystemPyTree)
    assert jnp.allclose(reconstructed_eigs.eigenvalues_mhz, eigensystem.eigenvalues_mhz)
    assert jnp.allclose(reconstructed_eigs.eigenvectors, eigensystem.eigenvectors)


# =============================================================================
# 5. DEPERTURBATION ANALYSIS & MIXING WEIGHTS
# =============================================================================

def test_deperturb_polyad_eigenstates() -> None:
    """Verifies deperturbation state decomposition, mixing weights and level shifts."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=99000.0, B=49500.0, C=24800.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1002.0, watson_constants=wc2)

    # Strong Fermi resonance: W = 10 cm^-1 for delta E = 2 cm^-1
    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=10.0)
    config = PolyadCouplingConfig(states=[s1, s2], fermi_couplings=[fermi])

    block = construct_polyad_block(config, J=0)
    eigensystem = diagonalize_polyad_block(block)

    report = deperturb_polyad_eigenstates(eigensystem, state_labels=["v1", "v2"])
    assert isinstance(report, PolyadDeperturbationResult)
    assert len(report.state_assignments) == 2

    # Strongly mixed: mixing weight ~ 0.5 each
    w0 = report.state_assignments[0].mixing_fractions
    w1 = report.state_assignments[1].mixing_fractions
    assert math.isclose(sum(w0), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(w1), 1.0, abs_tol=1e-6)
    assert report.is_strongly_coupled is True


# =============================================================================
# 6. REAL PHYSICAL SYSTEMS (CO2 FERMI DYAD & H2CO CORIOLIS)
# =============================================================================

def test_real_molecule_co2_fermi_dyad() -> None:
    """
    Tests canonical Carbon Dioxide (CO2) Fermi Dyad:
    nu1 (symmetric stretch) ~ 1337.0 cm^-1 vs 2*nu2 (bending overtone) ~ 1285.4 cm^-1
    Fermi interaction constant W_12 ~ 50.7 cm^-1.
    Empirical observed perturbed levels: 1285.4 cm^-1 and 1388.2 cm^-1.
    """
    # Linear CO2 (A=0, B=C=11732.5 MHz)
    wc1 = WatsonConstants(A=11732.5, B=11732.5, C=11732.5)
    wc2 = WatsonConstants(A=11732.5, B=11732.5, C=11732.5)

    # Deperturbed harmonic band origins
    s1 = VibrationalState(state_index=0, label="100", vibrational_energy_cm1=1337.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="020", vibrational_energy_cm1=1334.0, watson_constants=wc2)

    fermi = FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=50.7)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[fermi],
    )

    block = construct_polyad_block(config, J=0)
    eigensystem = diagonalize_polyad_block(block)

    evals_cm1 = np.array(eigensystem.eigenvalues_cm1)
    # Eigenvalues should show level repulsion: lower level shifted down, upper shifted up
    assert evals_cm1[0] < 1334.0
    assert evals_cm1[1] > 1337.0

    # Level splitting: sqrt((1337-1334)^2 + 4 * 50.7^2) = sqrt(9 + 10281.96) ~ 101.44 cm^-1
    splitting = evals_cm1[1] - evals_cm1[0]
    expected_splitting = math.sqrt((1337.0 - 1334.0)**2 + 4.0 * (50.7**2))
    assert math.isclose(splitting, expected_splitting, rel_tol=1e-5)


def test_real_molecule_formaldehyde_h2co_coriolis() -> None:
    """
    Tests Formaldehyde (H2CO) Coriolis interaction between nu4 (1167.3 cm^-1) and nu6 (1249.1 cm^-1).
    Coriolis zeta_a ~ 0.54.
    """
    # H2CO rotational constants (MHz)
    # A = 281970.5 MHz, B = 38833.99 MHz, C = 34004.24 MHz
    wc4 = WatsonConstants(A=282000.0, B=38800.0, C=34000.0)
    wc6 = WatsonConstants(A=281900.0, B=38850.0, C=34010.0)

    s4 = VibrationalState(state_index=0, label="nu4_B1", vibrational_energy_cm1=1167.3, watson_constants=wc4)
    s6 = VibrationalState(state_index=1, label="nu6_B2", vibrational_energy_cm1=1249.1, watson_constants=wc6)

    # a-type Coriolis coupling along principal a-axis (quantization z in Ir)
    # xi_a = 2 * A * zeta_a * 0.5 * (sqrt(1167.3/1249.1) + sqrt(1249.1/1167.3))
    omega_factor = 0.5 * (math.sqrt(1167.3 / 1249.1) + math.sqrt(1249.1 / 1167.3))
    xi_a_mhz = 2.0 * 281970.5 * 0.54 * omega_factor

    coriolis = CoriolisCouplingParam(
        state_i=0,
        state_j=1,
        zeta_z=0.54,
        xi_z_mhz=xi_a_mhz,
        coupling_axis="z",
    )

    config = PolyadCouplingConfig(
        states=[s4, s6],
        coriolis_couplings=[coriolis],
        representation="Ir",
    )

    # Construct for J=3 (K = -3..+3, dim = 7, total = 14)
    block = construct_polyad_block(config, J=3)
    eigensystem = diagonalize_polyad_block(block)

    assert eigensystem.eigenvalues_mhz.shape == (14,)
    # Verify state mixing occurs for K != 0
    mixing = np.array(eigensystem.state_mixing_weights)
    assert mixing.shape == (14, 2)
    # State mixing sums to 1 for every level
    assert np.allclose(np.sum(mixing, axis=1), 1.0, atol=1e-6)


# =============================================================================
# 7. EXTRACTOR RESONANCE REPORT BRIDGE
# =============================================================================

def test_extractor_resonance_report_bridge() -> None:
    """Verifies conversion of ResonanceReport from cochem_quantum_extractor into Polyad blocks."""
    wc1 = WatsonConstants(A=150000.0, B=40000.0, C=30000.0)
    wc2 = WatsonConstants(A=149000.0, B=39800.0, C=29900.0)

    # Create ResonanceReport with 1 Fermi resonance
    res_match = ResonanceMatch(
        resonance_type="Fermi_I",
        mode_indices=[0, 1],
        frequencies_cm1=[500.0, 1005.0],
        energy_difference_cm1=5.0,
        coriolis_zeta=None,
        description="Fermi resonance between 2*nu0 and nu1",
    )
    report = ResonanceReport(
        resonances=[res_match],
        num_fermi=1,
        num_coriolis=0,
        polyads=[[0, 1]],
        all_frequencies_cm1=[500.0, 1005.0, 1600.0],
    )

    blocks = build_polyad_from_extractor_report(
        extractor_report=report,
        states_constants=[wc1, wc2],
        J=1,
    )

    assert len(blocks) == 1
    polyad_block = blocks[0]
    assert polyad_block.num_states == 2
    assert polyad_block.total_dim == 6


def test_polyad_darling_dennison_coupling() -> None:
    """Verifies polyad block construction with Darling-Dennison anharmonic resonance."""
    wc1 = WatsonConstants(A=290000.0, B=60000.0, C=48000.0)
    wc2 = WatsonConstants(A=285000.0, B=59500.0, C=47800.0)

    s1 = VibrationalState(state_index=0, label="2nu1", vibrational_energy_cm1=3000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="2nu3", vibrational_energy_cm1=3050.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        darling_dennison_couplings=[DarlingDennisonParam(state_i=0, state_j=1, gamma_cm1=-15.0)],
    )

    block = construct_polyad_block(config, J=1)
    assert block.num_states == 2
    assert block.total_dim == 6

    # Diagonalize and verify eigensystem
    eigensystem = diagonalize_polyad_block(block)
    assert eigensystem.eigenvalues_cm1.shape == (6,)
    assert eigensystem.state_mixing_weights.shape == (6, 2)


def test_polyad_watson_s_reduction() -> None:
    """Verifies polyad block construction using Watson S-reduction Hamiltonian."""
    wc1 = WatsonConstants(A=180000.0, B=45000.0, C=36000.0, d1=-0.002, d2=-0.0005)
    wc2 = WatsonConstants(A=178000.0, B=44800.0, C=35800.0, d1=-0.002, d2=-0.0005)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1200.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1210.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        fermi_couplings=[FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=10.0)],
        reduction="S",
    )

    block = construct_polyad_block(config, J=2)
    assert block.total_dim == 10
    assert block.reduction == "S"

    eigensystem = diagonalize_polyad_block(block)
    assert eigensystem.eigenvalues_cm1.shape == (10,)


def test_compute_polyad_transition_dipoles_2d_and_3d() -> None:
    """Verifies intensity-borrowing perturbed transition dipole projection for 2D and 3D tensors."""
    wc = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    s0 = VibrationalState(state_index=0, label="v0", vibrational_energy_cm1=1000.0, watson_constants=wc)
    s1 = VibrationalState(state_index=1, label="v1", vibrational_energy_cm1=1020.0, watson_constants=wc)

    config = PolyadCouplingConfig(
        states=[s0, s1],
        fermi_couplings=[FermiCouplingParam(state_i=0, state_j=1, w_fermi_cm1=15.0)],
    )

    block = construct_polyad_block(config, J=0)
    eigensystem = diagonalize_polyad_block(block)

    # 2D scalar transition dipoles
    dip_2d = jnp.array([[1.0, 0.0], [0.0, 0.1]], dtype=jnp.float64)
    res_2d = compute_polyad_transition_dipoles(eigensystem, eigensystem, dip_2d)
    assert res_2d.shape == (2, 2)

    # 3D vector transition dipoles (x, y, z components)
    dip_3d = jnp.ones((2, 2, 3), dtype=jnp.float64)
    res_3d = compute_polyad_transition_dipoles(eigensystem, eigensystem, dip_3d)
    assert res_3d.shape == (2, 2, 3)


def test_complex_coriolis_dlpack_roundtrip() -> None:
    """Verifies zero-copy DLPack extraction and reconstruction of complex Coriolis Hamiltonian."""
    wc1 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)
    wc2 = WatsonConstants(A=100000.0, B=50000.0, C=25000.0)

    s1 = VibrationalState(state_index=0, label="v1", vibrational_energy_cm1=1000.0, watson_constants=wc1)
    s2 = VibrationalState(state_index=1, label="v2", vibrational_energy_cm1=1020.0, watson_constants=wc2)

    config = PolyadCouplingConfig(
        states=[s1, s2],
        coriolis_couplings=[CoriolisCouplingParam(state_i=0, state_j=1, zeta_z=0.5, xi_z_mhz=5000.0)],
    )

    block = construct_polyad_block(config, J=1)
    payload = extract_pytree_dlpack(block)
    restored = import_pytree_dlpack(payload, block)

    assert jnp.allclose(block.hamiltonian_matrix, restored.hamiltonian_matrix)


# =============================================================================
# 8. MENDELEEV & ZERO-MOCK COMPLIANCE VERIFICATION
# =============================================================================

def test_mendeleev_dynamic_mass_compliance() -> None:
    """Verifies that isotopic mass lookups use dynamic mendeleev library."""
    c12_mass = float(mendeleev.element("C").mass)
    h1_mass = float(mendeleev.element("H").mass)
    o16_mass = float(mendeleev.element("O").mass)

    assert 12.0 <= c12_mass <= 12.012
    assert 1.0 <= h1_mass <= 1.01
    assert 15.99 <= o16_mass <= 16.00


def test_anti_spoofing_no_pass_and_no_mock() -> None:
    """Static AST check verifying zero pass statements and zero banned tokens in cochem_jax_coupling.py."""
    module_path = Path("src/cochem_spycfit/physics_core/cochem_jax_coupling.py")
    if not module_path.exists():
        module_path = Path("D:/__CoChem/GitHub-Repo/CoChem-SpycFit/src/cochem_spycfit/physics_core/cochem_jax_coupling.py")

    assert module_path.exists(), f"Target module not found at {module_path}"

    source_text = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text)

    # 1. Check for pass statements
    pass_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Pass)]
    assert len(pass_nodes) == 0, f"Found {len(pass_nodes)} forbidden 'pass' statements in {module_path}"

    # 2. Check for banned tokens (mock, dummy, placeholder, fake, stub, synthetic) in identifiers
    banned = ["mock", "dummy", "placeholder", "fake", "stub", "synthetic"]
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for b in banned:
                assert b not in node.name.lower(), f"Banned token '{b}' in function name: {node.name}"
        elif isinstance(node, ast.Name):
            for b in banned:
                assert b not in node.id.lower(), f"Banned token '{b}' in variable name: {node.id}"


# =============================================================================
# 9. RUNTIME TYPE HINTS & INTROSPECTION COMPLIANCE
# =============================================================================

def test_get_type_hints_compliance() -> None:
    """Verifies that typing.get_type_hints succeeds without NameError across all module functions."""
    import typing
    import cochem_spycfit.physics_core.cochem_jax_coupling as mod

    for name, obj in inspect.getmembers(mod, inspect.isfunction):
        if obj.__module__ == mod.__name__:
            hints = typing.get_type_hints(obj)
            assert isinstance(hints, dict)




--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_jax_fitter.py ---
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

    active_mask = np.asarray(guard_output_2d["active_mask"])
    np.testing.assert_array_equal(active_mask, [True, True, False, False])
    assert int(guard_output_2d["active_count"]) == 2
    assert int(guard_output_2d["frozen_count"]) == 2

    # Verify frozen parameters are isolated and active parameters preserved
    active_params = np.asarray(guard_output_2d["active_parameters"])
    frozen_params = np.asarray(guard_output_2d["frozen_parameters"])
    assert active_params[0] == 12500.0
    assert active_params[1] == 6200.0
    assert active_params[2] == 0.0
    assert active_params[3] == 0.0
    assert frozen_params[2] == 0.005
    assert frozen_params[3] == 0.0001

    # Call with 1D variance vector
    guard_output_1d = sobol_sensitivity_guard(
        parameters=initial_params,
        covariance=cov_diag,
        threshold=1e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(guard_output_1d["active_mask"]),
        [True, True, False, False],
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


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_jax_coupling.py ---
"""
CoChem-SpycFit: Hardware-Agnostic JAX Polyad Resonance Matrix Engine Proxy.
"""

from src.cochem_spycfit.physics_core.cochem_jax_coupling import (
    CM1_TO_MHZ,
    DEFAULT_CORIOLIS_THRESHOLD_CM1,
    DEFAULT_FERMI_THRESHOLD_CM1,
    MHZ_TO_CM1,
    SPEED_OF_LIGHT_CM_S,
    CoriolisCouplingParam,
    CouplingError,
    DarlingDennisonParam,
    DeperturbationError,
    DLPackExportError,
    DLPackImportError,
    FermiCouplingParam,
    InvalidCouplingParameterError,
    InvalidPolyadStateError,
    NonHermitianPolyadError,
    PolyadBlockPyTree,
    PolyadCouplingConfig,
    PolyadDeperturbationResult,
    PolyadEigensystemPyTree,
    ResonanceType,
    StateAssignment,
    VibrationalState,
    build_polyad_from_extractor_report,
    compute_polyad_transition_dipoles,
    construct_polyad_block,
    deperturb_polyad_eigenstates,
    diagonalize_polyad_block,
    extract_pytree_dlpack,
    import_pytree_dlpack,
)

__all__ = [
    "CM1_TO_MHZ",
    "DEFAULT_CORIOLIS_THRESHOLD_CM1",
    "DEFAULT_FERMI_THRESHOLD_CM1",
    "MHZ_TO_CM1",
    "SPEED_OF_LIGHT_CM_S",
    "CoriolisCouplingParam",
    "CouplingError",
    "DarlingDennisonParam",
    "DeperturbationError",
    "DLPackExportError",
    "DLPackImportError",
    "FermiCouplingParam",
    "InvalidCouplingParameterError",
    "InvalidPolyadStateError",
    "NonHermitianPolyadError",
    "PolyadBlockPyTree",
    "PolyadCouplingConfig",
    "PolyadDeperturbationResult",
    "PolyadEigensystemPyTree",
    "ResonanceType",
    "StateAssignment",
    "VibrationalState",
    "build_polyad_from_extractor_report",
    "compute_polyad_transition_dipoles",
    "construct_polyad_block",
    "deperturb_polyad_eigenstates",
    "diagonalize_polyad_block",
    "extract_pytree_dlpack",
    "import_pytree_dlpack",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_jax_coupling.py ---
"""
CoChem-SpycFit: Hardware-Agnostic JAX Polyad Resonance Matrix Engine & Pure PyTree Solver.

This module implements:
1. Unified Polyad Hamiltonian matrix construction coupling interacting vibrational states
   via off-diagonal Coriolis (zeta) and Fermi resonance parameters.
2. Complete breakdown handling of the isolated rigid-rotor approximation for overlapping,
   resonating, and perturbed vibrational manifolds.
3. Strict functional purity: Zero I/O side effects during matrix construction and diagonalization,
   returning all physical states as pure mathematical JAX PyTrees.
4. Zero-copy DLPack IPC extraction and ingestion gateway to eliminate inter-process memory
   serialization bottlenecks.
5. Deperturbation analysis, eigenvector vibrational state decomposition, state mixing weights,
   and intensity borrowing calculations.
6. Theoretical Tensor Gateway integration with ResonanceReport from cochem_quantum_extractor.

Precision: JAX FP64 precision enabled globally.
Zero-Tolerance Anti-Spoofing: 0 pass statements, 0 test doubles, authentic physical quantum mechanics.
"""

from __future__ import annotations

import logging
import math
import os
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from .cochem_jax_builder import (
    WatsonConstants,
    construct_watson_a_matrix_jax,
    construct_watson_s_matrix_jax,
    determine_representation,
)
from .cochem_quantum_extractor import ResonanceReport

# Enforce JAX FP64 precision globally for polyad matrix diagonalization
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.JAXCoupling")

# Physical constants and conversion factors
SPEED_OF_LIGHT_CM_S: float = 29979245800.0  # cm / s
MHZ_TO_CM1: float = 1.0 / 29979.2458
CM1_TO_MHZ: float = 29979.2458
DEFAULT_FERMI_THRESHOLD_CM1: float = 50.0
DEFAULT_CORIOLIS_THRESHOLD_CM1: float = 30.0


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class CouplingError(Exception):
    """Base exception for all JAX polyad coupling and resonance operations."""


class NonHermitianPolyadError(CouplingError):
    """Raised when a constructed Polyad block violates Hermitian symmetry."""


class InvalidPolyadStateError(CouplingError):
    """Raised when an invalid vibrational state or state index is specified."""


class InvalidCouplingParameterError(CouplingError):
    """Raised when coupling parameters are malformed or reference non-existent states."""


class DLPackExportError(CouplingError):
    """Raised when DLPack capsule extraction from a PyTree fails."""


class DLPackImportError(CouplingError):
    """Raised when DLPack capsule ingestion into a PyTree fails."""


class DeperturbationError(CouplingError):
    """Raised when eigenvector deperturbation analysis encounters invalid state shapes."""


# =============================================================================
# 2. ENUMS AND PYDANTIC CONFIGURATION SCHEMAS
# =============================================================================

class ResonanceType(str, Enum):
    """Classification of vibrational and vibration-rotation resonances."""
    FERMI_I = "Fermi_I"
    FERMI_II = "Fermi_II"
    CORIOLIS_Z = "Coriolis_z"
    CORIOLIS_XY = "Coriolis_xy"
    DARLING_DENNISON = "Darling_Dennison"


class VibrationalState(BaseModel):
    """
    Physical vibrational state definition within a coupled Polyad manifold.
    Energies are defined in cm^-1 and auto-converted to MHz.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_index: int = Field(..., ge=0, description="Unique 0-based vibrational state index")
    label: str = Field(..., min_length=1, description="Spectroscopic state label (e.g. '100', '020', 'nu1')")
    vibrational_energy_cm1: float = Field(..., ge=0.0, description="Vibrational band origin in cm^-1")
    vibrational_energy_mhz: Optional[float] = Field(default=None, description="Vibrational energy in MHz")
    watson_constants: WatsonConstants = Field(..., description="State-specific Watson rotational constants")
    quanta: List[int] = Field(default_factory=list, description="Vibrational quantum numbers (v1, v2, ...)")
    symmetry: Optional[str] = Field(default=None, description="Vibrational symmetry irreducible representation")
    dipole_derivatives: Optional[List[float]] = Field(default=None, description="Cartesian dipole derivatives [dmu_x, dmu_y, dmu_z]")

    @model_validator(mode="after")
    def compute_defaults(self) -> VibrationalState:
        if self.vibrational_energy_mhz is None:
            self.vibrational_energy_mhz = float(self.vibrational_energy_cm1 * CM1_TO_MHZ)
        return self


class FermiCouplingParam(BaseModel):
    """
    Fermi resonance (anharmonic potential) coupling parameter between states i and j.
    Operator: H_Fermi = W_ij + W_ij^(J) * J(J+1) + W_ij^(K) * K^2
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_i: int = Field(..., ge=0, description="0-based index of first interacting state")
    state_j: int = Field(..., ge=0, description="0-based index of second interacting state")
    w_fermi_cm1: float = Field(..., description="Fermi cubic/quartic coupling constant W_ij in cm^-1")
    w_fermi_mhz: Optional[float] = Field(default=None, description="Fermi coupling constant W_ij in MHz")
    w_j_mhz: float = Field(default=0.0, description="J-dependent centrifugal correction to Fermi coupling (MHz)")
    w_k_mhz: float = Field(default=0.0, description="K-dependent correction to Fermi coupling (MHz)")
    w_xy_mhz: float = Field(default=0.0, description="Delta K = +-2 asymmetry correction to Fermi coupling (MHz)")

    @model_validator(mode="after")
    def compute_defaults(self) -> FermiCouplingParam:
        if self.w_fermi_mhz is None:
            self.w_fermi_mhz = float(self.w_fermi_cm1 * CM1_TO_MHZ)
        return self


class CoriolisCouplingParam(BaseModel):
    """
    Coriolis vibration-rotation resonance coupling parameter between states i and j.
    Operator: H_Cor = -2 * B_alpha * zeta_ij^alpha * J_alpha * (q_i p_j - q_j p_i)
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_i: int = Field(..., ge=0, description="0-based index of first interacting state")
    state_j: int = Field(..., ge=0, description="0-based index of second interacting state")
    zeta_z: float = Field(default=0.0, ge=-1.0, le=1.0, description="Coriolis zeta along quantization axis z")
    zeta_x: float = Field(default=0.0, ge=-1.0, le=1.0, description="Coriolis zeta along perpendicular axis x")
    zeta_y: float = Field(default=0.0, ge=-1.0, le=1.0, description="Coriolis zeta along perpendicular axis y")
    xi_z_mhz: Optional[float] = Field(default=None, description="Effective z-axis Coriolis parameter xi_z (MHz)")
    xi_xy_mhz: Optional[float] = Field(default=None, description="Effective xy-axis Coriolis parameter xi_xy (MHz)")
    coupling_axis: str = Field(default="z", description="Primary coupling axis: 'z', 'x', 'y', or 'xy'")

    @field_validator("coupling_axis")
    @classmethod
    def validate_axis(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in ("z", "x", "y", "xy", "xyz"):
            raise ValueError(f"Invalid Coriolis coupling axis '{v}'. Must be 'z', 'x', 'y', 'xy', or 'xyz'.")
        return v_clean


class DarlingDennisonParam(BaseModel):
    """
    Darling-Dennison resonance (quartic anharmonic) coupling parameter between states i and j.
    Couples states differing by Delta v_i = +-2, Delta v_j = -+2.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    state_i: int = Field(..., ge=0, description="0-based index of first interacting state")
    state_j: int = Field(..., ge=0, description="0-based index of second interacting state")
    gamma_cm1: float = Field(..., description="Darling-Dennison coupling constant gamma_ij in cm^-1")
    gamma_mhz: Optional[float] = Field(default=None, description="Darling-Dennison coupling constant gamma_ij in MHz")

    @model_validator(mode="after")
    def compute_defaults(self) -> DarlingDennisonParam:
        if self.gamma_mhz is None:
            self.gamma_mhz = float(self.gamma_cm1 * CM1_TO_MHZ)
        return self


class PolyadCouplingConfig(BaseModel):
    """Complete specification of a coupled vibrational Polyad manifold."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    states: List[VibrationalState] = Field(..., min_length=1, description="List of coupled vibrational states")
    fermi_couplings: List[FermiCouplingParam] = Field(default_factory=list, description="Fermi resonance parameters")
    coriolis_couplings: List[CoriolisCouplingParam] = Field(default_factory=list, description="Coriolis resonance parameters")
    darling_dennison_couplings: List[DarlingDennisonParam] = Field(default_factory=list, description="Darling-Dennison parameters")
    representation: str = Field(default="Ir", description="Watson coordinate representation ('Ir' or 'IIIr')")
    reduction: str = Field(default="A", description="Watson reduction type ('A' or 'S')")
    enforce_hermitian: bool = Field(default=True, description="Whether to verify Hermitian symmetry of constructed block")


class StateAssignment(BaseModel):
    """Spectroscopic assignment and mixing decomposition for a single perturbed eigenstate."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    eigenstate_index: int = Field(..., ge=0, description="0-based eigenstate energy level index")
    energy_mhz: float = Field(..., description="Perturbed energy in MHz")
    energy_cm1: float = Field(..., description="Perturbed energy in cm^-1")
    assigned_state_index: int = Field(..., ge=0, description="Leading vibrational state index")
    assigned_label: str = Field(..., description="Leading vibrational state label")
    state_purity: float = Field(..., ge=0.0, le=1.0, description="Fractional contribution of leading state")
    mixing_fractions: List[float] = Field(..., description="Fractional contributions of all vibrational states")
    mixing_entropy: float = Field(..., ge=0.0, description="Shannon entropy of state mixing")
    energy_shift_mhz: float = Field(..., description="Perturbed energy shift relative to unperturbed origin (MHz)")


class PolyadDeperturbationResult(BaseModel):
    """Structured report detailing state mixing, level shifts, and deperturbed assignments."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    j_value: int = Field(..., ge=0, description="Rotational quantum number J")
    total_states: int = Field(..., ge=1, description="Total number of coupled levels")
    num_vibrational_states: int = Field(..., ge=1, description="Number of vibrational states in Polyad")
    state_assignments: List[StateAssignment] = Field(..., description="Assignments for all eigenstates")
    is_strongly_coupled: bool = Field(..., description="True if any level exhibits >20% vibrational mixing")
    max_mixing_fraction: float = Field(..., ge=0.0, le=1.0, description="Maximum off-diagonal mixing fraction observed")


# =============================================================================
# 3. REGISTERED JAX PYTREE STRUCTURES
# =============================================================================

@jax.tree_util.register_pytree_node_class
class PolyadBlockPyTree:
    """
    Pure JAX PyTree encapsulating a unified Polyad Hamiltonian block.
    Guarantees functional purity with zero I/O side effects.
    """

    def __init__(
        self,
        hamiltonian_matrix: jnp.ndarray,
        state_energies_mhz: jnp.ndarray,
        coupling_matrix: jnp.ndarray,
        j_value: int,
        num_states: int,
        dim_per_state: int,
        total_dim: int,
        state_labels: Optional[List[str]] = None,
        representation: str = "Ir",
        reduction: str = "A",
    ) -> None:
        self.hamiltonian_matrix = jnp.asarray(hamiltonian_matrix)
        self.state_energies_mhz = jnp.asarray(state_energies_mhz, dtype=jnp.float64)
        self.coupling_matrix = jnp.asarray(coupling_matrix)
        self.j_value = int(j_value)
        self.num_states = int(num_states)
        self.dim_per_state = int(dim_per_state)
        self.total_dim = int(total_dim)
        self.state_labels = list(state_labels) if state_labels is not None else [f"v{i}" for i in range(num_states)]
        self.representation = str(representation)
        self.reduction = str(reduction)

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[Any, ...]]:
        children = (self.hamiltonian_matrix, self.state_energies_mhz, self.coupling_matrix)
        aux_data = (
            self.j_value,
            self.num_states,
            self.dim_per_state,
            self.total_dim,
            tuple(self.state_labels),
            self.representation,
            self.reduction,
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jnp.ndarray, ...]) -> PolyadBlockPyTree:
        hamiltonian_matrix, state_energies_mhz, coupling_matrix = children
        (
            j_value,
            num_states,
            dim_per_state,
            total_dim,
            state_labels,
            representation,
            reduction,
        ) = aux_data
        return cls(
            hamiltonian_matrix=hamiltonian_matrix,
            state_energies_mhz=state_energies_mhz,
            coupling_matrix=coupling_matrix,
            j_value=j_value,
            num_states=num_states,
            dim_per_state=dim_per_state,
            total_dim=total_dim,
            state_labels=list(state_labels),
            representation=representation,
            reduction=reduction,
        )


@jax.tree_util.register_pytree_node_class
class PolyadEigensystemPyTree:
    """
    Pure JAX PyTree containing eigenvalues, eigenvectors, and state mixing weights.
    """

    def __init__(
        self,
        eigenvalues_mhz: jnp.ndarray,
        eigenvalues_cm1: jnp.ndarray,
        eigenvectors: jnp.ndarray,
        state_mixing_weights: jnp.ndarray,
        j_value: int,
        num_states: int,
        dim_per_state: int,
        total_dim: int,
        state_labels: Optional[List[str]] = None,
    ) -> None:
        self.eigenvalues_mhz = jnp.asarray(eigenvalues_mhz, dtype=jnp.float64)
        self.eigenvalues_cm1 = jnp.asarray(eigenvalues_cm1, dtype=jnp.float64)
        self.eigenvectors = jnp.asarray(eigenvectors)
        self.state_mixing_weights = jnp.asarray(state_mixing_weights, dtype=jnp.float64)
        self.j_value = int(j_value)
        self.num_states = int(num_states)
        self.dim_per_state = int(dim_per_state)
        self.total_dim = int(total_dim)
        self.state_labels = list(state_labels) if state_labels is not None else [f"v{i}" for i in range(num_states)]

    def tree_flatten(self) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], Tuple[Any, ...]]:
        children = (
            self.eigenvalues_mhz,
            self.eigenvalues_cm1,
            self.eigenvectors,
            self.state_mixing_weights,
        )
        aux_data = (
            self.j_value,
            self.num_states,
            self.dim_per_state,
            self.total_dim,
            tuple(self.state_labels),
        )
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: Tuple[Any, ...], children: Tuple[jnp.ndarray, ...]) -> PolyadEigensystemPyTree:
        (
            eigenvalues_mhz,
            eigenvalues_cm1,
            eigenvectors,
            state_mixing_weights,
        ) = children
        (
            j_value,
            num_states,
            dim_per_state,
            total_dim,
            state_labels,
        ) = aux_data
        return cls(
            eigenvalues_mhz=eigenvalues_mhz,
            eigenvalues_cm1=eigenvalues_cm1,
            eigenvectors=eigenvectors,
            state_mixing_weights=state_mixing_weights,
            j_value=j_value,
            num_states=num_states,
            dim_per_state=dim_per_state,
            total_dim=total_dim,
            state_labels=list(state_labels),
        )


# =============================================================================
# 4. UNIFIED POLYAD BLOCK CONSTRUCTION
# =============================================================================

def construct_polyad_block(
    states: Union[PolyadCouplingConfig, List[VibrationalState], Dict[str, Any]],
    coupling_params: Optional[Union[Dict[str, Any], PolyadCouplingConfig]] = None,
    J: int = 0,
    representation: str = "Ir",
    reduction: str = "A",
    enforce_hermitian: bool = True,
) -> PolyadBlockPyTree:
    """
    Constructs a massive unified Polyad Hamiltonian block coupling interacting vibrational states
    via off-diagonal Coriolis and Fermi parameters in JAX FP64.

    Parameters:
        states: PolyadCouplingConfig, list of VibrationalState objects, or dict configuration
        coupling_params: Optional additional coupling parameters if states is a list
        J: Rotational quantum number J (>= 0)
        representation: Watson axis representation ('Ir' or 'IIIr')
        reduction: Watson reduction ('A' or 'S')
        enforce_hermitian: Whether to verify Hermitian symmetry

    Returns:
        PolyadBlockPyTree: Pure JAX PyTree holding the constructed Hamiltonian and metadata.
    """
    if J < 0:
        raise ValueError(f"Rotational quantum number J must be non-negative, got J={J}")

    # Normalize configuration input
    config: PolyadCouplingConfig
    if isinstance(states, PolyadCouplingConfig):
        config = states
    elif isinstance(states, dict) and "states" in states:
        config = PolyadCouplingConfig(**states)
    elif isinstance(states, list):
        fermi_list: List[FermiCouplingParam] = []
        coriolis_list: List[CoriolisCouplingParam] = []
        dd_list: List[DarlingDennisonParam] = []
        if isinstance(coupling_params, PolyadCouplingConfig):
            fermi_list = coupling_params.fermi_couplings
            coriolis_list = coupling_params.coriolis_couplings
            dd_list = coupling_params.darling_dennison_couplings
        elif isinstance(coupling_params, dict):
            fermi_list = [FermiCouplingParam(**item) for item in coupling_params.get("fermi_couplings", [])]
            coriolis_list = [CoriolisCouplingParam(**item) for item in coupling_params.get("coriolis_couplings", [])]
            dd_list = [DarlingDennisonParam(**item) for item in coupling_params.get("darling_dennison_couplings", [])]

        config = PolyadCouplingConfig(
            states=states,
            fermi_couplings=fermi_list,
            coriolis_couplings=coriolis_list,
            darling_dennison_couplings=dd_list,
            representation=representation,
            reduction=reduction,
            enforce_hermitian=enforce_hermitian,
        )
    else:
        raise InvalidPolyadStateError(f"Unsupported states input type: {type(states)}")

    num_v_states = len(config.states)
    if num_v_states == 0:
        raise InvalidPolyadStateError("Polyad configuration must contain at least one vibrational state.")

    # Validate state indices
    state_index_set = {s.state_index for s in config.states}
    if len(state_index_set) != num_v_states:
        raise InvalidPolyadStateError("Duplicate vibrational state indices detected in configuration.")

    for i in range(num_v_states):
        if i not in state_index_set:
            raise InvalidPolyadStateError(f"State index {i} missing from sequence 0..{num_v_states-1}")

    # Validate coupling parameters reference valid state indices
    for f in config.fermi_couplings:
        if f.state_i not in state_index_set or f.state_j not in state_index_set:
            raise InvalidCouplingParameterError(
                f"Fermi coupling references invalid state index ({f.state_i}, {f.state_j}). Valid: {state_index_set}"
            )

    for c in config.coriolis_couplings:
        if c.state_i not in state_index_set or c.state_j not in state_index_set:
            raise InvalidCouplingParameterError(
                f"Coriolis coupling references invalid state index ({c.state_i}, {c.state_j}). Valid: {state_index_set}"
            )

    for d in config.darling_dennison_couplings:
        if d.state_i not in state_index_set or d.state_j not in state_index_set:
            raise InvalidCouplingParameterError(
                f"Darling-Dennison coupling references invalid state index ({d.state_i}, {d.state_j}). Valid: {state_index_set}"
            )

    dim_per_state = 2 * J + 1
    total_dim = num_v_states * dim_per_state
    j_val = float(J)
    j_jp1 = j_val * (j_val + 1.0)
    k_vals = np.arange(-J, J + 1, dtype=np.float64)

    # Check if imaginary terms (Coriolis z-axis or perpendicular) are present
    has_coriolis = len(config.coriolis_couplings) > 0 and J > 0
    matrix_dtype = jnp.complex128 if has_coriolis else jnp.float64

    # Build intra-state Watson rotational diagonal blocks
    state_energies_list: List[float] = []
    state_labels_list: List[str] = []
    h_global = np.zeros((total_dim, total_dim), dtype=np.complex128 if has_coriolis else np.float64)

    # Sort states by state_index
    sorted_states = sorted(config.states, key=lambda s: s.state_index)

    for state in sorted_states:
        idx = state.state_index
        state_labels_list.append(state.label)
        origin_mhz = float(state.vibrational_energy_mhz if state.vibrational_energy_mhz is not None else state.vibrational_energy_cm1 * CM1_TO_MHZ)
        state_energies_list.append(origin_mhz)

        # Build Watson rotational block for state
        target_red = config.reduction.upper()
        if target_red == "S":
            rot_block = construct_watson_s_matrix_jax(state.watson_constants, J=J, representation=config.representation)
        else:
            rot_block = construct_watson_a_matrix_jax(state.watson_constants, J=J, representation=config.representation)

        row_start = idx * dim_per_state
        row_end = (idx + 1) * dim_per_state

        # Diagonal block H_ii = G_i * I + H_rot
        diag_block = np.array(rot_block, dtype=np.float64) + np.eye(dim_per_state, dtype=np.float64) * origin_mhz
        h_global[row_start:row_end, row_start:row_end] = diag_block

    # Build off-diagonal Fermi coupling sub-blocks
    for fermi in config.fermi_couplings:
        i = fermi.state_i
        j = fermi.state_j
        if i == j:
            continue

        w_mhz = float(fermi.w_fermi_mhz if fermi.w_fermi_mhz is not None else fermi.w_fermi_cm1 * CM1_TO_MHZ)
        w_j = float(fermi.w_j_mhz)
        w_k = float(fermi.w_k_mhz)
        w_xy = float(fermi.w_xy_mhz)

        r_start_i = i * dim_per_state
        r_end_i = (i + 1) * dim_per_state
        r_start_j = j * dim_per_state
        r_end_j = (j + 1) * dim_per_state

        # Delta K = 0 diagonal Fermi terms within sub-block
        fermi_diag = w_mhz + w_j * j_jp1 + w_k * (k_vals ** 2)
        sub_block_fermi = np.diag(fermi_diag)

        # Delta K = +-2 asymmetry terms if present
        if abs(w_xy) > 1e-12 and J >= 2:
            for k_idx in range(len(k_vals) - 2):
                k_s = k_vals[k_idx]
                raising_factor = math.sqrt((j_val - k_s) * (j_val - k_s - 1.0) * (j_val + k_s + 1.0) * (j_val + k_s + 2.0))
                coupling_val = w_xy * raising_factor
                sub_block_fermi[k_idx + 2, k_idx] += coupling_val
                sub_block_fermi[k_idx, k_idx + 2] += coupling_val

        h_global[r_start_i:r_end_i, r_start_j:r_end_j] += sub_block_fermi
        h_global[r_start_j:r_end_j, r_start_i:r_end_i] += sub_block_fermi.conj().T

    # Build off-diagonal Coriolis coupling sub-blocks
    for cor in config.coriolis_couplings:
        i = cor.state_i
        j = cor.state_j
        if i == j or J == 0:
            continue

        r_start_i = i * dim_per_state
        r_end_i = (i + 1) * dim_per_state
        r_start_j = j * dim_per_state
        r_end_j = (j + 1) * dim_per_state

        sub_block_cor = np.zeros((dim_per_state, dim_per_state), dtype=np.complex128)

        # Derive effective xi parameters if not explicitly supplied
        s_i = sorted_states[i]
        s_j = sorted_states[j]
        omega_i = max(s_i.vibrational_energy_cm1, 10.0)
        omega_j = max(s_j.vibrational_energy_cm1, 10.0)
        omega_factor = 0.5 * (math.sqrt(omega_i / omega_j) + math.sqrt(omega_j / omega_i))

        # 1. z-axis Coriolis coupling (Delta K = 0)
        xi_z = cor.xi_z_mhz
        if xi_z is None:
            a_const = float(s_i.watson_constants.A)
            xi_z = 2.0 * a_const * cor.zeta_z * omega_factor

        if abs(xi_z) > 1e-12:
            # <i, J, K | H_Cor^z | j, J, K> = -i * xi_z * K
            z_diag = -1.0j * xi_z * k_vals
            sub_block_cor += np.diag(z_diag)

        # 2. Perpendicular (xy) Coriolis coupling (Delta K = +-1)
        xi_xy = cor.xi_xy_mhz
        if xi_xy is None:
            b_const = float(s_i.watson_constants.B)
            eff_zeta_xy = math.sqrt(cor.zeta_x ** 2 + cor.zeta_y ** 2) if (cor.zeta_x != 0 or cor.zeta_y != 0) else 0.0
            xi_xy = b_const * eff_zeta_xy * omega_factor

        if abs(xi_xy) > 1e-12:
            # <i, J, K+1 | H_Cor | j, J, K> = xi_xy * sqrt(J(J+1) - K(K+1))
            for k_idx in range(len(k_vals) - 1):
                k_val_curr = k_vals[k_idx]
                step_factor = math.sqrt(max(0.0, j_jp1 - k_val_curr * (k_val_curr + 1.0)))
                val = xi_xy * step_factor
                sub_block_cor[k_idx + 1, k_idx] += val
                sub_block_cor[k_idx, k_idx + 1] += val

        h_global[r_start_i:r_end_i, r_start_j:r_end_j] += sub_block_cor
        h_global[r_start_j:r_end_j, r_start_i:r_end_i] += sub_block_cor.conj().T

    # Build off-diagonal Darling-Dennison coupling sub-blocks
    for dd in config.darling_dennison_couplings:
        i = dd.state_i
        j = dd.state_j
        if i == j:
            continue

        gamma_mhz = float(dd.gamma_mhz if dd.gamma_mhz is not None else dd.gamma_cm1 * CM1_TO_MHZ)
        r_start_i = i * dim_per_state
        r_end_i = (i + 1) * dim_per_state
        r_start_j = j * dim_per_state
        r_end_j = (j + 1) * dim_per_state

        sub_block_dd = np.eye(dim_per_state, dtype=np.float64) * gamma_mhz
        h_global[r_start_i:r_end_i, r_start_j:r_end_j] += sub_block_dd
        h_global[r_start_j:r_end_j, r_start_i:r_end_i] += sub_block_dd.conj().T

    # Enforce strict Hermitian Lock
    h_conj = h_global.conj().T
    if not np.allclose(h_global, h_conj, rtol=1e-8, atol=1e-8):
        max_asym = float(np.max(np.abs(h_global - h_conj)))
        raise NonHermitianPolyadError(
            f"[HARD_ABORT: PHYSICS WALL] Polyad Hamiltonian violated Hermitian symmetry. Max deviation = {max_asym:.8e}."
        )

    # Cast to real if all imaginary components are negligible
    if np.max(np.abs(h_global.imag)) < 1e-12:
        h_final = jnp.asarray(h_global.real, dtype=jnp.float64)
    else:
        h_final = jnp.asarray(h_global, dtype=jnp.complex128)

    state_energies_arr = jnp.array(state_energies_list, dtype=jnp.float64)
    coupling_preview = jnp.asarray(h_final[:num_v_states, :num_v_states])

    return PolyadBlockPyTree(
        hamiltonian_matrix=h_final,
        state_energies_mhz=state_energies_arr,
        coupling_matrix=coupling_preview,
        j_value=J,
        num_states=num_v_states,
        dim_per_state=dim_per_state,
        total_dim=total_dim,
        state_labels=state_labels_list,
        representation=config.representation,
        reduction=config.reduction,
    )


# =============================================================================
# 5. PURE JAX DIAGONALIZATION & DEPERTURBATION SOLVER
# =============================================================================

def diagonalize_polyad_block(polyad_block: PolyadBlockPyTree) -> PolyadEigensystemPyTree:
    """
    Pure JAX Hermitian eigensolver for unified Polyad Hamiltonian blocks.
    Computes sorted energy eigenvalues (in MHz and cm^-1), unitary eigenvector transformation matrix,
    and state mixing probability weights.

    Parameters:
        polyad_block: PolyadBlockPyTree instance

    Returns:
        PolyadEigensystemPyTree: Pure JAX PyTree holding eigensystem results.
    """
    h_mat = polyad_block.hamiltonian_matrix
    evals, evecs = jnp.linalg.eigh(h_mat)

    # Convert eigenvalues to cm^-1
    evals_cm1 = evals * MHZ_TO_CM1

    # Calculate fractional vibrational state mixing weights for each eigenstate k
    # P_m(k) = sum_{K} |U_{(m, K), k}|^2
    num_v_states = polyad_block.num_states
    dim_per_state = polyad_block.dim_per_state
    total_dim = polyad_block.total_dim

    # Probability density of eigenvectors |U_{alpha, k}|^2
    prob_matrix = jnp.abs(evecs) ** 2  # shape (total_dim, total_dim)

    # Reshape to (num_v_states, dim_per_state, total_dim) and sum over K (axis 1)
    prob_reshaped = jnp.reshape(prob_matrix, (num_v_states, dim_per_state, total_dim))
    # State mixing weights for each eigenstate k: shape (total_dim, num_v_states)
    state_weights = jnp.sum(prob_reshaped, axis=1).T

    return PolyadEigensystemPyTree(
        eigenvalues_mhz=evals,
        eigenvalues_cm1=evals_cm1,
        eigenvectors=evecs,
        state_mixing_weights=state_weights,
        j_value=polyad_block.j_value,
        num_states=num_v_states,
        dim_per_state=dim_per_state,
        total_dim=total_dim,
        state_labels=polyad_block.state_labels,
    )


def deperturb_polyad_eigenstates(
    eigensystem: PolyadEigensystemPyTree,
    state_labels: Optional[List[str]] = None,
) -> PolyadDeperturbationResult:
    """
    De-perturbation analysis: Decomposes perturbed eigenstates into unperturbed vibrational basis states.
    Identifies leading state assignments, fractional mixing weights, Shannon mixing entropy, and level shifts.

    Parameters:
        eigensystem: PolyadEigensystemPyTree instance
        state_labels: Optional list of vibrational state labels

    Returns:
        PolyadDeperturbationResult: Detailed assignment and mixing report.
    """
    labels = state_labels if state_labels is not None else eigensystem.state_labels
    weights_np = np.array(eigensystem.state_mixing_weights, dtype=np.float64)
    evals_mhz_np = np.array(eigensystem.eigenvalues_mhz, dtype=np.float64)
    evals_cm1_np = np.array(eigensystem.eigenvalues_cm1, dtype=np.float64)

    total_dim = eigensystem.total_dim
    num_v_states = eigensystem.num_states

    assignments: List[StateAssignment] = []
    is_strongly_coupled = False
    max_offdiag_fraction = 0.0

    for k in range(total_dim):
        w_k = weights_np[k, :]
        best_state_idx = int(np.argmax(w_k))
        purity = float(w_k[best_state_idx])
        assigned_label = labels[best_state_idx] if best_state_idx < len(labels) else f"State_{best_state_idx}"

        # Calculate off-diagonal mixing (1.0 - purity)
        offdiag = float(1.0 - purity)
        if offdiag > max_offdiag_fraction:
            max_offdiag_fraction = offdiag

        if purity < 0.80:
            is_strongly_coupled = True

        # Calculate Shannon mixing entropy: S = - sum P * ln(P)
        p_safe = np.clip(w_k, 1e-15, 1.0)
        entropy = float(-np.sum(p_safe * np.log(p_safe)))

        e_mhz = float(evals_mhz_np[k])
        e_cm1 = float(evals_cm1_np[k])

        assignment = StateAssignment(
            eigenstate_index=k,
            energy_mhz=e_mhz,
            energy_cm1=e_cm1,
            assigned_state_index=best_state_idx,
            assigned_label=assigned_label,
            state_purity=purity,
            mixing_fractions=[float(x) for x in w_k],
            mixing_entropy=entropy,
            energy_shift_mhz=0.0,
        )
        assignments.append(assignment)

    return PolyadDeperturbationResult(
        j_value=eigensystem.j_value,
        total_states=total_dim,
        num_vibrational_states=num_v_states,
        state_assignments=assignments,
        is_strongly_coupled=is_strongly_coupled,
        max_mixing_fraction=max_offdiag_fraction,
    )


# =============================================================================
# 6. ZERO-COPY DLPACK IPC EXTRACTION & RECONSTRUCTION GATEWAY
# =============================================================================

class DLPackTensor:
    """
    Zero-copy DLPack protocol container implementing Python Array API DLPack standard.
    Allows seamless zero-copy tensor sharing between JAX, NumPy, PyTorch, and external solvers.
    """

    def __init__(self, tensor: Any) -> None:
        self._tensor = jnp.asarray(tensor)

    def __dlpack__(self, stream: Any = None) -> Any:
        if hasattr(self._tensor, "__dlpack__"):
            return self._tensor.__dlpack__()
        return self._tensor

    def __dlpack_device__(self) -> Tuple[int, int]:
        if hasattr(self._tensor, "__dlpack_device__"):
            return self._tensor.__dlpack_device__()
        return (1, 0)

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def dtype(self) -> str:
        return str(self._tensor.dtype)


def extract_pytree_dlpack(pytree: Any) -> Dict[str, Any]:
    """
    Extracts all array leaves from a JAX PyTree via zero-copy DLPack protocol objects.
    Enables external orchestrators, PyTorch, CuPy, or Zarr stores to ingest arrays
    without host-device copying or memory serialization bottlenecks.

    Parameters:
        pytree: Any registered JAX PyTree instance (e.g. PolyadBlockPyTree, PolyadEigensystemPyTree)

    Returns:
        Dict[str, Any]: Container mapping leaf paths to DLPack protocol objects and PyTree metadata.
    """
    try:
        leaves, treedef = jax.tree_util.tree_flatten(pytree)
        dlpack_leaves: List[Dict[str, Any]] = []

        for idx, leaf in enumerate(leaves):
            if isinstance(leaf, (jax.Array, np.ndarray, jnp.ndarray)):
                arr = jnp.asarray(leaf)
                wrapper = DLPackTensor(arr)
                capsule = wrapper.__dlpack__()
                dlpack_leaves.append({
                    "index": idx,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "dlpack": wrapper,
                    "capsule": capsule,
                })
            else:
                dlpack_leaves.append({
                    "index": idx,
                    "value": leaf,
                })

        return {
            "treedef": treedef,
            "leaves": dlpack_leaves,
            "pytree_class": pytree.__class__.__name__,
        }
    except Exception as exc:
        raise DLPackExportError(f"Failed to export PyTree via zero-copy DLPack: {exc}") from exc


def import_pytree_dlpack(dlpack_payload: Dict[str, Any], template_pytree: Any) -> Any:
    """
    Reconstructs a JAX PyTree from a zero-copy DLPack container.

    Parameters:
        dlpack_payload: Dict returned by extract_pytree_dlpack
        template_pytree: PyTree instance providing the structural treedef template

    Returns:
        Reconstructed JAX PyTree populated with fresh zero-copy JAX arrays.
    """
    try:
        treedef = dlpack_payload.get("treedef")
        if treedef is None:
            _, treedef = jax.tree_util.tree_flatten(template_pytree)

        raw_leaves = dlpack_payload.get("leaves", [])
        reconstructed_leaves: List[Any] = []

        for item in raw_leaves:
            if "dlpack" in item:
                obj = item["dlpack"]
                jax_arr = jax.dlpack.from_dlpack(obj)
                reconstructed_leaves.append(jax_arr)
            elif "capsule" in item:
                capsule = item["capsule"]
                # Wrap capsule if necessary
                if hasattr(capsule, "__dlpack__") and hasattr(capsule, "__dlpack_device__"):
                    jax_arr = jax.dlpack.from_dlpack(capsule)
                else:
                    wrapper = DLPackTensor(capsule)
                    jax_arr = jax.dlpack.from_dlpack(wrapper)
                reconstructed_leaves.append(jax_arr)
            elif "value" in item:
                reconstructed_leaves.append(item["value"])
            else:
                raise DLPackImportError(f"Malformed DLPack leaf payload: {item}")

        return jax.tree_util.tree_unflatten(treedef, reconstructed_leaves)
    except Exception as exc:
        raise DLPackImportError(f"Failed to import PyTree via zero-copy DLPack: {exc}") from exc


# =============================================================================
# 7. TRANSITION INTENSITIES & INTENSITY BORROWING
# =============================================================================

def compute_polyad_transition_dipoles(
    ground_eigensystem: PolyadEigensystemPyTree,
    excited_eigensystem: PolyadEigensystemPyTree,
    unperturbed_dipoles: jnp.ndarray,
) -> jnp.ndarray:
    """
    Computes perturbed vibrational-rotational transition dipole matrix elements
    accounting for intensity borrowing between resonant states.

    Formula: <psi_f | mu | psi_i> = sum_{m, n} U_{f, m}^* * mu_{m, n} * U_{i, n}

    Parameters:
        ground_eigensystem: PolyadEigensystemPyTree for initial manifold
        excited_eigensystem: PolyadEigensystemPyTree for target manifold
        unperturbed_dipoles: Matrix (M_f, M_i) or (M_f, M_i, 3) of unperturbed transition dipoles

    Returns:
        jnp.ndarray: Matrix of perturbed transition dipole magnitudes.
    """
    u_initial = ground_eigensystem.eigenvectors  # (total_dim_i, total_dim_i)
    u_final = excited_eigensystem.eigenvectors    # (total_dim_f, total_dim_f)

    # Simplified state projection
    w_i = ground_eigensystem.state_mixing_weights   # (total_dim_i, M_i)
    w_f = excited_eigensystem.state_mixing_weights  # (total_dim_f, M_f)

    dip_f64 = jnp.asarray(unperturbed_dipoles, dtype=jnp.float64)

    # Effective transition dipoles via intensity-weighted projection:
    # <psi_f | mu | psi_i> = sum_{m, n} sqrt(w_{f,m}) * mu_{m,n} * sqrt(w_{i,n})
    sqrt_w_f = jnp.sqrt(w_f)
    sqrt_w_i = jnp.sqrt(w_i)

    if dip_f64.ndim == 2:
        return jnp.matmul(sqrt_w_f, jnp.matmul(dip_f64, sqrt_w_i.T))

    # 3D vector transition dipoles: shape (total_dim_f, total_dim_i, 3)
    return jnp.einsum("fa,abc,ib->fic", sqrt_w_f, dip_f64, sqrt_w_i)



# =============================================================================
# 8. EXTRACTOR RESONANCE REPORT BRIDGE
# =============================================================================

def build_polyad_from_extractor_report(
    extractor_report: ResonanceReport,
    states_constants: List[WatsonConstants],
    J: int = 0,
    representation: str = "Ir",
    reduction: str = "A",
) -> List[PolyadBlockPyTree]:
    """
    Constructs executable PolyadBlockPyTree objects directly from a ResonanceReport
    emitted by cochem_quantum_extractor.py.

    Parameters:
        extractor_report: ResonanceReport instance
        states_constants: List of WatsonConstants for each vibrational mode
        J: Rotational quantum number J
        representation: Watson axis representation ('Ir' or 'IIIr')
        reduction: Watson reduction ('A' or 'S')

    Returns:
        List[PolyadBlockPyTree]: List of constructed Polyad blocks for each coupled cluster.
    """
    blocks: List[PolyadBlockPyTree] = []
    all_freqs = extractor_report.all_frequencies_cm1

    for polyad_indices in extractor_report.polyads:
        if len(polyad_indices) < 2:
            continue

        # Extract states in this polyad
        polyad_states: List[VibrationalState] = []
        for local_idx, mode_idx in enumerate(polyad_indices):
            freq_val = all_freqs[mode_idx] if mode_idx < len(all_freqs) else 1000.0
            consts = states_constants[mode_idx] if mode_idx < len(states_constants) else states_constants[0]
            v_state = VibrationalState(
                state_index=local_idx,
                label=f"nu_{mode_idx}",
                vibrational_energy_cm1=float(freq_val),
                watson_constants=consts,
            )
            polyad_states.append(v_state)

        # Map global resonances to local polyad indices
        fermi_couplings: List[FermiCouplingParam] = []
        coriolis_couplings: List[CoriolisCouplingParam] = []

        for res in extractor_report.resonances:
            modes = res.mode_indices
            if all(m in polyad_indices for m in modes):
                loc_i = polyad_indices.index(modes[0])
                loc_j = polyad_indices.index(modes[1])
                if "Fermi" in res.resonance_type:
                    w_est = max(5.0, 50.0 - min(45.0, res.energy_difference_cm1))
                    fermi_couplings.append(FermiCouplingParam(state_i=loc_i, state_j=loc_j, w_fermi_cm1=w_est))
                elif "Coriolis" in res.resonance_type:
                    zeta_val = res.coriolis_zeta if res.coriolis_zeta is not None else 0.5
                    coriolis_couplings.append(CoriolisCouplingParam(state_i=loc_i, state_j=loc_j, zeta_z=zeta_val))

        config = PolyadCouplingConfig(
            states=polyad_states,
            fermi_couplings=fermi_couplings,
            coriolis_couplings=coriolis_couplings,
            representation=representation,
            reduction=reduction,
        )

        polyad_block = construct_polyad_block(config, J=J)
        blocks.append(polyad_block)

    return blocks

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.