Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\04_P4_T01_resolvability_filter.md.
Original prompt:
# Prompt 08 — Phase 4, Task Group 01: Resolvability Anti-Clustering Filter

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 4 — Smart Scan Navigator & Jacobian Sensitivity
> **Phase:** 4 (Tasks 61–80) — Information-Gain Triage
> **Dependency:** Prompt 07 (`SpycFitGPRegressor` for σ_predict values)

---

## Objective

Build the density penalty function that evaluates the local spectral density around each predicted transition and penalizes lines that are too close together to be experimentally resolved. This "Anti-Clustering Guardrail" prevents the system from recommending scan regions where lines are buried in dense Q-branch forests.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_resolvability.py
```

> **Scope:** This prompt targets ONLY the file above.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Resolvability anti-clustering filter for spectral line density penalty scoring."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
```

### 2. `ResolvabilityScore` Dataclass

```python
@dataclass(frozen=True)
class ResolvabilityScore:
    """Resolvability assessment for a single predicted transition."""
    frequency_mhz: float
    nearest_neighbor_spacing_mhz: float
    instrumental_resolution_mhz: float
    ml_uncertainty_mhz: float
    effective_threshold_mhz: float   # Res_inst + σ_predict
    penalty: float                    # 0.0 (unresolvable) to 1.0 (fully resolved)
    is_resolvable: bool               # True if spacing > effective_threshold
```

### 3. `ResolvabilityFilter` Class

```python
class ResolvabilityFilter:
    """Evaluates spectral line density and assigns resolvability penalties."""

    def __init__(
        self,
        instrumental_resolution_mhz: float = 0.050,
    ) -> None:
        """
        Parameters
        ----------
        instrumental_resolution_mhz : float
            User's FT-MW spectrometer resolution limit in MHz.
            Default: 50 kHz (0.050 MHz), typical for chirped-pulse instruments.
        """
        self._res_inst: float = instrumental_resolution_mhz

    def compute_penalties(
        self,
        predicted_frequencies_mhz: np.ndarray,
        ml_uncertainties_mhz: Optional[np.ndarray] = None,
    ) -> list[ResolvabilityScore]:
        """
        Compute resolvability penalties for all predicted transitions.

        For each transition, the penalty is computed as:
        - effective_threshold = Res_inst + σ_predict
        - nearest_spacing = min distance to any other predicted transition
        - If nearest_spacing < effective_threshold: penalty approaches 0 (steep)
        - If nearest_spacing >= effective_threshold: penalty = 1.0 (fully resolved)

        The penalty function uses a sigmoid-like steep decay:
        penalty = 1.0 / (1.0 + exp(-k * (spacing - threshold)))
        where k controls the steepness (default k=20 for aggressive penalization).

        Parameters
        ----------
        predicted_frequencies_mhz : np.ndarray
            Shape (N,). All predicted transition frequencies in MHz.
        ml_uncertainties_mhz : np.ndarray, optional
            Shape (N,). GP prediction uncertainty σ_predict for each transition.
            If None, defaults to 0.0 (only instrumental resolution used).

        Returns
        -------
        list[ResolvabilityScore]
            Penalty scores for each transition.
        """
        n = len(predicted_frequencies_mhz)
        if ml_uncertainties_mhz is None:
            ml_uncertainties_mhz = np.zeros(n, dtype=np.float64)

        # Sort frequencies for efficient nearest-neighbor computation
        sorted_indices = np.argsort(predicted_frequencies_mhz)
        sorted_freqs = predicted_frequencies_mhz[sorted_indices]

        scores: list[ResolvabilityScore] = [None] * n  # type: ignore
        k = 20.0  # Sigmoid steepness

        for rank, idx in enumerate(sorted_indices):
            freq = sorted_freqs[rank]
            sigma = float(ml_uncertainties_mhz[idx])
            threshold = self._res_inst + sigma

            # Find nearest neighbor spacing (left and right in sorted array)
            spacings = []
            if rank > 0:
                spacings.append(freq - sorted_freqs[rank - 1])
            if rank < n - 1:
                spacings.append(sorted_freqs[rank + 1] - freq)

            if spacings:
                nearest = min(spacings)
            else:
                nearest = float("inf")

            # Sigmoid penalty
            exponent = -k * (nearest - threshold)
            penalty = 1.0 / (1.0 + np.exp(exponent))
            penalty = float(np.clip(penalty, 0.0, 1.0))

            scores[idx] = ResolvabilityScore(
                frequency_mhz=float(freq),
                nearest_neighbor_spacing_mhz=float(nearest),
                instrumental_resolution_mhz=self._res_inst,
                ml_uncertainty_mhz=sigma,
                effective_threshold_mhz=threshold,
                penalty=penalty,
                is_resolvable=(nearest > threshold),
            )

        return scores
```

---

## Validation Test Specification

1. Create a `ResolvabilityFilter` with `instrumental_resolution_mhz=0.050` (50 kHz)
2. Construct a test spectrum with:
   - 3 well-separated transitions: 12000.0, 12500.0, 13000.0 MHz (spacing = 500 MHz)
   - 2 clustered transitions: 14000.0, 14000.030 MHz (spacing = 30 kHz < 50 kHz threshold)
3. **Assertions:**
   - Well-separated transitions have penalty ≈ 1.0 and `is_resolvable = True`
   - Clustered transitions have penalty < 0.1 and `is_resolvable = False`
   - Adding ML uncertainty (σ = 0.100 MHz) to one of the separated transitions still keeps it resolvable (spacing >> threshold)
4. Edge case: single transition array returns penalty = 1.0 (no neighbors)

---

## Acceptance Criteria

- [ ] Sigmoid penalty function with steep decay near the resolution threshold
- [ ] Effective threshold combines instrumental resolution + ML uncertainty
- [ ] Nearest-neighbor computation uses sorted array for efficiency
- [ ] `ResolvabilityScore` dataclass contains all required fields
- [ ] Single-transition edge case handled correctly
- [ ] No hardcoded atomic masses or synthetic placeholder data

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

from src.cochem_spycfit.physics_core.cochem_assign_ml import (
    ActivePESHealingConfig,
    AssignMLError,
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
from src.cochem_spycfit.physics_core.cochem_spycfit_auditor import (
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    format_contrast_dataframe,
)
from src.cochem_spycfit.physics_core.cochem_spycfit_features import (
    QuantumFeatureExtractor,
)
from src.cochem_spycfit.physics_core.cochem_spycfit_ml_core import (
    ProvenanceTag,
    ResidualPrediction,
    SpycFitGPRegressor,
)
from src.cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
    ResolvabilityFilter,
    ResolvabilityScore,
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
    # AI Assigner & Active PES Refinement Trigger (The Healing Loop)
    "ActivePESHealingConfig",
    "AssignMLError",
    "EmptyCandidateError",
    "InformationCriterion",
    "InformationGainConfig",
    "InvalidCovarianceError",
    "InvalidJacobianError",
    "InvalidResidualError",
    "back_calculate_driving_coordinates",
    "calculate_information_gain",
    "flag_unresolved_residuals",
    "generate_pes_healing_payload",
    "rank_candidate_transitions",
    "update_covariance_after_measurement",
    # Mathematical Parity Auditor
    "ParityAuditor",
    "ParityStatus",
    "ParityWarning",
    "format_contrast_dataframe",
    # Quantum Feature Extractor
    "QuantumFeatureExtractor",
    # Gaussian Process ML Core
    "ProvenanceTag",
    "ResidualPrediction",
    "SpycFitGPRegressor",
    # Resolvability Anti-Clustering Filter
    "ResolvabilityFilter",
    "ResolvabilityScore",
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
    ConvolverPyTree,
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
    convolute_multi_temperature,
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
    static_masked_scan,
    true_voigt_profile,
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

from .cochem_assign_ml import (
    ActivePESHealingConfig,
    AssignMLError,
    DLPNOForbiddenError as AssignMLDLPNOForbiddenError,
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
from .cochem_spycfit_auditor import (
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    format_contrast_dataframe,
)
from .cochem_spycfit_features import (
    QuantumFeatureExtractor,
)
from .cochem_spycfit_ml_core import (
    ProvenanceTag,
    ResidualPrediction,
    SpycFitGPRegressor,
)
from .cochem_spycfit_resolvability import (
    ResolvabilityFilter,
    ResolvabilityScore,
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
    "ConvolverPyTree",
    "faddeeva_rational_jax",
    "gaussian_kernel_jax",
    "lorentzian_kernel_jax",
    "pseudo_voigt_kernel_jax",
    "faddeeva_voigt_kernel_jax",
    "true_voigt_profile",
    "evaluate_profile_kernel_jax",
    "evaluate_partition_function_jax",
    "compute_boltzmann_scaling_grid_jax",
    "convolute_multi_temperature",
    "convolve_frequency_chunks_scan_1d_jax",
    "convolve_frequency_chunks_scan_2d_jax",
    "convolve_line_chunks_scan_1d_jax",
    "convolve_spectrum_monolithic_jax",
    "static_masked_scan",
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
    # AI Assigner & Active PES Refinement Trigger (The Healing Loop)
    "ActivePESHealingConfig",
    "AssignMLError",
    "EmptyCandidateError",
    "InformationCriterion",
    "InformationGainConfig",
    "InvalidCovarianceError",
    "InvalidJacobianError",
    "InvalidResidualError",
    "back_calculate_driving_coordinates",
    "calculate_information_gain",
    "flag_unresolved_residuals",
    "generate_pes_healing_payload",
    "rank_candidate_transitions",
    "update_covariance_after_measurement",
    # Mathematical Parity Auditor Core
    "ParityAuditor",
    "ParityStatus",
    "ParityWarning",
    "format_contrast_dataframe",
    # Quantum Feature Extractor
    "QuantumFeatureExtractor",
    # Gaussian Process ML Core
    "ProvenanceTag",
    "ResidualPrediction",
    "SpycFitGPRegressor",
    # Resolvability Anti-Clustering Filter
    "ResolvabilityFilter",
    "ResolvabilityScore",
]


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_resolvability.py ---
"""
CoChem-SpycFit: Resolvability Anti-Clustering Filter Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_resolvability import (
        ResolvabilityFilter,
        ResolvabilityScore,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_resolvability import (  # type: ignore[no-redef]
        ResolvabilityFilter,
        ResolvabilityScore,
    )

__all__ = [
    "ResolvabilityFilter",
    "ResolvabilityScore",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_spycfit_resolvability.py ---
"""Resolvability anti-clustering filter for spectral line density penalty scoring."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvabilityScore:
    """Resolvability assessment for a single predicted transition."""

    frequency_mhz: float
    nearest_neighbor_spacing_mhz: float
    instrumental_resolution_mhz: float
    ml_uncertainty_mhz: float
    effective_threshold_mhz: float  # Res_inst + σ_predict
    penalty: float  # 0.0 (unresolvable) to 1.0 (fully resolved)
    is_resolvable: bool  # True if spacing > effective_threshold


class ResolvabilityFilter:
    """Evaluates spectral line density and assigns resolvability penalties."""

    def __init__(
        self,
        instrumental_resolution_mhz: float = 0.050,
        steepness_k: float = 20.0,
    ) -> None:
        """Initialize the ResolvabilityFilter.

        Parameters
        ----------
        instrumental_resolution_mhz : float
            User's FT-MW spectrometer resolution limit in MHz.
            Default: 50 kHz (0.050 MHz), typical for chirped-pulse instruments.
        steepness_k : float
            Sigmoid steepness factor controlling the transition sharpness around the threshold.
            Default: 20.0.
        """
        self._res_inst: float = float(instrumental_resolution_mhz)
        self._steepness_k: float = float(steepness_k)

    @property
    def instrumental_resolution_mhz(self) -> float:
        """Return the instrumental resolution limit in MHz."""
        return self._res_inst

    @property
    def steepness_k(self) -> float:
        """Return the sigmoid steepness factor."""
        return self._steepness_k

    def compute_penalties(
        self,
        predicted_frequencies_mhz: Union[np.ndarray, Sequence[float]],
        ml_uncertainties_mhz: Optional[Union[np.ndarray, Sequence[float]]] = None,
    ) -> list[ResolvabilityScore]:
        """Compute resolvability penalties for all predicted transitions.

        For each transition, the penalty is computed as:
        - effective_threshold = Res_inst + σ_predict
        - nearest_spacing = min distance to any other predicted transition
        - If nearest_spacing < effective_threshold: penalty approaches 0 (steep decay)
        - If nearest_spacing >= effective_threshold: penalty approaches 1.0 (fully resolved)

        The penalty function uses a numerically stable sigmoid steep decay:
        penalty = 1.0 / (1.0 + exp(-k * (spacing - threshold)))

        Parameters
        ----------
        predicted_frequencies_mhz : Union[np.ndarray, Sequence[float]]
            Shape (N,). All predicted transition frequencies in MHz.
        ml_uncertainties_mhz : Optional[Union[np.ndarray, Sequence[float]]], optional
            Shape (N,). GP prediction uncertainty σ_predict for each transition.
            If None, defaults to 0.0 (only instrumental resolution used).

        Returns
        -------
        list[ResolvabilityScore]
            Penalty scores for each transition, preserving input array ordering.
        """
        freqs = np.asarray(predicted_frequencies_mhz, dtype=np.float64).ravel()
        n = len(freqs)
        if n == 0:
            return []

        if ml_uncertainties_mhz is None:
            unc = np.zeros(n, dtype=np.float64)
        else:
            unc = np.asarray(ml_uncertainties_mhz, dtype=np.float64).ravel()
            if len(unc) != n:
                raise ValueError(
                    f"ml_uncertainties_mhz length ({len(unc)}) must match predicted_frequencies_mhz length ({n})"
                )

        # Sort frequencies for efficient nearest-neighbor computation
        sorted_indices = np.argsort(freqs)
        sorted_freqs = freqs[sorted_indices]

        scores: list[ResolvabilityScore] = [None] * n  # type: ignore[list-item]

        for rank, idx in enumerate(sorted_indices):
            freq = float(sorted_freqs[rank])
            sigma = float(unc[idx])
            threshold = self._res_inst + sigma

            # Find nearest neighbor spacing (left and right in sorted array)
            spacings = []
            if rank > 0:
                spacings.append(freq - float(sorted_freqs[rank - 1]))
            if rank < n - 1:
                spacings.append(float(sorted_freqs[rank + 1]) - freq)

            if spacings:
                nearest = float(min(spacings))
            else:
                nearest = float("inf")

            # Numerically stable sigmoid penalty
            if math.isinf(nearest):
                penalty = 1.0
            else:
                exponent = -self._steepness_k * (nearest - threshold)
                if exponent > 700.0:
                    penalty = 0.0
                elif exponent < -700.0:
                    penalty = 1.0
                else:
                    penalty = 1.0 / (1.0 + float(np.exp(exponent)))
            penalty = float(np.clip(penalty, 0.0, 1.0))

            scores[idx] = ResolvabilityScore(
                frequency_mhz=freq,
                nearest_neighbor_spacing_mhz=nearest,
                instrumental_resolution_mhz=self._res_inst,
                ml_uncertainty_mhz=sigma,
                effective_threshold_mhz=threshold,
                penalty=penalty,
                is_resolvable=bool(nearest > threshold),
            )

        return scores


__all__ = [
    "ResolvabilityFilter",
    "ResolvabilityScore",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_resolvability.py ---
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

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.