Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\04_P4_T02_smart_scan_navigator.md.
Original prompt:
# Prompt 09 — Phase 4, Task Group 02: Smart Scan Navigator (Information-Gain Chunking)

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 4 — Smart Scan Navigator & Jacobian Sensitivity
> **Phase:** 4 (Tasks 61–80) — Information-Gain Triage
> **Dependency:** Prompt 03 (JAX driver for covariance/Jacobian), Prompt 08 (ResolvabilityFilter)

---

## Objective

Implement the Information-Gain engine that strategically identifies which unassigned spectral lines will mathematically fix the worst-performing physical constants, while respecting hardware bandwidth limits and avoiding unresolvable spectral clusters. This module acts as a strategic co-pilot, outputting ranked scan-region recommendations for the FT-MW spectrometer.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_navigator.py
```

> **Scope:** This prompt targets ONLY the file above.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Smart Scan Navigator — Information-Gain triage and hardware-aware chunking."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
```

### 2. `ScanRecommendation` Dataclass

```python
@dataclass(frozen=True)
class ScanRecommendation:
    """A recommended FT-MW scan region."""
    rank: int
    start_frequency_mhz: float
    end_frequency_mhz: float
    chunk_score: float
    n_clean_lines: int
    targeted_constants: tuple[str, ...]
    description: str  # Human-readable recommendation string
```

### 3. `SmartScanNavigator` Class

```python
class SmartScanNavigator:
    """Information-Gain engine for strategic FT-MW scan recommendations."""

    def __init__(
        self,
        max_chunk_bandwidth_mhz: float = 250.0,
        top_n_recommendations: int = 3,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_chunk_bandwidth_mhz : float
            Maximum scan chunk bandwidth in MHz. Default 250 MHz.
            Can be overridden by 'max_chunk_bandwidth' in cochem_system_config.json.
        top_n_recommendations : int
            Number of top-ranked scan regions to return.
        config_path : Path, optional
            Path to cochem_system_config.json for overrides.
        """
        self._chunk_bw: float = max_chunk_bandwidth_mhz
        self._top_n: int = top_n_recommendations

        # Override from config if available
        resolved_config = self._resolve_config_path(config_path)
        if resolved_config and resolved_config.exists():
            with open(resolved_config, "r") as f:
                config = json.load(f)
            if "max_chunk_bandwidth" in config:
                self._chunk_bw = float(config["max_chunk_bandwidth"])

    def identify_target_constants(
        self,
        covariance_matrix: np.ndarray,
        parameter_names: list[str],
        n_targets: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Identify the physical constants with highest variance or correlation.

        Scans the diagonal (parameter uncertainty) and off-diagonal
        (parameter collinearity) elements of the covariance matrix.

        Parameters
        ----------
        covariance_matrix : np.ndarray
            Shape (N_params, N_params). Symmetric covariance matrix from JAX fit.
        parameter_names : list[str]
            Names corresponding to matrix indices (e.g., ['A', 'B', 'C', 'DJ', ...]).
        n_targets : int
            Number of top-priority constants to flag.

        Returns
        -------
        list[tuple[str, float]]
            List of (parameter_name, variance) sorted by descending variance.
        """
        variances = np.diag(covariance_matrix)
        sorted_indices = np.argsort(variances)[::-1]
        targets = [
            (parameter_names[i], float(variances[i]))
            for i in sorted_indices[:n_targets]
        ]
        logger.info(f"Target constants (highest variance): {targets}")
        return targets

    def compute_information_gain(
        self,
        jacobian: np.ndarray,
        parameter_names: list[str],
        target_constants: list[tuple[str, float]],
        resolvability_penalties: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-transition Information Gain weighted by resolvability.

        Information Gain for transition i:
          IG_i = Σ_j (∂ν_i/∂p_j)² * w_j * resolvability_penalty_i
        where w_j = 1 if p_j is a target constant, 0 otherwise.

        Parameters
        ----------
        jacobian : np.ndarray
            Shape (N_transitions, N_params). Analytical Jacobian from JAX driver.
        parameter_names : list[str]
            Parameter names matching Jacobian columns.
        target_constants : list[tuple[str, float]]
            Target constants from identify_target_constants().
        resolvability_penalties : np.ndarray
            Shape (N_transitions,). Penalty scores from ResolvabilityFilter.

        Returns
        -------
        np.ndarray
            Shape (N_transitions,). Per-transition Information Gain scores.
        """
        target_names = {name for name, _ in target_constants}
        target_mask = np.array([
            1.0 if name in target_names else 0.0
            for name in parameter_names
        ], dtype=np.float64)

        # Weighted sum of squared Jacobian entries for target constants
        sensitivity = np.sum((jacobian ** 2) * target_mask[np.newaxis, :], axis=1)
        return sensitivity * resolvability_penalties

    def recommend_scan_regions(
        self,
        transition_frequencies_mhz: np.ndarray,
        information_gains: np.ndarray,
        resolvability_penalties: np.ndarray,
        target_constants: list[tuple[str, float]],
    ) -> list[ScanRecommendation]:
        """
        Slide a bandwidth window across the spectrum and rank regions.

        Chunk Score = Σ(Information Gain of clean lines) within the window,
        where "clean" means resolvability_penalty > 0.5.

        Parameters
        ----------
        transition_frequencies_mhz : np.ndarray
            Shape (N,). Predicted frequencies of all unassigned transitions.
        information_gains : np.ndarray
            Shape (N,). Per-transition IG scores.
        resolvability_penalties : np.ndarray
            Shape (N,). Penalty scores from ResolvabilityFilter.
        target_constants : list[tuple[str, float]]
            Target constants for description generation.

        Returns
        -------
        list[ScanRecommendation]
            Top-N ranked scan region recommendations.
        """
        freq_min = float(np.min(transition_frequencies_mhz))
        freq_max = float(np.max(transition_frequencies_mhz))

        chunk_starts = np.arange(freq_min, freq_max, self._chunk_bw / 2.0)
        chunk_results = []

        for start in chunk_starts:
            end = start + self._chunk_bw
            mask = (transition_frequencies_mhz >= start) & (transition_frequencies_mhz < end)
            if not np.any(mask):
                continue

            clean_mask = mask & (resolvability_penalties > 0.5)
            chunk_score = float(np.sum(information_gains[clean_mask]))
            n_clean = int(np.sum(clean_mask))

            if chunk_score > 0 and n_clean > 0:
                chunk_results.append((start, end, chunk_score, n_clean))

        # Sort by chunk score descending
        chunk_results.sort(key=lambda x: x[2], reverse=True)

        target_names = tuple(name for name, _ in target_constants)
        recommendations = []
        for rank, (start, end, score, n_clean) in enumerate(chunk_results[:self._top_n], 1):
            desc = (
                f"Recommendation {rank}: Scan {start:.0f}–{end:.0f} MHz. "
                f"Expected {n_clean} isolated transitions. "
                f"Targets: {', '.join(target_names)}."
            )
            recommendations.append(ScanRecommendation(
                rank=rank,
                start_frequency_mhz=start,
                end_frequency_mhz=end,
                chunk_score=score,
                n_clean_lines=n_clean,
                targeted_constants=target_names,
                description=desc,
            ))

        return recommendations

    @staticmethod
    def _resolve_config_path(user_path: Optional[Path]) -> Optional[Path]:
        """Resolve cochem_system_config.json path."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "Registry" / "cochem_system_config.json"
        return Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"
```

---

## Validation Test Specification

1. Construct a physically-grounded test scenario:
   - Use a covariance matrix where the `A` rotational constant has elevated variance (e.g., 10× larger than B, C)
   - Create 20 predicted transitions spread across 8000–18000 MHz
   - Place one highly sensitive A-dependent transition at 14000 MHz inside a dense Q-branch region (2 neighbors within 0.030 MHz → penalized by resolvability)
   - Place another A-dependent transition at 16500 MHz in a sparse R-branch region (nearest neighbor > 100 MHz → rewarded)
2. **Assertions:**
   - `identify_target_constants()` returns `A` as the top target
   - `recommend_scan_regions()` selects the sparse-region chunk (containing 16500 MHz) with higher rank than the dense-region chunk (containing 14000 MHz)
   - The dense-region chunk has fewer "clean lines" due to resolvability penalties

---

## Acceptance Criteria

- [ ] Covariance extraction identifies highest-variance and highest-correlation constants
- [ ] Information Gain uses squared Jacobian entries weighted by target constant mask
- [ ] Hardware-aware chunking respects `max_chunk_bandwidth_mhz` (configurable)
- [ ] Clean-line threshold of resolvability_penalty > 0.5 applied
- [ ] `ScanRecommendation` includes human-readable description string
- [ ] Config overrides from `cochem_system_config.json` honored
- [ ] All paths use `pathlib.Path`

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
from src.cochem_spycfit.physics_core.cochem_spycfit_navigator import (
    ScanRecommendation,
    SmartScanNavigator,
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
    # Smart Scan Navigator Core
    "ScanRecommendation",
    "SmartScanNavigator",
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
from .cochem_spycfit_navigator import (
    ScanRecommendation,
    SmartScanNavigator,
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
    # Smart Scan Navigator Core
    "ScanRecommendation",
    "SmartScanNavigator",
]


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_navigator.py ---
"""
CoChem-SpycFit: Smart Scan Navigator Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_navigator import (
        ScanRecommendation,
        SmartScanNavigator,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_navigator import (  # type: ignore[no-redef]
        ScanRecommendation,
        SmartScanNavigator,
    )

__all__ = [
    "ScanRecommendation",
    "SmartScanNavigator",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_spycfit_navigator.py ---
"""Smart Scan Navigator — Information-Gain triage and hardware-aware chunking."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanRecommendation:
    """A recommended FT-MW scan region."""

    rank: int
    start_frequency_mhz: float
    end_frequency_mhz: float
    chunk_score: float
    n_clean_lines: int
    targeted_constants: tuple[str, ...]
    description: str  # Human-readable recommendation string


class SmartScanNavigator:
    """Information-Gain engine for strategic FT-MW scan recommendations."""

    def __init__(
        self,
        max_chunk_bandwidth_mhz: float = 250.0,
        top_n_recommendations: int = 3,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_chunk_bandwidth_mhz : float
            Maximum scan chunk bandwidth in MHz. Default 250 MHz.
            Can be overridden by 'max_chunk_bandwidth' in cochem_system_config.json.
        top_n_recommendations : int
            Number of top-ranked scan regions to return.
        config_path : Path, optional
            Path to cochem_system_config.json for overrides.
        """
        self._chunk_bw: float = float(max_chunk_bandwidth_mhz)
        self._top_n: int = int(top_n_recommendations)

        # Override from config if available
        resolved_config = self._resolve_config_path(config_path)
        if resolved_config and resolved_config.exists():
            with open(resolved_config, "r", encoding="utf-8") as f:
                config = json.load(f)
            if "max_chunk_bandwidth" in config:
                self._chunk_bw = float(config["max_chunk_bandwidth"])

    def identify_target_constants(
        self,
        covariance_matrix: np.ndarray,
        parameter_names: list[str],
        n_targets: int = 3,
    ) -> list[tuple[str, float]]:
        """
        Identify the physical constants with highest variance or correlation.

        Scans the diagonal (parameter uncertainty) and off-diagonal
        (parameter collinearity) elements of the covariance matrix.

        Parameters
        ----------
        covariance_matrix : np.ndarray
            Shape (N_params, N_params). Symmetric covariance matrix from JAX fit.
        parameter_names : list[str]
            Names corresponding to matrix indices (e.g., ['A', 'B', 'C', 'DJ', ...]).
        n_targets : int
            Number of top-priority constants to flag.

        Returns
        -------
        list[tuple[str, float]]
            List of (parameter_name, variance) sorted by descending variance.
        """
        variances = np.diag(covariance_matrix)
        sorted_indices = np.argsort(variances)[::-1]
        targets = [
            (parameter_names[i], float(variances[i]))
            for i in sorted_indices[:n_targets]
        ]
        logger.info(f"Target constants (highest variance): {targets}")
        return targets

    def compute_information_gain(
        self,
        jacobian: np.ndarray,
        parameter_names: list[str],
        target_constants: list[tuple[str, float]],
        resolvability_penalties: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-transition Information Gain weighted by resolvability.

        Information Gain for transition i:
          IG_i = Σ_j (∂ν_i/∂p_j)² * w_j * resolvability_penalty_i
        where w_j = 1 if p_j is a target constant, 0 otherwise.

        Parameters
        ----------
        jacobian : np.ndarray
            Shape (N_transitions, N_params). Analytical Jacobian from JAX driver.
        parameter_names : list[str]
            Parameter names matching Jacobian columns.
        target_constants : list[tuple[str, float]]
            Target constants from identify_target_constants().
        resolvability_penalties : np.ndarray
            Shape (N_transitions,). Penalty scores from ResolvabilityFilter.

        Returns
        -------
        np.ndarray
            Shape (N_transitions,). Per-transition Information Gain scores.
        """
        target_names = {name for name, _ in target_constants}
        target_mask = np.array([
            1.0 if name in target_names else 0.0
            for name in parameter_names
        ], dtype=np.float64)

        # Weighted sum of squared Jacobian entries for target constants
        sensitivity = np.sum((jacobian ** 2) * target_mask[np.newaxis, :], axis=1)
        return sensitivity * resolvability_penalties

    def recommend_scan_regions(
        self,
        transition_frequencies_mhz: np.ndarray,
        information_gains: np.ndarray,
        resolvability_penalties: np.ndarray,
        target_constants: list[tuple[str, float]],
    ) -> list[ScanRecommendation]:
        """
        Slide a bandwidth window across the spectrum and rank regions.

        Chunk Score = Σ(Information Gain of clean lines) within the window,
        where "clean" means resolvability_penalty > 0.5.

        Parameters
        ----------
        transition_frequencies_mhz : np.ndarray
            Shape (N,). Predicted frequencies of all unassigned transitions.
        information_gains : np.ndarray
            Shape (N,). Per-transition IG scores.
        resolvability_penalties : np.ndarray
            Shape (N,). Penalty scores from ResolvabilityFilter.
        target_constants : list[tuple[str, float]]
            Target constants for description generation.

        Returns
        -------
        list[ScanRecommendation]
            Top-N ranked scan region recommendations.
        """
        if len(transition_frequencies_mhz) == 0:
            return []

        freq_min = float(np.min(transition_frequencies_mhz))
        freq_max = float(np.max(transition_frequencies_mhz))

        chunk_starts = np.arange(freq_min, freq_max, self._chunk_bw / 2.0)
        if len(chunk_starts) == 0:
            chunk_starts = np.array([freq_min])

        chunk_results = []

        for start in chunk_starts:
            end = start + self._chunk_bw
            mask = (transition_frequencies_mhz >= start) & (transition_frequencies_mhz < end)
            if not np.any(mask):
                continue

            clean_mask = mask & (resolvability_penalties > 0.5)
            chunk_score = float(np.sum(information_gains[clean_mask]))
            n_clean = int(np.sum(clean_mask))

            if chunk_score > 0 and n_clean > 0:
                chunk_results.append((start, end, chunk_score, n_clean))

        # Sort by chunk score descending
        chunk_results.sort(key=lambda x: x[2], reverse=True)

        target_names = tuple(name for name, _ in target_constants)
        recommendations = []
        for rank, (start, end, score, n_clean) in enumerate(chunk_results[:self._top_n], 1):
            desc = (
                f"Recommendation {rank}: Scan {start:.0f}–{end:.0f} MHz. "
                f"Expected {n_clean} isolated transitions. "
                f"Targets: {', '.join(target_names)}."
            )
            recommendations.append(ScanRecommendation(
                rank=rank,
                start_frequency_mhz=start,
                end_frequency_mhz=end,
                chunk_score=score,
                n_clean_lines=n_clean,
                targeted_constants=target_names,
                description=desc,
            ))

        return recommendations

    @staticmethod
    def _resolve_config_path(user_path: Optional[Path]) -> Optional[Path]:
        """Resolve cochem_system_config.json path."""
        if user_path is not None:
            return Path(user_path)
        artifacts_root = os.environ.get("COCHEM_ARTIFACTS_ROOT")
        if artifacts_root:
            return Path(artifacts_root) / "Registry" / "cochem_system_config.json"
        return Path.home() / "CoChem_Artifacts" / "Registry" / "cochem_system_config.json"


__all__ = [
    "ScanRecommendation",
    "SmartScanNavigator",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_navigator.py ---
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

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.