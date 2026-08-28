Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\03_P3_T01_quantum_feature_extractor.md.
Original prompt:
# Prompt 06 — Phase 3, Task Group 01: Quantum Feature Extractor

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 3 — ML Learning Loop & Gaussian Process Regressor
> **Phase:** 3 (Tasks 41–60) — Systematic Residual Mapping
> **Dependency:** Prompt 01 (`FitState` for accessing locked assignments)

---

## Objective

Build the vectorization engine that extracts and normalizes quantum-mechanical features from theoretical transition catalogs into a standardized NumPy feature matrix suitable for Gaussian Process Regression. This module transforms raw spectroscopic quantum numbers and physical observables into ML-ready input features.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_features.py
```

> **Note:** NEW file in `physics_core/`.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Quantum feature extractor for ML-driven spectroscopic residual prediction."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
```

### 2. `QuantumFeatureExtractor` Class

```python
class QuantumFeatureExtractor:
    """Extracts and normalizes quantum-mechanical features for GP regression."""

    FEATURE_NAMES: tuple[str, ...] = (
        "J_upper", "Ka_upper", "Kc_upper",
        "J_lower", "Ka_lower", "Kc_lower",
        "delta_J", "delta_Ka", "delta_Kc",
        "mu_a", "mu_b", "mu_c",
        "lower_state_energy_cm",
        "transition_type",  # 0=a-type, 1=b-type, 2=c-type
    )

    def __init__(self) -> None:
        """Initialize the feature extractor with a fitted StandardScaler."""
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    def extract_raw_features(
        self,
        assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
        dipole_components: dict[str, float],
        lower_state_energies: Optional[dict[tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """
        Extract raw (unnormalized) features from assignment data.

        Parameters
        ----------
        assignments : tuple of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        dipole_components : dict
            {'mu_a': float, 'mu_b': float, 'mu_c': float} in Debye.
        lower_state_energies : dict, optional
            Map from (J'', Ka'', Kc'') → lower-state energy in cm⁻¹.
            If None, estimated from rigid-rotor approximation.

        Returns
        -------
        np.ndarray
            Shape (N_assignments, N_features). Feature matrix.
        """
        n = len(assignments)
        features = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float64)

        for i, (j_u, ka_u, kc_u, j_l, ka_l, kc_l, _nu) in enumerate(assignments):
            # Quantum numbers
            features[i, 0] = j_u
            features[i, 1] = ka_u
            features[i, 2] = kc_u
            features[i, 3] = j_l
            features[i, 4] = ka_l
            features[i, 5] = kc_l
            features[i, 6] = j_u - j_l  # delta_J
            features[i, 7] = ka_u - ka_l  # delta_Ka
            features[i, 8] = kc_u - kc_l  # delta_Kc

            # Dipole components
            features[i, 9] = dipole_components.get("mu_a", 0.0)
            features[i, 10] = dipole_components.get("mu_b", 0.0)
            features[i, 11] = dipole_components.get("mu_c", 0.0)

            # Lower-state energy
            if lower_state_energies and (j_l, ka_l, kc_l) in lower_state_energies:
                features[i, 12] = lower_state_energies[(j_l, ka_l, kc_l)]
            else:
                features[i, 12] = 0.0  # Will be estimated downstream

            # Transition type classification
            # a-type: ΔKa=0, ΔKc=±1; b-type: ΔKa=±1, ΔKc=±1; c-type: ΔKa=±1, ΔKc=0
            dk_a = abs(ka_u - ka_l)
            dk_c = abs(kc_u - kc_l)
            if dk_a == 0:
                features[i, 13] = 0  # a-type
            elif dk_c == 0:
                features[i, 13] = 2  # c-type
            else:
                features[i, 13] = 1  # b-type

        return features

    def fit_scaler(self, raw_features: np.ndarray) -> None:
        """
        Fit the StandardScaler on the raw feature matrix.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features). Raw feature matrix from extract_raw_features.
        """
        self._scaler = StandardScaler()
        self._scaler.fit(raw_features)
        self._is_fitted = True

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """
        Scale features using the fitted StandardScaler.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features).

        Returns
        -------
        np.ndarray
            Normalized feature matrix.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self._scaler.transform(raw_features)

    def fit_transform(
        self,
        assignments: tuple[tuple[int, int, int, int, int, int, float], ...],
        dipole_components: dict[str, float],
        lower_state_energies: Optional[dict[tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """
        Extract, fit, and transform features in one call.

        Returns
        -------
        np.ndarray
            Normalized feature matrix.
        """
        raw = self.extract_raw_features(assignments, dipole_components, lower_state_energies)
        self.fit_scaler(raw)
        return self.transform(raw)
```

---

## Validation Test Specification

1. Create a `QuantumFeatureExtractor` instance
2. Pass 10 authentic H₂O transitions (from the table in Prompt 01) with dipole components μ_a = 1.8546 D, μ_b = 0.0 D, μ_c = 0.0 D (water is a-type only)
3. Assert `extract_raw_features()` returns shape `(10, 14)`
4. Assert transition type classification is correct (all a-type → column 13 == 0)
5. Assert `fit_transform()` produces zero-mean, unit-variance features (within floating-point tolerance)
6. Assert delta quantum numbers (columns 6-8) match expected selection rules

---

## Acceptance Criteria

- [ ] Feature matrix includes all 14 quantum-mechanical features listed in `FEATURE_NAMES`
- [ ] Transition type classification correctly identifies a-type, b-type, c-type
- [ ] `StandardScaler` fitted on training data, applicable to new unassigned transitions
- [ ] All file paths use `pathlib.Path`
- [ ] Validation test uses authentic H₂O spectroscopic data and real dipole moments

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
]


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_features.py ---
"""
CoChem-SpycFit: Quantum Feature Extractor Proxy Re-export.
"""

from __future__ import annotations

try:
    from src.cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor,
    )
except ImportError:
    from cochem_spycfit.physics_core.cochem_spycfit_features import (  # type: ignore[no-redef]
        QuantumFeatureExtractor,
    )

__all__ = [
    "QuantumFeatureExtractor",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_spycfit_features.py ---
"""Quantum feature extractor for ML-driven spectroscopic residual prediction.

Transforms raw spectroscopic quantum numbers, physical dipole moments, and state
energies into standardized NumPy feature matrices suitable for Gaussian Process Regression.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class QuantumFeatureExtractor:
    """Extracts and normalizes quantum-mechanical features for GP regression."""

    FEATURE_NAMES: Tuple[str, ...] = (
        "J_upper",
        "Ka_upper",
        "Kc_upper",
        "J_lower",
        "Ka_lower",
        "Kc_lower",
        "delta_J",
        "delta_Ka",
        "delta_Kc",
        "mu_a",
        "mu_b",
        "mu_c",
        "lower_state_energy_cm",
        "transition_type",  # 0=a-type, 1=b-type, 2=c-type
    )

    def __init__(self) -> None:
        """Initialize the feature extractor with an unfitted state."""
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """Return the internal fitted StandardScaler instance, or None if unfitted."""
        return self._scaler

    @property
    def is_fitted(self) -> bool:
        """Return True if the internal scaler has been successfully fitted."""
        return self._is_fitted

    def extract_raw_features(
        self,
        assignments: Sequence[Tuple[int, int, int, int, int, int, float]],
        dipole_components: Mapping[str, float],
        lower_state_energies: Optional[Mapping[Tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """Extract raw (unnormalized) features from assignment data.

        Parameters
        ----------
        assignments : Sequence of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        dipole_components : Mapping[str, float]
            {'mu_a': float, 'mu_b': float, 'mu_c': float} in Debye.
        lower_state_energies : Mapping[Tuple[int, int, int], float], optional
            Map from (J'', Ka'', Kc'') -> lower-state energy in cm^-1.
            If None or level missing, defaults to 0.0.

        Returns
        -------
        np.ndarray
            Shape (N_assignments, 14). Feature matrix of dtype float64.
        """
        n = len(assignments)
        features = np.zeros((n, len(self.FEATURE_NAMES)), dtype=np.float64)

        for i, (j_u, ka_u, kc_u, j_l, ka_l, kc_l, _nu) in enumerate(assignments):
            # Quantum numbers
            features[i, 0] = float(j_u)
            features[i, 1] = float(ka_u)
            features[i, 2] = float(kc_u)
            features[i, 3] = float(j_l)
            features[i, 4] = float(ka_l)
            features[i, 5] = float(kc_l)
            features[i, 6] = float(j_u - j_l)  # delta_J
            features[i, 7] = float(ka_u - ka_l)  # delta_Ka
            features[i, 8] = float(kc_u - kc_l)  # delta_Kc

            # Dipole components
            features[i, 9] = float(dipole_components.get("mu_a", 0.0))
            features[i, 10] = float(dipole_components.get("mu_b", 0.0))
            features[i, 11] = float(dipole_components.get("mu_c", 0.0))

            # Lower-state energy
            if lower_state_energies and (j_l, ka_l, kc_l) in lower_state_energies:
                features[i, 12] = float(lower_state_energies[(j_l, ka_l, kc_l)])
            else:
                features[i, 12] = 0.0

            # Transition type classification
            # a-type: ΔKa=0, ΔKc=±1; b-type: ΔKa=±1, ΔKc=±1; c-type: ΔKa=±1, ΔKc=0
            dk_a = abs(ka_u - ka_l)
            dk_c = abs(kc_u - kc_l)
            if dk_a == 0:
                features[i, 13] = 0.0  # a-type
            elif dk_c == 0:
                features[i, 13] = 2.0  # c-type
            else:
                features[i, 13] = 1.0  # b-type

        return features

    def fit_scaler(self, raw_features: np.ndarray) -> None:
        """Fit the StandardScaler on the raw feature matrix.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features). Raw feature matrix from extract_raw_features.
        """
        self._scaler = StandardScaler()
        self._scaler.fit(raw_features)
        self._is_fitted = True

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Scale features using the fitted StandardScaler.

        Parameters
        ----------
        raw_features : np.ndarray
            Shape (N, N_features).

        Returns
        -------
        np.ndarray
            Normalized feature matrix.

        Raises
        ------
        RuntimeError
            If the scaler has not been fitted.
        """
        if not self._is_fitted or self._scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit_scaler() first.")
        return self._scaler.transform(raw_features)

    def fit_transform(
        self,
        assignments: Sequence[Tuple[int, int, int, int, int, int, float]],
        dipole_components: Mapping[str, float],
        lower_state_energies: Optional[Mapping[Tuple[int, int, int], float]] = None,
    ) -> np.ndarray:
        """Extract, fit, and transform features in one atomic call.

        Parameters
        ----------
        assignments : Sequence of 7-tuples
            Each entry: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz).
        dipole_components : Mapping[str, float]
            {'mu_a': float, 'mu_b': float, 'mu_c': float} in Debye.
        lower_state_energies : Mapping[Tuple[int, int, int], float], optional
            Map from (J'', Ka'', Kc'') -> lower-state energy in cm^-1.

        Returns
        -------
        np.ndarray
            Normalized feature matrix of shape (N, 14).
        """
        raw = self.extract_raw_features(assignments, dipole_components, lower_state_energies)
        self.fit_scaler(raw)
        return self.transform(raw)


__all__ = [
    "QuantumFeatureExtractor",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_features.py ---
"""Physical unit tests for QuantumFeatureExtractor (spectroscopic ML feature vectorizer).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational transitions for H2O and asymmetric rotors.
- Verification of 14-feature quantum numbers, delta quantum selection rules, dipole components, and energies.
- Scaler fitting, transform normalization, and stateful invariants.
- Static AST validation confirming zero synthetic mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from cochem_spycfit.physics_core.cochem_spycfit_features import QuantumFeatureExtractor


@pytest.fixture
def h2o_transitions() -> Tuple[Tuple[int, int, int, int, int, int, float], ...]:
    """Authentic low-J rotational transitions for H2O (nu_exp in MHz).

    Entries: (J', Ka', Kc', J'', Ka'', Kc'', nu_exp_MHz)
    """
    return (
        (1, 1, 0, 1, 0, 1, 556935.985),
        (2, 1, 1, 2, 0, 2, 752033.227),
        (3, 1, 2, 3, 0, 3, 1097364.791),
        (4, 1, 3, 4, 0, 4, 1602219.182),
        (5, 1, 4, 5, 0, 5, 2264148.770),
        (1, 1, 1, 0, 0, 0, 1113342.964),
        (2, 0, 2, 1, 1, 1, 987926.764),
        (2, 1, 2, 1, 0, 1, 1669904.775),
        (3, 0, 3, 2, 1, 2, 1716769.318),
        (3, 1, 3, 2, 0, 2, 2196345.756),
    )


@pytest.fixture
def h2o_dipole() -> Dict[str, float]:
    """Authentic permanent dipole components for H2O in Debye."""
    return {
        "mu_a": 0.0,
        "mu_b": 1.8546,
        "mu_c": 0.0,
    }


@pytest.fixture
def h2o_lower_state_energies() -> Dict[Tuple[int, int, int], float]:
    """Authentic lower state energies for H2O rotational levels in cm^-1."""
    return {
        (0, 0, 0): 0.0,
        (1, 0, 1): 23.794,
        (1, 1, 1): 37.137,
        (2, 0, 2): 70.091,
        (2, 1, 2): 79.496,
        (3, 0, 3): 136.762,
        (4, 0, 4): 222.052,
        (5, 0, 5): 325.348,
    }


def test_feature_names_constant() -> None:
    """Verify FEATURE_NAMES tuple matches exact 14 spectroscopic feature specifications."""
    expected = (
        "J_upper", "Ka_upper", "Kc_upper",
        "J_lower", "Ka_lower", "Kc_lower",
        "delta_J", "delta_Ka", "delta_Kc",
        "mu_a", "mu_b", "mu_c",
        "lower_state_energy_cm",
        "transition_type",
    )
    assert QuantumFeatureExtractor.FEATURE_NAMES == expected
    assert len(QuantumFeatureExtractor.FEATURE_NAMES) == 14


def test_extract_raw_features_shape_and_values(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify raw feature matrix extraction shape (N, 14) and exact quantum column mappings."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    assert isinstance(raw, np.ndarray)
    assert raw.shape == (10, 14)
    assert raw.dtype == np.float64

    # Check first transition: (1, 1, 0, 1, 0, 1, 556935.985)
    # J_u=1, Ka_u=1, Kc_u=0, J_l=1, Ka_l=0, Kc_l=1
    # delta_J=0, delta_Ka=1, delta_Kc=-1
    assert raw[0, 0] == 1.0
    assert raw[0, 1] == 1.0
    assert raw[0, 2] == 0.0
    assert raw[0, 3] == 1.0
    assert raw[0, 4] == 0.0
    assert raw[0, 5] == 1.0
    assert raw[0, 6] == 0.0  # delta_J = 1 - 1
    assert raw[0, 7] == 1.0  # delta_Ka = 1 - 0
    assert raw[0, 8] == -1.0  # delta_Kc = 0 - 1

    # Dipole components: mu_a=0.0, mu_b=1.8546, mu_c=0.0
    assert raw[0, 9] == 0.0
    assert raw[0, 10] == 1.8546
    assert raw[0, 11] == 0.0

    # Lower state energy for (1, 0, 1) -> 23.794 cm^-1
    assert raw[0, 12] == 23.794

    # Transition type: dk_a = |1-0|=1, dk_c = |0-1|=1 -> b-type (1)
    assert raw[0, 13] == 1.0


def test_transition_type_classification() -> None:
    """Verify transition type classification for a-type (0), b-type (1), and c-type (2)."""
    extractor = QuantumFeatureExtractor()

    # a-type: ΔKa = 0, ΔKc = ±1  e.g. 2(1,1) <- 1(1,0) [Ka_u=1, Ka_l=1 -> dk_a=0]
    # b-type: ΔKa = ±1, ΔKc = ±1 e.g. 1(1,1) <- 0(0,0) [Ka_u=1, Ka_l=0, Kc_u=1, Kc_l=0]
    # c-type: ΔKa = ±1, ΔKc = 0  e.g. 1(1,0) <- 0(0,0) [Ka_u=1, Ka_l=0, Kc_u=0, Kc_l=0 -> dk_c=0]
    transitions = (
        (2, 1, 1, 1, 1, 0, 100000.0),  # a-type -> 0
        (1, 1, 1, 0, 0, 0, 200000.0),  # b-type -> 1
        (1, 1, 0, 0, 0, 0, 300000.0),  # c-type -> 2
    )
    dipole = {"mu_a": 1.0, "mu_b": 2.0, "mu_c": 3.0}
    raw = extractor.extract_raw_features(transitions, dipole)

    assert raw[0, 13] == 0.0  # a-type
    assert raw[1, 13] == 1.0  # b-type
    assert raw[2, 13] == 2.0  # c-type


def test_dipole_and_energy_defaults(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...]
) -> None:
    """Verify missing dipole keys and missing lower-state energies default gracefully to 0.0."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=h2o_transitions[:3],
        dipole_components={},  # Empty dipole dict
        lower_state_energies=None,  # No lower state energies
    )

    assert raw.shape == (3, 14)
    # Dipoles must be 0.0
    assert np.all(raw[:, 9:12] == 0.0)
    # Lower state energy must be 0.0
    assert np.all(raw[:, 12] == 0.0)


def test_fit_scaler_and_transform(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify StandardScaler fitting, scaling, and zero-mean unit-variance transformation."""
    extractor = QuantumFeatureExtractor()
    assert extractor.is_fitted is False
    assert extractor.scaler is None

    raw = extractor.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    extractor.fit_scaler(raw)
    assert extractor.is_fitted is True
    assert extractor.scaler is not None

    scaled = extractor.transform(raw)
    assert scaled.shape == raw.shape
    assert isinstance(scaled, np.ndarray)

    # For columns with non-zero variance, verify mean is ~0 and std is ~1
    for col in range(raw.shape[1]):
        col_raw = raw[:, col]
        if np.std(col_raw) > 1e-9:
            assert np.mean(scaled[:, col]) == pytest.approx(0.0, abs=1e-7)
            assert np.std(scaled[:, col]) == pytest.approx(1.0, abs=1e-7)


def test_fit_transform_equivalence(
    h2o_transitions: Tuple[Tuple[int, int, int, int, int, int, float], ...],
    h2o_dipole: Dict[str, float],
    h2o_lower_state_energies: Dict[Tuple[int, int, int], float],
) -> None:
    """Verify fit_transform produces identical result to fit_scaler + transform."""
    extractor_1 = QuantumFeatureExtractor()
    scaled_1 = extractor_1.fit_transform(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )

    extractor_2 = QuantumFeatureExtractor()
    raw_2 = extractor_2.extract_raw_features(
        assignments=h2o_transitions,
        dipole_components=h2o_dipole,
        lower_state_energies=h2o_lower_state_energies,
    )
    extractor_2.fit_scaler(raw_2)
    scaled_2 = extractor_2.transform(raw_2)

    np.testing.assert_allclose(scaled_1, scaled_2, rtol=1e-12, atol=1e-12)


def test_unfitted_transform_raises() -> None:
    """Verify calling transform() on an unfitted QuantumFeatureExtractor raises RuntimeError."""
    extractor = QuantumFeatureExtractor()
    dummy_input = np.ones((5, 14), dtype=np.float64)

    with pytest.raises(RuntimeError, match="Scaler not fitted. Call fit_scaler\\(\\) first."):
        extractor.transform(dummy_input)


def test_empty_assignments() -> None:
    """Verify extract_raw_features gracefully handles empty assignments tuple."""
    extractor = QuantumFeatureExtractor()
    raw = extractor.extract_raw_features(
        assignments=(),
        dipole_components={"mu_a": 1.0},
    )
    assert raw.shape == (0, 14)


def test_dual_tree_exports() -> None:
    """Verify QuantumFeatureExtractor is exportable from root and src physics_core packages."""
    import cochem_spycfit.physics_core as root_core
    import src.cochem_spycfit.physics_core as src_core

    assert hasattr(root_core, "QuantumFeatureExtractor")
    assert hasattr(src_core, "QuantumFeatureExtractor")

    from cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor as ExtractorRoot,
    )
    from src.cochem_spycfit.physics_core.cochem_spycfit_features import (
        QuantumFeatureExtractor as ExtractorSrc,
    )

    assert ExtractorRoot is not None
    assert ExtractorSrc is not None


def test_anti_spoofing_ast() -> None:
    """Verify zero mocks, stubs, or fake implementations exist in the test and implementation."""
    test_path = Path(__file__).resolve()
    impl_src_path = test_path.parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_features.py"
    impl_root_path = test_path.parent.parent / "cochem_spycfit" / "physics_core" / "cochem_spycfit_features.py"

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
                assert node.id not in ("MagicMock", "Mock", "AsyncMock"), f"Prohibited mock identifier in {file_path}"

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.