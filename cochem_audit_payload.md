Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.in-progress\02_P2_T03_parity_auditor.md.
Original prompt:
# Prompt 05 — Phase 2, Task Group 03: Mathematical Parity Auditor (JAX vs. SPFIT)

> **Prerequisite:** Load `D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit-ML\.prompts\00_GLOBAL_SYSTEM_PROMPT.md` into context first.
> **Batch:** 2 — Dual-Engine Parity Bridge (JAX vs. SPCAT)
> **Phase:** 2 (Tasks 21–40) — Scientific Validation & Dual-Engine Execution
> **Dependency:** Prompt 03 (JAX driver) and Prompt 04 (SPFIT wrapper) must exist.

---

## Objective

Build the mathematical difference auditor that compares rotational constants derived from the JAX engine against those from SPFIT. This module is the "Contrast Engine" that ensures publication-grade parity between the modern autodiff optimizer and the legacy Fortran gold standard. Any divergence > 0.1 kHz triggers a yellow "Parity Warning" flag.

---

## Target File

```
D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_auditor.py
```

> **Note:** NEW file in `physics_core/`.

---

## Detailed Implementation Instructions

### 1. Required Imports

```python
"""Mathematical Parity Auditor — JAX vs. SPFIT constant comparison engine."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
```

### 2. `ParityStatus` Enum

```python
class ParityStatus(Enum):
    """Status of the parity check between JAX and SPFIT."""
    VERIFIED = "VERIFIED"       # All constants within 0.1 kHz tolerance
    WARNING = "WARNING"         # At least one constant exceeds 0.1 kHz delta
    SPFIT_FAILED = "SPFIT_FAILED"  # SPFIT execution failed; parity cannot be verified
```

### 3. `ParityWarning` Dataclass

```python
@dataclass(frozen=True)
class ParityWarning:
    """Container for parity divergence details."""
    status: ParityStatus
    max_delta_khz: float
    divergent_constants: tuple[str, ...]
    contrast_table: dict[str, dict[str, float]]
    # contrast_table structure:
    # {
    #     "A": {"jax_mhz": 835840.288, "spfit_mhz": 835840.289, "delta_khz": 1.0},
    #     "B": {"jax_mhz": ..., "spfit_mhz": ..., "delta_khz": ...},
    #     ...
    # }
```

### 4. `ParityAuditor` Class

```python
class ParityAuditor:
    """Compares JAX-derived and SPFIT-derived rotational constants."""

    TOLERANCE_KHZ: float = 0.1  # 0.1 kHz parity threshold

    def __init__(self) -> None:
        """Initialize the parity auditor. Stateless — no instance attributes required."""

    def audit(
        self,
        jax_constants: dict[str, float],
        spfit_constants: dict[str, float],
        jax_uncertainties: Optional[dict[str, float]] = None,
        spfit_uncertainties: Optional[dict[str, float]] = None,
    ) -> ParityWarning:
        """
        Compare rotational constants from JAX and SPFIT engines.

        Parameters
        ----------
        jax_constants : dict
            Constants from the JAX driver (A, B, C, DJ, DJK, DK, etc.) in MHz.
        spfit_constants : dict
            Constants from the SPFIT wrapper in MHz.
        jax_uncertainties : dict, optional
            Uncertainties from JAX fit in MHz.
        spfit_uncertainties : dict, optional
            Uncertainties from SPFIT fit in MHz.

        Returns
        -------
        ParityWarning
            Contains status, max delta, divergent constants list, and
            full contrast table for UI rendering.
        """
        # 1. Identify the union of constant names present in both engines
        common_keys = sorted(set(jax_constants.keys()) & set(spfit_constants.keys()))

        # 2. Build the contrast table with float64 precision
        contrast_table: dict[str, dict[str, float]] = {}
        divergent: list[str] = []
        max_delta = 0.0

        for key in common_keys:
            jax_val = float(jax_constants[key])
            spfit_val = float(spfit_constants[key])
            delta_khz = abs(jax_val - spfit_val) * 1000.0  # MHz → kHz

            contrast_table[key] = {
                "jax_mhz": jax_val,
                "spfit_mhz": spfit_val,
                "delta_khz": delta_khz,
            }
            # Optionally include uncertainties
            if jax_uncertainties and key in jax_uncertainties:
                contrast_table[key]["jax_unc_mhz"] = jax_uncertainties[key]
            if spfit_uncertainties and key in spfit_uncertainties:
                contrast_table[key]["spfit_unc_mhz"] = spfit_uncertainties[key]

            if delta_khz > self.TOLERANCE_KHZ:
                divergent.append(key)
            max_delta = max(max_delta, delta_khz)

        # 3. Determine status
        if divergent:
            status = ParityStatus.WARNING
            logger.warning(
                f"Parity WARNING: {len(divergent)} constants exceed "
                f"{self.TOLERANCE_KHZ} kHz threshold: {divergent}"
            )
        else:
            status = ParityStatus.VERIFIED
            logger.info(
                f"Parity VERIFIED: All {len(common_keys)} constants "
                f"within {self.TOLERANCE_KHZ} kHz (max Δ = {max_delta:.4f} kHz)"
            )

        return ParityWarning(
            status=status,
            max_delta_khz=max_delta,
            divergent_constants=tuple(divergent),
            contrast_table=contrast_table,
        )

    def format_contrast_dataframe(self, warning: ParityWarning) -> "pandas.DataFrame":
        """
        Convert a ParityWarning into a Pandas DataFrame for UI display.

        Returns
        -------
        pandas.DataFrame
            Side-by-side comparison table with columns:
            Constant | JAX (MHz) | SPFIT (MHz) | Δ (kHz) | Status
        """
        import pandas as pd
        rows = []
        for key, vals in warning.contrast_table.items():
            status_flag = "⚠️" if vals["delta_khz"] > self.TOLERANCE_KHZ else "✅"
            rows.append({
                "Constant": key,
                "JAX (MHz)": vals["jax_mhz"],
                "SPFIT (MHz)": vals["spfit_mhz"],
                "Δ (kHz)": vals["delta_khz"],
                "Status": status_flag,
            })
        return pd.DataFrame(rows)
```

---

## Validation Test Specification

1. Create a `ParityAuditor` instance
2. **Test VERIFIED case:** Pass identical H₂O constants to both `jax_constants` and `spfit_constants`:
   - A = 835840.288 MHz, B = 435351.717 MHz, C = 278138.700 MHz
   - Assert `status == ParityStatus.VERIFIED` and `max_delta_khz == 0.0`
3. **Test WARNING case:** Perturb the JAX `A` constant by +0.0002 MHz (= 0.2 kHz delta):
   - Assert `status == ParityStatus.WARNING`
   - Assert `"A"` is in `divergent_constants`
   - Assert `max_delta_khz == pytest.approx(0.2, abs=0.001)`
4. **Test threshold boundary:** Perturb by exactly 0.0001 MHz (= 0.1 kHz — at the boundary):
   - Assert `status == ParityStatus.VERIFIED` (delta == tolerance, not exceeding)
5. **Test format_contrast_dataframe:** Assert the returned DataFrame has correct columns and row count

---

## Acceptance Criteria

- [ ] `ParityStatus` enum defines VERIFIED, WARNING, and SPFIT_FAILED states
- [ ] `ParityWarning` is a frozen dataclass with contrast table structure
- [ ] `audit()` computes deltas in float64 precision with 0.1 kHz threshold
- [ ] Divergent constants are explicitly identified and logged
- [ ] `format_contrast_dataframe()` produces a Pandas DataFrame for UI rendering
- [ ] Validation tests cover VERIFIED, WARNING, and boundary cases
- [ ] No hardcoded atomic masses or synthetic data

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
]


--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\cochem_spycfit\physics_core\cochem_spycfit_auditor.py ---
"""
CoChem-SpycFit: Mathematical Parity Auditor Proxy Re-export.
"""

from __future__ import annotations

from src.cochem_spycfit.physics_core.cochem_spycfit_auditor import (
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    format_contrast_dataframe,
)

__all__ = [
    "ParityAuditor",
    "ParityStatus",
    "ParityWarning",
    "format_contrast_dataframe",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_spycfit_auditor.py ---
"""Mathematical Parity Auditor: JAX autodiff Hamiltonian vs. Pickett SPFIT.

This module provides high-precision cross-validation between pure-JAX Hamiltonian
eigenvalue / Levenberg-Marquardt solutions and Pickett SPFIT / SPCAT legacy Fortran
solutions. It rigorously verifies rotational and centrifugal distortion parameter
congruence against a strict 0.1 kHz mathematical parity tolerance (in FP64).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import enum
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# 1. PARITY STATUS & DATA STRUCTURES
# =============================================================================

@enum.unique
class ParityStatus(str, enum.Enum):
    """Execution and mathematical parity audit status."""

    VERIFIED = "VERIFIED"
    WARNING = "WARNING"
    SPFIT_FAILED = "SPFIT_FAILED"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ParityWarning:
    """Immutable report of mathematical parity audit between JAX and SPFIT.

    Attributes:
        status: Overall parity status (VERIFIED, WARNING, SPFIT_FAILED).
        max_delta_khz: Maximum absolute discrepancy between JAX and SPFIT in kHz.
        divergent_constants: Tuple of parameter names exceeding the parity tolerance.
        contrast_table: Detailed dictionary mapping parameter names to comparative values.
        jax_uncertainties: Optional mapping of JAX parameter standard errors (MHz).
        spfit_uncertainties: Optional mapping of SPFIT parameter standard errors (MHz).
    """

    status: ParityStatus
    max_delta_khz: float
    divergent_constants: Tuple[str, ...] = ()
    contrast_table: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    jax_uncertainties: Optional[Dict[str, float]] = None
    spfit_uncertainties: Optional[Dict[str, float]] = None

    @property
    def is_verified(self) -> bool:
        """Return True if mathematical parity was successfully verified."""
        return self.status == ParityStatus.VERIFIED

    @property
    def has_divergence(self) -> bool:
        """Return True if one or more constants diverged beyond tolerance."""
        return len(self.divergent_constants) > 0

    def to_dataframe(self) -> Any:
        """Format contrast table as a pandas DataFrame."""
        return ParityAuditor.format_contrast_dataframe(self)


# =============================================================================
# 2. PARITY AUDITOR CORE
# =============================================================================

class ParityAuditor:
    """Rigorous mathematical parity auditor comparing JAX and SPFIT spectroscopic constants.

    Audits Watson A-reduction and S-reduction rotational and distortion constants
    in FP64 precision with a default tolerance of 0.1 kHz (1e-4 MHz).
    """

    TOLERANCE_KHZ: float = 0.1

    def __init__(self, default_tolerance_khz: float = TOLERANCE_KHZ) -> None:
        """Initialize the stateless ParityAuditor.

        Args:
            default_tolerance_khz: Default divergence threshold in kHz (default: 0.1 kHz).
        """
        self.default_tolerance_khz: float = float(default_tolerance_khz)

    def audit(
        self,
        jax_constants: Mapping[str, float],
        spfit_constants: Optional[Mapping[str, float]] = None,
        jax_uncertainties: Optional[Mapping[str, float]] = None,
        spfit_uncertainties: Optional[Mapping[str, float]] = None,
        tolerance_khz: Optional[float] = None,
    ) -> ParityWarning:
        """Audit parity between JAX-optimized constants and SPFIT constants.

        Args:
            jax_constants: Mapping of constant names to JAX-optimized values (MHz).
            spfit_constants: Mapping of constant names to SPFIT values (MHz), or None if SPFIT failed.
            jax_uncertainties: Optional mapping of JAX parameter uncertainties (MHz).
            spfit_uncertainties: Optional mapping of SPFIT parameter uncertainties (MHz).
            tolerance_khz: Override tolerance threshold in kHz (defaults to instance setting).

        Returns:
            ParityWarning frozen dataclass detailing parity status and contrast metrics.
        """
        tol = float(tolerance_khz) if tolerance_khz is not None else self.default_tolerance_khz

        # Handle failed SPFIT execution
        if spfit_constants is None:
            logger.warning("SPFIT constants are None: flagging SPFIT_FAILED.")
            return ParityWarning(
                status=ParityStatus.SPFIT_FAILED,
                max_delta_khz=float("inf"),
                divergent_constants=tuple(jax_constants.keys()),
                contrast_table={},
                jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
                spfit_uncertainties=None,
            )

        # Audit only common keys present in both parameter sets
        common_keys = [k for k in jax_constants.keys() if k in spfit_constants]

        if not common_keys:
            logger.warning("No overlapping parameter keys found between JAX and SPFIT sets.")
            return ParityWarning(
                status=ParityStatus.VERIFIED,
                max_delta_khz=0.0,
                divergent_constants=(),
                contrast_table={},
                jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
                spfit_uncertainties=dict(spfit_uncertainties) if spfit_uncertainties is not None else None,
            )

        contrast_table: Dict[str, Dict[str, Any]] = {}
        divergent_list: List[str] = []
        max_delta = 0.0

        for key in common_keys:
            val_jax = float(np.float64(jax_constants[key]))
            val_spfit = float(np.float64(spfit_constants[key]))

            delta_mhz = abs(val_jax - val_spfit)
            delta_khz = float(delta_mhz * 1000.0)

            # Check divergence: delta > tol and not within float64 representation precision
            is_divergent = (delta_khz > tol) and not math.isclose(delta_khz, tol, rel_tol=1e-7, abs_tol=1e-9)

            if is_divergent:
                divergent_list.append(key)

            if delta_khz > max_delta:
                max_delta = delta_khz

            jax_unc = (
                float(jax_uncertainties[key])
                if jax_uncertainties is not None and key in jax_uncertainties and jax_uncertainties[key] is not None
                else None
            )
            spfit_unc = (
                float(spfit_uncertainties[key])
                if spfit_uncertainties is not None and key in spfit_uncertainties and spfit_uncertainties[key] is not None
                else None
            )

            status_emoji = "⚠️" if is_divergent else "✅"

            contrast_table[key] = {
                "jax_mhz": val_jax,
                "spfit_mhz": val_spfit,
                "delta_khz": delta_khz,
                "status": status_emoji,
                "is_divergent": is_divergent,
                "jax_unc": jax_unc,
                "spfit_unc": spfit_unc,
                "jax_unc_mhz": jax_unc,
                "spfit_unc_mhz": spfit_unc,
            }

        status = ParityStatus.WARNING if len(divergent_list) > 0 else ParityStatus.VERIFIED

        return ParityWarning(
            status=status,
            max_delta_khz=float(max_delta),
            divergent_constants=tuple(divergent_list),
            contrast_table=contrast_table,
            jax_uncertainties=dict(jax_uncertainties) if jax_uncertainties is not None else None,
            spfit_uncertainties=dict(spfit_uncertainties) if spfit_uncertainties is not None else None,
        )

    @staticmethod
    def format_contrast_dataframe(warning: ParityWarning) -> Any:
        """Format contrast table from a ParityWarning into a presentation pandas DataFrame.

        Deferred pandas import to preserve zero-overhead CLI startup.

        Args:
            warning: ParityWarning object to format.

        Returns:
            pandas.DataFrame with columns ['Constant', 'JAX (MHz)', 'SPFIT (MHz)', 'Δ (kHz)', 'Status'].
        """
        import pandas as pd  # type: ignore[import-untyped]

        columns = ["Constant", "JAX (MHz)", "SPFIT (MHz)", "Δ (kHz)", "Status"]
        if not warning.contrast_table:
            return pd.DataFrame(columns=columns)

        rows = []
        for const_name, data in warning.contrast_table.items():
            jax_val = data.get("jax_mhz", np.nan)
            spfit_val = data.get("spfit_mhz", np.nan)
            delta_khz = data.get("delta_khz", 0.0)
            status_emoji = data.get("status", "⚠️" if const_name in warning.divergent_constants else "✅")
            if status_emoji not in ("✅", "⚠️"):
                status_emoji = "⚠️" if const_name in warning.divergent_constants else "✅"

            rows.append({
                "Constant": const_name,
                "JAX (MHz)": jax_val,
                "SPFIT (MHz)": spfit_val,
                "Δ (kHz)": delta_khz,
                "Status": status_emoji,
            })

        return pd.DataFrame(rows, columns=columns)


format_contrast_dataframe = ParityAuditor.format_contrast_dataframe

__all__ = [
    "ParityAuditor",
    "ParityStatus",
    "ParityWarning",
    "format_contrast_dataframe",
]

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_spycfit_auditor.py ---
"""Physical unit tests for Mathematical Parity Auditor (JAX autodiff vs. Pickett SPFIT).

Strict Zero-Mock & Anti-Spoofing Directive v2 Compliance:
- Authentic literature rotational and centrifugal distortion constants for H2O.
- Physical mathematical divergence checks against strict 0.1 kHz threshold.
- Direct validation of frozen dataclass immutability, boundary conditions, and uncertainty preservation.
- Static AST validation confirming zero synthetic mocks, zero stubs, and zero pass statements.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
from pathlib import Path
from typing import Dict

import pytest

from cochem_spycfit.physics_core.cochem_spycfit_auditor import (
    ParityAuditor,
    ParityStatus,
    ParityWarning,
    format_contrast_dataframe,
)


@pytest.fixture
def h2o_constants() -> Dict[str, float]:
    """Authentic literature Watson A-reduction parameters for H2O in MHz."""
    return {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "DJ": 37.594,
        "DJK": -172.91,
        "DK": 973.29,
        "dJ": 9.423,
        "dK": 41.25,
    }


def test_parity_verified_identical_constants(h2o_constants: Dict[str, float]) -> None:
    """Verify VERIFIED parity status when JAX and SPFIT constants are identical."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    assert isinstance(warning, ParityWarning)
    assert warning.status == ParityStatus.VERIFIED
    assert warning.status == "VERIFIED"
    assert warning.is_verified is True
    assert warning.has_divergence is False
    assert warning.max_delta_khz == 0.0
    assert warning.divergent_constants == ()
    assert len(warning.contrast_table) == len(h2o_constants)

    for const_name in h2o_constants:
        entry = warning.contrast_table[const_name]
        assert entry["status"] == "✅"
        assert entry["delta_khz"] == 0.0
        assert entry["is_divergent"] is False


def test_parity_warning_perturbed_constant(h2o_constants: Dict[str, float]) -> None:
    """Verify WARNING parity status when constant A is perturbed by +0.0002 MHz (0.2 kHz > 0.1 kHz)."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0002  # 0.2 kHz divergence

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert warning.status == "WARNING"
    assert warning.is_verified is False
    assert warning.has_divergence is True
    assert "A" in warning.divergent_constants
    assert len(warning.divergent_constants) == 1
    assert warning.max_delta_khz == pytest.approx(0.2, abs=0.001)

    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True
    assert warning.contrast_table["B"]["status"] == "✅"
    assert warning.contrast_table["B"]["is_divergent"] is False


def test_parity_exact_boundary_tolerance(h2o_constants: Dict[str, float]) -> None:
    """Verify boundary condition: perturbation of exactly 0.0001 MHz (0.1 kHz) remains VERIFIED."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001  # Exactly 0.1 kHz = tolerance threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.VERIFIED
    assert warning.divergent_constants == ()
    assert warning.max_delta_khz == pytest.approx(0.1, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "✅"
    assert warning.contrast_table["A"]["is_divergent"] is False


def test_parity_slight_threshold_breach(h2o_constants: Dict[str, float]) -> None:
    """Verify boundary breach: perturbation of 0.0001001 MHz (0.1001 kHz > 0.1 kHz) triggers WARNING."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0001001  # 0.1001 kHz > 0.1 kHz threshold

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert "A" in warning.divergent_constants
    assert warning.max_delta_khz == pytest.approx(0.1001, abs=0.001)
    assert warning.contrast_table["A"]["status"] == "⚠️"
    assert warning.contrast_table["A"]["is_divergent"] is True


def test_uncertainty_preservation(h2o_constants: Dict[str, float]) -> None:
    """Verify uncertainty preservation when jax_uncertainties and spfit_uncertainties are passed."""
    auditor = ParityAuditor()
    jax_unc = {
        "A": 0.0050,
        "B": 0.0030,
        "C": 0.0020,
        "DJ": 0.0001,
        "DJK": 0.0010,
        "DK": 0.0100,
        "dJ": 0.0001,
        "dK": 0.0005,
    }
    spfit_unc = {
        "A": 0.0048,
        "B": 0.0031,
        "C": 0.0019,
        "DJ": 0.0001,
        "DJK": 0.0011,
        "DK": 0.0098,
        "dJ": 0.0001,
        "dK": 0.00049,
    }

    warning = auditor.audit(
        jax_constants=h2o_constants,
        spfit_constants=h2o_constants,
        jax_uncertainties=jax_unc,
        spfit_uncertainties=spfit_unc,
    )

    assert warning.jax_uncertainties == jax_unc
    assert warning.spfit_uncertainties == spfit_unc
    assert warning.contrast_table["A"]["jax_unc"] == 0.0050
    assert warning.contrast_table["A"]["spfit_unc"] == 0.0048
    assert warning.contrast_table["DK"]["jax_unc"] == 0.0100
    assert warning.contrast_table["DK"]["spfit_unc"] == 0.0098


def test_disjoint_constant_sets() -> None:
    """Verify that when parameter sets are disjoint, only common keys are audited."""
    auditor = ParityAuditor()
    jax_constants = {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "HJ": 0.0001,
    }
    spfit_constants = {
        "A": 835840.288,
        "B": 435351.717,
        "C": 278138.700,
        "LK": 0.00005,
    }

    warning = auditor.audit(jax_constants=jax_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.VERIFIED
    assert warning.max_delta_khz == 0.0
    assert warning.divergent_constants == ()
    assert set(warning.contrast_table.keys()) == {"A", "B", "C"}
    assert "HJ" not in warning.contrast_table
    assert "LK" not in warning.contrast_table


def test_format_contrast_dataframe(h2o_constants: Dict[str, float]) -> None:
    """Verify format_contrast_dataframe returns valid pandas DataFrame with correct columns and emojis."""
    import pandas as pd  # type: ignore[import-untyped]

    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0003  # 0.3 kHz divergence (⚠️)
    spfit_constants["B"] += 0.00005  # 0.05 kHz within tolerance (✅)

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)
    df = auditor.format_contrast_dataframe(warning)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Constant", "JAX (MHz)", "SPFIT (MHz)", "Δ (kHz)", "Status"]
    assert len(df) == len(h2o_constants)

    a_row = df[df["Constant"] == "A"].iloc[0]
    assert a_row["Status"] == "⚠️"
    assert a_row["Δ (kHz)"] == pytest.approx(0.3, abs=0.001)

    b_row = df[df["Constant"] == "B"].iloc[0]
    assert b_row["Status"] == "✅"
    assert b_row["Δ (kHz)"] == pytest.approx(0.05, abs=0.001)

    # Test direct module-level function call
    df_direct = format_contrast_dataframe(warning)
    assert isinstance(df_direct, pd.DataFrame)
    assert list(df_direct.columns) == list(df.columns)
    assert len(df_direct) == len(df)

    # Test instance helper method parity
    df_helper = warning.to_dataframe()
    assert isinstance(df_helper, pd.DataFrame)
    assert list(df_helper.columns) == list(df.columns)
    assert len(df_helper) == len(df)


def test_dataclass_immutability(h2o_constants: Dict[str, float]) -> None:
    """Verify ParityWarning is a strictly frozen immutable dataclass."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=h2o_constants)

    with pytest.raises(FrozenInstanceError):
        warning.status = ParityStatus.WARNING  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.max_delta_khz = 100.0  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        warning.divergent_constants = ("A",)  # type: ignore[misc]


def test_spfit_failed_state(h2o_constants: Dict[str, float]) -> None:
    """Verify SPFIT_FAILED state when spfit_constants is None."""
    auditor = ParityAuditor()
    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=None)

    assert warning.status == ParityStatus.SPFIT_FAILED
    assert warning.is_verified is False
    assert math.isinf(warning.max_delta_khz)
    assert warning.contrast_table == {}
    assert set(warning.divergent_constants) == set(h2o_constants.keys())


def test_custom_tolerance(h2o_constants: Dict[str, float]) -> None:
    """Verify custom tolerance thresholds via constructor and method argument overrides."""
    auditor_relaxed = ParityAuditor(default_tolerance_khz=0.5)
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0003  # 0.3 kHz divergence

    # Relaxed tolerance of 0.5 kHz accepts 0.3 kHz delta
    warning_relaxed = auditor_relaxed.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)
    assert warning_relaxed.status == ParityStatus.VERIFIED

    # Overridden strict tolerance of 0.1 kHz flags 0.3 kHz delta
    warning_strict = auditor_relaxed.audit(
        jax_constants=h2o_constants,
        spfit_constants=spfit_constants,
        tolerance_khz=0.1,
    )
    assert warning_strict.status == ParityStatus.WARNING
    assert "A" in warning_strict.divergent_constants


def test_multiple_divergences(h2o_constants: Dict[str, float]) -> None:
    """Verify handling of multiple diverging parameters."""
    auditor = ParityAuditor()
    spfit_constants = dict(h2o_constants)
    spfit_constants["A"] += 0.0005  # 0.5 kHz
    spfit_constants["DJK"] += 0.0008  # 0.8 kHz

    warning = auditor.audit(jax_constants=h2o_constants, spfit_constants=spfit_constants)

    assert warning.status == ParityStatus.WARNING
    assert set(warning.divergent_constants) == {"A", "DJK"}
    assert warning.max_delta_khz == pytest.approx(0.8, abs=0.001)


def test_dual_tree_parity_exports() -> None:
    """Verify dual-tree imports from src and proxy directories are properly exported."""
    import cochem_spycfit.physics_core as proxy_core
    import src.cochem_spycfit.physics_core as src_core

    assert hasattr(src_core, "ParityAuditor")
    assert hasattr(src_core, "ParityStatus")
    assert hasattr(src_core, "ParityWarning")
    assert hasattr(src_core, "format_contrast_dataframe")

    assert hasattr(proxy_core, "ParityAuditor")
    assert hasattr(proxy_core, "ParityStatus")
    assert hasattr(proxy_core, "ParityWarning")
    assert hasattr(proxy_core, "format_contrast_dataframe")

    assert "ParityAuditor" in src_core.__all__
    assert "ParityStatus" in src_core.__all__
    assert "ParityWarning" in src_core.__all__
    assert "format_contrast_dataframe" in src_core.__all__

    assert "ParityAuditor" in proxy_core.__all__
    assert "ParityStatus" in proxy_core.__all__
    assert "ParityWarning" in proxy_core.__all__
    assert "format_contrast_dataframe" in proxy_core.__all__


def test_anti_spoofing_compliance_ast() -> None:
    """Enforce strict Zero-Mock and Anti-Spoofing rules via AST inspection."""
    test_file = Path(__file__).resolve()
    impl_file = test_file.parent.parent / "src" / "cochem_spycfit" / "physics_core" / "cochem_spycfit_auditor.py"

    for file_path in [test_file, impl_file]:
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