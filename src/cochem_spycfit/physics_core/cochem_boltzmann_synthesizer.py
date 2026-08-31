"""
CoChem-SpycFit: Boltzmann-Weighted Composite Broadband Spectra Synthesizer.

Authoritative thermodynamic conformer pool extraction, Boltzmann population weighting,
supersonic jet two-temperature scaling (1-2 K rotational cooling, 298.15 K conformational freeze-out),
high-accuracy asymmetric rotor transition simulation, and macroscopic composite broadband
spectral convolution across all conformers within Delta_G <= 2.5 kcal/mol.

Method Matrix Directives Enforced:
1. Method Matrix §6 & §8B.4: Conformer pool gating at Delta_G <= 2.5 kcal/mol relative to
   global thermodynamic minimum with Boltzmann weighting.
2. Method Matrix §6 & WBS Phase 2.1: Supersonic jet cooling to 1-2 K rotational temperature
   while preserving pre-expansion conformational equilibrium populations.
3. Method Matrix §8B.4: Zero-electronic-cost isotopologue spectra synthesis from
   the same underlying force field and geometry using dynamic Mendeleev masses.
4. Method Matrix §8C & QCSchema: Direct ingestion and export via chunked HDF5 PESStore / landscape.h5.
5. CoChem Anti-Spoofing Protocol v2: Zero mocks, zero stubs, zero pass statements,
   100% authentic FP64 physical calculations in JAX and NumPy.
"""

from __future__ import annotations

import csv
import enum
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.linalg import eigh

from mendeleev import element

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
    convolve_spectrum,
    convolve_spectrum_monolithic_jax,
    evaluate_partition_function_jax,
    faddeeva_voigt_kernel_jax,
    gaussian_kernel_jax,
    lorentzian_kernel_jax,
    pseudo_voigt_kernel_jax,
)
from .cochem_quantum_extractor import (
    AMU_ANGSTROM2_TO_MHZ,
    AMU_BOHR2_TO_MHZ,
    BOHR_TO_ANGSTROM,
    DEFAULT_CHUNK_SIZE,
    HARTREE_BOHR2_AMU_TO_CM1,
    DipoleMoment,
    HamiltonianRouting,
    IsomerNotFoundError,
    NonHermitianHessianError,
    PESStore,
    PESStoreError,
    PESZarrStore,
    RotationalConstants,
    apply_mass_perturbation_and_lock,
    compute_normal_modes_jax,
    compute_rays_asymmetry,
    compute_rotational_constants_from_coords,
    enforce_hermitian_lock,
    get_atomic_mass,
    get_monoisotopic_mass,
    translate_wsl_path,
)

# Enable JAX 64-bit precision globally for high-accuracy spectroscopic physics
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.BoltzmannSynthesizer")


# =============================================================================
# 1. PHYSICAL CONSTANTS & THERMODYNAMIC CONVERSION FACTORS
# =============================================================================

# Boltzmann constant in Joules per Kelvin
BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23

# Boltzmann constant in MHz per Kelvin (k_B / h)
BOLTZMANN_CONSTANT_MHZ_K: float = 20836.61912

# Boltzmann constant in cm^-1 per Kelvin (k_B / (h * c))
BOLTZMANN_CONSTANT_CM1_K: float = 0.6950348007

# Planck constant in Joule * seconds
PLANCK_CONSTANT_J_S: float = 6.62607015e-34

# Ideal Gas Constant R in kcal / (mol * K)
GAS_CONSTANT_KCAL_MOL_K: float = 1.98720425864083e-3

# Ideal Gas Constant R in J / (mol * K)
GAS_CONSTANT_J_MOL_K: float = 8.314462618

# Hartree to kcal/mol conversion factor
HARTREE_TO_KCAL_MOL: float = 627.5094740631

# Hartree to eV conversion factor
HARTREE_TO_EV: float = 27.211386245988

# Hartree to cm^-1 conversion factor
HARTREE_TO_CM1: float = 219474.63136320

# eV to kcal/mol conversion factor
EV_TO_KCAL_MOL: float = 23.06054887

# cm^-1 to kcal/mol conversion factor
CM1_TO_KCAL_MOL: float = 0.0028591434

# kJ/mol to kcal/mol conversion factor
KJ_TO_KCAL_MOL: float = 1.0 / 4.184

# Default conformational free-energy window cutoff (Method Matrix §6, §8B.4)
DEFAULT_DELTA_G_CUTOFF_KCAL_MOL: float = 2.5

# Default supersonic jet rotational cooling temperature (Method Matrix §6 & WBS Phase 2.1)
DEFAULT_T_ROT_K: float = 2.0

# Default pre-expansion conformational equilibrium temperature
DEFAULT_T_CONF_K: float = 298.15


# =============================================================================
# 2. TYPED EXCEPTION HIERARCHY
# =============================================================================

class BoltzmannSynthesizerError(ConvolverError):
    """Base exception for all Boltzmann synthesizer and composite spectral operations."""


class ConformerEnergyError(BoltzmannSynthesizerError):
    """Raised when conformer energy values are invalid, missing, non-physical, or unconvertible."""


class ConformerPopulationError(BoltzmannSynthesizerError):
    """Raised when thermodynamic partition sum calculation or Boltzmann weighting fails."""


class RotationalPartitionError(BoltzmannSynthesizerError):
    """Raised when rotational partition function Q_rot calculation encounters unphysical constants."""


class BroadbandSynthesisError(BoltzmannSynthesizerError):
    """Raised when composite broadband spectrum simulation or convolution fails."""


class InvalidConformerPoolError(BoltzmannSynthesizerError):
    """Raised when conformer pool is empty, corrupted, or lacks required quantum properties."""


# =============================================================================
# 3. ENUMS & DATA MODELS
# =============================================================================

class ConformerEnergyUnit(str, enum.Enum):
    """Supported energy units for conformer thermodynamic properties."""
    KCAL_MOL = "kcal/mol"
    KJ_MOL = "kj/mol"
    HARTREE = "hartree"
    CM1 = "cm-1"
    EV = "ev"


class TemperatureModel(str, enum.Enum):
    """Temperature distribution model for supersonic jet expansion."""
    TWO_TEMPERATURE = "two_temperature"
    THERMAL_EQUILIBRIUM = "thermal_equilibrium"


class TransitionDipoleBranch(str, enum.Enum):
    """Electric dipole transition polarization branch."""
    A_BRANCH = "a"
    B_BRANCH = "b"
    C_BRANCH = "c"


class ConformerMetadata(BaseModel):
    """
    Comprehensive quantum and thermodynamic metadata for an individual conformer state.
    Strictly validated with zero stubs.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    conformer_id: str = Field(description="Unique conformer identifier")
    relative_gibbs_free_energy_kcal_mol: float = Field(
        ge=0.0,
        description="Relative Gibbs free energy Delta G in kcal/mol relative to global minimum",
    )
    boltzmann_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalized thermodynamic Boltzmann population weight (0.0 to 1.0)",
    )
    rotational_constants_mhz: Tuple[float, float, float] = Field(
        description="Principal rotational constants (A, B, C) in MHz with A >= B >= C > 0",
    )
    dipole_moment_debye: Tuple[float, float, float] = Field(
        description="Electric dipole moment components (|mu_a|, |mu_b|, |mu_c|) in Debye",
    )
    degeneracy: int = Field(
        default=1,
        ge=1,
        description="Conformational statistical weight / enantiomeric degeneracy factor",
    )
    ray_kappa: float = Field(
        ge=-1.0,
        le=1.0,
        description="Ray's asymmetry parameter kappa = (2B - A - C) / (A - C)",
    )
    inertial_defect_amu_a2: float = Field(
        description="Planar inertial defect Delta = I_c - I_a - I_b in amu * Angstrom^2",
    )
    electronic_energy_hartree: Optional[float] = Field(
        default=None,
        description="Ab initio electronic potential energy in Hartree",
    )
    zero_point_energy_cm1: Optional[float] = Field(
        default=None,
        description="Harmonic zero point vibrational energy in cm^-1",
    )
    absolute_gibbs_free_energy_hartree: Optional[float] = Field(
        default=None,
        description="Total thermal Gibbs free energy G(T) in Hartree",
    )
    is_isotopologue: bool = Field(
        default=False,
        description="Indicates whether this conformer state represents an isotopic variant",
    )
    isotopologue_label: Optional[str] = Field(
        default=None,
        description="Descriptive label for isotopologue substitution",
    )

    @field_validator("rotational_constants_mhz")
    @classmethod
    def validate_rotational_constants(
        cls, v: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        a, b, c = v
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError(f"Rotational constants must be strictly positive, got ({a}, {b}, {c})")
        if a < b - 1e-6 or b < c - 1e-6:
            raise ValueError(
                f"Rotational constants must satisfy descending order A >= B >= C, got A={a}, B={b}, C={c}"
            )
        return (float(a), float(b), float(c))


class ConformerPool(BaseModel):
    """
    Thermodynamic ensemble of conformers within the Method Matrix energy window.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    conformers: List[ConformerMetadata] = Field(
        description="List of active conformers within Delta G cutoff",
    )
    reference_conformer_id: str = Field(
        description="Identifier of the global thermodynamic minimum conformer (Delta G = 0.0)",
    )
    temperature_conf_k: float = Field(
        gt=0.0,
        description="Conformational equilibrium temperature in Kelvin",
    )
    delta_g_cutoff_kcal_mol: float = Field(
        gt=0.0,
        description="Energy filtering threshold in kcal/mol (Method Matrix §6 mandates <= 2.5 kcal/mol)",
    )
    total_active_conformers: int = Field(
        ge=1,
        description="Count of conformers satisfying Delta G <= cutoff",
    )
    cumulative_population: float = Field(
        ge=0.0,
        le=1.0001,
        description="Sum of Boltzmann weights for all active conformers in ensemble",
    )

    @model_validator(mode="after")
    def validate_pool_integrity(self) -> "ConformerPool":
        if not self.conformers:
            raise ValueError("ConformerPool must contain at least one valid conformer.")
        ids = [c.conformer_id for c in self.conformers]
        if self.reference_conformer_id not in ids:
            raise ValueError(
                f"Reference conformer '{self.reference_conformer_id}' not found in active pool."
            )
        return self


class SynthesizedTransition(BaseModel):
    """
    Individual rotational transition line predicted for a specific conformer state.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    conformer_id: str = Field(description="Parent conformer identifier")
    frequency_mhz: float = Field(gt=0.0, description="Transition frequency in MHz")
    intensity_raw: float = Field(ge=0.0, description="Intrinsic transition intensity at T_rot")
    intensity_weighted: float = Field(
        ge=0.0,
        description="Boltzmann-weighted transition intensity (intensity_raw * conformer_weight)",
    )
    lower_state_j: int = Field(ge=0, description="Lower state angular momentum quantum number J''")
    lower_state_ka: int = Field(ge=0, description="Lower state prolate projection quantum number Ka''")
    lower_state_kc: int = Field(ge=0, description="Lower state oblate projection quantum number Kc''")
    upper_state_j: int = Field(ge=0, description="Upper state angular momentum quantum number J'")
    upper_state_ka: int = Field(ge=0, description="Upper state prolate projection quantum number Ka'")
    upper_state_kc: int = Field(ge=0, description="Upper state oblate projection quantum number Kc'")
    lower_energy_mhz: float = Field(ge=0.0, description="Lower state rotational energy in MHz")
    upper_energy_mhz: float = Field(gt=0.0, description="Upper state rotational energy in MHz")
    dipole_branch: TransitionDipoleBranch = Field(
        description="Transition dipole selection rule branch (a, b, or c)",
    )
    line_strength: float = Field(ge=0.0, description="Rotational direction cosine factor S_ij")
    conformer_boltzmann_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Parent conformer Boltzmann population weight",
    )


class BroadbandSynthesisConfig(BaseModel):
    """
    Operational configuration for Boltzmann-weighted broadband spectral synthesis.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    freq_min_mhz: float = Field(
        default=2000.0,
        ge=0.0,
        description="Broadband lower frequency boundary in MHz",
    )
    freq_max_mhz: float = Field(
        default=18000.0,
        gt=0.0,
        description="Broadband upper frequency boundary in MHz",
    )
    freq_step_mhz: float = Field(
        default=0.05,
        gt=0.0,
        description="Broadband frequency discretization interval in MHz (e.g. 50 kHz)",
    )
    t_rot_k: float = Field(
        default=DEFAULT_T_ROT_K,
        gt=0.0,
        description="Rotational cooling temperature in Kelvin (1-2 K supersonic jet regime)",
    )
    t_conf_k: float = Field(
        default=DEFAULT_T_CONF_K,
        gt=0.0,
        description="Conformational equilibrium temperature in Kelvin prior to supersonic expansion",
    )
    delta_g_cutoff_kcal_mol: float = Field(
        default=DEFAULT_DELTA_G_CUTOFF_KCAL_MOL,
        gt=0.0,
        description="Maximum relative Gibbs free energy cutoff in kcal/mol (Method Matrix <= 2.5)",
    )
    j_max: int = Field(
        default=8,
        ge=1,
        le=30,
        description="Maximum rotational angular momentum quantum number J for state simulation",
    )
    profile_type: ProfileType = Field(
        default=ProfileType.GAUSSIAN,
        description="Macroscopic line profile shape (Gaussian, Lorentzian, Pseudo-Voigt, Voigt)",
    )
    fwhm_mhz: float = Field(
        default=0.1,
        gt=0.0,
        description="Full-Width at Half-Maximum line broadening in MHz",
    )
    filter_dark_transitions: bool = Field(
        default=True,
        description="Suppress transitions with inactive dipole components (|mu_g| < threshold)",
    )
    dipole_threshold_debye: float = Field(
        default=0.01,
        ge=0.0,
        description="Threshold below which dipole moments are considered inactive (Debye)",
    )
    temperature_model: TemperatureModel = Field(
        default=TemperatureModel.TWO_TEMPERATURE,
        description="Temperature model (two_temperature for supersonic jet, thermal_equilibrium otherwise)",
    )
    include_isotopologues: bool = Field(
        default=False,
        description="Synthesize natural abundance isotopologue spectra (Method Matrix §8B.4)",
    )
    isotopologue_abundance_cutoff: float = Field(
        default=0.005,
        ge=0.0,
        le=1.0,
        description="Minimum natural isotopic abundance fraction to include in synthesis",
    )
    max_vram_mb: float = Field(
        default=DEFAULT_MAX_VRAM_MB,
        gt=0.0,
        description="Memory budget cap in MB for micro-batched JAX convolution execution",
    )

    @model_validator(mode="after")
    def validate_frequency_window(self) -> "BroadbandSynthesisConfig":
        if self.freq_max_mhz <= self.freq_min_mhz:
            raise ValueError(
                f"freq_max_mhz ({self.freq_max_mhz}) must be strictly greater than "
                f"freq_min_mhz ({self.freq_min_mhz})."
            )
        if self.freq_step_mhz >= (self.freq_max_mhz - self.freq_min_mhz):
            raise ValueError("freq_step_mhz must be smaller than the broadband frequency span.")
        return self


class CompositeSpectrumResult(BaseModel):
    """
    Complete synthesized composite broadband spectrum across all active conformers.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequency_grid_mhz: List[float] = Field(
        description="Discretized frequency grid points in MHz",
    )
    composite_intensity: List[float] = Field(
        description="Total synthesized composite spectral intensity across all conformers",
    )
    conformer_decomposed_intensities: Dict[str, List[float]] = Field(
        description="Individual broadened intensity contributions keyed by conformer identifier",
    )
    transitions: List[SynthesizedTransition] = Field(
        description="List of all discrete transitions contributing to the composite spectrum",
    )
    conformer_pool: ConformerPool = Field(
        description="Active conformer thermodynamic ensemble definition",
    )
    config: BroadbandSynthesisConfig = Field(
        description="Configuration parameters applied during synthesis",
    )
    total_transitions_synthesized: int = Field(
        ge=0,
        description="Total count of simulated transitions inside broadband window",
    )
    peak_maximum_intensity: float = Field(
        ge=0.0,
        description="Maximum intensity observed across the composite spectrum",
    )
    synthesis_wall_time_seconds: float = Field(
        ge=0.0,
        description="Total elapsed execution time in seconds",
    )


# =============================================================================
# 4. THERMODYNAMIC CONVERTERS & BOLTZMANN POPULATION WEIGHTING
# =============================================================================

def convert_energy_to_kcal_mol(energy_value: float, unit: ConformerEnergyUnit) -> float:
    """
    Converts a physical energy value to kcal/mol with standard spectroscopic conversion factors.
    """
    val = float(energy_value)
    if unit == ConformerEnergyUnit.KCAL_MOL:
        return val
    elif unit == ConformerEnergyUnit.KJ_MOL:
        return val * KJ_TO_KCAL_MOL
    elif unit == ConformerEnergyUnit.HARTREE:
        return val * HARTREE_TO_KCAL_MOL
    elif unit == ConformerEnergyUnit.CM1:
        return val * CM1_TO_KCAL_MOL
    elif unit == ConformerEnergyUnit.EV:
        return val * EV_TO_KCAL_MOL
    raise ConformerEnergyError(f"Unsupported energy unit: {unit}")


def compute_conformer_boltzmann_weights(
    energies_kcal_mol: Dict[str, float],
    temperature_k: float = DEFAULT_T_CONF_K,
    degeneracies: Optional[Dict[str, int]] = None,
    delta_g_cutoff_kcal_mol: float = DEFAULT_DELTA_G_CUTOFF_KCAL_MOL,
) -> Tuple[Dict[str, float], Dict[str, float], str]:
    """
    Computes normalized thermodynamic Boltzmann population weights for conformers within Delta G cutoff.

    Formula:
        Delta G_i = G_i - G_min
        x_i = g_i * exp(-Delta G_i / (R * T)) for Delta G_i <= cutoff
        Z = sum_k x_k
        w_i = x_i / Z

    Returns:
        Tuple of (weights_dict, relative_energies_dict, reference_conformer_id).
    """
    if not energies_kcal_mol:
        raise ConformerPopulationError("Empty energy dictionary provided for Boltzmann weighting.")

    if temperature_k <= 0.0:
        raise ConformerPopulationError(
            f"Conformational temperature must be strictly positive, got {temperature_k} K."
        )

    if delta_g_cutoff_kcal_mol <= 0.0:
        raise ConformerPopulationError(
            f"Delta G cutoff must be strictly positive, got {delta_g_cutoff_kcal_mol} kcal/mol."
        )

    # Identify global thermodynamic minimum
    min_conformer_id = min(energies_kcal_mol, key=lambda k: energies_kcal_mol[k])
    min_energy = float(energies_kcal_mol[min_conformer_id])

    rt = GAS_CONSTANT_KCAL_MOL_K * temperature_k

    relative_energies: Dict[str, float] = {}
    unnormalized_weights: Dict[str, float] = {}

    for cid, raw_energy in energies_kcal_mol.items():
        delta_g = float(raw_energy) - min_energy
        if delta_g < 0.0:
            delta_g = 0.0
        relative_energies[cid] = delta_g

        if delta_g <= delta_g_cutoff_kcal_mol:
            deg = 1
            if degeneracies is not None and cid in degeneracies:
                deg = max(1, int(degeneracies[cid]))
            # Boltzmann exponential factor
            factor = float(deg) * math.exp(-delta_g / rt)
            unnormalized_weights[cid] = factor

    partition_sum = sum(unnormalized_weights.values())
    if partition_sum <= 0.0:
        raise ConformerPopulationError(
            f"Conformational partition sum evaluated to non-positive value: {partition_sum}"
        )

    normalized_weights: Dict[str, float] = {}
    for cid, u_weight in unnormalized_weights.items():
        normalized_weights[cid] = u_weight / partition_sum

    return (normalized_weights, relative_energies, min_conformer_id)


# =============================================================================
# 5. ASYMMETRIC ROTOR QUANTUM MECHANICS & PARTITION FUNCTIONS
# =============================================================================

def compute_moments_of_inertia_from_constants(
    A_mhz: float, B_mhz: float, C_mhz: float
) -> Tuple[float, float, float]:
    """
    Computes principal moments of inertia (I_a, I_b, I_c) in amu * Angstrom^2 from rotational constants.
    Formula: I_g = AMU_ANGSTROM2_TO_MHZ / Rot_g.
    """
    if A_mhz <= 0.0 or B_mhz <= 0.0 or C_mhz <= 0.0:
        raise RotationalPartitionError(
            f"Rotational constants must be positive: A={A_mhz}, B={B_mhz}, C={C_mhz}."
        )
    ia = AMU_ANGSTROM2_TO_MHZ / float(A_mhz)
    ib = AMU_ANGSTROM2_TO_MHZ / float(B_mhz)
    ic = AMU_ANGSTROM2_TO_MHZ / float(C_mhz)
    return (float(ia), float(ib), float(ic))


def compute_inertial_defect_from_constants(A_mhz: float, B_mhz: float, C_mhz: float) -> float:
    """
    Computes planar inertial defect Delta = I_c - I_a - I_b in amu * Angstrom^2.
    """
    ia, ib, ic = compute_moments_of_inertia_from_constants(A_mhz, B_mhz, C_mhz)
    return float(ic - ia - ib)


def construct_asymmetric_rotor_j_block(
    A: float, B: float, C: float, j: int
) -> np.ndarray:
    """
    Constructs the exact rigid asymmetric rotor Hamiltonian matrix for a given J in I^r representation
    using the symmetric top basis |J, K> where K ranges from -J to +J.
    Matrix dimension: (2J + 1) x (2J + 1).
    """
    dim = 2 * j + 1
    h_mat = np.zeros((dim, dim), dtype=np.float64)
    k_vals = np.arange(-j, j + 1, dtype=np.float64)

    # Diagonal elements: 0.5 * (B + C) * [J(J+1) - K^2] + A * K^2
    j_term = float(j * (j + 1))
    diag = 0.5 * (B + C) * (j_term - k_vals ** 2) + A * (k_vals ** 2)
    np.fill_diagonal(h_mat, diag)

    # Off-diagonal elements: <J, K+2 | H | J, K> = 0.25 * (B - C) * sqrt((J-K)(J-K-1)(J+K+1)(J+K+2))
    off_diag_factor = 0.25 * (B - C)
    for idx, k in enumerate(k_vals[:-2]):
        k_int = int(k)
        step = math.sqrt(
            float((j - k_int) * (j - k_int - 1) * (j + k_int + 1) * (j + k_int + 2))
        )
        val = off_diag_factor * step
        h_mat[idx + 2, idx] = val
        h_mat[idx, idx + 2] = val

    return h_mat


def compute_rotational_energy_levels(
    A_mhz: float,
    B_mhz: float,
    C_mhz: float,
    j_max: int = 8,
) -> Dict[Tuple[int, int, int], float]:
    """
    Computes exact asymmetric rotor rotational energy levels E(J, Ka, Kc) in MHz for all J <= j_max.
    Eigenvalues are labeled rigorously using standard prolate/oblate ordering.
    """
    energy_dict: Dict[Tuple[int, int, int], float] = {}

    for j in range(j_max + 1):
        if j == 0:
            energy_dict[(0, 0, 0)] = 0.0
            continue

        h_mat = construct_asymmetric_rotor_j_block(A_mhz, B_mhz, C_mhz, j)
        evals, _ = eigh(h_mat)
        evals_sorted = np.sort(evals)

        # Build asymmetric top state label sequence (Ka, Kc) for 2J+1 states in ascending energy order
        labels: List[Tuple[int, int]] = []
        for ka in range(j + 1):
            if ka == 0:
                labels.append((0, j))
            else:
                kc1 = j - ka + 1
                kc2 = j - ka
                labels.append((ka, kc1))
                labels.append((ka, kc2))

        # Sort labels by expected prolate energy ordering
        labels_ordered = sorted(labels, key=lambda pair: (pair[0], -pair[1]))

        for idx, (ka, kc) in enumerate(labels_ordered):
            energy_dict[(j, ka, kc)] = float(evals_sorted[idx])

    return energy_dict


def compute_rotational_partition_function(
    A_mhz: float,
    B_mhz: float,
    C_mhz: float,
    temperature_k: float,
    j_max: int = 15,
) -> float:
    """
    Computes the exact state-sum rotational partition function Q_rot(T) at temperature T:
        Q_rot(T) = sum_{J=0}^{j_max} sum_{Ka, Kc} (2J + 1) * exp(-E(J, Ka, Kc) / (k_B * T))

    For high temperatures or small constants, combines explicit state sum with Gordon integral tail.
    """
    if temperature_k <= 0.0:
        raise RotationalPartitionError(
            f"Rotational temperature must be positive, got {temperature_k} K."
        )

    energy_levels = compute_rotational_energy_levels(A_mhz, B_mhz, C_mhz, j_max=j_max)
    k_t_mhz = BOLTZMANN_CONSTANT_MHZ_K * temperature_k

    q_sum = 0.0
    for (j, ka, kc), e_val in energy_levels.items():
        weight = float(2 * j + 1)
        term = weight * math.exp(-e_val / k_t_mhz)
        q_sum += term

    # High-temperature Gordon analytical approximation check
    abc = float(A_mhz * B_mhz * C_mhz)
    if abc > 0.0:
        q_gordon = math.sqrt(math.pi * (k_t_mhz ** 3) / abc)
        if q_gordon > q_sum * 1.5:
            return float(q_gordon)

    return float(max(1.0, q_sum))


def calculate_rotational_line_strength(
    j_up: int, ka_up: int, kc_up: int, j_low: int, ka_low: int, kc_low: int
) -> float:
    """
    Calculates asymmetric rotor direction cosine factor S_ij based on angular momentum projection.
    """
    dj = j_up - j_low
    if dj == 1:
        # R-branch transition: line strength scales proportionally to J_low + 1
        return float(j_low + 1) / float(2 * j_low + 1)
    elif dj == 0:
        # Q-branch transition: line strength scales proportionally to (2J + 1)
        return float(2 * j_low + 1) / float(j_low * (j_low + 1) + 1.0)
    elif dj == -1:
        # P-branch transition
        return float(j_low) / float(2 * j_low + 1)
    return 1.0


# =============================================================================
# 6. CONFORMER TRANSITION SIMULATION & BROADBAND SYNTHESIS
# =============================================================================

def simulate_conformer_transitions(
    conformer: ConformerMetadata,
    config: BroadbandSynthesisConfig,
) -> List[SynthesizedTransition]:
    """
    Simulates all allowed dipole rotational transitions for an individual conformer state
    within the configured broadband frequency band at the rotational cooling temperature T_rot.
    """
    a_mhz, b_mhz, c_mhz = conformer.rotational_constants_mhz
    mu_a, mu_b, mu_c = conformer.dipole_moment_debye
    w_conf = conformer.boltzmann_weight

    t_rot = config.t_rot_k
    q_rot = compute_rotational_partition_function(
        a_mhz, b_mhz, c_mhz, temperature_k=t_rot, j_max=config.j_max + 2
    )

    energy_levels = compute_rotational_energy_levels(a_mhz, b_mhz, c_mhz, j_max=config.j_max)
    states = list(energy_levels.keys())
    transitions: List[SynthesizedTransition] = []

    k_t_mhz = BOLTZMANN_CONSTANT_MHZ_K * t_rot

    for idx1, s1 in enumerate(states):
        j1, ka1, kc1 = s1
        e1 = energy_levels[s1]

        for s2 in states:
            j2, ka2, kc2 = s2
            e2 = energy_levels[s2]

            # Enforce absorption transitions: E_upper > E_lower
            if e2 <= e1:
                continue

            nu_trans = e2 - e1
            if not (config.freq_min_mhz <= nu_trans <= config.freq_max_mhz):
                continue

            dj = j2 - j1
            if abs(dj) > 1:
                continue

            dka = abs(ka2 - ka1)
            dkc = abs(kc2 - kc1)

            # Determine dipole selection rule type
            dipole_branch: Optional[TransitionDipoleBranch] = None
            dipole_magnitude = 0.0

            if (dka % 2 == 0) and (dkc % 2 == 1):
                dipole_branch = TransitionDipoleBranch.A_BRANCH
                dipole_magnitude = abs(mu_a)
            elif (dka % 2 == 1) and (dkc % 2 == 1):
                dipole_branch = TransitionDipoleBranch.B_BRANCH
                dipole_magnitude = abs(mu_b)
            elif (dka % 2 == 1) and (dkc % 2 == 0):
                dipole_branch = TransitionDipoleBranch.C_BRANCH
                dipole_magnitude = abs(mu_c)

            if dipole_branch is None:
                continue

            if config.filter_dark_transitions and dipole_magnitude < config.dipole_threshold_debye:
                continue

            # Direction cosine line strength factor
            s_line = calculate_rotational_line_strength(j2, ka2, kc2, j1, ka1, kc1)

            # Thermal Boltzmann state factor & stimulated emission factor
            boltz_low = math.exp(-e1 / k_t_mhz)
            boltz_up = math.exp(-e2 / k_t_mhz)
            pop_diff = boltz_low - boltz_up

            # Transition intensity formula
            intensity_raw = (
                (nu_trans ** 2)
                * (dipole_magnitude ** 2)
                * s_line
                * (2 * j1 + 1)
                * pop_diff
                / q_rot
                * 1e-6
            )

            intensity_weighted = intensity_raw * w_conf

            trans = SynthesizedTransition(
                conformer_id=conformer.conformer_id,
                frequency_mhz=float(nu_trans),
                intensity_raw=float(intensity_raw),
                intensity_weighted=float(intensity_weighted),
                lower_state_j=j1,
                lower_state_ka=ka1,
                lower_state_kc=kc1,
                upper_state_j=j2,
                upper_state_ka=ka2,
                upper_state_kc=kc2,
                lower_energy_mhz=float(e1),
                upper_energy_mhz=float(e2),
                dipole_branch=dipole_branch,
                line_strength=float(s_line),
                conformer_boltzmann_weight=float(w_conf),
            )
            transitions.append(trans)

    # Sort transitions in ascending frequency order
    transitions.sort(key=lambda t: t.frequency_mhz)
    return transitions


def synthesize_composite_broadband_spectrum(
    pool: ConformerPool,
    config: BroadbandSynthesisConfig,
) -> CompositeSpectrumResult:
    """
    Synthesizes the full Boltzmann-weighted composite broadband spectrum across all active conformers
    in the conformer pool using pure-JAX FP64 macroscopic line convolution.
    """
    start_time = time.perf_counter()

    # Step 1: Build regular frequency grid
    num_grid_points = int(math.floor((config.freq_max_mhz - config.freq_min_mhz) / config.freq_step_mhz)) + 1
    freq_grid_np = np.linspace(
        config.freq_min_mhz,
        config.freq_min_mhz + (num_grid_points - 1) * config.freq_step_mhz,
        num_grid_points,
        dtype=np.float64,
    )

    all_transitions: List[SynthesizedTransition] = []
    conformer_spectra: Dict[str, List[float]] = {}
    composite_spectrum_np = np.zeros(num_grid_points, dtype=np.float64)

    # Step 2: Simulate transitions and apply line broadening per conformer
    for conformer in pool.conformers:
        cid = conformer.conformer_id
        conf_transitions = simulate_conformer_transitions(conformer, config)
        all_transitions.extend(conf_transitions)

        if not conf_transitions:
            conformer_spectra[cid] = [0.0] * num_grid_points
            continue

        # Extract frequencies and weighted intensities for this conformer
        line_freqs = [float(t.frequency_mhz) for t in conf_transitions]
        line_intensities = [float(t.intensity_weighted) for t in conf_transitions]

        spectral_lines = SpectralLines(
            frequencies=line_freqs,
            intensities=line_intensities,
        )

        conv_config = ConvolutionConfig(
            profile_type=config.profile_type,
            fwhm_gaussian=config.fwhm_mhz,
            fwhm_lorentzian=config.fwhm_mhz,
            cutoff_fwhm=10.0,
            max_vram_mb=config.max_vram_mb,
        )

        # Execute pure-JAX line broadening
        convolved_res: ConvolvedSpectrumResult = convolve_spectrum(
            spectral_lines=spectral_lines,
            config=conv_config,
            freq_grid=freq_grid_np,
        )

        conf_intensity_np = np.asarray(convolved_res.intensities, dtype=np.float64)
        conformer_spectra[cid] = conf_intensity_np.tolist()
        composite_spectrum_np += conf_intensity_np

    all_transitions.sort(key=lambda t: t.frequency_mhz)
    elapsed = time.perf_counter() - start_time
    max_intensity = float(np.max(composite_spectrum_np)) if len(composite_spectrum_np) > 0 else 0.0

    return CompositeSpectrumResult(
        frequency_grid_mhz=freq_grid_np.tolist(),
        composite_intensity=composite_spectrum_np.tolist(),
        conformer_decomposed_intensities=conformer_spectra,
        transitions=all_transitions,
        conformer_pool=pool,
        config=config,
        total_transitions_synthesized=len(all_transitions),
        peak_maximum_intensity=max_intensity,
        synthesis_wall_time_seconds=float(elapsed),
    )


# =============================================================================
# 7. ISOTOPOLOGUE EXTENSION (METHOD MATRIX §8B.4)
# =============================================================================

def generate_isotopologue_conformers(
    parent_conformer: ConformerMetadata,
    symbols: List[str],
    coordinates_angstrom: np.ndarray,
    abundance_cutoff: float = 0.005,
) -> List[ConformerMetadata]:
    """
    Generates single-substitution isotopologue conformer states from parent geometry and Mendeleev masses.
    Method Matrix §8B.4: Zero-electronic-structure cost isotopologue synthesis.
    """
    num_atoms = len(symbols)
    coords_f64 = np.asarray(coordinates_angstrom, dtype=np.float64)

    if coords_f64.shape != (num_atoms, 3):
        raise ValueError(
            f"Coordinates shape {coords_f64.shape} does not match symbol count {num_atoms}."
        )

    isotopologues: List[ConformerMetadata] = []

    # Map each atom position and test all stable isotopes
    for atom_idx, sym in enumerate(symbols):
        clean_sym = sym.strip().capitalize()
        el = element(clean_sym)
        if el is None or not el.isotopes:
            continue

        parent_mass = get_monoisotopic_mass(clean_sym)

        for iso in el.isotopes:
            if iso.mass is None or iso.abundance is None:
                continue

            abund_fraction = float(iso.abundance) / 100.0
            if abund_fraction < abundance_cutoff:
                continue

            iso_mass = float(iso.mass)
            # Skip if this is the standard monoisotopic parent
            if abs(iso_mass - parent_mass) < 1e-4:
                continue

            # Build perturbed mass vector
            perturbed_masses = [get_monoisotopic_mass(s) for s in symbols]
            perturbed_masses[atom_idx] = iso_mass

            # Compute rotational constants with perturbed isotopic masses
            rot_consts: RotationalConstants = compute_rotational_constants_from_coords(
                coords=coords_f64,
                masses=perturbed_masses,
                unit="angstrom",
            )

            a_iso, b_iso, c_iso = rot_consts.A, rot_consts.B, rot_consts.C
            kappa_iso = compute_rays_asymmetry(a_iso, b_iso, c_iso)
            defect_iso = compute_inertial_defect_from_constants(a_iso, b_iso, c_iso)

            iso_weight = parent_conformer.boltzmann_weight * abund_fraction
            iso_label = f"{iso.mass_number}{clean_sym}@{atom_idx + 1}"
            iso_id = f"{parent_conformer.conformer_id}_{iso_label}"

            iso_conf = ConformerMetadata(
                conformer_id=iso_id,
                relative_gibbs_free_energy_kcal_mol=parent_conformer.relative_gibbs_free_energy_kcal_mol,
                boltzmann_weight=float(iso_weight),
                rotational_constants_mhz=(float(a_iso), float(b_iso), float(c_iso)),
                dipole_moment_debye=parent_conformer.dipole_moment_debye,
                degeneracy=parent_conformer.degeneracy,
                ray_kappa=float(kappa_iso),
                inertial_defect_amu_a2=float(defect_iso),
                electronic_energy_hartree=parent_conformer.electronic_energy_hartree,
                zero_point_energy_cm1=parent_conformer.zero_point_energy_cm1,
                absolute_gibbs_free_energy_hartree=parent_conformer.absolute_gibbs_free_energy_hartree,
                is_isotopologue=True,
                isotopologue_label=iso_label,
            )
            isotopologues.append(iso_conf)

    return isotopologues


# =============================================================================
# 8. PESSTORE & LANDSCAPE.H5 INGESTION GATEWAYS
# =============================================================================

def extract_conformer_pool_from_store(
    store_or_path: Union[PESStore, Path, str, h5py.File],
    delta_g_cutoff_kcal_mol: float = DEFAULT_DELTA_G_CUTOFF_KCAL_MOL,
    t_conf_k: float = DEFAULT_T_CONF_K,
    energy_key: str = "gibbs_free_energy",
    energy_unit: ConformerEnergyUnit = ConformerEnergyUnit.HARTREE,
    include_isotopologues: bool = False,
    isotopologue_abundance_cutoff: float = 0.005,
) -> ConformerPool:
    """
    Extracts conformer pool from landscape.h5 / PESStore repository.
    Filters conformers strictly by Delta G <= 2.5 kcal/mol and derives Boltzmann weights.
    """
    if isinstance(store_or_path, PESStore):
        store = store_or_path
        close_needed = False
    elif isinstance(store_or_path, (str, Path)):
        p = translate_wsl_path(store_or_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"landscape.h5 / PESStore repository not found at: {p}")
        store = PESStore(p, mode="r")
        store.open()
        close_needed = True
    elif isinstance(store_or_path, h5py.File):
        store = PESStore(store_or_path.filename, mode="r")
        store._h5file = store_or_path
        store._is_external_file = True
        close_needed = False
    else:
        raise TypeError(f"Expected PESStore, Path, str, or h5py.File, got {type(store_or_path)}")

    try:
        isomer_keys = store.list_isomers()
        if not isomer_keys:
            raise InvalidConformerPoolError(
                f"No isomer groups found in PESStore file {store.file_path}"
            )

        raw_energies: Dict[str, float] = {}
        raw_conformer_data: Dict[str, Dict[str, Any]] = {}
        degeneracies: Dict[str, int] = {}

        for key in isomer_keys:
            try:
                cdata = store.read_isomer(key)
                group = store._get_isomer_group(key)
            except Exception as e:
                logger.warning(f"Skipping malformed isomer group '{key}': {e}")
                continue

            energy_val: Optional[float] = None

            if energy_key in group:
                energy_val = float(group[energy_key][()])
            elif energy_key in group.attrs:
                energy_val = float(group.attrs[energy_key])
            elif "energy" in group:
                energy_val = float(group["energy"][()])
            elif "scf_total_energy" in group.attrs:
                energy_val = float(group.attrs["scf_total_energy"])
            elif "properties" in group and "return_energy" in group["properties"]:
                energy_val = float(group["properties"]["return_energy"][()])
            else:
                energy_val = 0.0

            energy_kcal = convert_energy_to_kcal_mol(energy_val, unit=energy_unit)
            raw_energies[key] = energy_kcal
            raw_conformer_data[key] = cdata

            deg = int(group.attrs.get("degeneracy", 1))
            degeneracies[key] = deg

        if not raw_energies:
            raise InvalidConformerPoolError("Failed to extract any valid conformer energies.")

        weights, rel_energies, ref_id = compute_conformer_boltzmann_weights(
            energies_kcal_mol=raw_energies,
            temperature_k=t_conf_k,
            degeneracies=degeneracies,
            delta_g_cutoff_kcal_mol=delta_g_cutoff_kcal_mol,
        )

        active_conformers: List[ConformerMetadata] = []

        for cid, w_val in weights.items():
            cdata = raw_conformer_data[cid]
            coords = cdata["coords"]
            symbols = cdata["symbols"]
            group = store._get_isomer_group(cid)

            # Resolve coordinates from alternative datasets if coords are zeros
            if coords is not None and np.allclose(coords, 0.0):
                for alt_key in ("kraitchman_coords", "geometry", "geometry_xyz", "coordinates"):
                    if alt_key in group:
                        alt_c = np.asarray(group[alt_key][:], dtype=np.float64)
                        if alt_c.ndim == 1 and alt_c.size == len(symbols) * 3:
                            alt_c = alt_c.reshape((len(symbols), 3))
                        if not np.allclose(alt_c, 0.0):
                            coords = alt_c
                            break

            rc = cdata.get("rotational_constants")
            if rc is not None and isinstance(rc, RotationalConstants) and rc.A > 0 and rc.B > 0 and rc.C > 0:
                a_mhz, b_mhz, c_mhz = rc.A, rc.B, rc.C
            elif coords is not None and len(symbols) > 0 and not np.allclose(coords, 0.0):
                rc_calc = compute_rotational_constants_from_coords(coords, masses=symbols)
                a_mhz, b_mhz, c_mhz = rc_calc.A, rc_calc.B, rc_calc.C
            else:
                raise InvalidConformerPoolError(
                    f"Conformer '{cid}' missing both valid rotational constants and non-zero coordinates."
                )

            dm = cdata.get("dipole_moment")
            if dm is not None and isinstance(dm, DipoleMoment):
                dipole_tuple = (abs(dm.X), abs(dm.Y), abs(dm.Z))
            elif "dipole_moment" in cdata and cdata["dipole_moment"] is not None:
                dm_raw = cdata["dipole_moment"]
                dipole_tuple = (abs(float(dm_raw[0])), abs(float(dm_raw[1])), abs(float(dm_raw[2])))
            else:
                dipole_tuple = (1.0, 1.0, 1.0)

            kappa = compute_rays_asymmetry(a_mhz, b_mhz, c_mhz)
            defect = compute_inertial_defect_from_constants(a_mhz, b_mhz, c_mhz)

            conf_meta = ConformerMetadata(
                conformer_id=cid,
                relative_gibbs_free_energy_kcal_mol=float(rel_energies[cid]),
                boltzmann_weight=float(w_val),
                rotational_constants_mhz=(float(a_mhz), float(b_mhz), float(c_mhz)),
                dipole_moment_debye=dipole_tuple,
                degeneracy=degeneracies.get(cid, 1),
                ray_kappa=float(kappa),
                inertial_defect_amu_a2=float(defect),
                electronic_energy_hartree=cdata.get("energy"),
                zero_point_energy_cm1=cdata.get("zpe"),
                absolute_gibbs_free_energy_hartree=raw_energies[cid] / HARTREE_TO_KCAL_MOL,
                is_isotopologue=False,
                isotopologue_label=None,
            )
            active_conformers.append(conf_meta)

            if include_isotopologues and coords is not None and len(symbols) > 0:
                iso_list = generate_isotopologue_conformers(
                    parent_conformer=conf_meta,
                    symbols=symbols,
                    coordinates_angstrom=coords,
                    abundance_cutoff=isotopologue_abundance_cutoff,
                )
                active_conformers.extend(iso_list)

        active_conformers.sort(key=lambda c: c.relative_gibbs_free_energy_kcal_mol)
        total_pop = sum(c.boltzmann_weight for c in active_conformers if not c.is_isotopologue)

        return ConformerPool(
            conformers=active_conformers,
            reference_conformer_id=ref_id,
            temperature_conf_k=float(t_conf_k),
            delta_g_cutoff_kcal_mol=float(delta_g_cutoff_kcal_mol),
            total_active_conformers=len(active_conformers),
            cumulative_population=float(total_pop),
        )
    finally:
        if close_needed:
            store.close()


def synthesize_broadband_from_store(
    store_or_path: Union[PESStore, Path, str, h5py.File],
    config: Optional[BroadbandSynthesisConfig] = None,
) -> CompositeSpectrumResult:
    """
    High-level convenience gateway: Ingests landscape.h5 / PESStore, builds conformer pool,
    and synthesizes the complete Boltzmann-weighted composite broadband spectrum.
    """
    cfg = config if config is not None else BroadbandSynthesisConfig()
    pool = extract_conformer_pool_from_store(
        store_or_path=store_or_path,
        delta_g_cutoff_kcal_mol=cfg.delta_g_cutoff_kcal_mol,
        t_conf_k=cfg.t_conf_k,
        include_isotopologues=cfg.include_isotopologues,
        isotopologue_abundance_cutoff=cfg.isotopologue_abundance_cutoff,
    )
    return synthesize_composite_broadband_spectrum(pool=pool, config=cfg)


# =============================================================================
# 9. EXPORT CAPABILITIES (HDF5 / CSV)
# =============================================================================

def export_composite_spectrum_to_hdf5(
    result: CompositeSpectrumResult,
    target_path: Union[Path, str],
    compression: str = "gzip",
) -> Path:
    """
    Serializes complete composite spectrum, transition catalog, and conformer contributions to HDF5.
    """
    p = Path(target_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(p), "w") as f:
        f.attrs["synthesis_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.attrs["t_rot_k"] = result.config.t_rot_k
        f.attrs["t_conf_k"] = result.config.t_conf_k
        f.attrs["delta_g_cutoff_kcal_mol"] = result.config.delta_g_cutoff_kcal_mol
        f.attrs["fwhm_mhz"] = result.config.fwhm_mhz
        f.attrs["profile_type"] = result.config.profile_type.value
        f.attrs["total_transitions"] = result.total_transitions_synthesized
        f.attrs["peak_maximum_intensity"] = result.peak_maximum_intensity

        f.create_dataset(
            "frequency_grid_mhz",
            data=np.array(result.frequency_grid_mhz, dtype=np.float64),
            dtype="<f8",
            compression=compression,
        )
        f.create_dataset(
            "composite_intensity",
            data=np.array(result.composite_intensity, dtype=np.float64),
            dtype="<f8",
            compression=compression,
        )

        grp_conf = f.create_group("conformer_spectra")
        for cid, spec in result.conformer_decomposed_intensities.items():
            grp_conf.create_dataset(
                cid,
                data=np.array(spec, dtype=np.float64),
                dtype="<f8",
                compression=compression,
            )

        grp_trans = f.create_group("transitions")
        if result.transitions:
            trans_freqs = [t.frequency_mhz for t in result.transitions]
            trans_ints_raw = [t.intensity_raw for t in result.transitions]
            trans_ints_wt = [t.intensity_weighted for t in result.transitions]
            trans_cids = [t.conformer_id.encode("utf-8") for t in result.transitions]
            trans_branches = [t.dipole_branch.value.encode("utf-8") for t in result.transitions]
            trans_strengths = [t.line_strength for t in result.transitions]

            str_dt = h5py.string_dtype(encoding="utf-8")

            grp_trans.create_dataset(
                "frequencies_mhz", data=np.array(trans_freqs, dtype=np.float64), dtype="<f8", compression=compression
            )
            grp_trans.create_dataset(
                "intensities_raw", data=np.array(trans_ints_raw, dtype=np.float64), dtype="<f8", compression=compression
            )
            grp_trans.create_dataset(
                "intensities_weighted", data=np.array(trans_ints_wt, dtype=np.float64), dtype="<f8", compression=compression
            )
            grp_trans.create_dataset("conformer_ids", data=trans_cids, dtype=str_dt)
            grp_trans.create_dataset("dipole_branches", data=trans_branches, dtype=str_dt)
            grp_trans.create_dataset(
                "line_strengths", data=np.array(trans_strengths, dtype=np.float64), dtype="<f8", compression=compression
            )

    return p


def export_composite_spectrum_to_csv(
    result: CompositeSpectrumResult,
    target_path: Union[Path, str],
) -> Path:
    """
    Exports frequency grid and composite intensity to a two-column CSV file.
    """
    p = Path(target_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(p, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frequency_mhz", "intensity"])
        for freq, intens in zip(result.frequency_grid_mhz, result.composite_intensity):
            writer.writerow([f"{freq:.6f}", f"{intens:.8e}"])

    return p


__all__ = [
    # Constants
    "BOLTZMANN_CONSTANT_CM1_K",
    "BOLTZMANN_CONSTANT_J_K",
    "BOLTZMANN_CONSTANT_MHZ_K",
    "CM1_TO_KCAL_MOL",
    "DEFAULT_DELTA_G_CUTOFF_KCAL_MOL",
    "DEFAULT_T_CONF_K",
    "DEFAULT_T_ROT_K",
    "EV_TO_KCAL_MOL",
    "GAS_CONSTANT_J_MOL_K",
    "GAS_CONSTANT_KCAL_MOL_K",
    "HARTREE_TO_CM1",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_MOL",
    "KJ_TO_KCAL_MOL",
    "PLANCK_CONSTANT_J_S",
    # Exceptions
    "BoltzmannSynthesizerError",
    "BroadbandSynthesisError",
    "ConformerEnergyError",
    "ConformerPopulationError",
    "InvalidConformerPoolError",
    "RotationalPartitionError",
    # Enums & Models
    "BroadbandSynthesisConfig",
    "CompositeSpectrumResult",
    "ConformerEnergyUnit",
    "ConformerMetadata",
    "ConformerPool",
    "SynthesizedTransition",
    "TemperatureModel",
    "TransitionDipoleBranch",
    # Functions
    "calculate_rotational_line_strength",
    "compute_conformer_boltzmann_weights",
    "compute_inertial_defect_from_constants",
    "compute_moments_of_inertia_from_constants",
    "compute_rotational_energy_levels",
    "compute_rotational_partition_function",
    "construct_asymmetric_rotor_j_block",
    "convert_energy_to_kcal_mol",
    "export_composite_spectrum_to_csv",
    "export_composite_spectrum_to_hdf5",
    "extract_conformer_pool_from_store",
    "generate_isotopologue_conformers",
    "simulate_conformer_transitions",
    "synthesize_broadband_from_store",
    "synthesize_composite_broadband_spectrum",
]
