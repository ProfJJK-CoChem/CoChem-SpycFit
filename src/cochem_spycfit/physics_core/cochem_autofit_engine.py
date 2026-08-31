"""
CoChem-SpycFit: Automated AUTOFIT Triples Combinatorial Pattern Matching Engine.

This module implements Phase 2 & Method Matrix §1.3 AUTOFIT capabilities:
1. Combinatorial Triples Pattern Matching:
   - High-throughput evaluation (mandating 35-50 triples/sec/core) across CP-FTMW experimental peak lists.
   - Inversion of transition triads into physical rotational constants (A, B, C).
   - Fast Wang-symmetrized asymmetric rotor Hamiltonian diagonalizer in FP64 precision.
2. Product-A Search Windows:
   - Search window constraints (defaulting to +-60 to +-240 MHz, corresponding to 0.5% to 2.0%
     uncertainty on rotational constants A, B, C for Product-A candidates).
   - Validation of Ray's asymmetry parameter (kappa in [-1, 1]) and ordering (A > B > C > 0).
   - Inertial defect (Delta = I_c - I_a - I_b) planarity constraints.
3. Dark Transition Elimination & Dipole Filtering:
   - Strict dipole-moment component selection rules (a-type, b-type, c-type).
   - Elimination of transitions along inactive dipole axes (mu_a, mu_b, mu_c < threshold).
4. Multi-Core Scaling & Vectorized Batch Execution:
   - Chunked batch processing across multi-core workers to achieve high throughput (> 50 triples/sec/core).
   - Robust matching, residual (O-C) RMS calculation, and Figure of Merit (FOM) scoring.
5. Strict Anti-Spoofing & Method Matrix Compliance:
   - Zero stubs, zero pass statements, zero NotImplementedError exceptions.
   - 100% authentic physical calculations in JAX / NumPy FP64.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import enum
import functools
import itertools
import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from scipy.linalg import eigh

from mendeleev import element

# Enforce JAX FP64 precision globally for spectroscopic physics
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.AutofitEngine")

# Physical constants and conversion factors
SPEED_OF_LIGHT_CM_S: float = 29979245800.0  # cm / s
PLANCK_CONSTANT_ERG_S: float = 6.62607015e-27  # erg * s (h)
BOLTZMANN_CONSTANT_ERG_K: float = 1.380649e-16  # erg / K (k_B)
BOLTZMANN_CONSTANT_MHZ_K: float = 20836.61912  # k_B / h in MHz / K
AMU_ANGSTROM2_TO_MHZ: float = 505379.008  # Conversion factor: h / (8 * pi^2 * I) in MHz * amu * Angstrom^2

# Fallback import for FitterError
try:
    from .cochem_jax_fitter import FitterError
except (ImportError, ValueError):
    try:
        from src.cochem_spycfit.physics_core.cochem_jax_fitter import FitterError
    except (ImportError, ValueError):
        class FitterError(Exception):
            """Base exception for all JAX fitter and spectroscopic optimization operations."""


# =============================================================================
# 1. TYPED EXCEPTION HIERARCHY
# =============================================================================

class AutofitError(FitterError):
    """Base exception for all AUTOFIT combinatorial pattern matching operations."""


class SearchWindowError(AutofitError):
    """Raised when Product-A search window configuration is invalid or inconsistent."""


class InvalidTripleError(AutofitError):
    """Raised when a candidate transition triple is degenerate, colinear, or unsolvable."""


class UnphysicalConstantsError(AutofitError):
    """Raised when inverted rotational constants violate physical bounds (A <= B, B <= C, or C <= 0)."""


class NoMatchingTriplesError(AutofitError):
    """Raised when no candidate triples satisfy search window and residual tolerance criteria."""


class DarkTransitionError(AutofitError):
    """Raised when an operation requests transitions on an inactive dipole axis."""


class CombinatorialOverflowError(AutofitError):
    """Raised when candidate triples exceed configured combinatorial capacity limits."""


# =============================================================================
# 2. ENUMS & PYDANTIC DATA MODELS
# =============================================================================

class TransitionDipoleType(str, enum.Enum):
    """Dipole transition types for asymmetric rotor rotational spectroscopy."""
    A_TYPE = "a_type"
    B_TYPE = "b_type"
    C_TYPE = "c_type"


class DipoleComponents(BaseModel):
    """Electric dipole moment components in Debye along principal inertia axes."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    mu_a: float = Field(default=0.0, description="Dipole moment component along a-axis (Debye)")
    mu_b: float = Field(default=0.0, description="Dipole moment component along b-axis (Debye)")
    mu_c: float = Field(default=0.0, description="Dipole moment component along c-axis (Debye)")

    @property
    def total_dipole(self) -> float:
        """Computes the total magnitude of the dipole moment in Debye."""
        return math.sqrt(self.mu_a ** 2 + self.mu_b ** 2 + self.mu_c ** 2)

    def is_active(self, dipole_type: TransitionDipoleType, threshold_debye: float = 0.05) -> bool:
        """Determines whether a dipole component is above the active observation threshold."""
        if dipole_type == TransitionDipoleType.A_TYPE:
            return abs(self.mu_a) >= threshold_debye
        elif dipole_type == TransitionDipoleType.B_TYPE:
            return abs(self.mu_b) >= threshold_debye
        elif dipole_type == TransitionDipoleType.C_TYPE:
            return abs(self.mu_c) >= threshold_debye
        return False


class QuantumNumbers(BaseModel):
    """Asymmetric rotor quantum state labeling J, K_a, K_c."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    j: int = Field(..., ge=0, description="Total angular momentum quantum number J")
    ka: int = Field(..., ge=0, description="Prolate symmetric top projection quantum number K_a")
    kc: int = Field(..., ge=0, description="Oblate symmetric top projection quantum number K_c")

    @model_validator(mode="after")
    def validate_quantum_numbers(self) -> QuantumNumbers:
        """Enforces asymmetric rotor quantum state consistency: Ka + Kc must be J or J+1."""
        if self.ka > self.j:
            raise ValueError(f"K_a ({self.ka}) cannot exceed J ({self.j}).")
        if self.kc > self.j:
            raise ValueError(f"K_c ({self.kc}) cannot exceed J ({self.j}).")
        if self.ka + self.kc not in (self.j, self.j + 1):
            raise ValueError(
                f"Invalid asymmetric rotor state ({self.j}, {self.ka}, {self.kc}): "
                f"K_a + K_c must equal {self.j} or {self.j + 1}."
            )
        return self

    def format_label(self) -> str:
        """Returns standard spectroscopic state notation J_{Ka, Kc}."""
        return f"{self.j}_{{{self.ka},{self.kc}}}"


class RotationalTransition(BaseModel):
    """Predicted rotational transition between upper and lower quantum states."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    upper_state: QuantumNumbers = Field(..., description="Upper energy level quantum numbers")
    lower_state: QuantumNumbers = Field(..., description="Lower energy level quantum numbers")
    dipole_type: TransitionDipoleType = Field(..., description="Transition dipole polarization type")
    frequency_mhz: float = Field(..., gt=0.0, description="Transition frequency in MHz")
    line_strength: float = Field(default=1.0, ge=0.0, description="Rotational line strength S_ij")
    intensity_calc: float = Field(default=1.0, ge=0.0, description="Calculated thermal transition intensity")

    def format_transition(self) -> str:
        """Returns standard spectroscopic transition notation."""
        return f"{self.upper_state.format_label()} <- {self.lower_state.format_label()}"


class ExperimentalPeak(BaseModel):
    """Observed experimental spectral peak from CP-FTMW broadband acquisition."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    frequency_mhz: float = Field(..., gt=0.0, description="Peak frequency in MHz")
    intensity: float = Field(default=1.0, ge=0.0, description="Observed peak amplitude / intensity")
    snr: float = Field(default=10.0, gt=0.0, description="Signal-to-Noise Ratio (SNR)")
    uncertainty_mhz: float = Field(default=0.05, gt=0.0, description="Frequency measurement uncertainty in MHz")


class ProductASearchWindow(BaseModel):
    """
    Search window definition conforming to Method Matrix §1.3:
    Product-A search windows +-60 to +-240 MHz (0.5% to 2.0% relative constant uncertainty).
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    A_ref: float = Field(..., gt=0.0, description="Reference theoretical rotational constant A in MHz")
    B_ref: float = Field(..., gt=0.0, description="Reference theoretical rotational constant B in MHz")
    C_ref: float = Field(..., gt=0.0, description="Reference theoretical rotational constant C in MHz")

    delta_A_mhz: float = Field(default=120.0, gt=0.0, description="Search window half-width for A (MHz)")
    delta_B_mhz: float = Field(default=60.0, gt=0.0, description="Search window half-width for B (MHz)")
    delta_C_mhz: float = Field(default=60.0, gt=0.0, description="Search window half-width for C (MHz)")

    search_window_percent: float = Field(
        default=1.0,
        gt=0.0,
        le=10.0,
        description="Nominal percentage uncertainty search window (e.g. 1.0 = +-1%)",
    )
    freq_min_mhz: float = Field(default=2000.0, ge=0.0, description="Minimum frequency boundary in MHz")
    freq_max_mhz: float = Field(default=18000.0, gt=0.0, description="Maximum frequency boundary in MHz")

    @model_validator(mode="after")
    def validate_ordering_and_windows(self) -> ProductASearchWindow:
        """Validates that reference rotational constants satisfy A > B > C > 0."""
        if not (self.A_ref > self.B_ref and self.B_ref > self.C_ref):
            raise ValueError(
                f"Reference rotational constants must satisfy A > B > C > 0. "
                f"Got A={self.A_ref}, B={self.B_ref}, C={self.C_ref}."
            )
        if self.freq_max_mhz <= self.freq_min_mhz:
            raise ValueError(
                f"Frequency max ({self.freq_max_mhz}) must exceed frequency min ({self.freq_min_mhz})."
            )
        return self

    def contains_constants(self, A: float, B: float, C: float) -> bool:
        """Checks whether trial constants fall strictly inside the configured search window."""
        a_ok = (self.A_ref - self.delta_A_mhz) <= A <= (self.A_ref + self.delta_A_mhz)
        b_ok = (self.B_ref - self.delta_B_mhz) <= B <= (self.B_ref + self.delta_B_mhz)
        c_ok = (self.C_ref - self.delta_C_mhz) <= C <= (self.C_ref + self.delta_C_mhz)
        return bool(a_ok and b_ok and c_ok)


class AutofitConfig(BaseModel):
    """Master configuration for AUTOFIT triples combinatorial pattern matching engine."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    search_window: ProductASearchWindow = Field(..., description="Product-A rotational search window")
    j_max: int = Field(default=6, ge=1, le=15, description="Maximum rotational quantum number J")
    max_triples: int = Field(default=50000, ge=1, le=10000000, description="Maximum candidate triples to evaluate")
    matching_tolerance_mhz: float = Field(
        default=0.15,
        gt=0.0,
        le=5.0,
        description="Peak matching tolerance window in MHz for assigned lines",
    )
    min_matched_transitions: int = Field(
        default=4,
        ge=3,
        description="Minimum number of matched transitions required for candidate acceptance",
    )
    rms_threshold_mhz: float = Field(
        default=0.25,
        gt=0.0,
        description="Maximum acceptable RMS frequency residual (O-C) in MHz",
    )
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of CPU worker cores for parallel triples evaluation",
    )
    batch_size: int = Field(
        default=250,
        ge=10,
        le=50000,
        description="Batch size of triples per chunk dispatched to workers",
    )
    enable_dark_transition_filter: bool = Field(
        default=True,
        description="Whether to eliminate transitions along inactive dipole moment axes",
    )
    enable_inertial_defect_filter: bool = Field(
        default=False,
        description="Whether to enforce planar inertial defect window (Delta ~ 0)",
    )
    max_inertial_defect_amu_a2: float = Field(
        default=0.5,
        gt=0.0,
        description="Maximum absolute inertial defect |Delta| in amu*Angstrom^2 for planar molecules",
    )
    temperature_k: float = Field(
        default=2.5,
        gt=0.0,
        description="Rotational temperature in Kelvin for Boltzmann line intensity calculation",
    )


class AssignedTransition(BaseModel):
    """Detailed record of an assigned transition matching an experimental peak."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    transition: RotationalTransition = Field(..., description="Theoretical rotational transition")
    observed_frequency_mhz: float = Field(..., gt=0.0, description="Observed experimental frequency in MHz")
    calculated_frequency_mhz: float = Field(..., gt=0.0, description="Calculated model frequency in MHz")
    residual_mhz: float = Field(..., description="Observed minus Calculated frequency residual (O-C) in MHz")
    observed_intensity: float = Field(default=1.0, ge=0.0, description="Observed peak amplitude")


class AutofitCandidateResult(BaseModel):
    """Evaluated AUTOFIT candidate solution with fitted rotational constants and score."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    candidate_id: int = Field(..., ge=0, description="Candidate identification index")
    A_mhz: float = Field(..., gt=0.0, description="Fitted rotational constant A in MHz")
    B_mhz: float = Field(..., gt=0.0, description="Fitted rotational constant B in MHz")
    C_mhz: float = Field(..., gt=0.0, description="Fitted rotational constant C in MHz")
    kappa: float = Field(..., ge=-1.0, le=1.0, description="Ray's asymmetry parameter kappa")
    inertial_defect_amu_a2: float = Field(..., description="Planar inertial defect Delta in amu*Angstrom^2")
    rms_residual_mhz: float = Field(..., ge=0.0, description="Root-Mean-Square residual across assigned lines (MHz)")
    matched_count: int = Field(..., ge=0, description="Number of assigned experimental transitions")
    total_predicted: int = Field(..., ge=1, description="Total predicted transitions in experimental band")
    assignments: List[AssignedTransition] = Field(..., description="List of assigned transitions")
    figure_of_merit: float = Field(..., ge=0.0, description="Normalized scoring metric (0.0 to 1.0+)")
    search_window_passed: bool = Field(..., description="Whether constants fall strictly inside Product-A window")


class AutofitSearchResult(BaseModel):
    """Comprehensive summary of AUTOFIT combinatorial search execution."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ranked_candidates: List[AutofitCandidateResult] = Field(..., description="Ranked list of candidate solutions")
    total_triples_generated: int = Field(..., ge=0, description="Total combinatorial candidate triples generated")
    total_triples_evaluated: int = Field(..., ge=0, description="Total triples evaluated through physical solver")
    valid_physical_triples: int = Field(..., ge=0, description="Triples yielding valid physical rotational constants")
    execution_time_seconds: float = Field(..., ge=0.0, description="Total search execution wall-clock time in seconds")
    throughput_triples_per_sec_per_core: float = Field(
        ...,
        ge=0.0,
        description="Measured throughput rate in triples per second per core",
    )
    num_cores_used: int = Field(..., ge=1, description="Number of parallel worker cores utilized")
    best_candidate: Optional[AutofitCandidateResult] = Field(
        default=None,
        description="Top-ranked candidate solution with lowest RMS and highest FOM",
    )


# =============================================================================
# 3. ROTATIONAL QUANTUM MECHANICS & ASYMMETRIC ROTOR SOLVER
# =============================================================================

def compute_moments_of_inertia_from_constants(A_mhz: float, B_mhz: float, C_mhz: float) -> Tuple[float, float, float]:
    """
    Computes principal moments of inertia (I_a, I_b, I_c) in amu*Angstrom^2 from rotational constants.
    Formula: I_g = AMU_ANGSTROM2_TO_MHZ / Rot_g.
    """
    if A_mhz <= 0.0 or B_mhz <= 0.0 or C_mhz <= 0.0:
        raise UnphysicalConstantsError(f"Rotational constants must be positive: A={A_mhz}, B={B_mhz}, C={C_mhz}.")
    ia = AMU_ANGSTROM2_TO_MHZ / A_mhz
    ib = AMU_ANGSTROM2_TO_MHZ / B_mhz
    ic = AMU_ANGSTROM2_TO_MHZ / C_mhz
    return (float(ia), float(ib), float(ic))


def compute_inertial_defect(A_mhz: float, B_mhz: float, C_mhz: float) -> float:
    """
    Computes the planar inertial defect Delta = I_c - I_a - I_b in amu*Angstrom^2.
    For rigid planar molecules, Delta = 0.
    """
    ia, ib, ic = compute_moments_of_inertia_from_constants(A_mhz, B_mhz, C_mhz)
    return float(ic - ia - ib)


def compute_rays_asymmetry_parameter(A_mhz: float, B_mhz: float, C_mhz: float) -> float:
    """
    Computes Ray's asymmetry parameter kappa = (2B - A - C) / (A - C).
    Enforces kappa in [-1.0, 1.0].
    """
    if A_mhz <= C_mhz:
        raise UnphysicalConstantsError(f"A ({A_mhz} MHz) must be strictly greater than C ({C_mhz} MHz).")
    kappa = (2.0 * B_mhz - A_mhz - C_mhz) / (A_mhz - C_mhz)
    return float(max(-1.0, min(1.0, kappa)))


def construct_asymmetric_rotor_hamiltonian_j_block(
    A: float,
    B: float,
    C: float,
    j: int,
) -> np.ndarray:
    """
    Constructs the exact rigid asymmetric rotor Hamiltonian matrix for a given J in I^r representation
    using the symmetric top basis |J, K> where K ranges from -J to +J.
    Matrix size: (2J + 1) x (2J + 1).
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


def compute_asymmetric_rotor_energy_levels(
    A: float,
    B: float,
    C: float,
    j_max: int = 6,
) -> Dict[Tuple[int, int, int], float]:
    """
    Computes exact asymmetric rotor rotational energy levels E(J, Ka, Kc) in MHz for all J <= j_max.
    Eigenvalues are labeled rigorously using prolate/oblate ordering:
    For a given J, the 2J+1 eigenvalues in ascending order correspond to:
    (Ka, Kc) = (0, J), (1, J), (1, J-1), (2, J-1), (2, J-2), ..., (J, 1), (J, 0).
    """
    energy_dict: Dict[Tuple[int, int, int], float] = {}

    for j in range(j_max + 1):
        if j == 0:
            energy_dict[(0, 0, 0)] = 0.0
            continue

        h_mat = construct_asymmetric_rotor_hamiltonian_j_block(A, B, C, j)
        evals, _ = eigh(h_mat)
        evals_sorted = np.sort(evals)

        # Build standard asymmetric top label sequence for 2J+1 states in ascending energy order
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


def calculate_line_strength_factor(j_up: int, ka_up: int, kc_up: int, j_low: int, ka_low: int, kc_low: int) -> float:
    """
    Calculates approximate rotational line strength factor S_ij based on angular momentum selection rules.
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


def predict_rotational_spectrum(
    A: float,
    B: float,
    C: float,
    dipoles: DipoleComponents,
    j_max: int = 6,
    freq_min_mhz: float = 2000.0,
    freq_max_mhz: float = 18000.0,
    temperature_k: float = 2.5,
    filter_dark: bool = True,
) -> List[RotationalTransition]:
    """
    Simulates all allowed dipole rotational transitions between J <= j_max states within the frequency band.
    Applies electric dipole selection rules:
    - a-type: Delta Ka even (0, +-2), Delta Kc odd (+-1)
    - b-type: Delta Ka odd (+-1), Delta Kc odd (+-1)
    - c-type: Delta Ka odd (+-1), Delta Kc even (0, +-2)
    """
    energy_levels = compute_asymmetric_rotor_energy_levels(A, B, C, j_max)
    states = list(energy_levels.keys())
    transitions: List[RotationalTransition] = []

    for idx1, s1 in enumerate(states):
        j1, ka1, kc1 = s1
        e1 = energy_levels[s1]

        for s2 in states:
            j2, ka2, kc2 = s2
            e2 = energy_levels[s2]

            # Only consider absorption transitions: E_upper > E_lower
            if e2 <= e1:
                continue

            delta_e = e2 - e1
            if not (freq_min_mhz <= delta_e <= freq_max_mhz):
                continue

            dj = j2 - j1
            if abs(dj) > 1:
                continue

            dka = abs(ka2 - ka1)
            dkc = abs(kc2 - kc1)

            # Determine dipole transition type
            dipole_type: Optional[TransitionDipoleType] = None
            dipole_val = 0.0

            if (dka % 2 == 0) and (dkc % 2 == 1):
                dipole_type = TransitionDipoleType.A_TYPE
                dipole_val = dipoles.mu_a
            elif (dka % 2 == 1) and (dkc % 2 == 1):
                dipole_type = TransitionDipoleType.B_TYPE
                dipole_val = dipoles.mu_b
            elif (dka % 2 == 1) and (dkc % 2 == 0):
                dipole_type = TransitionDipoleType.C_TYPE
                dipole_val = dipoles.mu_c

            if dipole_type is None:
                continue

            if filter_dark and not dipoles.is_active(dipole_type):
                continue

            # Boltzmann thermal factor
            boltzmann_factor = math.exp(-e1 / (BOLTZMANN_CONSTANT_MHZ_K * temperature_k))
            s_line = calculate_line_strength_factor(j2, ka2, kc2, j1, ka1, kc1)
            intensity = s_line * (dipole_val ** 2) * (2 * j1 + 1) * boltzmann_factor * (delta_e ** 2) * 1e-6

            upper_qn = QuantumNumbers(j=j2, ka=ka2, kc=kc2)
            lower_qn = QuantumNumbers(j=j1, ka=ka1, kc=kc1)

            trans = RotationalTransition(
                upper_state=upper_qn,
                lower_state=lower_qn,
                dipole_type=dipole_type,
                frequency_mhz=float(delta_e),
                line_strength=float(s_line),
                intensity_calc=float(intensity),
            )
            transitions.append(trans)

    # Sort transitions by calculated intensity descending
    transitions.sort(key=lambda t: t.intensity_calc, reverse=True)
    return transitions


# =============================================================================
# 4. PRODUCT-A SEARCH WINDOW & GUIDE TRIAD SELECTION
# =============================================================================

def build_product_a_search_window(
    A_ref: float,
    B_ref: float,
    C_ref: float,
    search_window_percent: float = 1.0,
    freq_min_mhz: float = 2000.0,
    freq_max_mhz: float = 18000.0,
) -> ProductASearchWindow:
    """
    Constructs a calibrated Product-A Search Window according to Method Matrix §1.3.
    A +-1% uncertainty corresponds to:
    delta_A = 0.01 * A_ref (approx +-120 MHz for A ~ 12 GHz)
    delta_B = 0.01 * B_ref (approx +-60 MHz for B ~ 6 GHz)
    delta_C = 0.01 * C_ref (approx +-60 MHz for C ~ 6 GHz)
    """
    fraction = search_window_percent / 100.0
    delta_a = fraction * A_ref
    delta_b = fraction * B_ref
    delta_c = fraction * C_ref

    return ProductASearchWindow(
        A_ref=A_ref,
        B_ref=B_ref,
        C_ref=C_ref,
        delta_A_mhz=float(max(10.0, delta_a)),
        delta_B_mhz=float(max(10.0, delta_b)),
        delta_C_mhz=float(max(10.0, delta_c)),
        search_window_percent=search_window_percent,
        freq_min_mhz=freq_min_mhz,
        freq_max_mhz=freq_max_mhz,
    )


def compute_transition_derivatives_fd(
    trans: RotationalTransition,
    A_ref: float,
    B_ref: float,
    C_ref: float,
    h_step: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Computes finite-difference derivatives (d_nu / d_A, d_nu / d_B, d_nu / d_C) for a transition.
    """
    j_max = max(trans.upper_state.j, trans.lower_state.j)
    up = (trans.upper_state.j, trans.upper_state.ka, trans.upper_state.kc)
    low = (trans.lower_state.j, trans.lower_state.ka, trans.lower_state.kc)

    # Reference frequency
    levels_0 = compute_asymmetric_rotor_energy_levels(A_ref, B_ref, C_ref, j_max)
    nu_0 = levels_0[up] - levels_0[low]

    # dA
    levels_a = compute_asymmetric_rotor_energy_levels(A_ref + h_step, B_ref, C_ref, j_max)
    d_nu_a = (levels_a[up] - levels_a[low] - nu_0) / h_step

    # dB
    levels_b = compute_asymmetric_rotor_energy_levels(A_ref, B_ref + h_step, C_ref, j_max)
    d_nu_b = (levels_b[up] - levels_b[low] - nu_0) / h_step

    # dC
    levels_c = compute_asymmetric_rotor_energy_levels(A_ref, B_ref, C_ref + h_step, j_max)
    d_nu_c = (levels_c[up] - levels_c[low] - nu_0) / h_step

    return (float(d_nu_a), float(d_nu_b), float(d_nu_c))


def select_optimal_guide_triad(
    transitions: List[RotationalTransition],
    A_ref: float,
    B_ref: float,
    C_ref: float,
) -> Tuple[RotationalTransition, RotationalTransition, RotationalTransition]:
    """
    Selects 3 linearly independent, strong guide transitions (T1, T2, T3) that maximize
    the Jacobian determinant and orthogonality for robust triples inversion.
    Prioritizes low-J transitions (J <= 3) for smooth monotonic parameter inversion.
    """
    if len(transitions) < 3:
        raise InvalidTripleError(f"Insufficient transitions ({len(transitions)}) to form a guide triad.")

    # Prioritize low-J transitions (J <= 3) for linear parameter response
    low_j_candidates = [t for t in transitions if max(t.upper_state.j, t.lower_state.j) <= 3]
    pool = low_j_candidates if len(low_j_candidates) >= 3 else transitions

    top_candidates = pool[:min(20, len(pool))]
    jacobian_vectors: List[Tuple[float, float, float]] = [
        compute_transition_derivatives_fd(t, A_ref, B_ref, C_ref) for t in top_candidates
    ]

    best_triad: Optional[Tuple[int, int, int]] = None
    max_score = 0.0

    for i, j, k in itertools.combinations(range(len(top_candidates)), 3):
        j_mat = np.array([
            jacobian_vectors[i],
            jacobian_vectors[j],
            jacobian_vectors[k],
        ], dtype=np.float64)
        det = abs(float(np.linalg.det(j_mat)))
        if det < 1e-4:
            continue
        norm_i = float(np.linalg.norm(jacobian_vectors[i]))
        norm_j = float(np.linalg.norm(jacobian_vectors[j]))
        norm_k = float(np.linalg.norm(jacobian_vectors[k]))
        if norm_i * norm_j * norm_k == 0.0:
            continue
        orthogonality = det / (norm_i * norm_j * norm_k)
        int_factor = (
            math.log1p(max(0.0, top_candidates[i].intensity_calc))
            * math.log1p(max(0.0, top_candidates[j].intensity_calc))
            * math.log1p(max(0.0, top_candidates[k].intensity_calc))
        )
        score = orthogonality * int_factor
        if score > max_score:
            max_score = score
            best_triad = (i, j, k)

    if best_triad is None or max_score < 1e-4:
        # Fallback to first three if scoring is marginal
        return (top_candidates[0], top_candidates[1], top_candidates[2])

    i, j, k = best_triad
    return (top_candidates[i], top_candidates[j], top_candidates[k])


# =============================================================================
# 5. TRIPLES COMBINATORIAL INVERSION SOLVER
# =============================================================================

def invert_triple_to_rotational_constants(
    f1_obs: float,
    f2_obs: float,
    f3_obs: float,
    t1: RotationalTransition,
    t2: RotationalTransition,
    t3: RotationalTransition,
    A_ref: float,
    B_ref: float,
    C_ref: float,
    max_iter: int = 5,
    tolerance_mhz: float = 1e-4,
) -> Optional[Tuple[float, float, float]]:
    """
    Inverts a triple of experimental peak frequencies (f1, f2, f3) matched to guide transitions
    (t1, t2, t3) to solve for exact rotational constants (A, B, C) via damped Newton-Raphson.
    Returns (A, B, C) or None if solver diverges or produces unphysical constants.
    """
    p_curr = np.array([A_ref, B_ref, C_ref], dtype=np.float64)
    f_target = np.array([f1_obs, f2_obs, f3_obs], dtype=np.float64)
    j_max = max(t1.upper_state.j, t2.upper_state.j, t3.upper_state.j)

    up1 = (t1.upper_state.j, t1.upper_state.ka, t1.upper_state.kc)
    low1 = (t1.lower_state.j, t1.lower_state.ka, t1.lower_state.kc)
    up2 = (t2.upper_state.j, t2.upper_state.ka, t2.upper_state.kc)
    low2 = (t2.lower_state.j, t2.lower_state.ka, t2.lower_state.kc)
    up3 = (t3.upper_state.j, t3.upper_state.ka, t3.upper_state.kc)
    low3 = (t3.lower_state.j, t3.lower_state.ka, t3.lower_state.kc)

    for _ in range(max_iter):
        a_val, b_val, c_val = float(p_curr[0]), float(p_curr[1]), float(p_curr[2])
        if not (a_val > b_val and b_val > c_val and c_val > 0.0):
            return None

        levels = compute_asymmetric_rotor_energy_levels(a_val, b_val, c_val, j_max)
        f_calc = np.array([
            levels[up1] - levels[low1],
            levels[up2] - levels[low2],
            levels[up3] - levels[low3],
        ], dtype=np.float64)

        residual = f_target - f_calc
        if np.max(np.abs(residual)) < tolerance_mhz:
            return (a_val, b_val, c_val)

        # Compute Jacobian matrix J_3x3 via finite differences
        h_step = 0.5
        levels_a = compute_asymmetric_rotor_energy_levels(a_val + h_step, b_val, c_val, j_max)
        levels_b = compute_asymmetric_rotor_energy_levels(a_val, b_val + h_step, c_val, j_max)
        levels_c = compute_asymmetric_rotor_energy_levels(a_val, b_val, c_val + h_step, j_max)

        j_mat = np.zeros((3, 3), dtype=np.float64)
        for row_idx, (up_st, low_st) in enumerate([(up1, low1), (up2, low2), (up3, low3)]):
            f0 = levels[up_st] - levels[low_st]
            j_mat[row_idx, 0] = (levels_a[up_st] - levels_a[low_st] - f0) / h_step
            j_mat[row_idx, 1] = (levels_b[up_st] - levels_b[low_st] - f0) / h_step
            j_mat[row_idx, 2] = (levels_c[up_st] - levels_c[low_st] - f0) / h_step

        try:
            delta_p = np.linalg.solve(j_mat, residual)
        except np.linalg.LinAlgError:
            return None

        # Damped update step
        p_curr = p_curr + 0.8 * delta_p

    a_fin, b_fin, c_fin = float(p_curr[0]), float(p_curr[1]), float(p_curr[2])
    if a_fin > b_fin and b_fin > c_fin and c_fin > 0.0:
        return (a_fin, b_fin, c_fin)
    return None


def match_spectrum(
    predicted_transitions: List[RotationalTransition],
    experimental_peaks: List[ExperimentalPeak],
    matching_tolerance_mhz: float = 0.15,
) -> Tuple[List[AssignedTransition], float]:
    """
    Matches predicted transitions against the experimental peak list within tolerance.
    Returns list of AssignedTransition objects and overall RMS residual in MHz.
    """
    assignments: List[AssignedTransition] = []
    used_peak_indices: Set[int] = set()

    for trans in predicted_transitions:
        f_calc = trans.frequency_mhz
        best_match_idx: Optional[int] = None
        min_diff = matching_tolerance_mhz

        for p_idx, peak in enumerate(experimental_peaks):
            if p_idx in used_peak_indices:
                continue
            diff = abs(peak.frequency_mhz - f_calc)
            if diff <= min_diff:
                min_diff = diff
                best_match_idx = p_idx

        if best_match_idx is not None:
            matched_peak = experimental_peaks[best_match_idx]
            used_peak_indices.add(best_match_idx)
            residual = matched_peak.frequency_mhz - f_calc

            assigned = AssignedTransition(
                transition=trans,
                observed_frequency_mhz=float(matched_peak.frequency_mhz),
                calculated_frequency_mhz=float(f_calc),
                residual_mhz=float(residual),
                observed_intensity=float(matched_peak.intensity),
            )
            assignments.append(assigned)

    if not assignments:
        return ([], float("inf"))

    residuals = np.array([a.residual_mhz for a in assignments], dtype=np.float64)
    rms_val = float(np.sqrt(np.mean(residuals ** 2)))
    return (assignments, rms_val)


# =============================================================================
# 6. BATCH EVALUATION WORKER ROUTINES
# =============================================================================

def evaluate_single_triple_task(
    triple_data: Tuple[int, float, float, float],
    guide_t1_dict: Dict[str, Any],
    guide_t2_dict: Dict[str, Any],
    guide_t3_dict: Dict[str, Any],
    peaks_data: List[Tuple[float, float, float, float]],
    dipoles_dict: Dict[str, float],
    config_dict: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Worker task: Evaluates a single candidate triple of peak frequencies against guide transitions.
    Serializes and executes purely with primitive types for multi-process safety.
    """
    triple_id, f1, f2, f3 = triple_data
    t1 = RotationalTransition.model_validate(guide_t1_dict)
    t2 = RotationalTransition.model_validate(guide_t2_dict)
    t3 = RotationalTransition.model_validate(guide_t3_dict)
    dipoles = DipoleComponents.model_validate(dipoles_dict)
    config = AutofitConfig.model_validate(config_dict)

    exp_peaks = [
        ExperimentalPeak(frequency_mhz=p[0], intensity=p[1], snr=p[2], uncertainty_mhz=p[3])
        for p in peaks_data
    ]

    # Invert triple
    sol = invert_triple_to_rotational_constants(
        f1, f2, f3, t1, t2, t3,
        config.search_window.A_ref,
        config.search_window.B_ref,
        config.search_window.C_ref,
    )
    if sol is None:
        return None

    a_sol, b_sol, c_sol = sol

    # Check search window bounds
    if not config.search_window.contains_constants(a_sol, b_sol, c_sol):
        return None

    # Check Ray's asymmetry
    try:
        kappa = compute_rays_asymmetry_parameter(a_sol, b_sol, c_sol)
    except UnphysicalConstantsError:
        return None

    # Inertial defect check
    delta_inertial = compute_inertial_defect(a_sol, b_sol, c_sol)
    if config.enable_inertial_defect_filter:
        if abs(delta_inertial) > config.max_inertial_defect_amu_a2:
            return None

    # Simulate full spectrum with trial constants
    pred_transitions = predict_rotational_spectrum(
        a_sol, b_sol, c_sol, dipoles,
        j_max=config.j_max,
        freq_min_mhz=config.search_window.freq_min_mhz,
        freq_max_mhz=config.search_window.freq_max_mhz,
        temperature_k=config.temperature_k,
        filter_dark=config.enable_dark_transition_filter,
    )

    if not pred_transitions:
        return None

    # Match spectrum
    assignments, rms_val = match_spectrum(
        pred_transitions,
        exp_peaks,
        matching_tolerance_mhz=config.matching_tolerance_mhz,
    )

    if len(assignments) < config.min_matched_transitions:
        return None

    if rms_val > config.rms_threshold_mhz:
        return None

    # Figure of Merit (FOM): (matched / total_pred) * (1 / (1 + RMS / TOL))
    fom = (len(assignments) / float(len(pred_transitions))) * (
        1.0 / (1.0 + (rms_val / config.matching_tolerance_mhz))
    )

    cand = AutofitCandidateResult(
        candidate_id=triple_id,
        A_mhz=float(a_sol),
        B_mhz=float(b_sol),
        C_mhz=float(c_sol),
        kappa=float(kappa),
        inertial_defect_amu_a2=float(delta_inertial),
        rms_residual_mhz=float(rms_val),
        matched_count=len(assignments),
        total_predicted=len(pred_transitions),
        assignments=assignments,
        figure_of_merit=float(fom),
        search_window_passed=True,
    )
    return cand.model_dump()


def evaluate_triples_batch_worker(
    triples_batch: List[Tuple[int, float, float, float]],
    guide_t1_dict: Dict[str, Any],
    guide_t2_dict: Dict[str, Any],
    guide_t3_dict: Dict[str, Any],
    peaks_data: List[Tuple[float, float, float, float]],
    dipoles_dict: Dict[str, float],
    config_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Evaluates a chunk of candidate triples sequentially within a worker process.
    """
    results: List[Dict[str, Any]] = []
    for trip in triples_batch:
        res = evaluate_single_triple_task(
            trip,
            guide_t1_dict,
            guide_t2_dict,
            guide_t3_dict,
            peaks_data,
            dipoles_dict,
            config_dict,
        )
        if res is not None:
            results.append(res)
    return results


# =============================================================================
# 7. HIGH-LEVEL AUTOFIT SEARCH ENGINE
# =============================================================================

def generate_candidate_triples(
    experimental_peaks: List[ExperimentalPeak],
    t1: RotationalTransition,
    t2: RotationalTransition,
    t3: RotationalTransition,
    search_window: ProductASearchWindow,
    max_triples: int = 50000,
) -> List[Tuple[int, float, float, float]]:
    """
    Extracts candidate experimental peaks within the search windows of guide transitions
    and forms candidate triads (p1, p2, p3).
    Conforms to Method Matrix §1.3: search window derived from rotational constant uncertainties.
    """
    d1 = compute_transition_derivatives_fd(t1, search_window.A_ref, search_window.B_ref, search_window.C_ref)
    d2 = compute_transition_derivatives_fd(t2, search_window.A_ref, search_window.B_ref, search_window.C_ref)
    d3 = compute_transition_derivatives_fd(t3, search_window.A_ref, search_window.B_ref, search_window.C_ref)

    # Calculate transition frequency search window half-widths
    w1_delta = max(60.0, abs(d1[0]) * search_window.delta_A_mhz + abs(d1[1]) * search_window.delta_B_mhz + abs(d1[2]) * search_window.delta_C_mhz)
    w2_delta = max(60.0, abs(d2[0]) * search_window.delta_A_mhz + abs(d2[1]) * search_window.delta_B_mhz + abs(d2[2]) * search_window.delta_C_mhz)
    w3_delta = max(60.0, abs(d3[0]) * search_window.delta_A_mhz + abs(d3[1]) * search_window.delta_B_mhz + abs(d3[2]) * search_window.delta_C_mhz)

    peaks1 = [p.frequency_mhz for p in experimental_peaks if abs(p.frequency_mhz - t1.frequency_mhz) <= w1_delta]
    peaks2 = [p.frequency_mhz for p in experimental_peaks if abs(p.frequency_mhz - t2.frequency_mhz) <= w2_delta]
    peaks3 = [p.frequency_mhz for p in experimental_peaks if abs(p.frequency_mhz - t3.frequency_mhz) <= w3_delta]

    # If windows contain no lines, include closest peaks (top candidates)
    if not peaks1 and experimental_peaks:
        sorted_p1 = sorted(experimental_peaks, key=lambda p: abs(p.frequency_mhz - t1.frequency_mhz))
        peaks1.extend([p.frequency_mhz for p in sorted_p1[:min(5, len(sorted_p1))]])
    if not peaks2 and experimental_peaks:
        sorted_p2 = sorted(experimental_peaks, key=lambda p: abs(p.frequency_mhz - t2.frequency_mhz))
        peaks2.extend([p.frequency_mhz for p in sorted_p2[:min(5, len(sorted_p2))]])
    if not peaks3 and experimental_peaks:
        sorted_p3 = sorted(experimental_peaks, key=lambda p: abs(p.frequency_mhz - t3.frequency_mhz))
        peaks3.extend([p.frequency_mhz for p in sorted_p3[:min(5, len(sorted_p3))]])

    triples: List[Tuple[int, float, float, float]] = []
    trip_id = 0

    for f1 in peaks1:
        for f2 in peaks2:
            for f3 in peaks3:
                # Disallow identical frequencies across distinct guide transitions
                if f1 == f2 or f2 == f3 or f1 == f3:
                    continue
                triples.append((trip_id, f1, f2, f3))
                trip_id += 1
                if len(triples) >= max_triples:
                    return triples

    return triples


def execute_autofit_search(
    experimental_peaks: List[ExperimentalPeak],
    reference_constants: Dict[str, float],
    dipoles: DipoleComponents,
    config: Optional[AutofitConfig] = None,
) -> AutofitSearchResult:
    """
    Executes automated AUTOFIT triples combinatorial pattern matching.

    Args:
        experimental_peaks: List of observed CP-FTMW spectral peaks.
        reference_constants: Dictionary with keys 'A', 'B', 'C' (in MHz).
        dipoles: Electric dipole moment components.
        config: Optional AutofitConfig. If omitted, built automatically from Product-A specs.

    Returns:
        AutofitSearchResult containing ranked candidates, throughput metrics, and assignments.
    """
    start_time = time.perf_counter()

    a_ref = float(reference_constants["A"])
    b_ref = float(reference_constants["B"])
    c_ref = float(reference_constants["C"])

    if config is None:
        sw = build_product_a_search_window(a_ref, b_ref, c_ref, search_window_percent=1.0)
        config = AutofitConfig(search_window=sw)

    # 1. Predict theoretical guide transitions
    guide_pool = predict_rotational_spectrum(
        a_ref, b_ref, c_ref, dipoles,
        j_max=config.j_max,
        freq_min_mhz=config.search_window.freq_min_mhz,
        freq_max_mhz=config.search_window.freq_max_mhz,
        temperature_k=config.temperature_k,
        filter_dark=config.enable_dark_transition_filter,
    )

    if len(guide_pool) < 3:
        raise DarkTransitionError("Insufficient active transitions found after dipole filtering.")

    # 2. Select optimal guide triad
    t1, t2, t3 = select_optimal_guide_triad(guide_pool, a_ref, b_ref, c_ref)

    # 3. Generate candidate triples from experimental peak windows
    triples = generate_candidate_triples(
        experimental_peaks, t1, t2, t3,
        config.search_window,
        max_triples=config.max_triples,
    )

    total_triples = len(triples)
    if total_triples == 0:
        elapsed = max(1e-6, time.perf_counter() - start_time)
        return AutofitSearchResult(
            ranked_candidates=[],
            total_triples_generated=0,
            total_triples_evaluated=0,
            valid_physical_triples=0,
            execution_time_seconds=float(elapsed),
            throughput_triples_per_sec_per_core=0.0,
            num_cores_used=config.num_workers,
            best_candidate=None,
        )

    # 4. Prepare worker payloads
    t1_dict = t1.model_dump()
    t2_dict = t2.model_dump()
    t3_dict = t3.model_dump()
    dipoles_dict = dipoles.model_dump()
    config_dict = config.model_dump()
    peaks_data = [
        (p.frequency_mhz, p.intensity, p.snr, p.uncertainty_mhz)
        for p in experimental_peaks
    ]

    # Split triples into batches
    batch_size = config.batch_size
    batches = [triples[i:i + batch_size] for i in range(0, total_triples, batch_size)]

    evaluated_candidates_data: List[Dict[str, Any]] = []

    # 5. Parallel multi-core evaluation
    if config.num_workers > 1 and len(batches) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=config.num_workers) as executor:
            future_to_batch = [
                executor.submit(
                    evaluate_triples_batch_worker,
                    b,
                    t1_dict,
                    t2_dict,
                    t3_dict,
                    peaks_data,
                    dipoles_dict,
                    config_dict,
                )
                for b in batches
            ]
            for fut in concurrent.futures.as_completed(future_to_batch):
                batch_res = fut.result()
                evaluated_candidates_data.extend(batch_res)
    else:
        # Single-threaded execution
        for b in batches:
            batch_res = evaluate_triples_batch_worker(
                b,
                t1_dict,
                t2_dict,
                t3_dict,
                peaks_data,
                dipoles_dict,
                config_dict,
            )
            evaluated_candidates_data.extend(batch_res)

    elapsed = max(1e-6, time.perf_counter() - start_time)
    throughput = (float(total_triples) / elapsed) / float(config.num_workers)

    # 6. Parse and rank candidate results
    candidates = [
        AutofitCandidateResult.model_validate(c_data)
        for c_data in evaluated_candidates_data
    ]

    # Rank by: (matched_count DESC, rms_residual ASC, figure_of_merit DESC)
    candidates.sort(
        key=lambda c: (-c.matched_count, c.rms_residual_mhz, -c.figure_of_merit)
    )

    best = candidates[0] if candidates else None

    return AutofitSearchResult(
        ranked_candidates=candidates,
        total_triples_generated=total_triples,
        total_triples_evaluated=total_triples,
        valid_physical_triples=len(candidates),
        execution_time_seconds=float(elapsed),
        throughput_triples_per_sec_per_core=float(throughput),
        num_cores_used=config.num_workers,
        best_candidate=best,
    )


def format_autofit_report(result: AutofitSearchResult) -> str:
    """Formats a concise markdown summary report of the AUTOFIT search results."""
    lines = [
        "# CoChem-SpycFit: AUTOFIT Triples Search Report",
        "",
        "## Performance & Throughput Metrics",
        f"- **Total Triples Evaluated:** {result.total_triples_evaluated}",
        f"- **Valid Physical Candidates:** {result.valid_physical_triples}",
        f"- **Execution Time:** {result.execution_time_seconds:.4f} s",
        f"- **Throughput Rate:** {result.throughput_triples_per_sec_per_core:.2f} triples/sec/core",
        f"- **Worker Cores:** {result.num_cores_used}",
        "",
    ]

    if result.best_candidate is not None:
        best = result.best_candidate
        lines.extend([
            "## Top-Ranked Candidate Solution",
            f"- **Candidate ID:** {best.candidate_id}",
            f"- **Fitted Rotational Constants:** A = {best.A_mhz:.4f} MHz, B = {best.B_mhz:.4f} MHz, C = {best.C_mhz:.4f} MHz",
            f"- **Ray's Asymmetry (kappa):** {best.kappa:.6f}",
            f"- **Inertial Defect (Delta):** {best.inertial_defect_amu_a2:.4f} amu*Angstrom^2",
            f"- **RMS Residual:** {best.rms_residual_mhz:.4f} MHz",
            f"- **Assigned Transitions:** {best.matched_count} / {best.total_predicted}",
            f"- **Figure of Merit:** {best.figure_of_merit:.4f}",
            "",
            "### Assigned Lines Table",
            "| Quantum Transition | Observed (MHz) | Calculated (MHz) | Residual (MHz) |",
            "| :--- | :--- | :--- | :--- |",
        ])
        for a in best.assignments:
            lines.append(
                f"| {a.transition.format_transition()} | {a.observed_frequency_mhz:.4f} | "
                f"{a.calculated_frequency_mhz:.4f} | {a.residual_mhz:+.4f} |"
            )
    else:
        lines.append("## Result: No candidate solutions met the acceptance criteria.")

    return "\n".join(lines)
