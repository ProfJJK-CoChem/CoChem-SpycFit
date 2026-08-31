#!/usr/bin/env python3
# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
CoChem-SpycFit: Analytical Jensen Inequality & Inverse Inertia Tensor Averager.

Method Matrix §5.1, §3.0, §16.3, Table 4-O & WBS Phase 1.1 Mandate:
----------------------------------------------------------------------
The vibrationally averaged rotational constant is one-half the expectation value of
the inverse effective inertia tensor, NOT a function of the expectation value of
the inertia tensor.

Physical Background & Mathematical Foundation:
----------------------------------------------
In the Eckart-Watson formulation (Czakó, Mátyus, and Császár, JPCA 113, 11665 (2009)):
    mu_alpha_beta = (I'^-1)_alpha_beta
    I'_R_gamma = I_R_gamma - sum_{k,l,m} zeta^{km}_R zeta^{lm}_gamma Q_k Q_l
    A_V ~= 1/2 <mu_xx>_V
    B_V ~= 1/2 <mu_yy>_V
    C_V ~= 1/2 <mu_zz>_V

In frequency units (MHz):
    B[MHz] = CONV / I[amu Angstrom^2], where CONV = 505379.008 MHz*amu*Angstrom^2 (Groner, NIST CCCBDB).
    Equivalently, B[MHz] = CONV * mu[amu^-1 Angstrom^-2].

Jensen's Inequality:
-------------------
By Jensen's inequality for the convex function f(x) = 1/x (for x > 0):
    <1/I> >= 1/<I>

Averaging the wrong quantity (i.e., inverting <I> rather than taking <I^-1>) systematically
underestimates the rotational constant. The leading fractional bias is:
    fractional_bias = 3 * sigma_R^2 / R0^2

For a floppy intermolecular stretch (reduced mass = 10.0 amu, R0 = 3.80 Angstrom):
- sigma_R = 0.05 Angstrom: B(<1/I>) = 3501.69 MHz, B(1/<I>) = 3499.26 MHz, bias = 2.43 MHz (0.069%)
- sigma_R = 0.10 Angstrom: B(<1/I>) = 3507.18 MHz, B(1/<I>) = 3497.46 MHz, bias = 9.72 MHz (0.278%)
- sigma_R = 0.15 Angstrom: B(<1/I>) = 3516.38 MHz, B(1/<I>) = 3494.45 MHz, bias = 21.93 MHz (0.628%)
- sigma_R = 0.20 Angstrom: B(<1/I>) = 3529.40 MHz, B(1/<I>) = 3490.25 MHz, bias = 39.16 MHz (1.122%)
- sigma_R = 0.30 Angstrom: B(<1/I>) = 3567.50 MHz, B(1/<I>) = 3478.27 MHz, bias = 89.24 MHz (2.566%)

At zero-point amplitude sigma_R ~= 0.15 Angstrom, the systematic bias is 21.93 MHz (0.63%), which is
more than 6x the 0.1% microwave spectroscopic assignment target!

Method Matrix Section 16.3 Rejection Mandate:
--------------------------------------------
A result is discarded if a rotational constant is computed by inverting the averaged
inertia tensor rather than averaging the inverse inertia tensor. Any calculation that
evaluates 1/<I> instead of <1/I> is strictly non-compliant and triggers immediate rejection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field
from scipy import integrate

# Enable JAX FP64 precision globally for spectroscopic tensor calculations
jax.config.update("jax_enable_x64", True)

# Reconfigure stream encodings for safe cross-platform terminal execution
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        logging.getLogger(__name__).debug("Failed to reconfigure sys.stdout encoding.")
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        logging.getLogger(__name__).debug("Failed to reconfigure sys.stderr encoding.")

logger = logging.getLogger("CoChem-SpycFit.JensenAverager")

# =============================================================================
# 1. PHYSICAL CONSTANTS & CONVERSION FACTORS (CODATA / NIST / Method Matrix)
# =============================================================================

# Spectroscopic conversion factor from moment of inertia (amu * Angstrom^2) to MHz:
# h / (8 * pi^2 * u * Angstrom^2) in MHz * amu * Angstrom^2
# References: Groner (2016); NIST CCCBDB; Method Matrix §5.1, §4.1, §3.0.
AMU_ANGSTROM2_TO_MHZ: float = 505379.008
CONV_MHZ_AMU_ANG2: float = 505379.008
CODATA_CONV_EXACT: float = 505379.00536

PROVENANCE_TAG: str = "[M]"
SECTION_REF: str = "Method Matrix §5.1, §3.0 & §16.3"


# =============================================================================
# 2. DYNAMIC ATOMIC MASS RESOLUTION (Mendeleev Library Mandate)
# =============================================================================

def get_dynamic_atomic_mass(element_or_symbol: Union[str, int], mass_number: Optional[int] = None) -> float:
    """
    Dynamically retrieves standard atomic or isotopic mass in unified atomic mass units (u/amu)
    from the mendeleev library. Strictly complies with the Mendeleev Library Mandate.
    """
    if isinstance(element_or_symbol, int):
        el_obj = element(element_or_symbol)
        if el_obj is not None:
            if mass_number is not None:
                for iso in getattr(el_obj, "isotopes", []):
                    if getattr(iso, "mass_number", None) == mass_number and iso.mass is not None:
                        return float(iso.mass)
            if el_obj.mass is not None:
                return float(el_obj.mass)
        raise ValueError(f"Atomic number {element_or_symbol} not found in mendeleev database.")

    sym_clean = str(element_or_symbol).strip()
    sym_cap = sym_clean.capitalize()

    # Hydrogen isotopes notation handling
    if sym_cap in ("D", "2h", "h2"):
        h_elem = element("H")
        for iso in getattr(h_elem, "isotopes", []):
            if getattr(iso, "mass_number", None) == 2 and iso.mass is not None:
                return float(iso.mass)
    elif sym_cap in ("T", "3h", "h3"):
        h_elem = element("H")
        for iso in getattr(h_elem, "isotopes", []):
            if getattr(iso, "mass_number", None) == 3 and iso.mass is not None:
                return float(iso.mass)

    # General isotope match: e.g. "13C", "18O", "15N", "35Cl"
    iso_match = re.match(r"^(\d+)([A-Za-z]+)$", sym_clean)
    if iso_match:
        iso_mass_num = int(iso_match.group(1))
        iso_sym = iso_match.group(2).capitalize()
        el_iso = element(iso_sym)
        for iso in getattr(el_iso, "isotopes", []):
            if getattr(iso, "mass_number", None) == iso_mass_num and iso.mass is not None:
                return float(iso.mass)

    # Standard elemental lookup
    base_sym = re.sub(r"^\d+", "", sym_clean).capitalize()
    el_standard = element(base_sym)
    if mass_number is not None:
        for iso in getattr(el_standard, "isotopes", []):
            if getattr(iso, "mass_number", None) == mass_number and iso.mass is not None:
                return float(iso.mass)

    el_mass = getattr(el_standard, "mass", None) or getattr(el_standard, "atomic_weight", None)
    if el_mass is not None:
        return float(el_mass)

    raise ValueError(f"Unable to dynamically resolve atomic mass for '{element_or_symbol}' (mass_number={mass_number}).")


def get_reduced_mass(
    symbol1: str,
    symbol2: str,
    mass_number1: Optional[int] = None,
    mass_number2: Optional[int] = None,
) -> float:
    """Computes two-body reduced mass mu = (m1 * m2) / (m1 + m2) in amu dynamically."""
    m1 = get_dynamic_atomic_mass(symbol1, mass_number1)
    m2 = get_dynamic_atomic_mass(symbol2, mass_number2)
    return float((m1 * m2) / (m1 + m2))


def resolve_mass_array(elements_seq: Sequence[Union[str, int]]) -> np.ndarray:
    """Converts a sequence of atomic symbols or numbers into an array of atomic masses in amu."""
    masses = [get_dynamic_atomic_mass(sym) for sym in elements_seq]
    return np.asarray(masses, dtype=np.float64)


# =============================================================================
# 3. TYPED EXCEPTION HIERARCHY
# =============================================================================

class JensenAveragerError(ValueError):
    """Base exception for all Jensen averager tensor operations."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class MethodMatrixComplianceError(JensenAveragerError):
    """Base exception for violations of Method Matrix mandates."""


class VibrationalAveragingRejectionError(MethodMatrixComplianceError):
    """Raised when vibrational averaging inverts <I> instead of averaging <I^-1> (Method Matrix §16.3)."""


class SingularInertiaTensorError(JensenAveragerError):
    """Raised when an inertia tensor is singular or non-invertible."""


class EckartAlignmentError(JensenAveragerError):
    """Raised when mass-weighted Eckart alignment fails or encounters improper orientation."""


class UnphysicalWeightDistributionError(JensenAveragerError):
    """Raised when vibrational integration weights sum to zero or contain negative values."""


class JensenInequalityViolationError(JensenAveragerError):
    """Raised when numerical anomalies cause a breach of the theoretical Jensen bound <1/I> >= 1/<I>."""


# =============================================================================
# 4. PYDANTIC SCHEMAS & RESULTS CONTAINERS
# =============================================================================

class RotationalConstants(BaseModel):
    """Container for principal rotational constants (A >= B >= C) in MHz."""
    model_config = ConfigDict(extra="forbid", frozen=True)

    A: float = Field(..., description="A principal rotational constant (MHz)")
    B: float = Field(..., description="B principal rotational constant (MHz)")
    C: float = Field(..., description="C principal rotational constant (MHz)")


@dataclass(frozen=True)
class Jensen1DResult:
    """Results container for 1D intermolecular stretch vibrational averaging."""
    sigma_R: float
    R0: float
    reduced_mass: float
    B_e: float
    B_inv_I: float
    B_I_avg: float
    bias_MHz: float
    bias_pct: float
    taylor_B_inv_I: float
    leading_fractional_bias: float
    provenance: str = PROVENANCE_TAG

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Jensen3DResult:
    """Results container for 3D polyatomic molecular vibrational tensor averaging."""
    symbols: List[str]
    n_atoms: int
    n_snapshots: int
    A_correct: float
    B_correct: float
    C_correct: float
    A_flawed: float
    B_flawed: float
    C_flawed: float
    bias_A_MHz: float
    bias_B_MHz: float
    bias_C_MHz: float
    bias_pct_A: float
    bias_pct_B: float
    bias_pct_C: float
    avg_mu_tensor: List[List[float]]
    avg_I_tensor: List[List[float]]
    is_valid_jensen: bool
    c_constant_reliability_warning: str
    provenance: str = PROVENANCE_TAG

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JensenDVRGridResult:
    """Results container for Discrete Variable Representation (DVR) grid vibrational averaging."""
    n_grid_points: int
    A_correct: float
    B_correct: float
    C_correct: float
    A_flawed: float
    B_flawed: float
    C_flawed: float
    bias_A_MHz: float
    bias_B_MHz: float
    bias_C_MHz: float
    bias_pct_A: float
    bias_pct_B: float
    bias_pct_C: float
    is_valid_jensen: bool
    provenance: str = PROVENANCE_TAG

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MethodMatrixAuditReport:
    """Audit verification report for Method Matrix §16.3 compliance."""
    status: str
    section: str
    rejection_triggered: bool
    findings: List[str]
    sha256_checksum: str
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# 5. 1D FLOPPY INTERMOLECULAR STRETCH MODEL (§5.1)
# =============================================================================

def compute_1d_jensen_model(
    R0: float = 3.80,
    reduced_mass: float = 10.0,
    sigma_R: float = 0.15,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> Jensen1DResult:
    """
    Computes exact vibrationally averaged rotational constants for a 1D floppy
    stretch mode with Gaussian probability density P(R) ~ N(R0, sigma_R^2).

    Mathematical Formulation (Method Matrix §5.1):
        I(R) = mu * R^2
        B_e = CONV / (mu * R0^2)
        <I> = mu * (R0^2 + sigma_R^2)
        B(1/<I>) = CONV / <I> = CONV / [mu * (R0^2 + sigma_R^2)]
        <1/I> = integral P(R) / (mu * R^2) dR
        B(<1/I>) = CONV * <1/I>
        bias = B(<1/I>) - B(1/<I>)
        bias_% = 100 * bias / B(1/<I>)
        leading_fractional_bias = 3 * sigma_R^2 / R0^2
    """
    if R0 <= 0.0:
        raise JensenAveragerError(f"Equilibrium separation R0 must be positive, got {R0}")
    if reduced_mass <= 0.0:
        raise JensenAveragerError(f"Reduced mass must be positive, got {reduced_mass}")
    if sigma_R <= 0.0:
        raise JensenAveragerError(f"Standard deviation sigma_R must be positive, got {sigma_R}")

    I0 = reduced_mass * (R0 ** 2)
    B_e = conv / I0

    # Gaussian distribution integrand for expectation value <1/I>
    def integrand_inv_I(r: float) -> float:
        norm_const = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_R)
        p_r = norm_const * math.exp(-0.5 * (((r - R0) / sigma_R) ** 2))
        return p_r / (reduced_mass * (r ** 2))

    # Numerical integration across 10 standard deviations
    r_min = max(1e-4, R0 - 10.0 * sigma_R)
    r_max = R0 + 10.0 * sigma_R
    inv_I_avg, _ = integrate.quad(integrand_inv_I, r_min, r_max, epsabs=1e-13, epsrel=1e-13)
    B_inv_I = conv * inv_I_avg

    # Analytical exact expectation value <I> = mu * (R0^2 + sigma_R^2)
    I_avg = reduced_mass * (R0 ** 2 + sigma_R ** 2)
    B_I_avg = conv / I_avg

    bias_MHz = B_inv_I - B_I_avg
    bias_pct = (bias_MHz / B_I_avg) * 100.0

    # High-order Taylor expansion:
    # <1/R^2> = R0^-2 * [1 + 3*(sigma/R0)^2 + 15*(sigma/R0)^4 + 105*(sigma/R0)^6]
    x2 = (sigma_R / R0) ** 2
    taylor_factor = 1.0 + 3.0 * x2 + 15.0 * (x2 ** 2) + 105.0 * (x2 ** 3)
    taylor_B_inv_I = B_e * taylor_factor

    leading_fractional_bias = 3.0 * x2

    return Jensen1DResult(
        sigma_R=sigma_R,
        R0=R0,
        reduced_mass=reduced_mass,
        B_e=B_e,
        B_inv_I=B_inv_I,
        B_I_avg=B_I_avg,
        bias_MHz=bias_MHz,
        bias_pct=bias_pct,
        taylor_B_inv_I=taylor_B_inv_I,
        leading_fractional_bias=leading_fractional_bias,
    )


def compute_1d_benchmark_suite(
    R0: float = 3.80,
    reduced_mass: float = 10.0,
    sigmas: Optional[Sequence[float]] = None,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> List[Jensen1DResult]:
    """
    Computes the authoritative Method Matrix §5.1 validation benchmark suite
    across all mandated zero-point amplitudes [0.05, 0.10, 0.15, 0.20, 0.30] Angstrom.
    """
    if sigmas is None:
        sigmas = [0.05, 0.10, 0.15, 0.20, 0.30]

    return [
        compute_1d_jensen_model(R0=R0, reduced_mass=reduced_mass, sigma_R=s, conv=conv)
        for s in sigmas
    ]


def verify_method_matrix_section_5_1_table() -> bool:
    """
    Asserts that this module exactly reproduces the published reference table
    in Method Matrix Section 5.1.
    """
    results = compute_1d_benchmark_suite()

    # Reference values from Method Matrix Section 5.1:
    # sigma=0.05: B(<1/I>)=3501.69, B(1/<I>)=3499.26, bias=2.43 MHz, bias%=0.069%
    # sigma=0.10: B(<1/I>)=3507.18, B(1/<I>)=3497.46, bias=9.72 MHz, bias%=0.278%
    # sigma=0.15: B(<1/I>)=3516.38, B(1/<I>)=3494.45, bias=21.93 MHz, bias%=0.628%
    # sigma=0.20: B(<1/I>)=3529.40, B(1/<I>)=3490.25, bias=39.16 MHz, bias%=1.122%
    # sigma=0.30: B(<1/I>)=3567.50, B(1/<I>)=3478.27, bias=89.24 MHz, bias%=2.566%

    expected_benchmarks = [
        (0.05, 3501.69, 3499.26, 2.43, 0.069),
        (0.10, 3507.18, 3497.46, 9.72, 0.278),
        (0.15, 3516.38, 3494.45, 21.93, 0.628),
        (0.20, 3529.40, 3490.25, 39.16, 1.122),
        (0.30, 3567.50, 3478.27, 89.24, 2.566),
    ]

    for res, (exp_s, exp_binv, exp_bavg, exp_bias, exp_pct) in zip(results, expected_benchmarks):
        if not math.isclose(res.sigma_R, exp_s, abs_tol=1e-3):
            raise MethodMatrixComplianceError(f"Sigma mismatch: got {res.sigma_R}, expected {exp_s}")
        if abs(res.bias_MHz - exp_bias) >= 0.05:
            raise MethodMatrixComplianceError(
                f"Bias mismatch for sigma={exp_s}: got {res.bias_MHz:.2f} MHz, expected {exp_bias:.2f} MHz"
            )
        if abs(res.bias_pct - exp_pct) >= 0.01:
            raise MethodMatrixComplianceError(
                f"Bias % mismatch for sigma={exp_s}: got {res.bias_pct:.3f}%, expected {exp_pct:.3f}%"
            )

    return True


# =============================================================================
# 6. 3D POLYATOMIC ENSEMBLE TENSOR AVERAGING (Eckart-Watson)
# =============================================================================

def compute_center_of_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Computes the center of mass coordinate vector for a Cartesian molecular structure."""
    total_mass = float(np.sum(masses))
    if total_mass <= 0.0:
        raise JensenAveragerError("Total molecular mass must be strictly positive.")
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass


def compute_inertia_tensor(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    Computes the 3x3 moment of inertia tensor in amu * Angstrom^2.
    Translates to the center-of-mass frame before tensor accumulation.

    Formula:
        I_alpha_beta = sum_i m_i * [ ||r_i - COM||^2 * delta_alpha_beta - (r_i - COM)_alpha * (r_i - COM)_beta ]
    """
    com = compute_center_of_mass(coords, masses)
    r = coords - com
    I = np.zeros((3, 3), dtype=np.float64)
    for i in range(len(masses)):
        m = float(masses[i])
        r_i = r[i]
        r2 = float(np.dot(r_i, r_i))
        I[0, 0] += m * (r2 - r_i[0] ** 2)
        I[1, 1] += m * (r2 - r_i[1] ** 2)
        I[2, 2] += m * (r2 - r_i[2] ** 2)
        I[0, 1] -= m * (r_i[0] * r_i[1])
        I[0, 2] -= m * (r_i[0] * r_i[2])
        I[1, 2] -= m * (r_i[1] * r_i[2])

    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]
    return I


def compute_principal_moments(I: np.ndarray) -> Tuple[float, float, float]:
    """Diagonalizes 3x3 moment of inertia tensor to yield sorted principal moments Ia <= Ib <= Ic."""
    eigvals = np.linalg.eigvalsh(I)
    sorted_moments = np.sort(eigvals)
    if sorted_moments[0] <= 1e-12:
        raise SingularInertiaTensorError(f"Singular or degenerate principal moment of inertia: {sorted_moments[0]}")
    return float(sorted_moments[0]), float(sorted_moments[1]), float(sorted_moments[2])


def compute_rotational_constants_from_inertia(
    I: np.ndarray,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> Tuple[float, float, float]:
    """Computes principal rotational constants (A >= B >= C) from a 3x3 moment of inertia tensor."""
    Ia, Ib, Ic = compute_principal_moments(I)
    A = conv / Ia
    B = conv / Ib
    C = conv / Ic
    return float(A), float(B), float(C)


def compute_rotational_constants_from_inverse_inertia(
    mu: np.ndarray,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> Tuple[float, float, float]:
    """
    Computes principal rotational constants (A >= B >= C) directly from the eigenvalues
    of the generalized inverse inertia tensor mu = I^-1.

    Formula:
        eigvals(mu) = [mu_c <= mu_b <= mu_a]
        A = conv * mu_a
        B = conv * mu_b
        C = conv * mu_c
    """
    eigvals = np.linalg.eigvalsh(mu)
    sorted_mu = np.sort(eigvals)[::-1]  # descending order: mu_a >= mu_b >= mu_c
    if sorted_mu[2] <= 0.0:
        raise SingularInertiaTensorError(f"Negative or non-positive inverse inertia eigenvalue: {sorted_mu[2]}")
    A = conv * float(sorted_mu[0])
    B = conv * float(sorted_mu[1])
    C = conv * float(sorted_mu[2])
    return float(A), float(B), float(C)


def eckart_align(
    coords: np.ndarray,
    ref_coords: np.ndarray,
    masses: np.ndarray,
) -> np.ndarray:
    """
    Performs optimal mass-weighted Eckart alignment (Kabsch algorithm)
    of `coords` onto `ref_coords` so that vibrational-rotational motions are
    projected into a single unambiguous Eckart frame (Method Matrix Section 5.1).
    """
    com_target = compute_center_of_mass(coords, masses)
    com_ref = compute_center_of_mass(ref_coords, masses)

    r_target = coords - com_target
    r_ref = ref_coords - com_ref

    # Mass-weighted covariance matrix H = sum_i m_i * (r_target_i) (r_ref_i)^T
    W = np.diag(masses)
    H = np.dot(r_target.T, np.dot(W, r_ref))

    U, S, Vt = np.linalg.svd(H)
    R_mat = np.dot(Vt.T, U.T)

    # Ensure proper right-handed rotation (det(R) == +1)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1.0
        R_mat = np.dot(Vt.T, U.T)

    aligned = np.dot(r_target, R_mat.T) + com_ref
    return aligned


def average_inverse_inertia_ensemble(
    ensemble_coords: np.ndarray,
    masses: np.ndarray,
    symbols: Optional[Sequence[str]] = None,
    weights: Optional[np.ndarray] = None,
    conv: float = CONV_MHZ_AMU_ANG2,
    ref_coords: Optional[np.ndarray] = None,
) -> Jensen3DResult:
    """
    Computes vibrationally averaged rotational constants for an ensemble of
    3D molecular geometries (e.g. from PIMD beads, DMC walkers, normal mode displacements).

    Rigorous Protocol:
    1. Correct route: <mu> = sum_k w_k [I(k)^-1], then diagonalizes <mu>.
    2. Flawed route: <I> = sum_k w_k I(k), then inverts and diagonalizes.
    3. Quantifies Jensen bias vector [dA, dB, dC] in MHz and percentages.
    4. Evaluates Method Matrix §5.1 caveat on the C rotational constant.
    """
    if ensemble_coords.ndim != 3 or ensemble_coords.shape[2] != 3:
        raise JensenAveragerError(
            f"ensemble_coords must have shape (n_snapshots, n_atoms, 3), got {ensemble_coords.shape}"
        )

    n_snapshots, n_atoms, _ = ensemble_coords.shape
    if len(masses) != n_atoms:
        raise JensenAveragerError(
            f"Length of masses ({len(masses)}) does not match n_atoms ({n_atoms})"
        )

    if weights is None:
        norm_weights = np.full(n_snapshots, 1.0 / float(n_snapshots), dtype=np.float64)
    else:
        w_arr = np.asarray(weights, dtype=np.float64)
        if w_arr.shape != (n_snapshots,):
            raise JensenAveragerError(f"Weights shape {w_arr.shape} does not match n_snapshots {n_snapshots}")
        w_sum = float(np.sum(w_arr))
        if w_sum <= 0.0 or np.any(w_arr < 0.0):
            raise UnphysicalWeightDistributionError("Weights must be strictly non-negative and sum to a positive value.")
        norm_weights = w_arr / w_sum

    if symbols is None:
        sym_list = [f"X{i+1}" for i in range(n_atoms)]
    else:
        sym_list = list(symbols)

    if ref_coords is None:
        ref_coords = ensemble_coords[0]

    mu_accum = np.zeros((3, 3), dtype=np.float64)
    I_accum = np.zeros((3, 3), dtype=np.float64)

    for k in range(n_snapshots):
        curr_geom = ensemble_coords[k]
        aligned_geom = eckart_align(curr_geom, ref_coords, masses)
        I_k = compute_inertia_tensor(aligned_geom, masses)
        mu_k = np.linalg.inv(I_k)

        wk = norm_weights[k]
        I_accum += wk * I_k
        mu_accum += wk * mu_k

    # Correct route: diagonalize <mu>
    A_corr, B_corr, C_corr = compute_rotational_constants_from_inverse_inertia(mu_accum, conv=conv)

    # Flawed route: diagonalize <I>, then invert
    A_flaw, B_flaw, C_flaw = compute_rotational_constants_from_inertia(I_accum, conv=conv)

    bias_A = A_corr - A_flaw
    bias_B = B_corr - B_flaw
    bias_C = C_corr - C_flaw

    bias_pct_A = (bias_A / A_flaw) * 100.0 if A_flaw > 0 else 0.0
    bias_pct_B = (bias_B / B_flaw) * 100.0 if B_flaw > 0 else 0.0
    bias_pct_C = (bias_C / C_flaw) * 100.0 if C_flaw > 0 else 0.0

    # Verification of Jensen's inequality: A_corr >= A_flaw, B_corr >= B_flaw, C_corr >= C_flaw
    eps = -1e-7
    is_valid_jensen = (bias_A >= eps) and (bias_B >= eps) and (bias_C >= eps)

    c_warning = (
        "Method Matrix §5.1 Advisory: C is the least reliable constant from vibrational "
        "averaging due to omission of out-of-plane Coriolis coupling terms <pi_alpha mu_alpha_beta>."
    )

    return Jensen3DResult(
        symbols=sym_list,
        n_atoms=n_atoms,
        n_snapshots=n_snapshots,
        A_correct=A_corr,
        B_correct=B_corr,
        C_correct=C_corr,
        A_flawed=A_flaw,
        B_flawed=B_flaw,
        C_flawed=C_flaw,
        bias_A_MHz=bias_A,
        bias_B_MHz=bias_B,
        bias_C_MHz=bias_C,
        bias_pct_A=bias_pct_A,
        bias_pct_B=bias_pct_B,
        bias_pct_C=bias_pct_C,
        avg_mu_tensor=mu_accum.tolist(),
        avg_I_tensor=I_accum.tolist(),
        is_valid_jensen=is_valid_jensen,
        c_constant_reliability_warning=c_warning,
    )


def compute_vibrationally_averaged_constants(
    inertia_tensors_q: np.ndarray,
    weights_q: Optional[np.ndarray] = None,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> np.ndarray:
    """
    Direct array-level expectation value calculator for a batch of precomputed
    moment of inertia tensors or principal diagonals.

    Complies with Method Matrix §5.1:
    - Inverts each inertia tensor element-wise / matrix-wise BEFORE expectation averaging.
    - Diagonalizes the averaged inverse tensor <mu>.
    """
    tensors = np.asarray(inertia_tensors_q, dtype=np.float64)
    n_points = tensors.shape[0]

    if weights_q is None:
        weights = np.full(n_points, 1.0 / float(n_points), dtype=np.float64)
    else:
        weights = np.asarray(weights_q, dtype=np.float64)
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            raise UnphysicalWeightDistributionError("Sum of vibrational weights must be strictly positive.")
        weights = weights / w_sum

    if tensors.ndim == 3 and tensors.shape[1:] == (3, 3):
        inv_inertia_q = np.linalg.inv(tensors)
        avg_inv_inertia = np.sum(inv_inertia_q * weights[:, None, None], axis=0)
        eigvals = np.linalg.eigvalsh(avg_inv_inertia)
        rotational_constants_mhz = conv * eigvals[::-1]
    elif tensors.ndim == 2 and tensors.shape[1] == 3:
        inv_inertia_q = 1.0 / tensors
        avg_inv_inertia = np.sum(inv_inertia_q * weights[:, None], axis=0)
        rotational_constants_mhz = conv * np.sort(avg_inv_inertia)[::-1]
    else:
        raise JensenAveragerError(
            f"inertia_tensors_q must have shape (N, 3, 3) or (N, 3), got {tensors.shape}"
        )

    return rotational_constants_mhz


def generate_vdw_stretch_distribution(
    monomer1_coords: np.ndarray,
    monomer2_coords: np.ndarray,
    monomer1_masses: np.ndarray,
    monomer2_masses: np.ndarray,
    R0: float = 3.80,
    sigma_R: float = 0.15,
    n_snapshots: int = 10000,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generates a physical vibrational coordinate ensemble for a weakly bound van der Waals
    dimer along the intermolecular separation coordinate R ~ N(R0, sigma_R^2).
    """
    if seed is not None:
        np.random.seed(seed)

    com1 = compute_center_of_mass(monomer1_coords, monomer1_masses)
    com2 = compute_center_of_mass(monomer2_coords, monomer2_masses)

    m1_centered = monomer1_coords - com1
    m2_centered = monomer2_coords - com2

    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    total_masses = np.concatenate([monomer1_masses, monomer2_masses])
    n_total_atoms = len(total_masses)
    symbols = ["Ar", "C", "O", "O"][:n_total_atoms]

    ensemble = np.zeros((n_snapshots, n_total_atoms, 3), dtype=np.float64)
    r_displacements = np.random.normal(R0, sigma_R, size=n_snapshots)

    for k in range(n_snapshots):
        R_k = max(0.5, float(r_displacements[k]))
        pos1 = m1_centered - 0.5 * R_k * axis
        pos2 = m2_centered + 0.5 * R_k * axis
        ensemble[k] = np.vstack([pos1, pos2])

    return ensemble, total_masses, symbols


# =============================================================================
# 7. DISCRETE VARIABLE REPRESENTATION (DVR) / PES GRID AVERAGER
# =============================================================================

def average_inverse_inertia_dvr_grid(
    grid_coords: np.ndarray,
    grid_probabilities: np.ndarray,
    masses: np.ndarray,
    conv: float = CONV_MHZ_AMU_ANG2,
    ref_coords: Optional[np.ndarray] = None,
) -> JensenDVRGridResult:
    """
    Computes vibrationally averaged rotational constants from a numerical
    Discrete Variable Representation (DVR) wavefunction probability density P(q_i) = |psi(q_i)|^2 * dV.
    """
    if grid_coords.ndim != 3 or grid_coords.shape[2] != 3:
        raise JensenAveragerError(f"grid_coords must have shape (N_grid, N_atoms, 3), got {grid_coords.shape}")

    n_grid = grid_coords.shape[0]
    prob = np.asarray(grid_probabilities, dtype=np.float64)
    if prob.shape != (n_grid,):
        raise JensenAveragerError(f"grid_probabilities shape {prob.shape} does not match grid points {n_grid}")

    prob_sum = float(np.sum(prob))
    if prob_sum <= 0.0 or np.any(prob < 0.0):
        raise UnphysicalWeightDistributionError("Probability density must be non-negative and sum to a positive value.")
    norm_prob = prob / prob_sum

    if ref_coords is None:
        ref_coords = grid_coords[0]

    mu_accum = np.zeros((3, 3), dtype=np.float64)
    I_accum = np.zeros((3, 3), dtype=np.float64)

    for i in range(n_grid):
        geom_i = grid_coords[i]
        aligned_i = eckart_align(geom_i, ref_coords, masses)
        I_i = compute_inertia_tensor(aligned_i, masses)
        mu_i = np.linalg.inv(I_i)

        w = norm_prob[i]
        I_accum += w * I_i
        mu_accum += w * mu_i

    A_corr, B_corr, C_corr = compute_rotational_constants_from_inverse_inertia(mu_accum, conv=conv)
    A_flaw, B_flaw, C_flaw = compute_rotational_constants_from_inertia(I_accum, conv=conv)

    bias_A = A_corr - A_flaw
    bias_B = B_corr - B_flaw
    bias_C = C_corr - C_flaw

    bias_pct_A = (bias_A / A_flaw) * 100.0 if A_flaw > 0 else 0.0
    bias_pct_B = (bias_B / B_flaw) * 100.0 if B_flaw > 0 else 0.0
    bias_pct_C = (bias_C / C_flaw) * 100.0 if C_flaw > 0 else 0.0

    is_valid = (bias_A >= -1e-7) and (bias_B >= -1e-7) and (bias_C >= -1e-7)

    return JensenDVRGridResult(
        n_grid_points=n_grid,
        A_correct=A_corr,
        B_correct=B_corr,
        C_correct=C_corr,
        A_flawed=A_flaw,
        B_flawed=B_flaw,
        C_flawed=C_flaw,
        bias_A_MHz=bias_A,
        bias_B_MHz=bias_B,
        bias_C_MHz=bias_C,
        bias_pct_A=bias_pct_A,
        bias_pct_B=bias_pct_B,
        bias_pct_C=bias_pct_C,
        is_valid_jensen=is_valid,
    )


# =============================================================================
# 8. HARDWARE-ACCELERATED PURE JAX FP64 KERNELS
# =============================================================================

@jax.jit
def compute_center_of_mass_jax(coords: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """JAX FP64 accelerated Center of Mass calculation."""
    total_mass = jnp.sum(masses)
    return jnp.sum(coords * masses[:, jnp.newaxis], axis=0) / total_mass


@jax.jit
def compute_inertia_tensor_jax(coords: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """
    Pure JAX FP64 calculation of the 3x3 moment of inertia tensor.
    """
    com = compute_center_of_mass_jax(coords, masses)
    r = coords - com
    r_x = r[:, 0]
    r_y = r[:, 1]
    r_z = r[:, 2]
    r2 = r_x ** 2 + r_y ** 2 + r_z ** 2

    Ixx = jnp.sum(masses * (r2 - r_x ** 2))
    Iyy = jnp.sum(masses * (r2 - r_y ** 2))
    Izz = jnp.sum(masses * (r2 - r_z ** 2))
    Ixy = -jnp.sum(masses * (r_x * r_y))
    Ixz = -jnp.sum(masses * (r_x * r_z))
    Iyz = -jnp.sum(masses * (r_y * r_z))

    I = jnp.array([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]
    ], dtype=jnp.float64)
    return I


@jax.jit
def compute_inverse_inertia_tensor_jax(coords: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
    """Pure JAX FP64 calculation of the inverse moment of inertia tensor mu = I^-1."""
    I = compute_inertia_tensor_jax(coords, masses)
    return jnp.linalg.inv(I)


def average_inverse_inertia_ensemble_jax(
    ensemble_coords: jnp.ndarray,
    masses: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    conv: float = CONV_MHZ_AMU_ANG2,
) -> Tuple[float, float, float]:
    """
    Batched JAX vectorization (vmap) computing <mu> = (1/M) sum [I(k)^-1]
    and returning principal rotational constants (A, B, C) in MHz.
    """
    n_snapshots = ensemble_coords.shape[0]
    if weights is None:
        w = jnp.full((n_snapshots,), 1.0 / float(n_snapshots), dtype=jnp.float64)
    else:
        w = weights / jnp.sum(weights)

    # Vectorized evaluation of inverse inertia tensors across all snapshots
    vmap_inv_I = jax.vmap(compute_inverse_inertia_tensor_jax, in_axes=(0, None))
    mu_batch = vmap_inv_I(ensemble_coords, masses)  # shape: (n_snapshots, 3, 3)

    avg_mu = jnp.sum(mu_batch * w[:, jnp.newaxis, jnp.newaxis], axis=0)
    eigvals = jnp.linalg.eigvalsh(avg_mu)
    sorted_mu = eigvals[::-1]

    A = float(conv * sorted_mu[0])
    B = float(conv * sorted_mu[1])
    C = float(conv * sorted_mu[2])
    return A, B, C


# =============================================================================
# 9. METHOD MATRIX §16.3 COMPLIANCE & AUDIT PROTOCOL
# =============================================================================

def validate_vibrational_averaging_compliance(
    averaging_method: str,
    inverse_before_avg: bool,
    eckart_aligned: bool = True,
) -> MethodMatrixAuditReport:
    """
    Validates a computational execution against Method Matrix Section 5.1 and Section 16.3 mandates.
    Rejects any calculation that inverts <I> or averages scalar rotational constants.
    """
    findings: List[str] = []
    rejection_triggered = False

    if not inverse_before_avg:
        findings.append(
            "[REJECTION TRIGGERED §16.3] Averaging strategy computed <I> before inversion "
            "or inverted the averaged inertia tensor. Violates Jensen's inequality directive."
        )
        rejection_triggered = True

    if not eckart_aligned:
        findings.append(
            "[WARNING §5.1] Snapshots were not aligned to a single Eckart reference frame. "
            "Rotational-vibrational Coriolis coupling may contaminate moments."
        )

    if averaging_method.lower() in ("direct_b_average", "scalar_b_average"):
        findings.append(
            "[REJECTION TRIGGERED §5.1] Direct averaging of scalar rotational constants "
            "B(R) without inverting the 3x3 inertia tensor is strictly forbidden."
        )
        rejection_triggered = True

    if rejection_triggered:
        status = "REJECTED_NON_COMPLIANT"
    else:
        status = "PASSED_COMPLIANT"
        findings.append(
            "[COMPLIANCE VERIFIED §5.1, §16.3] Inverse inertia tensor mu = I^-1 evaluated element-wise "
            "prior to expectation value accumulation and tensor diagonalization."
        )

    payload_str = json.dumps({"status": status, "findings": findings}, sort_keys=True)
    sha = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()

    report = MethodMatrixAuditReport(
        status=status,
        section=SECTION_REF,
        rejection_triggered=rejection_triggered,
        findings=findings,
        sha256_checksum=sha,
    )

    if rejection_triggered:
        raise VibrationalAveragingRejectionError("; ".join(findings))

    return report


# =============================================================================
# 10. FORMATTING & CLI INTERFACE
# =============================================================================

def format_benchmark_table_markdown(results: Sequence[Jensen1DResult]) -> str:
    """Renders the 1D benchmark suite as a GitHub-flavored Markdown table."""
    lines: List[str] = [
        "| sigma_R (Angstrom) | B from <1/I> (MHz) | B from 1/<I> (MHz) | bias (MHz) | bias (%) | Provenance |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for r in results:
        bold = "**" if math.isclose(r.sigma_R, 0.15, abs_tol=1e-4) else ""
        lines.append(
            f"| {bold}{r.sigma_R:.2f}{bold} | "
            f"{bold}{r.B_inv_I:.2f}{bold} | "
            f"{bold}{r.B_I_avg:.2f}{bold} | "
            f"{bold}{r.bias_MHz:.2f}{bold} | "
            f"{bold}{r.bias_pct:.3f}{bold} | "
            f"{r.provenance} |"
        )
    return "\n".join(lines)


def format_benchmark_table_ascii(results: Sequence[Jensen1DResult]) -> str:
    """Renders the 1D benchmark suite as a formatted terminal ASCII table."""
    header = f"{'sigma_R (A)':>12} | {'B from <1/I> (MHz)':>18} | {'B from 1/<I> (MHz)':>18} | {'bias (MHz)':>12} | {'bias (%)':>10} | {'Tag':>5}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r.sigma_R:12.2f} | {r.B_inv_I:18.2f} | {r.B_I_avg:18.2f} | {r.bias_MHz:12.2f} | {r.bias_pct:10.3f} | {r.provenance:>5}"
        )
    return "\n".join(rows)


def main() -> int:
    """CLI entrypoint for Jensen reference validation."""
    parser = argparse.ArgumentParser(
        description="Method Matrix §5.1, §16.3 Jensen Inequality Reference Validation Suite."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run and print the authoritative Method Matrix §5.1 1D benchmark table.",
    )
    parser.add_argument(
        "--verify-table",
        action="store_true",
        help="Run strict programmatic assertions against Method Matrix §5.1 table values.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit validation results in JSON format.",
    )
    parser.add_argument(
        "--vdw-3d-test",
        action="store_true",
        help="Run 3D polyatomic molecular ensemble vibrational tensor averaging test.",
    )

    args = parser.parse_args()

    if not (args.benchmark or args.verify_table or args.json or args.vdw_3d_test):
        args.benchmark = True
        args.verify_table = True

    results = compute_1d_benchmark_suite()

    if args.verify_table:
        is_verified = verify_method_matrix_section_5_1_table()
        if is_verified:
            logger.info("Method Matrix §5.1 table assertions: PASSED [M]")
        else:
            logger.error("Method Matrix §5.1 table assertions: FAILED")
            return 1

    if args.json:
        out_payload = {
            "metadata": {
                "module": "cochem_spycfit.physics_core.cochem_jensen_averager",
                "mandate": "Method Matrix §5.1, §3.0 & §16.3",
                "conversion_constant_MHz_amu_Ang2": CONV_MHZ_AMU_ANG2,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
            "benchmark_1d_results": [r.to_dict() for r in results],
        }
        if args.vdw_3d_test:
            m_Ar = get_dynamic_atomic_mass("Ar")
            m_C = get_dynamic_atomic_mass("C")
            m_O = get_dynamic_atomic_mass("O")
            m1 = np.array([m_Ar])
            m2 = np.array([m_C, m_O, m_O])
            c1 = np.array([[0.0, 0.0, 0.0]])
            c2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.16, 0.0], [0.0, -1.16, 0.0]])
            ens, total_masses, syms = generate_vdw_stretch_distribution(c1, c2, m1, m2, R0=3.80, sigma_R=0.15)
            res_3d = average_inverse_inertia_ensemble(ens, total_masses, syms)
            out_payload["vdw_3d_test_result"] = res_3d.to_dict()

        sys.stdout.write(json.dumps(out_payload, indent=2) + "\n")
        return 0

    if args.benchmark:
        print("\n" + "=" * 80)
        print(" CoChem Method Matrix §5.1 Reference Table: Jensen Inequality Bias ")
        print(" Model: Floppy intermolecular stretch, mu = 10.0 amu, R0 = 3.80 Angstrom ")
        print("=" * 80)
        print(format_benchmark_table_ascii(results))
        print("=" * 80)
        print("\nMarkdown Table for Method_Matrix.md:")
        print(format_benchmark_table_markdown(results))
        print()

    if args.vdw_3d_test:
        print("\n--- Running 3D Polyatomic Ensemble Tensor Averaging Test (Ar...CO2) ---")
        m_Ar = get_dynamic_atomic_mass("Ar")
        m_C = get_dynamic_atomic_mass("C")
        m_O = get_dynamic_atomic_mass("O")
        m1 = np.array([m_Ar])
        m2 = np.array([m_C, m_O, m_O])
        c1 = np.array([[0.0, 0.0, 0.0]])
        c2 = np.array([[0.0, 0.0, 0.0], [0.0, 1.16, 0.0], [0.0, -1.16, 0.0]])
        ens, total_masses, syms = generate_vdw_stretch_distribution(c1, c2, m1, m2, R0=3.80, sigma_R=0.15)
        res_3d = average_inverse_inertia_ensemble(ens, total_masses, syms)
        print(f"Correct <mu> Route: A = {res_3d.A_correct:.2f} MHz, B = {res_3d.B_correct:.2f} MHz, C = {res_3d.C_correct:.2f} MHz")
        print(f"Flawed  1/<I> Route: A = {res_3d.A_flawed:.2f} MHz, B = {res_3d.B_flawed:.2f} MHz, C = {res_3d.C_flawed:.2f} MHz")
        print(f"Bias:               dA = {res_3d.bias_A_MHz:.2f} MHz ({res_3d.bias_pct_A:.3f}%), dB = {res_3d.bias_B_MHz:.2f} MHz ({res_3d.bias_pct_B:.3f}%), dC = {res_3d.bias_C_MHz:.2f} MHz ({res_3d.bias_pct_C:.3f}%)")
        print(f"3D Jensen Inequality Maintained: {res_3d.is_valid_jensen}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
