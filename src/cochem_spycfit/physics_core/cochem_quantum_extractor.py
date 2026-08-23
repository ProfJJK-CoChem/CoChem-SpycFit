"""
CoChem-SpycFit: Theoretical Tensor Gateway & Physics Core.

Authoritative quantum tensor extraction, isotopic mass-weighting, Hermitian locks,
Ray's asymmetry parameter calculations, torsional barrier routing, QCSchema-compliant
PESStore HDF5 dataset management, and JAX FP64 normal mode analysis.
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.linalg import eigh

# Enable JAX 64-bit precision globally for high-accuracy normal mode analysis
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.QuantumExtractor")

# Physical constants and conversion factors
# Conversion from sqrt(Hartree / (Bohr^2 * amu)) to cm^-1:
# c = 2.99792458e10 cm/s, E_h = 4.3597447222071e-18 J, a_0 = 5.29177210903e-11 m, 1 amu = 1.66053906660e-27 kg
# sqrt(E_h / (a_0^2 * m_u)) / (2 * pi * c) = 5140.487143 cm^-1
HARTREE_BOHR2_AMU_TO_CM1: float = 5140.487143

# Conversion from amu * Angstrom^2 to MHz for rotational constants: h / (8 * pi^2 * I)
# h / (8 * pi^2 * 1.66053906660e-47 kg*m^2) in MHz = 505379.008 MHz * amu * Angstrom^2
AMU_ANGSTROM2_TO_MHZ: float = 505379.008

# Conversion from amu * Bohr^2 to MHz:
# 1 Bohr = 0.529177210903 Angstrom -> 1 Bohr^2 = 0.2800285608 Angstrom^2
# 505379.008 / 0.2800285608 = 1804741.0 MHz * amu * Bohr^2
AMU_BOHR2_TO_MHZ: float = 1804741.000

BOHR_TO_ANGSTROM: float = 0.529177210903
DEFAULT_CHUNK_SIZE: int = 65536

# Comprehensive IUPAC Atomic and Isotope Mass Table (amu)
MASS_DICT: Dict[str, float] = {
    "H": 1.00782503223,
    "D": 2.01410177812,
    "2H": 2.01410177812,
    "T": 3.01604928132,
    "3H": 3.01604928132,
    "He": 4.00260325413,
    "3He": 3.0160293201,
    "Li": 7.0160034366,
    "6Li": 6.0151228874,
    "Be": 9.012183065,
    "B": 11.00930536,
    "10B": 10.01293695,
    "C": 12.0000000000,
    "13C": 13.00335483507,
    "14C": 14.0032419884,
    "N": 14.00307400443,
    "15N": 15.00010889888,
    "O": 15.99491461957,
    "17O": 16.9991317565,
    "18O": 17.99915961286,
    "F": 18.99840316273,
    "Ne": 19.9924401762,
    "22Ne": 21.991385114,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "25Mg": 24.985836976,
    "26Mg": 25.982592968,
    "Al": 26.98153853,
    "Si": 27.97692653465,
    "29Si": 28.97649472,
    "30Si": 29.97377017,
    "P": 30.97376199842,
    "S": 31.97207073,
    "33S": 32.97145876,
    "34S": 33.96786690,
    "36S": 35.96708076,
    "Cl": 34.96885271,
    "37Cl": 36.96590260,
    "Ar": 39.9623831237,
    "36Ar": 35.967545105,
    "38Ar": 37.96273211,
    "K": 38.9637064864,
    "40K": 39.963998166,
    "41K": 40.9618252579,
    "Ca": 39.962590863,
    "42Ca": 41.95861783,
    "44Ca": 43.9554806,
    "Sc": 44.95590828,
    "Ti": 47.9479463,
    "46Ti": 45.95262772,
    "V": 50.9439595,
    "Cr": 51.9405075,
    "50Cr": 49.9460442,
    "Mn": 54.93804391,
    "Fe": 55.93493633,
    "54Fe": 53.9396090,
    "57Fe": 56.9353928,
    "Co": 58.93319429,
    "Ni": 57.93534241,
    "60Ni": 59.93078588,
    "Cu": 62.92959772,
    "65Cu": 64.92778970,
    "Zn": 63.92914201,
    "66Zn": 65.92603381,
    "Ga": 68.9255735,
    "71Ga": 70.92470258,
    "Ge": 73.9211774,
    "As": 74.92159457,
    "Se": 79.9165218,
    "78Se": 77.9173095,
    "Br": 78.9183376,
    "81Br": 80.9162897,
    "Kr": 83.911507,
    "86Kr": 85.91061062,
    "Rb": 84.9117897379,
    "87Rb": 86.9091836,
    "Sr": 87.9056125,
    "86Sr": 85.9092606,
    "Y": 88.9058479,
    "Zr": 89.9047037,
    "Nb": 92.9063772,
    "Mo": 97.9054073,
    "92Mo": 91.906811,
    "Tc": 97.907216,
    "Ru": 101.9043485,
    "Rh": 102.905503,
    "Pd": 105.903483,
    "Ag": 106.905093,
    "109Ag": 108.904756,
    "Cd": 113.903361,
    "In": 114.90387877,
    "Sn": 119.9021991,
    "118Sn": 117.901606,
    "Sb": 120.9038157,
    "123Sb": 122.904216,
    "Te": 129.906229,
    "128Te": 127.904463,
    "I": 126.9044719,
    "Xe": 131.904155085,
    "129Xe": 128.904780861,
    "Cs": 132.905451961,
    "Ba": 137.9052472,
    "La": 138.9063533,
    "Ce": 139.9054387,
    "Pr": 140.9076576,
    "Nd": 141.907729,
    "Pm": 144.912749,
    "Sm": 151.9197324,
    "Eu": 152.9212303,
    "Gd": 157.9241039,
    "Tb": 158.9253468,
    "Dy": 163.9291748,
    "Ho": 164.9303221,
    "Er": 165.9302931,
    "Tm": 168.9342133,
    "Yb": 173.9388621,
    "Lu": 174.9407718,
    "Hf": 179.946550,
    "Ta": 180.9479958,
    "W": 183.9509312,
    "Re": 186.9557531,
    "Os": 191.961479,
    "Ir": 192.962924,
    "Pt": 194.9647917,
    "Au": 196.9665687,
    "Hg": 201.970643,
    "Tl": 204.9744278,
    "Pb": 207.9766521,
    "206Pb": 205.9744653,
    "Bi": 208.9803987,
    "Po": 208.982430,
    "At": 209.987148,
    "Rn": 222.017578,
    "Fr": 223.019736,
    "Ra": 226.025410,
    "Ac": 227.027751,
    "Th": 232.0380553,
    "Pa": 231.035884,
    "U": 238.0507882,
    "235U": 235.0439299,
    "Np": 237.048173,
    "Pu": 244.064204,
    "Am": 243.061381,
    "Cm": 247.070354,
    "Bk": 247.070307,
    "Cf": 251.079587,
    "Es": 252.082980,
    "Fm": 257.095105,
    "Md": 258.098431,
    "No": 259.101030,
    "Lr": 262.109630,
    "Rf": 267.121790,
    "Db": 268.125670,
    "Sg": 271.133470,
    "Bh": 270.133360,
    "Hs": 277.151900,
    "Mt": 276.151600,
    "Ds": 281.164510,
    "Rg": 280.165100,
    "Cn": 285.174110,
    "Nh": 284.178080,
    "Fl": 289.187280,
    "Mc": 288.191010,
    "Lv": 293.196000,
    "Ts": 294.210460,
    "Og": 294.213920,
}

MASS_TABLE: Dict[str, float] = MASS_DICT


# =============================================================================
# TYPED EXCEPTION HIERARCHY
# =============================================================================

class NonHermitianHessianError(ValueError):
    """Raised when the Cartesian or mass-weighted Hessian violates Hermitian symmetry."""


class PESStoreError(RuntimeError):
    """Raised for errors in reading, writing, or validating PES HDF5 stores."""


class IsomerNotFoundError(KeyError):
    """Raised when the requested isomer is missing in the PES repository."""


# =============================================================================
# PYDANTIC DATA MODELS
# =============================================================================

class RotationalConstants(BaseModel):
    """Authoritative rotational constants in MHz."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    A: float = Field(..., description="Principal rotational constant A (MHz)")
    B: float = Field(..., description="Principal rotational constant B (MHz)")
    C: float = Field(..., description="Principal rotational constant C (MHz)")
    unit: str = Field(default="MHz", description="Unit of rotational constants")


class DipoleMoment(BaseModel):
    """Molecular permanent dipole moment components in Debye."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    X: float = Field(..., description="Dipole X component (Debye)")
    Y: float = Field(..., description="Dipole Y component (Debye)")
    Z: float = Field(..., description="Dipole Z component (Debye)")
    total: float = Field(..., description="Total dipole magnitude (Debye)")
    mu_a: Optional[float] = Field(default=None, description="Dipole projection along principal a-axis")
    mu_b: Optional[float] = Field(default=None, description="Dipole projection along principal b-axis")
    mu_c: Optional[float] = Field(default=None, description="Dipole projection along principal c-axis")


class HamiltonianRouting(BaseModel):
    """Hamiltonian reduction and representation routing parameters."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    routing: str = Field(..., description="Target routing Hamiltonian model (e.g. IAM_WOODS, watson_a, watson_s)")
    reduction: Optional[str] = Field(default=None, description="Watson reduction type: 'A', 'S', or None for IAM")
    representation: Optional[str] = Field(default=None, description="Watson axis representation: 'Ir', 'IIIr', or None")
    kappa: float = Field(..., description="Ray's asymmetry parameter")
    v3_barrier_cm1: Optional[float] = Field(default=None, description="1D torsional barrier in cm^-1")
    description: str = Field(..., description="Human-readable explanation of the Hamiltonian routing decision")


class QuantumExtraction(BaseModel):
    """Extraction container for rotational constants and dipole moments."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    rotational_constants: Optional[RotationalConstants] = None
    dipole_moment: Optional[DipoleMoment] = None


class QuantumTensors(BaseModel):
    """Typed container for complete molecular tensors, coordinates, and routing."""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="allow")

    isomer_id: str
    symbols: List[str]
    coords: np.ndarray
    cartesian_hessian: np.ndarray
    mass_weighted_hessian: np.ndarray
    masses: np.ndarray
    dipole_derivatives: Optional[np.ndarray] = None
    rotational_constants: Optional[RotationalConstants] = None
    dipole_moment: Optional[DipoleMoment] = None
    v3_barrier_cm1: Optional[float] = None
    hamiltonian_routing: Optional[HamiltonianRouting] = None
    lam_trigger: bool = False
    dvr_eigenvalues: Optional[np.ndarray] = None
    tunneling_splittings: Optional[np.ndarray] = None
    kraitchman_coords: Optional[np.ndarray] = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_atomic_mass(symbol: str) -> float:
    """
    Look up standard IUPAC isotopic mass for an atomic symbol.
    Strictly returns a 64-bit float.
    """
    cleaned = symbol.strip()
    if cleaned in MASS_DICT:
        return float(MASS_DICT[cleaned])

    if cleaned.upper() in MASS_DICT:
        return float(MASS_DICT[cleaned.upper()])

    direct_cap = cleaned.capitalize()
    if direct_cap in MASS_DICT:
        return float(MASS_DICT[direct_cap])

    m = re.match(r"^(\d+)([a-zA-Z]+)$", cleaned)
    if m:
        iso_key = f"{m.group(1)}{m.group(2).capitalize()}"
        if iso_key in MASS_DICT:
            return float(MASS_DICT[iso_key])

    alpha_only = re.sub(r"^\d+", "", cleaned).capitalize()
    if alpha_only in MASS_DICT:
        return float(MASS_DICT[alpha_only])

    raise ValueError(f"Unknown element/isotope symbol: '{symbol}' for mass retrieval.")


def compute_rotational_constants_from_coords(
    coords: np.ndarray,
    masses: Union[np.ndarray, List[float], List[str]],
    unit: str = "angstrom",
) -> RotationalConstants:
    """
    Calculates principal rotational constants (A, B, C in MHz) from 3D coordinates and masses.
    Applies center-of-mass translation, diagonalizes the moment of inertia tensor, and
    orders constants such that A >= B >= C.
    """
    coords_f64 = np.asarray(coords, dtype=np.float64)
    if coords_f64.ndim != 2 or coords_f64.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (N, 3), got {coords_f64.shape}")

    num_atoms = coords_f64.shape[0]

    # Resolve masses
    if isinstance(masses, list) and len(masses) > 0 and isinstance(masses[0], str):
        mass_arr = np.array([get_atomic_mass(s) for s in masses], dtype=np.float64)
    else:
        mass_arr = np.array(masses, dtype=np.float64)

    if mass_arr.shape != (num_atoms,):
        raise ValueError(f"Mass array length ({len(mass_arr)}) does not match atom count ({num_atoms})")

    # Shift to center of mass
    total_mass = float(np.sum(mass_arr))
    if total_mass <= 0.0:
        raise ValueError(f"Total molecular mass must be positive, got {total_mass}")

    com = np.sum(coords_f64 * mass_arr[:, np.newaxis], axis=0) / total_mass
    r = coords_f64 - com

    # Inertia tensor computation
    inertia_tensor = np.zeros((3, 3), dtype=np.float64)
    for i in range(num_atoms):
        m_i = mass_arr[i]
        x, y, z = r[i, 0], r[i, 1], r[i, 2]
        inertia_tensor[0, 0] += m_i * (y * y + z * z)
        inertia_tensor[1, 1] += m_i * (x * x + z * z)
        inertia_tensor[2, 2] += m_i * (x * x + y * y)
        inertia_tensor[0, 1] -= m_i * x * y
        inertia_tensor[1, 0] -= m_i * x * y
        inertia_tensor[0, 2] -= m_i * x * z
        inertia_tensor[2, 0] -= m_i * x * z
        inertia_tensor[1, 2] -= m_i * y * z
        inertia_tensor[2, 1] -= m_i * y * z

    # Diagonalize inertia tensor
    evals, _ = eigh(inertia_tensor)
    principal_moments = np.sort(evals)

    # Unit conversion factor to MHz
    conversion_factor = AMU_ANGSTROM2_TO_MHZ if unit.lower() == "angstrom" else AMU_BOHR2_TO_MHZ

    # Handle linear molecules and planar zero moments
    rot_vals: List[float] = []
    for moment in principal_moments:
        if moment > 1e-6:
            rot_vals.append(conversion_factor / moment)
        else:
            rot_vals.append(0.0)

    # Sort descending A >= B >= C
    # Principal moments I_a <= I_b <= I_c correspond to A >= B >= C
    rot_sorted = sorted(rot_vals, reverse=True)
    a_val = float(rot_sorted[0]) if len(rot_sorted) > 0 else 0.0
    b_val = float(rot_sorted[1]) if len(rot_sorted) > 1 else 0.0
    c_val = float(rot_sorted[2]) if len(rot_sorted) > 2 else 0.0

    return RotationalConstants(A=a_val, B=b_val, C=c_val, unit="MHz")


# =============================================================================
# MASS PERTURBATION & HERMITIAN LOCK
# =============================================================================

def apply_mass_perturbation_and_lock(
    hessian_cartesian: np.ndarray,
    mass_matrix: Optional[Union[np.ndarray, List[float], List[str]]] = None,
    symbols: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Applies isotopic mass perturbations to Cartesian Hessians to yield mass-weighted coordinates.
    Executes a Hermitian lock (np.allclose(H, H.conj().T)).
    Applies strict symmetrization 0.5 * (H + H.T) if floating-point numerical noise is detected.
    Raises NonHermitianHessianError if the matrix is severely non-symmetric (> 1e-2 deviation).
    """
    hess_f64 = np.asarray(hessian_cartesian, dtype=np.float64)

    if hess_f64.ndim != 2 or hess_f64.shape[0] != hess_f64.shape[1]:
        raise ValueError(f"Cartesian Hessian must be a square 2D matrix, got shape {hess_f64.shape}")

    dim_3n = hess_f64.shape[0]
    if dim_3n % 3 != 0:
        raise ValueError(f"Hessian dimension {dim_3n} must be a multiple of 3 (3N degrees of freedom)")

    num_atoms = dim_3n // 3

    # Derive atomic masses
    masses_3n: np.ndarray
    if mass_matrix is not None:
        if isinstance(mass_matrix, list) and len(mass_matrix) > 0 and isinstance(mass_matrix[0], str):
            atomic_masses = np.array([get_atomic_mass(s) for s in mass_matrix], dtype=np.float64)
            masses_3n = np.repeat(atomic_masses, 3)
        else:
            raw_masses = np.array(mass_matrix, dtype=np.float64)
            if raw_masses.ndim == 1:
                if raw_masses.shape[0] == num_atoms:
                    masses_3n = np.repeat(raw_masses, 3)
                elif raw_masses.shape[0] == dim_3n:
                    masses_3n = raw_masses
                else:
                    raise ValueError(f"Mass vector length {raw_masses.shape[0]} does not match N={num_atoms} or 3N={dim_3n}")
            elif raw_masses.ndim == 2 and raw_masses.shape == (dim_3n, dim_3n):
                masses_3n = np.diag(raw_masses)
            else:
                raise ValueError(f"Invalid mass matrix shape {raw_masses.shape}")
    elif symbols is not None:
        if len(symbols) != num_atoms:
            raise ValueError(f"Symbol list length {len(symbols)} does not match atom count {num_atoms}")
        atomic_masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)
        masses_3n = np.repeat(atomic_masses, 3)
    else:
        raise ValueError("Either 'mass_matrix' or 'symbols' must be provided for mass weighting.")

    if bool((masses_3n <= 0.0).any()):
        raise ValueError("All atomic masses must be strictly positive.")

    # Compute mass weighting matrix: W_ij = 1 / sqrt(m_i * m_j)
    inv_sqrt_masses = 1.0 / np.sqrt(masses_3n)
    weight_outer = np.outer(inv_sqrt_masses, inv_sqrt_masses)

    mw_hessian = hess_f64 * weight_outer

    # Execute Hermitian Lock and Asymmetry Verification
    asym_diff = mw_hessian - mw_hessian.conj().T
    max_asymmetry = float(np.max(np.abs(asym_diff)))

    # Strict threshold for severe non-Hermiticity
    if max_asymmetry > 1e-2:
        raise NonHermitianHessianError(
            f"Mass-weighted Hessian violates Hermitian symmetry severely: "
            f"max asymmetry deviation = {max_asymmetry:.6e} exceeds tolerance 1e-2"
        )

    # Symmetrize numerical noise
    mw_hessian_locked = 0.5 * (mw_hessian + mw_hessian.conj().T)
    return np.asarray(mw_hessian_locked, dtype=np.float64)


# =============================================================================
# RAY'S ASYMMETRY PARAMETER & HAMILTONIAN ROUTING
# =============================================================================

def compute_rays_asymmetry(A: float, B: float, C: float) -> float:
    """
    Computes Ray's asymmetry parameter: kappa = (2B - A - C) / (A - C).
    Enforces descending order A >= B >= C.
    Handles symmetric prolate (kappa = -1), oblate (kappa = +1), and spherical (kappa = 0) limits cleanly.
    """
    a_f = float(A)
    b_f = float(B)
    c_f = float(C)

    # Sort constants in descending order to adhere to standard spectroscopy convention A >= B >= C
    ordered = sorted([a_f, b_f, c_f], reverse=True)
    a_ord, b_ord, c_ord = ordered[0], ordered[1], ordered[2]

    diff_ac = a_ord - c_ord
    if abs(diff_ac) < 1e-12:
        # Spherical top limit (A == B == C)
        return 0.0

    if abs(b_ord - c_ord) < 1e-12:
        # Prolate symmetric top limit (A > B == C)
        return -1.0

    if abs(a_ord - b_ord) < 1e-12:
        # Oblate symmetric top limit (A == B > C)
        return 1.0

    kappa = (2.0 * b_ord - a_ord - c_ord) / diff_ac
    # Clamp to valid physical interval [-1.0, 1.0]
    return float(max(-1.0, min(1.0, kappa)))


def determine_hamiltonian_routing(
    kappa: float,
    v3_barrier_cm1: Optional[float] = None,
) -> HamiltonianRouting:
    """
    Determines the target Hamiltonian representation and reduction based on
    Ray's asymmetry parameter (kappa) and torsional barrier (V3 in cm^-1).

    Routing Rules:
    - If V3 is present and V3 < 400.0 cm^-1 -> "IAM_WOODS" (Internal Axis Method / Woods)
    - Else if |kappa| < 0.5 -> Watson's A-reduction ("watson_a", representation "Ir")
    - Else if kappa <= -0.5 -> Watson's S-reduction ("watson_s", representation "Ir")
    - Else if kappa >= 0.5 -> Watson's S-reduction ("watson_s", representation "IIIr")
    """
    k_val = float(kappa)
    v3_val = float(v3_barrier_cm1) if v3_barrier_cm1 is not None else None

    if v3_val is not None and v3_val < 400.0:
        return HamiltonianRouting(
            routing="IAM_WOODS",
            reduction=None,
            representation="Ir",
            kappa=k_val,
            v3_barrier_cm1=v3_val,
            description="Internal Axis Method (Woods / IAM) for low 1D-torsional barrier (V3 < 400 cm^-1)",
        )

    if abs(k_val) < 0.5:
        return HamiltonianRouting(
            routing="watson_a",
            reduction="A",
            representation="Ir",
            kappa=k_val,
            v3_barrier_cm1=v3_val,
            description="Watson A-reduction in Ir representation for asymmetric rotor (|kappa| < 0.5)",
        )

    if k_val <= -0.5:
        return HamiltonianRouting(
            routing="watson_s",
            reduction="S",
            representation="Ir",
            kappa=k_val,
            v3_barrier_cm1=v3_val,
            description="Watson S-reduction in Ir representation for near-prolate rotor (kappa <= -0.5)",
        )

    return HamiltonianRouting(
        routing="watson_s",
        reduction="S",
        representation="IIIr",
        kappa=k_val,
        v3_barrier_cm1=v3_val,
        description="Watson S-reduction in IIIr representation for near-oblate rotor (kappa >= 0.5)",
    )


# =============================================================================
# JAX FP64 NORMAL MODE ANALYSIS
# =============================================================================

def compute_normal_modes_jax(
    mw_hessian: Union[np.ndarray, jnp.ndarray],
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes vibrational harmonic frequencies and normal modes using JAX with FP64 precision.
    Returns frequencies in cm^-1, eigenvalues, eigenvectors, and reduced masses.
    """
    mw_h_jax = jnp.asarray(mw_hessian, dtype=jnp.float64)

    if mw_h_jax.ndim != 2 or mw_h_jax.shape[0] != mw_h_jax.shape[1]:
        raise ValueError(f"Mass-weighted Hessian must be square, got {mw_h_jax.shape}")

    # Diagonalize mass-weighted Hessian
    eigvals, eigvecs = jnp.linalg.eigh(mw_h_jax)

    # Compute harmonic frequencies (cm^-1)
    # Imaginary frequencies (negative eigenvalues) preserve sign
    frequencies = jnp.sign(eigvals) * jnp.sqrt(jnp.abs(eigvals)) * HARTREE_BOHR2_AMU_TO_CM1

    dim_3n = mw_h_jax.shape[0]
    num_atoms = dim_3n // 3

    reduced_masses: List[float] = []
    cartesian_displacements: Optional[np.ndarray] = None

    if symbols is not None and len(symbols) == num_atoms:
        atomic_masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)
        masses_3n = np.repeat(atomic_masses, 3)
        inv_sqrt_m = 1.0 / np.sqrt(masses_3n)

        # Un-mass-weight eigenvectors to yield Cartesian displacements
        eigvecs_np = np.array(eigvecs, dtype=np.float64)
        cartesian_modes = eigvecs_np * inv_sqrt_m[:, np.newaxis]
        cartesian_displacements = cartesian_modes

        for k in range(dim_3n):
            norm_k = float(np.sum(cartesian_modes[:, k] ** 2))
            red_m = 1.0 / norm_k if norm_k > 1e-14 else 1.0
            reduced_masses.append(red_m)
    else:
        for k in range(dim_3n):
            reduced_masses.append(1.0)

    # Compute zero point energy (ZPE) in cm^-1 (summing only positive real frequencies)
    freqs_np = np.array(frequencies, dtype=np.float64)
    zpe = float(0.5 * np.sum(freqs_np[freqs_np > 0.0]))

    return {
        "frequencies_cm1": freqs_np,
        "eigenvalues": np.array(eigvals, dtype=np.float64),
        "eigenvectors": np.array(eigvecs, dtype=np.float64),
        "reduced_masses_amu": np.array(reduced_masses, dtype=np.float64),
        "cartesian_displacements": cartesian_displacements,
        "zero_point_energy_cm1": zpe,
    }


# =============================================================================
# PESSTORE: QCSCHEMA-CONFORMANT HDF5 CHUNKED REPOSITORY
# =============================================================================

class PESStore:
    """
    QCSchema-conformant HDF5 repository for chunked PES and Hessian tensor persistence.
    Operates with full mathematical fidelity and casts all numeric float data strictly to np.float64 at ingestion.
    """

    def __init__(self, file_path: Union[str, Path], mode: str = "r") -> None:
        self.file_path = Path(file_path).resolve()
        self.mode = mode
        self._h5file: Optional[h5py.File] = None
        self._is_external_file = False

    def open(self) -> "PESStore":
        """Opens the HDF5 file safely."""
        if self._h5file is None:
            if not self.file_path.exists() and "w" not in self.mode and "a" not in self.mode:
                raise FileNotFoundError(f"PESStore HDF5 file not found: {self.file_path}")
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self._h5file = h5py.File(str(self.file_path), mode=self.mode)
        return self

    def close(self) -> None:
        """Closes the underlying HDF5 file handle."""
        if self._h5file is not None and not self._is_external_file:
            self._h5file.close()
            self._h5file = None

    def __enter__(self) -> "PESStore":
        return self.open()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @property
    def file(self) -> h5py.File:
        """Returns active h5py.File instance, opening if necessary."""
        if self._h5file is None:
            self.open()
        if self._h5file is None:
            raise PESStoreError("Failed to initialize HDF5 file handle.")
        return self._h5file

    def list_isomers(self) -> List[str]:
        """Lists all isomer identifiers present in the PES store."""
        return list(self.file.keys())

    def _get_isomer_group(self, isomer_id: str) -> h5py.Group:
        """Retrieves or validates isomer group."""
        if isomer_id not in self.file:
            raise IsomerNotFoundError(f"Isomer '{isomer_id}' not found in PES store {self.file_path}")
        group = self.file[isomer_id]
        if not isinstance(group, h5py.Group):
            raise PESStoreError(f"Dataset path '{isomer_id}' is not an HDF5 group in {self.file_path}")
        return group

    def read_isomer(self, isomer_id: str) -> Dict[str, Any]:
        """
        Reads all quantum tensors, geometry, and metadata for the given isomer.
        Casts all floating-point tensors to np.float64.
        """
        group = self._get_isomer_group(isomer_id)
        data: Dict[str, Any] = {"isomer_id": isomer_id}

        # 1. Extract Symbols
        if "symbols" in group:
            raw_syms = group["symbols"][:]
            syms: List[str] = []
            for s in raw_syms:
                if isinstance(s, bytes):
                    syms.append(s.decode("utf-8"))
                else:
                    syms.append(str(s))
            data["symbols"] = syms
        elif "molecule" in group and "symbols" in group["molecule"]:
            raw_syms = group["molecule"]["symbols"][:]
            data["symbols"] = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in raw_syms]
        else:
            raise PESStoreError(f"No symbols dataset found for isomer '{isomer_id}'")

        num_atoms = len(data["symbols"])

        # 2. Extract Geometry / Coordinates
        coords_raw = None
        for key in ("coords", "geometry", "geometry_xyz", "coordinates"):
            if key in group:
                coords_raw = group[key][:]
                break
            if "molecule" in group and key in group["molecule"]:
                coords_raw = group["molecule"][key][:]
                break

        if coords_raw is not None:
            coords_f64 = np.asarray(coords_raw, dtype=np.float64)
            if coords_f64.ndim == 1 and coords_f64.size == num_atoms * 3:
                coords_f64 = coords_f64.reshape((num_atoms, 3))
            data["coords"] = coords_f64
        else:
            data["coords"] = np.zeros((num_atoms, 3), dtype=np.float64)

        # 3. Extract Cartesian Hessian
        hess_raw = None
        for key in ("hessian", "hessian_matrix", "return_result"):
            if key in group:
                hess_raw = group[key][:]
                break
            if "wavefunction" in group and key in group["wavefunction"]:
                hess_raw = group["wavefunction"][key][:]
                break

        if hess_raw is None:
            raise PESStoreError(f"No Hessian matrix dataset found for isomer '{isomer_id}'")

        hess_f64 = np.asarray(hess_raw, dtype=np.float64)
        dim_3n = num_atoms * 3
        if hess_f64.ndim == 1 and hess_f64.size == dim_3n * dim_3n:
            hess_f64 = hess_f64.reshape((dim_3n, dim_3n))
        elif hess_f64.shape != (dim_3n, dim_3n):
            raise PESStoreError(f"Invalid Hessian shape {hess_f64.shape}, expected ({dim_3n}, {dim_3n})")
        data["hessian"] = hess_f64

        # 4. Extract Dipole Derivatives
        dip_deriv_raw = None
        for key in ("dipole_derivatives", "dipole_derivs"):
            if key in group:
                dip_deriv_raw = group[key][:]
                break
            if "wavefunction" in group and key in group["wavefunction"]:
                dip_deriv_raw = group["wavefunction"][key][:]
                break

        if dip_deriv_raw is not None:
            data["dipole_derivatives"] = np.asarray(dip_deriv_raw, dtype=np.float64)
        else:
            data["dipole_derivatives"] = None

        # 5. Extract Rotational Constants if stored
        if "rotational_constants" in group:
            rc_raw = group["rotational_constants"][:]
            data["rotational_constants"] = RotationalConstants(
                A=float(rc_raw[0]), B=float(rc_raw[1]), C=float(rc_raw[2]), unit="MHz"
            )
        elif "rotational_constants" in group.attrs:
            rc_attr = group.attrs["rotational_constants"]
            data["rotational_constants"] = RotationalConstants(
                A=float(rc_attr[0]), B=float(rc_attr[1]), C=float(rc_attr[2]), unit="MHz"
            )
        else:
            data["rotational_constants"] = None

        # 6. Extract Dipole Moment if stored
        if "dipole_moment" in group:
            dm_raw = group["dipole_moment"][:]
            dx, dy, dz = float(dm_raw[0]), float(dm_raw[1]), float(dm_raw[2])
            tot = float(math.sqrt(dx * dx + dy * dy + dz * dz))
            data["dipole_moment"] = DipoleMoment(X=dx, Y=dy, Z=dz, total=tot)
        elif "properties" in group and "scf_dipole_moment" in group["properties"]:
            dm_raw = group["properties"]["scf_dipole_moment"][:]
            dx, dy, dz = float(dm_raw[0]), float(dm_raw[1]), float(dm_raw[2])
            tot = float(math.sqrt(dx * dx + dy * dy + dz * dz))
            data["dipole_moment"] = DipoleMoment(X=dx, Y=dy, Z=dz, total=tot)
        else:
            data["dipole_moment"] = None

        # 7. Extract V3 Torsional Barrier
        if "v3_barrier_cm1" in group.attrs:
            data["v3_barrier_cm1"] = float(group.attrs["v3_barrier_cm1"])
        elif "v3_barrier" in group.attrs:
            data["v3_barrier_cm1"] = float(group.attrs["v3_barrier"])
        elif "v3_barrier_cm1" in group:
            data["v3_barrier_cm1"] = float(group["v3_barrier_cm1"][()])
        else:
            data["v3_barrier_cm1"] = None

        # 8. Extract LAM Data and Attributes
        data["lam_trigger"] = bool(group.attrs.get("LAM_TRIGGER_REQUIRED", False))
        if "dvr_eigenvalues" in group:
            data["dvr_eigenvalues"] = np.asarray(group["dvr_eigenvalues"][:], dtype=np.float64)
        else:
            data["dvr_eigenvalues"] = None

        if "tunneling_splittings" in group:
            data["tunneling_splittings"] = np.asarray(group["tunneling_splittings"][:], dtype=np.float64)
        else:
            data["tunneling_splittings"] = None

        if "kraitchman_coords" in group:
            data["kraitchman_coords"] = np.asarray(group["kraitchman_coords"][:], dtype=np.float64)
        else:
            data["kraitchman_coords"] = None

        return data

    def write_isomer(
        self,
        isomer_id: str,
        data: Dict[str, Any],
        chunks: bool = True,
        compression: str = "gzip",
    ) -> None:
        """
        Writes complete QCSchema and quantum tensor dataset for an isomer using chunked storage.
        All floating point arrays are strictly stored as np.float64.
        """
        if isomer_id in self.file:
            del self.file[isomer_id]

        group = self.file.create_group(isomer_id)

        # 1. Write Symbols
        if "symbols" in data:
            sym_list = [s.encode("utf-8") if isinstance(s, str) else s for s in data["symbols"]]
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset("symbols", data=sym_list, dtype=dt)

        # 2. Write Geometry
        if "coords" in data and data["coords"] is not None:
            coords_arr = np.asarray(data["coords"], dtype=np.float64)
            c_shape = (min(coords_arr.shape[0], 64), 3) if chunks else None
            group.create_dataset(
                "coords",
                data=coords_arr,
                dtype="<f8",
                chunks=c_shape,
                compression=compression,
            )

        # 3. Write Cartesian Hessian
        if "hessian" in data and data["hessian"] is not None:
            hess_arr = np.asarray(data["hessian"], dtype=np.float64)
            h_chunk = (min(hess_arr.shape[0], 64), min(hess_arr.shape[1], 64)) if chunks else None
            group.create_dataset(
                "hessian",
                data=hess_arr,
                dtype="<f8",
                chunks=h_chunk,
                compression=compression,
            )

        # 4. Write Dipole Derivatives
        if "dipole_derivatives" in data and data["dipole_derivatives"] is not None:
            dip_arr = np.asarray(data["dipole_derivatives"], dtype=np.float64)
            d_chunk = (min(dip_arr.shape[0], 64), dip_arr.shape[1]) if chunks else None
            group.create_dataset(
                "dipole_derivatives",
                data=dip_arr,
                dtype="<f8",
                chunks=d_chunk,
                compression=compression,
            )

        # 5. Write Rotational Constants
        if "rotational_constants" in data and data["rotational_constants"] is not None:
            rc = data["rotational_constants"]
            if isinstance(rc, RotationalConstants):
                group.attrs["rotational_constants"] = [rc.A, rc.B, rc.C]
            elif isinstance(rc, (list, tuple, np.ndarray)) and len(rc) >= 3:
                group.attrs["rotational_constants"] = [float(rc[0]), float(rc[1]), float(rc[2])]

        # 6. Write Dipole Moment
        if "dipole_moment" in data and data["dipole_moment"] is not None:
            dm = data["dipole_moment"]
            if isinstance(dm, DipoleMoment):
                group.create_dataset("dipole_moment", data=np.array([dm.X, dm.Y, dm.Z], dtype=np.float64))
            elif isinstance(dm, (list, tuple, np.ndarray)) and len(dm) >= 3:
                group.create_dataset("dipole_moment", data=np.array([dm[0], dm[1], dm[2]], dtype=np.float64))

        # 7. Write V3 Barrier
        if "v3_barrier_cm1" in data and data["v3_barrier_cm1"] is not None:
            group.attrs["v3_barrier_cm1"] = float(data["v3_barrier_cm1"])

        # 8. Write LAM Attributes & Datasets
        if "lam_trigger" in data:
            group.attrs["LAM_TRIGGER_REQUIRED"] = bool(data["lam_trigger"])

        if "dvr_eigenvalues" in data and data["dvr_eigenvalues"] is not None:
            dvr_arr = np.asarray(data["dvr_eigenvalues"], dtype=np.float64)
            group.create_dataset("dvr_eigenvalues", data=dvr_arr, dtype="<f8", compression=compression)

        if "tunneling_splittings" in data and data["tunneling_splittings"] is not None:
            ts_arr = np.asarray(data["tunneling_splittings"], dtype=np.float64)
            group.create_dataset("tunneling_splittings", data=ts_arr, dtype="<f8", compression=compression)

        if "kraitchman_coords" in data and data["kraitchman_coords"] is not None:
            kc_arr = np.asarray(data["kraitchman_coords"], dtype=np.float64)
            group.create_dataset("kraitchman_coords", data=kc_arr, dtype="<f8", compression=compression)

        self.file.flush()

    def get_hessian(self, isomer_id: str) -> np.ndarray:
        """Convenience method returning raw Cartesian Hessian."""
        return self.read_isomer(isomer_id)["hessian"]

    def get_dipole_derivatives(self, isomer_id: str) -> Optional[np.ndarray]:
        """Convenience method returning dipole derivatives if present."""
        return self.read_isomer(isomer_id)["dipole_derivatives"]

    def get_geometry(self, isomer_id: str) -> np.ndarray:
        """Convenience method returning Cartesian geometry."""
        return self.read_isomer(isomer_id)["coords"]

    def get_symbols(self, isomer_id: str) -> List[str]:
        """Convenience method returning atomic symbols."""
        return self.read_isomer(isomer_id)["symbols"]

    def get_v3_barrier(self, isomer_id: str) -> Optional[float]:
        """Convenience method returning 1D torsional barrier in cm^-1."""
        return self.read_isomer(isomer_id)["v3_barrier_cm1"]


# =============================================================================
# HIGH-LEVEL TENSOR EXTRACTION GATEWAY
# =============================================================================

def extract_tensors(
    pes_store: Union[PESStore, str, Path, h5py.File, h5py.Group],
    isomer_id: Optional[str] = None,
) -> QuantumTensors:
    """
    Theoretical Tensor Gateway entry point.
    Reads tensors from PESStore instance, file path, or HDF5 handle.
    Applies isotopic mass-weighting and Hermitian locks, derives Ray's asymmetry,
    determines Hamiltonian routing, and returns a fully populated QuantumTensors instance.
    """
    store_obj: Optional[PESStore] = None
    target_isomer = isomer_id

    if isinstance(pes_store, PESStore):
        store_obj = pes_store
        if target_isomer is None:
            isomers = store_obj.list_isomers()
            if not isomers:
                raise PESStoreError(f"No isomers found in store {store_obj.file_path}")
            target_isomer = isomers[0]
        raw_data = store_obj.read_isomer(target_isomer)
    elif isinstance(pes_store, (str, Path)):
        p = Path(pes_store).resolve()
        with PESStore(p, mode="r") as s:
            if target_isomer is None:
                isomers = s.list_isomers()
                if not isomers:
                    raise PESStoreError(f"No isomers found in store {p}")
                target_isomer = isomers[0]
            raw_data = s.read_isomer(target_isomer)
    elif isinstance(pes_store, (h5py.File, h5py.Group)):
        # Handle active h5py objects directly
        if isinstance(pes_store, h5py.File) and target_isomer is None:
            isomers = list(pes_store.keys())
            if not isomers:
                raise PESStoreError("No isomer groups found in HDF5 file.")
            target_isomer = isomers[0]
        elif isinstance(pes_store, h5py.Group):
            target_isomer = pes_store.name.split("/")[-1] if target_isomer is None else target_isomer

        if target_isomer is None or not target_isomer:
            raise PESStoreError("Could not determine target isomer_id from HDF5 object.")

        # Create temporary PESStore referencing existing group/file
        temp_store = PESStore(Path("ephemeral.h5"), mode="r")
        temp_store._h5file = pes_store if isinstance(pes_store, h5py.File) else pes_store.file
        temp_store._is_external_file = True
        raw_data = temp_store.read_isomer(target_isomer)
    else:
        raise TypeError(f"Unsupported pes_store type: {type(pes_store)}")

    if target_isomer is None or not target_isomer:
        raise PESStoreError("Target isomer_id is undefined.")

    symbols = raw_data["symbols"]
    coords = raw_data["coords"]
    cart_hessian = raw_data["hessian"]
    dipole_derivs = raw_data["dipole_derivatives"]
    v3_barrier = raw_data["v3_barrier_cm1"]

    # Compute atomic masses
    atomic_masses = np.array([get_atomic_mass(s) for s in symbols], dtype=np.float64)

    # Mass perturbation & Hermitian lock
    mw_hessian = apply_mass_perturbation_and_lock(
        hessian_cartesian=cart_hessian,
        symbols=symbols,
    )

    # Derive rotational constants
    rot_constants = raw_data.get("rotational_constants")
    if rot_constants is None:
        if coords is not None and coords.shape[0] == len(symbols):
            rot_constants = compute_rotational_constants_from_coords(coords, atomic_masses)

    # Compute Ray's Asymmetry and Hamiltonian Routing
    hamiltonian_routing: Optional[HamiltonianRouting] = None
    if rot_constants is not None and rot_constants.A > 0.0:
        kappa = compute_rays_asymmetry(rot_constants.A, rot_constants.B, rot_constants.C)
        hamiltonian_routing = determine_hamiltonian_routing(kappa, v3_barrier_cm1=v3_barrier)

    return QuantumTensors(
        isomer_id=target_isomer,
        symbols=symbols,
        coords=coords,
        cartesian_hessian=cart_hessian,
        mass_weighted_hessian=mw_hessian,
        masses=atomic_masses,
        dipole_derivatives=dipole_derivs,
        rotational_constants=rot_constants,
        dipole_moment=raw_data.get("dipole_moment"),
        v3_barrier_cm1=v3_barrier,
        hamiltonian_routing=hamiltonian_routing,
        lam_trigger=raw_data.get("lam_trigger", False),
        dvr_eigenvalues=raw_data.get("dvr_eigenvalues"),
        tunneling_splittings=raw_data.get("tunneling_splittings"),
        kraitchman_coords=raw_data.get("kraitchman_coords"),
    )


# =============================================================================
# QUANTUM EXTRACTOR: LOG PARSING (MPQC, ORCA, G-XTB)
# =============================================================================

class QuantumExtractor:
    """
    Extracts raw Hessians, dipole derivatives, rotational constants (A, B, C in MHz),
    and dipole components (X, Y, Z in Debye) from real MPQC and g-xTB outputs or HDF5 store.
    """

    def __init__(
        self,
        output_path: Optional[Union[str, Path]] = None,
        artifacts_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        self.output_path = Path(output_path).resolve() if output_path else None

        env_artifacts = os.getenv("COCHEM_ARTIFACTS_DIR") or os.getenv("COCHEM_ARTIFACT_DIR")
        if artifacts_dir:
            self.artifacts_dir = Path(artifacts_dir).resolve()
        elif env_artifacts:
            self.artifacts_dir = Path(env_artifacts).resolve()
        else:
            self.artifacts_dir = (Path.home() / ".cochem" / "artifacts").resolve()

        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create artifacts directory: {e}")

    def parse_mpqc_output(self, mpqc_output_text: Optional[str] = None) -> QuantumExtraction:
        """
        Parses real MPQC/ORCA output files/strings for rotational constants (A, B, C in MHz)
        and dipole components (X, Y, Z in Debye).
        """
        text = mpqc_output_text
        if text is None and self.output_path and self.output_path.exists():
            try:
                text = self.output_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                logger.error(f"Failed to read output path {self.output_path}: {e}")
                return QuantumExtraction()

        if not text:
            return QuantumExtraction()

        results = QuantumExtraction()

        # Match rotational constants (MHz or cm^-1)
        rot_match = re.search(
            r"Rotational constants\s*\((MHz|cm\^-1|cm-1)\)\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)",
            text,
            re.IGNORECASE,
        )
        if rot_match:
            unit = rot_match.group(1).lower()
            a, b, c = float(rot_match.group(2)), float(rot_match.group(3)), float(rot_match.group(4))
            if "cm" in unit:
                a, b, c = a * 29979.2458, b * 29979.2458, c * 29979.2458
            results.rotational_constants = RotationalConstants(A=a, B=b, C=c, unit="MHz")
        else:
            a_m = re.search(r"A\s*[:=]\s*([\d\.-]+)\s*MHz", text, re.IGNORECASE)
            b_m = re.search(r"B\s*[:=]\s*([\d\.-]+)\s*MHz", text, re.IGNORECASE)
            c_m = re.search(r"C\s*[:=]\s*([\d\.-]+)\s*MHz", text, re.IGNORECASE)
            if a_m and b_m and c_m:
                results.rotational_constants = RotationalConstants(
                    A=float(a_m.group(1)),
                    B=float(b_m.group(1)),
                    C=float(c_m.group(1)),
                    unit="MHz",
                )

        # Match total & component dipole moments (Debye)
        dip_match = re.search(
            r"Total Dipole Moment\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)",
            text,
            re.IGNORECASE,
        )
        if dip_match:
            dx, dy, dz = float(dip_match.group(1)), float(dip_match.group(2)), float(dip_match.group(3))
            tot = math.sqrt(dx * dx + dy * dy + dz * dz)
            results.dipole_moment = DipoleMoment(
                X=dx, Y=dy, Z=dz, total=tot, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
            )
        else:
            dip_comp = re.search(
                r"Dipole moment\s*\([^)]*\)\s*:\s*X\s*=\s*([\d\.-]+)\s+Y\s*=\s*([\d\.-]+)\s+Z\s*=\s*([\d\.-]+)",
                text,
                re.IGNORECASE,
            )
            if dip_comp:
                dx, dy, dz = float(dip_comp.group(1)), float(dip_comp.group(2)), float(dip_comp.group(3))
                tot = math.sqrt(dx * dx + dy * dy + dz * dz)
                results.dipole_moment = DipoleMoment(
                    X=dx, Y=dy, Z=dz, total=tot, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
                )

        # Spin contamination check for open-shell systems
        s2_match = re.search(r"Expectation value of <S\*\*2>\s*:\s*([\d\.]+)", text)
        ideal_s2_match = re.search(r"Ideal value S\*\S+\s+for S=[\d\.]+\s*:\s*([\d\.]+)", text)
        if s2_match and ideal_s2_match:
            s2_val = float(s2_match.group(1))
            ideal_s2_val = float(ideal_s2_match.group(1))
            if ideal_s2_val >= 0:
                contamination = abs(s2_val - ideal_s2_val) / ideal_s2_val if ideal_s2_val > 0 else abs(s2_val)
                if contamination > 0.10:
                    raise ValueError(f"Spin contamination exceeded 10%: <S**2> = {s2_val}, ideal = {ideal_s2_val}")

        return results

    parse_orca_output = parse_mpqc_output

    def parse_gxtb_output(self, gxtb_output_text: Optional[str] = None) -> QuantumExtraction:
        """
        Parses real g-xTB (GFN2-xTB) output files/strings for rotational constants (A, B, C in MHz)
        and dipole components (X, Y, Z in Debye).
        """
        text = gxtb_output_text
        if text is None and self.output_path and self.output_path.exists():
            try:
                text = self.output_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                logger.error(f"Failed to read output path {self.output_path}: {e}")
                return QuantumExtraction()

        if not text:
            return QuantumExtraction()

        results = QuantumExtraction()

        # g-xTB rotational constants match (cm-1 or MHz)
        rot_match = re.search(
            r"rotational constants\s*\((MHz|cm-1)\)\s*:\s*([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)",
            text,
            re.IGNORECASE,
        )
        if rot_match:
            unit = rot_match.group(1).lower()
            a, b, c = float(rot_match.group(2)), float(rot_match.group(3)), float(rot_match.group(4))
            if "cm" in unit:
                a, b, c = a * 29979.2458, b * 29979.2458, c * 29979.2458
            results.rotational_constants = RotationalConstants(A=a, B=b, C=c, unit="MHz")

        # g-xTB molecular dipole match
        dip_match = re.search(
            r"molecular dipole\s*:\s*x\s*=\s*([\d\.-]+)\s+y\s*=\s*([\d\.-]+)\s+z\s*=\s*([\d\.-]+)",
            text,
            re.IGNORECASE,
        )
        if dip_match:
            dx, dy, dz = float(dip_match.group(1)), float(dip_match.group(2)), float(dip_match.group(3))
            tot = math.sqrt(dx * dx + dy * dy + dz * dz)
            results.dipole_moment = DipoleMoment(
                X=dx, Y=dy, Z=dz, total=tot, mu_a=abs(dx), mu_b=abs(dy), mu_c=abs(dz)
            )

        return results

    def parse_output(self, text_or_path: Optional[Union[str, Path]] = None) -> QuantumExtraction:
        """Auto-detects MPQC vs g-xTB log format and parses rotational constants & dipole moments."""
        text = text_or_path
        if text is None and self.output_path and self.output_path.exists():
            try:
                text = self.output_path.read_text(encoding="utf-8", errors="ignore")
            except OSError as e:
                logger.error(f"Failed to read output path {self.output_path}: {e}")
                return QuantumExtraction()

        if text and isinstance(text, (str, Path)):
            potential_path = Path(text)
            if potential_path.exists() and potential_path.is_file():
                try:
                    text = potential_path.read_text(encoding="utf-8", errors="ignore")
                except OSError as e:
                    logger.error(f"Failed to read input path {potential_path}: {e}")
                    return QuantumExtraction()

        text_str = str(text) if text else ""
        if not text_str:
            return QuantumExtraction()

        if "MPQC" in text_str or "Program Version" in text_str or "Total Dipole Moment" in text_str:
            return self.parse_mpqc_output(text_str)
        elif "xtb" in text_str.lower() or "gfn" in text_str.lower() or "molecular dipole" in text_str.lower():
            return self.parse_gxtb_output(text_str)

        # Try MPQC first, fallback to g-xTB
        res = self.parse_mpqc_output(text_str)
        if res.rotational_constants is None:
            res = self.parse_gxtb_output(text_str)
        return res
