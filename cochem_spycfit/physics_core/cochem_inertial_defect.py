"""
CoChem-SpycFit: Analytical Inertial Defect & Planar Moment Physics Core.

Method Matrix Step 4 (§1.3, §6.4, §6.10) & WBS Phase 3.1 Mandate:
Computation, verification, and binding physical sign validation of:
1. Inertial defect:
       Delta = I_c - I_a - I_b = -2 * P_cc
2. Planar moments of inertia:
       P_aa = sum(m_i * a_i^2) = 0.5 * (-I_a + I_b + I_c) = 0.5 * (I_b + I_c - I_a)
       P_bb = sum(m_i * b_i^2) = 0.5 * ( I_a - I_b + I_c) = 0.5 * (I_a + I_c - I_b)
       P_cc = sum(m_i * c_i^2) = 0.5 * ( I_a + I_b - I_c) = 0.5 * (I_a + I_b - I_c)
3. Planar sum rules & identities:
       P_aa + P_bb = I_c
       P_aa + P_cc = I_b
       P_bb + P_cc = I_a
       P_aa + P_bb + P_cc = 0.5 * (I_a + I_b + I_c)
4. Binding Physical Sign Validation Rule:
       If sign(Delta_calc) != sign(Delta_exp), the assignment or structural candidate
       is physically invalid and must be rejected immediately, regardless of rotational
       constant proximity.
5. Planar Hierarchy & Permutation Guard:
       Validates physical ordering P_aa >= P_bb >= P_cc and enforces the vibrational
       zero-point floor on P_cc >= min_pcc_floor.
6. Pure JAX FP64 hardware-accelerated batch filtering and candidate pruning.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field

# Enable JAX FP64 precision globally for spectroscopic validation
jax.config.update("jax_enable_x64", True)

logger = logging.getLogger("CoChem-SpycFit.InertialDefect")

# =============================================================================
# 1. PHYSICAL CONSTANTS & CONVERSION FACTORS (CODATA / NIST)
# =============================================================================

# Conversion constant between rotational constants (MHz) and moments (amu*Angstrom^2):
# h / (8 * pi^2 * u * Angstrom^2) in MHz * amu * Angstrom^2
# CODATA recommended spectroscopic value: 505379.008 MHz * amu * Angstrom^2
AMU_ANGSTROM2_TO_MHZ: float = 505379.008
MHZ_TO_AMU_ANGSTROM2: float = 505379.008
DEFAULT_CONV_FACTOR_MHZ: float = 505379.008

# Default numerical tolerances
DEFAULT_ZERO_TOL: float = 1e-4       # Threshold for equilibrium planar Delta ~ 0 (amu*Angstrom^2)
DEFAULT_PLANAR_TOL: float = 1e-3     # Threshold for planar classification (amu*Angstrom^2)
DEFAULT_MIN_PCC_FLOOR: float = -0.25  # Lower vibrational zero-point floor for P_cc (amu*Angstrom^2)


# =============================================================================
# 2. DYNAMIC ATOMIC MASS RESOLUTION (Mendeleev Library Mandate)
# =============================================================================

def get_dynamic_atomic_mass(element_or_symbol: Union[str, int]) -> float:
    """
    Dynamically retrieves standard atomic or isotopic mass from the mendeleev library.
    Strictly forbids hardcoded mass values in compliance with Mendeleev Mandate.
    """
    if isinstance(element_or_symbol, int):
        el_obj = element(element_or_symbol)
        if el_obj is not None and el_obj.mass is not None:
            return float(el_obj.mass)
        raise ValueError(f"Atomic number {element_or_symbol} not found in mendeleev database.")

    sym_clean = str(element_or_symbol).strip()
    sym_cap = sym_clean.capitalize()

    # Hydrogen isotopes handling
    if sym_cap in ("D", "2h", "h2"):
        h_elem = element("H")
        for iso in h_elem.isotopes:
            if iso.mass_number == 2 and iso.mass is not None:
                return float(iso.mass)
    if sym_cap in ("T", "3h", "h3"):
        h_elem = element("H")
        for iso in h_elem.isotopes:
            if iso.mass_number == 3 and iso.mass is not None:
                return float(iso.mass)

    # General isotope match: e.g. "13C", "18O", "15N"
    iso_match = re.match(r"^(\d+)([A-Za-z]+)$", sym_clean)
    if iso_match:
        iso_mass_num = int(iso_match.group(1))
        iso_sym = iso_match.group(2).capitalize()
        el_iso = element(iso_sym)
        for iso in el_iso.isotopes:
            if iso.mass_number == iso_mass_num and iso.mass is not None:
                return float(iso.mass)

    # Standard element symbol
    base_sym = re.sub(r"^\d+", "", sym_clean).capitalize()
    el = element(base_sym)
    if el is not None and el.mass is not None:
        return float(el.mass)

    raise ValueError(f"Unable to dynamically resolve atomic mass for symbol '{element_or_symbol}'.")


def resolve_mass_array(elements: Sequence[Union[str, int]]) -> np.ndarray:
    """Converts a sequence of atomic symbols/numbers into an array of atomic masses."""
    masses = [get_dynamic_atomic_mass(sym) for sym in elements]
    return np.asarray(masses, dtype=np.float64)


# =============================================================================
# 3. TYPED EXCEPTION HIERARCHY
# =============================================================================

class InertialDefectError(ValueError):
    """Base exception for all inertial defect and planar moment validation operations."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InertialDefectSignError(InertialDefectError):
    """Raised when an unphysical sign inversion in the inertial defect (Delta) is detected."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlanarHierarchyError(InertialDefectError):
    """Raised when the physical planar moment ordering Paa >= Pbb >= Pcc is violated."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnphysicalAssignmentError(InertialDefectError):
    """Raised when a candidate assignment or fit state violates physical structural constraints."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PermutationError(InertialDefectError):
    """Raised when an axis permutation or irregular principal axis frame is encountered."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


# =============================================================================
# 4. PYDANTIC DATA MODELS & DIAGNOSTIC SCHEMAS
# =============================================================================

class PlanarMoments(BaseModel):
    """
    Planar moments of inertia P_aa, P_bb, P_cc in u*Angstrom^2 (amu*Angstrom^2).
    Definitions:
        P_aa = sum(m_i * a_i^2) = 0.5 * (-I_a + I_b + I_c)
        P_bb = sum(m_i * b_i^2) = 0.5 * ( I_a - I_b + I_c)
        P_cc = sum(m_i * c_i^2) = 0.5 * ( I_a + I_b - I_c)
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    P_aa: float = Field(..., description="Planar moment P_aa in u*Angstrom^2 (out of b-c plane mass)")
    P_bb: float = Field(..., description="Planar moment P_bb in u*Angstrom^2 (out of a-c plane mass)")
    P_cc: float = Field(..., description="Planar moment P_cc in u*Angstrom^2 (out of a-b plane mass)")

    @property
    def inertial_defect(self) -> float:
        """Inertial defect Delta = -2 * P_cc in u*Angstrom^2."""
        return -2.0 * self.P_cc

    @property
    def sum_planar(self) -> float:
        """Sum of planar moments P_aa + P_bb + P_cc = 0.5 * (Ia + Ib + Ic)."""
        return self.P_aa + self.P_bb + self.P_cc

    def to_principal_moments(self) -> Tuple[float, float, float]:
        """Reconstructs principal moments (Ia, Ib, Ic) from planar moments."""
        Ia = self.P_bb + self.P_cc
        Ib = self.P_aa + self.P_cc
        Ic = self.P_aa + self.P_bb
        return (float(Ia), float(Ib), float(Ic))


class MomentsOfInertia(BaseModel):
    """Principal moments of inertia (I_a <= I_b <= I_c in u*Angstrom^2)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    Ia: float = Field(..., gt=0.0, description="Principal moment Ia in u*Angstrom^2")
    Ib: float = Field(..., gt=0.0, description="Principal moment Ib in u*Angstrom^2")
    Ic: float = Field(..., gt=0.0, description="Principal moment Ic in u*Angstrom^2")

    @property
    def inertial_defect(self) -> float:
        """Inertial defect Delta = Ic - Ia - Ib in u*Angstrom^2."""
        return self.Ic - self.Ia - self.Ib

    def to_planar_moments(self) -> PlanarMoments:
        """Calculates planar moments P_aa, P_bb, P_cc."""
        P_aa = 0.5 * (-self.Ia + self.Ib + self.Ic)
        P_bb = 0.5 * (self.Ia - self.Ib + self.Ic)
        P_cc = 0.5 * (self.Ia + self.Ib - self.Ic)
        return PlanarMoments(P_aa=P_aa, P_bb=P_bb, P_cc=P_cc)

    def to_rotational_constants(self, conv_factor: float = DEFAULT_CONV_FACTOR_MHZ) -> Tuple[float, float, float]:
        """Converts principal moments to rotational constants (A >= B >= C in MHz)."""
        A = conv_factor / self.Ia
        B = conv_factor / self.Ib
        C = conv_factor / self.Ic
        return (float(A), float(B), float(C))


class RotationalConstants(BaseModel):
    """Rotational constants (A >= B >= C in MHz)."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    A: float = Field(..., gt=0.0, description="Rotational constant A in MHz")
    B: float = Field(..., gt=0.0, description="Rotational constant B in MHz")
    C: float = Field(..., gt=0.0, description="Rotational constant C in MHz")

    def to_moments(self, conv_factor: float = DEFAULT_CONV_FACTOR_MHZ) -> MomentsOfInertia:
        """Converts rotational constants to principal moments of inertia in u*Angstrom^2."""
        Ia = conv_factor / self.A
        Ib = conv_factor / self.B
        Ic = conv_factor / self.C
        return MomentsOfInertia(Ia=Ia, Ib=Ib, Ic=Ic)

    def compute_inertial_defect(self, conv_factor: float = DEFAULT_CONV_FACTOR_MHZ) -> float:
        """Computes the inertial defect Delta in u*Angstrom^2."""
        moms = self.to_moments(conv_factor)
        return moms.inertial_defect

    def compute_planar_moments(self, conv_factor: float = DEFAULT_CONV_FACTOR_MHZ) -> PlanarMoments:
        """Computes planar moments of inertia."""
        moms = self.to_moments(conv_factor)
        return moms.to_planar_moments()


class StructuralGateReport(BaseModel):
    """Full diagnostic payload generated by evaluate_structural_gate."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    is_valid: bool = Field(..., description="Overall physical validity boolean")
    delta_calc: float = Field(..., description="Calculated inertial defect in u*Angstrom^2")
    sign_calc: int = Field(..., description="Calculated sign (+1: ground state planar, 0: equilibrium planar, -1: non-planar)")
    is_planar_calc: bool = Field(..., description="Whether calculated system is planar within tolerance")
    planar_hierarchy_valid: bool = Field(..., description="Whether planar moments satisfy Paa >= Pbb >= Pcc")
    planar_permutation: str = Field(..., description="Axis permutation classification")
    planar_moments: Dict[str, float] = Field(..., description="Computed planar moments Paa, Pbb, Pcc")
    planar_violations: List[str] = Field(default_factory=list, description="List of planar hierarchy or floor violations")
    delta_exp: Optional[float] = Field(default=None, description="Experimental inertial defect in u*Angstrom^2")
    sign_exp: Optional[int] = Field(default=None, description="Experimental sign (+1, 0, -1)")
    is_planar_exp: Optional[bool] = Field(default=None, description="Whether experimental system is planar within tolerance")
    sign_valid: Optional[bool] = Field(default=None, description="Whether calculated and experimental signs agree physically")
    tolerance: float = Field(..., description="Applied numerical tolerance in u*Angstrom^2")


class CandidateFilteringResult(BaseModel):
    """Outcome schema for batch candidate filtering and unphysical assignment rejection."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    total_candidates: int = Field(..., description="Total input candidates evaluated")
    accepted_count: int = Field(..., description="Count of physically valid candidates")
    rejected_count: int = Field(..., description="Count of rejected candidates")
    accepted_indices: List[int] = Field(..., description="Indices of valid candidates")
    rejected_indices: List[int] = Field(..., description="Indices of rejected candidates")
    rejection_details: List[Dict[str, Any]] = Field(default_factory=list, description="Diagnostic records for rejected candidates")


# =============================================================================
# 5. CORE ANALYTICAL COMPUTATION FUNCTIONS
# =============================================================================

def compute_moments_of_inertia(
    coords: np.ndarray,
    masses: np.ndarray
) -> Tuple[float, float, float]:
    """
    Computes principal moments of inertia (I_a <= I_b <= I_c in u*Angstrom^2)
    from 3D Cartesian coordinates and atomic masses.

    Translates coordinates to the Center of Mass (COM), constructs the full
    symmetric Cartesian inertia tensor, and diagonalizes it via spectral decomposition.
    """
    coords_arr = np.asarray(coords, dtype=np.float64)
    masses_arr = np.asarray(masses, dtype=np.float64)

    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"Coordinates array must have shape (N, 3), received {coords_arr.shape}")
    if masses_arr.ndim != 1 or masses_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError(
            f"Masses length ({masses_arr.shape[0]}) must equal atom count ({coords_arr.shape[0]})"
        )

    total_mass = float(np.sum(masses_arr))
    if total_mass <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive.")

    # Center of mass translation
    center_of_mass = np.sum(coords_arr * masses_arr[:, np.newaxis], axis=0) / total_mass
    shifted = coords_arr - center_of_mass

    x = shifted[:, 0]
    y = shifted[:, 1]
    z = shifted[:, 2]

    # Symmetric inertia tensor elements
    I_xx = float(np.sum(masses_arr * (y**2 + z**2)))
    I_yy = float(np.sum(masses_arr * (x**2 + z**2)))
    I_zz = float(np.sum(masses_arr * (x**2 + y**2)))
    I_xy = float(-np.sum(masses_arr * x * y))
    I_xz = float(-np.sum(masses_arr * x * z))
    I_yz = float(-np.sum(masses_arr * y * z))

    inertia_tensor = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ], dtype=np.float64)

    eigenvalues = np.linalg.eigvalsh(inertia_tensor)
    sorted_eigenvalues = np.sort(eigenvalues)

    Ia = float(max(sorted_eigenvalues[0], 0.0))
    Ib = float(max(sorted_eigenvalues[1], 0.0))
    Ic = float(max(sorted_eigenvalues[2], 0.0))

    return Ia, Ib, Ic


def moments_from_rotational_constants(
    A: float,
    B: float,
    C: float,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[float, float, float]:
    """
    Converts rotational constants (A, B, C in MHz) into principal moments of inertia
    (I_a, I_b, I_c in u*Angstrom^2) using I = C_conv / nu.
    """
    if A <= 0.0 or B <= 0.0 or C <= 0.0:
        raise ValueError(f"Rotational constants must be strictly positive, received A={A}, B={B}, C={C}")

    Ia = conv_factor_mhz / float(A)
    Ib = conv_factor_mhz / float(B)
    Ic = conv_factor_mhz / float(C)

    return Ia, Ib, Ic


def rotational_constants_from_moments(
    Ia: float,
    Ib: float,
    Ic: float,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[float, float, float]:
    """
    Converts principal moments of inertia (I_a, I_b, I_c in u*Angstrom^2) into
    rotational constants (A, B, C in MHz) using nu = C_conv / I.
    """
    if Ia <= 0.0 or Ib <= 0.0 or Ic <= 0.0:
        raise ValueError(f"Principal moments must be strictly positive, received Ia={Ia}, Ib={Ib}, Ic={Ic}")

    A = conv_factor_mhz / float(Ia)
    B = conv_factor_mhz / float(Ib)
    C = conv_factor_mhz / float(Ic)

    return A, B, C


def compute_inertial_defect(
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    *,
    moments: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
    rot_constants: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
    coords: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> float:
    """
    Computes the inertial defect Delta = I_c - I_a - I_b (in u*Angstrom^2).

    Accepts:
    - Explicit moments: Ia, Ib, Ic
    - Sequence of moments: moments=(Ia, Ib, Ic)
    - Sequence of rotational constants: rot_constants=(A, B, C in MHz)
    - 3D Coordinates and atomic masses: coords=(N, 3), masses=(N,)
    """
    if coords is not None and masses is not None:
        Ia_val, Ib_val, Ic_val = compute_moments_of_inertia(coords, masses)
    elif rot_constants is not None:
        if len(rot_constants) != 3:
            raise ValueError(f"rot_constants sequence must contain 3 values (A, B, C), received {len(rot_constants)}")
        Ia_val, Ib_val, Ic_val = moments_from_rotational_constants(
            float(rot_constants[0]), float(rot_constants[1]), float(rot_constants[2]), conv_factor_mhz=conv_factor_mhz
        )
    elif moments is not None:
        if len(moments) != 3:
            raise ValueError(f"moments sequence must contain 3 values (Ia, Ib, Ic), received {len(moments)}")
        Ia_val, Ib_val, Ic_val = float(moments[0]), float(moments[1]), float(moments[2])
    elif Ia is not None and Ib is not None and Ic is not None:
        Ia_val, Ib_val, Ic_val = float(Ia), float(Ib), float(Ic)
    else:
        raise ValueError(
            "Insufficient arguments for inertial defect computation. "
            "Provide (Ia, Ib, Ic), moments, rot_constants, or (coords, masses)."
        )

    delta = Ic_val - Ia_val - Ib_val
    return float(delta)


def compute_inertial_defect_from_constants(
    A_mhz: float,
    B_mhz: float,
    C_mhz: float,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> float:
    """
    Convenience function computing inertial defect Delta from rotational constants (A, B, C in MHz).
    """
    return compute_inertial_defect(
        rot_constants=(float(A_mhz), float(B_mhz), float(C_mhz)),
        conv_factor_mhz=conv_factor_mhz
    )


def compute_planar_moments(
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    *,
    moments: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
    rot_constants: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
    coords: Optional[np.ndarray] = None,
    masses: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[float, float, float]:
    """
    Computes the three planar moments of inertia (in u*Angstrom^2):
        P_aa = 0.5 * (-I_a + I_b + I_c)
        P_bb = 0.5 * ( I_a - I_b + I_c)
        P_cc = 0.5 * ( I_a + I_b - I_c)
    """
    if coords is not None and masses is not None:
        Ia_val, Ib_val, Ic_val = compute_moments_of_inertia(coords, masses)
    elif rot_constants is not None:
        if len(rot_constants) != 3:
            raise ValueError(f"rot_constants sequence must contain 3 values (A, B, C), received {len(rot_constants)}")
        Ia_val, Ib_val, Ic_val = moments_from_rotational_constants(
            float(rot_constants[0]), float(rot_constants[1]), float(rot_constants[2]), conv_factor_mhz=conv_factor_mhz
        )
    elif moments is not None:
        if len(moments) != 3:
            raise ValueError(f"moments sequence must contain 3 values (Ia, Ib, Ic), received {len(moments)}")
        Ia_val, Ib_val, Ic_val = float(moments[0]), float(moments[1]), float(moments[2])
    elif Ia is not None and Ib is not None and Ic is not None:
        Ia_val, Ib_val, Ic_val = float(Ia), float(Ib), float(Ic)
    else:
        raise ValueError("Insufficient arguments for planar moments computation.")

    Paa = 0.5 * (-Ia_val + Ib_val + Ic_val)
    Pbb = 0.5 * (Ia_val - Ib_val + Ic_val)
    Pcc = 0.5 * (Ia_val + Ib_val - Ic_val)

    return float(Paa), float(Pbb), float(Pcc)


def resolve_inertial_defect(
    input_val: Any,
    masses: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> float:
    """
    Flexible input resolver extracting the scalar inertial defect Delta (in u*Angstrom^2)
    from diverse representations:
    - Scalar float or integer
    - 2D (N, 3) coordinates ndarray (with masses)
    - 1D 3-element tuple/list/ndarray of rotational constants (MHz) or moments
    - Dictionary payload with keys ('A', 'B', 'C'), ('Ia', 'Ib', 'Ic'), or ('Paa', 'Pbb', 'Pcc')
    """
    if isinstance(input_val, (int, float, np.floating, np.integer)):
        return float(input_val)

    if isinstance(input_val, dict):
        # Direct defect key
        for d_k in ("delta", "Delta", "inertial_defect", "inertial_defect_amu_a2"):
            if d_k in input_val:
                return float(input_val[d_k])
        # Planar moments: Delta = -2 * Pcc
        for p_c in ("Pcc", "P_cc", "pcc"):
            if p_c in input_val:
                return float(-2.0 * float(input_val[p_c]))
        # Rotational constants
        for a_k, b_k, c_k in (("A", "B", "C"), ("a", "b", "c"), ("rot_a", "rot_b", "rot_c")):
            if a_k in input_val and b_k in input_val and c_k in input_val:
                return compute_inertial_defect(
                    rot_constants=(float(input_val[a_k]), float(input_val[b_k]), float(input_val[c_k])),
                    conv_factor_mhz=conv_factor_mhz
                )
        # Principal moments
        for ia_k, ib_k, ic_k in (("Ia", "Ib", "Ic"), ("I_a", "I_b", "I_c"), ("ia", "ib", "ic")):
            if ia_k in input_val and ib_k in input_val and ic_k in input_val:
                return compute_inertial_defect(
                    Ia=float(input_val[ia_k]), Ib=float(input_val[ib_k]), Ic=float(input_val[ic_k]),
                    conv_factor_mhz=conv_factor_mhz
                )
        raise ValueError(f"Dictionary input lacks recognized keys for inertial defect: {list(input_val.keys())}")

    if isinstance(input_val, np.ndarray):
        if input_val.ndim == 2 and input_val.shape[1] == 3:
            if masses is None:
                raise ValueError("Atomic masses array required when 2D coordinates ndarray is provided.")
            return compute_inertial_defect(coords=input_val, masses=masses, conv_factor_mhz=conv_factor_mhz)
        elif input_val.ndim == 1 and len(input_val) == 3:
            input_val = tuple(input_val)

    if isinstance(input_val, (tuple, list)):
        if len(input_val) == 3:
            vals = [float(v) for v in input_val]
            # Heuristic distinction: A >= B >= C in MHz (rotational constants) vs Ia <= Ib <= Ic
            if vals[0] >= vals[1] >= vals[2] and (vals[0] > 100.0 or vals[0] > vals[2]):
                return compute_inertial_defect(rot_constants=vals, conv_factor_mhz=conv_factor_mhz)
            else:
                return compute_inertial_defect(moments=vals, conv_factor_mhz=conv_factor_mhz)

    raise TypeError(f"Cannot resolve inertial defect from input of type {type(input_val)}: {input_val}")


def resolve_planar_moments(
    input_val: Any = None,
    *,
    masses: Optional[np.ndarray] = None,
    is_planar_moments: bool = False,
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    A: Optional[float] = None,
    B: Optional[float] = None,
    C: Optional[float] = None,
    Paa: Optional[float] = None,
    Pbb: Optional[float] = None,
    Pcc: Optional[float] = None,
    coords: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[float, float, float]:
    """
    Resolves planar moments (P_aa, P_bb, P_cc in u*Angstrom^2) from diverse representations:
    - Explicit Paa, Pbb, Pcc arguments
    - Rotational constants (A, B, C in MHz)
    - Principal moments (Ia, Ib, Ic in u*Angstrom^2)
    - Cartesian coordinates (N, 3) and atomic masses (N,)
    - Dictionary payloads containing keys like 'Paa'/'P_aa', 'A'/'B'/'C', or 'Ia'/'Ib'/'Ic'
    - Tuples/lists/1D ndarrays of 3 elements.
    """
    if Paa is not None and Pbb is not None and Pcc is not None:
        return float(Paa), float(Pbb), float(Pcc)

    if A is not None and B is not None and C is not None:
        return compute_planar_moments(rot_constants=(float(A), float(B), float(C)), conv_factor_mhz=conv_factor_mhz)

    if Ia is not None and Ib is not None and Ic is not None:
        return compute_planar_moments(Ia=float(Ia), Ib=float(Ib), Ic=float(Ic), conv_factor_mhz=conv_factor_mhz)

    if coords is not None:
        if masses is None:
            raise ValueError("Atomic masses required when coordinates array is provided.")
        return compute_planar_moments(coords=np.asarray(coords, dtype=float), masses=np.asarray(masses, dtype=float))

    if isinstance(input_val, dict):
        for p_a, p_b, p_c in (("Paa", "Pbb", "Pcc"), ("P_aa", "P_bb", "P_cc"), ("paa", "pbb", "pcc")):
            if p_a in input_val and p_b in input_val and p_c in input_val:
                return float(input_val[p_a]), float(input_val[p_b]), float(input_val[p_c])
        for a_k, b_k, c_k in (("A", "B", "C"), ("a", "b", "c"), ("rot_a", "rot_b", "rot_c")):
            if a_k in input_val and b_k in input_val and c_k in input_val:
                return compute_planar_moments(
                    rot_constants=(float(input_val[a_k]), float(input_val[b_k]), float(input_val[c_k])),
                    conv_factor_mhz=conv_factor_mhz
                )
        for ia_k, ib_k, ic_k in (("Ia", "Ib", "Ic"), ("I_a", "I_b", "I_c"), ("ia", "ib", "ic")):
            if ia_k in input_val and ib_k in input_val and ic_k in input_val:
                return compute_planar_moments(
                    Ia=float(input_val[ia_k]), Ib=float(input_val[ib_k]), Ic=float(input_val[ic_k]),
                    conv_factor_mhz=conv_factor_mhz
                )
        raise ValueError(f"Dictionary input contains unrecognized keys for planar moments: {list(input_val.keys())}")

    if isinstance(input_val, np.ndarray) and input_val.ndim == 2:
        if masses is None:
            raise ValueError("Atomic masses required when coordinates ndarray is provided.")
        return compute_planar_moments(coords=input_val, masses=masses)

    if isinstance(input_val, (tuple, list, np.ndarray)):
        flat_vals = [float(v) for v in np.asarray(input_val, dtype=float).ravel()]
        if len(flat_vals) != 3:
            raise ValueError(f"Expected 3 values for planar moments resolution, received {len(flat_vals)}")
        if is_planar_moments:
            return flat_vals[0], flat_vals[1], flat_vals[2]
        if flat_vals[0] >= flat_vals[1] >= flat_vals[2] and (flat_vals[0] > 100.0 or flat_vals[0] > flat_vals[2]):
            return compute_planar_moments(rot_constants=flat_vals, conv_factor_mhz=conv_factor_mhz)
        else:
            return compute_planar_moments(moments=flat_vals, conv_factor_mhz=conv_factor_mhz)

    raise TypeError(f"Cannot resolve planar moments from input: {input_val} (type={type(input_val)})")


def get_defect_sign(delta: float, tol: float = DEFAULT_ZERO_TOL) -> int:
    """
    Returns physical classification sign for an inertial defect value:
      0 : Strictly planar / equilibrium ( |Delta| <= tol )
     +1 : Planar vibrational ground state ( Delta > tol )
     -1 : Non-planar out-of-plane structure ( Delta < -tol )
    """
    t = float(tol)
    if abs(delta) <= t:
        return 0
    return 1 if delta > 0.0 else -1


def validate_inertial_defect_sign(
    calc_input: Any,
    exp_input: Any,
    masses_calc: Optional[np.ndarray] = None,
    masses_exp: Optional[np.ndarray] = None,
    tol: float = DEFAULT_PLANAR_TOL,
    allow_equilibrium_planar_match: bool = True,
    raise_on_error: bool = False,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> bool:
    """
    Validates that the calculated inertial defect sign matches the experimental sign:
        sign(Delta_calc) == sign(Delta_exp)

    Method Matrix §6.4 & §6.10 Binding Validation Rule:
    - Planar systems (Delta >= 0): Both calc and exp must have Delta >= -tol.
    - Non-planar systems (Delta < 0): Both calc and exp must have Delta < -tol.
    - Sign inversion (planar vs non-planar conflict): Returns False or raises InertialDefectSignError.
    """
    t = float(tol)
    delta_calc = resolve_inertial_defect(calc_input, masses=masses_calc, conv_factor_mhz=conv_factor_mhz)
    delta_exp = resolve_inertial_defect(exp_input, masses=masses_exp, conv_factor_mhz=conv_factor_mhz)

    sign_calc = get_defect_sign(delta_calc, tol=t)
    sign_exp = get_defect_sign(delta_exp, tol=t)

    # 1. Exact sign match
    if sign_calc == sign_exp:
        logger.debug(
            f"Inertial defect sign match: Delta_calc={delta_calc:+.6f} ({sign_calc}), "
            f"Delta_exp={delta_exp:+.6f} ({sign_exp})"
        )
        return True

    # 2. Planar equilibrium (re, Delta ~ 0) vs planar ground state (r0, Delta > 0)
    if allow_equilibrium_planar_match:
        is_planar_calc = delta_calc >= -t
        is_planar_exp = delta_exp >= -t
        if is_planar_calc and is_planar_exp:
            logger.debug(
                f"Planar match accepted: Delta_calc={delta_calc:+.6f}, Delta_exp={delta_exp:+.6f}"
            )
            return True

    # 3. Physical sign inversion detected
    err_msg = (
        f"[REJECT: UNPHYSICAL INERTIAL DEFECT SIGN] Calculated Delta ({delta_calc:+.6f} u*A^2, sign={sign_calc:+d}) "
        f"and experimental Delta ({delta_exp:+.6f} u*A^2, sign={sign_exp:+d}) have contradictory signs "
        f"(tolerance = {t:.1e} u*A^2)."
    )
    logger.warning(err_msg)

    if raise_on_error:
        raise InertialDefectSignError(err_msg)

    return False


def detect_planar_permutation(
    input_val: Any = None,
    *,
    masses: Optional[np.ndarray] = None,
    tol: float = DEFAULT_ZERO_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    is_planar_moments: bool = False,
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    A: Optional[float] = None,
    B: Optional[float] = None,
    C: Optional[float] = None,
    Paa: Optional[float] = None,
    Pbb: Optional[float] = None,
    Pcc: Optional[float] = None,
    coords: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[str, Dict[str, Any]]:
    """
    Analyzes the planar moment ordering and detects any axis permutations (a <-> b, b <-> c, a <-> c)
    or unphysical vibrational floor violations.

    Returns:
        Tuple[str, Dict[str, Any]]:
            permutation_type: One of:
                - 'canonical': Paa >= Pbb >= Pcc (standard right-handed principal axes)
                - 'ab_swapped': Pbb > Paa >= Pcc (axes a and b swapped, Ia > Ib)
                - 'bc_swapped': Paa >= Pcc > Pbb (axes b and c swapped, Ib > Ic)
                - 'ac_swapped': Pcc > Paa >= Pbb (axes a and c swapped, Ia > Ic)
                - 'unphysical_floor': Pcc < min_pcc_floor (unphysical out-of-plane distortion)
                - 'irregular': general permutation violation
            diagnostics: Dictionary containing computed moments, differences, and validity flags.
    """
    Paa_val, Pbb_val, Pcc_val = resolve_planar_moments(
        input_val,
        masses=masses,
        is_planar_moments=is_planar_moments,
        Ia=Ia, Ib=Ib, Ic=Ic,
        A=A, B=B, C=C,
        Paa=Paa, Pbb=Pbb, Pcc=Pcc,
        coords=coords,
        conv_factor_mhz=conv_factor_mhz
    )

    diff_ab = Paa_val - Pbb_val
    diff_bc = Pbb_val - Pcc_val
    diff_ac = Paa_val - Pcc_val

    ordering_ab = (diff_ab + tol) >= 0.0
    ordering_bc = (diff_bc + tol) >= 0.0
    pcc_floor_ok = Pcc_val >= (min_pcc_floor - tol)

    diagnostics: Dict[str, Any] = {
        "Paa": float(Paa_val),
        "Pbb": float(Pbb_val),
        "Pcc": float(Pcc_val),
        "diff_ab": float(diff_ab),
        "diff_bc": float(diff_bc),
        "diff_ac": float(diff_ac),
        "ordering_ab_valid": bool(ordering_ab),
        "ordering_bc_valid": bool(ordering_bc),
        "pcc_floor_valid": bool(pcc_floor_ok),
        "min_pcc_floor": float(min_pcc_floor),
        "tolerance": float(tol)
    }

    if not pcc_floor_ok:
        permutation_type = "unphysical_floor"
    elif ordering_ab and ordering_bc:
        permutation_type = "canonical"
    elif not ordering_ab and ordering_bc and (Paa_val + tol >= Pcc_val):
        permutation_type = "ab_swapped"
    elif ordering_ab and not ordering_bc and (Paa_val + tol >= Pcc_val):
        permutation_type = "bc_swapped"
    elif Pcc_val > (Paa_val + tol) and (Paa_val + tol >= Pbb_val):
        permutation_type = "ac_swapped"
    else:
        permutation_type = "irregular"

    diagnostics["permutation_type"] = permutation_type
    diagnostics["is_canonical"] = bool(permutation_type == "canonical")
    return permutation_type, diagnostics


def validate_planar_hierarchy(
    input_val: Any = None,
    *,
    masses: Optional[np.ndarray] = None,
    tol: float = DEFAULT_ZERO_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    raise_on_error: bool = False,
    return_diagnostics: bool = False,
    is_planar_moments: bool = False,
    Ia: Optional[float] = None,
    Ib: Optional[float] = None,
    Ic: Optional[float] = None,
    A: Optional[float] = None,
    B: Optional[float] = None,
    C: Optional[float] = None,
    Paa: Optional[float] = None,
    Pbb: Optional[float] = None,
    Pcc: Optional[float] = None,
    coords: Optional[np.ndarray] = None,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
    """
    Verifies the physical planar moment ordering: P_aa >= P_bb >= P_cc.
    For any physically valid right-handed principal axis system with sorted moments (Ia <= Ib <= Ic):
        P_aa - P_bb = I_b - I_a >= 0
        P_bb - P_cc = I_c - I_b >= 0

    Also verifies that P_cc does not take unphysical large negative values beyond the vibrational
    zero-point defect floor (P_cc >= min_pcc_floor, where min_pcc_floor defaults to -0.25 u*A^2).
    """
    perm_type, diag = detect_planar_permutation(
        input_val,
        masses=masses,
        tol=tol,
        min_pcc_floor=min_pcc_floor,
        is_planar_moments=is_planar_moments,
        Ia=Ia, Ib=Ib, Ic=Ic,
        A=A, B=B, C=C,
        Paa=Paa, Pbb=Pbb, Pcc=Pcc,
        coords=coords,
        conv_factor_mhz=conv_factor_mhz
    )

    Paa_val = diag["Paa"]
    Pbb_val = diag["Pbb"]
    Pcc_val = diag["Pcc"]
    diff_ab = diag["diff_ab"]
    diff_bc = diag["diff_bc"]
    ordering_ab = diag["ordering_ab_valid"]
    ordering_bc = diag["ordering_bc_valid"]
    pcc_floor_ok = diag["pcc_floor_valid"]

    is_valid = ordering_ab and ordering_bc and pcc_floor_ok
    diag["is_valid"] = bool(is_valid)

    violations: List[str] = []
    if not ordering_ab:
        violations.append(f"Paa ({Paa_val:.6f}) < Pbb ({Pbb_val:.6f}) [diff={diff_ab:+.6f}]")
    if not ordering_bc:
        violations.append(f"Pbb ({Pbb_val:.6f}) < Pcc ({Pcc_val:.6f}) [diff={diff_bc:+.6f}]")
    if not pcc_floor_ok:
        violations.append(f"Pcc ({Pcc_val:.6f}) < min_floor ({min_pcc_floor:+.6f})")

    diag["violations"] = violations

    if not is_valid:
        err_msg = (
            f"[REJECT: PLANAR MOMENT HIERARCHY VIOLATION] Planar moments (Paa={Paa_val:.6f}, Pbb={Pbb_val:.6f}, Pcc={Pcc_val:.6f}) "
            f"violate physical ordering Paa >= Pbb >= Pcc or exceed vibrational floor ({min_pcc_floor} u*A^2): "
            f"{'; '.join(violations)} (classification='{perm_type}')."
        )
        logger.warning(err_msg)
        if raise_on_error:
            raise PlanarHierarchyError(err_msg)

    if return_diagnostics:
        return is_valid, diag
    return is_valid


def evaluate_structural_gate(
    calc_input: Any,
    exp_input: Optional[Any] = None,
    masses_calc: Optional[np.ndarray] = None,
    masses_exp: Optional[np.ndarray] = None,
    tol: float = DEFAULT_PLANAR_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    allow_equilibrium_planar_match: bool = True,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Dict[str, Any]:
    """
    Comprehensive evaluation gate combining sign validation and planar hierarchy checks.
    Returns a dictionary with full structural diagnostics compliant with StructuralGateReport.
    """
    t = float(tol)
    floor = float(min_pcc_floor)
    delta_calc = resolve_inertial_defect(calc_input, masses=masses_calc, conv_factor_mhz=conv_factor_mhz)
    sign_calc = get_defect_sign(delta_calc, tol=t)

    result: Dict[str, Any] = {
        "delta_calc": float(delta_calc),
        "sign_calc": int(sign_calc),
        "is_planar_calc": bool(delta_calc >= -t),
        "tolerance": float(t)
    }

    try:
        hierarchy_ok, diag = validate_planar_hierarchy(
            calc_input,
            masses=masses_calc,
            tol=t,
            min_pcc_floor=floor,
            return_diagnostics=True,
            conv_factor_mhz=conv_factor_mhz
        )
        result["planar_hierarchy_valid"] = bool(hierarchy_ok)
        result["planar_moments"] = {
            "Paa": diag["Paa"],
            "Pbb": diag["Pbb"],
            "Pcc": diag["Pcc"]
        }
        result["planar_permutation"] = diag["permutation_type"]
        result["planar_violations"] = diag.get("violations", [])
    except Exception as exc:
        result["planar_hierarchy_valid"] = False
        result["planar_moments"] = {"Paa": 0.0, "Pbb": 0.0, "Pcc": 0.0}
        result["planar_permutation"] = "error"
        result["planar_violations"] = [str(exc)]

    if exp_input is not None:
        delta_exp = resolve_inertial_defect(exp_input, masses=masses_exp, conv_factor_mhz=conv_factor_mhz)
        sign_exp = get_defect_sign(delta_exp, tol=t)
        sign_valid = validate_inertial_defect_sign(
            calc_input, exp_input,
            masses_calc=masses_calc, masses_exp=masses_exp,
            tol=t,
            allow_equilibrium_planar_match=allow_equilibrium_planar_match,
            conv_factor_mhz=conv_factor_mhz
        )
        result["delta_exp"] = float(delta_exp)
        result["sign_exp"] = int(sign_exp)
        result["is_planar_exp"] = bool(delta_exp >= -t)
        result["sign_valid"] = bool(sign_valid)
        result["is_valid"] = bool(sign_valid and result.get("planar_hierarchy_valid", True))
    else:
        result["is_valid"] = bool(result.get("planar_hierarchy_valid", True))

    return result


# =============================================================================
# 6. BATCH CANDIDATE & TRAJECTORY FILTERING ROUTINES
# =============================================================================

def filter_unphysical_assignments(
    candidates: Sequence[Union[Tuple[float, float, float], Dict[str, float], np.ndarray]],
    exp_reference: Optional[Any] = None,
    masses: Optional[np.ndarray] = None,
    tol: float = DEFAULT_PLANAR_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    return_diagnostics: bool = False,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Union[List[Any], Tuple[List[Any], Dict[str, Any]]]:
    """
    Filters a sequence of candidate rotational constant sets or moments, rejecting
    any candidate violating planar moment hierarchy or showing sign conflicts against
    the experimental reference.
    """
    accepted_candidates: List[Any] = []
    accepted_indices: List[int] = []
    rejected_indices: List[int] = []
    rejection_records: List[Dict[str, Any]] = []

    total_candidates = len(candidates)

    for idx, cand in enumerate(candidates):
        gate_res = evaluate_structural_gate(
            calc_input=cand,
            exp_input=exp_reference,
            masses_calc=masses,
            tol=tol,
            min_pcc_floor=min_pcc_floor,
            conv_factor_mhz=conv_factor_mhz
        )
        if gate_res["is_valid"]:
            accepted_candidates.append(cand)
            accepted_indices.append(idx)
        else:
            rejected_indices.append(idx)
            reasons: List[str] = []
            if gate_res.get("sign_valid") is False:
                reasons.append(
                    f"Inertial defect sign conflict: calc={gate_res['delta_calc']:+.4f} vs exp={gate_res.get('delta_exp', 0.0):+.4f}"
                )
            if gate_res.get("planar_hierarchy_valid") is False:
                reasons.extend(gate_res.get("planar_violations", ["Planar moment hierarchy violation"]))
            rejection_records.append({
                "candidate_index": idx,
                "reasons": reasons,
                "gate_report": gate_res
            })

    diagnostics: Dict[str, Any] = {
        "total_candidates": total_candidates,
        "accepted_count": len(accepted_candidates),
        "rejected_count": len(rejected_indices),
        "accepted_indices": accepted_indices,
        "rejected_indices": rejected_indices,
        "rejection_details": rejection_records
    }

    if return_diagnostics:
        return accepted_candidates, diagnostics
    return accepted_candidates


def filter_unphysical_trajectories(
    trajectories: np.ndarray,
    masses: np.ndarray,
    exp_reference: Optional[Any] = None,
    elements: Optional[Sequence[Union[str, int]]] = None,
    tol: float = DEFAULT_PLANAR_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    return_diagnostics: bool = False,
    conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Filters candidate 3D trajectory steps (N_frames, N_atoms, 3) or coordinate frames,
    discarding unphysical steps with coordinate divergence (NaN/Inf, overlapping atoms),
    planar moment hierarchy violations, or inertial defect sign conflicts.
    """
    traj_arr = np.asarray(trajectories, dtype=np.float64)
    if traj_arr.ndim == 2 and traj_arr.shape[1] == 3:
        traj_arr = traj_arr[np.newaxis, :, :]

    if traj_arr.ndim != 3 or traj_arr.shape[2] != 3:
        raise ValueError(f"Trajectories array must have shape (N_frames, N_atoms, 3), received {traj_arr.shape}")

    masses_arr = resolve_mass_array(elements) if elements is not None else np.asarray(masses, dtype=np.float64)

    total_frames = traj_arr.shape[0]
    accepted_frames_list: List[np.ndarray] = []
    accepted_indices: List[int] = []
    rejected_indices: List[int] = []
    rejection_records: List[Dict[str, Any]] = []

    for frame_idx in range(total_frames):
        frame_coords = traj_arr[frame_idx]

        # Check for NaN / Inf
        if not np.all(np.isfinite(frame_coords)):
            rejected_indices.append(frame_idx)
            rejection_records.append({
                "frame_index": frame_idx,
                "reasons": ["Non-finite coordinates detected (NaN or Inf)"]
            })
            continue

        # Check for unphysical atom overlaps (< 0.2 Angstrom between distinct atoms)
        n_atoms = frame_coords.shape[0]
        has_overlap = False
        if n_atoms > 1:
            diffs = frame_coords[:, np.newaxis, :] - frame_coords[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diffs**2, axis=-1))
            np.fill_diagonal(dists, 999.0)
            if np.any(dists < 0.2):
                has_overlap = True

        if has_overlap:
            rejected_indices.append(frame_idx)
            rejection_records.append({
                "frame_index": frame_idx,
                "reasons": ["Unphysical atomic overlap detected (interatomic distance < 0.2 Angstrom)"]
            })
            continue

        # Physical structural gate evaluation
        try:
            gate_res = evaluate_structural_gate(
                calc_input=frame_coords,
                exp_input=exp_reference,
                masses_calc=masses_arr,
                tol=tol,
                min_pcc_floor=min_pcc_floor,
                conv_factor_mhz=conv_factor_mhz
            )
        except Exception as exc:
            rejected_indices.append(frame_idx)
            rejection_records.append({
                "frame_index": frame_idx,
                "reasons": [f"Evaluation exception: {str(exc)}"]
            })
            continue

        if gate_res["is_valid"]:
            accepted_frames_list.append(frame_coords)
            accepted_indices.append(frame_idx)
        else:
            rejected_indices.append(frame_idx)
            reasons = []
            if gate_res.get("sign_valid") is False:
                reasons.append(
                    f"Inertial defect sign conflict: calc={gate_res['delta_calc']:+.4f} vs exp={gate_res.get('delta_exp', 0.0):+.4f}"
                )
            if gate_res.get("planar_hierarchy_valid") is False:
                reasons.extend(gate_res.get("planar_violations", ["Planar moment hierarchy violation"]))
            rejection_records.append({
                "frame_index": frame_idx,
                "reasons": reasons,
                "gate_report": gate_res
            })

    filtered_trajectories = np.array(accepted_frames_list, dtype=np.float64) if accepted_frames_list else np.empty((0, masses_arr.shape[0], 3), dtype=np.float64)

    diagnostics = {
        "total_frames": total_frames,
        "accepted_count": len(accepted_indices),
        "rejected_count": len(rejected_indices),
        "accepted_indices": accepted_indices,
        "rejected_indices": rejected_indices,
        "rejection_details": rejection_records
    }

    if return_diagnostics:
        return filtered_trajectories, diagnostics
    return filtered_trajectories


# =============================================================================
# 7. JAX FP64 VECTORIZED HIGH-PERFORMANCE ROUTINES
# =============================================================================

@jax.jit
def compute_inertial_defect_jax(
    a_mhz: Union[float, jnp.ndarray],
    b_mhz: Union[float, jnp.ndarray],
    c_mhz: Union[float, jnp.ndarray],
    conv_factor: float = DEFAULT_CONV_FACTOR_MHZ
) -> jnp.ndarray:
    """
    Pure-JAX hardware-accelerated calculation of the inertial defect Delta.
    Delta = I_c - I_a - I_b = conv_factor * (1/C - 1/A - 1/B).
    """
    Ia = conv_factor / a_mhz
    Ib = conv_factor / b_mhz
    Ic = conv_factor / c_mhz
    return Ic - Ia - Ib


@jax.jit
def compute_planar_moments_jax(
    a_mhz: Union[float, jnp.ndarray],
    b_mhz: Union[float, jnp.ndarray],
    c_mhz: Union[float, jnp.ndarray],
    conv_factor: float = DEFAULT_CONV_FACTOR_MHZ
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Pure-JAX hardware-accelerated calculation of planar moments (Paa, Pbb, Pcc).
    """
    Ia = conv_factor / a_mhz
    Ib = conv_factor / b_mhz
    Ic = conv_factor / c_mhz

    Paa = 0.5 * (-Ia + Ib + Ic)
    Pbb = 0.5 * (Ia - Ib + Ic)
    Pcc = 0.5 * (Ia + Ib - Ic)

    return Paa, Pbb, Pcc


@jax.jit
def filter_unphysical_constants_jax(
    a_batch: jnp.ndarray,
    b_batch: jnp.ndarray,
    c_batch: jnp.ndarray,
    exp_delta: float,
    tol: float = DEFAULT_PLANAR_TOL,
    min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
    conv_factor: float = DEFAULT_CONV_FACTOR_MHZ
) -> jnp.ndarray:
    """
    Vectorized JAX evaluation of physical validity mask across candidate rotational constants.
    Returns a boolean mask where True indicates physical validity.
    """
    Paa, Pbb, Pcc = compute_planar_moments_jax(a_batch, b_batch, c_batch, conv_factor)
    calc_delta = -2.0 * Pcc

    # Hierarchy validation
    hierarchy_ok = (Paa + tol >= Pbb) & (Pbb + tol >= Pcc) & (Pcc >= min_pcc_floor - tol)

    # Sign validation
    is_planar_calc = calc_delta >= -tol
    is_planar_exp = exp_delta >= -tol
    sign_ok = (is_planar_calc & is_planar_exp) | ((calc_delta < -tol) & (exp_delta < -tol))

    return hierarchy_ok & sign_ok


# =============================================================================
# 8. THE InertialDefectValidator CLASS
# =============================================================================

class InertialDefectValidator:
    """
    CoChem High-Performance Inertial Defect & Planar Moment Validator.

    Enforces Method Matrix Step 4 (§1.3, §6.4, §6.10) & WBS Phase 3.1:
    - Verifies sign consistency sign(Delta_calc) == sign(Delta_exp).
    - Rejects unphysical planar <-> non-planar assignment contradictions.
    - Validates planar moment ordering P_aa >= P_bb >= P_cc and vibrational floor P_cc >= min_floor.
    - Gating mechanism for SpycFit JAX fitting loops, AI assignment engines, and candidate trajectories.
    """

    DEFAULT_CONV_FACTOR_MHZ: float = DEFAULT_CONV_FACTOR_MHZ

    def __init__(
        self,
        conv_factor_mhz: float = DEFAULT_CONV_FACTOR_MHZ,
        zero_tol: float = DEFAULT_ZERO_TOL,
        planar_tol: float = DEFAULT_PLANAR_TOL,
        min_pcc_floor: float = DEFAULT_MIN_PCC_FLOOR,
        logger_instance: Optional[logging.Logger] = None
    ) -> None:
        self.conv_factor_mhz = float(conv_factor_mhz)
        self.zero_tol = float(zero_tol)
        self.planar_tol = float(planar_tol)
        self.min_pcc_floor = float(min_pcc_floor)
        self.logger = logger_instance or logging.getLogger("CoChem-SpycFit.InertialDefectValidator")

    def compute_moments_of_inertia(
        self,
        coords: np.ndarray,
        masses: np.ndarray
    ) -> Tuple[float, float, float]:
        """Computes principal moments of inertia from coordinates and atomic masses."""
        return compute_moments_of_inertia(coords, masses)

    def moments_from_rotational_constants(
        self,
        A: float,
        B: float,
        C: float
    ) -> Tuple[float, float, float]:
        """Converts rotational constants (A, B, C in MHz) into principal moments (Ia, Ib, Ic)."""
        return moments_from_rotational_constants(A, B, C, conv_factor_mhz=self.conv_factor_mhz)

    def rotational_constants_from_moments(
        self,
        Ia: float,
        Ib: float,
        Ic: float
    ) -> Tuple[float, float, float]:
        """Converts principal moments (Ia, Ib, Ic) into rotational constants (A, B, C in MHz)."""
        return rotational_constants_from_moments(Ia, Ib, Ic, conv_factor_mhz=self.conv_factor_mhz)

    def compute_inertial_defect(
        self,
        Ia: Optional[float] = None,
        Ib: Optional[float] = None,
        Ic: Optional[float] = None,
        *,
        moments: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        rot_constants: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        coords: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None
    ) -> float:
        """Computes the inertial defect Delta = Ic - Ia - Ib (in u*Angstrom^2)."""
        return compute_inertial_defect(
            Ia=Ia, Ib=Ib, Ic=Ic,
            moments=moments,
            rot_constants=rot_constants,
            coords=coords,
            masses=masses,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def compute_planar_moments(
        self,
        Ia: Optional[float] = None,
        Ib: Optional[float] = None,
        Ic: Optional[float] = None,
        *,
        moments: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        rot_constants: Optional[Union[Tuple[float, float, float], List[float], np.ndarray]] = None,
        coords: Optional[np.ndarray] = None,
        masses: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Computes planar moments of inertia (P_aa, P_bb, P_cc in u*Angstrom^2)."""
        return compute_planar_moments(
            Ia=Ia, Ib=Ib, Ic=Ic,
            moments=moments,
            rot_constants=rot_constants,
            coords=coords,
            masses=masses,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def resolve_inertial_defect(
        self,
        input_val: Any,
        masses: Optional[np.ndarray] = None
    ) -> float:
        """Extracts inertial defect Delta from scalar, sequence, dict, or coordinates."""
        return resolve_inertial_defect(input_val, masses=masses, conv_factor_mhz=self.conv_factor_mhz)

    def resolve_planar_moments(
        self,
        input_val: Any = None,
        *,
        masses: Optional[np.ndarray] = None,
        is_planar_moments: bool = False,
        Ia: Optional[float] = None,
        Ib: Optional[float] = None,
        Ic: Optional[float] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        C: Optional[float] = None,
        Paa: Optional[float] = None,
        Pbb: Optional[float] = None,
        Pcc: Optional[float] = None,
        coords: Optional[np.ndarray] = None
    ) -> Tuple[float, float, float]:
        """Resolves planar moments (P_aa, P_bb, P_cc) from diverse input representations."""
        return resolve_planar_moments(
            input_val,
            masses=masses,
            is_planar_moments=is_planar_moments,
            Ia=Ia, Ib=Ib, Ic=Ic,
            A=A, B=B, C=C,
            Paa=Paa, Pbb=Pbb, Pcc=Pcc,
            coords=coords,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def get_defect_sign(self, delta: float, tol: Optional[float] = None) -> int:
        """Returns physical classification sign for an inertial defect."""
        t = self.zero_tol if tol is None else float(tol)
        return get_defect_sign(delta, tol=t)

    def validate_sign(
        self,
        calc_input: Any,
        exp_input: Any,
        masses_calc: Optional[np.ndarray] = None,
        masses_exp: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        allow_equilibrium_planar_match: bool = True,
        raise_on_error: bool = False
    ) -> bool:
        """Validates that calculated and experimental inertial defects have compatible physical signs."""
        t = self.planar_tol if tol is None else float(tol)
        return validate_inertial_defect_sign(
            calc_input=calc_input,
            exp_input=exp_input,
            masses_calc=masses_calc,
            masses_exp=masses_exp,
            tol=t,
            allow_equilibrium_planar_match=allow_equilibrium_planar_match,
            raise_on_error=raise_on_error,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def detect_planar_permutation(
        self,
        input_val: Any = None,
        *,
        masses: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        min_pcc_floor: Optional[float] = None,
        is_planar_moments: bool = False,
        Ia: Optional[float] = None,
        Ib: Optional[float] = None,
        Ic: Optional[float] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        C: Optional[float] = None,
        Paa: Optional[float] = None,
        Pbb: Optional[float] = None,
        Pcc: Optional[float] = None,
        coords: Optional[np.ndarray] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Detects principal axis permutations or vibrational floor violations."""
        t = self.zero_tol if tol is None else float(tol)
        floor = self.min_pcc_floor if min_pcc_floor is None else float(min_pcc_floor)
        return detect_planar_permutation(
            input_val=input_val,
            masses=masses,
            tol=t,
            min_pcc_floor=floor,
            is_planar_moments=is_planar_moments,
            Ia=Ia, Ib=Ib, Ic=Ic,
            A=A, B=B, C=C,
            Paa=Paa, Pbb=Pbb, Pcc=Pcc,
            coords=coords,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def validate_planar_hierarchy(
        self,
        input_val: Any = None,
        *,
        masses: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        min_pcc_floor: Optional[float] = None,
        raise_on_error: bool = False,
        return_diagnostics: bool = False,
        is_planar_moments: bool = False,
        Ia: Optional[float] = None,
        Ib: Optional[float] = None,
        Ic: Optional[float] = None,
        A: Optional[float] = None,
        B: Optional[float] = None,
        C: Optional[float] = None,
        Paa: Optional[float] = None,
        Pbb: Optional[float] = None,
        Pcc: Optional[float] = None,
        coords: Optional[np.ndarray] = None
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        """Validates that planar moments satisfy physical ordering Paa >= Pbb >= Pcc and floor."""
        t = self.zero_tol if tol is None else float(tol)
        floor = self.min_pcc_floor if min_pcc_floor is None else float(min_pcc_floor)
        return validate_planar_hierarchy(
            input_val=input_val,
            masses=masses,
            tol=t,
            min_pcc_floor=floor,
            raise_on_error=raise_on_error,
            return_diagnostics=return_diagnostics,
            is_planar_moments=is_planar_moments,
            Ia=Ia, Ib=Ib, Ic=Ic,
            A=A, B=B, C=C,
            Paa=Paa, Pbb=Pbb, Pcc=Pcc,
            coords=coords,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def evaluate_structural_gate(
        self,
        calc_input: Any,
        exp_input: Optional[Any] = None,
        masses_calc: Optional[np.ndarray] = None,
        masses_exp: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        min_pcc_floor: Optional[float] = None,
        allow_equilibrium_planar_match: bool = True
    ) -> Dict[str, Any]:
        """Evaluates complete structural validity combining sign and hierarchy gates."""
        t = self.planar_tol if tol is None else float(tol)
        floor = self.min_pcc_floor if min_pcc_floor is None else float(min_pcc_floor)
        return evaluate_structural_gate(
            calc_input=calc_input,
            exp_input=exp_input,
            masses_calc=masses_calc,
            masses_exp=masses_exp,
            tol=t,
            min_pcc_floor=floor,
            allow_equilibrium_planar_match=allow_equilibrium_planar_match,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def filter_unphysical_assignments(
        self,
        candidates: Sequence[Union[Tuple[float, float, float], Dict[str, float], np.ndarray]],
        exp_reference: Optional[Any] = None,
        masses: Optional[np.ndarray] = None,
        tol: Optional[float] = None,
        min_pcc_floor: Optional[float] = None,
        return_diagnostics: bool = False
    ) -> Union[List[Any], Tuple[List[Any], Dict[str, Any]]]:
        """Filters candidate rotational constants or moments against sign and hierarchy rules."""
        t = self.planar_tol if tol is None else float(tol)
        floor = self.min_pcc_floor if min_pcc_floor is None else float(min_pcc_floor)
        return filter_unphysical_assignments(
            candidates=candidates,
            exp_reference=exp_reference,
            masses=masses,
            tol=t,
            min_pcc_floor=floor,
            return_diagnostics=return_diagnostics,
            conv_factor_mhz=self.conv_factor_mhz
        )

    def filter_unphysical_trajectories(
        self,
        trajectories: np.ndarray,
        masses: np.ndarray,
        exp_reference: Optional[Any] = None,
        elements: Optional[Sequence[Union[str, int]]] = None,
        tol: Optional[float] = None,
        min_pcc_floor: Optional[float] = None,
        return_diagnostics: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Filters candidate 3D trajectory steps."""
        t = self.planar_tol if tol is None else float(tol)
        floor = self.min_pcc_floor if min_pcc_floor is None else float(min_pcc_floor)
        return filter_unphysical_trajectories(
            trajectories=trajectories,
            masses=masses,
            exp_reference=exp_reference,
            elements=elements,
            tol=t,
            min_pcc_floor=floor,
            return_diagnostics=return_diagnostics,
            conv_factor_mhz=self.conv_factor_mhz
        )


__all__ = [
    "AMU_ANGSTROM2_TO_MHZ",
    "CandidateFilteringResult",
    "DEFAULT_CONV_FACTOR_MHZ",
    "DEFAULT_MIN_PCC_FLOOR",
    "DEFAULT_PLANAR_TOL",
    "DEFAULT_ZERO_TOL",
    "InertialDefectError",
    "InertialDefectSignError",
    "InertialDefectValidator",
    "MHZ_TO_AMU_ANGSTROM2",
    "MomentsOfInertia",
    "PermutationError",
    "PlanarHierarchyError",
    "PlanarMoments",
    "RotationalConstants",
    "StructuralGateReport",
    "UnphysicalAssignmentError",
    "compute_inertial_defect",
    "compute_inertial_defect_from_constants",
    "compute_inertial_defect_jax",
    "compute_moments_of_inertia",
    "compute_planar_moments",
    "compute_planar_moments_jax",
    "detect_planar_permutation",
    "evaluate_structural_gate",
    "filter_unphysical_assignments",
    "filter_unphysical_constants_jax",
    "filter_unphysical_trajectories",
    "get_defect_sign",
    "get_dynamic_atomic_mass",
    "moments_from_rotational_constants",
    "resolve_inertial_defect",
    "resolve_mass_array",
    "resolve_planar_moments",
    "rotational_constants_from_moments",
    "validate_inertial_defect_sign",
    "validate_planar_hierarchy",
]
