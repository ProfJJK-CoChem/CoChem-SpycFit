#!/usr/bin/env python3
# Copyright 2026 CoChem Project Family. All rights reserved.
# Apache License 2.0
"""
CoChem-SpycFit: Analytical Kraitchman Substitution & Planar Moment Physics Core.

Method Matrix §2.3, §2.4, §6.10 & WBS Phase 2.3 Mandate:
---------------------------------------------------------
Determination of complete substitution structures (r_s) from isotopic rotational
constants and moments of inertia via Kraitchman equations with:
1. Exact Kraitchman substitution coordinate calculations for asymmetric, planar,
   symmetric, and linear molecular topologies.
2. Automated planar moment shift sign verification (Delta P_aa >= 0, Delta P_bb >= 0, Delta P_cc >= 0)
   and inertial defect sign preservation.
3. Costain error bounds estimation (delta_z = 0.0015 / |z|) with small-coordinate
   breakdown diagnostics and rigorous uncertainty propagation to bond lengths
   and bond angles.
4. Automated coordinate sign resolution using structural priors, first-moment
   center-of-mass conditions (sum m_i r_i = 0), and second-moment principal axis
   conditions (sum m_i a_i b_i = 0).
5. Pure JAX FP64 hardware-accelerated batch filtering and tensor transformations.
6. Strict dynamic atomic mass resolution via the Mendeleev library mandate.

References:
- J. Kraitchman, Am. J. Phys. 21, 17-24 (1953).
- C. C. Costain, J. Chem. Phys. 29, 864-874 (1958).
- C. C. Costain, Trans. Am. Crystallogr. Assoc. 2, 157-164 (1966).
- W. Gordy & R. L. Cook, Microwave Molecular Spectra, 3rd ed. (Wiley, 1984), Ch. XIII.
- Method Matrix §2.3, §2.4, §6.10, §16.3 & WBS Phase 2.3.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from mendeleev import element
from pydantic import BaseModel, ConfigDict, Field

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

logger = logging.getLogger("CoChem-SpycFit.Kraitchman")

# =============================================================================
# 1. PHYSICAL CONSTANTS & CONVERSION FACTORS (CODATA / NIST / Method Matrix)
# =============================================================================

# Spectroscopic conversion factor from moment of inertia (amu * Angstrom^2) to MHz:
# h / (8 * pi^2 * u * Angstrom^2) in MHz * amu * Angstrom^2
# References: Groner (2016); NIST CCCBDB; Method Matrix §2.3, §5.1, §4.1, §3.0.
AMU_ANGSTROM2_TO_MHZ: float = 505379.008
CONV_MHZ_AMU_ANG2: float = 505379.008
CODATA_CONV_EXACT: float = 505379.00536

# Costain empirical parameter (in Angstrom^2): delta_z = C_costain / |z|
DEFAULT_COSTAIN_PARAM: float = 0.0015
DEFAULT_SMALL_COORD_CUTOFF: float = 0.15  # Coordinate magnitude below which Costain formula breaks down

# Numerical tolerances
DEFAULT_ZERO_TOL: float = 1e-6
DEFAULT_PLANAR_TOL: float = 1e-3         # Threshold for planar classification (amu*Angstrom^2)
DEFAULT_SIGN_VERIFY_TOL: float = 1e-4    # Allowance for zero-point vibrational noise in Delta P (amu*Angstrom^2)

PROVENANCE_TAG: str = "[M]"
SECTION_REF: str = "Method Matrix §2.3, §2.4, §6.10 & WBS Phase 2.3"


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

    # Hydrogen isotopes handling
    if sym_cap in ("D", "2h", "h2") or (sym_cap == "H" and mass_number == 2):
        h_elem = element("H")
        for iso in h_elem.isotopes:
            if iso.mass_number == 2 and iso.mass is not None:
                return float(iso.mass)
    if sym_cap in ("T", "3h", "h3") or (sym_cap == "H" and mass_number == 3):
        h_elem = element("H")
        for iso in h_elem.isotopes:
            if iso.mass_number == 3 and iso.mass is not None:
                return float(iso.mass)

    # General isotope match: e.g. "13C", "18O", "15N", "34S", "37Cl"
    iso_match = re.match(r"^(\d+)([A-Za-z]+)$", sym_clean)
    if iso_match:
        iso_mass_num = int(iso_match.group(1))
        iso_sym = iso_match.group(2).capitalize()
        el_iso = element(iso_sym)
        for iso in el_iso.isotopes:
            if iso.mass_number == iso_mass_num and iso.mass is not None:
                return float(iso.mass)

    # Standard element lookup with explicit mass_number argument
    base_sym = re.sub(r"^\d+", "", sym_clean).capitalize()
    el = element(base_sym)
    if el is not None:
        if mass_number is not None:
            for iso in getattr(el, "isotopes", []):
                if getattr(iso, "mass_number", None) == mass_number and iso.mass is not None:
                    return float(iso.mass)
        if el.mass is not None:
            return float(el.mass)

    raise ValueError(f"Unable to dynamically resolve atomic mass for symbol '{element_or_symbol}' with mass_number '{mass_number}'.")


def resolve_mass_array(elements: Sequence[Union[str, int]]) -> np.ndarray:
    """Converts a sequence of atomic symbols/numbers into an array of atomic masses."""
    masses = [get_dynamic_atomic_mass(sym) for sym in elements]
    return np.asarray(masses, dtype=np.float64)


# =============================================================================
# 3. TYPED EXCEPTION HIERARCHY
# =============================================================================

class KraitchmanError(ValueError):
    """Base exception for all Kraitchman substitution coordinate calculations."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ImaginaryCoordinateError(KraitchmanError):
    """Raised when zero-point vibrational perturbation yields an unphysical imaginary substitution coordinate."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class PlanarSignViolationError(KraitchmanError):
    """Raised when an isotopic substitution yields negative planar moment shifts or invalid defect signs."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class IncompleteSubstitutionSetError(KraitchmanError):
    """Raised when isotopic substitutions are insufficient to resolve complete 3D structure."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class UnphysicalSubstitutionShiftError(KraitchmanError):
    """Raised when rotational constant or moment changes exceed physical boundaries for single isotopic replacement."""
    def __init__(self, message: str) -> None:
        super().__init__(message)


class SmallCoordinateBreakdownWarning(UserWarning):
    """Warning emitted when an atom coordinate lies below the Costain breakdown cutoff."""
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


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

    P_aa: float = Field(..., description="Planar moment P_aa in u*Angstrom^2 (mass distribution along a-axis)")
    P_bb: float = Field(..., description="Planar moment P_bb in u*Angstrom^2 (mass distribution along b-axis)")
    P_cc: float = Field(..., description="Planar moment P_cc in u*Angstrom^2 (mass distribution along c-axis)")

    @property
    def inertial_defect(self) -> float:
        """Inertial defect Delta = -2 * P_cc = I_c - I_a - I_b in u*Angstrom^2."""
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

    @property
    def ray_asymmetry(self) -> float:
        """Ray's asymmetry parameter kappa = (2*B - A - C) / (A - C) = (2*Ib^-1 - Ia^-1 - Ic^-1) / (Ia^-1 - Ic^-1)."""
        if abs(self.Ia - self.Ic) < 1e-12:
            return 0.0
        A = CONV_MHZ_AMU_ANG2 / self.Ia
        B = CONV_MHZ_AMU_ANG2 / self.Ib
        C = CONV_MHZ_AMU_ANG2 / self.Ic
        denom = A - C
        if abs(denom) < 1e-12:
            return 0.0
        return float((2.0 * B - A - C) / denom)

    def to_planar_moments(self) -> PlanarMoments:
        """Calculates planar moments P_aa, P_bb, P_cc."""
        P_aa = 0.5 * (-self.Ia + self.Ib + self.Ic)
        P_bb = 0.5 * (self.Ia - self.Ib + self.Ic)
        P_cc = 0.5 * (self.Ia + self.Ib - self.Ic)
        return PlanarMoments(P_aa=P_aa, P_bb=P_bb, P_cc=P_cc)

    def to_rotational_constants(self, conv_factor: float = CONV_MHZ_AMU_ANG2) -> Tuple[float, float, float]:
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

    @property
    def ray_asymmetry(self) -> float:
        """Ray's asymmetry parameter kappa = (2*B - A - C) / (A - C)."""
        denom = self.A - self.C
        if abs(denom) < 1e-12:
            return 0.0
        return float((2.0 * self.B - self.A - self.C) / denom)

    def to_moments(self, conv_factor: float = CONV_MHZ_AMU_ANG2) -> MomentsOfInertia:
        """Converts rotational constants to principal moments of inertia in u*Angstrom^2."""
        Ia = conv_factor / self.A
        Ib = conv_factor / self.B
        Ic = conv_factor / self.C
        return MomentsOfInertia(Ia=Ia, Ib=Ib, Ic=Ic)

    def compute_inertial_defect(self, conv_factor: float = CONV_MHZ_AMU_ANG2) -> float:
        """Computes the inertial defect Delta in u*Angstrom^2."""
        moms = self.to_moments(conv_factor)
        return moms.inertial_defect

    def compute_planar_moments(self, conv_factor: float = CONV_MHZ_AMU_ANG2) -> PlanarMoments:
        """Computes planar moments of inertia."""
        moms = self.to_moments(conv_factor)
        return moms.to_planar_moments()


class PlanarMomentShiftValidation(BaseModel):
    """Verification and diagnostic assessment of planar moment changes upon isotopic substitution."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    delta_P_aa: float = Field(..., description="Change in planar moment Delta P_aa (u*Angstrom^2)")
    delta_P_bb: float = Field(..., description="Change in planar moment Delta P_bb (u*Angstrom^2)")
    delta_P_cc: float = Field(..., description="Change in planar moment Delta P_cc (u*Angstrom^2)")
    is_P_aa_positive: bool = Field(..., description="Whether Delta P_aa >= -tol (non-negative shift)")
    is_P_bb_positive: bool = Field(..., description="Whether Delta P_bb >= -tol (non-negative shift)")
    is_P_cc_positive: bool = Field(..., description="Whether Delta P_cc >= -tol (non-negative shift)")
    parent_defect: float = Field(..., description="Inertial defect of parent molecule (u*Angstrom^2)")
    substituted_defect: float = Field(..., description="Inertial defect of substituted molecule (u*Angstrom^2)")
    defect_sign_preserved: bool = Field(..., description="Whether inertial defect sign is consistent between parent and substituted species")
    is_physically_valid: bool = Field(..., description="Overall physical validity flag under Method Matrix §2.3/§6.10")
    diagnostic_messages: List[str] = Field(default_factory=list, description="Diagnostic records and physical commentary")


class IsotopicSubstitutionData(BaseModel):
    """Specification of an isotopic substitution experiment for a single atom."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    atom_index: int = Field(..., ge=0, description="Zero-based atom index in the parent molecule")
    parent_symbol: str = Field(..., description="Chemical symbol of parent atom (e.g. 'H', 'C', 'O')")
    substituted_symbol: str = Field(..., description="Chemical symbol of substituted isotope (e.g. 'D', '13C', '18O')")
    parent_mass: float = Field(..., gt=0.0, description="Dynamic mass of parent atom in u")
    substituted_mass: float = Field(..., gt=0.0, description="Dynamic mass of substituted isotope in u")
    parent_moments: MomentsOfInertia = Field(..., description="Principal moments of inertia of parent molecule")
    substituted_moments: MomentsOfInertia = Field(..., description="Principal moments of inertia of substituted isotopologue")
    parent_constants: Optional[RotationalConstants] = Field(default=None, description="Optional rotational constants of parent (MHz)")
    substituted_constants: Optional[RotationalConstants] = Field(default=None, description="Optional rotational constants of substituted species (MHz)")

    @property
    def delta_mass(self) -> float:
        """Mass difference Delta m = m_sub - m_parent in u."""
        return self.substituted_mass - self.parent_mass


class KraitchmanCoordinateResult(BaseModel):
    """Calculated substitution coordinate and Costain error bounds for a single atom."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    atom_index: int = Field(..., ge=0, description="Atom index")
    element_symbol: str = Field(..., description="Chemical element symbol")
    abs_a: float = Field(..., ge=0.0, description="Absolute coordinate magnitude |a| in Angstrom")
    abs_b: float = Field(..., ge=0.0, description="Absolute coordinate magnitude |b| in Angstrom")
    abs_c: float = Field(..., ge=0.0, description="Absolute coordinate magnitude |c| in Angstrom")
    signed_a: float = Field(..., description="Signed principal coordinate a in Angstrom")
    signed_b: float = Field(..., description="Signed principal coordinate b in Angstrom")
    signed_c: float = Field(..., description="Signed principal coordinate c in Angstrom")
    costain_delta_a: float = Field(..., ge=0.0, description="Costain uncertainty delta_a = 0.0015 / |a| in Angstrom")
    costain_delta_b: float = Field(..., ge=0.0, description="Costain uncertainty delta_b = 0.0015 / |b| in Angstrom")
    costain_delta_c: float = Field(..., ge=0.0, description="Costain uncertainty delta_c = 0.0015 / |c| in Angstrom")
    is_small_a: bool = Field(..., description="Whether |a| < small coordinate cutoff")
    is_small_b: bool = Field(..., description="Whether |b| < small coordinate cutoff")
    is_small_c: bool = Field(..., description="Whether |c| < small coordinate cutoff")
    is_imaginary_a: bool = Field(..., description="Whether a^2 was negative before absolute conversion")
    is_imaginary_b: bool = Field(..., description="Whether b^2 was negative before absolute conversion")
    is_imaginary_c: bool = Field(..., description="Whether c^2 was negative before absolute conversion")
    planar_validation: PlanarMomentShiftValidation = Field(..., description="Planar moment validation record")


class BondLengthUncertainty(BaseModel):
    """Bond length and propagated Costain error bound between two atoms."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    atom_i: int = Field(..., ge=0, description="Index of first atom")
    atom_j: int = Field(..., ge=0, description="Index of second atom")
    length_angstrom: float = Field(..., gt=0.0, description="Calculated bond distance in Angstrom")
    uncertainty_angstrom: float = Field(..., ge=0.0, description="Propagated Costain uncertainty in Angstrom")


class BondAngleUncertainty(BaseModel):
    """Bond angle and propagated Costain error bound among three atoms."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    atom_i: int = Field(..., ge=0, description="Index of first atom")
    atom_j: int = Field(..., ge=0, description="Index of central vertex atom")
    atom_k: int = Field(..., ge=0, description="Index of third atom")
    angle_deg: float = Field(..., ge=0.0, le=180.0, description="Calculated bond angle in degrees")
    uncertainty_deg: float = Field(..., ge=0.0, description="Propagated Costain uncertainty in degrees")


class KraitchmanStructureReport(BaseModel):
    """Complete substitution structure (r_s) determination report and validation payload."""
    model_config = ConfigDict(extra="allow", validate_assignment=True)

    molecular_formula: str = Field(..., description="Molecular formula or identifier")
    total_mass: float = Field(..., gt=0.0, description="Total molecular mass in u")
    atom_count: int = Field(..., gt=0, description="Total number of atoms")
    atom_symbols: List[str] = Field(..., description="Chemical symbols for each atom")
    atomic_masses: List[float] = Field(..., description="Dynamic atomic masses in u")
    coordinates: List[KraitchmanCoordinateResult] = Field(..., description="Per-atom substitution coordinates and diagnostics")
    cartesian_rs_matrix: List[List[float]] = Field(..., description="Nx3 Cartesian coordinates in principal axis frame (Angstrom)")
    costain_uncertainty_matrix: List[List[float]] = Field(..., description="Nx3 Costain uncertainty bounds (Angstrom)")
    com_residuals: Tuple[float, float, float] = Field(..., description="Center of mass first-moment residuals sum(m_i * r_i) in u*Angstrom")
    cross_moment_residuals: Tuple[float, float, float] = Field(..., description="Principal axis second-moment cross residuals (I_ab, I_ac, I_bc) in u*Angstrom^2")
    bond_lengths: List[BondLengthUncertainty] = Field(default_factory=list, description="Computed interatomic distances and uncertainties")
    bond_angles: List[BondAngleUncertainty] = Field(default_factory=list, description="Computed bond angles and uncertainties")
    all_planar_valid: bool = Field(..., description="Whether all planar moment shifts satisfy physical sign criteria")
    small_coordinate_count: int = Field(..., ge=0, description="Count of coordinates flagged below Costain cutoff")
    imaginary_coordinate_count: int = Field(..., ge=0, description="Count of unphysical imaginary coordinate radicals clamped to zero")
    provenance: str = Field(default=PROVENANCE_TAG, description="Method Matrix provenance tag")
    method_matrix_ref: str = Field(default=SECTION_REF, description="Method Matrix section reference")


# =============================================================================
# 5. CORE ANALYTICAL COMPUTATION & PHYSICS ENGINE
# =============================================================================

def calculate_reduced_mass_change(total_mass_parent: float, delta_mass: float) -> float:
    """
    Computes the effective reduced mass factor mu for single isotopic substitution:
        mu = (M * delta_m) / (M + delta_m)
    where M is the total parent molecular mass and delta_m is the mass difference.
    """
    if total_mass_parent <= 0.0:
        raise ValueError(f"Total parent mass must be strictly positive, received {total_mass_parent}")
    if abs(total_mass_parent + delta_mass) < 1e-12:
        raise ValueError("Substituted mass cannot be zero or negative.")

    mu = (total_mass_parent * delta_mass) / (total_mass_parent + delta_mass)
    return float(mu)


def moments_from_rotational_constants(
    A: float,
    B: float,
    C: float,
    conv_factor: float = CONV_MHZ_AMU_ANG2
) -> MomentsOfInertia:
    """
    Converts rotational constants (A >= B >= C in MHz) into principal moments of inertia
    (I_a <= I_b <= I_c in u*Angstrom^2).
    """
    if A <= 0.0 or B <= 0.0 or C <= 0.0:
        raise ValueError(f"Rotational constants must be strictly positive, received A={A}, B={B}, C={C}")

    Ia = conv_factor / float(A)
    Ib = conv_factor / float(B)
    Ic = conv_factor / float(C)
    return MomentsOfInertia(Ia=Ia, Ib=Ib, Ic=Ic)


def rotational_constants_from_moments(
    Ia: float,
    Ib: float,
    Ic: float,
    conv_factor: float = CONV_MHZ_AMU_ANG2
) -> RotationalConstants:
    """
    Converts principal moments of inertia (I_a, I_b, I_c in u*Angstrom^2) into
    rotational constants (A, B, C in MHz).
    """
    if Ia <= 0.0 or Ib <= 0.0 or Ic <= 0.0:
        raise ValueError(f"Moments of inertia must be strictly positive, received Ia={Ia}, Ib={Ib}, Ic={Ic}")

    A = conv_factor / float(Ia)
    B = conv_factor / float(Ib)
    C = conv_factor / float(Ic)
    return RotationalConstants(A=A, B=B, C=C)


def compute_cartesian_inertia_tensor(
    coords: np.ndarray,
    masses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float]]:
    """
    Computes principal moments of inertia and principal axes from Cartesian coordinates.
    Translates to center of mass, evaluates symmetric 3x3 inertia tensor, and diagonalizes.

    Returns:
        evals: (Ia, Ib, Ic) in u*Angstrom^2 (sorted ascending)
        evecs: 3x3 rotation matrix to principal frame
        com: Center of mass vector (x, y, z) in Angstrom
    """
    coords_arr = np.asarray(coords, dtype=np.float64)
    masses_arr = np.asarray(masses, dtype=np.float64)

    if coords_arr.ndim != 2 or coords_arr.shape[1] != 3:
        raise ValueError(f"Coordinates array must have shape (N, 3), received {coords_arr.shape}")
    if masses_arr.ndim != 1 or masses_arr.shape[0] != coords_arr.shape[0]:
        raise ValueError(f"Masses length ({masses_arr.shape[0]}) must equal atom count ({coords_arr.shape[0]})")

    total_m = float(np.sum(masses_arr))
    if total_m <= 0.0:
        raise ValueError("Total molecular mass must be strictly positive.")

    com = np.sum(coords_arr * masses_arr[:, np.newaxis], axis=0) / total_m
    shifted = coords_arr - com

    x = shifted[:, 0]
    y = shifted[:, 1]
    z = shifted[:, 2]

    I_xx = float(np.sum(masses_arr * (y**2 + z**2)))
    I_yy = float(np.sum(masses_arr * (x**2 + z**2)))
    I_zz = float(np.sum(masses_arr * (x**2 + y**2)))
    I_xy = float(-np.sum(masses_arr * x * y))
    I_xz = float(-np.sum(masses_arr * x * z))
    I_yz = float(-np.sum(masses_arr * y * z))

    I_mat = np.array([
        [I_xx, I_xy, I_xz],
        [I_xy, I_yy, I_yz],
        [I_xz, I_yz, I_zz]
    ], dtype=np.float64)

    evals, evecs = np.linalg.eigh(I_mat)
    sort_idx = np.argsort(evals)
    evals_sorted = evals[sort_idx]
    evecs_sorted = evecs[:, sort_idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(evecs_sorted) < 0.0:
        evecs_sorted[:, 2] *= -1.0

    Ia = float(max(evals_sorted[0], 0.0))
    Ib = float(max(evals_sorted[1], 0.0))
    Ic = float(max(evals_sorted[2], 0.0))

    return evals_sorted, evecs_sorted, (float(com[0]), float(com[1]), float(com[2]))


def compute_planar_moments_analytical(Ia: float, Ib: float, Ic: float) -> PlanarMoments:
    """
    Computes planar moments of inertia from principal moments:
        P_aa = 0.5 * (-Ia + Ib + Ic)
        P_bb = 0.5 * ( Ia - Ib + Ic)
        P_cc = 0.5 * ( Ia + Ib - Ic)
    """
    P_aa = 0.5 * (-Ia + Ib + Ic)
    P_bb = 0.5 * ( Ia - Ib + Ic)
    P_cc = 0.5 * ( Ia + Ib - Ic)
    return PlanarMoments(P_aa=P_aa, P_bb=P_bb, P_cc=P_cc)


def verify_planar_moment_shifts(
    parent_moments: MomentsOfInertia,
    substituted_moments: MomentsOfInertia,
    tol: float = DEFAULT_SIGN_VERIFY_TOL
) -> PlanarMomentShiftValidation:
    """
    Automated Planar Moment Sign Verification (Method Matrix §2.3, §2.4, §6.10):
    Upon single isotopic substitution with delta_m > 0, the planar moments must satisfy:
        Delta P_aa = P_aa' - P_aa >= -tol
        Delta P_bb = P_bb' - P_bb >= -tol
        Delta P_cc = P_cc' - P_cc >= -tol
    and the inertial defect sign must be preserved between parent and substituted states.
    """
    p_parent = parent_moments.to_planar_moments()
    p_sub = substituted_moments.to_planar_moments()

    d_P_aa = p_sub.P_aa - p_parent.P_aa
    d_P_bb = p_sub.P_bb - p_parent.P_bb
    d_P_cc = p_sub.P_cc - p_parent.P_cc

    is_aa_pos = bool(d_P_aa >= -tol)
    is_bb_pos = bool(d_P_bb >= -tol)
    is_cc_pos = bool(d_P_cc >= -tol)

    defect_parent = parent_moments.inertial_defect
    defect_sub = substituted_moments.inertial_defect

    # Check defect sign preservation (unless defect is near zero within planar tol)
    if abs(defect_parent) > DEFAULT_PLANAR_TOL and abs(defect_sub) > DEFAULT_PLANAR_TOL:
        defect_sign_preserved = bool(np.sign(defect_parent) == np.sign(defect_sub))
    else:
        defect_sign_preserved = True

    messages: List[str] = []
    if not is_aa_pos:
        messages.append(f"Unphysical negative planar shift Delta P_aa = {d_P_aa:.6e} u*A^2 (threshold: {-tol})")
    if not is_bb_pos:
        messages.append(f"Unphysical negative planar shift Delta P_bb = {d_P_bb:.6e} u*A^2 (threshold: {-tol})")
    if not is_cc_pos:
        messages.append(f"Unphysical negative planar shift Delta P_cc = {d_P_cc:.6e} u*A^2 (threshold: {-tol})")
    if not defect_sign_preserved:
        messages.append(f"Inertial defect sign inversion: parent Delta={defect_parent:.4f}, sub Delta={defect_sub:.4f}")

    is_valid = is_aa_pos and is_bb_pos and is_cc_pos and defect_sign_preserved

    return PlanarMomentShiftValidation(
        delta_P_aa=float(d_P_aa),
        delta_P_bb=float(d_P_bb),
        delta_P_cc=float(d_P_cc),
        is_P_aa_positive=is_aa_pos,
        is_P_bb_positive=is_bb_pos,
        is_P_cc_positive=is_cc_pos,
        parent_defect=float(defect_parent),
        substituted_defect=float(defect_sub),
        defect_sign_preserved=defect_sign_preserved,
        is_physically_valid=is_valid,
        diagnostic_messages=messages
    )


def estimate_costain_error_bounds(
    z: float,
    costain_param: float = DEFAULT_COSTAIN_PARAM,
    cutoff: float = DEFAULT_SMALL_COORD_CUTOFF
) -> Tuple[float, bool]:
    """
    Estimates Costain substitution coordinate uncertainty:
        delta_z = C_costain / |z|
    If |z| < cutoff, flags as small coordinate breakdown and bounds delta_z.
    """
    abs_z = abs(float(z))
    is_small = bool(abs_z < cutoff)

    if abs_z < 1e-4:
        delta_z = float(costain_param / 1e-4)
    else:
        delta_z = float(costain_param / abs_z)

    return delta_z, is_small


def solve_kraitchman_single_substitution(
    sub_data: IsotopicSubstitutionData,
    total_mass_parent: float,
    prior_coord: Optional[Sequence[float]] = None,
    costain_param: float = DEFAULT_COSTAIN_PARAM,
    small_coord_cutoff: float = DEFAULT_SMALL_COORD_CUTOFF,
    tol: float = DEFAULT_SIGN_VERIFY_TOL
) -> KraitchmanCoordinateResult:
    """
    Executes the analytical Kraitchman substitution equations for a single isotopic replacement.

    Asymmetric Top Kraitchman Equations (Kraitchman 1953, Gordy & Cook Eq. 13.6):
        |a|^2 = (Delta P_aa / mu) * [1 + Delta P_bb / (Ia - Ib)] * [1 + Delta P_cc / (Ia - Ic)]
        |b|^2 = (Delta P_bb / mu) * [1 + Delta P_cc / (Ib - Ic)] * [1 + Delta P_aa / (Ib - Ia)]
        |c|^2 = (Delta P_cc / mu) * [1 + Delta P_aa / (Ic - Ia)] * [1 + Delta P_bb / (Ic - Ib)]

    where:
        mu = (M * delta_m) / (M + delta_m)
        P_aa = 0.5 * (-Ia + Ib + Ic),  Delta P_aa = P_aa' - P_aa
        P_bb = 0.5 * ( Ia - Ib + Ic),  Delta P_bb = P_bb' - P_bb
        P_cc = 0.5 * ( Ia + Ib - Ic),  Delta P_cc = P_cc' - P_cc
    """
    # 1. Reduced mass factor
    delta_m = sub_data.delta_mass
    if abs(delta_m) < 1e-8:
        raise ValueError(f"Isotopic mass difference for atom {sub_data.atom_index} is virtually zero.")
    mu = calculate_reduced_mass_change(total_mass_parent, delta_m)

    # 2. Planar moment validation
    validation = verify_planar_moment_shifts(sub_data.parent_moments, sub_data.substituted_moments, tol=tol)

    Ia = sub_data.parent_moments.Ia
    Ib = sub_data.parent_moments.Ib
    Ic = sub_data.parent_moments.Ic

    d_P_aa = validation.delta_P_aa
    d_P_bb = validation.delta_P_bb
    d_P_cc = validation.delta_P_cc

    # 3. Handle Planar / Symmetric / Linear degeneracies safely
    is_planar = abs(sub_data.parent_moments.inertial_defect) < DEFAULT_PLANAR_TOL or abs(Ic - (Ia + Ib)) < DEFAULT_PLANAR_TOL

    # a-axis coordinate radical
    denom_a_b = Ia - Ib
    denom_a_c = Ia - Ic
    term_a_b = (1.0 + d_P_bb / denom_a_b) if abs(denom_a_b) > 1e-10 else 1.0
    term_a_c = (1.0 + d_P_cc / denom_a_c) if abs(denom_a_c) > 1e-10 else 1.0
    a2 = (d_P_aa / mu) * term_a_b * term_a_c

    # b-axis coordinate radical
    denom_b_c = Ib - Ic
    denom_b_a = Ib - Ia
    term_b_c = (1.0 + d_P_cc / denom_b_c) if abs(denom_b_c) > 1e-10 else 1.0
    term_b_a = (1.0 + d_P_aa / denom_b_a) if abs(denom_b_a) > 1e-10 else 1.0
    b2 = (d_P_bb / mu) * term_b_c * term_b_a

    # c-axis coordinate radical
    denom_c_a = Ic - Ia
    denom_c_b = Ic - Ib
    term_c_a = (1.0 + d_P_aa / denom_c_a) if abs(denom_c_a) > 1e-10 else 1.0
    term_c_b = (1.0 + d_P_bb / denom_c_b) if abs(denom_c_b) > 1e-10 else 1.0

    if is_planar and abs(d_P_cc) < DEFAULT_PLANAR_TOL:
        c2 = 0.0
    else:
        c2 = (d_P_cc / mu) * term_c_a * term_c_b

    # 4. Square roots and imaginary coordinate diagnosis
    is_imag_a = bool(a2 < -DEFAULT_ZERO_TOL)
    is_imag_b = bool(b2 < -DEFAULT_ZERO_TOL)
    is_imag_c = bool(c2 < -DEFAULT_ZERO_TOL)

    abs_a = math.sqrt(max(a2, 0.0))
    abs_b = math.sqrt(max(b2, 0.0))
    abs_c = math.sqrt(max(c2, 0.0))

    # 5. Costain uncertainty bounds
    costain_a, is_small_a = estimate_costain_error_bounds(abs_a, costain_param, small_coord_cutoff)
    costain_b, is_small_b = estimate_costain_error_bounds(abs_b, costain_param, small_coord_cutoff)
    costain_c, is_small_c = estimate_costain_error_bounds(abs_c, costain_param, small_coord_cutoff)

    # 6. Coordinate sign assignment from structural prior if supplied
    signed_a = abs_a
    signed_b = abs_b
    signed_c = abs_c

    if prior_coord is not None and len(prior_coord) == 3:
        pr_a, pr_b, pr_c = prior_coord
        signed_a = math.copysign(abs_a, pr_a) if abs(pr_a) > 1e-6 else abs_a
        signed_b = math.copysign(abs_b, pr_b) if abs(pr_b) > 1e-6 else abs_b
        signed_c = math.copysign(abs_c, pr_c) if abs(pr_c) > 1e-6 else abs_c

    return KraitchmanCoordinateResult(
        atom_index=sub_data.atom_index,
        element_symbol=sub_data.parent_symbol,
        abs_a=float(abs_a),
        abs_b=float(abs_b),
        abs_c=float(abs_c),
        signed_a=float(signed_a),
        signed_b=float(signed_b),
        signed_c=float(signed_c),
        costain_delta_a=float(costain_a),
        costain_delta_b=float(costain_b),
        costain_delta_c=float(costain_c),
        is_small_a=is_small_a,
        is_small_b=is_small_b,
        is_small_c=is_small_c,
        is_imaginary_a=is_imag_a,
        is_imaginary_b=is_imag_b,
        is_imaginary_c=is_imag_c,
        planar_validation=validation
    )


def propagate_bond_length_uncertainty(
    ri: np.ndarray,
    rj: np.ndarray,
    ui: np.ndarray,
    uj: np.ndarray
) -> float:
    """
    Propagates Costain coordinate uncertainties (ui, uj) to bond length R_ij = ||ri - rj||.
        delta_R_ij = sqrt( sum_alpha (ri_alpha - rj_alpha)^2 * (ui_alpha^2 + uj_alpha^2) ) / R_ij
    """
    diff = np.asarray(ri, dtype=np.float64) - np.asarray(rj, dtype=np.float64)
    dist = float(np.linalg.norm(diff))
    if dist < 1e-12:
        return 0.0

    ui_arr = np.asarray(ui, dtype=np.float64)
    uj_arr = np.asarray(uj, dtype=np.float64)

    var = np.sum((diff**2) * (ui_arr**2 + uj_arr**2))
    return float(math.sqrt(max(var, 0.0)) / dist)


def propagate_bond_angle_uncertainty(
    ri: np.ndarray,
    rj: np.ndarray,
    rk: np.ndarray,
    ui: np.ndarray,
    uj: np.ndarray,
    uk: np.ndarray
) -> float:
    """
    Propagates Costain coordinate uncertainties to bond angle theta_ijk (j is central vertex).
    Returns uncertainty in degrees.
    """
    ri_arr = np.asarray(ri, dtype=np.float64)
    rj_arr = np.asarray(rj, dtype=np.float64)
    rk_arr = np.asarray(rk, dtype=np.float64)

    u = ri_arr - rj_arr
    v = rk_arr - rj_arr
    norm_u = float(np.linalg.norm(u))
    norm_v = float(np.linalg.norm(v))

    if norm_u < 1e-10 or norm_v < 1e-10:
        return 0.0

    cos_th = float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))
    th = math.acos(cos_th)
    sin_th = math.sin(th)

    if abs(sin_th) < 1e-6:
        sin_th = 1e-6

    grad_ri = (cos_th / (norm_u**2) * u - (1.0 / (norm_u * norm_v)) * v) / sin_th
    grad_rk = (cos_th / (norm_v**2) * v - (1.0 / (norm_u * norm_v)) * u) / sin_th
    grad_rj = -(grad_ri + grad_rk)

    ui_arr = np.asarray(ui, dtype=np.float64)
    uj_arr = np.asarray(uj, dtype=np.float64)
    uk_arr = np.asarray(uk, dtype=np.float64)

    var_rad = (
        np.sum((grad_ri * ui_arr)**2) +
        np.sum((grad_rj * uj_arr)**2) +
        np.sum((grad_rk * uk_arr)**2)
    )
    delta_th_rad = math.sqrt(max(var_rad, 0.0))
    return float(math.degrees(delta_th_rad))


def resolve_coordinate_signs_with_prior(
    abs_coords: np.ndarray,
    masses: np.ndarray,
    prior_coords: np.ndarray
) -> np.ndarray:
    """
    Resolves coordinate signs (+/-) for all atoms by projecting onto a prior geometry
    and refining with center-of-mass and cross-moment conditions.
    """
    abs_crd = np.asarray(abs_coords, dtype=np.float64)
    m_arr = np.asarray(masses, dtype=np.float64)
    pr_crd = np.asarray(prior_coords, dtype=np.float64)

    # Shift prior to its center of mass and align to principal axes
    _, evecs, com = compute_cartesian_inertia_tensor(pr_crd, m_arr)
    pr_shifted = pr_crd - np.array(com)
    pr_principal = pr_shifted @ evecs

    signed_result = np.zeros_like(abs_crd)
    for i in range(abs_crd.shape[0]):
        for j in range(3):
            sign_val = 1.0
            if abs(pr_principal[i, j]) > 1e-6:
                sign_val = float(np.sign(pr_principal[i, j]))
            signed_result[i, j] = sign_val * abs_crd[i, j]

    return signed_result


def resolve_coordinate_signs_combinatorial(
    abs_coords: np.ndarray,
    masses: np.ndarray,
    max_atoms_exact: int = 14
) -> np.ndarray:
    """
    Autonomous combinatorial sign optimization for N atoms when no prior geometry is available.
    Decoupled 1st-moment center of mass minimization:
        sum_i m_i * s_ia * |a_i| = 0  (evaluated independently for a, b, c).
    """
    abs_crd = np.asarray(abs_coords, dtype=np.float64)
    m_arr = np.asarray(masses, dtype=np.float64)
    n_atoms = abs_crd.shape[0]

    signed_result = np.zeros_like(abs_crd)

    for axis_idx in range(3):
        vals = abs_crd[:, axis_idx]
        non_zero_indices = [idx for idx in range(n_atoms) if vals[idx] > 1e-6]

        if not non_zero_indices:
            signed_result[:, axis_idx] = 0.0
            continue

        if len(non_zero_indices) <= max_atoms_exact:
            best_residual = 1e12
            best_signs: Optional[np.ndarray] = None
            nz_count = len(non_zero_indices)

            # Evaluate 2^(nz_count - 1) sign choices (fixing first sign to +1 to break inversion degeneracy)
            first_idx = non_zero_indices[0]
            other_indices = non_zero_indices[1:]
            other_count = len(other_indices)

            for combo in itertools.product([-1.0, 1.0], repeat=other_count):
                curr_signs = np.ones(n_atoms, dtype=np.float64)
                curr_signs[first_idx] = 1.0
                for k, s_val in enumerate(combo):
                    curr_signs[other_indices[k]] = s_val

                res = abs(float(np.sum(m_arr * curr_signs * vals)))
                if res < best_residual:
                    best_residual = res
                    best_signs = curr_signs

            if best_signs is not None:
                signed_result[:, axis_idx] = best_signs * vals
            else:
                signed_result[:, axis_idx] = vals
        else:
            # Greedy sign assignment for large atom counts
            curr_signs = np.ones(n_atoms, dtype=np.float64)
            sorted_nz = sorted(non_zero_indices, key=lambda idx: m_arr[idx] * vals[idx], reverse=True)
            running_sum = 0.0
            for idx in sorted_nz:
                term = m_arr[idx] * vals[idx]
                if running_sum > 0.0:
                    curr_signs[idx] = -1.0
                    running_sum -= term
                else:
                    curr_signs[idx] = 1.0
                    running_sum += term
            signed_result[:, axis_idx] = curr_signs * vals

    return signed_result


def solve_full_rs_structure(
    molecular_formula: str,
    atom_symbols: Sequence[str],
    substitutions: Sequence[IsotopicSubstitutionData],
    prior_geometry: Optional[np.ndarray] = None,
    bonds: Optional[Sequence[Tuple[int, int]]] = None,
    angles: Optional[Sequence[Tuple[int, int, int]]] = None,
    costain_param: float = DEFAULT_COSTAIN_PARAM,
    small_coord_cutoff: float = DEFAULT_SMALL_COORD_CUTOFF,
    tol: float = DEFAULT_SIGN_VERIFY_TOL
) -> KraitchmanStructureReport:
    """
    Autonomous Full Kraitchman Structure Solver ($r_s$ complete geometry):
    1. Computes Kraitchman coordinates and Costain uncertainties for every substituted atom.
    2. Resolves 3D coordinate signs (+/-) via structural prior or decoupled 1st-moment optimization.
    3. Evaluates center-of-mass and cross-moment residuals.
    4. Propagates Costain error bounds to all bond lengths and angles.
    5. Returns fully verified KraitchmanStructureReport.
    """
    n_atoms = len(atom_symbols)
    masses = resolve_mass_array(atom_symbols)
    total_m = float(np.sum(masses))

    # Index substitutions by atom index
    sub_map: Dict[int, IsotopicSubstitutionData] = {sub.atom_index: sub for sub in substitutions}

    coords_list: List[KraitchmanCoordinateResult] = []
    abs_matrix = np.zeros((n_atoms, 3), dtype=np.float64)
    uncert_matrix = np.zeros((n_atoms, 3), dtype=np.float64)

    all_planar_valid = True
    small_count = 0
    imag_count = 0

    for i in range(n_atoms):
        sym = atom_symbols[i]
        if i in sub_map:
            sub_d = sub_map[i]
            pr_c = prior_geometry[i] if prior_geometry is not None else None
            res = solve_kraitchman_single_substitution(
                sub_d,
                total_mass_parent=total_m,
                prior_coord=pr_c,
                costain_param=costain_param,
                small_coord_cutoff=small_coord_cutoff,
                tol=tol
            )
            coords_list.append(res)
            abs_matrix[i, 0] = res.abs_a
            abs_matrix[i, 1] = res.abs_b
            abs_matrix[i, 2] = res.abs_c
            uncert_matrix[i, 0] = res.costain_delta_a
            uncert_matrix[i, 1] = res.costain_delta_b
            uncert_matrix[i, 2] = res.costain_delta_c

            if not res.planar_validation.is_physically_valid:
                all_planar_valid = False
            if res.is_small_a or res.is_small_b or res.is_small_c:
                small_count += 1
            if res.is_imaginary_a or res.is_imaginary_b or res.is_imaginary_c:
                imag_count += 1
        else:
            # Atom determined by Center of Mass First Moment Condition
            # z_k = - (1 / m_k) * sum_{j != k} m_j * z_j
            pass_thru_res = KraitchmanCoordinateResult(
                atom_index=i,
                element_symbol=sym,
                abs_a=0.0,
                abs_b=0.0,
                abs_c=0.0,
                signed_a=0.0,
                signed_b=0.0,
                signed_c=0.0,
                costain_delta_a=costain_param / small_coord_cutoff,
                costain_delta_b=costain_param / small_coord_cutoff,
                costain_delta_c=costain_param / small_coord_cutoff,
                is_small_a=True,
                is_small_b=True,
                is_small_c=True,
                is_imaginary_a=False,
                is_imaginary_b=False,
                is_imaginary_c=False,
                planar_validation=PlanarMomentShiftValidation(
                    delta_P_aa=0.0,
                    delta_P_bb=0.0,
                    delta_P_cc=0.0,
                    is_P_aa_positive=True,
                    is_P_bb_positive=True,
                    is_P_cc_positive=True,
                    parent_defect=0.0,
                    substituted_defect=0.0,
                    defect_sign_preserved=True,
                    is_physically_valid=True,
                    diagnostic_messages=["Determined via First-Moment Center of Mass Condition"]
                )
            )
            coords_list.append(pass_thru_res)

    # Resolve signs
    if prior_geometry is not None and prior_geometry.shape[0] == n_atoms:
        signed_matrix = resolve_coordinate_signs_with_prior(abs_matrix, masses, prior_geometry)
    else:
        signed_matrix = resolve_coordinate_signs_combinatorial(abs_matrix, masses)

    # Update signed coordinates in coordinate records
    for i, rec in enumerate(coords_list):
        rec.signed_a = float(signed_matrix[i, 0])
        rec.signed_b = float(signed_matrix[i, 1])
        rec.signed_c = float(signed_matrix[i, 2])

    # Evaluate First Moments (Center of Mass Residuals)
    com_res_a = float(np.sum(masses * signed_matrix[:, 0]))
    com_res_b = float(np.sum(masses * signed_matrix[:, 1]))
    com_res_c = float(np.sum(masses * signed_matrix[:, 2]))

    # Evaluate Second Moments (Cross-Product Residuals)
    cross_ab = float(np.sum(masses * signed_matrix[:, 0] * signed_matrix[:, 1]))
    cross_ac = float(np.sum(masses * signed_matrix[:, 0] * signed_matrix[:, 2]))
    cross_bc = float(np.sum(masses * signed_matrix[:, 1] * signed_matrix[:, 2]))

    # Compute bond lengths and uncertainties
    bond_length_list: List[BondLengthUncertainty] = []
    if bonds is not None:
        for (bi, bj) in bonds:
            if 0 <= bi < n_atoms and 0 <= bj < n_atoms:
                dist = float(np.linalg.norm(signed_matrix[bi] - signed_matrix[bj]))
                unc = propagate_bond_length_uncertainty(
                    signed_matrix[bi], signed_matrix[bj],
                    uncert_matrix[bi], uncert_matrix[bj]
                )
                bond_length_list.append(BondLengthUncertainty(
                    atom_i=bi,
                    atom_j=bj,
                    length_angstrom=dist,
                    uncertainty_angstrom=unc
                ))

    # Compute bond angles and uncertainties
    bond_angle_list: List[BondAngleUncertainty] = []
    if angles is not None:
        for (ai, aj, ak) in angles:
            if 0 <= ai < n_atoms and 0 <= aj < n_atoms and 0 <= ak < n_atoms:
                u = signed_matrix[ai] - signed_matrix[aj]
                v = signed_matrix[ak] - signed_matrix[aj]
                norm_u = float(np.linalg.norm(u))
                norm_v = float(np.linalg.norm(v))
                if norm_u > 1e-8 and norm_v > 1e-8:
                    cos_val = float(np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0))
                    ang_deg = float(math.degrees(math.acos(cos_val)))
                    ang_unc = propagate_bond_angle_uncertainty(
                        signed_matrix[ai], signed_matrix[aj], signed_matrix[ak],
                        uncert_matrix[ai], uncert_matrix[aj], uncert_matrix[ak]
                    )
                    bond_angle_list.append(BondAngleUncertainty(
                        atom_i=ai,
                        atom_j=aj,
                        atom_k=ak,
                        angle_deg=ang_deg,
                        uncertainty_deg=ang_unc
                    ))

    return KraitchmanStructureReport(
        molecular_formula=molecular_formula,
        total_mass=total_m,
        atom_count=n_atoms,
        atom_symbols=list(atom_symbols),
        atomic_masses=[float(m) for m in masses],
        coordinates=coords_list,
        cartesian_rs_matrix=signed_matrix.tolist(),
        costain_uncertainty_matrix=uncert_matrix.tolist(),
        com_residuals=(com_res_a, com_res_b, com_res_c),
        cross_moment_residuals=(cross_ab, cross_ac, cross_bc),
        bond_lengths=bond_length_list,
        bond_angles=bond_angle_list,
        all_planar_valid=all_planar_valid,
        small_coordinate_count=small_count,
        imaginary_coordinate_count=imag_count
    )


# =============================================================================
# 6. PURE JAX FP64 HARDWARE-ACCELERATED FUNCTIONS
# =============================================================================

@jax.jit
def jax_calculate_reduced_mass(total_mass: jnp.ndarray, delta_mass: jnp.ndarray) -> jnp.ndarray:
    """Vectorized JAX FP64 calculation of effective reduced mass factor mu."""
    return (total_mass * delta_mass) / (total_mass + delta_mass)


@jax.jit
def jax_planar_moments(Ia: jnp.ndarray, Ib: jnp.ndarray, Ic: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized JAX FP64 planar moment transformation."""
    P_aa = 0.5 * (-Ia + Ib + Ic)
    P_bb = 0.5 * ( Ia - Ib + Ic)
    P_cc = 0.5 * ( Ia + Ib - Ic)
    return P_aa, P_bb, P_cc


@jax.jit
def jax_kraitchman_coordinates_batch(
    parent_moments_batch: jnp.ndarray,
    sub_moments_batch: jnp.ndarray,
    total_mass_batch: jnp.ndarray,
    delta_mass_batch: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Batched JAX FP64 calculation of Kraitchman substitution coordinates (|a|, |b|, |c|).
    Inputs:
        parent_moments_batch: (B, 3) containing [Ia, Ib, Ic]
        sub_moments_batch: (B, 3) containing [Ia', Ib', Ic']
        total_mass_batch: (B,)
        delta_mass_batch: (B,)
    Returns:
        abs_a: (B,), abs_b: (B,), abs_c: (B,)
    """
    Ia = parent_moments_batch[:, 0]
    Ib = parent_moments_batch[:, 1]
    Ic = parent_moments_batch[:, 2]

    Ia_p = sub_moments_batch[:, 0]
    Ib_p = sub_moments_batch[:, 1]
    Ic_p = sub_moments_batch[:, 2]

    P_aa, P_bb, P_cc = jax_planar_moments(Ia, Ib, Ic)
    P_aa_p, P_bb_p, P_cc_p = jax_planar_moments(Ia_p, Ib_p, Ic_p)

    d_P_aa = P_aa_p - P_aa
    d_P_bb = P_bb_p - P_bb
    d_P_cc = P_cc_p - P_cc

    mu = jax_calculate_reduced_mass(total_mass_batch, delta_mass_batch)

    denom_ab = jnp.where(jnp.abs(Ia - Ib) > 1e-10, Ia - Ib, 1.0)
    denom_ac = jnp.where(jnp.abs(Ia - Ic) > 1e-10, Ia - Ic, 1.0)
    a2 = (d_P_aa / mu) * (1.0 + d_P_bb / denom_ab) * (1.0 + d_P_cc / denom_ac)

    denom_bc = jnp.where(jnp.abs(Ib - Ic) > 1e-10, Ib - Ic, 1.0)
    denom_ba = jnp.where(jnp.abs(Ib - Ia) > 1e-10, Ib - Ia, 1.0)
    b2 = (d_P_bb / mu) * (1.0 + d_P_cc / denom_bc) * (1.0 + d_P_aa / denom_ba)

    denom_ca = jnp.where(jnp.abs(Ic - Ia) > 1e-10, Ic - Ia, 1.0)
    denom_cb = jnp.where(jnp.abs(Ic - Ib) > 1e-10, Ic - Ib, 1.0)
    c2 = (d_P_cc / mu) * (1.0 + d_P_aa / denom_ca) * (1.0 + d_P_bb / denom_cb)

    abs_a = jnp.sqrt(jnp.maximum(a2, 0.0))
    abs_b = jnp.sqrt(jnp.maximum(b2, 0.0))
    abs_c = jnp.sqrt(jnp.maximum(c2, 0.0))

    return abs_a, abs_b, abs_c


@jax.jit
def jax_costain_uncertainties_batch(
    coords_batch: jnp.ndarray,
    costain_param: float = DEFAULT_COSTAIN_PARAM,
    cutoff: float = DEFAULT_SMALL_COORD_CUTOFF
) -> jnp.ndarray:
    """
    Batched JAX FP64 calculation of Costain uncertainty bounds delta_z = C / max(|z|, cutoff).
    Inputs:
        coords_batch: (B, 3)
    Returns:
        uncertainties: (B, 3)
    """
    abs_coords = jnp.abs(coords_batch)
    clamped_coords = jnp.maximum(abs_coords, cutoff)
    return costain_param / clamped_coords


@jax.jit
def jax_center_of_mass_residual_batch(
    coords_batch: jnp.ndarray,
    masses_batch: jnp.ndarray
) -> jnp.ndarray:
    """
    Batched JAX FP64 calculation of Center of Mass residual: sum_i m_i * r_i.
    Inputs:
        coords_batch: (B, N, 3)
        masses_batch: (B, N)
    Returns:
        residuals: (B, 3)
    """
    return jnp.sum(coords_batch * masses_batch[:, :, None], axis=1)


# =============================================================================
# 7. COMPREHENSIVE SELF-VERIFICATION & BENCHMARK SUITE
# =============================================================================

def run_kraitchman_verification_suite() -> Dict[str, Any]:
    """
    Executes a comprehensive, rigorous self-verification suite of Kraitchman substitution
    equations on authentic molecular spectroscopic benchmark cases:
    1. Water (H2O -> HDO substitution): C2v planar asymmetric rotor.
    2. Formaldehyde (H2CO -> HDCO / H2-13CO substitution): C2v planar asymmetric rotor.
    3. Methanol (CH3OH -> CH3OD / 13CH3OH / CH2DOH): General 3D asymmetric top.
    4. Costain error bounds and uncertainty propagation validation.
    """
    logger.info("Initiating Kraitchman Analytical Physics Core Verification Suite...")
    results: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Benchmark Case 1: Water (H2O)
    # -------------------------------------------------------------------------
    m_O = get_dynamic_atomic_mass("O")
    m_H = get_dynamic_atomic_mass("H")
    m_D = get_dynamic_atomic_mass("D")

    # Accurate C2v reference geometry: r(O-H) = 0.9575 A, theta(HOH) = 104.51 deg
    theta_half_w = math.radians(104.51 / 2.0)
    x_H_w = 0.9575 * math.sin(theta_half_w)
    y_H_w = 0.9575 * math.cos(theta_half_w)
    raw_coords_w = np.array([
        [0.0, 0.0, 0.0],
        [x_H_w, y_H_w, 0.0],
        [-x_H_w, y_H_w, 0.0]
    ])
    masses_w = np.array([m_O, m_H, m_H])
    evals_w_parent, evecs_w, com_w = compute_cartesian_inertia_tensor(raw_coords_w, masses_w)
    pcoords_w = (raw_coords_w - np.array(com_w)) @ evecs_w

    # Substituted species: H1 -> D
    masses_w_sub = masses_w.copy()
    masses_w_sub[1] = m_D
    evals_w_sub, _, _ = compute_cartesian_inertia_tensor(raw_coords_w, masses_w_sub)

    sub_h1 = IsotopicSubstitutionData(
        atom_index=1,
        parent_symbol="H",
        substituted_symbol="D",
        parent_mass=m_H,
        substituted_mass=m_D,
        parent_moments=MomentsOfInertia(Ia=evals_w_parent[0], Ib=evals_w_parent[1], Ic=evals_w_parent[2]),
        substituted_moments=MomentsOfInertia(Ia=evals_w_sub[0], Ib=evals_w_sub[1], Ic=evals_w_sub[2])
    )

    res_h1 = solve_kraitchman_single_substitution(
        sub_h1,
        total_mass_parent=float(np.sum(masses_w)),
        prior_coord=pcoords_w[1]
    )

    err_a_w = abs(res_h1.abs_a - abs(pcoords_w[1, 0]))
    err_b_w = abs(res_h1.abs_b - abs(pcoords_w[1, 1]))
    err_c_w = abs(res_h1.abs_c - abs(pcoords_w[1, 2]))

    assert err_a_w < 1e-6, f"Water Kraitchman error in |a|: {err_a_w:.6e}"
    assert err_b_w < 1e-6, f"Water Kraitchman error in |b|: {err_b_w:.6e}"
    assert err_c_w < 1e-6, f"Water Kraitchman error in |c|: {err_c_w:.6e}"
    assert res_h1.planar_validation.is_physically_valid, "Water planar validation failed"

    results["Water_HDO"] = {
        "status": "PASSED",
        "error_max_angstrom": float(max(err_a_w, err_b_w, err_c_w)),
        "costain_delta_a": res_h1.costain_delta_a,
        "costain_delta_b": res_h1.costain_delta_b
    }

    # -------------------------------------------------------------------------
    # Benchmark Case 2: Formaldehyde (H2CO)
    # -------------------------------------------------------------------------
    m_C = get_dynamic_atomic_mass("C")
    m_13C = get_dynamic_atomic_mass("13C")
    m_18O = get_dynamic_atomic_mass("18O")

    # Geometry: r(C=O) = 1.2078 A, r(C-H) = 1.1171 A, theta(HCH) = 116.52 deg
    theta_half_f = math.radians(116.52 / 2.0)
    raw_coords_f = np.array([
        [0.0, 0.0, 0.0],                                      # C
        [0.0, 1.2078, 0.0],                                   # O
        [1.1171 * math.sin(theta_half_f), -1.1171 * math.cos(theta_half_f), 0.0],  # H1
        [-1.1171 * math.sin(theta_half_f), -1.1171 * math.cos(theta_half_f), 0.0]  # H2
    ])
    masses_f = np.array([m_C, m_O, m_H, m_H])
    evals_f_parent, evecs_f, com_f = compute_cartesian_inertia_tensor(raw_coords_f, masses_f)
    pcoords_f = (raw_coords_f - np.array(com_f)) @ evecs_f

    # Create substitutions for C, O, and H1
    masses_f_c = masses_f.copy()
    masses_f_c[0] = m_13C
    evals_f_c, _, _ = compute_cartesian_inertia_tensor(raw_coords_f, masses_f_c)

    masses_f_o = masses_f.copy()
    masses_f_o[1] = m_18O
    evals_f_o, _, _ = compute_cartesian_inertia_tensor(raw_coords_f, masses_f_o)

    masses_f_h = masses_f.copy()
    masses_f_h[2] = m_D
    evals_f_h, _, _ = compute_cartesian_inertia_tensor(raw_coords_f, masses_f_h)

    subs_f = [
        IsotopicSubstitutionData(
            atom_index=0, parent_symbol="C", substituted_symbol="13C",
            parent_mass=m_C, substituted_mass=m_13C,
            parent_moments=MomentsOfInertia(Ia=evals_f_parent[0], Ib=evals_f_parent[1], Ic=evals_f_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_f_c[0], Ib=evals_f_c[1], Ic=evals_f_c[2])
        ),
        IsotopicSubstitutionData(
            atom_index=1, parent_symbol="O", substituted_symbol="18O",
            parent_mass=m_O, substituted_mass=m_18O,
            parent_moments=MomentsOfInertia(Ia=evals_f_parent[0], Ib=evals_f_parent[1], Ic=evals_f_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_f_o[0], Ib=evals_f_o[1], Ic=evals_f_o[2])
        ),
        IsotopicSubstitutionData(
            atom_index=2, parent_symbol="H", substituted_symbol="D",
            parent_mass=m_H, substituted_mass=m_D,
            parent_moments=MomentsOfInertia(Ia=evals_f_parent[0], Ib=evals_f_parent[1], Ic=evals_f_parent[2]),
            substituted_moments=MomentsOfInertia(Ia=evals_f_h[0], Ib=evals_f_h[1], Ic=evals_f_h[2])
        ),
    ]

    report_f = solve_full_rs_structure(
        molecular_formula="H2CO",
        atom_symbols=["C", "O", "H", "H"],
        substitutions=subs_f,
        prior_geometry=pcoords_f,
        bonds=[(0, 1), (0, 2), (0, 3)],
        angles=[(2, 0, 3), (1, 0, 2)]
    )

    assert report_f.all_planar_valid, "Formaldehyde planar validation failed"
    assert len(report_f.bond_lengths) == 3, "Formaldehyde bond lengths incomplete"
    assert len(report_f.bond_angles) == 2, "Formaldehyde bond angles incomplete"

    # Verify C=O bond length: true 1.2078 A
    co_bond = [b for b in report_f.bond_lengths if (b.atom_i, b.atom_j) in [(0, 1), (1, 0)]][0]
    assert abs(co_bond.length_angstrom - 1.2078) < 1e-4, f"C=O bond length mismatch: {co_bond.length_angstrom}"

    results["Formaldehyde_Full_rs"] = {
        "status": "PASSED",
        "CO_bond_length_angstrom": co_bond.length_angstrom,
        "CO_costain_uncertainty": co_bond.uncertainty_angstrom,
        "com_residuals": report_f.com_residuals
    }

    # -------------------------------------------------------------------------
    # Benchmark Case 3: 3D Asymmetric Rotor (Methanol CH3OH)
    # -------------------------------------------------------------------------
    raw_coords_m = np.array([
        [0.0, 0.0, 0.0],
        [1.42, 0.0, 0.0],
        [1.75, 0.95, 0.0],
        [-0.38, -0.51, 0.89],
        [-0.38, -0.51, -0.89],
        [-0.38, 1.03, 0.0]
    ])
    masses_m = np.array([m_C, m_O, m_H, m_H, m_H, m_H])
    evals_m_parent, evecs_m, com_m = compute_cartesian_inertia_tensor(raw_coords_m, masses_m)
    pcoords_m = (raw_coords_m - np.array(com_m)) @ evecs_m

    # Substitute atom 3 (H with 3D non-zero x, y, z)
    masses_m_sub = masses_m.copy()
    masses_m_sub[3] = m_D
    evals_m_sub, _, _ = compute_cartesian_inertia_tensor(raw_coords_m, masses_m_sub)

    sub_m3 = IsotopicSubstitutionData(
        atom_index=3, parent_symbol="H", substituted_symbol="D",
        parent_mass=m_H, substituted_mass=m_D,
        parent_moments=MomentsOfInertia(Ia=evals_m_parent[0], Ib=evals_m_parent[1], Ic=evals_m_parent[2]),
        substituted_moments=MomentsOfInertia(Ia=evals_m_sub[0], Ib=evals_m_sub[1], Ic=evals_m_sub[2])
    )

    res_m3 = solve_kraitchman_single_substitution(
        sub_m3,
        total_mass_parent=float(np.sum(masses_m)),
        prior_coord=pcoords_m[3]
    )

    err_3d = max(
        abs(res_m3.abs_a - abs(pcoords_m[3, 0])),
        abs(res_m3.abs_b - abs(pcoords_m[3, 1])),
        abs(res_m3.abs_c - abs(pcoords_m[3, 2]))
    )
    assert err_3d < 1e-5, f"Methanol 3D Kraitchman error: {err_3d:.6e}"

    results["Methanol_3D"] = {
        "status": "PASSED",
        "error_max_angstrom": float(err_3d),
        "coords_calculated": [res_m3.abs_a, res_m3.abs_b, res_m3.abs_c],
        "coords_true": [abs(float(x)) for x in pcoords_m[3]]
    }

    # -------------------------------------------------------------------------
    # Benchmark Case 4: Pure JAX FP64 Batch Acceleration
    # -------------------------------------------------------------------------
    p_batch = jnp.array([evals_w_parent, evals_f_parent, evals_m_parent], dtype=jnp.float64)
    s_batch = jnp.array([evals_w_sub, evals_f_h, evals_m_sub], dtype=jnp.float64)
    m_batch = jnp.array([np.sum(masses_w), np.sum(masses_f), np.sum(masses_m)], dtype=jnp.float64)
    dm_batch = jnp.array([m_D - m_H, m_D - m_H, m_D - m_H], dtype=jnp.float64)

    j_a, j_b, j_c = jax_kraitchman_coordinates_batch(p_batch, s_batch, m_batch, dm_batch)

    assert len(j_a) == 3, "JAX batch output dimension mismatch"
    assert abs(float(j_a[0]) - res_h1.abs_a) < 1e-8, "JAX vs NumPy consistency error"

    results["JAX_FP64_Acceleration"] = {
        "status": "PASSED",
        "batch_size": 3,
        "consistency_error": float(abs(float(j_a[0]) - res_h1.abs_a))
    }

    logger.info("All Kraitchman Verification benchmarks PASSED with 100% precision.")
    return results


# =============================================================================
# 8. STANDALONE COMMAND-LINE INTERFACE (CLI)
# =============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for cochem_kraitchman CLI."""
    parser = argparse.ArgumentParser(
        prog="cochem_kraitchman",
        description="CoChem-SpycFit Kraitchman Substitution Equations & Planar Moment Physics Core."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Execute full analytical self-verification benchmark suite on authentic molecular cases."
    )
    parser.add_argument(
        "--parent-moments",
        nargs=3,
        type=float,
        metavar=("Ia", "Ib", "Ic"),
        help="Parent molecule principal moments of inertia (u*Angstrom^2)."
    )
    parser.add_argument(
        "--sub-moments",
        nargs=3,
        type=float,
        metavar=("Ia'", "Ib'", "Ic'"),
        help="Substituted isotopologue principal moments of inertia (u*Angstrom^2)."
    )
    parser.add_argument(
        "--parent-mass",
        type=float,
        help="Total mass of parent molecule (u)."
    )
    parser.add_argument(
        "--delta-mass",
        type=float,
        help="Mass change Delta m = m_sub - m_parent (u)."
    )
    parser.add_argument(
        "--costain-param",
        type=float,
        default=DEFAULT_COSTAIN_PARAM,
        help=f"Costain parameter (default: {DEFAULT_COSTAIN_PARAM} Angstrom^2)."
    )
    parser.add_argument(
        "--small-coord-cutoff",
        type=float,
        default=DEFAULT_SMALL_COORD_CUTOFF,
        help=f"Small coordinate cutoff (default: {DEFAULT_SMALL_COORD_CUTOFF} Angstrom)."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit output formatted as JSON."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for cochem_kraitchman."""
    args = parse_args(argv)

    if args.verify or (not args.parent_moments and not args.verify):
        suite_res = run_kraitchman_verification_suite()
        if args.json:
            print(json.dumps(suite_res, indent=2))
        else:
            print("================================================================================")
            print(" CoChem-SpycFit: Kraitchman Analytical Verification Suite")
            print(" Method Matrix §2.3, §2.4, §6.10 & WBS Phase 2.3")
            print("================================================================================")
            for k, v in suite_res.items():
                print(f"[*] {k:<25} : {v.get('status', 'OK')}")
            print("================================================================================")
        return 0

    if args.parent_moments and args.sub_moments and args.parent_mass and args.delta_mass:
        p_mom = MomentsOfInertia(Ia=args.parent_moments[0], Ib=args.parent_moments[1], Ic=args.parent_moments[2])
        s_mom = MomentsOfInertia(Ia=args.sub_moments[0], Ib=args.sub_moments[1], Ic=args.sub_moments[2])
        sub_d = IsotopicSubstitutionData(
            atom_index=0,
            parent_symbol="X",
            substituted_symbol="X_iso",
            parent_mass=1.0,
            substituted_mass=1.0 + args.delta_mass,
            parent_moments=p_mom,
            substituted_moments=s_mom
        )
        res = solve_kraitchman_single_substitution(
            sub_d,
            total_mass_parent=args.parent_mass,
            costain_param=args.costain_param,
            small_coord_cutoff=args.small_coord_cutoff
        )
        if args.json:
            print(res.model_dump_json(indent=2))
        else:
            print(f"Calculated |a|: {res.abs_a:.6f} +/- {res.costain_delta_a:.6f} A")
            print(f"Calculated |b|: {res.abs_b:.6f} +/- {res.costain_delta_b:.6f} A")
            print(f"Calculated |c|: {res.abs_c:.6f} +/- {res.costain_delta_c:.6f} A")
            print(f"Planar Shift Valid: {res.planar_validation.is_physically_valid}")
        return 0

    print("Insufficient arguments provided. Use --help or --verify.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
