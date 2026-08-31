"""
Scientific Computational Chemistry Glossary and Standard Unit Conversion Constants.

Adheres strictly to CODATA 2022 and IUPAC physical constant definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class UnitConversionConstants:
    """IUPAC & CODATA 2022 exact physical unit conversion multipliers."""

    # Energy conversions
    HARTREE_TO_KCAL_MOL: float = 627.509474
    HARTREE_TO_EV: float = 27.211386245988
    HARTREE_TO_KJ_MOL: float = 2625.499639
    HARTREE_TO_JOULE: float = 4.3597447222071e-18
    HARTREE_TO_WAVENUMBER: float = 219474.63136320
    EV_TO_KCAL_MOL: float = 23.060548867
    KCAL_MOL_TO_KJ_MOL: float = 4.184

    # Distance conversions
    BOHR_TO_ANGSTROM: float = 0.529177210903
    ANGSTROM_TO_BOHR: float = 1.88972612462577

    # Fundamental physical constants (SI)
    SPEED_OF_LIGHT_CM_S: float = 2.99792458e10
    SPEED_OF_LIGHT_M_S: float = 2.99792458e8
    PLANCK_CONSTANT_J_S: float = 6.62607015e-34
    HBAR_J_S: float = 1.054571817e-34
    BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23
    AVOGADRO_CONSTANT: float = 6.02214076e23
    AMU_TO_KG: float = 1.66053906660e-27
    GAS_CONSTANT_CAL_MOL_K: float = 1.98720425864083
    GAS_CONSTANT_J_MOL_K: float = 8.31446261815324
    STANDARD_TEMPERATURE_K: float = 298.15
    STANDARD_PRESSURE_ATM: float = 1.0
    STANDARD_PRESSURE_BAR: float = 1.0e5
    ATM_TO_PASCAL: float = 101325.0


class CalculationFidelity(str, Enum):
    """Supported computational quantum chemistry fidelity tiers."""

    U_HF = "U_HF"
    R_DFT = "R_DFT"
    XTB2 = "XTB2"
    DLPNO_CCSD_T = "DLPNO_CCSD_T"
    CASSCF = "CASSCF"
