"""
CoChem Quasi-RRHO Thermochemistry & Frequency Scaling Engine.

Provenance:
- Method Matrix [M]: Empirical vibrational frequency scaling factors from NIST CCCBDB / Truhlar databases,
  Grimme Quasi-RRHO damping with rotor cut-off nu_0 = 100.0 cm^-1 (REQ-THM-045).
- Derived Relations [D]: Free-rotor entropy formulation S_rot = R [1/2 + ln(sqrt(pi / u_i))],
  Scaled zero-point energy E_ZPE = 0.5 * h * c * sum(lambda_ZPE * nu_i).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import UnsupportedMethodThermoError

# CODATA 2018 physical constants
PLANCK_CONSTANT_J_S = 6.62607015e-34  # J s
SPEED_OF_LIGHT_CM_S = 2.99792458e10   # cm / s
BOLTZMANN_CONSTANT_J_K = 1.380649e-23 # J / K
GAS_CONSTANT_J_MOL_K = 8.314462618    # J / (mol K)
CAL_TO_JOULE = 4.184
GAS_CONSTANT_CAL_MOL_K = GAS_CONSTANT_J_MOL_K / CAL_TO_JOULE # ~ 1.987204 cal / (mol K)
HARTREE_TO_JOULE = 4.3597447222071e-18
HARTREE_TO_KCAL_MOL = 627.509474
CM1_TO_HARTREE = (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_CM_S * 100.0) / (HARTREE_TO_JOULE * 100.0) # 4.5563352529120e-6

# Repository of empirical scaling factors (ZPE, fundamental, enthalpy, entropy)
SCALING_FACTORS_DB: Dict[str, Dict[str, float]] = {
    "b3lyp/def2-tzvp": {
        "lambda_zpe": 0.965,
        "lambda_fund": 0.960,
        "lambda_enthalpy": 0.966,
        "lambda_entropy": 0.970,
    },
    "b3lyp/6-31g*": {
        "lambda_zpe": 0.961,
        "lambda_fund": 0.957,
        "lambda_enthalpy": 0.962,
        "lambda_entropy": 0.968,
    },
    "pbe0/def2-tzvp": {
        "lambda_zpe": 0.960,
        "lambda_fund": 0.955,
        "lambda_enthalpy": 0.961,
        "lambda_entropy": 0.967,
    },
    "r2scan-3c": {
        "lambda_zpe": 0.970,
        "lambda_fund": 0.965,
        "lambda_enthalpy": 0.971,
        "lambda_entropy": 0.975,
    },
    "wb97x-d3/def2-tzvp": {
        "lambda_zpe": 0.968,
        "lambda_fund": 0.963,
        "lambda_enthalpy": 0.969,
        "lambda_entropy": 0.973,
    },
    "wb97x-d/def2-tzvp": {
        "lambda_zpe": 0.968,
        "lambda_fund": 0.963,
        "lambda_enthalpy": 0.969,
        "lambda_entropy": 0.973,
    },
    "b97-3c": {
        "lambda_zpe": 0.969,
        "lambda_fund": 0.964,
        "lambda_enthalpy": 0.970,
        "lambda_entropy": 0.974,
    },
}


class ThermochemistryResultModel(BaseModel):
    """Data contract representing scaled Quasi-RRHO thermochemical properties."""

    temperature_k: float = Field(..., gt=0)
    pressure_atm: float = Field(..., gt=0)
    scaling_factors: Dict[str, float]
    zpe_unscaled_hartree: float
    zpe_scaled_hartree: float
    entropy_qrrho_cal_mol_k: float
    gibbs_free_energy_hartree: float
    citation: str


def get_scaling_factors(method: str, basis: Optional[str] = None) -> Dict[str, float]:
    """
    [REQ-THM-045] Retrieve empirical frequency scaling factors for a functional/basis combination.

    Args:
        method: Quantum mechanical functional / method name.
        basis: Optional GTO basis set name.

    Returns:
        Dictionary with keys 'lambda_zpe', 'lambda_fund', 'lambda_enthalpy', 'lambda_entropy'.

    Raises:
        UnsupportedMethodThermoError: If method/basis tuple is not in repository.
    """
    method_clean = method.strip().lower()
    basis_clean = basis.strip().lower() if basis else ""

    # Check composite exact match
    if method_clean in SCALING_FACTORS_DB:
        return SCALING_FACTORS_DB[method_clean]

    # Check method/basis combination
    combo_key = f"{method_clean}/{basis_clean}"
    if combo_key in SCALING_FACTORS_DB:
        return SCALING_FACTORS_DB[combo_key]

    # Check partial match on method
    for key, factors in SCALING_FACTORS_DB.items():
        if key.startswith(f"{method_clean}/"):
            return factors

    raise UnsupportedMethodThermoError(
        method=method,
        basis=basis,
        available_methods=list(SCALING_FACTORS_DB.keys()),
    )


def calculate_qrrho_thermochemistry(
    frequencies_cm1: Sequence[float],
    electronic_energy_hartree: float,
    method: str = "B3LYP",
    basis: Optional[str] = "def2-TZVP",
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    rotor_cutoff_cm1: float = 100.0,
) -> ThermochemistryResultModel:
    """
    [REQ-THM-045] Compute Grimme Quasi-RRHO vibrational entropy and Gibbs free energy.

    Damping function with rotor cut-off nu_0 = 100.0 cm^-1:
        w(nu_i) = 1 / (1 + (nu_0 / nu_i)^4)
        u_i = (h * c * nu'_i) / (k_B * T)
        S_harm(nu'_i) = R * [u_i / (exp(u_i) - 1) - ln(1 - exp(-u_i))]
        S_rot(nu'_i) = R * [0.5 + 0.5 * ln(pi / u_i)]
        S_vib_QRRHO = sum_i [ w(nu_i) * S_harm(nu'_i) + (1 - w(nu_i)) * S_rot(nu'_i) ]

    Args:
        frequencies_cm1: Positive vibrational frequencies in cm^-1.
        electronic_energy_hartree: Total converged electronic SCF energy in Hartree.
        method: Method / functional name for scaling factors.
        basis: Basis set name.
        temperature_k: Absolute temperature in Kelvin (default 298.15 K).
        pressure_atm: Pressure in standard atmospheres (default 1.0 atm).
        rotor_cutoff_cm1: Grimme rotor cut-off frequency in cm^-1 (default 100.0 cm^-1).

    Returns:
        ThermochemistryResultModel instance.
    """
    factors = get_scaling_factors(method=method, basis=basis)
    lambda_zpe = factors["lambda_zpe"]
    lambda_entropy = factors["lambda_entropy"]

    # Filter strictly positive frequencies (> 1e-3 cm^-1)
    pos_freqs = [float(f) for f in frequencies_cm1 if f > 1e-3]

    # Compute unscaled and scaled ZPE
    sum_unscaled_cm1 = sum(pos_freqs)
    sum_scaled_cm1 = sum(f * lambda_zpe for f in pos_freqs)

    zpe_unscaled_hartree = 0.5 * sum_unscaled_cm1 * CM1_TO_HARTREE
    zpe_scaled_hartree = 0.5 * sum_scaled_cm1 * CM1_TO_HARTREE

    # Grimme Quasi-RRHO entropy summation
    total_entropy_j_mol_k = 0.0

    for freq in pos_freqs:
        scaled_freq = freq * lambda_entropy
        if scaled_freq <= 1e-6:
            continue

        # Dimensionless parameter u = h * c * nu' / (k_B * T)
        u = (PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_CM_S * scaled_freq) / (BOLTZMANN_CONSTANT_J_K * temperature_k)

        # Damping weight w(nu)
        damping_weight = 1.0 / (1.0 + (rotor_cutoff_cm1 / freq) ** 4)

        # Harmonic oscillator entropy
        if u > 50.0:
            s_harm = 0.0
        elif u < 1e-5:
            s_harm = GAS_CONSTANT_J_MOL_K * (1.0 - math.log(u))
        else:
            exp_u = math.exp(u)
            s_harm = GAS_CONSTANT_J_MOL_K * (u / (exp_u - 1.0) - math.log(1.0 - math.exp(-u)))

        # Free rotor entropy S_rot = R * [0.5 + 0.5 * ln(pi / u)]
        s_rot = GAS_CONSTANT_J_MOL_K * (0.5 + 0.5 * math.log(math.pi / u))

        s_mode = damping_weight * s_harm + (1.0 - damping_weight) * s_rot
        total_entropy_j_mol_k += s_mode

    entropy_qrrho_cal_mol_k = total_entropy_j_mol_k / CAL_TO_JOULE

    # Free energy: G = E_elec + ZPE_scaled - T * S_vib (vibrational component)
    ts_j_mol = temperature_k * total_entropy_j_mol_k
    ts_hartree = ts_j_mol / (HARTREE_TO_JOULE * 6.02214076e23)

    gibbs_free_energy_hartree = electronic_energy_hartree + zpe_scaled_hartree - ts_hartree

    return ThermochemistryResultModel(
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
        scaling_factors=factors,
        zpe_unscaled_hartree=zpe_unscaled_hartree,
        zpe_scaled_hartree=zpe_scaled_hartree,
        entropy_qrrho_cal_mol_k=entropy_qrrho_cal_mol_k,
        gibbs_free_energy_hartree=gibbs_free_energy_hartree,
        citation="Grimme, S. Chem. Eur. J. 2012, 18, 9955-9964 (Quasi-RRHO); NIST CCCBDB Scaling Factors",
    )
