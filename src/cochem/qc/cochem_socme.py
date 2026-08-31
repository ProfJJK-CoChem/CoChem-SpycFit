"""
CoChem Spin-Orbit Coupling Matrix Elements (SOCME) and ISC Photophysics Kinetics Engine (REQ-QC-089).

Provenance:
- Method Matrix [M]: Spin-Orbit Mean-Field (SOMF) operator matrix elements, Fermi's Golden Rule ISC rate constants,
  and Marcus Franck-Condon Weighted Density (FCWD) of states.
- Derived Relations [D]: CODATA 2022 constants (hbar = 6.582119569e-16 eV s, k_B = 8.617333262e-5 eV/K, 1 cm^-1 = 1.239841984e-4 eV).
- Empirical [E]: Intersystem Crossing (ISC) kinetics and radiative phosphorescence lifetime estimations.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

from cochem.qc.models import SOCMEResult

# CODATA 2022 Physical Constants
H_BAR_EV_S: float = 6.582119569e-16
K_BOLTZMANN_EV_K: float = 8.617333262e-5
CM1_TO_EV: float = 1.239841984e-4


def compute_socme_and_isc_rate(
    initial_state: str,
    final_state: str,
    delta_e_st_ev: float,
    socme_components_cm1: Tuple[float, float, float],
    reorganization_energy_ev: float = 0.2,
    temperature_k: float = 298.15,
) -> SOCMEResult:
    """
    [M][D] Compute Spin-Orbit Coupling Matrix Elements and Intersystem Crossing rate (REQ-QC-089).

    Formulation:
        1. Effective SOMF coupling magnitude:
           ||V_SO|| = sqrt(H_SO,x^2 + H_SO,y^2 + H_SO,z^2) in cm^-1
        2. Marcus Franck-Condon Weighted Density (FCWD):
           FCWD(Delta E_ST) = (1 / sqrt(4 * pi * lambda * k_B * T)) * exp(-(Delta E_ST + lambda)^2 / (4 * lambda * k_B * T))
        3. Fermi's Golden Rule ISC rate constant:
           k_ISC = (2 * pi / hbar) * |V_SO,eV|^2 * FCWD
        4. Phosphorescence radiative lifetime tau_phos approx 1 / k_ISC.

    Args:
        initial_state: Label of initial electronic state (e.g. 'S1').
        final_state: Label of target electronic state (e.g. 'T1').
        delta_e_st_ev: State adiabatic or vertical energy gap Delta E_ST in eV.
        socme_components_cm1: Cartesian components (H_SO,x, H_SO,y, H_SO,z) in cm^-1.
        reorganization_energy_ev: Reorganization energy lambda in eV (default: 0.2 eV).
        temperature_k: Absolute temperature in Kelvin (default: 298.15 K).

    Returns:
        SOCMEResult containing matrix element norms, FCWD, k_ISC rate, and lifetime.

    Raises:
        ValueError: If temperature_k <= 0 or reorganization_energy_ev <= 0.
    """
    if temperature_k <= 0.0:
        raise ValueError(f"Temperature must be positive, got T={temperature_k} K.")
    if reorganization_energy_ev <= 0.0:
        raise ValueError(f"Reorganization energy must be positive, got lambda={reorganization_energy_ev} eV.")

    # 1. Total SOCME magnitude
    lx, ly, lz = socme_components_cm1
    socme_mag_cm1 = math.sqrt(lx ** 2 + ly ** 2 + lz ** 2)
    socme_mag_ev = socme_mag_cm1 * CM1_TO_EV

    # 2. Marcus Franck-Condon Weighted Density of States FCWD (eV^-1)
    k_b_t = K_BOLTZMANN_EV_K * temperature_k
    fcwd_denom = math.sqrt(4.0 * math.pi * reorganization_energy_ev * k_b_t)
    exponent = -((delta_e_st_ev + reorganization_energy_ev) ** 2) / (4.0 * reorganization_energy_ev * k_b_t)
    fcwd_val = (1.0 / fcwd_denom) * math.exp(exponent)

    # 3. Fermi's Golden Rule ISC Rate Constant k_ISC (s^-1)
    prefactor = (2.0 * math.pi) / H_BAR_EV_S
    v_so_squared = socme_mag_ev ** 2
    k_isc = prefactor * v_so_squared * fcwd_val

    # 4. Phosphorescence lifetime estimation
    tau_phos: Optional[float] = None
    if k_isc > 1e-12:
        tau_phos = 1.0 / k_isc

    return SOCMEResult(
        initial_state=initial_state,
        final_state=final_state,
        delta_e_singlet_triplet_ev=round(delta_e_st_ev, 5),
        socme_magnitude_cm1=round(socme_mag_cm1, 5),
        socme_components_cm1=(round(lx, 5), round(ly, 5), round(lz, 5)),
        fcwd_density_states=round(fcwd_val, 6),
        k_isc_s_inv=k_isc,
        phosphorescence_lifetime_seconds=tau_phos,
    )
