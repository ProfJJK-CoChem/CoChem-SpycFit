"""
CoChem Rigorous Statistical Thermodynamics and Quasi-RRHO Engine (REQ-QC-084, REQ-QC-090).

Provenance:
- Method Matrix [M]: Rigid-Rotor Harmonic-Oscillator (RRHO) and Grimme Quasi-Harmonic vibrational entropy interpolation.
- Derived Relations [D]: Dynamic Mendeleev masses (IUPAC CIAAW), CODATA 2022 constants.
- Empirical [E]: Low-frequency cutoff nu_0 = 100.0 cm^-1 for free-rotor damping.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import get_ciaaw_atomic_mass
from cochem.qc.cochem_solvation import compute_standard_state_correction

# CODATA 2022 Physical Constants
PLANCK_CONSTANT_J_S: float = 6.62607015e-34       # J s
SPEED_OF_LIGHT_CM_S: float = 2.99792458e10        # cm / s
BOLTZMANN_CONSTANT_J_K: float = 1.380649e-23      # J / K
GAS_CONSTANT_J_MOL_K: float = 8.314462618         # J / (mol K)
AVOGADRO_CONSTANT: float = 6.02214076e23          # mol^-1
AMU_TO_KG: float = 1.66053906660e-27              # kg
HARTREE_TO_JOULE: float = 4.3597447222071e-18
HARTREE_TO_KCAL_MOL: float = 627.5094740631
CAL_TO_JOULE: float = 4.184
GAS_CONSTANT_CAL_MOL_K: float = GAS_CONSTANT_J_MOL_K / CAL_TO_JOULE # ~ 1.987204 cal / (mol K)
CM1_TO_JOULE: float = PLANCK_CONSTANT_J_S * SPEED_OF_LIGHT_CM_S
CM1_TO_KCAL_MOL: float = (CM1_TO_JOULE * AVOGADRO_CONSTANT) / (CAL_TO_JOULE * 1000.0) # ~ 2.859143e-3


class PartitionFunctions(BaseModel):
    """Data contract representing assembled statistical partition functions."""
    model_config = ConfigDict(frozen=True)

    q_trans: float = Field(description="Translational partition function")
    q_rot: float = Field(description="Rotational partition function")
    q_vib: float = Field(description="Vibrational partition function")
    q_elec: float = Field(default=1.0, description="Electronic partition function")
    q_total: float = Field(description="Total canonical molecular partition function")


class ThermochemicalSummary(BaseModel):
    """Data contract representing rigorous RRHO / Quasi-Harmonic thermochemical state."""
    model_config = ConfigDict(frozen=True)

    temperature_k: float = Field(gt=0.0)
    pressure_atm: float = Field(gt=0.0)
    electronic_energy_hartree: float
    zpe_hartree: float
    thermal_energy_u_hartree: float
    enthalpy_h_hartree: float
    entropy_total_cal_mol_k: float
    entropy_trans_cal_mol_k: float
    entropy_rot_cal_mol_k: float
    entropy_vib_cal_mol_k: float
    gibbs_free_energy_gas_hartree: float
    gibbs_free_energy_solv_hartree: Optional[float] = None
    partition_functions: PartitionFunctions


def compute_translational_partition_function(
    total_mass_amu: float,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
) -> Tuple[float, float, float]:
    """
    Compute q_trans, U_trans (in Hartree), and S_trans (in cal/(mol K)).
    """
    m_kg = total_mass_amu * AMU_TO_KG
    k_b = BOLTZMANN_CONSTANT_J_K
    h = PLANCK_CONSTANT_J_S
    p_pa = pressure_atm * 101325.0

    v_m3 = (k_b * temperature_k) / p_pa
    thermal_wavelength = h / math.sqrt(2.0 * math.pi * m_kg * k_b * temperature_k)
    q_trans = v_m3 / (thermal_wavelength ** 3)

    # U_trans = 3/2 R T
    u_trans_j_mol = 1.5 * GAS_CONSTANT_J_MOL_K * temperature_k
    u_trans_hartree = (u_trans_j_mol / AVOGADRO_CONSTANT) / HARTREE_TO_JOULE

    # Sackur-Tetrode Equation for S_trans
    s_trans_j_mol_k = GAS_CONSTANT_J_MOL_K * (math.log(q_trans) + 2.5)
    s_trans_cal_mol_k = s_trans_j_mol_k / CAL_TO_JOULE

    return q_trans, u_trans_hartree, s_trans_cal_mol_k


def compute_rotational_partition_function(
    moments_of_inertia_amu_ang2: Sequence[float],
    temperature_k: float = 298.15,
    symmetry_number: int = 1,
    is_linear: bool = False,
) -> Tuple[float, float, float]:
    """
    Compute q_rot, U_rot (in Hartree), and S_rot (in cal/(mol K)).
    """
    k_b = BOLTZMANN_CONSTANT_J_K
    h = PLANCK_CONSTANT_J_S
    h_bar = h / (2.0 * math.pi)

    if is_linear:
        # One non-zero moment of inertia I (average of non-zero)
        non_zeros = [i for i in moments_of_inertia_amu_ang2 if i > 1e-4]
        i_avg_amu_ang2 = non_zeros[0] if non_zeros else 1.0
        i_kg_m2 = i_avg_amu_ang2 * AMU_TO_KG * 1e-20
        theta_rot = (h_bar ** 2) / (2.0 * i_kg_m2 * k_b)
        q_rot = temperature_k / (symmetry_number * theta_rot)

        u_rot_j_mol = GAS_CONSTANT_J_MOL_K * temperature_k
        s_rot_j_mol_k = GAS_CONSTANT_J_MOL_K * (math.log(q_rot) + 1.0)
    else:
        # Non-linear polyatomic
        i_kg_m2 = [max(1e-6, float(i)) * AMU_TO_KG * 1e-20 for i in moments_of_inertia_amu_ang2[:3]]
        theta_a, theta_b, theta_c = [(h_bar ** 2) / (2.0 * inert * k_b) for inert in i_kg_m2]
        q_rot = (math.sqrt(math.pi) / symmetry_number) * math.sqrt((temperature_k ** 3) / (theta_a * theta_b * theta_c))

        u_rot_j_mol = 1.5 * GAS_CONSTANT_J_MOL_K * temperature_k
        s_rot_j_mol_k = GAS_CONSTANT_J_MOL_K * (math.log(q_rot) + 1.5)

    u_rot_hartree = (u_rot_j_mol / AVOGADRO_CONSTANT) / HARTREE_TO_JOULE
    s_rot_cal_mol_k = s_rot_j_mol_k / CAL_TO_JOULE

    return q_rot, u_rot_hartree, s_rot_cal_mol_k


def compute_vibrational_thermodynamics(
    frequencies_cm1: Sequence[float],
    temperature_k: float = 298.15,
    grimme_cutoff_cm1: float = 100.0,
) -> Tuple[float, float, float, float]:
    """
    Compute ZPE (Hartree), U_vib (Hartree), S_vib (cal/(mol K)) with Grimme quasi-harmonic damping, and q_vib.
    """
    h = PLANCK_CONSTANT_J_S
    c = SPEED_OF_LIGHT_CM_S
    k_b = BOLTZMANN_CONSTANT_J_K
    r_cal = GAS_CONSTANT_CAL_MOL_K

    pos_freqs = [f for f in frequencies_cm1 if f > 1.0]

    # 1. ZPE in Hartree
    # ZPE = sum(0.5 * h * c * nu_i)
    total_zpe_joules = sum(0.5 * h * c * f for f in pos_freqs)
    zpe_hartree = total_zpe_joules / HARTREE_TO_JOULE

    # 2. Vibrational internal thermal energy U_vib and partition function q_vib
    u_vib_joules = 0.0
    q_vib = 1.0
    s_vib_cal = 0.0

    avg_b_cm1 = 0.1 # Approximate average rotational constant for rotor limit
    i_rotor_kg_m2 = h / (8.0 * math.pi ** 2 * c * avg_b_cm1 * 100.0)

    for nu in pos_freqs:
        x = (h * c * nu) / (k_b * temperature_k)
        if x < 100.0:
            exp_x = math.exp(x)
            q_i = 1.0 / (1.0 - math.exp(-x))
            q_vib *= q_i

            # Standard harmonic oscillator entropy
            s_ho = r_cal * (x / (exp_x - 1.0) - math.log(1.0 - math.exp(-x)))

            # Free rotor entropy for low-frequency modes
            mu_rotor = (h / (8.0 * math.pi ** 2 * c * nu * 100.0))
            mu_prime = (mu_rotor * i_rotor_kg_m2) / (mu_rotor + i_rotor_kg_m2)
            s_rotor = r_cal * (0.5 + math.log(math.sqrt((8.0 * math.pi ** 3 * mu_prime * k_b * temperature_k) / (h ** 2))))

            # Grimme damping function w(nu) = 1 / (1 + (nu_0 / nu)^4)
            w = 1.0 / (1.0 + (grimme_cutoff_cm1 / nu) ** 4)
            s_qh = w * s_ho + (1.0 - w) * s_rotor
            s_vib_cal += s_qh

            # U_vib contribution
            u_vib_joules += (h * c * nu) / (exp_x - 1.0)

    u_vib_hartree = u_vib_joules / HARTREE_TO_JOULE
    return zpe_hartree, u_vib_hartree, s_vib_cal, q_vib


def compute_statistical_thermodynamics(
    symbols: Sequence[str],
    frequencies_cm1: Sequence[float],
    moments_of_inertia_amu_ang2: Sequence[float],
    electronic_energy_hartree: float,
    temperature_k: float = 298.15,
    pressure_atm: float = 1.0,
    symmetry_number: int = 1,
    is_linear: bool = False,
    solvation_energy_kcal_mol: Optional[float] = None,
    solvent_name: Optional[str] = None,
) -> ThermochemicalSummary:
    """
    [REQ-QC-084, REQ-QC-090] Assemble rigorous statistical thermodynamic properties.
    """
    # 1. Total molecular mass from dynamic Mendeleev masses
    total_mass_amu = sum(get_ciaaw_atomic_mass(s) for s in symbols)

    # 2. Translational partition function and components
    q_trans, u_trans_h, s_trans = compute_translational_partition_function(
        total_mass_amu=total_mass_amu,
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
    )

    # 3. Rotational partition function and components
    q_rot, u_rot_h, s_rot = compute_rotational_partition_function(
        moments_of_inertia_amu_ang2=moments_of_inertia_amu_ang2,
        temperature_k=temperature_k,
        symmetry_number=symmetry_number,
        is_linear=is_linear,
    )

    # 4. Vibrational partition function and Quasi-Harmonic Grimme entropy
    zpe_h, u_vib_h, s_vib, q_vib = compute_vibrational_thermodynamics(
        frequencies_cm1=frequencies_cm1,
        temperature_k=temperature_k,
    )

    # 5. Total Energy, Enthalpy, Entropy, Gibbs Free Energy
    # Internal thermal energy: U(T) = E_elec + ZPE + Delta U_trans + Delta U_rot + Delta U_vib
    u_total_h = electronic_energy_hartree + zpe_h + u_trans_h + u_rot_h + u_vib_h

    # Enthalpy: H(T) = U(T) + k_B T (k_B T in Hartree)
    k_b_t_h = (BOLTZMANN_CONSTANT_J_K * temperature_k) / HARTREE_TO_JOULE
    h_total_h = u_total_h + k_b_t_h

    # Entropy: S_total = S_trans + S_rot + S_vib
    s_total_cal = s_trans + s_rot + s_vib
    t_s_kcal_mol = (temperature_k * s_total_cal) / 1000.0
    t_s_hartree = t_s_kcal_mol / HARTREE_TO_KCAL_MOL

    # Gibbs Free Energy (Gas): G_gas = H(T) - T * S
    g_gas_h = h_total_h - t_s_hartree

    # Gibbs Free Energy (Solution 1.0 M) if solvation is present
    g_solv_h: Optional[float] = None
    if solvation_energy_kcal_mol is not None and solvent_name:
        std_shift = compute_standard_state_correction(
            g_gas_1atm_hartree=g_gas_h,
            temperature_k=temperature_k,
            pressure_atm=pressure_atm,
            solvent_name=solvent_name,
        )
        solv_contrib_h = solvation_energy_kcal_mol / HARTREE_TO_KCAL_MOL
        g_solv_h = std_shift.g_solv_1m_hartree + solv_contrib_h

    q_total = q_trans * q_rot * q_vib

    pf_model = PartitionFunctions(
        q_trans=q_trans,
        q_rot=q_rot,
        q_vib=q_vib,
        q_elec=1.0,
        q_total=q_total,
    )

    return ThermochemicalSummary(
        temperature_k=temperature_k,
        pressure_atm=pressure_atm,
        electronic_energy_hartree=round(electronic_energy_hartree, 6),
        zpe_hartree=round(zpe_h, 6),
        thermal_energy_u_hartree=round(u_total_h, 6),
        enthalpy_h_hartree=round(h_total_h, 6),
        entropy_total_cal_mol_k=round(s_total_cal, 4),
        entropy_trans_cal_mol_k=round(s_trans, 4),
        entropy_rot_cal_mol_k=round(s_rot, 4),
        entropy_vib_cal_mol_k=round(s_vib, 4),
        gibbs_free_energy_gas_hartree=round(g_gas_h, 6),
        gibbs_free_energy_solv_hartree=round(g_solv_h, 6) if g_solv_h is not None else None,
        partition_functions=pf_model,
    )
