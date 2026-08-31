"""
Statistical Mechanics & Quasi-RRHO (qRRHO) Molecular Thermochemistry Governor.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple
import numpy as np

from cochem_base.core.models import MolecularTopology
from cochem_base.core.metadata import get_atomic_mass
from cochem_base.core.glossary import UnitConversionConstants


class ThermalGovernor:
    """
    Computes enthalpy H(T), entropy S(T), and Gibbs Free Energy G(T) via Grimme qRRHO.
    """

    @classmethod
    def compute_moments_of_inertia(
        cls, topology: MolecularTopology
    ) -> Tuple[float, float, float]:
        """Compute principal moments of inertia in kg * m^2."""
        coords_ang = np.array(topology.get_coordinates_angstrom(), dtype=float)
        masses_amu = np.array([get_atomic_mass(z) for z in topology.atomic_numbers], dtype=float)

        com = np.sum(coords_ang * masses_amu[:, None], axis=0) / np.sum(masses_amu)
        r = coords_ang - com

        I_mat = np.zeros((3, 3), dtype=float)
        for i in range(len(masses_amu)):
            m = masses_amu[i]
            x, y, z = r[i]
            I_mat[0, 0] += m * (y**2 + z**2)
            I_mat[1, 1] += m * (x**2 + z**2)
            I_mat[2, 2] += m * (x**2 + y**2)
            I_mat[0, 1] -= m * x * y
            I_mat[0, 2] -= m * x * z
            I_mat[1, 2] -= m * y * z
        I_mat[1, 0] = I_mat[0, 1]
        I_mat[2, 0] = I_mat[0, 2]
        I_mat[2, 1] = I_mat[1, 2]

        eigvals = np.linalg.eigvalsh(I_mat)
        eigvals.sort()

        amu_to_kg = UnitConversionConstants.AMU_TO_KG
        ang_to_m = 1.0e-10
        factor = amu_to_kg * (ang_to_m**2)

        return (
            float(max(1e-50, eigvals[0] * factor)),
            float(max(1e-50, eigvals[1] * factor)),
            float(max(1e-50, eigvals[2] * factor)),
        )

    @classmethod
    def compute_thermochemistry(
        cls,
        topology: MolecularTopology,
        frequencies_cm1: List[float],
        electronic_energy_hartree: float,
        temperature_k: float = 298.15,
        pressure_atm: float = 1.0,
        sigma: int = 1,
        multiplicity: int = 1,
        cutoff_cm1: float = 100.0,
    ) -> Dict[str, float]:
        """
        Compute standard thermodynamic properties using Grimme / Head-Gordon qRRHO.
        """
        k_B = UnitConversionConstants.BOLTZMANN_CONSTANT_J_K
        h = UnitConversionConstants.PLANCK_CONSTANT_J_S
        hbar = UnitConversionConstants.HBAR_J_S
        R_cal = UnitConversionConstants.GAS_CONSTANT_CAL_MOL_K
        R_J = UnitConversionConstants.GAS_CONSTANT_J_MOL_K
        c_cm = UnitConversionConstants.SPEED_OF_LIGHT_CM_S
        cal_to_j = 4.184
        hartree_to_kcal = UnitConversionConstants.HARTREE_TO_KCAL_MOL
        N_A = UnitConversionConstants.AVOGADRO_CONSTANT

        # 1. Translational entropy
        m_tot_kg = sum(get_atomic_mass(z) for z in topology.atomic_numbers) * UnitConversionConstants.AMU_TO_KG
        p_pa = pressure_atm * UnitConversionConstants.ATM_TO_PASCAL
        V_molecule = k_B * temperature_k / p_pa
        Lambda = h / math.sqrt(2.0 * math.pi * m_tot_kg * k_B * temperature_k)
        S_trans_J = R_J * (math.log(V_molecule / (Lambda**3)) + 2.5)
        S_trans_cal = S_trans_J / cal_to_j

        # 2. Rotational entropy
        Ia, Ib, Ic = cls.compute_moments_of_inertia(topology)
        I_av = (Ia + Ib + Ic) / 3.0
        is_linear = (topology.num_atoms == 2) or (Ia < 1e-45)
        if is_linear:
            I_lin = max(Ia, Ib, Ic)
            T_rot = (h**2) / (8.0 * (math.pi**2) * I_lin * k_B)
            q_rot = temperature_k / (sigma * T_rot)
            S_rot_cal = (R_J * (math.log(max(1e-10, q_rot)) + 1.0)) / cal_to_j
        else:
            term = (8.0 * (math.pi**2) * k_B * temperature_k / (h**2)) ** 1.5
            q_rot = (math.sqrt(math.pi) / sigma) * term * math.sqrt(Ia * Ib * Ic)
            S_rot_cal = (R_J * (math.log(max(1e-10, q_rot)) + 1.5)) / cal_to_j

        # 3. Vibrational entropy & ZPVE
        real_freqs = [f for f in frequencies_cm1 if f > 1.0]
        zpve_kcal = 0.0
        S_vib_cal = 0.0
        E_vib_thermal_kcal = 0.0

        for nu in real_freqs:
            zpve_kcal += 0.5 * h * c_cm * nu * N_A / (cal_to_j * 1000.0)

            x = (h * c_cm * nu) / (k_B * temperature_k)
            if x < 100.0:
                S_ho = R_cal * (x / (math.exp(x) - 1.0) - math.log(1.0 - math.exp(-x)))
                E_vib_thermal_kcal += (R_cal * temperature_k * x / (math.exp(x) - 1.0)) / 1000.0
            else:
                S_ho = 0.0

            mu_eff = (hbar / (4.0 * math.pi * c_cm * nu)) * (I_av / (I_av + hbar / (4.0 * math.pi * c_cm * nu)))
            S_rotor = R_cal * (0.5 + math.log(max(1e-10, math.sqrt((8.0 * (math.pi**3) * mu_eff * k_B * temperature_k) / (h**2)))))

            w = 1.0 / (1.0 + (cutoff_cm1 / nu) ** 4)
            S_vib_cal += w * S_ho + (1.0 - w) * S_rotor

        # 4. Electronic entropy
        S_elec_cal = R_cal * math.log(multiplicity)

        S_tot_cal_mol_k = S_trans_cal + S_rot_cal + S_vib_cal + S_elec_cal

        H_trans_rot_kcal = (3.0 if not is_linear else 2.5) * (R_cal * temperature_k / 1000.0)
        H_thermal_kcal = zpve_kcal + E_vib_thermal_kcal + H_trans_rot_kcal + (R_cal * temperature_k / 1000.0)
        
        zpve_hartree = zpve_kcal / hartree_to_kcal
        H_tot_hartree = electronic_energy_hartree + (H_thermal_kcal / hartree_to_kcal)
        
        TS_kcal = (temperature_k * S_tot_cal_mol_k) / 1000.0
        G_tot_hartree = H_tot_hartree - (TS_kcal / hartree_to_kcal)

        return {
            "temperature_k": temperature_k,
            "pressure_atm": pressure_atm,
            "electronic_energy_hartree": electronic_energy_hartree,
            "zpve_hartree": zpve_hartree,
            "zpve_kcal_mol": zpve_kcal,
            "entropy_trans_cal_mol_k": S_trans_cal,
            "entropy_rot_cal_mol_k": S_rot_cal,
            "entropy_vib_cal_mol_k": S_vib_cal,
            "entropy_total_cal_mol_k": S_tot_cal_mol_k,
            "enthalpy_hartree": H_tot_hartree,
            "gibbs_free_energy_hartree": G_tot_hartree,
        }
