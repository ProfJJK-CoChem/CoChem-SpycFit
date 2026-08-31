"""
Colbert-Miller Sinc-DVR Quantum Hindered Rotor & Tunneling Splitting Solver.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple
import numpy as np

from cochem_base.core.glossary import UnitConversionConstants


class TorqThermoSolver:
    """
    Solves 1D torsional Schrödinger equation using Colbert-Miller periodic Sinc-DVR.
    """

    @classmethod
    def solve_sinc_dvr(
        cls,
        potential_func_kcal: Callable[[float], float],
        reduced_moment_amu_ang2: float = 0.5,
        num_grid_points: int = 61,
        temperature_k: float = 298.15,
    ) -> Dict[str, Any]:
        """
        Diagonalize periodic Sinc-DVR / Fourier Hamiltonian.
        """
        N = num_grid_points if num_grid_points % 2 == 1 else num_grid_points + 1
        theta_grid = np.linspace(0, 2.0 * math.pi, N, endpoint=False)

        # Kinetic operator in Fourier DVR basis
        m_vals = np.arange(-(N - 1) // 2, (N - 1) // 2 + 1)
        F = np.exp(1j * np.outer(theta_grid, m_vals)) / math.sqrt(N)
        T_op = np.real(F @ np.diag(m_vals**2) @ F.conj().T)

        # Convert I_red from amu * Angstrom^2 to Joules / (rad/s)^2
        amu_to_kg = UnitConversionConstants.AMU_TO_KG
        ang_to_m = 1.0e-10
        I_kg_m2 = reduced_moment_amu_ang2 * amu_to_kg * (ang_to_m**2)

        hbar = UnitConversionConstants.HBAR_J_S
        k_B = UnitConversionConstants.BOLTZMANN_CONSTANT_J_K
        cal_to_j = 4.184
        N_A = UnitConversionConstants.AVOGADRO_CONSTANT

        # Rotational constant B = hbar^2 / (2 * I_red) in Joules
        B_J = (hbar**2) / (2.0 * I_kg_m2)
        # Convert to kcal/mol
        B_kcal = (B_J * N_A) / (cal_to_j * 1000.0)

        # Potential energy matrix
        V_diag = np.array([potential_func_kcal(t) for t in theta_grid], dtype=float)
        V_mat = np.diag(V_diag)

        # Hamiltonian matrix H = B * T + V
        H = B_kcal * T_op + V_mat
        H = 0.5 * (H + H.T)

        eigvals, eigvecs = np.linalg.eigh(H)
        rel_eigvals = eigvals - eigvals[0]  # relative to ground state

        # Tunneling splitting (E1 - E0 in cm^-1)
        kcal_to_cm = 349.755
        tunneling_cm1 = (rel_eigvals[1] - rel_eigvals[0]) * kcal_to_cm if N > 1 else 0.0

        # Quantum partition function
        R_cal = UnitConversionConstants.GAS_CONSTANT_CAL_MOL_K
        RT_kcal = (R_cal * temperature_k) / 1000.0
        q_tor = float(np.sum(np.exp(-rel_eigvals / RT_kcal)))

        return {
            "eigenvalues_kcal_mol": rel_eigvals[:10].tolist(),
            "eigenvalues_cm1": (rel_eigvals[:10] * kcal_to_cm).tolist(),
            "tunneling_splitting_cm1": float(tunneling_cm1),
            "partition_function": q_tor,
            "ground_state_energy_kcal_mol": float(eigvals[0]),
        }
