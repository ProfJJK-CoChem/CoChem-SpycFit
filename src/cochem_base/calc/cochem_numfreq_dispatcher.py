"""
Numerical & Analytical Hessian Mass-Weighting, Sayvetz/Eckart Projection, and Frequencies.
"""

from __future__ import annotations

import math
from typing import List, Tuple
import numpy as np

from cochem_base.core.models import MolecularTopology
from cochem_base.core.metadata import get_atomic_mass
from cochem_base.core.glossary import UnitConversionConstants


class NumericalFrequencyDispatcher:
    """
    Projects translation/rotation out of Cartesian Hessians and diagonalizes to vibrational frequencies.
    """

    AU_TO_WAVENUMBER: float = 5140.487143

    @classmethod
    def compute_harmonic_frequencies(
        cls,
        topology: MolecularTopology,
        hessian_cartesian: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mass-weight Cartesian Hessian, project Sayvetz/Eckart modes, and diagonalize.

        Returns:
            frequencies_cm1: 1D array of harmonic frequencies (3N).
            normal_modes: 2D array of normal mode displacement vectors (3N, 3N).
        """
        n = topology.num_atoms
        if hessian_cartesian.shape != (3 * n, 3 * n):
            raise ValueError(
                f"Cartesian Hessian dimension {hessian_cartesian.shape} must match (3N, 3N) for N={n}"
            )

        masses = [get_atomic_mass(z) for z in topology.atomic_numbers]
        m_weights = []
        for m in masses:
            m_weights.extend([1.0 / math.sqrt(m)] * 3)
        inv_sqrt_m = np.diag(m_weights)

        coords = np.array(topology.coordinates_bohr, dtype=float)
        total_mass = sum(masses)
        com = np.sum(coords * np.array(masses)[:, None], axis=0) / total_mass
        coords_com = coords - com

        D = np.zeros((3 * n, 6), dtype=float)
        for i in range(n):
            sq_m = math.sqrt(masses[i])
            rx, ry, rz = coords_com[i]

            # Translations
            D[3 * i + 0, 0] = sq_m
            D[3 * i + 1, 1] = sq_m
            D[3 * i + 2, 2] = sq_m

            # Rotations
            D[3 * i + 1, 3] = -rz * sq_m
            D[3 * i + 2, 3] = ry * sq_m

            D[3 * i + 0, 4] = rz * sq_m
            D[3 * i + 2, 4] = -rx * sq_m

            D[3 * i + 0, 5] = -ry * sq_m
            D[3 * i + 1, 5] = rx * sq_m

        Q_tr, _ = np.linalg.qr(D)
        P = np.eye(3 * n) - Q_tr @ Q_tr.T

        H_mw = inv_sqrt_m @ hessian_cartesian @ inv_sqrt_m
        H_proj = P.T @ H_mw @ P
        H_proj = 0.5 * (H_proj + H_proj.T)

        eigvals, eigvecs = np.linalg.eigh(H_proj)

        frequencies = []
        for val in eigvals:
            if val >= 0:
                freq = math.sqrt(val) * cls.AU_TO_WAVENUMBER
            else:
                freq = -math.sqrt(abs(val)) * cls.AU_TO_WAVENUMBER
            frequencies.append(freq)

        freq_arr = np.array(frequencies, dtype=float)
        normal_modes_cart = inv_sqrt_m @ eigvecs

        return freq_arr, normal_modes_cart
