"""
Periodic Cubic Spline & Fourier Series Slicer for 1D/2D Torsional PES.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.interpolate import CubicSpline


class TorqSlicer:
    """
    Fits and interpolates periodic potential energy surfaces with analytical derivatives.
    """

    @classmethod
    def fit_fourier_series(
        cls,
        angles_rad: np.ndarray,
        energies_kcal: np.ndarray,
        num_harmonics: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit periodic Fourier expansion V(theta) = a0/2 + sum(an cos(n theta) + bn sin(n theta)).
        """
        N = len(angles_rad)
        A = np.zeros((N, 2 * num_harmonics + 1), dtype=float)
        A[:, 0] = 0.5

        for n in range(1, num_harmonics + 1):
            A[:, 2 * n - 1] = np.cos(n * angles_rad)
            A[:, 2 * n] = np.sin(n * angles_rad)

        coeffs, _, _, _ = np.linalg.lstsq(A, energies_kcal, rcond=None)
        return coeffs, A

    @classmethod
    def evaluate_fourier_series(
        cls,
        theta_rad: np.ndarray,
        coeffs: np.ndarray,
        num_harmonics: int = 6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate Fourier potential V(theta) and analytical derivatives V'(theta), V''(theta).
        """
        V = np.full_like(theta_rad, coeffs[0] * 0.5, dtype=float)
        dV = np.zeros_like(theta_rad, dtype=float)
        d2V = np.zeros_like(theta_rad, dtype=float)

        for n in range(1, num_harmonics + 1):
            an = coeffs[2 * n - 1]
            bn = coeffs[2 * n]

            cos_nt = np.cos(n * theta_rad)
            sin_nt = np.sin(n * theta_rad)

            V += an * cos_nt + bn * sin_nt
            dV += -n * an * sin_nt + n * bn * cos_nt
            d2V += - (n**2) * an * cos_nt - (n**2) * bn * sin_nt

        return V, dV, d2V

    @classmethod
    def analyze_barrier(
        cls,
        angles_deg: np.ndarray,
        energies_kcal: np.ndarray,
        num_harmonics: int = 6,
    ) -> Dict[str, float]:
        """
        Extract barrier height V_barrier and transition state locations.
        """
        angles_rad = np.radians(angles_deg)
        coeffs, _ = cls.fit_fourier_series(angles_rad, energies_kcal, num_harmonics=num_harmonics)

        fine_grid = np.linspace(0, 2 * math.pi, 360, endpoint=False)
        V_fine, dV_fine, _ = cls.evaluate_fourier_series(fine_grid, coeffs, num_harmonics=num_harmonics)

        min_val = float(np.min(V_fine))
        max_val = float(np.max(V_fine))
        barrier = max_val - min_val

        return {
            "min_energy_kcal_mol": min_val,
            "max_energy_kcal_mol": max_val,
            "barrier_height_kcal_mol": barrier,
        }
