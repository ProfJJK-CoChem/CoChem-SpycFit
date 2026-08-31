"""
CoChem Semi-Empirical xTB Pre-Scanner & Gaussian RBF Hybrid PES.

Provenance:
- Method Matrix [M]: Multi-fidelity PES interpolation, thermal window 4.0 kcal/mol for stationary minima (REQ-TORQ-049).
- Derived Relations [D]: Exact Gaussian RBF residual learning anchors V_hybrid(phi_m) = E_DFT(phi_m).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import StationaryPointExtractionError


class HybridPesScanResult(BaseModel):
    """Data contract representing the multi-fidelity xTB/DFT hybrid PES and anchor points."""

    dihedral_indices: List[Tuple[int, int, int, int]]
    xtb_grid_points: int
    extracted_minima_count: int
    dft_single_points: List[Dict[str, float]]
    rbf_gamma: float
    max_energy_barrier_kcal_mol: float


class GaussianRbfResidualLearner:
    """
    [REQ-TORQ-049] Exact Gaussian Radial Basis Function (RBF) residual learner.

    Solves weights w such that:
        K * w = delta
        where K_{ij} = exp(-gamma * ||phi_i - phi_j||^2)
        delta_m = E_DFT(phi_m) - V_xTB(phi_m)
    """

    def __init__(self, anchor_angles: np.ndarray, residuals: np.ndarray, gamma: float = 0.05) -> None:
        self.anchor_angles = np.asarray(anchor_angles, dtype=np.float64)  # (K, d) or (K,)
        if self.anchor_angles.ndim == 1:
            self.anchor_angles = self.anchor_angles[:, np.newaxis]
        self.residuals = np.asarray(residuals, dtype=np.float64)  # (K,)
        self.gamma = float(gamma)

        # Build kernel matrix K
        diff = self.anchor_angles[:, np.newaxis, :] - self.anchor_angles[np.newaxis, :, :]  # (K, K, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (K, K)
        k_matrix = np.exp(-self.gamma * sq_dist) + 1e-10 * np.eye(len(self.residuals))

        self.weights = np.linalg.solve(k_matrix, self.residuals)

    def predict_residual(self, angles: np.ndarray) -> np.ndarray:
        """Evaluate RBF residual correction at arbitrary angle query points."""
        q_angles = np.asarray(angles, dtype=np.float64)
        if q_angles.ndim == 1:
            q_angles = q_angles[:, np.newaxis]

        diff = q_angles[:, np.newaxis, :] - self.anchor_angles[np.newaxis, :, :]  # (N, K, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (N, K)
        k_query = np.exp(-self.gamma * sq_dist)  # (N, K)

        return np.dot(k_query, self.weights)


def extract_pes_local_minima_1d(
    angles_deg: np.ndarray,
    energies_kcal: np.ndarray,
    thermal_window_kcal_mol: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REQ-TORQ-049] Extract stationary local minima where grad=0, hessian>0 within thermal window.

    Args:
        angles_deg: 1D array of sampled angles.
        energies_kcal: 1D array of energies in kcal/mol.
        thermal_window_kcal_mol: Maximum relative energy above global minimum (default: 4.0 kcal/mol [M]).

    Returns:
        Tuple of (minima_angles, minima_energies).

    Raises:
        StationaryPointExtractionError: If no local minima are identified.
    """
    n = len(energies_kcal)
    if n < 3:
        raise StationaryPointExtractionError(
            grid_size=n,
            energy_range_kcal_mol=float(energies_kcal.max() - energies_kcal.min()) if n > 0 else 0.0,
        )

    min_indices = []
    # Identify interior local minima
    for i in range(1, n - 1):
        if energies_kcal[i] < energies_kcal[i - 1] and energies_kcal[i] < energies_kcal[i + 1]:
            min_indices.append(i)

    # Periodic boundaries or endpoints check
    if energies_kcal[0] < energies_kcal[1] and energies_kcal[0] < energies_kcal[-1]:
        min_indices.append(0)
    if energies_kcal[-1] < energies_kcal[-2] and energies_kcal[-1] < energies_kcal[0]:
        min_indices.append(n - 1)

    if not min_indices:
        # Fallback to index of absolute minimum if monotonic
        min_indices = [int(np.argmin(energies_kcal))]

    min_energies = energies_kcal[min_indices]
    min_angles = angles_deg[min_indices]

    global_min_e = np.min(min_energies)
    rel_energies = min_energies - global_min_e

    # Filter within thermal window (Delta E <= 4.0 kcal/mol)
    mask = rel_energies <= thermal_window_kcal_mol
    filtered_angles = min_angles[mask]
    filtered_energies = min_energies[mask]

    if len(filtered_angles) == 0:
        raise StationaryPointExtractionError(
            grid_size=n,
            energy_range_kcal_mol=float(energies_kcal.max() - energies_kcal.min()),
        )

    return filtered_angles, filtered_energies


def run_hybrid_xtb_dft_pes_pipeline(
    dihedral_indices: List[Tuple[int, int, int, int]],
    xtb_angles_deg: np.ndarray,
    xtb_energies_kcal: np.ndarray,
    dft_single_point_eval_func: Callable[[float], float],
    thermal_window_kcal_mol: float = 4.0,
    rbf_gamma: float = 0.05,
) -> Tuple[HybridPesScanResult, Callable[[np.ndarray], np.ndarray]]:
    """
    [REQ-TORQ-049] Execute 3-stage multi-fidelity xTB + DFT Gaussian RBF hybrid pipeline.

    Args:
        dihedral_indices: 4-tuple atom indices defining rotating dihedrals.
        xtb_angles_deg: Sampled angles on baseline xTB grid.
        xtb_energies_kcal: Baseline xTB energies in kcal/mol.
        dft_single_point_eval_func: Function evaluating high-level DFT single point energy at angle phi.
        thermal_window_kcal_mol: Thermal window cutoff (default 4.0 kcal/mol).
        rbf_gamma: Gaussian RBF kernel hyperparameter.

    Returns:
        Tuple of (HybridPesScanResult, hybrid_pes_eval_func).
    """
    min_angles, min_xtb_energies = extract_pes_local_minima_1d(
        angles_deg=xtb_angles_deg,
        energies_kcal=xtb_energies_kcal,
        thermal_window_kcal_mol=thermal_window_kcal_mol,
    )

    dft_points: List[Dict[str, float]] = []
    dft_energies = []
    residuals = []

    for angle, xtb_e in zip(min_angles, min_xtb_energies):
        dft_e = dft_single_point_eval_func(float(angle))
        res = dft_e - xtb_e
        dft_points.append({
            "dihedral_angle_deg": float(angle),
            "xtb_energy_kcal_mol": float(xtb_e),
            "dft_energy_kcal_mol": float(dft_e),
            "residual_kcal_mol": float(res),
        })
        dft_energies.append(dft_e)
        residuals.append(res)

    learner = GaussianRbfResidualLearner(
        anchor_angles=min_angles,
        residuals=np.array(residuals, dtype=np.float64),
        gamma=rbf_gamma,
    )

    # Construct hybrid function
    def evaluate_hybrid_pes(query_angles: np.ndarray) -> np.ndarray:
        # Interpolate baseline xTB at query angles
        v_xtb = np.interp(query_angles, xtb_angles_deg, xtb_energies_kcal)
        rbf_correction = learner.predict_residual(query_angles)
        return v_xtb + rbf_correction

    barrier = float(xtb_energies_kcal.max() - xtb_energies_kcal.min())

    result = HybridPesScanResult(
        dihedral_indices=dihedral_indices,
        xtb_grid_points=len(xtb_angles_deg),
        extracted_minima_count=len(min_angles),
        dft_single_points=dft_points,
        rbf_gamma=rbf_gamma,
        max_energy_barrier_kcal_mol=barrier,
    )

    return result, evaluate_hybrid_pes
