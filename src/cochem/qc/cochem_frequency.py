"""
CoChem Sayvetz-Eckart Rotational/Translational Projection Engine (REQ-QC-090).

Provenance:
- Method Matrix [M]: Sayvetz-Eckart coordinate projector for mass-weighted Cartesian Hessians,
  Gram-Schmidt / QR orthogonalization of 6 rigid-body translational/rotational modes.
- Derived Relations [D]: Dynamic Mendeleev atomic masses (IUPAC CIAAW), CODATA 2022 constants.
- Empirical [E]: Strict numerical suppression of non-vibrational zero-modes (|omega_k| <= 1.0e-4 cm^-1).
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cochem.core.cochem_elements import get_ciaaw_atomic_mass
from cochem.qc.models import ProjectedFrequencyResult

# Physical Constants
AMU_TO_AU: float = 1822.888486209
ANGSTROM_TO_BOHR: float = 1.88972612462577
AU_TO_CM1: float = 219474.63136320
ZERO_MODE_TOLERANCE_CM1: float = 1.0e-4


class ProjectedVibrationalSpectrum(BaseModel):
    """Data contract representing projected harmonic vibrational spectrum."""
    model_config = ConfigDict(frozen=True)

    unprojected_frequencies_cm1: List[float]
    projected_frequencies_cm1: List[float]
    zero_mode_residuals_cm1: List[float]
    num_zero_modes: int = 6
    is_linear: bool = False
    is_clean: bool = True


class SayvetzEckartProjectionResult(BaseModel):
    """Data contract representing Sayvetz-Eckart projector diagnostics."""
    model_config = ConfigDict(frozen=True)

    projected_frequencies_cm1: List[float]
    rot_trans_residual_max_cm1: float
    is_projection_clean: bool
    num_projected_zero_modes: int


def project_sayvetz_eckart_hessian(
    symbols: Sequence[str],
    coordinates_angstrom: Union[np.ndarray, Sequence[Sequence[float]]],
    cartesian_hessian_hartree_bohr2: Union[np.ndarray, Sequence[Sequence[float]]],
    is_linear: bool = False,
) -> ProjectedFrequencyResult:
    """
    [M][D][E] Project translational and rotational rigid-body modes from mass-weighted Cartesian Hessian (REQ-QC-090).

    Sayvetz-Eckart Projection Algorithm:
        1. Dynamically retrieve CIAAW standard atomic weights m_i via mendeleev.
        2. Center Cartesian coordinates at molecular Center-of-Mass (COM):
           r_COM = sum(m_i * r_i) / sum(m_i),  r'_i = r_i - r_COM
        3. Assemble 3 translational basis vectors D_x, D_y, D_z in mass-weighted space:
           D_a,3i+b = sqrt(m_i) * delta_ab
        4. Assemble 3 rotational basis vectors R_x, R_y, R_z:
           R_i = sqrt(m_i) * (e_a x r'_i)
        5. Form orthonormal basis Q in R^(3N x 6) (or R^(3N x 5) for linear molecules) via QR decomposition.
        6. Assemble idempotent projector:
           P = I_3N - Q * Q^T
        7. Construct projected mass-weighted Hessian:
           H_MW,proj = P * H_MW * P
        8. Diagonalize both unprojected and projected Hessians, verifying zero-mode residual <= 1.0e-4 cm^-1.

    Args:
        symbols: Elemental symbols for all N atoms.
        coordinates_angstrom: (N, 3) Cartesian coordinates in Å.
        cartesian_hessian_hartree_bohr2: (3N, 3N) Cartesian electronic Hessian in Hartree/Bohr^2.
        is_linear: Whether the molecule is linear (5 zero modes instead of 6).

    Returns:
        ProjectedFrequencyResult containing unprojected and projected frequencies, residual max, and cleanliness flag.

    Raises:
        ValueError: If array dimensions mismatch or coordinate shape != (N, 3).
    """
    n_atoms = len(symbols)
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    h_cart = np.asarray(cartesian_hessian_hartree_bohr2, dtype=np.float64)

    if coords.shape != (n_atoms, 3):
        raise ValueError(f"Coordinate shape {coords.shape} mismatch expected ({n_atoms}, 3).")
    if h_cart.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(f"Hessian shape {h_cart.shape} mismatch expected ({3 * n_atoms}, {3 * n_atoms}).")

    # 1. Dynamic atomic masses
    masses_amu = np.array([get_ciaaw_atomic_mass(s) for s in symbols], dtype=np.float64)
    masses_au = masses_amu * AMU_TO_AU

    # 2. Coordinates in Bohr and Center-of-Mass
    coords_bohr = coords * ANGSTROM_TO_BOHR
    total_mass = np.sum(masses_au)
    com_bohr = np.sum(coords_bohr * masses_au[:, np.newaxis], axis=0) / total_mass
    r_prime = coords_bohr - com_bohr

    # 3. Assemble Mass-Weighted Diagonal
    m_diag = np.repeat(masses_au, 3)
    inv_sqrt_m = 1.0 / np.sqrt(m_diag)

    # 4. Form 6 Rigid-Body Modes (Translational and Rotational)
    d_mat = np.zeros((3 * n_atoms, 6), dtype=np.float64)

    for i in range(n_atoms):
        sqrt_m = math.sqrt(masses_au[i])
        rx, ry, rz = r_prime[i]

        # Translational modes (columns 0, 1, 2)
        d_mat[3 * i + 0, 0] = sqrt_m
        d_mat[3 * i + 1, 1] = sqrt_m
        d_mat[3 * i + 2, 2] = sqrt_m

        # Rotational modes (columns 3, 4, 5): sqrt(m) * (e_a x r)
        # R_x = sqrt(m) * (0, -z, y)
        d_mat[3 * i + 1, 3] = -rz * sqrt_m
        d_mat[3 * i + 2, 3] = ry * sqrt_m

        # R_y = sqrt(m) * (z, 0, -x)
        d_mat[3 * i + 0, 4] = rz * sqrt_m
        d_mat[3 * i + 2, 4] = -rx * sqrt_m

        # R_z = sqrt(m) * (-y, x, 0)
        d_mat[3 * i + 0, 5] = -ry * sqrt_m
        d_mat[3 * i + 1, 5] = rx * sqrt_m

    # 5. QR Decomposition to produce orthonormal basis Q
    q_full, r_qr = np.linalg.qr(d_mat)

    # Determine genuine non-zero rigid-body columns
    num_zero_modes = 5 if is_linear else 6
    # Select columns corresponding to non-trivial R_diag elements
    diag_r = np.abs(np.diag(r_qr))
    valid_cols = np.argsort(-diag_r)[:num_zero_modes]
    q_basis = q_full[:, valid_cols]

    # 6. Idempotent Projector P = I - Q * Q^T
    eye_3n = np.eye(3 * n_atoms, dtype=np.float64)
    projector = eye_3n - np.dot(q_basis, q_basis.T)

    # 7. Form Mass-Weighted Hessian and Project
    h_mw = h_cart * np.outer(inv_sqrt_m, inv_sqrt_m)
    h_mw_proj = np.dot(projector, np.dot(h_mw, projector))
    # Symmetrize numerical projection
    h_mw_proj = 0.5 * (h_mw_proj + h_mw_proj.T)

    # 8. Diagonalize both Hessians
    eigenvals_raw, _ = np.linalg.eigh(h_mw)
    eigenvals_proj, _ = np.linalg.eigh(h_mw_proj)

    # Frequencies in cm^-1: imaginary frequencies represented with negative sign
    freqs_raw = np.sign(eigenvals_raw) * np.sqrt(np.abs(eigenvals_raw)) * AU_TO_CM1
    freqs_proj = np.sign(eigenvals_proj) * np.sqrt(np.abs(eigenvals_proj)) * AU_TO_CM1

    # Identify rigid-body zero modes by smallest absolute magnitude (handles transition states where lambda_TS < 0)
    abs_freqs_proj = np.abs(freqs_proj)
    zero_mode_indices = np.argsort(abs_freqs_proj)[:num_zero_modes]
    zero_mode_residuals = abs_freqs_proj[zero_mode_indices]
    max_residual = float(np.max(zero_mode_residuals)) if len(zero_mode_residuals) > 0 else 0.0
    is_clean = max_residual <= ZERO_MODE_TOLERANCE_CM1

    # Clean display frequencies: zero only the true rigid-body zero modes
    cleaned_freqs_proj = freqs_proj.copy()
    cleaned_freqs_proj[zero_mode_indices] = 0.0

    return ProjectedFrequencyResult(
        unprojected_frequencies_cm1=[round(float(f), 4) for f in freqs_raw],
        projected_frequencies_cm1=[round(float(f), 4) for f in cleaned_freqs_proj],
        rot_trans_residual_max_cm1=max_residual,
        num_projected_zero_modes=num_zero_modes,
        is_linear_molecule=is_linear,
        is_projection_clean=is_clean,
    )
