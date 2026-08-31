"""
CoChem Distributed Sayvetz/Eckart 6N Numerical Frequency Engine.

Provenance:
- Method Matrix [M]: 6N Cartesian displacements (delta = 0.005 Å), non-Hermitian gradient asymmetry gate (Delta_asym < 1e-4),
  dynamic Mendeleev mass-weighting, Sayvetz-Eckart orthogonal projector, and .hess serialization (REQ-DIST-060).
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Optional, Sequence, Tuple
import numpy as np

from cochem.core.cochem_elements import batch_get_masses
from cochem.core.models import (
    AsymmetricNumericalHessianError,
    DistributedNumFreqTask,
)

# Conversion constant from (Hartree / (Bohr^2 * amu)) or (mdyn / (Å * amu)) to cm^-1
# Standard conversion: 1 Hartree / (Bohr^2 * amu) = 219474.63 cm^-1 (approx factor: 5140.48 cm^-1 / sqrt(a.u.))
# For Hessian in a.u. / amu: nu (cm^-1) = 5140.485 * sqrt(lambda)
HESSIAN_CONVERSION_FACTOR_CM1 = 5140.485


def generate_6n_cartesian_displacements(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    delta_angstrom: float = 0.005,
    scratch_directory: str = "/tmp/cochem_scratch",
) -> List[DistributedNumFreqTask]:
    """
    [REQ-DIST-060] Generate 6N central-difference Cartesian displacement tasks for a molecular structure.

    Formula:
        R_{k, alpha}^(+/-) = R_0 +/- delta * e_{k, alpha}
        for atom k in 0..N-1, axis alpha in {x, y, z}.

    Args:
        symbols: Sequence of chemical element symbols.
        coordinates_angstrom: Equilibrium Cartesian geometry R_0 in Å.
        delta_angstrom: Central-difference displacement step size in Å (default: 0.005 Å).
        scratch_directory: Local NVMe scratch path for worker execution.

    Returns:
        List of 6N DistributedNumFreqTask instances.
    """
    tasks: List[DistributedNumFreqTask] = []
    axes: List[Literal["x", "y", "z"]] = ["x", "y", "z"]

    for atom_idx, sym in enumerate(symbols):
        for axis_name in axes:
            for direction in (+1, -1):
                dir_str = "pos" if direction == 1 else "neg"
                job_id = f"numfreq_{sym}_{atom_idx}_{axis_name}_{dir_str}"
                tasks.append(
                    DistributedNumFreqTask(
                        job_id=job_id,
                        atom_index=atom_idx,
                        cartesian_axis=axis_name,
                        direction=direction,  # type: ignore[arg-type]
                        displacement_angstrom=delta_angstrom,
                        scratch_directory=scratch_directory,
                        status="PENDING",
                    )
                )

    return tasks


def reconstruct_hessian_from_gradients(
    symbols: Sequence[str],
    gradients_plus: Dict[Tuple[int, str], np.ndarray],
    gradients_minus: Dict[Tuple[int, str], np.ndarray],
    delta_angstrom: float = 0.005,
) -> np.ndarray:
    """
    [REQ-DIST-060] Reconstruct 3N x 3N Cartesian Hessian from scattered central-difference gradient responses.

    Formula:
        H_{ia, jb} = (g_{ia}(jb+) - g_{ia}(jb-)) / (2 * delta)

    Args:
        symbols: Chemical element symbols.
        gradients_plus: Mapping of (displaced_atom_index, axis) -> gradient (N_atoms, 3).
        gradients_minus: Mapping of (displaced_atom_index, axis) -> gradient (N_atoms, 3).
        delta_angstrom: Step size delta in Å.

    Returns:
        (3N, 3N) Cartesian Hessian matrix.
    """
    n_atoms = len(symbols)
    dim = 3 * n_atoms
    hessian = np.zeros((dim, dim), dtype=np.float64)
    axis_map = {"x": 0, "y": 1, "z": 2}

    for atom_j in range(n_atoms):
        for axis_j_name, axis_j_idx in axis_map.items():
            col_idx = 3 * atom_j + axis_j_idx
            key = (atom_j, axis_j_name)

            g_plus = gradients_plus[key].reshape(dim)
            g_minus = gradients_minus[key].reshape(dim)

            # Central difference column
            hessian[:, col_idx] = (g_plus - g_minus) / (2.0 * delta_angstrom)

    return hessian


def validate_hessian_symmetry(
    hessian: np.ndarray,
    threshold: float = 1.0e-4,
) -> float:
    """
    [REQ-DIST-060] Non-Hermitian gradient asymmetry validation gate.

    Formula:
        Delta_asym = max |H - H^T| / max |H|

    Args:
        hessian: (3N, 3N) raw Cartesian Hessian.
        threshold: Maximum allowable asymmetry ratio (default: 1.0e-4).

    Returns:
        Calculated Delta_asym ratio.

    Raises:
        AsymmetricNumericalHessianError: If Delta_asym >= threshold.
    """
    max_h = float(np.max(np.abs(hessian)))
    if max_h < 1e-12:
        return 0.0

    asym_matrix = np.abs(hessian - hessian.T)
    max_asym = float(np.max(asym_matrix))
    delta_asym = max_asym / max_h

    if delta_asym >= threshold:
        raise AsymmetricNumericalHessianError(
            delta_asymmetry=round(delta_asym, 6),
            threshold=threshold,
        )

    return delta_asym


def mass_weight_hessian(
    symbols: Sequence[str],
    hessian: np.ndarray,
) -> np.ndarray:
    """
    [REQ-DIST-060] Symmetrize and dynamically mass-weight Cartesian Hessian.

    Formula:
        H_sym = 0.5 * (H + H^T)
        M_{ia, jb} = H_sym_{ia, jb} / sqrt(m_i * m_j)

    Args:
        symbols: Chemical element symbols.
        hessian: (3N, 3N) Cartesian Hessian.

    Returns:
        (3N, 3N) mass-weighted Hessian matrix M.
    """
    h_sym = 0.5 * (hessian + hessian.T)
    masses = batch_get_masses(list(symbols))
    n_atoms = len(symbols)

    # Expand masses to 3N vector: [m0, m0, m0, m1, m1, m1, ...]
    expanded_masses = np.repeat(masses, 3)
    sqrt_m = np.sqrt(expanded_masses)

    # Outer product for 2D broadcast
    m_denom = np.outer(sqrt_m, sqrt_m)
    m_weighted = h_sym / m_denom

    return m_weighted


def construct_sayvetz_eckart_projector(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REQ-DIST-060] Construct Sayvetz/Eckart translational and rotational projection operator P.

    Projection operator:
        P = I_3N - D (D^T D)^(-1) D^T
        where D in R^(3N x 6) spans mass-weighted translations and rotations about COM.

    Verifications:
        Idempotency: P^2 = P
        Orthogonality: P @ D = 0

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.

    Returns:
        Tuple of (D_matrix, P_projector).
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    masses = batch_get_masses(list(symbols))
    n_atoms = len(symbols)
    dim = 3 * n_atoms

    # Mass-weighted COM
    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    centered = coords - com

    d_basis = np.zeros((dim, 6), dtype=np.float64)

    # Translations: D[:, 0..2]
    for i in range(n_atoms):
        sm = math.sqrt(masses[i])
        d_basis[3 * i + 0, 0] = sm  # Tx
        d_basis[3 * i + 1, 1] = sm  # Ty
        d_basis[3 * i + 2, 2] = sm  # Tz

    # Rotations: D[:, 3..5] -> mass-weighted cross product: r_i x e_alpha
    for i in range(n_atoms):
        sm = math.sqrt(masses[i])
        xi, yi, zi = centered[i]

        # Rx: cross product with x-axis
        d_basis[3 * i + 1, 3] = -zi * sm
        d_basis[3 * i + 2, 3] = yi * sm

        # Ry: cross product with y-axis
        d_basis[3 * i + 0, 4] = zi * sm
        d_basis[3 * i + 2, 4] = -xi * sm

        # Rz: cross product with z-axis
        d_basis[3 * i + 0, 5] = -yi * sm
        d_basis[3 * i + 1, 5] = xi * sm

    # QR or SVD to orthogonalize basis D
    q_mat, r_mat = np.linalg.qr(d_basis)

    # Retain non-zero columns of Q (for linear molecules, 5 columns; non-linear, 6)
    r_diag = np.abs(np.diag(r_mat)) if r_mat.shape[0] >= 6 else np.abs(np.diag(r_mat[:r_mat.shape[1], :]))
    valid_cols = r_diag > 1e-10
    q_active = q_mat[:, :len(valid_cols)][:, valid_cols]

    # Projector P = I - Q Q^T
    projector = np.eye(dim) - (q_active @ q_active.T)

    return d_basis, projector


def project_and_diagonalize_hessian(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    mass_weighted_hessian: np.ndarray,
) -> Tuple[List[float], np.ndarray]:
    """
    [REQ-DIST-060] Project external translations/rotations and diagonalize to obtain harmonic frequencies.

    Formula:
        M_proj = P @ M_mw @ P^T
        M_proj v_k = lambda_k v_k
        nu_k = (1 / 2*pi*c) * sqrt(lambda_k) (cm^-1)

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        mass_weighted_hessian: (3N, 3N) mass-weighted Hessian matrix.

    Returns:
        Tuple of (frequencies_cm1, normal_modes_matrix).
    """
    _, p_proj = construct_sayvetz_eckart_projector(symbols, coordinates_angstrom)
    m_proj = p_proj @ mass_weighted_hessian @ p_proj.T

    # Symmetrize projected matrix
    m_proj_sym = 0.5 * (m_proj + m_proj.T)

    evals, evecs = np.linalg.eigh(m_proj_sym)

    frequencies_cm1: List[float] = []
    for val in evals:
        if abs(val) < 1e-8:
            frequencies_cm1.append(0.0)
        elif val > 0:
            freq = HESSIAN_CONVERSION_FACTOR_CM1 * math.sqrt(val)
            frequencies_cm1.append(round(freq, 2))
        else:
            freq = -HESSIAN_CONVERSION_FACTOR_CM1 * math.sqrt(abs(val))
            frequencies_cm1.append(round(freq, 2))

    return frequencies_cm1, evecs


def serialize_to_hess_format(
    frequencies_cm1: Sequence[float],
    normal_modes: np.ndarray,
) -> str:
    """
    [REQ-DIST-060] Format frequencies and eigenvectors into standard ORCA/CFOUR .hess text block.

    Args:
        frequencies_cm1: List of harmonic frequencies in cm^-1.
        normal_modes: (3N, 3N) normal mode eigenvectors matrix.

    Returns:
        Formatted .hess block string.
    """
    dim = len(frequencies_cm1)
    lines: List[str] = [
        "$vibrational_frequencies",
        str(dim),
    ]

    for idx, f in enumerate(frequencies_cm1):
        lines.append(f"{idx:4d}  {f:12.4f}")

    lines.append("")
    lines.append("$normal_modes")
    lines.append(f"{dim} {dim}")

    # Column indices
    col_hdr = "    " + " ".join(f"{c:8d}" for c in range(min(dim, 5)))
    lines.append(col_hdr)

    for row_idx in range(dim):
        row_vals = " ".join(f"{normal_modes[row_idx, col_idx]:8.4f}" for col_idx in range(min(dim, 5)))
        lines.append(f"{row_idx:4d}  {row_vals}")

    return "\n".join(lines) + "\n"
