"""
CoChem Dynamic Adaptive Step-Size TORQ Scanner.

Provenance:
- Method Matrix [M]: Dynamic Bondi-Mantina vdW radii, topological graph distance filtering d_graph >= 4 (REQ-TORQ-044).
- Derived Relations [D]: C^1 continuous quadratic step scheduling algorithm, Cubic Hermite spline / Pchip interpolation.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.interpolate import PchipInterpolator

from cochem.core.cochem_elements import get_vdw_radius
from cochem.core.cochem_exceptions import StericGridLockError


class StericOverlapReport(BaseModel):
    """Data contract reporting steric overlap metrics and adaptive torsional step allocation."""

    dihedral_angle_deg: float
    min_steric_ratio: float
    clashing_atom_pair: Tuple[int, int]
    allocated_step_deg: float
    regime: Literal["unconstrained", "repulsion_onset", "severe_clash"]


def compute_nonbonded_steric_ratio(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    bonded_pairs_exclude: Set[Tuple[int, int]],
    fragment_a_indices: Optional[Set[int]] = None,
    fragment_b_indices: Optional[Set[int]] = None,
) -> Tuple[float, Tuple[int, int]]:
    """
    [REQ-TORQ-044] Calculate the minimum non-bonded steric ratio across rotating fragments.

    Formula:
        rho_steric = min_{i in A, j in B, d_graph(i, j) >= 4} ( r_ij / (R_vdW,i + R_vdW,j) )

    Args:
        symbols: Chemical element symbols for all atoms.
        coordinates_angstrom: Cartesian coordinates per atom.
        bonded_pairs_exclude: Set of atom index pairs (i, j) with d_graph < 4 to exclude.
        fragment_a_indices: Atom indices in Fragment A.
        fragment_b_indices: Atom indices in Fragment B.

    Returns:
        Tuple of (min_steric_ratio, (clashing_i, clashing_j)).
    """
    coords = np.array(coordinates_angstrom, dtype=np.float64)
    n_atoms = len(symbols)
    vdw_radii = np.array([get_vdw_radius(s) for s in symbols], dtype=np.float64)

    min_ratio = float("inf")
    clashing_pair = (0, 1 if n_atoms > 1 else 0)

    set_a = fragment_a_indices if fragment_a_indices is not None else set(range(n_atoms))
    set_b = fragment_b_indices if fragment_b_indices is not None else set(range(n_atoms))

    for i in set_a:
        for j in set_b:
            if i >= j and fragment_a_indices is None:
                continue
            if i == j:
                continue

            pair = (min(i, j), max(i, j))
            if pair in bonded_pairs_exclude:
                continue

            dist = float(np.linalg.norm(coords[i] - coords[j]))
            sum_vdw = vdw_radii[i] + vdw_radii[j]
            ratio = dist / sum_vdw if sum_vdw > 0 else 1.0

            if ratio < min_ratio:
                min_ratio = ratio
                clashing_pair = (i, j)

    if min_ratio == float("inf"):
        min_ratio = 1.0

    return min_ratio, clashing_pair


def schedule_adaptive_torsional_step(
    dihedral_angle_deg: float,
    min_steric_ratio: float,
    clashing_atom_pair: Tuple[int, int],
    delta_coarse_deg: float = 15.0,
    delta_fine_deg: float = 2.5,
    delta_micro_deg: float = 1.0,
    min_step_floor_deg: float = 0.1,
) -> StericOverlapReport:
    """
    [REQ-TORQ-044] Continuous C^1 quadratic step scheduling based on steric ratio rho.

    Regimes:
        - Unconstrained (rho >= 0.85): Delta theta = Delta theta_coarse (15.0°)
        - Repulsion Onset (0.65 <= rho < 0.85):
            Delta theta(rho) = Delta theta_fine + (Delta theta_coarse - Delta theta_fine) * ((rho - 0.65)/0.20)^2
        - Severe Clash (rho < 0.65): Delta theta = Delta theta_micro (1.0°)

    Raises:
        StericGridLockError: If allocated step < min_step_floor_deg (0.1°).

    Returns:
        StericOverlapReport model.
    """
    rho = min_steric_ratio

    if rho >= 0.85:
        regime: Literal["unconstrained", "repulsion_onset", "severe_clash"] = "unconstrained"
        step = delta_coarse_deg
    elif rho >= 0.65:
        regime = "repulsion_onset"
        fraction = (rho - 0.65) / 0.20
        step = delta_fine_deg + (delta_coarse_deg - delta_fine_deg) * (fraction ** 2)
    else:
        regime = "severe_clash"
        step = delta_micro_deg

    if step < min_step_floor_deg:
        raise StericGridLockError(
            dihedral_angle_deg=dihedral_angle_deg,
            step_deg=step,
            min_steric_ratio=rho,
        )

    return StericOverlapReport(
        dihedral_angle_deg=dihedral_angle_deg,
        min_steric_ratio=rho,
        clashing_atom_pair=clashing_atom_pair,
        allocated_step_deg=step,
        regime=regime,
    )


def interpolate_torsional_pes(
    angles_deg: Sequence[float],
    energies: Sequence[float],
    eval_grid_deg: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    [REQ-TORQ-044] Interpolate non-uniform torsional PES using monotonic cubic Hermite splines (PCHIP).

    Args:
        angles_deg: Evaluated dihedral angles in degrees.
        energies: Evaluated energies (e.g. kcal/mol or Hartree).
        eval_grid_deg: Optional dense evaluation grid in degrees.

    Returns:
        Tuple of (grid_angles, interpolated_energies).
    """
    sorted_pairs = sorted(zip(angles_deg, energies), key=lambda p: p[0])
    x_arr = np.array([p[0] for p in sorted_pairs], dtype=np.float64)
    y_arr = np.array([p[1] for p in sorted_pairs], dtype=np.float64)

    # Remove duplicate angles if any
    unique_x, unique_indices = np.unique(x_arr, return_index=True)
    unique_y = y_arr[unique_indices]

    interpolator = PchipInterpolator(unique_x, unique_y)

    if eval_grid_deg is None:
        grid_x = np.linspace(unique_x.min(), unique_x.max(), 361)
    else:
        grid_x = np.array(eval_grid_deg, dtype=np.float64)

    grid_y = interpolator(grid_x)
    return grid_x, grid_y
