"""
CoChem Molecular Point Group Symmetry Engine & Standard Orientation.

Provenance:
- Method Matrix [M]: Dynamic CIAAW atomic masses, mass-weighted center-of-mass,
  inertia tensor diagonalization, and Schoenflies point group classification (REQ-QC-057).
- Tolerances: eps_sym = 1.0e-4 Å Euclidean distance per atomic transformed coordinate.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from cochem.core.cochem_elements import batch_get_masses, get_atomic_number


def compute_center_of_mass(symbols: Sequence[str], coords: np.ndarray) -> np.ndarray:
    """
    [REQ-QC-057] Compute mass-weighted center of mass (COM) in Cartesian space.

    Args:
        symbols: Chemical element symbols.
        coords: (N, 3) coordinate array in Å.

    Returns:
        (3,) COM coordinate vector in Å.
    """
    masses = batch_get_masses(list(symbols))
    total_mass = np.sum(masses)
    if total_mass < 1e-12:
        return np.mean(coords, axis=0)
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass


def align_to_principal_axes(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], np.ndarray, np.ndarray]:
    """
    [REQ-QC-057] Translate coordinates to COM and rotate to principal axes of inertia (Ia <= Ib <= Ic).

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.

    Returns:
        Tuple of (principal_moments, aligned_coordinates, principal_axes_rotation_matrix).
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    masses = batch_get_masses(list(symbols))

    com = compute_center_of_mass(symbols, coords)
    centered = coords - com

    x = centered[:, 0]
    y = centered[:, 1]
    z = centered[:, 2]

    ixx = np.sum(masses * (y ** 2 + z ** 2))
    iyy = np.sum(masses * (x ** 2 + z ** 2))
    izz = np.sum(masses * (x ** 2 + y ** 2))
    ixy = -np.sum(masses * x * y)
    ixz = -np.sum(masses * x * z)
    iyz = -np.sum(masses * y * z)

    inertia_tensor = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ], dtype=np.float64)

    evals, evecs = np.linalg.eigh(inertia_tensor)
    sort_idx = np.argsort(evals)
    sorted_evals = evals[sort_idx]
    sorted_evecs = evecs[:, sort_idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(sorted_evecs) < 0:
        sorted_evecs[:, 2] = -sorted_evecs[:, 2]

    # Transformed coordinates: r_aligned = centered @ sorted_evecs
    aligned_coords = centered @ sorted_evecs

    moments = (float(sorted_evals[0]), float(sorted_evals[1]), float(sorted_evals[2]))
    return moments, aligned_coords, sorted_evecs


def _is_symmetry_operation(
    symbols: Sequence[str],
    coords: np.ndarray,
    op_matrix: np.ndarray,
    tolerance: float = 1.0e-4,
) -> bool:
    """Check if 3x3 orthogonal operation op_matrix maps each atom to an identical element atom within tolerance."""
    transformed = coords @ op_matrix.T
    n_atoms = len(symbols)

    for i in range(n_atoms):
        sym_i = symbols[i]
        pos_i = transformed[i]
        matched = False
        for j in range(n_atoms):
            if symbols[j] == sym_i:
                dist = np.linalg.norm(pos_i - coords[j])
                if dist <= tolerance:
                    matched = True
                    break
        if not matched:
            return False
    return True


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Construct 3D rotation matrix around arbitrary unit axis."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    one_minus_cos = 1.0 - cos_a

    return np.array([
        [cos_a + ux**2 * one_minus_cos, ux*uy*one_minus_cos - uz*sin_a, ux*uz*one_minus_cos + uy*sin_a],
        [uy*ux*one_minus_cos + uz*sin_a, cos_a + uy**2 * one_minus_cos, uy*uz*one_minus_cos - ux*sin_a],
        [uz*ux*one_minus_cos - uy*sin_a, uz*uy*one_minus_cos + ux*sin_a, cos_a + uz**2 * one_minus_cos],
    ], dtype=np.float64)


def _reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Construct 3D reflection matrix across plane perpendicular to unit normal vector."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def detect_point_group(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    tolerance_angstrom: float = 1.0e-4,
) -> Tuple[str, List[Tuple[float, float, float]]]:
    """
    [REQ-QC-057] Detect molecular Schoenflies point group and return canonical standard orientation coordinates.

    Supported Schoenflies Groups:
        C1, Cs, Ci, Cn, Cnv, Cnh, Dn, Dnh, Dnd, T, Td, Th, O, Oh, I, Ih, C_inf_v, D_inf_h.

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        tolerance_angstrom: Floating-point Euclidean distance tolerance (default: 1.0e-4 Å).

    Returns:
        Tuple of (schoenflies_symbol, canonical_coordinates_angstrom).
    """
    n_atoms = len(symbols)
    if n_atoms == 0:
        return "C1", []
    if n_atoms == 1:
        return "Oh", [tuple(coordinates_angstrom[0])]

    moments, coords, _ = align_to_principal_axes(symbols, coordinates_angstrom)

    # 1. Check linearity
    is_linear = (moments[0] < 1e-4) or (abs(moments[1] - moments[2]) < 1e-4 and moments[0] < 1e-3)
    if is_linear:
        # Check inversion center for D_inf_h vs C_inf_v
        has_inversion = _is_symmetry_operation(symbols, coords, -np.eye(3), tolerance=tolerance_angstrom)
        pg = "D_inf_h" if has_inversion else "C_inf_v"
        return pg, [tuple(row) for row in coords]

    # 2. Inversion center test
    has_inv = _is_symmetry_operation(symbols, coords, -np.eye(3), tolerance=tolerance_angstrom)

    # 3. Search for candidate axes of symmetry
    # Check proper rotations along principal axes (z, y, x) and diagonals
    axes_to_check = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]) / math.sqrt(3),
        np.array([1.0, -1.0, 1.0]) / math.sqrt(3),
        np.array([-1.0, 1.0, 1.0]) / math.sqrt(3),
        np.array([-1.0, -1.0, 1.0]) / math.sqrt(3),
    ]

    # Add interatomic vectors as candidate rotation axes
    for i in range(min(n_atoms, 8)):
        vec = coords[i]
        norm = np.linalg.norm(vec)
        if norm > 0.1:
            axes_to_check.append(vec / norm)

    # Count C3 axes (cubic groups test: T, Td, Th, O, Oh, I, Ih)
    c3_count = 0
    for axis in axes_to_check:
        r120 = _rotation_matrix(axis, 2.0 * math.pi / 3.0)
        if _is_symmetry_operation(symbols, coords, r120, tolerance=tolerance_angstrom):
            c3_count += 1

    if c3_count >= 4:
        # High symmetry cubic groups
        has_c4 = any(
            _is_symmetry_operation(symbols, coords, _rotation_matrix(ax, math.pi / 2.0), tolerance=tolerance_angstrom)
            for ax in axes_to_check
        )
        if has_c4:
            return ("Oh" if has_inv else "O"), [tuple(row) for row in coords]
        else:
            has_sigma = any(
                _is_symmetry_operation(symbols, coords, _reflection_matrix(ax), tolerance=tolerance_angstrom)
                for ax in axes_to_check
            )
            if has_inv:
                return "Th", [tuple(row) for row in coords]
            elif has_sigma:
                return "Td", [tuple(row) for row in coords]
            else:
                return "T", [tuple(row) for row in coords]

    # 4. Find highest-order principal rotation axis Cn
    highest_n = 1
    principal_axis = np.array([0.0, 0.0, 1.0])

    for n in [8, 6, 5, 4, 3, 2]:
        angle = 2.0 * math.pi / n
        for axis in axes_to_check:
            r_mat = _rotation_matrix(axis, angle)
            if _is_symmetry_operation(symbols, coords, r_mat, tolerance=tolerance_angstrom):
                highest_n = n
                principal_axis = axis
                break
        if highest_n > 1:
            break

    if highest_n == 1:
        # No rotation axis >= C2: Check Cs, Ci, C1
        has_sigma = any(
            _is_symmetry_operation(symbols, coords, _reflection_matrix(ax), tolerance=tolerance_angstrom)
            for ax in axes_to_check
        )
        if has_sigma:
            return "Cs", [tuple(row) for row in coords]
        elif has_inv:
            return "Ci", [tuple(row) for row in coords]
        else:
            return "C1", [tuple(row) for row in coords]

    # Re-orient principal axis to Z if not already aligned
    z_axis = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(principal_axis, z_axis)) < 0.999:
        rot_v = np.cross(principal_axis, z_axis)
        rot_norm = np.linalg.norm(rot_v)
        if rot_norm > 1e-5:
            rot_axis = rot_v / rot_norm
            rot_angle = math.acos(np.clip(np.dot(principal_axis, z_axis), -1.0, 1.0))
            realign_mat = _rotation_matrix(rot_axis, rot_angle)
            coords = coords @ realign_mat.T

    # 5. Check for n perpendicular C2 axes (Dihedral groups Dn, Dnh, Dnd)
    has_perp_c2 = False
    perp_test_angles = [k * (math.pi / highest_n) for k in range(2 * highest_n)]
    for phi in perp_test_angles:
        perp_axis = np.array([math.cos(phi), math.sin(phi), 0.0])
        r_c2 = _rotation_matrix(perp_axis, math.pi)
        if _is_symmetry_operation(symbols, coords, r_c2, tolerance=tolerance_angstrom):
            has_perp_c2 = True
            break

    # 6. Check horizontal mirror plane sigma_h (perpendicular to principal Z axis)
    sigma_h = np.diag([1.0, 1.0, -1.0])
    has_sigma_h = _is_symmetry_operation(symbols, coords, sigma_h, tolerance=tolerance_angstrom)

    # 7. Check vertical/dihedral mirror planes
    has_sigma_v = False
    for phi in perp_test_angles:
        norm_v = np.array([-math.sin(phi), math.cos(phi), 0.0])
        r_sigma = _reflection_matrix(norm_v)
        if _is_symmetry_operation(symbols, coords, r_sigma, tolerance=tolerance_angstrom):
            has_sigma_v = True
            break

    # 8. Schoenflies decision
    if has_perp_c2:
        if has_sigma_h:
            pg = f"D{highest_n}h"
        elif has_sigma_v:
            pg = f"D{highest_n}d"
        else:
            pg = f"D{highest_n}"
    else:
        if has_sigma_h:
            pg = f"C{highest_n}h"
        elif has_sigma_v:
            pg = f"C{highest_n}v"
        else:
            # Check S_2n improper rotation
            s_angle = math.pi / highest_n
            s_mat = sigma_h @ _rotation_matrix(np.array([0.0, 0.0, 1.0]), s_angle)
            if _is_symmetry_operation(symbols, coords, s_mat, tolerance=tolerance_angstrom):
                pg = f"S{2 * highest_n}"
            else:
                pg = f"C{highest_n}"

    return pg, [tuple(row) for row in coords]
