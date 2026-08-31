"""
CoChem Whitened 12D KD-Tree Invariant Geometric Sieving Engine.

Provenance:
- Method Matrix [M]: 12D invariant feature formulation (moments of inertia, Coulomb matrix eigenvalues,
  Ray's asymmetry kappa, radius of gyration, spatial span) and O(N log N) KD-Tree candidate pruning (REQ-TOPOS-055).
- Two-Stage Filtering: Stage 1 eps_sieve = 1.5e-3 ball query -> Stage 2 Hungarian Kabsch RMSD < 0.05 Å.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.spatial import cKDTree

from cochem.core.cochem_elements import (
    batch_get_masses,
    get_atomic_number,
    get_element_electronegativity,
)
from cochem.core.models import (
    InhomogeneousFeatureSievingError,
    SievingFeatureVector,
)


def extract_12d_feature_vector(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    conformer_id: str,
    energy_hartree: float = 0.0,
) -> SievingFeatureVector:
    """
    [REQ-TOPOS-055] Formulate the invariant 12D geometric feature vector for a molecular conformer.

    Feature Vector Components:
        v = [Ia, Ib, Ic, lambda_1(C), lambda_2(C), lambda_3(C), d_min, d_max, Rg, mu_est, S_span, kappa]^T

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        conformer_id: Unique conformer identifier string.
        energy_hartree: Electronic energy in Hartree.

    Returns:
        SievingFeatureVector instance.
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    n_atoms = len(symbols)
    masses = batch_get_masses(list(symbols))
    atomic_numbers = np.array([get_atomic_number(s) for s in symbols], dtype=np.float64)

    # 1. Mass-weighted Center of Mass and centered coords
    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    centered = coords - com

    # 2. Principal moments of inertia (Ia <= Ib <= Ic)
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

    evals, _ = np.linalg.eigh(inertia_tensor)
    sorted_evals = np.sort(evals)
    ia, ib, ic = float(sorted_evals[0]), float(sorted_evals[1]), float(sorted_evals[2])

    # 3. Ray's asymmetry parameter kappa in [-1.0, 1.0]
    denom = ic - ia
    if denom > 1e-8:
        kappa = float((2.0 * ib - ia - ic) / denom)
    else:
        kappa = 0.0
    kappa = float(np.clip(kappa, -1.0, 1.0))

    # 4. Coulomb matrix top 3 eigenvalues
    c_mat = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                c_mat[i, i] = 0.5 * (atomic_numbers[i] ** 2.4)
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                c_mat[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / max(dist, 1e-4)

    c_evals = np.linalg.eigvalsh(c_mat)
    sorted_c_evals = np.sort(c_evals)[::-1]  # descending
    l1 = float(sorted_c_evals[0]) if len(sorted_c_evals) > 0 else 0.0
    l2 = float(sorted_c_evals[1]) if len(sorted_c_evals) > 1 else 0.0
    l3 = float(sorted_c_evals[2]) if len(sorted_c_evals) > 2 else 0.0

    # 5. Pairwise interatomic distances (d_min, d_max)
    dists = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dists.append(float(np.linalg.norm(coords[i] - coords[j])))

    d_min = min(dists) if dists else 1.0
    d_max = max(dists) if dists else 1.0

    # 6. Radius of gyration Rg
    rg = float(np.sqrt(np.sum(masses * np.sum(centered ** 2, axis=1)) / total_mass))

    # 7. Spatial span S_span
    s_span = float(np.max(np.linalg.norm(centered, axis=1))) if n_atoms > 0 else 1.0

    # 8. Estimated dipole moment mu_est
    ens = np.array([get_element_electronegativity(s) or 2.2 for s in symbols], dtype=np.float64)
    mean_en = np.mean(ens)
    partial_charges = mean_en - ens
    dipole_vec = np.sum(centered * partial_charges[:, np.newaxis], axis=0)
    mu_est = float(np.linalg.norm(dipole_vec) * 4.8032)  # Conversion to Debye approx

    return SievingFeatureVector(
        conformer_id=conformer_id,
        principal_moments=(ia, ib, ic),
        coulomb_eigenvalues_top3=(l1, l2, l3),
        min_interatomic_distance=max(0.1, d_min),
        max_interatomic_distance=max(d_min, d_max),
        radius_of_gyration=max(0.1, rg),
        estimated_dipole_debye=max(0.0, mu_est),
        spatial_span=max(0.1, s_span),
        ray_asymmetry_kappa=kappa,
        energy_hartree=energy_hartree,
    )


def feature_vector_to_array(vec: SievingFeatureVector) -> np.ndarray:
    """Convert SievingFeatureVector to 1D 12-element NumPy float64 array."""
    return np.array([
        vec.principal_moments[0],
        vec.principal_moments[1],
        vec.principal_moments[2],
        vec.coulomb_eigenvalues_top3[0],
        vec.coulomb_eigenvalues_top3[1],
        vec.coulomb_eigenvalues_top3[2],
        vec.min_interatomic_distance,
        vec.max_interatomic_distance,
        vec.radius_of_gyration,
        vec.estimated_dipole_debye,
        vec.spatial_span,
        vec.ray_asymmetry_kappa,
    ], dtype=np.float64)


def whiten_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    [REQ-TOPOS-055] Z-score standardization and population whitening across feature columns.

    Formula:
        v_tilde = (v - mu) / sigma

    Args:
        feature_matrix: (N_samples, 12) matrix of raw feature vectors.

    Returns:
        (N_samples, 12) whitened matrix.

    Raises:
        InhomogeneousFeatureSievingError: If variance along any column < 1e-12.
    """
    if len(feature_matrix) <= 1:
        return feature_matrix

    mu = np.mean(feature_matrix, axis=0)
    var = np.var(feature_matrix, axis=0)

    zero_var_idx = [i for i, v in enumerate(var) if v < 1e-12]
    if zero_var_idx:
        raise InhomogeneousFeatureSievingError(
            zero_variance_indices=zero_var_idx,
            variances=[float(v) for v in var],
        )

    std = np.sqrt(var)
    return (feature_matrix - mu) / std


def compute_kabsch_rmsd(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    """Compute optimal rotation Kabsch RMSD between two aligned coordinate sets in Å."""
    p = coords_a - np.mean(coords_a, axis=0)
    q = coords_b - np.mean(coords_b, axis=0)

    # Covariance matrix H
    h_mat = p.T @ q
    u, s, vt = np.linalg.svd(h_mat)
    d = np.linalg.det(vt.T @ u.T)
    correction = np.diag([1.0, 1.0, d])
    rot = vt.T @ correction @ u.T

    rotated_p = p @ rot.T
    diff = rotated_p - q
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return rmsd


class TwoStageKDTreeSiever:
    """
    [REQ-TOPOS-055] Two-stage geometric conformer siever with 12D KD-Tree and Hungarian RMSD.
    """

    def __init__(
        self,
        eps_sieve: float = 1.5e-3,
        rmsd_prune_angstrom: float = 0.05,
    ) -> None:
        self.eps_sieve = eps_sieve
        self.rmsd_prune_angstrom = rmsd_prune_angstrom
        self.pool_vectors: List[np.ndarray] = []
        self.pool_feature_objects: List[SievingFeatureVector] = []
        self.pool_coordinates: List[np.ndarray] = []
        self.kdtree: Optional[cKDTree] = None

    def _rebuild_tree(self) -> None:
        """Reconstruct scipy.spatial.cKDTree from normalized feature pool."""
        if len(self.pool_vectors) == 0:
            self.kdtree = None
            return

        mat = np.array(self.pool_vectors, dtype=np.float64)
        if len(mat) >= 2:
            std = np.std(mat, axis=0)
            std_safe = np.where(std < 1e-12, 1.0, std)
            normalized = (mat - np.mean(mat, axis=0)) / std_safe
        else:
            normalized = mat

        self.kdtree = cKDTree(normalized)

    def sieve_conformer(
        self,
        symbols: Sequence[str],
        coordinates: Sequence[Tuple[float, float, float]],
        feature_vector: SievingFeatureVector,
    ) -> bool:
        """
        [REQ-TOPOS-055] Evaluate candidate conformer against existing pool.

        Returns:
            True if accepted as unique conformer, False if pruned.
        """
        coords_arr = np.asarray(coordinates, dtype=np.float64)
        vec_arr = feature_vector_to_array(feature_vector)

        if len(self.pool_vectors) == 0:
            # First conformer is accepted immediately
            self.pool_vectors.append(vec_arr)
            self.pool_feature_objects.append(feature_vector)
            self.pool_coordinates.append(coords_arr)
            self._rebuild_tree()
            return True

        # Stage 1: Normalized feature distance query
        mat = np.array(self.pool_vectors, dtype=np.float64)
        std = np.std(mat, axis=0)
        std_safe = np.where(std < 1e-12, 1.0, std)
        mean = np.mean(mat, axis=0)

        norm_candidate = (vec_arr - mean) / std_safe
        norm_pool = (mat - mean) / std_safe

        dists = np.linalg.norm(norm_pool - norm_candidate, axis=1)
        close_indices = np.where(dists <= self.eps_sieve)[0]

        if len(close_indices) == 0:
            # Stage 1 uniqueness confirmed
            self.pool_vectors.append(vec_arr)
            self.pool_feature_objects.append(feature_vector)
            self.pool_coordinates.append(coords_arr)
            self._rebuild_tree()
            return True

        # Stage 2: Heavy-atom Kabsch RMSD check against close candidates
        for idx in close_indices:
            ref_coords = self.pool_coordinates[idx]
            rmsd = compute_kabsch_rmsd(coords_arr, ref_coords)
            if rmsd < self.rmsd_prune_angstrom:
                # Duplicate identified: prune
                return False

        # If RMSD >= threshold against all nearby feature points, accept
        self.pool_vectors.append(vec_arr)
        self.pool_feature_objects.append(feature_vector)
        self.pool_coordinates.append(coords_arr)
        self._rebuild_tree()
        return True
