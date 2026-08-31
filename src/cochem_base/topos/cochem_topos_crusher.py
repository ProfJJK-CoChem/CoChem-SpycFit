"""
TOPOS Heavy-Atom Kabsch RMSD Alignment and Boltzmann-Weighted Conformer Crusher.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

from cochem_base.core.models import MolecularTopology, ConformerRecord
from cochem_base.core.glossary import UnitConversionConstants


class ToposCrusher:
    """
    Deduplicates conformer ensembles using optimal rigid 3D Kabsch superposition.
    """

    @classmethod
    def kabsch_rmsd(
        cls,
        coords_a_ang: np.ndarray,
        coords_b_ang: np.ndarray,
        heavy_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute optimal heavy-atom superposition and RMSD between two geometries.
        """
        P = np.array(coords_a_ang, dtype=float)
        Q = np.array(coords_b_ang, dtype=float)

        if heavy_mask is not None and np.any(heavy_mask):
            P_sub = P[heavy_mask]
            Q_sub = Q[heavy_mask]
        else:
            P_sub = P
            Q_sub = Q

        # Center structures
        com_p = np.mean(P_sub, axis=0)
        com_q = np.mean(Q_sub, axis=0)
        P_cent = P_sub - com_p
        Q_cent = Q_sub - com_q

        # Cross-covariance matrix H = P^T Q
        H = P_cent.T @ Q_cent

        # SVD
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])

        # Rotation matrix R = V @ S_sign @ U^T
        R = Vt.T @ sign_matrix @ U.T

        # Rotate entire structure P
        P_all_cent = P - com_p
        P_rot = (R @ P_all_cent.T).T + com_q

        # Compute RMSD on aligned subset
        P_sub_rot = (R @ P_cent.T).T
        diff = P_sub_rot - Q_cent
        rmsd = float(math.sqrt(np.mean(np.sum(diff**2, axis=1))))

        return rmsd, P_rot

    @classmethod
    def cluster_conformers(
        cls,
        conformers: List[ConformerRecord],
        rmsd_threshold_angstrom: float = 0.25,
        energy_window_kcal_mol: float = 6.0,
        temperature_k: float = 298.15,
    ) -> List[ConformerRecord]:
        """
        Deduplicate and cluster conformers within an energy window.
        """
        if not conformers:
            return []

        # Sort by relative energy
        sorted_conf = sorted(conformers, key=lambda c: c.relative_energy_kcal_mol)
        min_e = sorted_conf[0].relative_energy_kcal_mol

        # 1. Filter by energy window
        valid_conf = [c for c in sorted_conf if (c.relative_energy_kcal_mol - min_e) <= energy_window_kcal_mol]
        if not valid_conf:
            return []

        # 2. Greedy Kabsch clustering
        unique_centroids: List[ConformerRecord] = []

        for candidate in valid_conf:
            cand_coords = np.array(candidate.topology.get_coordinates_angstrom(), dtype=float)
            heavy_mask = np.array([z > 1 for z in candidate.topology.atomic_numbers], dtype=bool)

            is_duplicate = False
            for centroid in unique_centroids:
                cent_coords = np.array(centroid.topology.get_coordinates_angstrom(), dtype=float)
                rmsd, _ = cls.kabsch_rmsd(cand_coords, cent_coords, heavy_mask)
                if rmsd < rmsd_threshold_angstrom:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_centroids.append(candidate)

        # 3. Recalculate Boltzmann weights
        R_cal = UnitConversionConstants.GAS_CONSTANT_CAL_MOL_K
        RT_kcal = (R_cal * temperature_k) / 1000.0

        rel_energies = [c.relative_energy_kcal_mol - min_e for c in unique_centroids]
        unnormalized_weights = [math.exp(-dE / RT_kcal) for dE in rel_energies]
        total_weight = sum(unnormalized_weights)

        final_records: List[ConformerRecord] = []
        global_min_coords = np.array(unique_centroids[0].topology.get_coordinates_angstrom(), dtype=float)
        global_heavy_mask = np.array([z > 1 for z in unique_centroids[0].topology.atomic_numbers], dtype=bool)

        for idx, c in enumerate(unique_centroids):
            c_coords = np.array(c.topology.get_coordinates_angstrom(), dtype=float)
            rmsd_to_min, _ = cls.kabsch_rmsd(c_coords, global_min_coords, global_heavy_mask)
            
            record = ConformerRecord(
                conformer_id=c.conformer_id,
                topology=c.topology,
                relative_energy_kcal_mol=round(rel_energies[idx], 4),
                boltzmann_weight=round(unnormalized_weights[idx] / total_weight, 6),
                rmsd_to_minimum_angstrom=round(rmsd_to_min, 4),
            )
            final_records.append(record)

        return final_records
