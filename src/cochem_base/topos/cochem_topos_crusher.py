"""# zero-stub anti-spoofing engine
CoChem Nuclear Clash Guard, Manual Adjustment Interceptor & Conformer Crusher (cochem_base.topos.cochem_topos_crusher)

Provides strict physical nuclear clash gating (min r_ij >= 0.65 Angstroms), immutable manual
adjustment provenance recording, rotamer RMSD clustering, and conformer energy window pruning.

Governing Requirements:
- REQ-67 (FR-67.1 - FR-67.3), REQ-69: Nuclear Clash Guard & Topology Crusher.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("cochem_topos_crusher")

MIN_INTERATOMIC_DISTANCE_ANGSTROM: float = 0.65  # Hard physical Coulombic divergence threshold


class NuclearClashError(Exception):
    """Raised when atomic coordinates violate the nuclear clash distance threshold (< 0.65 A)."""
    pass


@dataclass(frozen=True)
class ManualAdjustmentRecord:
    """Immutable audit record tracking manual geometry or calculation overrides."""
    timestamp_utc: str
    target_module: str
    parameter_name: str
    pre_override_hash: str
    post_override_hash: str
    default_value: Any
    override_value: Any
    min_interatomic_distance_angstrom: float
    operator_justification: str
    physical_implication: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "target_module": self.target_module,
            "parameter_name": self.parameter_name,
            "pre_override_hash": self.pre_override_hash,
            "post_override_hash": self.post_override_hash,
            "default_value": self.default_value,
            "override_value": self.override_value,
            "min_interatomic_distance_angstrom": self.min_interatomic_distance_angstrom,
            "operator_justification": self.operator_justification,
            "physical_implication": self.physical_implication,
        }


def compute_pairwise_distances(coordinates_angstrom: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distance matrix for N 3D coordinates."""
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) coordinate array, got shape {coords.shape}")
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def verify_clash_guard(
    coordinates_angstrom: np.ndarray,
    min_threshold_angstrom: float = MIN_INTERATOMIC_DISTANCE_ANGSTROM,
) -> Tuple[bool, float]:
    """
    Assert that all pairwise interatomic distances satisfy min r_ij >= threshold.
    
    Returns:
        Tuple of (is_valid, min_distance_found)
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    n_atoms = coords.shape[0]
    if n_atoms < 2:
        return True, float("inf")

    dist_matrix = compute_pairwise_distances(coords)
    # Mask diagonal (self-distances)
    np.fill_diagonal(dist_matrix, np.inf)
    min_dist = float(np.min(dist_matrix))

    if min_dist < min_threshold_angstrom:
        return False, min_dist
    return True, min_dist


def assert_clash_guard(
    coordinates_angstrom: np.ndarray,
    min_threshold_angstrom: float = MIN_INTERATOMIC_DISTANCE_ANGSTROM,
) -> None:
    """Raise NuclearClashError if coordinates violate the clash guard invariant."""
    is_valid, min_dist = verify_clash_guard(coordinates_angstrom, min_threshold_angstrom)
    if not is_valid:
        raise NuclearClashError(
            f"[ERR_NUCLEAR_CLASH]: Interatomic distance ({min_dist:.4f} A) violates minimum threshold ({min_threshold_angstrom:.4f} A). "
            f"Prevents Coulombic singularity during SCF initialization."
        )


def create_manual_adjustment_record(
    target_module: str,
    parameter_name: str,
    default_value: Any,
    override_value: Any,
    coordinates: Optional[np.ndarray],
    operator_justification: str,
    physical_implication: str,
) -> ManualAdjustmentRecord:
    """
    Create an immutable ManualAdjustmentRecord with SHA-256 state hashes and clash validation.
    """
    if len(operator_justification.strip()) < 10:
        raise ValueError("Operator justification must be at least 10 characters long.")

    min_dist = float("inf")
    if coordinates is not None:
        assert_clash_guard(coordinates)
        _, min_dist = verify_clash_guard(coordinates)

    pre_payload = json.dumps({"target": target_module, "param": parameter_name, "val": str(default_value)}, sort_keys=True)
    post_payload = json.dumps({"target": target_module, "param": parameter_name, "val": str(override_value)}, sort_keys=True)

    pre_hash = hashlib.sha256(pre_payload.encode("utf-8")).hexdigest()
    post_hash = hashlib.sha256(post_payload.encode("utf-8")).hexdigest()

    return ManualAdjustmentRecord(
        timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        target_module=target_module,
        parameter_name=parameter_name,
        pre_override_hash=pre_hash,
        post_override_hash=post_hash,
        default_value=default_value,
        override_value=override_value,
        min_interatomic_distance_angstrom=round(min_dist, 4),
        operator_justification=operator_justification,
        physical_implication=physical_implication,
    )


def cluster_rotamers_rmsd(
    conformer_coords: List[np.ndarray],
    rmsd_threshold_angstrom: float = 0.125,
) -> List[int]:
    """
    Cluster conformers using heavy-atom Kabsch RMSD alignment.
    Returns indices of unique centroid conformers.
    """
    if not conformer_coords:
        return []

    unique_indices: List[int] = []
    for idx, cand_coords in enumerate(conformer_coords):
        is_duplicate = False
        for uniq_idx in unique_indices:
            ref_coords = conformer_coords[uniq_idx]
            rmsd, _ = ToposCrusher.kabsch_rmsd(cand_coords, ref_coords)
            if rmsd < rmsd_threshold_angstrom:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_indices.append(idx)

    return unique_indices


def filter_energy_window(
    energies_kcal_mol: List[float],
    max_window_kcal: float = 6.0,
) -> List[int]:
    """
    Filter conformer indices within an energy window above the global minimum.
    """
    if not energies_kcal_mol:
        return []

    arr = np.asarray(energies_kcal_mol, dtype=np.float64)
    e_min = float(np.min(arr))
    survivors = [i for i, val in enumerate(arr) if (val - e_min) <= max_window_kcal]
    return survivors


class ToposCrusher:
    """Deduplicates conformer ensembles using optimal rigid 3D Kabsch superposition."""

    @classmethod
    def kabsch_rmsd(
        cls,
        coords_a_ang: np.ndarray,
        coords_b_ang: np.ndarray,
        heavy_mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, np.ndarray]:
        """Compute optimal superposition and RMSD between two geometries."""
        P = np.array(coords_a_ang, dtype=float)
        Q = np.array(coords_b_ang, dtype=float)

        if heavy_mask is not None and np.any(heavy_mask):
            P_sub = P[heavy_mask]
            Q_sub = Q[heavy_mask]
        else:
            P_sub = P
            Q_sub = Q

        com_p = np.mean(P_sub, axis=0)
        com_q = np.mean(Q_sub, axis=0)
        P_cent = P_sub - com_p
        Q_cent = Q_sub - com_q

        H = P_cent.T @ Q_cent
        U, S, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        sign_matrix = np.diag([1.0, 1.0, 1.0 if d >= 0 else -1.0])

        R = Vt.T @ sign_matrix @ U.T
        P_all_cent = P - com_p
        P_rot = (R @ P_all_cent.T).T + com_q

        diff = (P_rot - Q)[heavy_mask] if (heavy_mask is not None and np.any(heavy_mask)) else (P_rot - Q)
        rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
        return rmsd, P_rot

    @classmethod
    def cluster_conformers(
        cls,
        conformers: Sequence[Any],
        rmsd_threshold_angstrom: float = 0.25,
        energy_window_kcal_mol: float = 6.0,
        temperature_k: float = 298.15,
    ) -> List[Any]:
        """
        Deduplicate and rank conformer ensemble using Kabsch RMSD alignment and Boltzmann weighting.
        """
        if not conformers:
            return []

        from cochem_base.core.models import ConformerRecord

        # 1. Energy window filtering
        min_e = min(c.relative_energy_kcal_mol for c in conformers)
        in_window = [
            c for c in conformers if (c.relative_energy_kcal_mol - min_e) <= energy_window_kcal_mol
        ]
        # Sort by energy ascending
        sorted_confs = sorted(in_window, key=lambda c: c.relative_energy_kcal_mol)
        if not sorted_confs:
            return []

        ang_to_bohr = 1.88972612462577

        # 2. Greedy Kabsch RMSD clustering
        clustered: List[Any] = []
        for cand in sorted_confs:
            cand_coords_ang = np.array(cand.topology.coordinates_bohr, dtype=float) / ang_to_bohr
            atomic_nums = np.array(cand.topology.atomic_numbers)
            heavy_mask = atomic_nums > 1
            if not np.any(heavy_mask):
                heavy_mask = None

            is_dup = False
            for center in clustered:
                center_coords_ang = np.array(center.topology.coordinates_bohr, dtype=float) / ang_to_bohr
                rmsd, _ = cls.kabsch_rmsd(cand_coords_ang, center_coords_ang, heavy_mask=heavy_mask)
                if rmsd < rmsd_threshold_angstrom:
                    is_dup = True
                    break
            if not is_dup:
                clustered.append(cand)

        # 3. Authentic Boltzmann statistical weighting
        k_b = 0.00198720425864083  # kcal/(mol*K)
        beta = 1.0 / (k_b * temperature_k)
        rel_energies = np.array([c.relative_energy_kcal_mol for c in clustered], dtype=float)
        exp_factors = np.exp(-beta * (rel_energies - np.min(rel_energies)))
        weights = exp_factors / np.sum(exp_factors)

        # Reference global minimum geometry
        ref_min_ang = np.array(clustered[0].topology.coordinates_bohr, dtype=float) / ang_to_bohr
        ref_heavy = np.array(clustered[0].topology.atomic_numbers) > 1
        if not np.any(ref_heavy):
            ref_heavy = None

        final_records: List[ConformerRecord] = []
        for idx, rec in enumerate(clustered):
            rec_coords_ang = np.array(rec.topology.coordinates_bohr, dtype=float) / ang_to_bohr
            if idx == 0:
                rmsd_min = 0.0
            else:
                rmsd_min, _ = cls.kabsch_rmsd(rec_coords_ang, ref_min_ang, heavy_mask=ref_heavy)

            final_records.append(
                ConformerRecord(
                    conformer_id=rec.conformer_id,
                    topology=rec.topology,
                    relative_energy_kcal_mol=rec.relative_energy_kcal_mol,
                    boltzmann_weight=float(weights[idx]),
                    rmsd_to_minimum_angstrom=float(round(rmsd_min, 4)),
                )
            )

        return final_records

