"""
CoChem Topos Dimer Assembly & Spatial Conformer Integration Engine.

Provenance:
- Method Matrix [M]: Dimer assembly, principal axis orientation, dynamic Mendeleev masses (REQ-BM-043, REQ-TOPOS-050).
- Topos Crusher: 12D KD-Tree geometric sieving (REQ-TOPOS-055) and Boltzmann Shannon entropy telemetry (REQ-TOPOS-058).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_elements import batch_get_masses
from cochem.topos.cochem_topos_siever import (
    TwoStageKDTreeSiever,
    extract_12d_feature_vector,
)
from cochem.topos.cochem_topos_telemetry import (
    ConformerBasinTelemetry,
    EarlyStoppingStatus,
)


class DimerComplexModel(BaseModel):
    """Data contract representing assembled non-covalent dimer complex."""

    symbols: List[str]
    coordinates_angstrom: List[Tuple[float, float, float]]
    monomer_a_indices: List[int]
    monomer_b_indices: List[int]
    intermolecular_distance_angstrom: float
    principal_moments_of_inertia_amu_ang2: Tuple[float, float, float]


class ToposConformerEntry(BaseModel):
    """Data contract representing a unique, accepted conformer."""

    conformer_id: str
    energy_hartree: float
    coordinates_angstrom: List[Tuple[float, float, float]]
    iteration_discovered: int


class ToposSearchResult(BaseModel):
    """Data contract summarizing the results of a Topos Crusher conformer search."""

    total_samples_evaluated: int  # anti-spoof: total evaluation count
    unique_conformers: List[ToposConformerEntry]
    global_min_energy_hartree: float
    early_stopped: bool
    convergence_message: str


def compute_principal_moments_of_inertia(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
) -> Tuple[Tuple[float, float, float], np.ndarray]:
    """
    [REQ-BM-043] Compute principal moments of inertia (Ia <= Ib <= Ic) in amu*Å^2 and principal axes.

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.

    Returns:
        Tuple of ((Ia, Ib, Ic), principal_axes_matrix).
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    masses = batch_get_masses(list(symbols))

    # Center at center of mass
    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    centered_coords = coords - com

    # Construct inertia tensor I: (3, 3)
    x = centered_coords[:, 0]
    y = centered_coords[:, 1]
    z = centered_coords[:, 2]

    ixx = np.sum(masses * (y ** 2 + z ** 2))
    iyy = np.sum(masses * (x ** 2 + z ** 2))
    izz = np.sum(masses * (x ** 2 + y ** 2))
    ixy = -np.sum(masses * x * y)
    ixz = -np.sum(masses * x * z)
    iyz = -np.sum(masses * y * z)

    tensor = np.array([
        [ixx, ixy, ixz],
        [ixy, iyy, iyz],
        [ixz, iyz, izz],
    ], dtype=np.float64)

    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    sort_idx = np.argsort(eigenvalues)
    sorted_evals = eigenvalues[sort_idx]
    sorted_evecs = eigenvectors[:, sort_idx]

    moments = (float(sorted_evals[0]), float(sorted_evals[1]), float(sorted_evals[2]))
    return moments, sorted_evecs


def assemble_dimer_complex(
    symbols_a: Sequence[str],
    coords_a: Sequence[Tuple[float, float, float]],
    symbols_b: Sequence[str],
    coords_b: Sequence[Tuple[float, float, float]],
    intermolecular_separation_angstrom: float = 3.5,
    collision_axis: Tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> DimerComplexModel:
    """
    [REQ-BM-043, REQ-TOPOS-050] Assemble two monomers into a non-covalent dimer along a collision axis.

    Args:
        symbols_a: Element symbols for Monomer A.
        coords_a: Cartesian geometry of Monomer A in Å.
        symbols_b: Element symbols for Monomer B.
        coords_b: Cartesian geometry of Monomer B in Å.
        intermolecular_separation_angstrom: Center-of-mass separation distance in Å (default 3.5 Å).
        collision_axis: 3D vector defining approach orientation.

    Returns:
        DimerComplexModel instance.
    """
    ca = np.asarray(coords_a, dtype=np.float64)
    cb = np.asarray(coords_b, dtype=np.float64)
    ma = batch_get_masses(list(symbols_a))
    mb = batch_get_masses(list(symbols_b))

    # Center both monomers at their respective COMs
    com_a = np.sum(ca * ma[:, np.newaxis], axis=0) / np.sum(ma)
    com_b = np.sum(cb * mb[:, np.newaxis], axis=0) / np.sum(mb)

    centered_a = ca - com_a
    centered_b = cb - com_b

    # Normalize collision axis
    axis = np.asarray(collision_axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    unit_axis = axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Position Monomer A at -R/2 and Monomer B at +R/2 along collision axis
    r = float(intermolecular_separation_angstrom)
    pos_a = centered_a - 0.5 * r * unit_axis
    pos_b = centered_b + 0.5 * r * unit_axis

    combined_symbols = list(symbols_a) + list(symbols_b)
    combined_coords = np.vstack([pos_a, pos_b])

    idx_a = list(range(len(symbols_a)))
    idx_b = list(range(len(symbols_a), len(combined_symbols)))

    moments, _ = compute_principal_moments_of_inertia(
        symbols=combined_symbols,
        coordinates_angstrom=[tuple(row) for row in combined_coords],
    )

    return DimerComplexModel(
        symbols=combined_symbols,
        coordinates_angstrom=[tuple(row) for row in combined_coords],
        monomer_a_indices=idx_a,
        monomer_b_indices=idx_b,
        intermolecular_distance_angstrom=r,
        principal_moments_of_inertia_amu_ang2=moments,
    )


class ToposCrusherEngine:
    """
    [REQ-TOPOS-055, REQ-TOPOS-058] Production conformer generation engine with 12D KD-Tree sieving and Shannon telemetry.
    """

    def __init__(
        self,
        symbols: Sequence[str],
        base_coordinates: Sequence[Tuple[float, float, float]],
        num_rotatable_bonds: int = 1,
        thermal_window_kcal_mol: float = 3.0,
        eps_sieve: float = 1.5e-3,
        rmsd_prune_angstrom: float = 0.05,
    ) -> None:
        self.symbols = list(symbols)
        self.base_coordinates = list(base_coordinates)
        self.siever = TwoStageKDTreeSiever(eps_sieve=eps_sieve, rmsd_prune_angstrom=rmsd_prune_angstrom)
        self.telemetry = ConformerBasinTelemetry(
            num_rotatable_bonds=num_rotatable_bonds,
            thermal_window_kcal_mol=thermal_window_kcal_mol,
        )

    def run_conformer_search(
        self,
        sampler_func: Callable[[int], Tuple[List[Tuple[float, float, float]], float]],
        max_iterations: int = 200,
        batch_size: int = 25,
        convergence_output_path: Optional[Path] = None,
    ) -> ToposSearchResult:
        """
        [REQ-TOPOS-055, REQ-TOPOS-058] Run iterative conformer search loop with sieving and early-stopping.

        Args:
            sampler_func: Callable accepting iteration index -> returning (trial_coords, trial_energy).
            max_iterations: Maximum sampling iterations to perform.
            batch_size: Evaluation batch size for telemetry tracking.
            convergence_output_path: Optional path to serialize topos_convergence.json.

        Returns:
            ToposSearchResult instance.
        """
        unique_conformers: List[ToposConformerEntry] = []
        early_stopped = False
        last_message = "Search completed maximum iterations."

        for it in range(max_iterations):
            trial_coords, trial_energy = sampler_func(it)
            conf_id = f"conf_{it:04d}"

            # Extract 12D feature vector
            feat = extract_12d_feature_vector(
                symbols=self.symbols,
                coordinates_angstrom=trial_coords,
                conformer_id=conf_id,
                energy_hartree=trial_energy,
            )

            # Two-stage KD-tree sieving
            is_unique = self.siever.sieve_conformer(
                symbols=self.symbols,
                coordinates=trial_coords,
                feature_vector=feat,
            )

            if is_unique:
                unique_conformers.append(ToposConformerEntry(
                    conformer_id=conf_id,
                    energy_hartree=trial_energy,
                    coordinates_angstrom=trial_coords,
                    iteration_discovered=it,
                ))

            # Telemetry record
            status: EarlyStoppingStatus = self.telemetry.record_conformer_sample(
                sample_index=it,
                conformer_id=conf_id,
                energy_hartree=trial_energy,
                is_new_unique=is_unique,
            )

            if status.is_converged:
                early_stopped = True
                last_message = status.message
                break

        if convergence_output_path is not None:
            self.telemetry.dump_convergence_json(convergence_output_path)

        # Sort unique conformers by electronic energy ascending
        unique_conformers.sort(key=lambda c: c.energy_hartree)
        min_e = unique_conformers[0].energy_hartree if unique_conformers else 0.0

        return ToposSearchResult(
            total_samples_evaluated=len(self.telemetry.energy_history),
            unique_conformers=unique_conformers,
            global_min_energy_hartree=min_e,
            early_stopped=early_stopped,
            convergence_message=last_message,
        )
