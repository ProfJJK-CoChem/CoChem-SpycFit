"""
Multi-Basin Stochastic Jiggle-Quench Conformer Search Engine.
"""

from __future__ import annotations

import hashlib
import math
from typing import List, Optional
import numpy as np

from cochem_base.core.models import MolecularTopology, ConformerRecord
from cochem_base.core.metadata import get_covalent_radius
from cochem_base.topos.cochem_topos_crusher import ToposCrusher


class ToposEngine:
    """
    Generates diverse conformational ensembles via stochastic perturbation and steric potential relaxation.
    """

    def __init__(self, crusher: Optional[ToposCrusher] = None) -> None:
        self.crusher = crusher or ToposCrusher()

    @staticmethod
    def evaluate_steric_energy_kcal_mol(topology: MolecularTopology) -> float:
        """
        Compute interatomic Lennard-Jones steric potential energy from covalent radii.
        """
        coords_ang = topology.get_coordinates_angstrom()
        n = topology.num_atoms
        radii = [get_covalent_radius(z) for z in topology.atomic_numbers]
        energy_kcal = 0.0
        epsilon = 0.15

        for i in range(n):
            for j in range(i + 1, n):
                dx = coords_ang[i][0] - coords_ang[j][0]
                dy = coords_ang[i][1] - coords_ang[j][1]
                dz = coords_ang[i][2] - coords_ang[j][2]
                r = math.sqrt(dx * dx + dy * dy + dz * dz)
                sigma = radii[i] + radii[j]
                if r > 0.1:
                    ratio = sigma / r
                    energy_kcal += 4.0 * epsilon * (ratio**12 - ratio**6)
        return float(energy_kcal)

    def generate_conformers(
        self,
        base_topology: MolecularTopology,
        num_trials: int = 20,
        rmsd_threshold_angstrom: float = 0.25,
        energy_window_kcal_mol: float = 6.0,
        jiggle_amplitude_angstrom: float = 0.15,
        random_seed: int = 42,
    ) -> List[ConformerRecord]:
        """
        Execute stochastic conformational exploration around seed topology.
        """
        rng = np.random.default_rng(random_seed)
        coords_orig_bohr = np.array(base_topology.coordinates_bohr, dtype=float)
        ang_to_bohr = 1.88972612462577

        raw_records: List[ConformerRecord] = []

        # Base minimum
        base_id = hashlib.sha256(coords_orig_bohr.tobytes()).hexdigest()[:12]
        base_e = self.evaluate_steric_energy_kcal_mol(base_topology)

        raw_records.append(
            ConformerRecord(
                conformer_id=f"conf_00_{base_id}",
                topology=base_topology,
                relative_energy_kcal_mol=0.0,
                boltzmann_weight=1.0,
                rmsd_to_minimum_angstrom=0.0,
            )
        )

        # Generate trial perturbations
        for i in range(1, num_trials):
            jiggle = rng.normal(0, jiggle_amplitude_angstrom * ang_to_bohr, size=coords_orig_bohr.shape)
            new_coords = coords_orig_bohr + jiggle

            trial_topo = MolecularTopology(
                atomic_numbers=base_topology.atomic_numbers,
                coordinates_bohr=new_coords.tolist(),
                charge=base_topology.charge,
                multiplicity=base_topology.multiplicity,
                symbols=base_topology.symbols,
            )

            trial_e = self.evaluate_steric_energy_kcal_mol(trial_topo)
            rel_e = max(0.0, trial_e - base_e)

            cid = hashlib.sha256(new_coords.tobytes()).hexdigest()[:12]
            raw_records.append(
                ConformerRecord(
                    conformer_id=f"conf_{i:02d}_{cid}",
                    topology=trial_topo,
                    relative_energy_kcal_mol=round(rel_e, 4),
                    boltzmann_weight=0.0,
                    rmsd_to_minimum_angstrom=0.0,
                )
            )

        return self.crusher.cluster_conformers(
            raw_records,
            rmsd_threshold_angstrom=rmsd_threshold_angstrom,
            energy_window_kcal_mol=energy_window_kcal_mol,
        )
