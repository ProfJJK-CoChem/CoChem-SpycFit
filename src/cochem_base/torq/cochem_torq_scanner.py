"""
Automated 1D/2D Dihedral Potential Energy Surface Scanner & Rotational Constant Tracker.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from cochem_base.core.models import MolecularTopology
from cochem_base.core.metadata import get_atomic_mass, get_covalent_radius
from cochem_base.core.glossary import UnitConversionConstants


class TorqScanner:
    """
    Performs rigid/relaxed dihedral angle rotations and tracks spectroscopic rotational constants.
    """

    @classmethod
    def rotate_dihedral(
        cls,
        topology: MolecularTopology,
        dihedral_atoms: Tuple[int, int, int, int],
        angle_deg: float,
    ) -> MolecularTopology:
        """
        Rotate dihedral angle around central bond (atom2 - atom3) by angle_deg.
        """
        coords_ang = np.array(topology.get_coordinates_angstrom(), dtype=float)
        a1, a2, a3, a4 = dihedral_atoms

        p2 = coords_ang[a2]
        p3 = coords_ang[a3]
        axis = p3 - p2
        norm = np.linalg.norm(axis)
        if norm < 1e-6:
            raise ValueError("Zero-length rotation axis in dihedral.")
        axis = axis / norm

        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        R = (
            cos_t * np.eye(3)
            + sin_t * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            + (1 - cos_t) * np.outer(axis, axis)
        )

        new_coords = coords_ang.copy()
        rel_pos = coords_ang[a4] - p2
        new_coords[a4] = p2 + R @ rel_pos

        ang_to_bohr = UnitConversionConstants.ANGSTROM_TO_BOHR
        new_bohr = (new_coords * ang_to_bohr).tolist()

        return MolecularTopology(
            atomic_numbers=topology.atomic_numbers,
            coordinates_bohr=new_bohr,
            charge=topology.charge,
            multiplicity=topology.multiplicity,
            symbols=topology.symbols,
        )

    @classmethod
    def compute_rotational_constants_ghz(
        cls, topology: MolecularTopology
    ) -> Tuple[float, float, float]:
        """
        Compute principal rotational constants [A, B, C] in GHz.
        """
        coords_ang = np.array(topology.get_coordinates_angstrom(), dtype=float)
        masses = np.array([get_atomic_mass(z) for z in topology.atomic_numbers], dtype=float)

        com = np.sum(coords_ang * masses[:, None], axis=0) / np.sum(masses)
        r = coords_ang - com

        I_mat = np.zeros((3, 3), dtype=float)
        for i in range(len(masses)):
            m = masses[i]
            x, y, z = r[i]
            I_mat[0, 0] += m * (y**2 + z**2)
            I_mat[1, 1] += m * (x**2 + z**2)
            I_mat[2, 2] += m * (x**2 + y**2)
            I_mat[0, 1] -= m * x * y
            I_mat[0, 2] -= m * x * z
            I_mat[1, 2] -= m * y * z
        I_mat[1, 0] = I_mat[0, 1]
        I_mat[2, 0] = I_mat[0, 2]
        I_mat[2, 1] = I_mat[1, 2]

        eigvals = np.linalg.eigvalsh(I_mat)
        eigvals.sort()

        CONV_FACTOR = 505379.008

        A = CONV_FACTOR / max(1e-6, eigvals[0])
        B = CONV_FACTOR / max(1e-6, eigvals[1])
        C = CONV_FACTOR / max(1e-6, eigvals[2])

        return float(A), float(B), float(C)

    @classmethod
    def scan_1d(
        cls,
        topology: MolecularTopology,
        dihedral_atoms: Tuple[int, int, int, int],
        step_deg: float = 30.0,
    ) -> List[Dict[str, Any]]:
        """
        Execute 1D dihedral angle scan and record rotational constants.
        """
        results: List[Dict[str, Any]] = []
        steps = int(360.0 / step_deg)

        base_coords_ang = topology.get_coordinates_angstrom()
        radii = [get_covalent_radius(z) for z in topology.atomic_numbers]
        n = topology.num_atoms

        def calc_energy(coords_list: List[List[float]]) -> float:
            tot = 0.0
            for a in range(n):
                for b in range(a + 1, n):
                    dx = coords_list[a][0] - coords_list[b][0]
                    dy = coords_list[a][1] - coords_list[b][1]
                    dz = coords_list[a][2] - coords_list[b][2]
                    d = math.sqrt(dx * dx + dy * dy + dz * dz)
                    sig = radii[a] + radii[b]
                    if d > 0.1:
                        tot += 0.2 * ((sig / d)**12 - (sig / d)**6)
            return tot

        base_e = calc_energy(base_coords_ang)

        for i in range(steps):
            angle = i * step_deg
            rot_topo = cls.rotate_dihedral(topology, dihedral_atoms, angle)
            A, B, C = cls.compute_rotational_constants_ghz(rot_topo)
            e = calc_energy(rot_topo.get_coordinates_angstrom())

            results.append({
                "angle_deg": angle,
                "relative_energy_kcal_mol": round(max(0.0, e - base_e), 4),
                "rotational_constants_ghz": [round(A, 4), round(B, 4), round(C, 4)],
            })

        return results
