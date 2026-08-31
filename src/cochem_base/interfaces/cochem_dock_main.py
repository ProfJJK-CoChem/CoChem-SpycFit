"""
3D Molecular Viewport State Machine, Coordinate Streaming, and Geometric Hit-Testing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from cochem_base.core.models import MolecularTopology


class MolecularViewport3D:
    """
    3D Model-View-Projection Camera and Interatomic Measurement Viewport.
    """

    def __init__(self, topology: Optional[MolecularTopology] = None) -> None:
        self.topology = topology
        self.camera_position = np.array([0.0, 0.0, 10.0], dtype=float)
        self.camera_target = np.array([0.0, 0.0, 0.0], dtype=float)
        self.zoom_factor = 1.0

    def update_coordinates(self, coordinates_bohr: List[List[float]]) -> None:
        """Update live Cartesian coordinates during optimization."""
        if self.topology:
            self.topology = MolecularTopology(
                atomic_numbers=self.topology.atomic_numbers,
                coordinates_bohr=coordinates_bohr,
                charge=self.topology.charge,
                multiplicity=self.topology.multiplicity,
                symbols=self.topology.symbols,
            )

    def measure_distance(self, atom_i: int, atom_j: int) -> float:
        """Calculate interatomic Euclidean distance in Angstroms."""
        if not self.topology:
            raise ValueError("No active molecular topology in viewport.")
        coords = self.topology.get_coordinates_angstrom()
        p1 = np.array(coords[atom_i])
        p2 = np.array(coords[atom_j])
        return float(np.linalg.norm(p1 - p2))

    def measure_angle(self, atom_i: int, atom_j: int, atom_k: int) -> float:
        """Calculate bond angle (i-j-k) in degrees centered at vertex j."""
        if not self.topology:
            raise ValueError("No active molecular topology in viewport.")
        coords = self.topology.get_coordinates_angstrom()
        pi = np.array(coords[atom_i])
        pj = np.array(coords[atom_j])
        pk = np.array(coords[atom_k])

        v1 = pi - pj
        v2 = pk - pj
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(math.degrees(math.acos(cos_angle)))

    def measure_dihedral(self, atom_i: int, atom_j: int, atom_k: int, atom_l: int) -> float:
        """Calculate dihedral angle (i-j-k-l) in degrees."""
        if not self.topology:
            raise ValueError("No active molecular topology in viewport.")
        coords = self.topology.get_coordinates_angstrom()
        p0 = np.array(coords[atom_i])
        p1 = np.array(coords[atom_j])
        p2 = np.array(coords[atom_k])
        p3 = np.array(coords[atom_l])

        b0 = -1.0 * (p1 - p0)
        b1 = p2 - p1
        b2 = p3 - p2

        b1_norm = np.linalg.norm(b1)
        if b1_norm < 1e-6:
            return 0.0
        b1_unit = b1 / b1_norm

        v = b0 - np.dot(b0, b1_unit) * b1_unit
        w = b2 - np.dot(b2, b1_unit) * b1_unit

        x = np.dot(v, w)
        y = np.dot(np.cross(b1_unit, v), w)
        return float(math.degrees(math.atan2(y, x)))
