"""
MINT Molecular Graph Ingestor, Connectivity Analyzer, and Fragment Partitioner.

Adheres strictly to dynamic Mendeleev mass and covalent radius lookups.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
from mendeleev import element as m_element

from cochem_base.core.models import MolecularTopology
from cochem_base.core.metadata import get_atomic_mass, get_covalent_radius
from cochem_base.core.glossary import UnitConversionConstants


class MINTSStructureIngestor:
    """
    Ingests molecular structures from standard formats and builds molecular graphs.
    """

    @classmethod
    def parse_xyz_string(cls, xyz_text: str) -> MolecularTopology:
        """
        Parse standard multi-line XYZ coordinate string into MolecularTopology.
        """
        lines = [line.strip() for line in xyz_text.strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError("Empty XYZ coordinate string provided.")

        try:
            num_atoms = int(lines[0])
            start_idx = 2  # line 1 is comment
        except ValueError:
            # Format lacks header line
            num_atoms = len(lines)
            start_idx = 0

        atomic_numbers: List[int] = []
        coordinates_bohr: List[List[float]] = []
        symbols: List[str] = []

        ang_to_bohr = UnitConversionConstants.ANGSTROM_TO_BOHR

        for line in lines[start_idx : start_idx + num_atoms]:
            tokens = line.split()
            if len(tokens) < 4:
                continue
            sym_or_z = tokens[0]
            try:
                z = int(sym_or_z)
                el = m_element(z)
                sym = str(el.symbol)
            except ValueError:
                sym = sym_or_z.capitalize()
                el = m_element(sym)
                z = int(el.atomic_number)

            x = float(tokens[1]) * ang_to_bohr
            y = float(tokens[2]) * ang_to_bohr
            z_coord = float(tokens[3]) * ang_to_bohr

            atomic_numbers.append(z)
            symbols.append(sym)
            coordinates_bohr.append([x, y, z_coord])

        return MolecularTopology(
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coordinates_bohr,
            symbols=symbols,
        )

    @classmethod
    def parse_pdb_string(cls, pdb_text: str) -> MolecularTopology:
        """
        Parse PDB file text into MolecularTopology.
        """
        atomic_numbers: List[int] = []
        coordinates_bohr: List[List[float]] = []
        symbols: List[str] = []
        ang_to_bohr = UnitConversionConstants.ANGSTROM_TO_BOHR

        for line in pdb_text.splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    sym = line[76:78].strip()
                    if not sym:
                        sym = line[12:16].strip()
                        sym = "".join([c for c in sym if c.isalpha()])[:2]
                    
                    sym = sym.capitalize()
                    el = m_element(sym)
                    z = int(el.atomic_number)

                    x = float(line[30:38].strip()) * ang_to_bohr
                    y = float(line[38:46].strip()) * ang_to_bohr
                    z_coord = float(line[46:54].strip()) * ang_to_bohr

                    atomic_numbers.append(z)
                    symbols.append(sym)
                    coordinates_bohr.append([x, y, z_coord])
                except Exception:
                    continue

        return MolecularTopology(
            atomic_numbers=atomic_numbers,
            coordinates_bohr=coordinates_bohr,
            symbols=symbols,
        )

    @classmethod
    def compute_connectivity_graph(
        cls, topology: MolecularTopology, tolerance_angstrom: float = 0.4
    ) -> List[Tuple[int, int]]:
        """
        Derive covalent adjacency connectivity graph using dynamic Mendeleev covalent radii.
        """
        coords_ang = topology.get_coordinates_angstrom()
        n = topology.num_atoms
        bonds: List[Tuple[int, int]] = []

        radii = [get_covalent_radius(z) for z in topology.atomic_numbers]

        for i in range(n):
            for j in range(i + 1, n):
                dist = math.sqrt(
                    (coords_ang[i][0] - coords_ang[j][0]) ** 2
                    + (coords_ang[i][1] - coords_ang[j][1]) ** 2
                    + (coords_ang[i][2] - coords_ang[j][2]) ** 2
                )
                max_bond_dist = radii[i] + radii[j] + tolerance_angstrom
                if dist <= max_bond_dist and dist > 0.4:
                    bonds.append((i, j))

        return bonds

    @classmethod
    def compute_molecular_weight(cls, topology: MolecularTopology) -> float:
        """
        Compute molecular weight dynamically by querying Mendeleev atomic masses.
        """
        return float(sum(get_atomic_mass(z) for z in topology.atomic_numbers))

    @classmethod
    def partition_fragments(cls, topology: MolecularTopology) -> List[MolecularTopology]:
        """
        Partition topology into disconnected molecular fragments / complexes via BFS.
        """
        n = topology.num_atoms
        bonds = cls.compute_connectivity_graph(topology)

        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        for u, v in bonds:
            adj[u].append(v)
            adj[v].append(u)

        visited: Set[int] = set()
        fragments: List[MolecularTopology] = []

        for i in range(n):
            if i not in visited:
                comp_indices: List[int] = []
                queue = deque([i])
                visited.add(i)

                while queue:
                    curr = queue.popleft()
                    comp_indices.append(curr)
                    for neighbor in adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                comp_indices.sort()
                sub_z = [topology.atomic_numbers[idx] for idx in comp_indices]
                sub_coords = [topology.coordinates_bohr[idx] for idx in comp_indices]
                sub_sym = [topology.symbols[idx] for idx in comp_indices] if topology.symbols else None

                frag = MolecularTopology(
                    atomic_numbers=sub_z,
                    coordinates_bohr=sub_coords,
                    symbols=sub_sym,
                    charge=0,
                    multiplicity=1,
                )
                fragments.append(frag)

        return fragments
