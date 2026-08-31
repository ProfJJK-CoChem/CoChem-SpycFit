"""
CoChem Molecular Graph Topology & Gyration Tensor Asphericity Analyzer.

Provenance:
- Method Matrix [M]: Covalent connectivity from Cordero/Pyykkö radii, Floyd-Warshall graph diameter,
  3D gyration tensor asphericity, and DLPNO-CCSD(T) auto-promotion decision gating (REQ-QC-059).
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional, Sequence, Set, Tuple
import numpy as np

from cochem.core.cochem_elements import get_covalent_radius


def build_covalent_molecular_graph(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    covalent_factor: float = 1.25,
) -> Dict[int, List[int]]:
    """
    [REQ-QC-059] Construct molecular adjacency list from interatomic distances and dynamic covalent radii.

    Bond condition:
        d_ij <= covalent_factor * (R_cov,i + R_cov,j)

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        covalent_factor: Scaling multiplier for covalent radius sum (default: 1.25).

    Returns:
        Adjacency list mapping atom index -> list of bonded atom indices.
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    n_atoms = len(symbols)
    radii = np.array([get_covalent_radius(s) for s in symbols], dtype=np.float64)

    adj: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dist = float(np.linalg.norm(coords[i] - coords[j]))
            cutoff = covalent_factor * (radii[i] + radii[j])
            if dist <= cutoff:
                adj[i].append(j)
                adj[j].append(i)

    return adj


def compute_graph_diameter_and_sparsity(
    adj: Dict[int, List[int]],
) -> Tuple[int, float]:
    """
    [REQ-QC-059] Compute the shortest-path graph diameter D(G) and graph sparsity index sigma_graph.

    Diameter D(G) = max_{u,v} d_G(u, v)
    Sparsity sigma_graph = 2 * |E| / (|V| * (|V| - 1))

    Args:
        adj: Molecular graph adjacency list.

    Returns:
        Tuple of (diameter, sparsity_index).
    """
    n_nodes = len(adj)
    if n_nodes <= 1:
        return 0, 1.0

    # Count total edges
    total_edges = sum(len(neighbors) for neighbors in adj.values()) // 2
    max_possible_edges = (n_nodes * (n_nodes - 1)) // 2
    sparsity = float(total_edges) / float(max_possible_edges) if max_possible_edges > 0 else 1.0

    # BFS all-pairs shortest paths
    max_distance = 0

    for start_node in range(n_nodes):
        distances = {start_node: 0}
        queue = collections.deque([start_node])

        while queue:
            curr = queue.popleft()
            curr_dist = distances[curr]
            for neighbor in adj.get(curr, []):
                if neighbor not in distances:
                    distances[neighbor] = curr_dist + 1
                    queue.append(neighbor)

        if distances:
            max_d_from_start = max(distances.values())
            if max_d_from_start > max_distance:
                max_distance = max_d_from_start

    return max_distance, round(sparsity, 4)


def compute_gyration_tensor_and_asphericity(
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
) -> Tuple[np.ndarray, Tuple[float, float, float], float]:
    """
    [REQ-QC-059] Compute 3D gyration tensor, principal gyration radii (R1 >= R2 >= R3), and normalized asphericity Omega.

    Gyration Tensor:
        S_alpha_beta = (1 / N) * sum_i (r_i_alpha - r_COM_alpha)(r_i_beta - r_COM_beta)

    Normalized Asphericity:
        Omega = ((R1 - R2)^2 + (R2 - R3)^2 + (R3 - R1)^2) / (2 * (R1 + R2 + R3)^2)
        (Omega = 1 for 1D rod, 0.25 for 2D disc, 0 for 3D sphere).

    Args:
        coordinates_angstrom: Cartesian coordinates in Å.

    Returns:
        Tuple of (gyration_tensor_3x3, (R1, R2, R3), asphericity_omega).
    """
    coords = np.asarray(coordinates_angstrom, dtype=np.float64)
    n_atoms = len(coords)
    if n_atoms <= 1:
        return np.zeros((3, 3)), (0.0, 0.0, 0.0), 0.0

    com = np.mean(coords, axis=0)
    centered = coords - com

    # Gyration tensor S: (1/N) * centered^T @ centered
    gyration_tensor = (centered.T @ centered) / float(n_atoms)

    eigenvalues, _ = np.linalg.eigh(gyration_tensor)
    # Principal gyration radii (R1 >= R2 >= R3)
    sort_desc = np.sort(np.maximum(0.0, eigenvalues))[::-1]
    r1, r2, r3 = float(sort_desc[0]), float(sort_desc[1]), float(sort_desc[2])

    radius_sum = r1 + r2 + r3
    if radius_sum < 1e-12:
        omega = 0.0
    else:
        numerator = (r1 - r2) ** 2 + (r2 - r3) ** 2 + (r3 - r1) ** 2
        denominator = 2.0 * (radius_sum ** 2)
        omega = float(numerator / denominator)

    omega_clamped = float(np.clip(omega, 0.0, 1.0))
    return gyration_tensor, (r1, r2, r3), round(omega_clamped, 4)


def should_promote_to_dlpno(
    n_atoms: int,
    n_basis: int,
    omega: float,
    diameter: int,
) -> bool:
    """
    [REQ-QC-059] Dynamic DLPNO-CCSD(T) auto-promotion decision gating.

    Gating Formula:
        promote = (n_atoms >= 30 or n_basis >= 350) and (omega >= 0.40 or diameter >= 8)

    Args:
        n_atoms: Number of atoms in molecular system.
        n_basis: Number of contracted Gaussian basis functions.
        omega: Normalized 3D asphericity parameter in [0.0, 1.0].
        diameter: Shortest-path topological graph diameter D(G).

    Returns:
        True if linear-scaling DLPNO-CCSD(T) should replace canonical CCSD(T).
    """
    size_gate = (n_atoms >= 30) or (n_basis >= 350)
    topology_gate = (omega >= 0.40) or (diameter >= 8)

    return bool(size_gate and topology_gate)
