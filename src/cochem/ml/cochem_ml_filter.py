"""
CoChem NNIP Conformer Screening, Fallback Ladder & Automorphism RMSD.

Provenance:
- Method Matrix [M]: 4-tier fallback ladder (TorchANI -> MACE -> GFN2-xTB -> MMFF94s),
  energy pruning window Delta E <= 6.0 kcal/mol, Hungarian automorphism RMSD < 0.25 Å (REQ-ML-048).
- Derived Relations [D]: Hungarian optimal linear sum assignment for heavy-atom permutation invariant RMSD.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Set, Tuple
import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import linear_sum_assignment

from cochem.core.cochem_elements import get_atomic_number
from cochem.core.cochem_exceptions import ModelCoverageError

# Element subsets for each tier
ANI2X_ELEMENTS: Set[str] = {"H", "C", "N", "O", "S", "F", "CL"}
MACE_OFF23_ELEMENTS: Set[str] = {
    "H", "C", "N", "O", "F", "P", "S", "CL", "BR", "I", "B", "SI", "LI", "NA", "K", "MG", "CA", "ZN"
}
XTB_MAX_Z: int = 86  # Hydrogen to Radon
MMFF94_ELEMENTS: Set[str] = {
    "H", "C", "N", "O", "F", "SI", "P", "S", "CL", "BR", "I", "FE"
}


class ConformerPruningReport(BaseModel):
    """Data contract reporting conformer ensemble pruning and Hungarian clustering metrics."""

    initial_conformer_count: int
    selected_nnip_backend: Literal["torchani_ani2x", "mace_off23", "gfn2_xtb", "mmff94s"]
    energy_filtered_count: int
    rmsd_clustered_count: int
    final_retained_count: int
    energy_threshold_kcal_mol: float = 6.0
    rmsd_threshold_angstrom: float = 0.25


def select_nnip_backend(symbols: Sequence[str]) -> Literal["torchani_ani2x", "mace_off23", "gfn2_xtb", "mmff94s"]:
    """
    [REQ-ML-048] Select the highest available NNIP / semi-empirical / FF backend via 4-tier fallback ladder.

    Tier 1: TorchANI (ANI-2x) if all elements in {H, C, N, O, S, F, Cl}
    Tier 2: MACE-OFF23 if all elements in extended organic/halogen set
    Tier 3: GFN2-xTB if all Z <= 86
    Tier 4: MMFF94s if supported by MMFF
    Raises ModelCoverageError if all tiers fail.
    """
    upper_symbols = {s.strip().upper() for s in symbols}
    z_list = [get_atomic_number(s) for s in symbols]

    # Tier 1: ANI-2x
    if upper_symbols.issubset(ANI2X_ELEMENTS):
        return "torchani_ani2x"

    # Tier 2: MACE-OFF23
    if upper_symbols.issubset(MACE_OFF23_ELEMENTS):
        return "mace_off23"

    # Tier 3: GFN2-xTB
    if all(z <= XTB_MAX_Z for z in z_list):
        return "gfn2_xtb"

    # Tier 4: MMFF94s
    if upper_symbols.issubset(MMFF94_ELEMENTS):
        return "mmff94s"

    uncovered = [s for s, z in zip(symbols, z_list) if z > XTB_MAX_Z]
    raise ModelCoverageError(
        uncovered_elements=uncovered or list(symbols),
        attempted_backends=["torchani_ani2x", "mace_off23", "gfn2_xtb", "mmff94s"],
    )


def compute_hungarian_rmsd(
    symbols: Sequence[str],
    coords1: np.ndarray,
    coords2: np.ndarray,
    heavy_atoms_only: bool = True,
) -> float:
    """
    [REQ-ML-048] Calculate Hungarian graph-automorphism permutation-invariant RMSD between two conformers.

    Args:
        symbols: Chemical element symbols.
        coords1: (N_atoms, 3) Cartesian coordinates for Conformer 1 in Å.
        coords2: (N_atoms, 3) Cartesian coordinates for Conformer 2 in Å.
        heavy_atoms_only: If True, only evaluate heavy (non-hydrogen) atoms.

    Returns:
        RMSD in Ångströms after optimal Hungarian atom mapping within each element group.
    """
    c1 = np.asarray(coords1, dtype=np.float64)
    c2 = np.asarray(coords2, dtype=np.float64)

    # Group atom indices by chemical symbol
    element_groups: Dict[str, List[int]] = {}
    for idx, sym in enumerate(symbols):
        norm_sym = sym.strip().upper()
        if heavy_atoms_only and norm_sym == "H":
            continue
        element_groups.setdefault(norm_sym, []).append(idx)

    # If no heavy atoms (e.g. H2), evaluate all atoms
    if not element_groups:
        for idx, sym in enumerate(symbols):
            norm_sym = sym.strip().upper()
            element_groups.setdefault(norm_sym, []).append(idx)

    total_sq_diff = 0.0
    total_atoms_evaluated = 0

    for elem, indices in element_groups.items():
        n_elem = len(indices)
        sub_c1 = c1[indices]  # (K, 3)
        sub_c2 = c2[indices]  # (K, 3)

        if n_elem == 1:
            total_sq_diff += float(np.sum((sub_c1[0] - sub_c2[0]) ** 2))
            total_atoms_evaluated += 1
            continue

        # Pairwise distance cost matrix for this element group: (K, K)
        # cost[i, j] = ||sub_c1[i] - sub_c2[j]||^2
        diff = sub_c1[:, np.newaxis, :] - sub_c2[np.newaxis, :, :]  # (K, K, 3)
        cost_matrix = np.sum(diff ** 2, axis=-1)  # (K, K)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_sq_diff += float(np.sum(cost_matrix[row_ind, col_ind]))
        total_atoms_evaluated += n_elem

    if total_atoms_evaluated == 0:
        return 0.0

    return float(np.sqrt(total_sq_diff / total_atoms_evaluated))


def prune_conformer_ensemble(
    symbols: Sequence[str],
    conformers: Sequence[np.ndarray],
    energies_kcal_mol: Sequence[float],
    energy_threshold_kcal_mol: float = 6.0,
    rmsd_threshold_angstrom: float = 0.25,
) -> Tuple[List[np.ndarray], List[float], ConformerPruningReport]:
    """
    [REQ-ML-048] Prune conformer ensemble by relative energy window and Hungarian RMSD clustering.

    Steps:
        1. Select NNIP backend.
        2. Filter out conformers with Delta E > energy_threshold_kcal_mol (6.0 kcal/mol).
        3. Sort surviving conformers by energy.
        4. Greedily cluster conformers: if RMSD(C_i, C_j) < rmsd_threshold (0.25 Å), prune C_j.

    Returns:
        Tuple of (retained_conformers, retained_energies, ConformerPruningReport).
    """
    n_initial = len(conformers)
    backend = select_nnip_backend(symbols)

    if n_initial == 0:
        return [], [], ConformerPruningReport(
            initial_conformer_count=0,
            selected_nnip_backend=backend,
            energy_filtered_count=0,
            rmsd_clustered_count=0,
            final_retained_count=0,
            energy_threshold_kcal_mol=energy_threshold_kcal_mol,
            rmsd_threshold_angstrom=rmsd_threshold_angstrom,
        )

    # 1. Energy window filtering
    min_energy = min(energies_kcal_mol)
    energy_filtered_indices = [
        i for i, e in enumerate(energies_kcal_mol)
        if (e - min_energy) <= energy_threshold_kcal_mol
    ]
    energy_filtered_count = n_initial - len(energy_filtered_indices)

    # Sort surviving conformers by energy
    energy_filtered_indices.sort(key=lambda idx: energies_kcal_mol[idx])

    # 2. Hungarian RMSD clustering
    retained_indices: List[int] = []
    rmsd_clustered_count = 0

    for idx in energy_filtered_indices:
        candidate_coords = np.asarray(conformers[idx], dtype=np.float64)
        is_duplicate = False

        for ret_idx in retained_indices:
            ref_coords = np.asarray(conformers[ret_idx], dtype=np.float64)
            rmsd = compute_hungarian_rmsd(
                symbols=symbols,
                coords1=candidate_coords,
                coords2=ref_coords,
                heavy_atoms_only=True,
            )
            if rmsd < rmsd_threshold_angstrom:
                is_duplicate = True
                rmsd_clustered_count += 1
                break

        if not is_duplicate:
            retained_indices.append(idx)

    final_conformers = [conformers[i] for i in retained_indices]
    final_energies = [energies_kcal_mol[i] for i in retained_indices]

    report = ConformerPruningReport(
        initial_conformer_count=n_initial,
        selected_nnip_backend=backend,
        energy_filtered_count=energy_filtered_count,
        rmsd_clustered_count=rmsd_clustered_count,
        final_retained_count=len(final_conformers),
        energy_threshold_kcal_mol=energy_threshold_kcal_mol,
        rmsd_threshold_angstrom=rmsd_threshold_angstrom,
    )

    return final_conformers, final_energies, report
