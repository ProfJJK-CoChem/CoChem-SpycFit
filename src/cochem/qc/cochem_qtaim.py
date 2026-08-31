"""
CoChem QTAIM Electron Density and Bond Critical Point Topological Engine (REQ-QC-088).

Provenance:
- Method Matrix [M]: Quantum Theory of Atoms in Molecules (QTAIM) gradient vector field topology,
  Poincaré-Hopf relationship, and Cremer-Kraka / Bader bond interaction classifications.
- Derived Relations [D]: Local energy densities (G(r) kinetic, V(r) potential, H(r) total) and bond ellipticity.
- Empirical [E]: Bond Critical Point (3,-1) eigenvalues (lambda_1 <= lambda_2 < 0 < lambda_3).
"""

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from cochem.qc.exceptions import QTAIMTopologyError
from cochem.qc.models import BondCriticalPointDescriptor


def classify_bond_critical_point(
    bcp_idx: int,
    atom_1: int,
    atom_2: int,
    coords: Tuple[float, float, float],
    rho: float,
    hessian_eigenvals: Tuple[float, float, float],
    g_density: float,
    v_density: float,
) -> BondCriticalPointDescriptor:
    """
    [M][D] Classify a (3,-1) Bond Critical Point using Bader and Cremer-Kraka criteria (REQ-QC-088).

    Topological & Physical Criteria:
        1. Laplacian: nabla^2 rho = lambda_1 + lambda_2 + lambda_3
        2. Total Energy Density: H(r_BCP) = G(r_BCP) + V(r_BCP)
        3. Bond Ellipticity: epsilon = (lambda_1 / lambda_2) - 1 >= 0
        4. Interaction Classification:
           - nabla^2 rho < 0, H < 0: COVALENT_SHARED (covalent, electron sharing)
           - nabla^2 rho > 0, H < 0: TRANSIT_POLAR_COVALENT (polar covalent, coordination)
           - nabla^2 rho > 0, H > 0: NON_COVALENT_CLOSED_SHELL (van der Waals, ionic, steric)

    Args:
        bcp_idx: Unique integer index of the BCP.
        atom_1: 0-based index of first bonded atom.
        atom_2: 0-based index of second bonded atom.
        coords: Cartesian coordinates (x, y, z) in Å.
        rho: Electron density rho(r_BCP) in a.u. (must be > 0).
        hessian_eigenvals: 3 eigenvalues of the electron density Hessian matrix.
        g_density: Lagrangian kinetic energy density G(r) in a.u. (>= 0).
        v_density: Virial potential energy density V(r) in a.u. (<= 0).

    Returns:
        BondCriticalPointDescriptor containing topological scalar quantities and interaction class.

    Raises:
        ValueError: If rho <= 0, g_density < 0, or v_density > 0.
    """
    if rho <= 0.0:
        raise ValueError(f"Electron density at BCP must be positive, got rho={rho} a.u.")
    if g_density < 0.0:
        raise ValueError(f"Kinetic energy density G must be >= 0, got G={g_density} a.u.")
    if v_density > 0.0:
        raise ValueError(f"Potential energy density V must be <= 0, got V={v_density} a.u.")

    # 1. Sort Hessian eigenvalues: lambda_1 <= lambda_2 <= lambda_3
    sorted_lambdas = sorted(hessian_eigenvals)
    l1, l2, l3 = sorted_lambdas[0], sorted_lambdas[1], sorted_lambdas[2]

    # 2. Laplacian nabla^2 rho = l1 + l2 + l3
    laplacian = l1 + l2 + l3

    # 3. Total energy density H = G + V
    h_density = g_density + v_density

    # 4. Bond Ellipticity epsilon = (lambda_1 / lambda_2) - 1
    # For a genuine (3,-1) BCP, l1 and l2 are negative perpendicular curvatures
    if abs(l2) > 1e-9:
        ellipticity = max(0.0, (l1 / l2) - 1.0)
    else:
        ellipticity = 0.0

    # 5. Cremer-Kraka & Bader interaction classification
    classification: Literal[
        "COVALENT_SHARED", "TRANSIT_POLAR_COVALENT", "NON_COVALENT_CLOSED_SHELL", "HYDROGEN_BOND", "METALLIC"
    ]
    if laplacian < 0.0 and h_density < 0.0:
        classification = "COVALENT_SHARED"
    elif laplacian >= 0.0 and h_density < 0.0:
        classification = "TRANSIT_POLAR_COVALENT"
    else:
        classification = "NON_COVALENT_CLOSED_SHELL"

    return BondCriticalPointDescriptor(
        bcp_index=bcp_idx,
        atom_1_index=atom_1,
        atom_2_index=atom_2,
        coordinates_angstrom=coords,
        electron_density_au=round(rho, 6),
        laplacian_au=round(laplacian, 6),
        kinetic_energy_density_g_au=round(g_density, 6),
        potential_energy_density_v_au=round(v_density, 6),
        total_energy_density_h_au=round(h_density, 6),
        ellipticity=round(ellipticity, 6),
        interaction_classification=classification,
    )


def validate_poincare_hopf_invariant(
    n_ncp: int,
    n_bcp: int,
    n_rcp: int,
    n_ccp: int,
) -> bool:
    """
    [M] Enforce the global Poincaré-Hopf topological invariant on QTAIM critical points (REQ-QC-088).

    Invariant:
        n_NCP - n_BCP + n_RCP - n_CCP = 1 (for an isolated finite molecule in R^3)

    Where:
        - n_NCP: Number of Nuclear Critical Points (3,-3)
        - n_BCP: Number of Bond Critical Points (3,-1)
        - n_RCP: Number of Ring Critical Points (3,+1)
        - n_CCP: Number of Cage Critical Points (3,+3)

    Args:
        n_ncp: Count of Nuclear Critical Points.
        n_bcp: Count of Bond Critical Points.
        n_rcp: Count of Ring Critical Points.
        n_ccp: Count of Cage Critical Points.

    Returns:
        True if the topological relationship is satisfied.

    Raises:
        QTAIMTopologyError: If Euler characteristic != 1.
    """
    euler_char = n_ncp - n_bcp + n_rcp - n_ccp
    if euler_char != 1:
        raise QTAIMTopologyError(
            n_ncp=n_ncp,
            n_bcp=n_bcp,
            n_rcp=n_rcp,
            n_ccp=n_ccp,
            euler_char=euler_char,
        )
    return True
