"""
CoChem S22 / S66 Non-Covalent Benchmark Verification & Deformation Prep Energy.

Provenance:
- Method Matrix [M]: Canonical CCSD(T)/CBS benchmark reference comparisons (REQ-BM-043).
- Derived Relations [D]: Counterpoise binding energy Delta E_bind_CP = Delta E_int_CP + Delta E_prep,
  3D principal moment of inertia 5% tolerance window.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np
from pydantic import BaseModel, Field

from cochem.core.cochem_exceptions import BenchmarkMotifMismatchError
from cochem.topos.cochem_topos_crusher import compute_principal_moments_of_inertia

HARTREE_TO_KCAL_MOL = 627.509474

# Canonical S22 reference dataset: CCSD(T)/CBS binding energies (kcal/mol) and canonical reference moments of inertia (amu*Å^2)
S22_REFERENCE_DATA: Dict[str, Dict[str, Any]] = {
    "S22-01": {
        "name": "Water dimer (C_s)",
        "formula_a": "H2O",
        "formula_b": "H2O",
        "reference_energy_kcal_mol": -4.99,
        "ref_moments_amu_ang2": (2.36007, 77.54238, 77.57357),
    },
    "S22-02": {
        "name": "Formic acid dimer (C_2h)",
        "formula_a": "HCOOH",
        "formula_b": "HCOOH",
        "reference_energy_kcal_mol": -18.75,
        "ref_moments_amu_ang2": (98.40, 245.10, 343.50),
    },
    "S22-08": {
        "name": "Methane dimer (D_3d)",
        "formula_a": "CH4",
        "formula_b": "CH4",
        "reference_energy_kcal_mol": -0.53,
        "ref_moments_amu_ang2": (42.10, 158.40, 158.40),
    },
    "S22-11": {
        "name": "Benzene-Methane (C_s)",
        "formula_a": "C6H6",
        "formula_b": "CH4",
        "reference_energy_kcal_mol": -1.45,
        "ref_moments_amu_ang2": (175.20, 310.50, 485.70),
    },
}


class BenchmarkComparisonResult(BaseModel):
    """Data contract reporting S22/S66 benchmark binding energy and geometric verification."""

    benchmark_id: str
    formula_monomer_a: str
    formula_monomer_b: str
    delta_e_int_cp_kcal_mol: float
    delta_e_prep_kcal_mol: float
    delta_e_bind_cp_kcal_mol: float
    reference_energy_kcal_mol: float
    absolute_relative_error_pct: float
    inertia_tensor_matched: bool


def compute_counterpoise_binding_energy(
    e_complex_dimer_basis_hartree: float,
    e_monomer_a_dimer_basis_hartree: float,
    e_monomer_b_dimer_basis_hartree: float,
    e_monomer_a_monomer_basis_hartree: float,
    e_monomer_a_opt_hartree: float,
    e_monomer_b_monomer_basis_hartree: float,
    e_monomer_b_opt_hartree: float,
) -> Tuple[float, float, float]:
    """
    [REQ-BM-043] Compute Counterpoise interaction energy, monomer deformation prep energy, and total binding energy.

    Formulas:
        Delta E_int_CP = E_AB(AB) - E_A(AB) - E_B(AB)  [Hartree -> kcal/mol]
        Delta E_prep = (E_A(A) - E_A_opt(A)) + (E_B(B) - E_B_opt(B))  [Hartree -> kcal/mol]
        Delta E_bind_CP = Delta E_int_CP + Delta E_prep

    Returns:
        Tuple of (delta_e_int_cp_kcal_mol, delta_e_prep_kcal_mol, delta_e_bind_cp_kcal_mol).
    """
    # Interaction energy in Hartree
    delta_e_int_hartree = (
        e_complex_dimer_basis_hartree
        - e_monomer_a_dimer_basis_hartree
        - e_monomer_b_dimer_basis_hartree
    )

    # Deformation prep energy in Hartree
    delta_e_prep_hartree = (
        (e_monomer_a_monomer_basis_hartree - e_monomer_a_opt_hartree)
        + (e_monomer_b_monomer_basis_hartree - e_monomer_b_opt_hartree)
    )

    delta_e_bind_hartree = delta_e_int_hartree + delta_e_prep_hartree

    return (
        delta_e_int_hartree * HARTREE_TO_KCAL_MOL,
        delta_e_prep_hartree * HARTREE_TO_KCAL_MOL,
        delta_e_bind_hartree * HARTREE_TO_KCAL_MOL,
    )


def verify_benchmark_complex(
    benchmark_id: str,
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    delta_e_bind_cp_kcal_mol: float,
    delta_e_int_cp_kcal_mol: Optional[float] = None,
    delta_e_prep_kcal_mol: Optional[float] = None,
    tolerance_pct: float = 5.0,
) -> BenchmarkComparisonResult:
    """
    [REQ-BM-043] Validate complex against canonical benchmark reference with 5% moment of inertia check.

    Args:
        benchmark_id: Canonical identifier (e.g. 'S22-01').
        symbols: Complex atomic symbols.
        coordinates_angstrom: Complex 3D geometry in Å.
        delta_e_bind_cp_kcal_mol: Calculated total CP binding energy in kcal/mol.
        delta_e_int_cp_kcal_mol: Interaction energy in kcal/mol.
        delta_e_prep_kcal_mol: Deformation prep energy in kcal/mol.
        tolerance_pct: Allowable deviation percentage for moments of inertia (default: 5.0%).

    Returns:
        BenchmarkComparisonResult instance.

    Raises:
        BenchmarkMotifMismatchError: If moments of inertia deviate > tolerance_pct from reference.
    """
    if benchmark_id not in S22_REFERENCE_DATA:
        raise ValueError(f"Unknown benchmark ID '{benchmark_id}'. Available: {list(S22_REFERENCE_DATA.keys())}")

    ref_entry = S22_REFERENCE_DATA[benchmark_id]
    ref_energy = ref_entry["reference_energy_kcal_mol"]
    ref_moments = ref_entry["ref_moments_amu_ang2"]

    # Calculate actual principal moments of inertia
    calc_moments, _ = compute_principal_moments_of_inertia(symbols, coordinates_angstrom)

    # Compute maximum percentage deviation among the 3 principal moments
    deviations = [
        abs(c - r) / r * 100.0
        for c, r in zip(calc_moments, ref_moments)
        if r > 0
    ]
    max_dev = max(deviations) if deviations else 0.0

    if max_dev > tolerance_pct:
        raise BenchmarkMotifMismatchError(
            benchmark_id=benchmark_id,
            deviation_pct=max_dev,
            calculated_inertia=calc_moments,
            reference_inertia=ref_moments,
        )

    # Calculate relative error in binding energy
    rel_error_pct = abs(delta_e_bind_cp_kcal_mol - ref_energy) / abs(ref_energy) * 100.0

    e_int = delta_e_int_cp_kcal_mol if delta_e_int_cp_kcal_mol is not None else delta_e_bind_cp_kcal_mol
    e_prep = delta_e_prep_kcal_mol if delta_e_prep_kcal_mol is not None else 0.0

    return BenchmarkComparisonResult(
        benchmark_id=benchmark_id,
        formula_monomer_a=ref_entry["formula_a"],
        formula_monomer_b=ref_entry["formula_b"],
        delta_e_int_cp_kcal_mol=e_int,
        delta_e_prep_kcal_mol=e_prep,
        delta_e_bind_cp_kcal_mol=delta_e_bind_cp_kcal_mol,
        reference_energy_kcal_mol=ref_energy,
        absolute_relative_error_pct=rel_error_pct,
        inertia_tensor_matched=True,
    )
