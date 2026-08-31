"""
CoChem High-Performance CFOUR ZMAT & Input Generator.

Provenance:
- Method Matrix [M]: Standardized Cartesian/ZMAT input blocks, dynamic CIAAW atomic coordinates,
  Abelian subgroup symmetry detection, and MEMORY_SIZE allocation (REQ-QC-057).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from cochem.core.cochem_elements import get_atomic_number
from cochem.qc.cochem_symmetry import align_to_principal_axes


def generate_cfour_zmat_deck(
    symbols: Sequence[str],
    coordinates_angstrom: Sequence[Tuple[float, float, float]],
    point_group: str = "C1",
    calc_level: str = "CCSD(T)",
    basis: str = "PVTZ",
    memory_mb: int = 4000,
    reference: str = "RHF",
    charge: int = 0,
    multiplicity: int = 1,
    title: str = "CoChem CFOUR Job",
) -> str:
    """
    [REQ-QC-057] Assemble a complete CFOUR ZMAT input file with dynamic symmetry and memory keywords.

    Args:
        symbols: Chemical element symbols.
        coordinates_angstrom: Cartesian coordinates in Å.
        point_group: Detected Schoenflies point group string.
        calc_level: Post-SCF electronic correlation level (e.g. 'CCSD(T)', 'CCSD', 'HF').
        basis: CFOUR basis set name (e.g. 'PVTZ', 'PVDZ', 'ANO0').
        memory_mb: Allocated memory size in MB.
        reference: Reference wavefunction ('RHF', 'UHF', 'ROHF').
        charge: Total net molecular charge.
        multiplicity: Total spin multiplicity.
        title: Header comment line.

    Returns:
        Formatted CFOUR ZMAT string.
    """
    # 1. Align coordinates along principal axes for Abelian symmetry
    if point_group and point_group != "C1":
        _, aligned_coords, _ = align_to_principal_axes(symbols, coordinates_angstrom)
        coords_to_use = [tuple(row) for row in aligned_coords]
        symmetry_flag = "ON"
    else:
        coords_to_use = list(coordinates_angstrom)
        symmetry_flag = "OFF"

    # Multiplicity to UHF if open-shell
    ref = reference
    if multiplicity > 1 and ref == "RHF":
        ref = "UHF"

    lines: List[str] = [
        title,
    ]

    # Cartesian coordinate block
    for sym, (x, y, z) in zip(symbols, coords_to_use):
        lines.append(f"{sym:<3} {x:14.8f} {y:14.8f} {z:14.8f}")

    lines.append("")
    lines.append("*CFOUR(")
    lines.append(f"CALC_LEVEL={calc_level.upper()}")
    lines.append(f"BASIS={basis.upper()}")
    lines.append(f"SYMMETRY={symmetry_flag}")
    lines.append(f"MEMORY_SIZE={memory_mb}")
    lines.append(f"REF={ref}")
    if charge != 0:
        lines.append(f"CHARGE={charge}")
    if multiplicity != 1:
        lines.append(f"MULTIPLICITY={multiplicity}")
    lines.append("COORD=CARTESIAN")
    lines.append("UNITS=ANGSTROM")
    lines.append(")")

    return "\n".join(lines) + "\n"
