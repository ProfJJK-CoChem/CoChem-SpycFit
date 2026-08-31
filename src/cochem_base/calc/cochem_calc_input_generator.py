"""
Multi-Engine Quantum Chemistry Input Deck Generator (ORCA, CFOUR, xTB, PySCF).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from cochem_base.core.models import MolecularTopology
from cochem_base.core.metadata import get_atomic_symbol
from cochem_base.core.glossary import UnitConversionConstants


class QuantumInputGenerator:
    """
    Constructs syntactically compliant input decks for quantum chemistry codes.
    """

    @classmethod
    def generate_orca_deck(
        cls,
        topology: MolecularTopology,
        method: str = "r2SCAN-3c",
        job_type: str = "Opt NumFreq",
        maxcore_mb: int = 4000,
        nprocs: int = 4,
    ) -> str:
        """Generate official ORCA input deck."""
        coords_ang = topology.get_coordinates_angstrom()
        lines = [
            f"! {method} {job_type}",
            f"%maxcore {maxcore_mb}",
            f"%pal nprocs {nprocs} end",
            f"* xyz {topology.charge} {topology.multiplicity}",
        ]
        for idx, (z, (x, y, zc)) in enumerate(zip(topology.atomic_numbers, coords_ang)):
            sym = topology.symbols[idx] if topology.symbols else get_atomic_symbol(z)
            lines.append(f"  {sym:<3} {x:14.8f} {y:14.8f} {zc:14.8f}")
        lines.append("*")
        return "\n".join(lines) + "\n"

    @classmethod
    def generate_cfour_deck(
        cls,
        topology: MolecularTopology,
        method: str = "CCSD(T)",
        basis: str = "cc-pVTZ",
        calc_type: str = "GRADIENT",
    ) -> str:
        """Generate CFOUR ZMAT Cartesian input deck."""
        coords_ang = topology.get_coordinates_angstrom()
        lines = [
            "CoChem Auto-Generated CFOUR Deck",
        ]
        for idx, (z, (x, y, zc)) in enumerate(zip(topology.atomic_numbers, coords_ang)):
            sym = topology.symbols[idx] if topology.symbols else get_atomic_symbol(z)
            lines.append(f"{sym:<3} {x:14.8f} {y:14.8f} {zc:14.8f}")
        lines.append("")
        lines.append(f"*CFOUR(CALC={method},BASIS={basis},DERIV={calc_type}")
        lines.append(f"CHARGE={topology.charge},MULT={topology.multiplicity}")
        lines.append("REF=RHF,MEMORY=4,MEM_UNIT=GB)")
        lines.append("")
        return "\n".join(lines) + "\n"

    @classmethod
    def generate_xtb_command(
        cls,
        coord_file: Path,
        gfn_version: int = 2,
        opt: bool = True,
        freq: bool = False,
        charge: int = 0,
        unpaired: int = 0,
    ) -> List[str]:
        """Construct xTB CLI command arguments."""
        cmd = ["xtb", str(coord_file), "--gfn", str(gfn_version), "--chrg", str(charge), "--uhf", str(unpaired)]
        if opt:
            cmd.append("--opt")
        if freq:
            cmd.append("--hess")
        return cmd

    @classmethod
    def generate_pyscf_script(
        cls,
        topology: MolecularTopology,
        xc: str = "B3LYP",
        basis: str = "def2-TZVP",
    ) -> str:
        """Generate standalone Python PySCF execution script."""
        coords_ang = topology.get_coordinates_angstrom()
        atom_entries = []
        for idx, (z, (x, y, zc)) in enumerate(zip(topology.atomic_numbers, coords_ang)):
            sym = topology.symbols[idx] if topology.symbols else get_atomic_symbol(z)
            atom_entries.append(f"    '{sym} {x:.8f} {y:.8f} {zc:.8f}'")
        atom_block = ",\n".join(atom_entries)

        code_lines = [
            "from pyscf import gto, dft",
            "",
            "mol = gto.Mole()",
            "mol.atom = [",
            atom_block,
            "]",
            f"mol.basis = '{basis}'",
            f"mol.charge = {topology.charge}",
            f"mol.spin = {topology.multiplicity - 1}",
            "mol.build()",
            "",
            "mf = dft.RKS(mol)",
            f"mf.xc = '{xc}'",
            "energy = mf.kernel()",
            "print(f'TOTAL_ENERGY_HARTREE = {energy:.12f}')",
        ]
        return "\n".join(code_lines) + "\n"
