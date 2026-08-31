"""
CoChem Quantum Chemistry Output & Hessian Parser.

Provenance:
- Method Matrix [M]: Frequency threshold for stationary minimum vs saddle point (nu_1 >= -15.0 cm^-1).
- Derived Relations [D]: 3N-6 / 3N-5 vibrational degrees of freedom filter, filtering external translational/rotational modes (|nu| <= 10.0 cm^-1).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class ParsedQcOutput(BaseModel):
    """Data contract representing extracted quantum chemistry properties and vibrational analysis."""

    total_energy_hartree: Optional[float] = None
    scf_converged: bool = False
    num_atoms: int = Field(default=0, ge=0)
    is_linear: bool = False
    expected_vibrational_dof: int = 0
    all_frequencies_cm1: List[float] = Field(default_factory=list)
    vibrational_frequencies_cm1: List[float] = Field(default_factory=list)
    lowest_internal_frequency_cm1: Optional[float] = None
    is_minimum: bool = True
    is_saddle_point: bool = False
    dipole_moment_debye: Optional[Tuple[float, float, float]] = None
    dipole_total_debye: Optional[float] = None
    rotational_constants_ghz: Optional[Tuple[float, float, float]] = None
    normal_modes: Optional[List[List[float]]] = None  # (N_modes, 3*N_atoms)


def parse_orca_hessian_text(hess_text: str) -> Tuple[List[float], np.ndarray]:
    """
    Parse vibrational frequencies and normal mode eigenvectors from an ORCA .hess file content.

    Args:
        hess_text: Raw text of the ORCA .hess file.

    Returns:
        Tuple of (list_of_frequencies_cm1, normal_modes_matrix).
    """
    freqs: List[float] = []

    # 1. Parse $vibrational_frequencies
    freq_match = re.search(r"\$vibrational_frequencies\s+(\d+)(.*?)(?=\$|\Z)", hess_text, re.DOTALL)
    if freq_match:
        n_freqs = int(freq_match.group(1))
        lines = freq_match.group(2).strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].isdigit():
                try:
                    val = float(parts[1].replace("D", "E").replace("d", "e"))
                    freqs.append(val)
                except ValueError:
                    pass

    # 2. Parse $normal_modes
    modes_match = re.search(r"\$normal_modes\s+(\d+)\s+(\d+)(.*?)(?=\$|\Z)", hess_text, re.DOTALL)
    if modes_match and freqs:
        dim1 = int(modes_match.group(1))
        dim2 = int(modes_match.group(2))
        matrix = np.zeros((dim1, dim2), dtype=np.float64)

        # Parse column-blocked matrix
        block_text = modes_match.group(3)
        col_blocks = re.findall(r"(\s+\d+(?:\s+\d+)*)\n((?:(?:\s*\d+\s+[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?.*?\n)+))", block_text)

        for col_header, rows_str in col_blocks:
            cols = [int(c) for c in col_header.strip().split()]
            for row_line in rows_str.strip().splitlines():
                parts = row_line.strip().split()
                if not parts:
                    continue
                row_idx = int(parts[0])
                vals = [float(p.replace("D", "E").replace("d", "e")) for p in parts[1:]]
                for c_idx, val in zip(cols, vals):
                    if row_idx < dim1 and c_idx < dim2:
                        matrix[row_idx, c_idx] = val

        # Transpose so rows correspond to modes (dim2 modes, each with dim1 components)
        normal_modes = matrix.T
    else:
        normal_modes = np.empty((0, 0), dtype=np.float64)

    return freqs, normal_modes


def parse_orca_output_text(output_text: str, is_linear: bool = False) -> ParsedQcOutput:
    """
    [REQ-QC-042, REQ-THM-045] Extract energy, frequencies, dipole, and classify stationary points.

    Args:
        output_text: Raw string content of ORCA .out log.
        is_linear: Whether the molecular system is linear (determines 3N-5 vs 3N-6 DOF).

    Returns:
        ParsedQcOutput model.
    """
    total_energy: Optional[float] = None
    scf_converged = "SCF CONVERGED AFTER" in output_text or "SUCCESSFULLY CONVERGED" in output_text

    # Match Total Energy: e.g. "FINAL SINGLE POINT ENERGY      -154.87239487"
    energy_match = re.search(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", output_text)
    if energy_match:
        total_energy = float(energy_match.group(1))
    else:
        # Alt match: "Total Energy       :    -154.87239487 Eh"
        alt_energy = re.search(r"Total Energy\s*:\s*([-+]?\d+\.\d+)", output_text)
        if alt_energy:
            total_energy = float(alt_energy.group(1))

    # Match number of atoms: "Number of atoms ................ 6"
    natoms_match = re.search(r"Number of atoms\s*[\.:\s]*\s*(\d+)", output_text, re.IGNORECASE)
    num_atoms = int(natoms_match.group(1)) if natoms_match else 0

    # Match dipole moment: "Total Dipole Moment    :     1.8542"
    dipole_match = re.search(r"Total Dipole Moment\s*:\s*([-+]?\d+\.\d+)", output_text)
    dipole_total = float(dipole_match.group(1)) if dipole_match else None

    # Match dipole vector: "Magnitude (Debye)\s*:\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)"
    dip_vec_match = re.search(r"Magnitude \(Debye\)\s*:\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", output_text)
    dipole_vec = None
    if dip_vec_match:
        dipole_vec = (float(dip_vec_match.group(1)), float(dip_vec_match.group(2)), float(dip_vec_match.group(3)))

    # Match rotational constants: "Rotational constants in GHz\s*:\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)"
    rot_match = re.search(r"Rotational constants in GHz\s*:\s*([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)", output_text)
    rot_consts = None
    if rot_match:
        rot_consts = (float(rot_match.group(1)), float(rot_match.group(2)), float(rot_match.group(3)))

    # Parse frequencies
    all_freqs = [
        float(m.group(2))
        for m in re.finditer(r"^\s*(\d+):\s*([-+]?\d+\.\d+)\s*cm", output_text, re.MULTILINE)
    ]

    # Calculate expected vibrational DOF
    if num_atoms == 0:
        if all_freqs:
            num_atoms = max(1, len(all_freqs) // 3)

    if num_atoms <= 1:
        expected_dof = 0
    elif is_linear:
        expected_dof = max(0, 3 * num_atoms - 5)
    else:
        expected_dof = max(0, 3 * num_atoms - 6)

    n_ext = 5 if is_linear else 6

    # Filter external translational/rotational modes (|nu| <= 10.0 cm^-1 or leading 5/6 modes)
    if len(all_freqs) > n_ext:
        # If the first n_ext modes are external ~ 0 cm^-1, take remaining
        candidate_internal = all_freqs[n_ext:]
        # Also check for any internal imaginary modes (< -10.0 cm^-1) in candidate
        vib_freqs = candidate_internal
    else:
        # Filter out near-zero modes (|nu| <= 10.0 cm^-1)
        vib_freqs = [f for f in all_freqs if abs(f) > 10.0]

    lowest_internal_freq = min(vib_freqs) if vib_freqs else (min(all_freqs) if all_freqs else None)

    # State classification: threshold -15.0 cm^-1 [M]
    if lowest_internal_freq is not None:
        is_minimum = lowest_internal_freq >= -15.0
        is_saddle_point = lowest_internal_freq < -15.0
    else:
        is_minimum = True
        is_saddle_point = False

    return ParsedQcOutput(
        total_energy_hartree=total_energy,
        scf_converged=scf_converged,
        num_atoms=num_atoms,
        is_linear=is_linear,
        expected_vibrational_dof=expected_dof,
        all_frequencies_cm1=all_freqs,
        vibrational_frequencies_cm1=vib_freqs,
        lowest_internal_frequency_cm1=lowest_internal_freq,
        is_minimum=is_minimum,
        is_saddle_point=is_saddle_point,
        dipole_moment_debye=dipole_vec,
        dipole_total_debye=dipole_total,
        rotational_constants_ghz=rot_consts,
    )
