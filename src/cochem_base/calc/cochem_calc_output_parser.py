"""
Multi-Engine Quantum Chemistry Output Extractor & QCSchema Normalizer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
import numpy as np

from cochem_base.core.models import QCResultsRecord


class QuantumOutputParser:
    """
    Parses output streams and binary matrix files from ORCA, CFOUR, and xTB.
    """

    @classmethod
    def parse_orca_output(cls, log_text: str, hess_text: Optional[str] = None) -> QCResultsRecord:
        """Parse ORCA output log text and Hessian file into QCResultsRecord."""
        # 1. Total Energy
        energy: Optional[float] = None
        for m in re.finditer(r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+(-?\d+\.\d+)", log_text):
            energy = float(m.group(1))
        if energy is None:
            for m in re.finditer(r"Total\s+Energy\s+:\s+(-?\d+\.\d+)", log_text):
                energy = float(m.group(1))
        if energy is None:
            raise ValueError("Could not extract final electronic energy from ORCA log.")

        # 2. Dipole moment
        dipole: Optional[List[float]] = None
        m_dip = re.search(r"Total\s+Dipole\s+Moment\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", log_text)
        if m_dip:
            dipole = [float(m_dip.group(1)), float(m_dip.group(2)), float(m_dip.group(3))]

        # 3. Rotational constants (in GHz)
        rot_consts: Optional[List[float]] = None
        m_rot = re.search(r"Rotational\s+constants\s+in\s+GHz\s+:\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)", log_text)
        if m_rot:
            rot_consts = [float(m_rot.group(1)), float(m_rot.group(2)), float(m_rot.group(3))]

        # 4. Frequencies
        freqs: List[float] = []
        in_freq_block = False
        for line in log_text.splitlines():
            if "VIBRATIONAL FREQUENCIES" in line:
                in_freq_block = True
                continue
            if in_freq_block:
                if "NORMAL MODES" in line or "IR SPECTRUM" in line or ("-----------" in line and len(freqs) > 0):
                    in_freq_block = False
                m_f = re.match(r"\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1", line)
                if m_f:
                    freqs.append(float(m_f.group(1)))

        # 5. ZPVE
        zpve: Optional[float] = None
        m_z = re.search(r"Zero\s+point\s+energy\s+\.\.\.\s+(-?\d+\.\d+)\s+Eh", log_text)
        if m_z:
            zpve = float(m_z.group(1))

        # 6. Parse Cartesian Hessian matrix from ORCA .hess text block
        hessian_matrix: Optional[List[List[float]]] = None
        if hess_text and "$hessian" in hess_text:
            try:
                lines = [l.strip() for l in hess_text.splitlines() if l.strip()]
                dim = 0
                start_data_idx = 0
                for idx, line in enumerate(lines):
                    if line.startswith("$hessian"):
                        if idx + 1 < len(lines):
                            dim = int(lines[idx + 1])
                            start_data_idx = idx + 2
                            break
                if dim > 0:
                    mat = np.zeros((dim, dim), dtype=float)
                    curr_idx = start_data_idx
                    while curr_idx < len(lines) and not lines[curr_idx].startswith("$"):
                        col_indices = [int(c) for c in lines[curr_idx].split() if c.isdigit()]
                        if not col_indices:
                            curr_idx += 1
                            continue
                        curr_idx += 1
                        for _ in range(dim):
                            if curr_idx >= len(lines) or lines[curr_idx].startswith("$"):
                                break
                            parts = lines[curr_idx].split()
                            if len(parts) >= len(col_indices) + 1:
                                row_idx = int(parts[0])
                                vals = [float(x) for x in parts[1 : len(col_indices) + 1]]
                                for col_i, val in zip(col_indices, vals):
                                    mat[row_idx, col_i] = val
                            curr_idx += 1
                    hessian_matrix = mat.tolist()
            except Exception:
                hessian_matrix = None

        return QCResultsRecord(
            energy_hartree=energy,
            dipole_moment_debye=dipole,
            rotational_constants_ghz=rot_consts,
            frequencies_cm1=freqs if freqs else None,
            zpve_hartree=zpve,
            hessian=hessian_matrix,
            provenance={"engine": "ORCA"},
        )

    @classmethod
    def parse_cfour_output(cls, log_text: str) -> QCResultsRecord:
        """Parse CFOUR output log."""
        energy: Optional[float] = None
        for m in re.finditer(r"E\(CCSD\(T\)\)\s*=\s*(-?\d+\.\d+)", log_text):
            energy = float(m.group(1))
        if energy is None:
            for m in re.finditer(r"The\s+energy\s+is\s+(-?\d+\.\d+)", log_text):
                energy = float(m.group(1))
        if energy is None:
            raise ValueError("Could not extract energy from CFOUR log.")

        return QCResultsRecord(
            energy_hartree=energy,
            provenance={"engine": "CFOUR"},
        )

    @classmethod
    def parse_xtb_output(cls, log_text: str) -> QCResultsRecord:
        """Parse GFN-xTB calculation log."""
        energy: Optional[float] = None
        for m in re.finditer(r"::\s+total\s+energy\s+(-?\d+\.\d+)\s+Eh", log_text):
            energy = float(m.group(1))
        if energy is None:
            for m in re.finditer(r"TOTAL\s+ENERGY\s+(-?\d+\.\d+)\s+Eh", log_text):
                energy = float(m.group(1))
        if energy is None:
            raise ValueError("Could not extract final electronic energy from xTB log.")

        return QCResultsRecord(
            energy_hartree=energy,
            provenance={"engine": "xTB"},
        )
