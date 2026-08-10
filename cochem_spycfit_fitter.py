#!/usr/bin/env python3
"""
CoChem-SpycFit: Quantum Extractor & Spectroscopic Fitter Module
Exposes QuantumExtractor class to parse real MPQC/g-xTB calculations and calculate
rotational constants (A, B, C in MHz) and dipole components (mu_a, mu_b, mu_c).
"""

import os
import sys
from pathlib import Path
from cochem_quantum_extractor import QuantumExtractor, parse_mpqc_freq_output, apply_mass_weighting
from cochem_spycfit_master import compute_vibrationally_averaged_constants, reweight_isotopologue_hessian

__all__ = ["QuantumExtractor", "parse_mpqc_freq_output", "apply_mass_weighting", "compute_vibrationally_averaged_constants", "reweight_isotopologue_hessian"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = sys.argv[1]
        extractor = QuantumExtractor(path)
        res = extractor.parse_output()
        print("Extracted Quantum Observables:", res)
    else:
        print("CoChem-SpycFit Quantum Extractor module loaded.")
