#!/usr/bin/env python3
"""
Facade Module for CoChem-SpycFit: Quantum Extractor & Spectroscopic Fitter Module
This is a Facade Module that provides a unified interface by re-exporting the extractor and master modules.
Exposes QuantumExtractor class to parse real MPQC/g-xTB calculations and calculate
rotational constants (A, B, C in MHz) and dipole components (mu_a, mu_b, mu_c).
"""
import logging
logger = logging.getLogger(__name__)

import os
import sys
from pathlib import Path
from cochem_quantum_extractor import QuantumExtractor, parse_mpqc_freq_output, apply_mass_weighting
from cochem_spycfit_master import compute_vibrationally_averaged_constants, reweight_isotopologue_hessian

__all__ = ["QuantumExtractor", "parse_mpqc_freq_output", "apply_mass_weighting", "compute_vibrationally_averaged_constants", "reweight_isotopologue_hessian"]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1:
        path: str = sys.argv[1]
        import cochem_quantum_extractor
        extractor: cochem_quantum_extractor.QuantumExtractor = QuantumExtractor(path)
        res: dict = extractor.parse_output()
        logger.info(f"Extracted Quantum Observables: {res}")
    else:
        logger.info("CoChem-SpycFit Quantum Extractor module loaded.")