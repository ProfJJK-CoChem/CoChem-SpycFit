#!/usr/bin/env python3
"""
Facade Module for CoChem-SpycFit: Quantum Extractor & Spectroscopic Fitter Module
This is a Facade Module that provides a unified interface by re-exporting the extractor and master modules.
Exposes QuantumExtractor class to parse real MPQC/g-xTB calculations and calculate
rotational constants (A, B, C in MHz) and dipole components (mu_a, mu_b, mu_c).
"""
import logging
import sys
import os
from pathlib import Path

# Setup rigorous logging
logger = logging.getLogger(__name__)

from cochem_spycfit.quantum_extractor import (
    QuantumExtractor,
    parse_mpqc_freq_output,
    apply_mass_weighting,
    ExtractedOutput
)
from cochem_spycfit.master import (
    compute_vibrationally_averaged_constants,
    reweight_isotopologue_hessian
)

__all__ = [
    "QuantumExtractor",
    "parse_mpqc_freq_output",
    "apply_mass_weighting",
    "compute_vibrationally_averaged_constants",
    "reweight_isotopologue_hessian"
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    
    if len(sys.argv) > 1:
        path_str: str = sys.argv[1]
        target_path: Path = Path(path_str).resolve()
        
        if not target_path.exists():
            logger.error(f"[MISSING DATA] Target path does not exist: {target_path}")
            sys.exit(1)
            
        extractor: QuantumExtractor = QuantumExtractor(str(target_path))
        try:
            res: ExtractedOutput = extractor.parse_output()
            # Pydantic v2 uses model_dump_json, fallback to dict if needed
            if hasattr(res, "model_dump_json"):
                logger.info(f"Extracted Quantum Observables: {res.model_dump_json(indent=2)}")
            else:
                logger.info(f"Extracted Quantum Observables: {res}")
        except Exception as e:
            logger.error(f"Failed to extract quantum observables: {e}")
            sys.exit(1)
    else:
        logger.error("[MISSING DATA] No path provided. Expected file path as sys.argv[1].")
        sys.exit(1)