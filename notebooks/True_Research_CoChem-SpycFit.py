#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolve paths dynamically to remove hardcoded paths
notebook_dir = Path(__file__).parent.resolve() if '__file__' in globals() else Path.cwd()
if notebook_dir.name == "notebooks":
    module_dir = notebook_dir.parent
else:
    module_dir = notebook_dir

logger.info(f"[CoChem-SpycFit] Native UI Execution")

# Set up environment variables BEFORE importing modules to ensure paths are correct
if not os.environ.get("COCHEM_ARTIFACT_DIR"):
    artifact_dir = Path.home() / ".cochem" / "artifacts"
    os.environ["COCHEM_ARTIFACT_DIR"] = str(artifact_dir)

if not os.environ.get("COCHEM_TARGET_ISOMER"):
    os.environ["COCHEM_TARGET_ISOMER"] = "isomer_001"

# Ensure module is in path
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

# Import the actual domain logic
try:
    from cochem_spycfit import quantum_extractor
    from cochem_spycfit import convolver
except ImportError as e:
    logger.error(f"[MISSING DATA] Failed to import domain logic modules. Ensure they are present in {module_dir}. Reason: {e}")
    sys.exit(1)

old_argv = sys.argv.copy()
sys.argv = [old_argv[0]]

try:
    logger.info("Launching real quantum data extraction...")
    # Execute native extraction natively without subprocess.run
    try:
        quantum_extractor.main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError("Physical execution failed for quantum extractor.") from e

    logger.info("Running Bayesian convolution...")
    # Execute native convolution
    try:
        convolver.main()
    except SystemExit as e:
        if e.code != 0:
            raise RuntimeError("Physical execution failed for convolver.") from e

    logger.info("[SUCCESS] Genuine Native Execution Complete. Quantum Tensors Extracted and Convolved.")
finally:
    sys.argv = old_argv
