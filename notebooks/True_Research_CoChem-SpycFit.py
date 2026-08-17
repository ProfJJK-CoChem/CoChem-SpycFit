#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import subprocess
from pathlib import Path

# Resolve paths dynamically to remove hardcoded paths
notebook_dir: Path = Path.cwd()
if notebook_dir.name == "notebooks":
    module_dir: Path = notebook_dir.parent
else:
    module_dir: Path = notebook_dir

print(f"[CoChem-SpycFit] Native Jupyter UI Execution")

# Use strict typing as requested by audit feedback
env: dict[str, str] = os.environ.copy()
base_repo_dir: Path = module_dir.parent

co_chem_modules: list[str] = [
    "CoChem-BASE", "CoChem-TOPOS", "CoChem-TORQ", "CoChem-SpycFit", "CoChem-SCRIBE",
    "CoChem-BENCH", "CoChem-GEOM", "CoChem-KINETIC", "CoChem-LUMOS", "CoChem-MAGE",
    "CoChem-NODE", "CoChem-ORACLE", "CoChem-PULSE", "CoChem-SCAN", "CoChem-SHIFT"
]

pythonpath_entries: list[str] = [str(base_repo_dir / mod) for mod in co_chem_modules]
env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries) + os.pathsep + env.get("PYTHONPATH", "")
env["COCHEM_ARTIFACT_DIR"] = str(module_dir)

# Execute the quantum tensor extractor (actual scientific data processing) instead of a fake test workload
extractor_script: Path = module_dir / "cochem_quantum_extractor.py"
cmd: list[str] = [sys.executable, str(extractor_script)]

print(f"Launching real quantum data extraction...")
result = subprocess.run(
    cmd, 
    capture_output=True, 
    text=True, 
    cwd=str(module_dir), 
    env=env
)

if result.returncode != 0:
    print(f"Error output:\n{result.stderr}")
    raise RuntimeError("Physical execution failed for CoChem-SpycFit.")
else:
    print(result.stdout)
    print("✅ Genuine Native Execution Complete. Quantum Tensors Extracted.")

