Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task5_Prompt2_triage_phase2.md.
Original prompt:
# Task 5 (Phase 2): Experimental Triage - cochem_spycfit_triage.py

**Objective:**
Update `cochem_spycfit_triage.py` to implement Phase 2 requirements including the Isotope Natural Abundance Filter, Headless Degradation, Resonance Trapping, and Eckart Tracking.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\intake\cochem_spycfit_triage.py`

**Instructions:**
You are the `cochem-coder` agent. Implement Phase 2 features in `cochem_spycfit_triage.py`.
It must:
1. Implement the Isotope Natural Abundance Filter (< 0.01%) via `mendeleev`. Include a HITL trigger, but enforce Headless Degradation if `COCHEM_HEADLESS=1` is detected (auto-quarantine and raise exception).
2. Scan theoretical frequencies from `landscape.zarr` for Fermi/Coriolis resonance conditions. Emit asynchronous heartbeats during this scan to prevent HPC timeouts. Alert the Orchestrator via ZeroMQ or Zarr state locks to construct a Polyad matrix.
3. Detect Axis Reorientation (Eckart Tracking) upon isotopic substitution (determinant swap) and automatically re-index dipole derivative matrices.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Read real system environment variables (`os.environ`).
- Must actually implement the math for Eckart tracking.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import os
from mendeleev import element
import numpy as np
import zarr

def check_isotope_abundance(isotopologue):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def scan_resonance_traps(frequencies):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def eckart_axis_reorientation(dipole_derivatives, old_inertia, new_inertia):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.