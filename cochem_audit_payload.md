Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\CoChem-SpycFit\.in-progress\Task6_Prompt2_jax_builder.md.
Original prompt:
# Task 6 (Phase 3): Native JAX Builder - cochem_jax_builder.py

**Objective:**
Create `cochem_jax_builder.py` to construct Watson asymmetric rotor matrices natively utilizing hardware-agnostic JAX acceleration.

**Target File:**
`D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\physics_core\cochem_jax_builder.py`

**Instructions:**
You are the `cochem-coder` agent. Implement `cochem_jax_builder.py`.
It must:
1. Use `jax.pmap` or `jax.sharding` by default. Dynamically elevate to `mpi4jax` ONLY if the HPC flag is detected.
2. Calculate Ray's Asymmetry Parameter ($\kappa$). Switch representation to $III^r$ if $\kappa > 0.5$. Enforce a hysteresis buffer by checking $\kappa_{prev}$.
3. Implement J-block batching (grouping blocks into compilation buckets like J=1-10) to satisfy JAX XLA static shape constraints without $O(N^3)$ computational waste.
4. Implement the Internal Rotation (LAM) Guard. Poll the 1D-torsional barrier ($V_3, V_6$). If a hyperparameter threshold is breached, halt rigid-rotor construction and route to the IAM/Woods Method block.

**Constraints:**
- No mocks, placeholders, or dummy execution.
- Actually implement the matrix construction logic.
- Do NOT use `pass` statements in your generated code.

**Proposed Snippet Outline:**
```python
import jax
import jax.numpy as jnp
import os

def determine_representation(A, B, C, prev_kappa=None):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def build_j_block_batches(J_max, bucket_size=10):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")

def construct_hamiltonian(constants, J_max):
    # MUST IMPLEMENT FULL LOGIC HERE - NO PASS STATEMENTS OR STUBS
    raise NotImplementedError("Implementation required.")
```

Modified files content:

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.