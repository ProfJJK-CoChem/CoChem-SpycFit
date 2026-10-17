# CoChem-SpycFit Master Execution Workflow (2026-08-07)

This document maps the complete, step-by-step execution workflow for the `CoChem-SpycFit` module. This workflow transforms SpycFit into an interactive, JAX-accelerated Bayesian Active Learning platform designed specifically for resolving dense CP-FTMW rotational spectra.

## Stage 1: Environment Lock & Tensor Initialization
1. **JAX Pre-Flight:** `cochem_spycfit_init.py` locks the XLA compiler to strict `float64` precision via `JAX_ENABLE_X64=True`. This is a mandatory guardrail; operating in standard ML `float32` precision leads to catastrophic numerical degradation when diagonalizing large rotational partition functions.
2. **Memory-Mapped Assimilation:** The quantum extractor hooks into the massive `cochem_state.h5` shared tensor. To prevent GPU memory exhaustion when calculating states up to $J=80$, SpycFit enforces strict memory-mapped (`mmap`) batch streaming to the VRAM.
3. **SCRIBE Sync:** SpycFit pings the `CoChem-SCRIBE` background daemon to prepare for incremental data streaming.

## Stage 2: Prior Assembly (LAM vs Rigid Rotor)
1. **Metadata Branching:** SpycFit scans the HDF5 metadata for the complex currently being analyzed.
2. **Rigid Rotor Pathway (VPT2):** If the `LAM_TRIGGER_REQUIRED` flag is FALSE, SpycFit extracts the *ab initio* rotational constants ($A, B, C$), quartic centrifugal distortion constants ($D_J, D_{JK}, D_K, d_1, d_2$), and Coriolis coupling matrices. These serve as the rigid Bayesian priors.
3. **Large Amplitude Motion Pathway (Sinc-DVR):** If the LAM flag is TRUE, SpycFit bypasses standard constants. Instead, it extracts the exact 3D probability eigenvalues and A/E tunneling splittings pre-calculated by TORQ. It also extracts the theoretical Kraitchman $r_s$ substitution coordinates to act as geometric priors for the isotopologues.

## Stage 3: Interactive Bayesian Fitting (Human-in-the-Loop)
1. **Experimental Ingestion:** The user provides an initial `.lin` file containing the first few manually assigned CP-FTMW experimental frequencies.
2. **Levenberg-Marquardt Optimization:** The Bayesian optimizer attempts to fit the theoretical *ab initio* priors against the experimental dataset. 
3. **Physical Constraints Gate:**
   - **Decision Gate:** Does the optimizer attempt to step into unphysical parameter space (e.g., negative centrifugal distortion) to achieve a lower RMS error?
   - *Yes:* SpycFit instantly applies a strict mathematical penalty function, rejects the step, recalculates, and logs the penalty event to the HDF5 tensor for SCRIBE documentation with an `[E]` provenance tag.
4. **Information-Gain Transition Mapping (Active Learning):**
   - Once the initial fit converges, the JAX `vmap` convolver broadcasts the theoretical spectral grid.
   - Using the statistical variance matrix, SpycFit actively calculates the "Next Best Microwave Transitions."
   - The UI pauses and presents the user with a dynamic table of unassigned frequencies that have the highest theoretical intensity combined with the lowest prediction error.
5. **Iterative Assignment:** The user searches the experimental spectrum for the ML-suggested frequencies, assigns them, and reruns the optimizer. This loop repeats until the RMS error is minimized across all assignments.

## Stage 4: LOIO Cross-Validation
1. **Leave-One-Isotopologue-Out Routine:** Once the primary fit is finalized, SpycFit executes the LOIO statistical cross-validation.
2. **Variance Testing:** The engine systematically drops one isotopic species from the dataset, re-optimizes the fit, and calculates the variance leverage. 
3. **Leverage Reporting:** A box-and-whisker plot (via Plotly) is generated in the UI, proving to the user (and reviewers) that no single misassigned isotopologue is falsely skewing the global molecular structure.

## Stage 5: Final Serialization & SCRIBE Publication Handoff
1. **Legacy Pickett Assembly:** The `quantum_extractor` parses the final Bayesian posteriors and generates standard Pickett-formatted `.lin`, `.cat`, and `.fit` files, strictly embedding `[M]` and `[D]` provenance tags into the output metadata.
2. **HDF5 Commit & Zipping:** 
   - SpycFit flushes the final rotational constants, statistical variances, LOIO plots, and Kraitchman coordinates to `cochem_state.h5`.
   - The generated Pickett files are specifically bundled into `Legacy_Verification.zip` to allow classical spectroscopists to independently cross-validate the results.
3. **SCRIBE Notification:** The `CoChem-SCRIBE` background daemon acknowledges the payload commit. SCRIBE now possesses the complete dataset required to automatically generate the final LaTeX publication methodology, BibTeX references, Supporting Information (SI) tables, and interactive 3D structures.
4. **Pipeline Conclusion:** SpycFit gracefully terminates, displaying a massive green success banner. The computational chemistry pipeline is formally complete.
