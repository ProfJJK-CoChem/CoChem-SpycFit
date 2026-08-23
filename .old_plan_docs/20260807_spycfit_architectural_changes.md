# CoChem-SpycFit Architectural Changes (2026-08-07)

This document outlines the requisite architectural refactoring for the `CoChem-SpycFit` module. These changes transition the module from a standard linear IR/Raman convolution engine into a rigorous Bayesian spectral fitting and Active Learning platform designed specifically for deciphering dense rotational microwave spectra.

## 1. Multi-Pathway Quantum Tensor Assimilation
**Target Files:** `cochem_quantum_extractor.py` and `cochem_spycfit_init.py`

**Current State:** 
The extractor pulls the raw Cartesian Hessian, applies isotopic mass-weighting, and passes it to the convolver to generate static harmonic infrared spectra.

**Required Architectural Change:**
- **LAM Protocol Detection:** The extractor must parse the `cochem_state.h5` tensor metadata. If it detects that `TORQ` executed the **Stage 3B (Sinc-DVR)** pathway, SpycFit must bypass standard VPT2 Hessian diagonalization entirely. Instead, it must extract the pre-diagonalized 3D probability eigenvalues and A/E tunneling splittings directly from the DVR tensor.
- **Kraitchman Substitution Hook:** The extractor must assimilate the theoretical Kraitchman $r_s$ substitution coordinates mapped by TORQ, preparing them as Bayesian priors for the isotopic fitting loop.

## 2. Bayesian Active Learning & LOIO Cross-Validation
**Target File:** `cochem_spycfit_fitter.py` (Microwave Bayesian fitting module, differentiated from the continuous spectral grid convolution engine `cochem_convolver.py`)

**Current State:** 
The convolver uses JAX `vmap` to blindly broadcast Gaussian line profiles across a frequency grid based on harmonic normal modes.

**Required Architectural Change:**
- **Rigorous Typing & Validation:** All statistical configurations, parameters, and JSON payloads must be strictly enforced through `Pydantic` models utilizing exhaustive Python 3.10+ type hints.
- **Partition Function Thermodynamics:** To accurately fit absolute peak intensities against experimental data, SpycFit must dynamically calculate the Rigid-Rotor Harmonic-Oscillator (RRHO) partition function $Q_{\text{RRHO}}(T)$, extracting the exact experimental temperature from the user's .int files or configuration inputs.
- **Levenberg-Marquardt Optimizer:** `CoChem-SpycFit` must implement a non-linear Levenberg-Marquardt Bayesian optimizer. The *ab initio* rotational constants ($A, B, C$) and quartic centrifugal distortion constants (passed from TORQ) will serve as the rigid priors. The optimizer will fit these against experimental CP-FTMW transition frequencies (provided by the user in a `.lin` file).
- **LOIO Routine:** Implement the Leave-One-Isotopologue-Out (LOIO) cross-validation statistical method. The engine must systematically drop one isotopic species from the fit, re-optimize, and calculate the variance leverage to prove that no single isotopologue is falsely skewing the global structure.
- **Information-Gain Transition Mapping:** SpycFit must dynamically calculate the statistical variance matrix. Utilizing active learning, the UI must suggest the "Next Best Microwave Transitions" to the user—specifically highlighting unassigned frequencies that have the highest theoretical intensity combined with the lowest prediction error, drastically accelerating human assignment.

## 3. Legacy Pickett SPFIT Handoff Integration
**Target File:** `cochem_quantum_extractor.py` (New Export Subroutines)

**Current State:** 
The module dumps a basic `.csv` array of wavenumber vs. intensity.

**Required Architectural Change:**
- **SPFIT/SPCAT Generation:** To maintain strict academic integrity and allow classical spectroscopists to cross-validate the ML results, SpycFit must dynamically write Pickett-formatted `.lin` (transition lines), `.par` (catalog parameters), `.cat` (catalog predictions), and `.fit` files. **Crucially, all file paths must use dynamic lookups via `pathlib.Path.home()` and environment variables. Hardcoded paths are strictly forbidden. All outputs must be forced into a configurable artifacts directory.**
- **SCRIBE Triggering:** These legacy files must be automatically routed to the `CoChem-SCRIBE` background daemon for Zstandard isolation and bundling (`Legacy_Verification.zip`). **All `subprocess.run` calls to trigger SCRIBE must be wrapped in `try/except` blocks with `check=True` and explicit timeouts. `psutil` or `atexit` must be used to sweep zombie processes, and `print()` calls must be replaced entirely by `logging`.**

## 4. Hardware-Accelerated Tensor Constraints
**Target File:** `cochem_spycfit_init.py`

**Current State:** 
JAX is locked to `float64`, but memory limits are unbounded.

**Required Architectural Change:**
- **GPU Memory-Mapped Tensors:** Calculating rotational states from $J=0$ up to $J=80$ across multiple vibrational states generates enormous matrices. SpycFit must implement strict memory-mapped file (`mmap`) reads and JAX host-to-device streaming. Attempting to load entire multi-dimensional DVR / PES tensor grids or massive multi-transition high-resolution spectral convolution grids into GPU VRAM at once will crash the XLA compiler; batching logic is mandatory.
- **Penalty Function Reporting:** If the Levenberg-Marquardt optimizer attempts to force a mathematically unphysical negative centrifugal distortion constant (strictly applying ONLY to diagonal constants such as $D_J, D_K, \Delta_J, \Delta_K$, not cross-term constants which can legitimately be negative) to achieve a lower RMS error, SpycFit must instantly apply a hard mathematical penalty, reject the step, and log the intervention securely for `SCRIBE` to include in the Supporting Information.

## 5. SCOPE EXCLUSION: UV-Vis & TD-DFT Spectral Convolution
**Target File:** `cochem_convolver.py`

**Current State:** 
Previous drafts incorrectly proposed ingesting TD-DFT and STEOM-CCSD vertical excitation energies.

**Required Architectural Change (Adversarial Correction):**
- **Strict Method Matrix Enforcement:** Any proposal to expand SpycFit into UV-Vis, TD-DFT, STEOM-CCSD, or time-dependent laser dynamics is **REJECTED** as a direct violation of the CoChem Scope Exclusions. The ecosystem is explicitly confined to static *ab initio* geometry optimization, electronic structure, and VPT2 / anharmonic force fields for predicting rotational constants in the microwave regime.
- **Code Pruning:** Any code attempting to broadcast UV-Vis Voigt profiles must be immediately purged.

---
**Next Step Readiness:** 
By implementing these structural pillars, `CoChem-SpycFit` will safely transform from a static IR spectrum generator into an interactive, Active Learning-driven microwave spectroscopy fitting engine. Once these changes are approved, the exceptionally detailed and complete UI/Execution Workflow for SpycFit can be securely generated.
