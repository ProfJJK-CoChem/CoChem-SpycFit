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
**Target File:** `cochem_convolver.py` (To be refactored into `cochem_bayesian_fitter.py`)

**Current State:** 
The convolver uses JAX `vmap` to blindly broadcast Gaussian line profiles across a frequency grid based on harmonic normal modes.

**Required Architectural Change:**
- **Partition Function Thermodynamics:** To accurately fit absolute peak intensities against experimental data, SpycFit must dynamically calculate the Rigid-Rotor Harmonic-Oscillator (RRHO) partition function (T)$, extracting the exact experimental temperature from the user's .lin file metadata.
- **Levenberg-Marquardt Optimizer:** `CoChem-SpycFit` must implement a non-linear Levenberg-Marquardt Bayesian optimizer. The *ab initio* rotational constants ($A, B, C$) and quartic centrifugal distortion constants (passed from TORQ) will serve as the rigid priors. The optimizer will fit these against experimental CP-FTMW transition frequencies (provided by the user in a `.lin` file).
- **LOIO Routine:** Implement the Leave-One-Isotopologue-Out (LOIO) cross-validation statistical method. The engine must systematically drop one isotopic species from the fit, re-optimize, and calculate the variance leverage to prove that no single isotopologue is falsely skewing the global structure.
- **Information-Gain Transition Mapping:** SpycFit must dynamically calculate the statistical variance matrix. Utilizing active learning, the UI must suggest the "Next Best Microwave Transitions" to the user—specifically highlighting unassigned frequencies that have the highest theoretical intensity combined with the lowest prediction error, drastically accelerating human assignment.

## 3. Legacy Pickett SPFIT Handoff Integration
**Target File:** `cochem_quantum_extractor.py` (New Export Subroutines)

**Current State:** 
The module dumps a basic `.csv` array of wavenumber vs. intensity.

**Required Architectural Change:**
- **SPFIT/SPCAT Generation:** To maintain strict academic integrity and allow classical spectroscopists to cross-validate the ML results, SpycFit must dynamically write Pickett-formatted `.lin` (transition lines), `.cat` (catalog parameters), and `.fit` files.
- **SCRIBE Triggering:** These legacy files must be automatically routed to the `CoChem-SCRIBE` background daemon for Zstandard isolation and bundling (`Legacy_Verification.zip`).

## 4. Hardware-Accelerated Tensor Constraints
**Target File:** `cochem_spycfit_init.py`

**Current State:** 
JAX is locked to `float64`, but memory limits are unbounded.

**Required Architectural Change:**
- **GPU Memory-Mapped Tensors:** Calculating rotational states from $J=0$ up to $J=80$ across multiple vibrational states generates enormous matrices. SpycFit must implement strict memory-mapped file (`mmap`) reads and JAX host-to-device streaming. Attempting to load the entire rotational partition function into GPU VRAM at once will crash the XLA compiler; batching logic is mandatory.
- **Penalty Function Reporting:** If the Levenberg-Marquardt optimizer attempts to force a mathematically unphysical negative centrifugal distortion constant to achieve a lower RMS error, SpycFit must instantly apply a hard mathematical penalty, reject the step, and log the intervention securely for `SCRIBE` to include in the Supporting Information.

## 5. UV-Vis Spectral Convolution (STEOM-CCSD)
**Target File:** `cochem_convolver.py`

**Required Architectural Change:**
- **Excited State Ingestion:** SpycFit must be expanded beyond Microwave/IR convolutions to ingest TD-DFT and STEOM-CCSD vertical excitation energies and Natural Transition Orbitals (NTOs) from the `cochem_state.h5` tensor.
- **JAX Absorption Profiling:** The convolver must utilize JAX arrays to broadcast Voigt or log-normal profiles over the excitation frequencies to rapidly generate theoretical UV-Vis absorption spectra, enabling active learning for photochemically active mixtures.

---
**Next Step Readiness:** 
By implementing these structural pillars, `CoChem-SpycFit` will safely transform from a static IR spectrum generator into an interactive, Active Learning-driven microwave spectroscopy fitting engine. Once these changes are approved, the exceptionally detailed and complete UI/Execution Workflow for SpycFit can be securely generated.
