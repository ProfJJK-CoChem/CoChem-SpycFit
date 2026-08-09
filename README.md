# CoChem-SpycFit

**CoChem-SpycFit** is the Bayesian Interactive Spectral Assignment Engine of the CoChem suite.

Moving far beyond static `.csv` frequency generation, SpycFit:
- Translates classical VPT2 priors (rotational constants, distortion scalars) or quantum Sinc-DVR wavefunctions from `CoChem-TORQ` into a JAX-accelerated Bayesian optimizer (Levenberg-Marquardt).
- Dynamically parses the experimental temperature to calculate the Rigid-Rotor Harmonic-Oscillator (RRHO) partition function $Q(T)$, ensuring theoretical intensities precisely match experiment.
- Deploys **Information-Gain Transition Mapping** to present the user with the mathematically "Next Best Transitions," rapidly accelerating human-in-the-loop assignment of dense Microwave spectra.
- Executes automated Leave-One-Isotopologue-Out (LOIO) cross-validation statistical routines to rigorously validate the global minimum geometry.

## Usage
Please refer to the authoritative [CoChem Master User Manual](../CoChem-BASE/CoChem_Master_User_Manual.md) for full execution instructions across the entire 5-module pipeline.