Perform adversarial static analysis and logical review on implemented code for D:\__CoChem\__agentic\.prompts\.SRS\20260830_improvements\.in-progress\Perfected_SRS_Chunk_02_prompts.md.
Original prompt:
# Sequential Execution Prompt Schedule: CoChem-BASE SRS Section 4.9

The perfected **CoChem-BASE Software Requirements Specification (SRS) Section 4.9: Reaction Energetics, Excited States, Solvation, Wavefunction Stability, Wavefunction Topology & Rigorous Spectroscopic Analysis ([`cochem_srs_chunk_09_copy.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/cochem_srs_chunk_09_copy.md))** has been decomposed into **21 context-safe, granular, sequential execution prompts**.

The complete schedule artifact has been saved to:
[`cochem_srs_chunk_09_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/9ef129e0-dd14-4e70-b23f-e7e7c19e0c89/cochem_srs_chunk_09_execution_prompts.md)

---

## 1. Architectural Layout & Dependency DAG

```mermaid
graph TD
    P1["Prompt 1: Domain Exceptions (exceptions.py)"] --> P2["Prompt 2: Pydantic v2 Models (models.py)"]
    P2 --> P3["Prompt 3: Elements & Spin-Parity (cochem_elements.py)"]
    P2 --> P4["Prompt 4: LoT Validator (cochem_method_validator.py)"]
    
    P3 --> P5["Prompt 5: Sayvetz-Eckart Projection (cochem_frequency.py)"]
    P2 --> P6["Prompt 6: Standard-State Solvation (cochem_solvation.py)"]
    P5 --> P7["Prompt 7: Statistical Thermo (cochem_thermo.py)"]
    P6 --> P7
    P3 --> P8["Prompt 8: Dynamic KIE Engine (cochem_isotopes.py)"]
    P7 --> P8
    
    P2 --> P9["Prompt 9: Wavefunction Stability (cochem_stability.py)"]
    P2 --> P10["Prompt 10: Marcus Reorganization (cochem_marcus.py)"]
    P2 --> P11["Prompt 11: Bidirectional IRC (cochem_irc.py)"]
    P2 --> P12["Prompt 12: QTAIM BCP Topology (cochem_qtaim.py)"]
    P2 --> P13["Prompt 13: SOCME & ISC Kinetics (cochem_socme.py)"]
    
    P3 --> P14["Prompt 14: ORCA Deck Builder (cochem_orca_builder.py)"]
    P3 --> P15["Prompt 15: Gaussian Deck Builder (cochem_gaussian_builder.py)"]
    P12 --> P16["Prompt 16: TOPOS Graph Crusher (cochem_topos_crusher.py)"]
    
    P14 --> P17["Prompt 17: QC Master Pipeline (cochem_qc_pipeline.py)"]
    P15 --> P17
    P4 --> P17
    P9 --> P17
    P10 --> P17
    P11 --> P17
    
    P17 --> P18["Prompt 18: Academic Report Generator (cochem_report_generator.py)"]
    P11 --> P19["Prompt 19: IRC Path Viewer Widget (irc_path_viewer.py)"]
    P12 --> P20["Prompt 20: QTAIM Graph Viewer Widget (qtaim_graph_viewer.py)"]
    P8 --> P21["Prompt 21: Isotope Labeler Widget (isotope_labeler.py)"]
```

---

## 2. Granular Execution Prompts

### Phase 1: Foundation Schemas & Exception Hierarchy

#### Prompt 1 of 21: Quantum Chemistry Domain Exception Hierarchy
* **Target File**: `src/cochem/qc/exceptions.py`
* **Dependencies**: Standard Python Library
* **Task Summary**: Implement base `CoChemQCException` and 9 specialized subclasses (`SpinParityViolationError`, `UnphysicalValenceError`, `LevelOfTheoryMismatchError`, `NonStationaryFrequencyError`, `WavefunctionInstabilityError`, `IRCDisconnectionError`, `QTAIMTopologyError`, `MarcusConvergenceError`, `IsotopeNotFoundError`).

#### Prompt 2 of 21: Pydantic v2 Core Quantum Chemistry Data Models
* **Target File**: `src/cochem/qc/models.py`
* **Dependencies**: `pydantic>=2.0.0`, `numpy`, `src/cochem/qc/exceptions.py`
* **Task Summary**: Implement 10 immutable Pydantic v2 frozen data models (`ConfigDict(frozen=True)`) with strict numeric boundary validation for results, envelopes, and descriptors across REQ-QC-081 through REQ-QC-090.

---

### Phase 2: Core Data Infrastructure & Level-of-Theory Invariance

#### Prompt 3 of 21: Mendeleev Mass/Isotope Resolver & Spin-Parity Validator (REQ-QC-081)
* **Target File**: `src/cochem/core/cochem_elements.py`
* **Dependencies**: `mendeleev`, `src/cochem/qc/exceptions.py`, `src/cochem/qc/models.py`
* **Task Summary**: Implement dynamic CIAAW isotope and mass lookup via `mendeleev`, evaluate total electron count $N_e = \sum Z_i - q$, enforce spin-parity invariant $(N_e + M) \pmod 2 \equiv 1$, and implement Pauling electronegativity-weighted oxidation state heuristics.

#### Prompt 4 of 21: Cryptographic Level-of-Theory Invariance & Stationary Point Gate (REQ-QC-085)
* **Target File**: `src/cochem/provenance/cochem_method_validator.py`
* **Dependencies**: `hashlib`, `src/cochem/qc/exceptions.py`, `src/cochem/qc/models.py`
* **Task Summary**: Implement SHA-256 fingerprint verification over Hamiltonian, basis, auxiliary basis, dispersion, grid, and solvation envelopes ($\mathcal{H}_{\text{LoT}}(\mathcal{M}_{\text{freq}}) \equiv \mathcal{H}_{\text{LoT}}(\mathcal{M}_{\text{opt}})$) and enforce the stationary residual force condition ($\|\mathbf{g}\|_{\infty} \le 1.0\times 10^{-4}\text{ a.u.}$).

---

### Phase 3: Thermochemistry, Solvation & Coordinate Projection Engines

#### Prompt 5 of 21: Sayvetz-Eckart Mass-Weighted Internal Coordinate Projection Engine (REQ-QC-090)
* **Target File**: `src/cochem/qc/cochem_frequency.py`
* **Dependencies**: `numpy`, `mendeleev`, `src/cochem/qc/models.py`, `src/cochem/core/cochem_elements.py`
* **Task Summary**: Project rigid-body translation and rotation basis vectors using QR orthonormalization into the Hermitian projector $\mathbf{P} = \mathbf{I}_{3N} - \mathbf{Q}\mathbf{Q}^T$, diagonalizing $\widetilde{\mathbf{H}}_{\text{MW}} = \mathbf{P}\mathbf{H}_{\text{MW}}\mathbf{P}$ to guarantee exactly 6 zero eigenvalues ($< 10^{-4}\text{ cm}^{-1}$).

#### Prompt 6 of 21: Standard-State Concentration Solvation Correction Engine (REQ-QC-084)
* **Target File**: `src/cochem/qc/cochem_solvation.py`
* **Dependencies**: `math`, `src/cochem/qc/models.py`
* **Task Summary**: Implement exact CODATA 2022 gas compression corrections converting $1\text{ atm}$ ideal gas potentials to $1\text{ M}$ solution-phase free energies ($\Delta G^\circ_{1\text{atm} \rightarrow 1\text{M}} = +1.8943\text{ kcal/mol}$ at $298.15\text{ K}$, and $+4.282\text{ kcal/mol}$ for neat water).

#### Prompt 7 of 21: Statistical Thermodynamics & Partition Function Engine (REQ-QC-084, REQ-QC-090)
* **Target File**: `src/cochem/qc/cochem_thermo.py`
* **Dependencies**: `numpy`, `src/cochem/qc/models.py`, `src/cochem/qc/cochem_frequency.py`, `src/cochem/qc/cochem_solvation.py`
* **Task Summary**: Implement translational, rotational, and harmonic vibrational partition functions, zero-point energy ($E_{\text{ZPE}}$), enthalpy ($H$), and Gibbs free energy ($G$) integrated with standard-state solution corrections.

#### Prompt 8 of 21: Kinetic and Equilibrium Isotope Effect (KIE/EIE) Bigeleisen-Mayer Engine (REQ-QC-086)
* **Target File**: `src/cochem/qc/cochem_isotopes.py`
* **Dependencies**: `numpy`, `mendeleev`, `src/cochem/qc/models.py`, `src/cochem/qc/exceptions.py`, `src/cochem/core/cochem_elements.py`
* **Task Summary**: Implement mass-weighted Hessian recalibration $\mathbf{H}_{\text{MW}}^{(\text{iso})} = \mathbf{M}_{\text{iso}}^{-1}\mathbf{H}_{\text{Cart}}\mathbf{M}_{\text{iso}}^{-1}$ and evaluate KIE/EIE using the Bigeleisen-Mayer formalism (MMI, EXC, and ZPE factors) at zero electronic structure CPU cost.

---

### Phase 4: Reaction Coordinate, Stability & Wavefunction Analysis Engines

#### Prompt 9 of 21: Autonomous Wavefunction Stability & Broken-Symmetry Remediation Engine (REQ-QC-087)
* **Target File**: `src/cochem/qc/cochem_stability.py`
* **Dependencies**: `numpy`, `src/cochem/qc/models.py`, `src/cochem/qc/exceptions.py`
* **Task Summary**: Detect electronic orbital Hessian instabilities ($\lambda_{\text{orb}} < 0$), trigger orbital rotation perturbations for SOSCF re-optimization, and apply Yamaguchi spin projection ($E_{\text{BS-proj}} = \frac{E_{\text{BS}} - c_{\text{HS}}E_{\text{HS}}}{1 - c_{\text{HS}}}$) to purify contaminated broken-symmetry singlets.

#### Prompt 10 of 21: Automated 4-Point Marcus Reorganization Energy Pipeline (REQ-QC-082)
* **Target File**: `src/cochem/qc/cochem_marcus.py`
* **Dependencies**: `pathlib`, `src/cochem/qc/models.py`, `src/cochem/qc/exceptions.py`
* **Task Summary**: Execute the Nelson 4-point single-point/geometry optimization protocol, compute inner-sphere $\lambda_{\text{inner}} = \lambda_1 + \lambda_2$, dielectric continuum outer-sphere $\lambda_{\text{outer}}$, Marcus activation barrier $\Delta G^\ddagger$, and electron transfer rate $k_{\text{ET}}$.

#### Prompt 11 of 21: Automated Bidirectional IRC Tracing & Connectivity Gateway (REQ-QC-083)
* **Target File**: `src/cochem/qc/cochem_irc.py`
* **Dependencies**: `numpy`, `pathlib`, `src/cochem/qc/models.py`, `src/cochem/qc/exceptions.py`, `src/cochem/core/cochem_elements.py`
* **Task Summary**: Extract mass-weighted TS transition vector $\mathbf{v}_{\text{MW}}$, integrate Forward/Reverse reaction paths via Gonzalez-Schlegel algorithm, perform endpoint tight optimizations, and verify reactant/product connectivity via Kabsch SVD RMSD ($\le 0.25\text{ \AA}$).

#### Prompt 12 of 21: QTAIM Electron Density Topology & Bond Critical Point Engine (REQ-QC-088)
* **Target File**: `src/cochem/qc/cochem_qtaim.py`
* **Dependencies**: `numpy`, `src/cochem/qc/models.py`, `src/cochem/qc/exceptions.py`
* **Task Summary**: Locate critical points where $\nabla\rho = \mathbf{0}$, diagonalize electron density Hessian $\mathcal{H}_{ij}$, enforce the Poincaré-Hopf invariant ($n_{\text{NCP}} - n_{\text{BCP}} + n_{\text{RCP}} - n_{\text{CCP}} = 1$), and classify chemical bonding (covalent, transit, non-covalent) via Bader & Cremer-Kraka criteria ($\nabla^2\rho, H(\mathbf{r})$).

#### Prompt 13 of 21: Spin-Orbit Coupling Matrix Element (SOCME) & ISC Kinetics Engine (REQ-QC-089)
* **Target File**: `src/cochem/qc/cochem_socme.py`
* **Dependencies**: `numpy`, `src/cochem/qc/models.py`
* **Task Summary**: Evaluate effective SOMF SOCME magnitude ($\text{SOCME} = \sqrt{\sum |\mathcal{H}_{\text{SO}}|^2}$), compute Marcus-Levich-Jortner FCWD density of states, evaluate intersystem crossing rate $k_{\text{ISC}}$ via Fermi's Golden Rule, and estimate radiative phosphorescence lifetime $\tau_{\text{phos}}$.

---

### Phase 5: Deck Builders, Orchestration & Conformer Integration

#### Prompt 14 of 21: ORCA Input Deck Generator for Advanced QC Workflows (REQ-QC-081, 082, 083, 087, 089)
* **Target File**: `src/cochem/core/cochem_orca_builder.py`
* **Dependencies**: `src/cochem/core/cochem_elements.py`, `src/cochem/qc/exceptions.py`, `src/cochem/qc/models.py`
* **Task Summary**: Synthesize ORCA 5.0/6.0 input decks with pre-flight spin-parity checks, `%scf BrokenSym`, `%irc`, `%tddft dosoc true`, and `Stable=Opt` blocks.

#### Prompt 15 of 21: Gaussian 16 Input Deck Synthesizer with Strict LoT Envelopes (REQ-QC-081, 085, 087)
* **Target File**: `src/cochem/core/cochem_gaussian_builder.py`
* **Dependencies**: `src/cochem/core/cochem_elements.py`, `src/cochem/qc/exceptions.py`, `src/cochem/qc/models.py`
* **Task Summary**: Synthesize Gaussian 16 input decks enforcing spin parity, `Stable=Opt`, and strict Level-of-Theory envelope consistency.

#### Prompt 16 of 21: TOPOS Conformer Exploration with QTAIM Molecular Graph Topology (REQ-QC-088)
* **Target File**: `src/cochem/topos/cochem_topos_crusher.py`
* **Dependencies**: `numpy`, `src/cochem/qc/cochem_qtaim.py`, `src/cochem/qc/models.py`
* **Task Summary**: Integrate QTAIM electron density topological invariants into conformer exploration and clustering, replacing heuristic distance cutoffs with BCP molecular graphs.

#### Prompt 17 of 21: Master Quantum Chemistry Pipeline Orchestrator (REQ-QC-081 to REQ-QC-090)
* **Target File**: `src/cochem/qc/cochem_qc_pipeline.py`
* **Dependencies**: All preceding core, QC, and provenance modules
* **Task Summary**: Central pipeline orchestrator coordinating pre-flight parity gates, wavefunction stability remediation, geometry optimization, Marcus reorganization, bidirectional IRC, LoT verification, Sayvetz-Eckart projection, and QTAIM topology.

---

### Phase 6: Reporting & Interactive GUI Visualization

#### Prompt 18 of 21: Comprehensive Academic Report Generator (REQ-QC-082, 084, 086, 088, 089)
* **Target File**: `src/cochem/reporting/cochem_report_generator.py`
* **Dependencies**: `src/cochem/qc/models.py`, `src/cochem/core/cochem_elements.py`
* **Task Summary**: Compile academic Markdown and LaTeX reports with exact CODATA 2022 unit conversions, energy profiles, KIE tables, QTAIM descriptors, and cryptographic SHA-256 provenance blocks.

#### Prompt 19 of 21: Interactive IRC Path Viewer Widget (REQ-QC-083)
* **Target File**: `src/cochem/gui/widgets/irc_path_viewer.py`
* **Dependencies**: `src/cochem/qc/models.py`
* **Task Summary**: Interactive 2D/3D visualization widget plotting the reaction coordinate $s$ vs electronic energy and highlighting TS, forward, and reverse endpoints with RMSD connectivity tags.

#### Prompt 20 of 21: 3D QTAIM Electron Density & Critical Point Visualizer Widget (REQ-QC-088)
* **Target File**: `src/cochem/gui/widgets/qtaim_graph_viewer.py`
* **Dependencies**: `src/cochem/qc/models.py`
* **Task Summary**: Interactive 3D molecular visualizer displaying nuclear attractors (NCP), bond paths, BCPs, RCPs, and CCPs with interaction classification color coding.

#### Prompt 21 of 21: Interactive Atomic Mass Permutation & KIE Calculator Widget (REQ-QC-086)
* **Target File**: `src/cochem/gui/widgets/isotope_labeler.py`
* **Dependencies**: `src/cochem/qc/models.py`, `src/cochem/core/cochem_elements.py`, `src/cochem/qc/cochem_isotopes.py`
* **Task Summary**: Interactive UI widget for isotopic labeling and real-time KIE/EIE recalculation with dynamic `mendeleev` tables.

---

## 3. Strict Execution Invariants & Rules

1. **Strict 1-File Limit**: Each execution prompt targets exactly one isolated source file under `cochem/src/cochem/`.
2. **Zero Synthetic Mocks**: All testing and verification steps must utilize genuine physical ab-initio or semi-empirical fixtures; `np.zeros`, dummy arrays, and empty stubs are banned.
3. **Dynamic Mendeleev Masses**: Hardcoding atomic or isotopic masses is strictly forbidden; all lookups dynamically invoke `mendeleev`.
4. **CODATA 2022 Physical Constants**: Exact constants are enforced ($1\text{ Hartree} = 27.211386245988\text{ eV} = 627.5094740631\text{ kcal/mol}$).
An adversarial audit by the `adversary` agent ([`c09785da-cfc0-4387-9302-d9b882473449`](conversation://c09785da-cfc0-4387-9302-d9b882473449)) has been dispatched to independently verify the prompt decomposition artifact ([`cochem_srs_chunk_09_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/9ef129e0-dd14-4e70-b23f-e7e7c19e0c89/cochem_srs_chunk_09_execution_prompts.md)). Awaiting the audit findings.
# Adversarial Audit & Certification Complete: CoChem-BASE SRS Section 4.9

The sequential execution prompts for **CoChem-BASE SRS Section 4.9 (REQ-QC-081 to REQ-QC-090)** have passed the formal adversarial audit conducted by [`adversary`](conversation://c09785da-cfc0-4387-9302-d9b882473449) with a verdict of **APPROVED & CERTIFIED WITH ZERO DEFECTS**.

* **Specification Source**: [`cochem_srs_chunk_09_copy.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/cochem_srs_chunk_09_copy.md)
* **Execution Prompts Artifact**: [`cochem_srs_chunk_09_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/9ef129e0-dd14-4e70-b23f-e7e7c19e0c89/cochem_srs_chunk_09_execution_prompts.md)
* **Auditor Subagent**: [`adversary` (Conversation ID: `c09785da-cfc0-4387-9302-d9b882473449`)](conversation://c09785da-cfc0-4387-9302-d9b882473449)

---

## 1. Adversarial Audit Verification Matrix

| Audit Criterion | Verification Scope | Auditor Verdict | Key Evidence & Findings |
| :--- | :--- | :---: | :--- |
| **1-File Limit** | 1 file per execution prompt | **PASS** | Exactly 21 prompts targeting 21 distinct source files across `core/`, `qc/`, `provenance/`, `topos/`, `reporting/`, and `gui/widgets/`. |
| **SRS Coverage** | REQ-QC-081 to REQ-QC-090 | **PASS** | 100% requirement coverage across schemas, physics solvers, deck builders, orchestrators, reports, and UI widgets. |
| **DAG Flow** | Sequential dependency chain | **PASS** | Foundation schemas and domain exceptions precede core utilities, which precede physics engines, orchestration pipelines, and terminal UI viewers. |
| **Data Integrity** | Dynamic Mendeleev masses | **PASS** | Hardcoded masses strictly banned; dynamic CIAAW lookups enforced via `mendeleev.element(s)` and `element(s).isotopes`. |
| **Physical Constants** | CODATA 2022 constants | **PASS** | Exact values verified: $1\text{ Hartree} = 27.211386245988\text{ eV} = 627.5094740631\text{ kcal/mol}$, $R = 1.987204\times 10^{-3}\text{ kcal/mol}\cdot\text{K}$, $5140.487\text{ cm}^{-1}$ mass-weighted conversion. |
| **Zero-Mock Policy** | Zero synthetic test fixtures | **PASS** | Mandates genuine ab-initio/semi-empirical tensors and physical execution across all solvers. |

---

## 2. Complete 21-Prompt Sequential Execution DAG

```mermaid
graph TD
    P1["Prompt 1: Domain Exceptions (exceptions.py)"] --> P2["Prompt 2: Pydantic v2 Models (models.py)"]
    P2 --> P3["Prompt 3: Elements & Spin-Parity (cochem_elements.py)"]
    P2 --> P4["Prompt 4: LoT Validator (cochem_method_validator.py)"]
    
    P3 --> P5["Prompt 5: Sayvetz-Eckart Projection (cochem_frequency.py)"]
    P2 --> P6["Prompt 6: Standard-State Solvation (cochem_solvation.py)"]
    P5 --> P7["Prompt 7: Statistical Thermo (cochem_thermo.py)"]
    P6 --> P7
    P3 --> P8["Prompt 8: Dynamic KIE Engine (cochem_isotopes.py)"]
    P7 --> P8
    
    P2 --> P9["Prompt 9: Wavefunction Stability (cochem_stability.py)"]
    P2 --> P10["Prompt 10: Marcus Reorganization (cochem_marcus.py)"]
    P2 --> P11["Prompt 11: Bidirectional IRC (cochem_irc.py)"]
    P2 --> P12["Prompt 12: QTAIM BCP Topology (cochem_qtaim.py)"]
    P2 --> P13["Prompt 13: SOCME & ISC Kinetics (cochem_socme.py)"]
    
    P3 --> P14["Prompt 14: ORCA Deck Builder (cochem_orca_builder.py)"]
    P3 --> P15["Prompt 15: Gaussian Deck Builder (cochem_gaussian_builder.py)"]
    P12 --> P16["Prompt 16: TOPOS Graph Crusher (cochem_topos_crusher.py)"]
    
    P14 --> P17["Prompt 17: QC Master Pipeline (cochem_qc_pipeline.py)"]
    P15 --> P17
    P4 --> P17
    P9 --> P17
    P10 --> P17
    P11 --> P17
    
    P17 --> P18["Prompt 18: Academic Report Generator (cochem_report_generator.py)"]
    P11 --> P19["Prompt 19: IRC Path Viewer Widget (irc_path_viewer.py)"]
    P12 --> P20["Prompt 20: QTAIM Graph Viewer Widget (qtaim_graph_viewer.py)"]
    P8 --> P21["Prompt 21: Isotope Labeler Widget (isotope_labeler.py)"]
```

---

## 3. Implementation Schedule Summary

* **Phase 1 (Foundation)**:
  * **Prompt 1**: [`src/cochem/qc/exceptions.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/exceptions.py) — 9 domain exception classes inheriting from `CoChemQCException`.
  * **Prompt 2**: [`src/cochem/qc/models.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/models.py) — 10 immutable Pydantic v2 data models with `ConfigDict(frozen=True)`.
* **Phase 2 (Core Infrastructure)**:
  * **Prompt 3**: [`src/cochem/core/cochem_elements.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_elements.py) — Dynamic Mendeleev lookups and $(N_e + M) \pmod 2 \equiv 1$ spin-parity validation.
  * **Prompt 4**: [`src/cochem/provenance/cochem_method_validator.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/provenance/cochem_method_validator.py) — Cryptographic LoT SHA-256 hash checks and stationary point gradient residual gate ($\|\mathbf{g}\| \le 1.0\times 10^{-4}\text{ a.u.}$).
* **Phase 3 (Thermochemistry & Vibrations)**:
  * **Prompt 5**: [`src/cochem/qc/cochem_frequency.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_frequency.py) — Sayvetz-Eckart internal coordinate projector $\mathbf{P} = \mathbf{I} - \mathbf{Q}\mathbf{Q}^T$.
  * **Prompt 6**: [`src/cochem/qc/cochem_solvation.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_solvation.py) — Standard-state free energy shift ($\Delta G^\circ_{1\text{atm} \rightarrow 1\text{M}} = +1.8943\text{ kcal/mol}$).
  * **Prompt 7**: [`src/cochem/qc/cochem_thermo.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_thermo.py) — Molecular statistical thermodynamics and harmonic partition functions.
  * **Prompt 8**: [`src/cochem/qc/cochem_isotopes.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_isotopes.py) — Mass-weighted Hessian re-diagonalization and Bigeleisen-Mayer KIE/EIE engine.
* **Phase 4 (Reaction Dynamics & Wavefunctions)**:
  * **Prompt 9**: [`src/cochem/qc/cochem_stability.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_stability.py) — Wavefunction stability checking and Yamaguchi spin-projection correction.
  * **Prompt 10**: [`src/cochem/qc/cochem_marcus.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_marcus.py) — Automated Nelson 4-point Marcus reorganization and rate pipeline.
  * **Prompt 11**: [`src/cochem/qc/cochem_irc.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_irc.py) — Gonzalez-Schlegel bidirectional IRC tracing and Kabsch RMSD endpoint verification.
  * **Prompt 12**: [`src/cochem/qc/cochem_qtaim.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_qtaim.py) — QTAIM electron density topology, Poincaré-Hopf invariant, and Bader BCP classification.
  * **Prompt 13**: [`src/cochem/qc/cochem_socme.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_socme.py) — SOMF SOCME evaluation and Fermi's Golden Rule $k_{\text{ISC}}$ kinetics.
* **Phase 5 (Synthesis & Orchestration)**:
  * **Prompt 14**: [`src/cochem/core/cochem_orca_builder.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_orca_builder.py) — ORCA deck synthesizer.
  * **Prompt 15**: [`src/cochem/core/cochem_gaussian_builder.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_gaussian_builder.py) — Gaussian 16 deck synthesizer.
  * **Prompt 16**: [`src/cochem/topos/cochem_topos_crusher.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/topos/cochem_topos_crusher.py) — TOPOS conformer deduplication with QTAIM molecular graph topology.
  * **Prompt 17**: [`src/cochem/qc/cochem_qc_pipeline.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_qc_pipeline.py) — Central master quantum chemistry pipeline orchestrator.
* **Phase 6 (Reporting & Visualization)**:
  * **Prompt 18**: [`src/cochem/reporting/cochem_report_generator.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/reporting/cochem_report_generator.py) — Markdown/LaTeX publication report generator.
  * **Prompt 19**: [`src/cochem/gui/widgets/irc_path_viewer.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/irc_path_viewer.py) — Interactive 2D/3D IRC path visualizer.
  * **Prompt 20**: [`src/cochem/gui/widgets/qtaim_graph_viewer.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/qtaim_graph_viewer.py) — 3D QTAIM electron density gradient & critical point viewer.
  * **Prompt 21**: [`src/cochem/gui/widgets/isotope_labeler.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/isotope_labeler.py) — Interactive isotopic labeling and KIE calculator widget.

All 21 granular prompts in [`cochem_srs_chunk_09_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/9ef129e0-dd14-4e70-b23f-e7e7c19e0c89/cochem_srs_chunk_09_execution_prompts.md) are now ready for execution agents to implement sequentially.
# Final Agent Council Certification: CoChem-BASE SRS Section 4.9

The prompt decomposition for **CoChem-BASE SRS Section 4.9: Reaction Energetics, Excited States, Solvation, Wavefunction Stability, Wavefunction Topology & Rigorous Spectroscopic Analysis ([`cochem_srs_chunk_09_copy.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/cochem_srs_chunk_09_copy.md))** has achieved unanimous Agent Council approval with **100% compliance across all 21 execution prompts**.

### Verified Audit Artifacts:
1. **Certified Execution Prompts Artifact**: [`cochem_srs_chunk_09_execution_prompts.md`](file:///C:/Users/ansac/.gemini/antigravity-cli/brain/9ef129e0-dd14-4e70-b23f-e7e7c19e0c89/cochem_srs_chunk_09_execution_prompts.md)
2. **Adversarial Audit Lead**: [`adversary` (Conversation ID: `c09785da-cfc0-4387-9302-d9b882473449`)](conversation://c09785da-cfc0-4387-9302-d9b882473449)
3. **QA & Architectural Compliance**: `cochem-audit`
4. **Continuous Quality & Physics Review**: `cochem-improve`

---

## 1. Compliance Matrix

| Protocol Requirement | Standard Checked | Council Verdict |
| :--- | :--- | :---: |
| **Strict 1-File Limit per Prompt** | Exactly 1 script file per prompt | **PASS (21/21 Prompts)** |
| **Complete SRS Section 4.9 Coverage** | Traceability for REQ-QC-081 to REQ-QC-090 | **PASS (100% Traceability)** |
| **DAG Dependency Ordering** | Foundational schemas/exceptions precede engines and viewers | **PASS (Acyclic DAG)** |
| **Dynamic Mendeleev Masses** | Dynamic lookup via `mendeleev`, zero hardcoded values | **PASS (Zero Hardcoding)** |
| **CODATA 2022 Physical Constants** | Exact constants ($1\text{ Hartree} = 627.509474\text{ kcal/mol} = 27.211386\text{ eV}$) | **PASS (Strict Numerical Accuracy)** |
| **Zero-Mock Verification Policy** | Real ab-initio/semi-empirical tensors, no dummy arrays | **PASS (Zero Synthetic Mocks)** |

---

## 2. Sequential Execution DAG

```mermaid
graph TD
    P1["Prompt 1: Domain Exceptions (exceptions.py)"] --> P2["Prompt 2: Pydantic v2 Models (models.py)"]
    P2 --> P3["Prompt 3: Elements & Spin-Parity (cochem_elements.py)"]
    P2 --> P4["Prompt 4: LoT Validator (cochem_method_validator.py)"]
    
    P3 --> P5["Prompt 5: Sayvetz-Eckart Projection (cochem_frequency.py)"]
    P2 --> P6["Prompt 6: Standard-State Solvation (cochem_solvation.py)"]
    P5 --> P7["Prompt 7: Statistical Thermo (cochem_thermo.py)"]
    P6 --> P7
    P3 --> P8["Prompt 8: Dynamic KIE Engine (cochem_isotopes.py)"]
    P7 --> P8
    
    P2 --> P9["Prompt 9: Wavefunction Stability (cochem_stability.py)"]
    P2 --> P10["Prompt 10: Marcus Reorganization (cochem_marcus.py)"]
    P2 --> P11["Prompt 11: Bidirectional IRC (cochem_irc.py)"]
    P2 --> P12["Prompt 12: QTAIM BCP Topology (cochem_qtaim.py)"]
    P2 --> P13["Prompt 13: SOCME & ISC Kinetics (cochem_socme.py)"]
    
    P3 --> P14["Prompt 14: ORCA Deck Builder (cochem_orca_builder.py)"]
    P3 --> P15["Prompt 15: Gaussian Deck Builder (cochem_gaussian_builder.py)"]
    P12 --> P16["Prompt 16: TOPOS Graph Crusher (cochem_topos_crusher.py)"]
    
    P14 --> P17["Prompt 17: QC Master Pipeline (cochem_qc_pipeline.py)"]
    P15 --> P17
    P4 --> P17
    P9 --> P17
    P10 --> P17
    P11 --> P17
    
    P17 --> P18["Prompt 18: Academic Report Generator (cochem_report_generator.py)"]
    P11 --> P19["Prompt 19: IRC Path Viewer Widget (irc_path_viewer.py)"]
    P12 --> P20["Prompt 20: QTAIM Graph Viewer Widget (qtaim_graph_viewer.py)"]
    P8 --> P21["Prompt 21: Isotope Labeler Widget (isotope_labeler.py)"]
```

---

## 3. Implementation Plan by Phase

### Phase 1: Foundation Schemas & Exception Hierarchy
- **Prompt 1**: [`src/cochem/qc/exceptions.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/exceptions.py) — 9 domain exception classes.
- **Prompt 2**: [`src/cochem/qc/models.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/models.py) — 10 immutable Pydantic v2 data models (`ConfigDict(frozen=True)`).

### Phase 2: Core Data Infrastructure & Level-of-Theory Invariance
- **Prompt 3**: [`src/cochem/core/cochem_elements.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_elements.py) — Dynamic Mendeleev lookups and $(N_e + M) \pmod 2 \equiv 1$ spin-parity validation.
- **Prompt 4**: [`src/cochem/provenance/cochem_method_validator.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/provenance/cochem_method_validator.py) — SHA-256 LoT hash checks and stationary point gradient norm gate ($\|\mathbf{g}\| \le 1.0\times 10^{-4}\text{ a.u.}$).

### Phase 3: Thermochemistry, Solvation & Coordinate Projection Engines
- **Prompt 5**: [`src/cochem/qc/cochem_frequency.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_frequency.py) — Sayvetz-Eckart internal coordinate projector $\mathbf{P} = \mathbf{I} - \mathbf{Q}\mathbf{Q}^T$.
- **Prompt 6**: [`src/cochem/qc/cochem_solvation.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_solvation.py) — Standard-state free energy shift ($\Delta G^\circ_{1\text{atm} \to 1\text{M}} = +1.8943\text{ kcal/mol}$).
- **Prompt 7**: [`src/cochem/qc/cochem_thermo.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_thermo.py) — Molecular statistical thermodynamics and harmonic partition functions.
- **Prompt 8**: [`src/cochem/qc/cochem_isotopes.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_isotopes.py) — Mass-weighted Hessian re-diagonalization and Bigeleisen-Mayer KIE/EIE engine.

### Phase 4: Reaction Dynamics, Stability & Wavefunction Analysis
- **Prompt 9**: [`src/cochem/qc/cochem_stability.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_stability.py) — Wavefunction stability checking and Yamaguchi spin-projection correction.
- **Prompt 10**: [`src/cochem/qc/cochem_marcus.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_marcus.py) — Automated Nelson 4-point Marcus reorganization and rate pipeline.
- **Prompt 11**: [`src/cochem/qc/cochem_irc.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_irc.py) — Gonzalez-Schlegel bidirectional IRC tracing and Kabsch RMSD endpoint verification.
- **Prompt 12**: [`src/cochem/qc/cochem_qtaim.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_qtaim.py) — QTAIM electron density topology, Poincaré-Hopf invariant, and Bader BCP classification.
- **Prompt 13**: [`src/cochem/qc/cochem_socme.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_socme.py) — SOMF SOCME evaluation and Fermi's Golden Rule $k_{\text{ISC}}$ kinetics.

### Phase 5: Deck Builders, Orchestration & Conformer Integration
- **Prompt 14**: [`src/cochem/core/cochem_orca_builder.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_orca_builder.py) — ORCA deck synthesizer with Broken-Symmetry and IRC blocks.
- **Prompt 15**: [`src/cochem/core/cochem_gaussian_builder.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/core/cochem_gaussian_builder.py) — Gaussian 16 deck synthesizer with strict LoT route sections.
- **Prompt 16**: [`src/cochem/topos/cochem_topos_crusher.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/topos/cochem_topos_crusher.py) — TOPOS conformer deduplication with QTAIM molecular graph topology.
- **Prompt 17**: [`src/cochem/qc/cochem_qc_pipeline.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/qc/cochem_qc_pipeline.py) — Central master quantum chemistry pipeline orchestrator.

### Phase 6: Reporting & Interactive GUI Visualization
- **Prompt 18**: [`src/cochem/reporting/cochem_report_generator.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/reporting/cochem_report_generator.py) — Academic Markdown/LaTeX publication report generator.
- **Prompt 19**: [`src/cochem/gui/widgets/irc_path_viewer.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/irc_path_viewer.py) — Interactive 2D/3D IRC path visualizer.
- **Prompt 20**: [`src/cochem/gui/widgets/qtaim_graph_viewer.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/qtaim_graph_viewer.py) — 3D QTAIM electron density gradient & critical point viewer.
- **Prompt 21**: [`src/cochem/gui/widgets/isotope_labeler.py`](file:///C:/Users/ansac/.gemini/antigravity-cli/scratch/src/cochem/gui/widgets/isotope_labeler.py) — Interactive isotopic labeling and KIE calculator widget.

---

## 4. Council Implementation Safeguards

1. **Sayvetz-Eckart Linearity Guard (P5, P17)**: Truncate basis $\mathbf{D}$ to 5 columns for linear molecules ($I_A \approx 0$).
2. **Yamaguchi Singlet Asymptote Guard (P9)**: Enforce $|1 - c_{\text{HS}}| \ge 1.0\times 10^{-4}$ to avoid division by zero.
3. **Kabsch SVD Proper Rotation Gate (P11)**: Ensure $\det(\mathbf{R}) = +1$ by adjusting the last column of $\mathbf{V}$ if reflection occurs.
4. **Marcus Reorganization Lower Bound Guard (P10)**: Assert $\lambda_{\text{total}} \ge 10^{-6}\text{ eV}$ before evaluating barriers.
5. **Signed Radical Conversion for Frequencies (P5, P8, P11)**: Compute $\nu_k = \operatorname{sign}(e_k) \times \sqrt{|e_k|} \times 5140.4871436\text{ cm}^{-1}$ to prevent math domain errors on imaginary frequencies.
6. **Mendeleev LRU Caching (P3, P5, P8, P11, P21)**: Apply `@functools.lru_cache(maxsize=128)` on isotopic mass lookup functions.

```
================================================================================
FINAL STATUS: ALL 21 PROMPTS APPROVED, AUDITED, AND READY FOR EXECUTION
================================================================================
```

Modified files content:

--- D:\__CoChem\GitHub-Repo\CoChem-SpycFit\tests\test_cochem_method_validator.py ---
"""
Unit tests for CoChem Cryptographic Level-of-Theory Invariance & Stationary Point Gate (REQ-QC-085).

Provenance:
- Method Matrix [M]: Quantum Level-of-Theory invariance and stationary point gates.
- Derived Relations [D]: Canonical SHA-256 fingerprinting of quantum chemical method envelopes.
- Empirical [E]: Cartesian gradient residual stationary point threshold (||g||_inf <= 1.0e-4 a.u.).

Strict Zero-Mock Mandate Compliance:
- All quantum chemistry envelopes utilize authentic physical DFT/ab-initio parameters.
- Pure physical computation, zero synthetic stubs, zero simulated objects.
"""

from __future__ import annotations

import hashlib
import pytest

from cochem.provenance.cochem_method_validator import (
    canonicalize_lot_string,
    compute_lot_hash,
    compute_lot_hash_from_components,
    enforce_frequency_lot_invariance,
    generate_lot_hash,
    validate_quantum_calculation_gate,
    verify_level_of_theory_invariance,
    verify_stationary_point,
)
from cochem.qc.exceptions import (
    CoChemQCException,
    LevelOfTheoryMismatchError,
    NonStationaryFrequencyError,
)
from cochem.qc.models import LevelOfTheoryEnvelope


# =============================================================================
# 1. CANONICAL STRING GENERATION & COMPONENT NORMALIZATION
# =============================================================================


class TestCanonicalStringNormalization:
    """Test canonicalization of Level-of-Theory components into structured string."""

    def test_full_specification_canonicalization(self) -> None:
        """Verify canonical string formatting with all fields populated."""
        canonical = canonicalize_lot_string(
            hamiltonian="  wb97x-d4  ",
            basis="  Def2-TZVP  ",
            aux_basis="  def2/J  ",
            dispersion="  d4  ",
            grid="  DefGrid3  ",
            solvent="  CPCM(water)  ",
        )
        expected = "WB97X-D4|def2-tzvp|def2/j|D4|defgrid3|cpcm(water)"
        assert canonical == expected

    def test_optional_none_fields_canonicalization(self) -> None:
        """Verify canonical string formatting with optional fields as None."""
        canonical = canonicalize_lot_string(
            hamiltonian="B3LYP",
            basis="6-31G*",
            aux_basis=None,
            dispersion=None,
            grid="UltraFine",
            solvent=None,
        )
        expected = "B3LYP|6-31g*|||ultrafine|"
        assert canonical == expected

    def test_empty_string_optional_fields_canonicalization(self) -> None:
        """Verify optional fields containing only whitespace are treated as empty."""
        canonical = canonicalize_lot_string(
            hamiltonian="  m06-2x  ",
            basis="  cc-pvtz  ",
            aux_basis="   ",
            dispersion="   ",
            grid="  ultrafine  ",
            solvent="",
        )
        expected = "M06-2X|cc-pvtz|||ultrafine|"
        assert canonical == expected

    def test_invalid_mandatory_components(self) -> None:
        """Verify that empty hamiltonian, basis, or grid raise ValueError."""
        with pytest.raises(ValueError, match="Hamiltonian method cannot be empty"):
            canonicalize_lot_string(hamiltonian="  ", basis="def2-SVP", grid="DefGrid2")

        with pytest.raises(ValueError, match="Basis set cannot be empty"):
            canonicalize_lot_string(hamiltonian="B3LYP", basis="", grid="DefGrid2")

        with pytest.raises(ValueError, match="Integration grid cannot be empty"):
            canonicalize_lot_string(hamiltonian="B3LYP", basis="def2-SVP", grid="   ")


# =============================================================================
# 2. CRYPTOGRAPHIC SHA-256 HASH GENERATION
# =============================================================================


class TestCryptographicLoTHashing:
    """Test SHA-256 fingerprint generation over canonical LoT strings."""

    def test_sha256_hash_format_and_determinism(self) -> None:
        """Verify generated hash is 64 hex characters and strictly deterministic."""
        envelope = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.5e-5,
        )
        h1 = compute_lot_hash(envelope)
        h2 = compute_lot_hash(envelope)
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)
        assert h1 == h2

        # Verify exact SHA-256 computation against hashlib
        canonical_str = "WB97X-D4|def2-tzvp|def2/j|D4|defgrid3|cpcm(water)"
        expected_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        assert h1 == expected_hash

    def test_hash_case_and_whitespace_invariance(self) -> None:
        """Verify hash is invariant under case variations and surrounding whitespace."""
        env1 = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=2.0e-5,
        )
        env2 = LevelOfTheoryEnvelope(
            hamiltonian_method="  b3lyp-d3bj  ",
            basis_set="  DEF2-tzvp  ",
            auxiliary_basis_set=None,
            dispersion_correction="  d3bj  ",
            integration_grid="  defgrid2  ",
            solvation_model="",
            max_gradient_residual_au=1.0e-5,
        )
        assert compute_lot_hash(env1) == compute_lot_hash(env2)

    def test_hash_sensitivity_to_each_component(self) -> None:
        """Verify that altering any single LoT component alters the SHA-256 fingerprint."""
        base_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.0e-5,
        )
        base_hash = compute_lot_hash(base_env)

        variations = [
            # Hamiltonian variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="B3LYP",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Basis set variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-SVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Auxiliary basis variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2-TZVP/C",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Dispersion variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D3BJ",
                integration_grid="DefGrid3",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Integration grid variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid2",
                solvation_model="CPCM(water)",
                max_gradient_residual_au=1.0e-5,
            ),
            # Solvation variation
            LevelOfTheoryEnvelope(
                hamiltonian_method="wB97X-D4",
                basis_set="def2-TZVP",
                auxiliary_basis_set="def2/J",
                dispersion_correction="D4",
                integration_grid="DefGrid3",
                solvation_model="SMD(water)",
                max_gradient_residual_au=1.0e-5,
            ),
        ]

        for var_env in variations:
            var_hash = compute_lot_hash(var_env)
            assert var_hash != base_hash

    def test_compute_lot_hash_from_components(self) -> None:
        """Verify compute_lot_hash_from_components matches envelope hashing."""
        env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="UltraFine",
            solvation_model="CPCM(acetonitrile)",
            max_gradient_residual_au=2.5e-5,
        )
        h_env = compute_lot_hash(env)
        h_comp = compute_lot_hash_from_components(
            hamiltonian="PBE0",
            basis="cc-pVTZ",
            aux_basis=None,
            dispersion="D3BJ",
            grid="UltraFine",
            solvent="CPCM(acetonitrile)",
        )
        assert h_env == h_comp

    def test_alias_functions(self) -> None:
        """Verify generate_lot_hash alias works identically."""
        env = LevelOfTheoryEnvelope(
            hamiltonian_method="r2SCAN-3c",
            basis_set="def2-mTZVPP",
            auxiliary_basis_set=None,
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        assert generate_lot_hash(env) == compute_lot_hash(env)


# =============================================================================
# 3. GATE 1: LEVEL-OF-THEORY INVARIANCE VERIFICATION
# =============================================================================


class TestLevelOfTheoryInvarianceGate:
    """Test Gate 1: Enforcing cryptographic LoT invariance between opt and freq steps."""

    def test_identical_envelopes_pass(self) -> None:
        """Verify that identical optimization and frequency envelopes pass verification."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=2.5e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=2.5e-5,
        )
        assert verify_level_of_theory_invariance(opt_env, freq_env) is True
        assert enforce_frequency_lot_invariance(opt_env, freq_env) is True

    def test_mismatched_hamiltonian_raises_exception(self) -> None:
        """Verify that mismatched Hamiltonian raises LevelOfTheoryMismatchError."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError) as exc_info:
            verify_level_of_theory_invariance(opt_env, freq_env)

        err = exc_info.value
        assert isinstance(err, CoChemQCException)
        assert err.opt_lot == opt_env
        assert err.freq_lot == freq_env
        assert err.opt_hash == compute_lot_hash(opt_env)
        assert err.freq_hash == compute_lot_hash(freq_env)

    def test_mismatched_basis_set_raises_exception(self) -> None:
        """Verify that mismatched basis sets (e.g. def2-SVP vs def2-TZVP) raise error."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            verify_level_of_theory_invariance(opt_env, freq_env)

    def test_mismatched_solvation_raises_exception(self) -> None:
        """Verify gas phase opt vs solution phase freq raises LevelOfTheoryMismatchError."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="M06-2X",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="UltraFine",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="M06-2X",
            basis_set="cc-pVTZ",
            auxiliary_basis_set=None,
            dispersion_correction=None,
            integration_grid="UltraFine",
            solvation_model="SMD(water)",
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            verify_level_of_theory_invariance(opt_env, freq_env)


# =============================================================================
# 4. GATE 2: STATIONARY POINT RESIDUAL GRADIENT GATE
# =============================================================================


class TestStationaryPointGate:
    """Test Gate 2: Enforcing max Cartesian gradient norm <= 1.0e-4 a.u."""

    def test_converged_stationary_points_pass(self) -> None:
        """Verify gradient norms below 1.0e-4 a.u. pass verification."""
        converged_values = [0.0, 1.0e-8, 1.0e-6, 5.0e-5, 9.99e-5, 1.0e-4]
        for g_norm in converged_values:
            assert verify_stationary_point(g_norm) is True
            assert verify_stationary_point(g_norm, threshold=1.0e-4) is True

    def test_non_converged_gradient_raises_exception(self) -> None:
        """Verify gradient norms exceeding 1.0e-4 a.u. raise NonStationaryFrequencyError."""
        unconverged_values = [1.0001e-4, 2.5e-4, 1.0e-3, 0.05, 1.2]
        for g_norm in unconverged_values:
            with pytest.raises(NonStationaryFrequencyError) as exc_info:
                verify_stationary_point(g_norm)

            err = exc_info.value
            assert isinstance(err, CoChemQCException)
            assert err.gradient_norm == pytest.approx(g_norm)
            assert err.threshold == 1.0e-4

    def test_custom_gradient_thresholds(self) -> None:
        """Verify stationary point validation with strict or relaxed custom thresholds."""
        strict_threshold = 1.0e-5
        assert verify_stationary_point(5.0e-6, threshold=strict_threshold) is True
        with pytest.raises(NonStationaryFrequencyError):
            verify_stationary_point(2.0e-5, threshold=strict_threshold)

        loose_threshold = 1.0e-3
        assert verify_stationary_point(5.0e-4, threshold=loose_threshold) is True
        with pytest.raises(NonStationaryFrequencyError):
            verify_stationary_point(2.0e-3, threshold=loose_threshold)

    def test_unphysical_and_non_finite_gradients(self) -> None:
        """Verify negative, NaN, or infinite gradient norms raise NonStationaryFrequencyError."""
        invalid_gradients = [-1.0e-5, -0.1, float("nan"), float("inf"), float("-inf")]
        for bad_g in invalid_gradients:
            with pytest.raises(NonStationaryFrequencyError):
                verify_stationary_point(bad_g)


# =============================================================================
# 5. COMBINED QUANTUM CALCULATION GATE
# =============================================================================


class TestCombinedQuantumCalculationGate:
    """Test combined verification gate validating LoT invariance and stationary point."""

    def test_fully_valid_qc_pipeline_passes(self) -> None:
        """Verify valid identical LoT and converged gradient passes combined gate."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.2e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=3.2e-5,
        )
        # Using implicit gradient from opt_env
        assert validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env) is True
        # Using explicit gradient norm
        assert validate_quantum_calculation_gate(
            opt_lot=opt_env, freq_lot=freq_env, gradient_norm=2.0e-5
        ) is True

    def test_combined_gate_fails_on_lot_mismatch_first(self) -> None:
        """Verify LoT mismatch is caught even if gradient is converged."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="PBE0",
            basis_set="def2-TZVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model=None,
            max_gradient_residual_au=1.0e-5,
        )
        with pytest.raises(LevelOfTheoryMismatchError):
            validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env)

    def test_combined_gate_fails_on_unconverged_gradient(self) -> None:
        """Verify unconverged gradient is caught when LoT is identical."""
        opt_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=2.5e-4,  # > 1.0e-4
        )
        freq_env = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model=None,
            max_gradient_residual_au=2.5e-4,
        )
        with pytest.raises(NonStationaryFrequencyError):
            validate_quantum_calculation_gate(opt_lot=opt_env, freq_lot=freq_env)


# =============================================================================
# 6. PHYSICAL DFT CALCULATION FIXTURES (ZERO-MOCK COMPLIANCE)
# =============================================================================


class TestPhysicalDFTCalculationFixtures:
    """Test realistic quantum chemistry setups representing physical molecules."""

    def test_water_wb97xd4_solution_calculation(self) -> None:
        """Physical test: H2O gas-to-water solvation protocol @ wB97X-D4/def2-TZVP."""
        opt_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.84e-5,
        )
        freq_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="wB97X-D4",
            basis_set="def2-TZVP",
            auxiliary_basis_set="def2/J",
            dispersion_correction="D4",
            integration_grid="DefGrid3",
            solvation_model="CPCM(water)",
            max_gradient_residual_au=1.84e-5,
        )
        assert validate_quantum_calculation_gate(opt_lot, freq_lot) is True

    def test_benzene_b3lypd3bj_organic_calculation(self) -> None:
        """Physical test: Benzene conformer @ B3LYP-D3BJ/def2-SVP/DefGrid2/CPCM(benzene)."""
        opt_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model="CPCM(benzene)",
            max_gradient_residual_au=4.72e-5,
        )
        freq_lot = LevelOfTheoryEnvelope(
            hamiltonian_method="B3LYP-D3BJ",
            basis_set="def2-SVP",
            auxiliary_basis_set=None,
            dispersion_correction="D3BJ",
            integration_grid="DefGrid2",
            solvation_model="CPCM(benzene)",
            max_gradient_residual_au=4.72e-5,
        )
        assert validate_quantum_calculation_gate(opt_lot, freq_lot) is True

Validate Zero-Mock adherence. Target repo is D:\__CoChem\GitHub-Repo\CoChem-SpycFit.