# CoChem-SpycFit Software Requirements Specification, version 2

| Field | Value |
| :--- | :--- |
| Status | **Draft 1, proposed.** No requirement below is approved. Agents write only `proposed`. Review and approval may proceed chapter by chapter. |
| Author | PI: Dr. Joshua John Klaassen. Agents draft text; only the PI approves it. |
| Approval | The PI's tag `srs-v2.0` on the commit the PI approves (gate G1). |
| Revises | The PI's requirements baseline on `main` at `488dc87`: `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md` (consolidated SRS) and `docs/wiki/CoChem-SpycFit_SRS_2026-10-16_Amendment.md` (amendment). Until `srs-v2.0` exists, those two documents remain the baseline. |
| Historical | `docs/archive/CoChem-SpycFit_Comprehensive_SRS_7e42507.md`; `.docs/task1_subsystems_architectural_specification.md` (a CoChem-BASE document); `docs/legacy/2026-08-07_*` plans. |
| Parent documents | `CoChem-BASE/Method_Matrix.md` v4. `CoChem-TORQ/docs/wiki/Method_Matrix_Implementation_Contract.md`. `CoChem-TORQ/docs/wiki/CoChem-TORQ_Implementation_SRS.md` §1.2, §9.1, §12, §15-16. `CoChem-TORQ/docs/wiki/review/Decisions_and_Risks.md` D05, D08, D10, D11, D12. |
| Working inputs | `docs/HANDOFF.md` (the PI's 2026-10-16 decisions); `docs/handoff/PI_DECISIONS.md`; `docs/wiki/review/CoChem-SpycFit_SRS_2026-10-16_Literature_Review.md` (the literature review behind this draft, findings R1-R5). |

---

## 0. Document control

### 0.1 Versions

| Version | Date | Change |
| :--- | :--- | :--- |
| Draft 0 | 2026-10-11 | Skeleton (PR #1). |
| Draft 1 | 2026-10-16 | Full proposed text. Restructures the consolidated SRS and amendment at `488dc87` into single-statement requirements, keeps every SF id, retires nine duplicate ids as aliases (§0.6), and adds requirements from the literature review. |

### 0.2 Modal verbs

SHALL, SHALL NOT, SHOULD and MAY carry the meanings of RFC 2119 as clarified by RFC 8174 [R-1]
and only in capitals. Each requirement states one verifiable condition. A requirement may pair an
obligation with the prohibition or consequence that defines its failure (for example, a value
SHALL be recorded and SHALL NOT be inferred) when one acceptance test checks both. Text without
these words is informative.

This document is prepared with reference to ISO/IEC/IEEE 29148:2018 [R-2]. No conformance is
claimed. The individual-requirement characteristics used in review are those listed by the
standard: necessary, appropriate, unambiguous, complete, singular, feasible, verifiable, correct
and conforming [R-3].

### 0.3 Requirement form

Each requirement is one bullet:

> - **SF-AREA-NNNx** Title. One SHALL statement with one verifiable predicate.
>   *Source* · *Priority* · *Verification* and acceptance id · *Status*

- **Id.** Ids are never reused and are independent of section numbers. A letter suffix marks a
  part split from a compound requirement at `488dc87`; the unsuffixed id names the whole.
- **Source.** "Amd §n" is the amendment and "SRS §n" the consolidated SRS at `488dc87`.
  "PI 2026-10-16" marks substance the PI settled on that date (`docs/HANDOFF.md`, amendment §2
  and §7); the wording here is new and is proposed. "PI 2026-10-17" marks an answer the PI gave
  in conversation on that date, recorded in `docs/decisions/`. "R1-nn" to "R5-nn" are
  literature-review findings, whose citations are in §1.6.
- **Priority.** Core (the semirigid comparison profile and anything it depends on), Profile
  (one of the later capability profiles in §1.2) or Optional.
- **Verification.** T = Test, A = Analysis, I = Inspection, D = Demonstration, followed by an
  acceptance id from §8.2.
- **Status.** Agents write only `proposed`. The `srs-v2.0` tag approves every requirement in the
  tagged commit. `implemented` and `verified` come only from the CI traceability report (§10).
- **Tolerances.** Every tolerance reads "TBD (D-SPYC-14)" until the PI sets it. A requirement
  that depends on an open decision says "blocked on D-SPYC-nn".
- **Owner** is given per section as the component in `docs/architecture/CoChem-SpycFit_Architecture.md`.

### 0.4 Status and absent-state vocabulary

One vocabulary serves comparisons, engine runs and missing values (R5 §4 item 6):

| Kind | Values | Source |
| :--- | :--- | :--- |
| Comparison status | NOT_RUN, UNAVAILABLE, FAILED, INCOMPATIBLE_MODEL, INCOMPLETE, DIVERGENT, WITHIN_TOLERANCE | Amd SF-CMP-012 |
| Engine run outcome | COMPLETED, FAILED, TIMED_OUT, CANCELLED, UNSUPPORTED | SRS SF-RUN-004 |
| Absent value | NOT_REQUESTED, NOT_RUN, UNAVAILABLE, FAILED, NOT_APPLICABLE, each with a reason | TORQ-DATA-002 |

A comparison whose engine run is not COMPLETED is FAILED (the run failed or timed out),
UNAVAILABLE (the program is missing) or NOT_RUN (not requested or cancelled).

### 0.5 Evidence classes

Every test and every acceptance record carries one evidence class (SRS §11; Oberkampf and
Trucano's separation of code verification, solution verification and validation [R-4]):

| Class | Meaning | Oracle |
| :--- | :--- | :--- |
| software | Parsing, transport, process handling, interfaces | Specified behaviour; malformed inputs test robustness only |
| analytical | Numerical correctness against closed forms or limits | Analytic expressions |
| statistical | Estimator recovery and uncertainty calibration | Designed studies with known truth |
| external-engine | Agreement with a genuine pinned external program | Outputs of that program on hash-verified inputs |
| experimental | Agreement with measured data | Published or PI-supplied measurements with provenance |

### 0.6 Retired ids

These ids duplicated a confirmed amendment requirement. They are aliases and are never reused (R5 §4).

| Retired | Canonical | Retired | Canonical |
| :--- | :--- | :--- | :--- |
| SF-MODEL-001 | SF-SCOPE-003 | SF-UI-001 | SF-HITL-001 |
| SF-IO-001 | SF-ING-001 | SF-UI-002 | SF-HITL-002 |
| SF-IO-002 | SF-ING-002 | SF-OUT-003 | SF-PUB-001 |
| SF-IO-006 | SF-DATA-001 | SF-FIT-002 | SF-INF-001 |
| SF-FIT-006 | SF-INF-002 | | |

Sentences that only partly overlapped were moved: SF-UI-004 second sentence and SF-UI-005 first
sentence to SF-HITL-003; the divergence clause of SF-OUT-004 to SF-CMP-015; the label clause of
SF-RUN-002 to SF-CMP-013; the field list of SF-SCOPE-005 to SF-ML-002; the shared-source clause
of SF-DATA-001 to SF-IO-007; "one coherent record" of SF-PUB-001 to SF-OUT-001.

---

## 1. Introduction

### 1.1 Purpose

SpycFit is an expert-directed module of the CoChem ecosystem for assigning and fitting molecular
spectra. It helps researchers inspect measured spectra, interpret predictions, compare assignment
hypotheses, fit physical models, state justified uncertainty and produce reproducible publication
artifacts (SRS §1; PI 2026-10-16).

Improvement over existing workflows is measured by correctly supported assignments, expert
effort, predictive accuracy, calibrated uncertainty, run time and traceability, each against a
declared workflow, data set, hardware and tolerance. No blanket claim of superiority over
SPCAT/SPFIT is made (SRS §1).

### 1.2 Scope and capability profiles

**Is.** SpycFit supports, as required research workflows (PI 2026-10-16):

1. broadband rotational assignment and fitting, including isotopologues and uncertainty;
2. joint rotational and resolved rovibrational assignment and fitting with explicit shared
   parameters;
3. computation-led assignment for flexible molecules using qualified computational and MLFF
   providers;
4. vibrational band and line fitting, mode interpretation and spectral-envelope decomposition.

IR, Raman, THz, near-IR and microwave are first-class workflows. Gas-phase data anchor
validation; solution and solid-state work are separate tracks. Genuine SPCAT and SPFIT run
beside the native SpycFit path for compatible microwave models. Semirigid comparison is
accepted first, then hyperfine, excited-state and large-amplitude extensions, each as its own
profile (PI 2026-10-16; Amd §7 D-03).

**Is not.** SpycFit does not reduce raw instrument data (FIDs, interferograms, THz time traces):
adapters or other CoChem modules do (SF-ING-001). It does not run quantum chemistry: TORQ and
other providers do (§1.3); TORQ produces vibrational and Raman predictions (D-SPYC-06). It does
not determine molecular structures (r₀, r_s, r_m): GEOM does, from constants SpycFit exports
(D-SPYC-10; SF-IF-008). It does not train PES or MLFF models: a provider does, through the
interface in §4.3 (SF-SCOPE-005). It does not accept assignments autonomously (SF-HITL-001).
It does not write another module's database.

"First-class" is defined by the capability matrix below: a modality is first-class when each of
its rows has an accepted end-to-end demonstration (SF-SCOPE-002). The rows split THz and NIR
into gas-phase and condensed-phase work, because the observables and models differ (R3-03).

| Profile | Observable and model | Reference engines and data | Order |
| :--- | :--- | :--- | :--- |
| P1 Semirigid rotational | Line frequencies; Watson A/S, linear and symmetric-top Hamiltonians | Pinned SPFIT/SPCAT; CDMS/JPL fit files for the §8.3 panel | First (D-03) |
| P2 Hyperfine rotational | Resolved quadrupole and spin-rotation structure | SPFIT/SPCAT; PGOPHER [R-10] | After P1 |
| P3 Excited and multi-state | Vibrational satellites; Coriolis and Fermi interactions | SPFIT/SPCAT; PGOPHER | After P1 |
| P4 Large-amplitude motion | Internal rotation, inversion, tunnelling | XIAM, BELGI, RAM36 (observable level only) [R-11..R-13] | After P1 |
| P5 Sub-mm and THz, gas | Rotational lines above 300 GHz | As P1-P4 | With P1-P4 |
| P6 Resolved rovibrational, gas | Assigned IR lines; band origins and upper-state constants; joint MW+IR | SPFIT (IR flag), PGOPHER; HITRAN line lists [R-14] | After P1 |
| P7 Band contour, gas | Partially resolved band envelopes from a rovibrational model | PGOPHER contour fits | After P6 |
| P8 Condensed-phase envelopes | Band positions, widths, areas, baselines | Synthetic recovery studies; measured reference spectra | Separate track |
| P9 Vibrational Raman | Raman shifts, depolarisation, intensities | Shift and intensity standards [R-15, R-16] | After P1 |
| P10 Rotational Raman | Rotational Raman lines with spin weights | Dataset to be chosen | After P9 |
| P11 THz-TDS, condensed | Absorption coefficient and refractive index | Published extraction methods [R-17] | Separate track |
| P12 Near-IR overtones | Overtone and combination bands | HITRAN overtone lines; anharmonic providers [R-18] | After P6 |
| P13 Computation-led assignment | Ranked hypotheses scored on held-out data | Blind-challenge data such as HyDRA [R-19] | After P1 |
| P14 PES and force-field refinement | Inverse fits of provider models to observables | A provider module (SF-SCOPE-005) | Interface only |

The order column is a proposal for the plan. Every profile stays in scope (PI 2026-10-16).
TORQ produces the vibrational and Raman predictions these profiles use (D-SPYC-06, PI
2026-10-17; SF-IF-006). UV-Vis is an open question (D-SPYC-06).

### 1.3 Product perspective

| Module | Owns | SpycFit's relation |
| :--- | :--- | :--- |
| BASE | Module discovery (`cochem.modules`), handoff manifests (`cochem.module-handoff/1`), artifact and scratch roots, ingestion, the unified GUI, durable task state (TORQ D08) | Hard requirement (D-SPYC-19); registered module; receives and returns handoffs (§4.4) |
| TOPOS | Conformer and binding-topology search | Dependency of the normal workflow, upstream of TORQ; no direct interface (D-SPYC-19) |
| TORQ | Quantum chemistry, predicted constants (TORQ-EXP-004, HDF5 bundle, not yet built), Pickett prediction inputs (TORQ-EXP-002), predicted catalogs (TORQ-SPEC-001, D12) | Consumer of TORQ bundles (§4.3); sender of refinement requests (SF-RUN-005) |
| GEOM | Molecular geometry, including structure determination (r₀, r_s, r_m) | Receives fitted constants per isotopologue (SF-IF-008; D-SPYC-10) |
| User-supplied files | Predicted ORCA output and Pickett files | Direct-input path that does not need TOPOS or TORQ (SF-IF-005; D-SPYC-19) |
| SpycFit | Observation semantics, assignment interaction, fitting, engine comparison, reporting | This document |

Where TORQ's predicted catalog ends and SpycFit's re-prediction from fitted or edited constants
begins is open decision D-SPYC-16.

### 1.4 Users

| Class | Needs |
| :--- | :--- |
| Expert spectroscopist (the PI and research staff) | Decides every assignment; reviews divergence before export |
| Student in BASE Codespaces | Guided workflows on defaults recorded in provenance |
| Operator | Installs, provisions pinned engines, runs batch jobs |
| Autonomous agent | Headless API only; cannot accept assignments or approve exports |

### 1.5 Glossary

| Term | Meaning |
| :--- | :--- |
| B_e, B_0, B_v | Equilibrium, ground-state effective and state-v effective rotational constants. ΔB_vib = B_0 − B_e (TORQ contract §3.1). |
| Watson A and S reductions | Reduced forms of the asymmetric-top Hamiltonian; the reduction removes indeterminable combinations [R-20]. |
| Representation I^r, III^r, III^l | Assignment of a, b, c axes to z, x, y (King, Hainer and Cross notation [R-21]). |
| Pickett files | `.par` (parameters, input to SPFIT; rewritten by it), `.var` (parameters with fitted errors and correlations), `.lin` (assigned lines), `.int` (intensity settings), `.cat` (catalog), `.fit` (fit log), `.egy` (energies) [R-5, R-6]. |
| RMS ERROR, MICROWAVE RMS | SPFIT's dimensionless weighted rms (denominator: fitted blends) and its unweighted microwave rms in MHz (§1.7.4). |
| FRAC, ERPAR, ERRTST | SPFIT controls: error scaling, a priori parameter error, rejection threshold (§1.7.4). |
| Profile | A row of the capability matrix in §1.2. |
| Evidence class | §0.5. |
| Lineage key | The originating measurement publications of an observation and every fit or catalog derived from them (SF-IO-007c). |
| T_rot | Rotational temperature of a supersonic expansion. |

### 1.6 References

Each reference was resolved (DOI or URL) during the 2026-10-16 literature review unless marked
UNVERIFIED. "Not read" means only the citation was checked. The review document records which
claim each reference supports.

- R-1 RFC 2119 and RFC 8174, https://www.rfc-editor.org/rfc/rfc8174. Modal keywords in capitals.
- R-2 ISO/IEC/IEEE 29148:2018, doi:10.1109/IEEESTD.2018.8559686. Not read; characteristics via R-3.
- R-3 Lubos et al. 2024, arXiv:2408.10886, Table I. The nine individual-requirement characteristics.
- R-4 Oberkampf and Trucano 2002, Prog. Aerosp. Sci. 38, 209, doi:10.1016/S0376-0421(02)00005-2. Code verification, solution verification, validation.
- R-5 Pickett 1991, J. Mol. Spectrosc. 148, 371, doi:10.1016/0022-2852(91)90393-O. SPFIT/SPCAT.
- R-6 JPL calpgm sources and documentation (`calfit.c`, `calcat.c`, `catutil.c`, `spinv.pdf`, `README`), https://spec.jpl.nasa.gov/ftp/pub/calpgm/; hashes in `docs/handoff/calpgm_sources_2026-10-11.sha256` (sources and examples).
- R-7 Kisiel, crib-sheet for SPFIT/SPCAT (updated 2024-08-29), http://info.ifpan.edu.pl/~kisiel/asym/pickett/crib.htm. Parameter signs, RMS denominators, FRAC, DIAG.
- R-8 Pickett, Cohen, Drouin and Pearson, JPL catalog documentation (2003), https://spec.jpl.nasa.gov/ftp/pub/catalog/doc/catdoc.pdf; Pickett et al. 1998, JQSRT 60, 883, doi:10.1016/S0022-4073(98)00091-0. `.cat` format, LGINT at 300 K, Q conventions.
- R-9 Endres et al. 2016, J. Mol. Spectrosc. 327, 95, doi:10.1016/j.jms.2016.03.005; Müller et al. 2005, J. Mol. Struct. 742, 215, doi:10.1016/j.molstruc.2005.01.027. CDMS.
- R-10 Western 2017, JQSRT 186, 221, doi:10.1016/j.jqsrt.2016.04.010; PGOPHER command line, https://pgopher.chm.bris.ac.uk/Help/cmdline.htm. Open source; fit, contour fit.
- R-11 Hartwig and Dreizler 1996, Z. Naturforsch. 51a, 923, doi:10.1515/zna-1996-0807. XIAM.
- R-12 Hougen, Kleiner and Godefroid 1994, J. Mol. Spectrosc. 163, 559, doi:10.1006/jmsp.1994.1047. BELGI.
- R-13 Ilyushin et al. 2010, J. Mol. Spectrosc. 259, 26, doi:10.1016/j.jms.2009.10.005. RAM36.
- R-14 Gordon et al. 2022, JQSRT 277, 107949, doi:10.1016/j.jqsrt.2021.107949 (HITRAN2020); HITRAN definitions, https://hitran.org/docs/definitions-and-units/.
- R-15 Choquette et al. 2007, Appl. Spectrosc. 61, 117, doi:10.1366/000370207779947585. Raman intensity standards (NIST SRM 2241-2243).
- R-16 ASTM E1840, doi:10.1520/E1840-96R22. Raman shift standards.
- R-17 Withayachumnankul and Naftaly 2014, J. Infrared Millim. THz Waves 35, 610, doi:10.1007/s10762-013-0042-z; Jepsen and Fischer 2005, Opt. Lett. 30, 29, doi:10.1364/OL.30.000029. THz-TDS extraction and dynamic range.
- R-18 Bloino 2015, J. Phys. Chem. A 119, 5269, doi:10.1021/jp509985u. Overtones and combinations by VPT2.
- R-19 Fischer et al. 2022, PCCP 24, 11442, doi:10.1039/D2CP01119K. HyDRA blind challenge.
- R-20 Watson 1967, J. Chem. Phys. 46, 1935, doi:10.1063/1.1840957. Indeterminacy and reduction.
- R-21 King, Hainer and Cross 1943, J. Chem. Phys. 11, 27, doi:10.1063/1.1723778. Representations.
- R-22 JCGM 100:2008 (GUM), https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf; JCGM 101:2008. Uncertainty terms, reporting, Monte Carlo.
- R-23 Le Roy 1998, J. Mol. Spectrosc. 191, 223, doi:10.1006/jmsp.1998.7646; Watson 1977, J. Mol. Spectrosc. 66, 500, doi:10.1016/0022-2852(77)90308-3. Rounding of correlated parameters (content via the NLLSSRR source and PGOPHER help; full texts not read).
- R-24 Saltelli et al. 2010, Comput. Phys. Commun. 181, 259, doi:10.1016/j.cpc.2009.09.018; Jansen 1999, Comput. Phys. Commun. 117, 35, doi:10.1016/S0010-4655(98)00154-4; Ishigami and Homma 1990, doi:10.1109/ISUMA.1990.151285. Sobol estimators and test function.
- R-25 Transtrum, Machta and Sethna 2011, Phys. Rev. E 83, 036701, doi:10.1103/PhysRevE.83.036701; Raue et al. 2009, Bioinformatics 25, 1923, doi:10.1093/bioinformatics/btp358. Sloppiness and identifiability.
- R-26 Pernot 2022, J. Chem. Phys. 157, 144103, doi:10.1063/5.0109572. Calibration statistics for prediction uncertainty.
- R-27 Tennyson et al. 2014, Pure Appl. Chem. 86, 1931, doi:10.1515/pac-2014-0208. Hartmann-Tran profile; Kochanov et al. 2016, JQSRT 177, 15, doi:10.1016/j.jqsrt.2016.03.005 (HAPI).
- R-28 Irikura, Johnson and Kacker 2005, J. Phys. Chem. A 109, 8430, doi:10.1021/jp052793n; Kesharwani, Brauer and Martin 2015, J. Phys. Chem. A 119, 1701, doi:10.1021/jp508422u. Scale-factor uncertainty and kinds.
- R-29 Barone 2005, J. Chem. Phys. 122, 014108, doi:10.1063/1.1824881; Franke et al. 2021, J. Phys. Chem. A 125, 1301, doi:10.1021/acs.jpca.0c09526. VPT2 and resonance treatment.
- R-30 Meier 2005, Vib. Spectrosc. 39, 266, doi:10.1016/j.vibspec.2005.03.003; Maddams 1980, Appl. Spectrosc. 34, 245, doi:10.1366/0003702804730312. Band-fitting pitfalls.
- R-31 Wang et al. 2021, Chin. Phys. C 45, 030003, doi:10.1088/1674-1137/abddaf (AME2020); Mohr et al. 2025, Rev. Mod. Phys. 97, 025002, doi:10.1103/RevModPhys.97.025002 (CODATA 2022); Meija et al. 2016, Pure Appl. Chem. 88, 293, doi:10.1515/pac-2015-0503 (isotopic compositions).
- R-32 Kapoor and Narayanan 2023, Patterns 4, 100804, doi:10.1016/j.patter.2023.100804; Wu et al. 2018, Chem. Sci. 9, 513, doi:10.1039/C7SC02664A. Leakage and split design.
- R-33 Kanewala and Bieman 2014, Inf. Softw. Technol. 56, 1219, doi:10.1016/j.infsof.2014.05.006. Test oracles for scientific software.
- R-34 Barker et al. 2022, Sci. Data 9, 622, doi:10.1038/s41597-022-01710-x (FAIR4RS); W3C PROV-O, https://www.w3.org/TR/prov-o/.
- R-35 Guo et al. 2017, arXiv:1706.04599; Angelopoulos and Bates 2023, doi:10.1561/2200000101. Probability calibration; conformal prediction.
- R-36 Müller et al. 2022, MNRAS 518, 185, doi:10.1093/mnras/stac2525; Cabezas, Peña and Cernicharo 2023, MNRAS 519, 2590, doi:10.1093/mnras/stac3698. Line uncertainties by technique; reduction and representation dependence.

### 1.7 Conventions

#### 1.7.1 Units and signs

- Frequencies in MHz, wavenumbers in cm⁻¹, energies in cm⁻¹ or MHz as labelled, temperature in
  K, dipole moments in D, masses in u, lengths in Å.
- E′ is the upper and E″ the lower level; ν = (E′ − E″)/h (SRS SF-PHYS-002).
- Residual r = ν_obs − ν_calc. Weight w = 1/u² for a standard uncertainty u.
- Abscissae are vacuum values unless an observation declares air (SF-INT-001b).

#### 1.7.2 Pickett parameter codes and signs

SPFIT and SPCAT store several Watson constants with a sign or factor (R1-04). The 2026-10-16
review checked the rows below by comparing SPCAT energies (J ≤ 6) with an independent Watson
Hamiltonian; acceptance test V-PICK-03 repeats the check against the pinned binary.

| Code | Stored quantity, A reduction | Stored quantity, S reduction |
| :--- | :--- | :--- |
| 10000, 20000, 30000 | A, B, C | A, B, C |
| 200 | −Δ_J | −D_J |
| 1100 | −Δ_JK | −D_JK |
| 2000 | −Δ_K | −D_K |
| 40100 | −δ_J | +d_1 |
| 41000 | −δ_K | (not used) |
| 50000 | (not used) | +d_2 |

The reduction follows from the codes used; the option-line character and the `.nam` label files
do not set it, and JPL's `spins.nam` labels 40100 and 50000 "-d1" and "-d2", contrary to the
program's behaviour (R1-04). The prolate basis behaves as I^r; III^r is not provided natively.

#### 1.7.3 Intensities

The JPL and CDMS catalogs give log₁₀ of the line intensity in nm²·MHz at 300 K; dividing by
2.99792458 × 10¹⁸ converts to cm⁻¹/(molecule·cm⁻²). They apply no isotopic-abundance weighting,
and their partition function covers rotation and spin. HITRAN uses 296 K, weights by natural
abundance and uses total internal partition sums [R-8, R-14]. SPCAT computes LGINT at the TEMP
and with the QROT given in `.int` (R1-07).

#### 1.7.4 SPFIT statistics

From the pinned source (`calfit.c`) [R-6, R-7]:

- MICROWAVE RMS = √(Σ r²/N_MW) in MHz, unweighted, microwave lines only; IR RMS likewise in cm⁻¹.
- RMS ERROR = √((Σ (r/u)² + prior terms)/N), where N counts each blend once. Its denominator
  is not N − p, and it includes the a priori parameter terms.
- Parameter errors are multiplied by FRAC; with FRAC < 0 by |FRAC| times the standard error of
  the fit (`PARAMETER ERRORS SCALED BY ... times the standard error`).
- ERPAR in `.par` is an a priori error: a finite value is a prior centred on that run's input
  value; a tiny value fixes the parameter.
- Lines with |r|/u above ERRTST are dropped (`NEXT LINE NOT USED IN FIT`).
- The rewritten `.par` keeps the a priori errors. The fitted errors and correlations are in `.var`.

#### 1.7.5 Constants, masses and defaults

- Isotope masses: AME2020 [R-31], with standard uncertainty and the estimated-value flag.
  Isotopic abundances: CIAAW. Fundamental constants: CODATA 2022 through TORQ's profile
  `cochem.constants.codata-2022-torq-compatible/1`. Blocked on D-SPYC-02.
- Temperature, frequency window and J_max are session parameters. The user sets them before the
  first prediction and may change them at any time (D-SPYC-20, PI 2026-10-17; SF-SET-001 to
  SF-SET-004). Whether the fields start empty or with per-technique presets is open; a preset is
  a default only if D-SPYC-20 names it, and is then recorded in provenance as a default.

---

## 2. Scientific invariants

These requirements bound every claim SpycFit makes. Owner: all components.

- **SF-FIT-007** Weak evidence is not acceptance. No result SHALL receive the status `accepted`
  or WITHIN_TOLERANCE from convergence, low rms, small conditional uncertainties, agreement
  between programs or a plausible reconstructed spectrum alone.
  *Source* SRS SF-FIT-007; R5 · Core · T, V-INV-01 · proposed
- **SF-INF-002a** Uncertainty categories. Every reported uncertainty SHALL carry one category
  from {measurement, conditional statistical, fixed-parameter contribution, model-definition
  difference, assignment alternative, computational prediction, numerical}, its evaluation type
  (GUM Type A or B), its coverage factor k and its conditioning set.
  *Source* Amd SF-INF-002 (PI 2026-10-16); R2-11; GUM [R-22] · Core · T, V-UNC-01 · proposed
- **SF-INF-002b** Unknown stays unknown. An uncertainty or covariance that was not determined
  SHALL be stored and exported as absent with a reason, never as zero.
  *Source* Amd SF-INF-002, SF-CMP-009 · Core · T, V-UNC-02 · proposed
- **SF-INF-002c** Assignment ambiguity is not an uncertainty term. Competing assignments SHALL be
  carried as separate fit hypotheses, each with its own result, and SHALL NOT be added to a
  parameter uncertainty.
  *Source* R2-11 · Core · T, V-UNC-03 · proposed
- **SF-PHYS-005** Capability claims need evidence. A large-amplitude or other extended method
  SHALL be selectable only when its capability record (SF-MODEL-002a) cites a reference case
  with recorded acceptance results.
  *Source* SRS SF-PHYS-005 · Profile · I, V-INV-02 · proposed
- **SF-VER-001** Evidence classes stay separate. Every test and acceptance record SHALL carry one
  evidence class from §0.5.
  *Source* SRS §11 · Core · I, V-VER-01 · proposed
- **SF-VER-002** No substitution. A failed or unavailable external engine, or a failed
  experimental check, SHALL NOT be replaced by an analytical check, a process stand-in or another
  physical calculation, and the original failure SHALL remain in the record.
  *Source* SRS §11; Amd SF-CMP-013 · Core · T, V-VER-02 · proposed
- **SF-VER-003** Non-empty coverage. An acceptance record SHALL report its coverage, and a
  record with empty coverage SHALL NOT count as passed.
  *Source* SRS §11; Amd SF-CMP-012 · Core · T, V-VER-03 · proposed
- **SF-VER-004** Criteria first. Numerical, experimental and predictive criteria for an
  acceptance run SHALL be committed before the run.
  *Source* SRS §11; Amd SF-CMP-008 · Core · I, V-VER-04 · proposed
- **SF-VER-005** Improvement claims count failures. A claim that SpycFit improves on a workflow
  SHALL count failed, abstained and expert-corrected assignments in its statistics.
  *Source* SRS §11 · Optional · A, V-VER-05 · proposed

---

## 3. Use cases

| Id | Use case | Main requirements |
| :--- | :--- | :--- |
| UC-1 | Assign and fit a broadband rotational spectrum, with isotopologues and uncertainty | SF-ING, SF-HITL, SF-FIT, SF-CMP, SF-OUT |
| UC-2 | Fit rotational and resolved rovibrational data jointly | SF-FIT-003, SF-FIT-009 |
| UC-3 | Assign a flexible molecule from computational candidates (conformers, MLFF) | SF-ML, SF-PHYS-001, SF-HITL |
| UC-4 | Fit vibrational bands and decompose envelopes | SF-SCOPE-004, SF-VIB |
| UC-5 | Compare native results with genuine SPCAT and SPFIT, review divergence, export | SF-CMP |
| UC-6 | Send a refinement request to a theory provider and receive a new version | SF-RUN-005 |
| UC-7 | Run a fit headless in a batch | SF-API-001, SF-RUN-004 |
| UC-8 | Start from user-supplied ORCA or Pickett prediction files, without TOPOS or TORQ | SF-IF-005 |
| UC-9 | Send fitted isotopologue constants to GEOM for structure determination | SF-IF-008 |

The PI selects and orders the use cases for the first release.

---

## 4. External interfaces

### 4.1 Pickett files and the pinned SPFIT/SPCAT

Owner: `io/pickett`, `fit`, `catalog`.

- **SF-IO-004a** Supported Pickett file types. SpycFit SHALL read and write `.par`, `.var`,
  `.lin`, `.int` and `.cat`, and read `.fit` and `.egy`, as listed in the field-coverage table of
  SF-IO-004d.
  *Source* SRS SF-IO-004; R5 · Core · T, V-PICK-01 · proposed
- **SF-IO-004b** Round trip through the pinned programs. For each writer, output SHALL reproduce
  the original outputs of the pinned program when that program runs on the written file, using
  hash-verified reference inputs.
  *Source* SRS SF-IO-004; R1-14 · Core · T, V-PICK-02 · proposed
- **SF-IO-004c** Unsupported content is reported. A field or record a parser or writer does not
  support SHALL raise a typed error that names the field and line.
  *Source* SRS SF-IO-004 · Core · T, V-PICK-01 · proposed
- **SF-IO-004d** Field coverage table. Each Pickett adapter SHALL publish a coverage table that
  includes, at minimum: the `.lin` quantum-number layout (12I3 default; up to 10 quanta per state
  when NLINE < 0), the negative-ERR wavenumber flag, blends and blend-width pseudo-lines; `.par`
  option-line chaining through VSYM and the `.var` correlation block; `.int` line-2 fields and
  dipole codes; `.cat` letter encoding of quantum numbers and of GUP.
  *Source* R1-14; [R-6, R-8] · Core · I, V-PICK-01 · proposed
- **SF-IO-008** Scratch copies only. SpycFit SHALL run SPFIT and SPCAT only on copies of their
  inputs in a fresh scratch directory, because SPFIT rewrites `.par`.
  *Source* errata row 9; [R-6] · Core · T, V-PICK-04 · proposed
- **SF-PICK-001** Code and sign table. Parameters SHALL map to Pickett codes only through the
  versioned table of §1.7.2, each row verified by V-PICK-03 against the pinned SPCAT.
  *Source* R1-04 · Core · T, V-PICK-03 · proposed
- **SF-PICK-002** Reduction is explicit. The reduction and representation of every parameter set
  SHALL be recorded as data, and SHALL NOT be inferred from the option-line character or from
  `.nam` labels.
  *Source* R1-04 · Core · T, V-PICK-03 · proposed
- **SF-RUN-006a** Verified program identity. SpycFit SHALL run an external program only if its
  SHA-256 matches a recorded manifest entry, and SHALL treat a manifest entry equal to the SHA-256
  of the empty input as absent.
  *Source* SRS SF-RUN-006; Amd §5 · Core · T, V-RUN-06 · proposed
- **SF-RUN-007** Provisioning acceptance per library. Each pinned Pickett build SHALL pass an
  acceptance run on scratch copies of the example shipped for its library (SPINV: HCO with SPFIT
  and SPCAT; DPI: NO with DPFIT and DPCAT), recording byte equality or the listed differences.
  *Source* R1-02, R4-07 · Core · T, V-PICK-05 · proposed; the DPI part is blocked on D-SPYC-17
  (DPFIT/DPCAT)
- **SF-RUN-008** Use rights recorded. Each external engine record SHALL state the source, the
  PI-determined use rights and whether the engine may be redistributed, and an engine without
  recorded rights SHALL NOT be bundled.
  *Source* R1-16; errata row 9 · Core · I, V-RUN-08 · proposed; use rights blocked on D-SPYC-17
- **SF-RUN-009** Pickett sources. SPFIT and SPCAT SHALL be built only from the calpgm source files
  published at `https://spec.jpl.nasa.gov/ftp/pub/calpgm/` whose SHA-256 match a recorded source
  manifest.
  *Source* D-SPYC-17 (PI 2026-10-17) · Core · T, V-RUN-09 · proposed

### 4.2 Experimental input

Owner: `io/experimental`.

- **SF-ING-001a** Calibrated input. The core SHALL import calibrated spectra and peak lists in
  the formats listed for each profile.
  *Source* Amd SF-ING-001 (PI 2026-10-16) · Core · T, V-ING-01 · proposed
- **SF-ING-001b** One observation contract. Records produced by raw-data adapters SHALL validate
  against the same observation schema as direct imports.
  *Source* Amd SF-ING-001 · Core · T, V-ING-01 · proposed
- **SF-ING-002a** Observation identity. Each observation record SHALL hold a stable id, source,
  observable, units, conditions and processing history.
  *Source* Amd SF-ING-002 (PI 2026-10-16) · Core · T, V-ING-02 · proposed
- **SF-ING-002b** Missing metadata. Missing metadata SHALL be stored as absent with a reason.
  *Source* Amd SF-ING-002 · Core · T, V-ING-02 · proposed
- **SF-ING-002c** Missing measurement uncertainty. A missing measurement uncertainty SHALL be
  stored as absent and SHALL NOT be replaced by any number.
  *Source* Amd SF-ING-002; SRS SF-IO-002 · Core · T, V-ING-03 · proposed
- **SF-ING-002d** Calibration kinds. Frequency calibration SHALL NOT set the flag that an
  observation is intensity-calibrated.
  *Source* Amd SF-ING-002 · Core · T, V-ING-02 · proposed
- **SF-ING-002e** No silent drops. A record rejected at import SHALL be reported with a reason.
  *Source* SRS SF-IO-002 · Core · T, V-ING-02 · proposed
- **SF-IO-003a** Source preservation. Source files SHALL be stored byte-identical with their SHA-256.
  *Source* SRS SF-IO-003 · Core · T, V-ING-04 · proposed
- **SF-IO-003b** Derived records. Each conversion, normalisation, baseline correction, mask,
  subtraction or resampling SHALL create a derived record linked to its parent and parameters.
  *Source* SRS SF-IO-003 · Core · T, V-ING-04 · proposed
- **SF-IO-003c** Record kinds. Each record SHALL carry one kind from {profile, peak list,
  assigned line, constant set, computed quantity}.
  *Source* SRS SF-IO-003 · Core · T, V-ING-04 · proposed
- **SF-THZ-001** THz-TDS observations. A THz-TDS observation SHALL carry α(ν) and n(ν) with
  uncertainties, the extraction method and version, sample thickness, time-window length and
  α_max(ν), with points above α_max flagged.
  *Source* R3-11; [R-17] · Profile P11 · T, V-THZ-01 · proposed
- **SF-RAMAN-001** Raman observations. A Raman observation SHALL record the excitation vacuum
  wavenumber, scattering geometry and polarisation, detector response type (power or photon
  counting), the intensity-calibration standard or `uncalibrated`, and temperature.
  *Source* R3-10; [R-15, R-16] · Profile P9 · T, V-RAMAN-01 · proposed

### 4.3 TORQ and computational providers

Owner: `contract`.

- **SF-PHYS-001a** Provider tensors keep provenance. Imported geometries, Hessians, dipole
  derivatives, modes, constants and other tensors SHALL retain atom mapping, frame, units,
  method and version, and source.
  *Source* SRS SF-PHYS-001 · Core · T, V-PROV-01 · proposed
- **SF-PHYS-001b** Shape and symmetry checks. An imported tensor whose shape or symmetry violates
  its declared kind SHALL raise a typed error.
  *Source* SRS SF-PHYS-001 · Core · T, V-PROV-01 · proposed
- **SF-ML-002** Provider declarations. A computational or MLFF provider artifact SHALL be rejected
  unless it declares molecular identity, atom mapping, units, supported derivatives,
  training and reference provenance, applicability domain and model version.
  *Source* SRS SF-ML-002; Amd SF-SCOPE-005 · Core · T, V-PROV-02 · proposed
- **SF-SCOPE-005** PES refinement through an interface. PES and force-field refinement SHALL be
  exchanged with providers only through the versioned provider interface of this section.
  *Source* Amd SF-SCOPE-005 (PI 2026-10-16) · Profile P14 · T, V-PROV-03 · proposed
- **SF-IF-001** TORQ bundle consumer. SpycFit SHALL read TORQ spectroscopy bundles
  (TORQ-EXP-004) only through a schema-version check that rejects unknown versions.
  *Source* TORQ SRS §12, V-CONSUMER · Profile · T, V-IF-01 · proposed; blocked on D-SPYC-16 and on
  TORQ's exporter, which does not exist yet
- **SF-IF-005a** Direct input of Pickett files. SpycFit SHALL accept user-supplied Pickett `.par`,
  `.var`, `.int`, `.lin` and `.cat` files as prediction inputs without TOPOS or TORQ.
  *Source* D-SPYC-19 (PI 2026-10-17) · Core · T, V-IF-05 · proposed
- **SF-IF-005b** Direct input of ORCA output. SpycFit SHALL accept user-supplied ORCA output files
  as prediction inputs without TOPOS or TORQ, reading the quantities listed in a published
  coverage table.
  *Source* D-SPYC-19 (PI 2026-10-17) · Core · T, V-IF-05 · proposed; blocked on D-SPYC-19 (which
  ORCA outputs and quantities, and whose parser)
- **SF-IF-005c** Direct inputs are labelled. A prediction from the direct-input path SHALL record
  its source file, format, SHA-256 and the producing program and version where the file states
  them, and SHALL be labelled user-supplied, never as a TORQ prediction.
  *Source* D-SPYC-19 (PI 2026-10-17); SF-PHYS-001a · Core · T, V-IF-05 · proposed
- **SF-IF-006** Vibrational and Raman predictions are imported. SpycFit SHALL obtain
  quantum-chemical vibrational and Raman predictions from TORQ or from the direct-input path,
  and SHALL NOT compute them itself.
  *Source* D-SPYC-06 (PI 2026-10-17) · Profile P6-P12 · I, V-IF-06 · proposed; blocked on TORQ's
  exporter for these quantities

### 4.4 BASE integration

Owner: `contract`, `ui`.

- **SF-IF-002** Module registration. SpycFit SHALL register through the `cochem.modules` entry
  point and receive work only through `cochem.module-handoff/1` manifests.
  *Source* BASE; architecture §1; D-SPYC-19 · Core · T, V-IF-02 · proposed
- **SF-IF-003** Roots from BASE. SpycFit SHALL write run outputs only under the artifact and
  scratch roots BASE supplies, and SHALL NOT write the source tree at run time.
  *Source* architecture §3 · Core · T, V-IF-03 · proposed
- **SF-IF-004** Declared BASE boundaries. For each SpycFit boundary BASE already declares
  (`pickett_export`, `pgopher_export`, `spectroscopy_snapshot`), the SRS SHALL state whether it is
  accepted, renamed or retired before G1.
  *Source* architecture §1 · Core · I, V-IF-04 · proposed
- **SF-IF-007a** BASE is required. The SpycFit package SHALL declare `cochem-base` as a required
  dependency.
  *Source* D-SPYC-19 (PI 2026-10-17: "CoChem-BASE though is a hard requirement") · Core · I, V-IF-07 · proposed
- **SF-IF-007b** No vendored BASE. SpycFit SHALL NOT ship its own copy of BASE modules.
  *Source* D-SPYC-19 (PI 2026-10-17) · Core · I, V-IF-07 · proposed

### 4.5 Command-line and Python API

- **SF-API-001** One headless API. Every UI action that changes scientific state SHALL be
  available through the documented Python API, which the UI calls.
  *Source* architecture §2 · Core · T, V-API-01 · proposed

### 4.6 Reference data

Owner: `bench` (benchmark registry).

- **SF-DATA-001a** Reference record identity. Each reference record SHALL hold the source,
  database-declared tag, version and date, retrieval URL and UTC time, HTTP Last-Modified,
  SHA-256, experimental or computational status, conditions, resolution and uncertainty or an
  absent reason.
  *Source* Amd SF-DATA-001 (PI 2026-10-16); R4-04 · Core · I, V-DATA-01 · proposed
- **SF-DATA-001b** Version mismatches flagged. A record whose directory listing and
  documentation disagree on version or date SHALL be flagged.
  *Source* R4-04 · Core · T, V-DATA-01 · proposed
- **SF-DATA-001c** Rotational sources named. Each rotational benchmark case SHALL cite a CDMS or
  JPL record or a primary publication, and SHALL state whether its SPFIT inputs came from that
  record, were reconstructed from publications, or are absent.
  *Source* R4-01; [R-8, R-9] · Core · I, V-DATA-01 · proposed
- **SF-DATA-001d** Redistribution class. Each reference record SHALL carry one redistribution
  class from {open licence (named), written permission on file, not redistributable, unknown},
  and records in the last two classes SHALL NOT be committed to a public repository.
  *Source* R4-03 · Core · I, V-DATA-02 · proposed
- **SF-DATA-001e** Fetch and verify. A test that uses a record not committed to the repository
  SHALL fetch it from its recorded location and verify its SHA-256 before use, and SHALL fail,
  not skip, if either step fails.
  *Source* R4-03; AGENTS.md · Core · T, V-DATA-02 · proposed
- **SF-DATA-002** Published catalogs. Forward-prediction acceptance against a published `.cat`
  SHALL compare only computed lines (positive TAG) with SPCAT output from the identical
  `.par`/`.var`/`.int`, and SHALL compare merged measured lines with the `.lin` observations.
  *Source* R4-05; [R-8] · Core · T, V-CMP-02 · proposed
- **SF-DATA-003** Database attributions are checked. Every fixture value attributed to a
  database SHALL be reproduced by lookup in the hash-pinned record, and unmatched values SHALL be
  labelled synthetic.
  *Source* R4-11 · Core · T, V-DATA-03 · proposed
- **SF-IO-005** Masses and constants offline. Isotope masses and constants SHALL come from the
  versioned snapshot of §1.7.5, recorded with source, version and SHA-256, without network
  queries at run time.
  *Source* SRS SF-IO-005; R4-10 · Core · T, V-DATA-04 · proposed; blocked on D-SPYC-02
- **SF-IO-007a** Shared-source evidence. Records that share a source SHALL count as one unit of
  evidence: a spectrum, its peaks and constants fitted to those peaks are not independent.
  *Source* SRS SF-IO-007; Amd SF-DATA-001 · Core · T, V-DATA-05 · proposed
- **SF-IO-007b** Lineage-keyed partitions. Benchmark partitions SHALL be formed by lineage key,
  isotopologue, conformer and source publication, so that no observation or value derived from it
  appears on both sides.
  *Source* R4-09; [R-32] · Core · T, V-DATA-05 · proposed
- **SF-IO-007c** Extrapolation labels. Each predicted transition used in validation SHALL carry an
  interpolation or extrapolation label computed from the per-state ranges of J, Ka, Kc (or K)
  and v in the fitted data and from validity limits the source states.
  *Source* R4-08 · Core · T, V-DATA-06 · proposed
- **SF-IO-007d** Temporal holdout. Where later measurements of a benchmark molecule exist, the
  benchmark SHALL include a publication-level holdout of the later data.
  *Source* R4-09 · Core · I, V-DATA-05 · proposed
- **SF-DATA-004** Vibrational sources by observable. Each vibrational profile SHALL name its
  reference source for each observable class (line positions, absolute cross sections, band
  envelopes, computed frequencies), and computed sources SHALL NOT count as experimental
  validation.
  *Source* R4-13 · Profile P6-P12 · I, V-DATA-07 · proposed

### 4.7 GEOM

Owner: `contract`, `export`.

- **SF-IF-008** Constants for structure determination. SpycFit SHALL export, for each fitted
  isotopologue, the fitted rotational constants with uncertainties, correlations, reduction,
  representation and isotopologue identity, in a versioned payload that GEOM reads.
  *Source* D-SPYC-10 (PI 2026-10-17); `CoChem-GEOM/CoChem_GEOM_Workflow_Plan_v4.md` Stage 1.0 ·
  Profile · T, V-IF-08 · proposed; blocked on the payload schema agreed with GEOM

---

## 5. Functional requirements

### 5.1 Workflows and observation model

Owner: `model` (observation schema), `ui`.

- **SF-SCOPE-001a** Rotational workflow. SpycFit SHALL demonstrate broadband rotational
  assignment and fitting with isotopologues and uncertainty in an accepted V-DEMO-1 record.
  *Source* Amd SF-SCOPE-001 (PI 2026-10-16) · Core · D, V-DEMO-1 · proposed
- **SF-SCOPE-001b** Joint workflow. SpycFit SHALL demonstrate joint rotational and resolved
  rovibrational fitting in an accepted V-DEMO-2 record.
  *Source* Amd SF-SCOPE-001 · Profile P6 · D, V-DEMO-2 · proposed
- **SF-SCOPE-001c** Computation-led workflow. SpycFit SHALL demonstrate computation-led
  assignment of a flexible molecule in an accepted V-DEMO-3 record.
  *Source* Amd SF-SCOPE-001 · Profile P13 · D, V-DEMO-3 · proposed
- **SF-SCOPE-001d** Vibrational workflow. SpycFit SHALL demonstrate vibrational band fitting,
  mode interpretation and envelope decomposition in an accepted V-DEMO-4 record.
  *Source* Amd SF-SCOPE-001 · Profile P7/P8 · D, V-DEMO-4 · proposed
- **SF-SCOPE-002** First-class modalities. For each row of the §1.2 capability matrix, SpycFit
  SHALL provide the sequence import, inspect, propose, assign, fit, compare and export, and list
  any unsupported step in the matrix.
  *Source* Amd SF-SCOPE-002 (PI 2026-10-16); R3-03, R3-15 · Profile · D and I, V-DEMO-5 · proposed
- **SF-SCOPE-003a** Separate fields. Each observation SHALL hold separate fields for spectral
  region, measurement technique, measured observable, experimental conditions and physical
  transition model.
  *Source* Amd SF-SCOPE-003; SRS SF-MODEL-001 · Core · T, V-OBS-01 · proposed
- **SF-SCOPE-003b** Raman shift is its own quantity. Raman shift and absolute frequency SHALL be
  distinct quantity types with no implicit conversion.
  *Source* Amd SF-SCOPE-003 · Profile P9 · T, V-OBS-01 · proposed
- **SF-SCOPE-003c** Intensity kinds are distinct. Absorbance, transmittance, integrated
  intensity and power spectral density SHALL be distinct quantity types.
  *Source* Amd SF-SCOPE-003; R3-15 ("spectral density" defined as power spectral density) · Core · T, V-OBS-01 · proposed
- **SF-SCOPE-006** Phase is recorded. Each observation and result SHALL carry its phase (gas,
  solution, solid), and statistics SHALL be aggregated within one phase only.
  *Source* Amd SF-SCOPE-006 (PI 2026-10-16) · Core · T, V-OBS-02 · proposed
- **SF-MODEL-002a** Capability records. Each solver or backend SHALL publish a versioned
  capability record giving supported species and state classes, Hamiltonian or force-field
  assumptions, operator conventions, truncation and basis limits, observables, precision and
  applicability limits.
  *Source* SRS SF-MODEL-002 · Core · I, V-OBS-03 · proposed
- **SF-MODEL-002b** Out-of-capability requests. A request outside a backend's capability record
  SHALL return an `unsupported` result naming the unmet field, and no numbers.
  *Source* SRS SF-MODEL-002 · Core · T, V-OBS-03 · proposed
- **SF-INT-001a** Intensity registry. Every intensity record SHALL carry a quantity kind from a
  controlled registry (line intensity, Einstein A, band intensity, cross-section, absorption
  coefficient, decadic or napierian absorbance, transmittance, Raman differential cross-section,
  Raman activity), units, reference temperature, partition-function source and convention,
  statistical-weight and spin-scaling convention, isotopic-abundance scaling, whether stimulated
  emission is included, and the zero of lower-state energy.
  *Source* R3-06; [R-8, R-14] · Core · T, V-INT-01 · proposed
- **SF-INT-001b** Abscissa medium. Every abscissa SHALL be declared vacuum or air, and any
  conversion SHALL name its refractive-index formula.
  *Source* R3-06 · Profile P9/P12 · T, V-INT-01 · proposed
- **SF-INT-001c** Unmappable conversions. A conversion between intensity conventions that cannot
  be mapped SHALL raise a typed error.
  *Source* R3-06 · Core · T, V-INT-02 · proposed

### 5.2 Spectroscopic models

Owner: `model`, `catalog`.

- **SF-PHYS-002** Prediction conventions recorded. Each rotational prediction SHALL record its
  reduction, representation, operators, sign and unit conventions, state labels, basis and
  truncation, and numerical convergence.
  *Source* SRS SF-PHYS-002 · Core · T, V-HAM-01 · proposed
- **SF-HAM-001** Native model equals SPCAT where comparable. Each native Hamiltonian term SHALL
  reproduce the pinned SPCAT's energies and frequencies for the same parameters, within the T1
  tolerance of SF-CMP-008a, before it is enabled.
  *Source* PI 2026-10-16 (native beside SPCAT); D-SPYC-18 · Core · T, V-HAM-02 · proposed; scope
  of terms blocked on D-SPYC-18
- **SF-HAM-002** Label swaps are visible. When eigenvector labelling changes a transition's
  quantum-number assignment between iterations or parameter updates, SpycFit SHALL halt the
  update and show the swap to the expert.
  *Source* D-SPYC-07 · Core · T, V-HAM-03 · proposed; blocked on D-SPYC-07
- **SF-PHYS-003a** Constant types. Constants SHALL be typed as equilibrium (B_e), state-averaged
  (B_v) or fitted effective, and a comparison between unlike types without a recorded correction
  SHALL be refused.
  *Source* SRS SF-PHYS-003; TORQ contract §3.1 · Core · T, V-HAM-04 · proposed
- **SF-PHYS-003b** No automatic conformer rejection. No conformer SHALL be rejected
  automatically from a percentage difference between constants of unlike types.
  *Source* SRS SF-PHYS-003; TORQ contract §3.1 · Core · T, V-HAM-04 · proposed
- **SF-PHYS-003c** Vibrational frequency provenance. Every predicted vibrational frequency SHALL
  be labelled harmonic, scaled-harmonic or anharmonic with its method, and a scaled value SHALL
  record the scale factor's kind (harmonic, fundamental or ZPVE), value, standard uncertainty,
  source and version, and the level of theory it was derived for.
  *Source* R3-08; [R-28] · Profile P6-P12 · T, V-VIB-03 · proposed
- **SF-PHYS-003d** Anharmonic resonance treatment. A VPT2-family value SHALL record the resonance
  tests and thresholds, the resonances treated and how, and whether intensities include
  electrical anharmonicity.
  *Source* R3-08; [R-29] · Profile P6-P12 · T, V-VIB-03 · proposed
- **SF-PHYS-003e** VPT2 applicability. Large-amplitude modes SHALL be flagged as outside VPT2
  applicability.
  *Source* R3-08; [R-29] · Profile P6-P12 · T, V-VIB-03 · proposed
- **SF-PHYS-004a** Frame transformations. Axis alignment, atom mapping, isotope substitution and
  frame transformations SHALL preserve right-handedness and the declared permutation and tensor
  conventions, checked on every transform.
  *Source* SRS SF-PHYS-004 · Core · T, V-HAM-05 · proposed
- **SF-PHYS-004b** Near-degenerate moments. Near-degenerate principal moments (threshold
  declared) SHALL produce a recorded warning naming the ambiguous axes.
  *Source* SRS SF-PHYS-004 · Core · T, V-HAM-05 · proposed
- **SF-PHYS-004c** Isotopic reweighting approximations. Isotopic reweighting within a shared
  electronic model SHALL record its approximations.
  *Source* SRS SF-PHYS-004 · Core · T, V-HAM-05 · proposed
- **SF-PHYS-006** Many-to-many vibrational assignments. Vibrational assignments SHALL allow
  combinations, overtones, resonance mixing, ambiguous mode correspondence and several
  contributors to one feature.
  *Source* SRS SF-PHYS-006 · Profile P6-P12 · T, V-VIB-01 · proposed
- **SF-PHYS-007a** IR and Raman intensity models. IR absorption and Raman scattering intensities
  SHALL use distinct, recorded models that state temperature, populations, partition function,
  selection rules and instrument response.
  *Source* SRS SF-PHYS-007 · Profile P6-P12 · T, V-INT-03 · proposed
- **SF-PHYS-007b** Uncalibrated intensities. Without intensity calibration, intensity-based
  results SHALL be labelled `uncalibrated` and SHALL NOT enter quantitative intensity claims.
  *Source* SRS SF-PHYS-007; Amd SF-ING-002 · Core · T, V-INT-03 · proposed
- **SF-RAMAN-002** Raman intensity kind. A predicted Raman intensity SHALL state whether it is an
  activity or a cross-section, the excitation wavenumber and frequency-power factor used
  (ν⁴ for power, ν³ for photon counting), and the Bose-Einstein factor applied.
  *Source* R3-10 · Profile P9/P10 · T, V-RAMAN-02 · proposed
- **SF-RAMAN-003** Rotational Raman rules. A rotational Raman prediction SHALL apply the declared
  selection rules and nuclear-spin statistical weights.
  *Source* R3-10 · Profile P10 · T, V-RAMAN-03 · proposed
- **SF-PHYS-008a** Line-shape declaration. Every line-shape model SHALL declare its profile family
  and version (Gaussian, Lorentzian, Voigt, speed-dependent, Hartmann-Tran or its revisions,
  Van Vleck-Weisskopf), whether widths are HWHM or FWHM, area normalisation in the declared
  abscissa unit, and for each width and shift its reference temperature, pressure and
  temperature law.
  *Source* SRS SF-PHYS-008; R3-05; [R-27] · Profile · T, V-LS-01 · proposed
- **SF-PHYS-008b** Instrument line shape. Every profile model SHALL declare the instrument line
  shape (for FTIR, apodisation and optical path difference; for THz-TDS, the time window).
  *Source* R3-05 · Profile · T, V-LS-01 · proposed
- **SF-PHYS-008c** Line-shape oracle. Each profile implementation SHALL reproduce an independent
  implementation (for example HAPI) and the analytic Gaussian and Lorentzian limits on a versioned
  grid within TBD (D-SPYC-14).
  *Source* R3-05; [R-27] · Profile · T, V-LS-02 · proposed
- **SF-PHYS-008d** Mixed predictions declare populations. Multi-temperature or mixed-species
  predictions SHALL declare the population of each component.
  *Source* SRS SF-PHYS-008; TORQ contract §3.1 (mixtures are sums of spectra) · Core · T, V-LS-03 · proposed
- **SF-PHYS-008e** Chunked equals unchunked. A chunked or windowed computation SHALL equal the
  unchunked reference within TBD (D-SPYC-14).
  *Source* SRS SF-PHYS-008 · Core · T, V-LS-03 · proposed
- **SF-VIB-001** Gas-phase contours. For a gas-phase band, SpycFit SHALL model the envelope as a
  rotational contour from a declared rovibrational model (band type, constants, temperature,
  instrument line shape), and SHALL label any peak-sum decomposition of such a band as empirical.
  *Source* R3-02; [R-10] · Profile P7 · T, V-VIB-02 · proposed
- **SF-NIR-001** Overtone models. Overtone and combination predictions SHALL come from a declared
  anharmonic model (VPT2-family up to a stated number of quanta, local-mode or variational), and
  a harmonic provider SHALL report such intensities as unsupported.
  *Source* R3-12; [R-18] · Profile P12 · T, V-NIR-01 · proposed
- **SF-ROVIB-001** Rovibrational references. Each resolved-rovibrational benchmark SHALL name at
  least one qualified external engine and one assigned reference line list, and its
  mode-numbering convention.
  *Source* R3-13 · Profile P6 · I, V-ROVIB-01 · proposed
- **SF-PHASE-001** Condensed-phase models. A condensed-phase model SHALL declare its environment
  treatment (implicit, explicit or periodic) and SHALL label band widths as empirical.
  *Source* R3-14 · Profile P8/P11 · T, V-PHASE-01 · proposed

### 5.3 Expert control

Owner: `assign`, `session`, `ui`.

- **SF-HITL-001a** Acceptance by expert action. An assignment SHALL become accepted only through
  an expert-action record naming the actor, time and assignment revision.
  *Source* Amd SF-HITL-001 (PI 2026-10-16); SRS SF-UI-001 · Core · T, V-HITL-01 · proposed
- **SF-HITL-001b** Computation never accepts. No automated computation SHALL change an
  assignment's acceptance status.
  *Source* Amd SF-HITL-001 · Core · T, V-HITL-01 · proposed
- **SF-HITL-002a** Assignment actions. Accept, reject, tentative, blend, reassign and undo SHALL
  be available in every modality of the capability matrix.
  *Source* Amd SF-HITL-002 (PI 2026-10-16); SRS SF-UI-002 · Core · T and D, V-HITL-02 · proposed
- **SF-HITL-002b** Non-drag path. Each action SHALL be available without dragging.
  *Source* SRS SF-UI-002 · Core · T, V-HITL-02 · proposed
- **SF-HITL-002c** Reassignment keeps measurements. Reassigning SHALL change the assignment
  relationship and leave the observation record unchanged.
  *Source* Amd SF-HITL-002 · Core · T, V-HITL-02 · proposed
- **SF-HITL-002d** Rejections persist. A rejection SHALL persist until the expert reverses it,
  unless new evidence is explicitly surfaced to the expert.
  *Source* Amd SF-HITL-002 · Core · T, V-HITL-02 · proposed
- **SF-HITL-003a** Stale results are marked. A reassignment SHALL mark every dependent residual,
  prediction, parameter, uncertainty and output stale until recomputed.
  *Source* Amd SF-HITL-003; SRS SF-UI-004 · Core · T, V-HITL-03 · proposed
- **SF-HITL-003b** No older overwrites. A result computed from revision n SHALL NOT overwrite a
  result or decision of a later revision.
  *Source* Amd SF-HITL-003; SRS SF-UI-005 · Core · T, V-HITL-03 · proposed
- **SF-UI-003a** Assignment view. The assignment view SHALL show overlays, residuals,
  alternatives, physical labels, the uncertainty category and supporting evidence.
  *Source* SRS SF-UI-003 · Core · D, V-UI-01 · proposed
- **SF-UI-003b** Accepted is not unique. The view SHALL distinguish an expert-accepted
  interpretation from a uniquely demonstrated one.
  *Source* SRS SF-UI-003 · Core · D, V-UI-01 · proposed
- **SF-UI-004** Previews are labelled. An approximate preview of a parameter change SHALL be
  labelled approximate and SHALL NOT be stored as a result.
  *Source* SRS SF-UI-004 · Core · T and D, V-UI-02 · proposed
- **SF-UI-005** Reversible edits. Rejections, masks and subtractions SHALL be stored as reversible
  revision events.
  *Source* SRS SF-UI-005 · Core · T, V-HITL-03 · proposed

- **SF-SET-001** Parameters before predictions. SpycFit SHALL NOT generate a prediction, window
  or catalog for a project until the project's temperature, frequency window and J_max are set.
  *Source* D-SPYC-20 (PI 2026-10-17) · Core · T, V-SET-01 · proposed
- **SF-SET-002** Editable parameters. Temperature, frequency window and J_max SHALL be editable
  at any time through input fields in the UI and through the API.
  *Source* D-SPYC-20 (PI 2026-10-17) · Core · T and D, V-SET-02 · proposed
- **SF-SET-003** Edits mark results stale. A change to any of these parameters SHALL mark every
  prediction, window and catalog made with the previous values stale until it is regenerated.
  *Source* D-SPYC-20; SF-HITL-003a · Core · T, V-SET-03 · proposed
- **SF-SET-004** Parameters in provenance. Every prediction, catalog and fit SHALL record the
  temperature, frequency window and J_max it used, and whether each value was entered by the
  user or taken from a preset.
  *Source* D-SPYC-20; AGENTS.md (defaults recorded as defaults) · Core · T, V-SET-04 · proposed

The UI toolkit is open decision D-SPYC-05. These requirements hold for any toolkit.

### 5.4 Fitting and inference

Owner: `fit`.

- **SF-FIT-001** Fit binding. Each fit record SHALL reference the observation revision,
  assignment revision, model, parameter states, objective, weights, starting values, priors and
  constraints, optimiser settings and convergence diagnostics.
  *Source* SRS SF-FIT-001 · Core · T, V-FIT-01 · proposed
- **SF-FIT-012** Parameter states. Every parameter SHALL carry one state from {free, constrained
  (prior value, uncertainty, source), fixed (value, source), ratio-tied, absent}, derived from the
  engine input (for SPFIT, from ERPAR and negative IDPAR under thresholds declared in the record).
  *Source* R2-09; [R-6] · Core · T, V-FIT-02 · proposed
- **SF-INF-001a** Experiment-only label. A fit SHALL be labelled experiment-only only if no
  parameter is constrained, fixed or ratio-tied to a computed or assumed non-zero value.
  *Source* Amd SF-INF-001 (PI 2026-10-16); R2-09 · Core · T, V-FIT-02 · proposed
- **SF-INF-001b** Side-by-side results. A fit that uses computational priors or fixed computed
  values SHALL be reported beside the experiment-only fit of the same data, each with its own
  values, covariance, engine, units and model revision.
  *Source* Amd SF-INF-001 (PI 2026-10-16) · Core · T, V-FIT-03 · proposed
- **SF-INF-001c** Prior sensitivity. Each informed fit SHALL report refits with each prior width
  multiplied by 0.5 and 2 and each fixed value moved by ±1 standard uncertainty, as parameter
  shifts in units of their uncertainty.
  *Source* Amd SF-INF-001; R2-10 · Core · T, V-FIT-03 · proposed
- **SF-INF-001d** Starting values are not evidence. The origin of starting values SHALL NOT change
  a fit's evidence label.
  *Source* SRS SF-FIT-002 · Core · T, V-FIT-02 · proposed
- **SF-FIT-013** Fixed-parameter uncertainty. Each fit with fixed or constrained parameters SHALL
  report the conditional covariance, the contribution of each fixed parameter's declared
  uncertainty (consider covariance), and the total; a fixed parameter without a declared
  uncertainty SHALL make the total `unknown`.
  *Source* R2-10; GUM §4.3.1, §5.2.2 [R-22] · Core · T, V-FIT-04 · proposed
- **SF-FIT-014** Fit statistics. Every fit record SHALL report, with formula and value: N (each
  blend once), the number of free parameters M, ν = N − M, χ² from the declared weights,
  σ_f = √(χ²/ν), mean and rms residual per data class and unit, and max |r|/u; SPFIT's RMS ERROR,
  MICROWAVE RMS and IR RMS SHALL appear beside these under their own names.
  *Source* R1-03, R2-01; §1.7.4 · Core · T, V-FIT-05 · proposed
- **SF-FIT-015** Uncertainty scaling. Each parameter SHALL be reported with its unscaled standard
  uncertainty and with the value scaled by σ_f, with the primary choice declared, SPFIT's FRAC
  recorded, and a flag when σ_f lies outside a declared χ²_ν interval.
  *Source* R1-03, R2-02 · Core · T, V-FIT-05 · proposed
- **SF-FIT-016a** Reporting convention. Uncertainties SHALL be reported as standard uncertainties
  (k = 1) in value(u) notation unless k, its basis and the coverage probability are stated.
  *Source* R2-03; GUM §7.2 [R-22] · Core · T, V-FIT-06 · proposed
- **SF-FIT-016b** Covariance export. The full covariance and correlation matrices SHALL be
  exported, transformed as C′ = D·C·D under unit or sign conversion (D the diagonal of signed
  conversion factors).
  *Source* R2-03; GUM §7.2.5 [R-22] · Core · T, V-FIT-06 · proposed
- **SF-FIT-004a** Correlation definition. Correlations SHALL be computed as C_ij/√(C_ii·C_jj).
  *Source* SRS SF-FIT-004; GUM C.3.6 · Core · T, V-FIT-07 · proposed
- **SF-FIT-004b** Conditioning. Each fit SHALL report the singular values and condition number of
  the column-scaled weighted Jacobian, with the scaling declared, and for each singular value
  below a declared threshold its right-singular vector in named parameters.
  *Source* SRS SF-FIT-004; R2-05; [R-25] · Core · T, V-FIT-07 · proposed
- **SF-FIT-004c** Structural indeterminacy. SpycFit SHALL refuse a fit configuration that is
  structurally indeterminate (terms removed by the declared reduction, or exact rank loss of the
  normal matrix at a random parameter point), naming the dependent combination.
  *Source* R2-06; [R-20] · Core · T, V-FIT-07 · proposed
- **SF-FIT-005a** Sensitivity records. A sensitivity result SHALL name the scalar observable,
  input distribution and its source, the independence assumption, the estimators, the sample
  size, the sampling scheme and bootstrap intervals, and SHALL report unclipped estimates.
  *Source* SRS SF-FIT-005; R2-07; [R-24] · Core · T, V-FIT-08 · proposed
- **SF-FIT-005b** Estimator oracle. Each sensitivity estimator SHALL reproduce the analytic
  indices of the linear additive and Ishigami test functions within three bootstrap standard
  errors.
  *Source* R2-07; Amd §5 (convolver defect) · Core · T, V-FIT-08 · proposed
- **SF-FIT-005c** Sensitivity never sets state. No sensitivity result SHALL change a parameter's
  state; a proposal to fix, constrain or drop a parameter SHALL take effect only by a recorded
  expert action.
  *Source* SRS SF-FIT-005; Amd §4 · Core · T, V-FIT-08 · proposed
- **SF-FIT-011** Evidence for freezing. A proposal to fix, constrain or drop a parameter SHALL show
  the motivating singular-vector combination, Δχ², Δσ_f, the change in predictions over a
  declared transition set, and the shift of every other parameter in units of its uncertainty
  when the fixed value moves by ±1 standard uncertainty.
  *Source* R2-08; [R-25] · Core · T, V-FIT-09 · proposed
- **SF-FIT-003a** Shared parameters. A joint fit SHALL declare which physical parameters are shared
  between data sets.
  *Source* SRS SF-FIT-003 · Profile P6 · T, V-FIT-10 · proposed
- **SF-FIT-003b** Per-data-set models. Each data set in a joint fit SHALL keep its own signal
  model and calibration.
  *Source* SRS SF-FIT-003 · Profile P6 · T, V-FIT-10 · proposed
- **SF-FIT-003c** No double counting. A joint fit SHALL count each blend once and SHALL reject a
  second copy of an observation with the same lineage key.
  *Source* SRS SF-FIT-003; R2-13 · Core · T, V-FIT-10 · proposed
- **SF-FIT-009** Joint MW and IR fits. A joint fit SHALL weight each observation by its stated
  uncertainty in its native unit, record unit conversions (for SPFIT, the ERR sign and CAL value),
  report weighted rms and mean residual per data set and per state, and declare for each
  lower-state parameter whether it was co-fitted, fixed to a cited value or constrained by a prior.
  *Source* R3-09; [R-6] · Profile P6 · T, V-FIT-10 · proposed
- **SF-FIT-017** Exclusions are recorded. Every line excluded or reweighted by a rule SHALL be
  recorded with the rule, threshold and iteration, with N and ν recomputed and the excluded lines
  exported; robust weighting SHALL be off unless selected and labelled.
  *Source* R2-13; [R-6] · Core · T, V-FIT-11 · proposed
- **SF-FIT-018** Nonlinear intervals. For parameters declared strongly nonlinear, SpycFit SHALL
  offer a profile-χ² or Monte Carlo interval beside the linearised uncertainty.
  *Source* R2-14; JCGM 101 [R-22] · Profile P4 · T, V-FIT-12 · proposed
- **SF-FIT-008a** Derivatives verified. Analytic or automatic derivatives SHALL agree with finite
  differences at two step sizes within TBD (D-SPYC-14), including a case with degenerate levels.
  *Source* SRS SF-FIT-008 · Core · T, V-FIT-13 · proposed
- **SF-FIT-008b** Speed claims. A claim of acceleration SHALL cite the workload, host and a
  precision check.
  *Source* SRS SF-FIT-008 · Optional · A, V-PERF-01 · proposed
- **SF-INF-003** Predictive calibration. Predictive calibration SHALL be measured on observations
  excluded from the fit (blends, hyperfine components and same-source isotopologues kept on one
  side), reporting n, mean z, Var(z) with a bootstrap interval and coverage at k = 1, 2, 3, overall
  and stratified by the extrapolation label of SF-IO-007c.
  *Source* R2-12; [R-26] · Core · T, V-CAL-01 · proposed

### 5.5 Computational assistance

Owner: `assign`.

- **SF-ML-001a** Scores and probabilities. An ML output SHALL be labelled `heuristic` unless a
  held-out reliability diagram and expected calibration error (classification) or the SF-INF-003
  coverage (regression) on exchangeable data support the label `calibrated`.
  *Source* SRS SF-ML-001; R2-15; [R-35] · Profile P13 · T, V-ML-01 · proposed
- **SF-ML-001b** Abstention. Outside its declared applicability domain an ML component SHALL
  return `out_of_domain` and no recommendation.
  *Source* SRS SF-ML-001; R2-15 · Profile P13 · T, V-ML-01 · proposed
- **SF-ML-001c** Conformal intervals. Conformal intervals SHALL follow the TORQ contract §3.2
  rank rule and SHALL distinguish marginal from conditional coverage.
  *Source* R2-15; TORQ contract §3.2 · Profile P13 · T, V-ML-02 · proposed
- **SF-ML-003a** Inverse-fit records. PES or force-field inverse fits SHALL record model
  versions, objectives, constraints and training and evaluation partitions.
  *Source* SRS SF-ML-003 · Profile P14 · T, V-PROV-03 · proposed
- **SF-ML-003b** Observable-level validation. An MLFF-derived spectroscopic quantity SHALL be
  validated against measured frequencies or constants before it is used as a prior.
  *Source* SRS SF-ML-002, SF-ML-003; R2-15 · Profile P13 · T, V-ML-03 · proposed

### 5.6 SPCAT/SPFIT comparison

Owner: `compare`. The comparison stages are those of amendment §3.2: forward prediction,
experimental fitting, cross-prediction (a two-by-two matrix), assignment review and the
informed-fit extension.

- **SF-CMP-001a** Genuine engines beside the native path. For each model in a profile that
  SPCAT/SPFIT support, SpycFit SHALL run the pinned SPCAT and SPFIT and the native path on the
  same inputs.
  *Source* Amd SF-CMP-001 (PI 2026-10-16); D-SPYC-17 · Core · T, V-CMP-01 · proposed
- **SF-CMP-001b** Results kept per engine. Each engine's results SHALL be stored separately.
  *Source* Amd SF-CMP-001 · Core · T, V-CMP-01 · proposed
- **SF-CMP-002** Comparability first. A comparability record (isotopologue, conformer, state,
  reduction, representation, operator and sign conventions, truncation, axis convention,
  quantum-number mapping, units, hyperfine treatment, parameter states) SHALL be completed before
  any metric, and any mismatch SHALL yield INCOMPATIBLE_MODEL.
  *Source* Amd SF-CMP-002 · Core · T, V-CMP-03 · proposed
- **SF-CMP-002a** Truncations listed. The forward-stage record SHALL list every SPCAT truncation
  and cutoff (KNMIN/KNMAX, frequency and intensity limits, MAXV).
  *Source* R1-10 · Core · T, V-CMP-03 · proposed
- **SF-CMP-002b** Same limits. The native model SHALL apply the truncations and cutoffs listed
  under SF-CMP-002a.
  *Source* R1-10 · Core · T, V-CMP-03 · proposed
- **SF-CMP-002c** Cutoff losses counted apart. Lines missing only because of a cutoff SHALL be
  counted separately from unmatched lines.
  *Source* R1-10 · Core · T, V-CMP-03 · proposed
- **SF-CMP-003** Intensity comparability. An intensity comparison SHALL require matched
  temperature, partition-function and degeneracy conventions, dipole definitions, intensity units
  and populations, and SHALL otherwise yield INCOMPATIBLE_MODEL for intensities only.
  *Source* Amd SF-CMP-003 · Core · T, V-CMP-04 · proposed
- **SF-CMP-003a** Intensity record. An intensity comparison SHALL record TEMP, the QROT supplied
  to SPCAT, SPCAT's summed Q at TEMP, the spin-weight convention and the signed dipole components.
  *Source* R1-07; [R-8] · Core · T, V-CMP-04 · proposed
- **SF-CMP-003b** Rescaled intensities. Intensities rescaled from a 300 K LGINT to another
  temperature SHALL be labelled approximate and SHALL NOT yield WITHIN_TOLERANCE.
  *Source* R1-07; [R-8] · Core · T, V-CMP-04 · proposed
- **SF-CMP-004a** Matching by state identity. Transitions SHALL be matched by state identity, J,
  symmetry block and energy rank within the block, never by frequency alone.
  *Source* Amd SF-CMP-004; R1-08 · Core · T, V-CMP-05 · proposed
- **SF-CMP-004b** Labelling rule recorded. Each engine's labelling rule (for SPFIT/SPCAT the DIAG
  value) SHALL be recorded with the comparison.
  *Source* Amd SF-CMP-004; R1-08 · Core · T, V-CMP-05 · proposed
- **SF-CMP-004c** Ambiguity flagged. A transition with more than one candidate inside the
  tolerance SHALL be flagged ambiguous.
  *Source* Amd SF-CMP-004; R1-08 · Core · T, V-CMP-05 · proposed
- **SF-CMP-004d** Blends by declared rule. Unresolved blends and mixed-state labels SHALL be
  compared under a declared rule.
  *Source* Amd SF-CMP-004 · Core · T, V-CMP-05 · proposed
- **SF-CMP-005a** Incompatible models. Incompatible models or objectives SHALL yield
  INCOMPATIBLE_MODEL.
  *Source* Amd SF-CMP-005 · Core · T, V-CMP-03 · proposed
- **SF-CMP-005b** Convention mappings tested. Each transformation between parameter conventions
  SHALL have its own acceptance test and record its truncation limitations.
  *Source* Amd SF-CMP-005 · Core · T, V-CMP-06 · proposed
- **SF-CMP-005c** Observable comparison stays separate. Where parameter comparison is invalid but
  observable comparison is meaningful, the two SHALL be reported separately.
  *Source* Amd SF-CMP-005 · Core · T, V-CMP-06 · proposed
- **SF-CMP-006** Transition metrics. SpycFit SHALL report per-transition signed and absolute
  differences, maximum and rms differences, residuals per engine, matched and unmatched counts and
  fit diagnostics, with summaries linked to each divergent transition.
  *Source* Amd SF-CMP-006 · Core · T, V-CMP-07 · proposed
- **SF-CMP-007** Parameter differences. Parameter differences SHALL be reported only between
  parameters of the same type and convention, with native and normalised units, uncertainty
  category, parameter state and covariance where supplied.
  *Source* Amd SF-CMP-007 · Core · T, V-CMP-08 · proposed
- **SF-CMP-008a** Four tolerance tiers. Each profile SHALL declare before evaluation: T1, forward
  agreement as half the print quantum of the compared reference output plus a stated numerical
  bound; T2, fitted-parameter agreement in units of standard uncertainty with each engine's
  convergence criterion; T3, limits on normalised residuals and weighted rms per data set from
  per-line uncertainties; T4, predictive coverage on held-out lines; and WITHIN_TOLERANCE SHALL
  name its tier.
  *Source* Amd SF-CMP-008; R1-09 · Core · I and T, V-CMP-09 · proposed; values TBD (D-SPYC-14)
- **SF-CMP-009** Shared-data covariance. An uncertainty-normalised comparison of fits to the same
  observations SHALL state its treatment of shared-data covariance or carry the label
  `approximation`.
  *Source* Amd SF-CMP-009 · Core · T, V-CMP-08 · proposed
- **SF-CMP-010** Divergence persists. A DIVERGENT status SHALL persist when each engine converges
  separately, and SHALL carry a cause from {expected model difference, unresolved mismatch,
  parsing or convention error, numerical discrepancy, failure to explain observations}.
  *Source* Amd SF-CMP-010 · Core · T, V-CMP-10 · proposed
- **SF-CMP-011a** Engine run record. Each engine run SHALL store executable path and SHA-256,
  source-manifest hash and build recipe, platform, exact input files, command, exit status,
  stdout, stderr, all output files, parser version, run id and comparison configuration.
  *Source* Amd SF-CMP-011; R1-11 · Core · T, V-CMP-11 · proposed
- **SF-CMP-012a** One status. Each comparison record SHALL carry exactly one status from §0.4.
  *Source* Amd SF-CMP-012 · Core · T, V-CMP-12 · proposed
- **SF-CMP-012b** Empty is not agreement. An empty match set, missing statistics, empty parsed
  output or unsupported requirement SHALL NOT yield WITHIN_TOLERANCE.
  *Source* Amd SF-CMP-012; Amd §5 (auditor defect) · Core · T, V-CMP-12 · proposed
- **SF-CMP-012c** Partial coverage. A comparison with partial coverage SHALL be INCOMPLETE unless
  a compared quantity diverges, in which case it SHALL be DIVERGENT with the missing quantities
  listed.
  *Source* Amd SF-CMP-012 · Core · T, V-CMP-12 · proposed
- **SF-CMP-012d** Diagnostics over exit codes. A Pickett run SHALL be classified FAILED or
  INCOMPLETE whatever its exit status when its outputs show `Bad Line` records, lines not used in
  the final iteration, an empty or truncated `.cat`, `Fit Diverging`, or termination at NITR with
  the last relative change in RMS ERROR above the declared criterion.
  *Source* R1-05; [R-6] · Core · T, V-CMP-13 · proposed
- **SF-CMP-013** Results come from the named engine. A result labelled SPCAT or SPFIT SHALL have
  an engine run record with a matching executable digest, and a failure SHALL NOT be replaced by
  another calculation.
  *Source* Amd SF-CMP-013; SRS SF-RUN-002 · Core · T, V-CMP-11 · proposed
- **SF-CMP-014a** Comparison views. The comparison view SHALL show spectral overlays, transition
  and constant tables, residual and difference views, coverage and the provenance record, with
  navigation from a discrepancy to its observations, alternatives and model assumptions.
  *Source* Amd SF-CMP-014 · Core · D, V-UI-03 · proposed
- **SF-CMP-014b** Rounding never hides differences. Displayed differences SHALL show at least one
  more significant digit than the tolerance of their tier, and a non-zero difference SHALL never
  display as zero.
  *Source* Amd SF-CMP-014; R1-15 · Core · T, V-UI-03 · proposed
- **SF-CMP-015a** Comparison does not decide. A comparison SHALL NOT change an assignment,
  constraint or model selection.
  *Source* Amd SF-CMP-015 (PI 2026-10-16, D-01) · Core · T, V-CMP-14 · proposed
- **SF-CMP-015b** Review before export. Exporting a result whose comparison is DIVERGENT SHALL
  require an expert-review record.
  *Source* Amd SF-CMP-015 (D-01); SRS SF-OUT-004 · Core · T, V-CMP-14 · proposed
- **SF-CMP-015c** Review is not agreement. The export package SHALL keep the review record and the
  DIVERGENT status unchanged.
  *Source* Amd SF-CMP-015 (D-01) · Core · T, V-CMP-14 · proposed
- **SF-CMP-016** SPFIT statistics stored with their meaning. Each SPFIT result SHALL store
  MICROWAVE RMS and IR RMS, RMS ERROR with its denominator, the numbers of fitted blends, rejected
  lines and free parameters, FRAC, and which file supplied each uncertainty, and parameter
  uncertainties SHALL be converted to one declared convention before any comparison across
  engines.
  *Source* R1-03; §1.7.4 · Core · T, V-CMP-15 · proposed
- **SF-CMP-017** Priors in the objective. The fitting-stage record SHALL state the objective as a
  data term plus a prior term, give each parameter's ERPAR and role, and state whether the input
  `.par` held a variance block, and a native fit SHALL be compared with SPFIT only if it implements
  the same prior and blend terms.
  *Source* R1-06; [R-6, R-7] · Core · T, V-CMP-16 · proposed
- **SF-CMP-018a** Predicted uncertainties keep their convention. Predicted uncertainties read
  from a `.cat` SHALL carry the FRAC convention of the fit that produced them.
  *Source* R1-12; [R-8] · Core · T, V-CMP-15 · proposed
- **SF-CMP-018b** Predicted is not measured. Predicted uncertainties SHALL be distinguished from
  experimental uncertainties (negative TAG).
  *Source* R1-12; [R-8] · Core · T, V-CMP-15 · proposed
- **SF-CMP-018c** Overflowed errors. ERR = 999.9999 in a `.cat` SHALL be stored as `exceeds field`,
  not as a number.
  *Source* R1-12; [R-8] · Core · T, V-CMP-15 · proposed
- **SF-CMP-019** Reference engines named. Each extended profile SHALL name its reference engines
  and their Hamiltonian methods and state whether comparison is valid at parameter or only at
  observable level.
  *Source* R1-13; [R-10..R-13] · Profile · I, V-CMP-17 · proposed
- **SF-CMP-020** Fitting comparisons follow the PI's model scope. Fitting-stage and
  cross-prediction comparisons SHALL run only for model terms that the native path implements
  under D-SPYC-18, and other terms SHALL carry NOT_RUN with the reason.
  *Source* R1-01 · Core · T, V-CMP-01 · proposed

### 5.7 Outputs and publication

Owner: `export`.

- **SF-OUT-001a** One record per number. Each exported number and plot SHALL reference one
  result record.
  *Source* SRS SF-OUT-001; Amd SF-PUB-001 · Core · T, V-OUT-01 · proposed
- **SF-OUT-001b** No mixed pairs. No export SHALL pair a value and an uncertainty from different
  result records or engines.
  *Source* SRS SF-OUT-001; Amd §5 (export defect) · Core · T, V-OUT-01 · proposed
- **SF-OUT-002a** Units follow numbers. A unit label SHALL be written only after the number has
  been converted to that unit.
  *Source* SRS SF-OUT-002; Amd §5 · Core · T, V-OUT-02 · proposed
- **SF-OUT-002b** Unknown uncertainty exported as absent. An unknown uncertainty SHALL be exported
  as an absent token and never as zero.
  *Source* SRS SF-OUT-002; Amd SF-ING-002 · Core · T, V-OUT-02 · proposed
- **SF-OUT-002c** Full precision kept. Machine-readable outputs SHALL keep full precision beside
  any display rounding.
  *Source* SRS SF-OUT-002; R2-04 · Core · T, V-OUT-02 · proposed
- **SF-OUT-005** Rounding checked. A publication table SHALL quote each uncertainty to one or two
  significant digits and each value to at least the digit of its uncertainty, and SHALL be
  accepted only if the rounded parameter set reproduces the unrounded predictions of all fitted
  data with rms(Δν/u) ≤ 0.1·σ_f, otherwise adding digits or applying recorded sequential rounding.
  *Source* R2-04; [R-23] · Core · T, V-OUT-03 · proposed
- **SF-OUT-006** SPFIT first. When a result exists from both SPFIT and the native path, publication
  tables SHALL present the SPFIT result first, with the native result in a separate table
  labelled with its engine.
  *Source* PI 2026-10-16 (conversation; `docs/decisions/D-SPYC-01.md`) · Core · T, V-OUT-06 · proposed
- **SF-OUT-004** Limitations exported. Exports SHALL include unresolved assignments, blends and
  model limitations.
  *Source* SRS SF-OUT-004 · Core · T, V-OUT-04 · proposed
- **SF-PUB-001a** Provenance graph. A publication package SHALL contain a provenance graph from
  each number and plot to its result record, observations, expert decisions, model, fit settings
  and validation status, expressed in W3C PROV terms.
  *Source* Amd SF-PUB-001 (PI 2026-10-16); [R-34] · Core · T, V-OUT-05 · proposed
- **SF-PUB-001b** Package contents. A publication package SHALL identify sources, molecular and
  state identities, conditions, models and conventions, assignments, expert decisions, the
  experiment-only and informed distinction, priors, the uncertainty interpretation, engine and run
  versions and validation status.
  *Source* SRS SF-OUT-003 · Core · I, V-OUT-05 · proposed
- **SF-VIB-002** Envelope decomposition evidence. Every envelope decomposition SHALL report the
  component count with the declared selection criterion and the competing models' scores, a
  jointly fitted baseline or the sensitivity to declared alternative baselines, the parameter
  correlation matrix, and a flag on each parameter whose profile or posterior interval is
  unbounded.
  *Source* R3-07; [R-30] · Profile P8 · T, V-VIB-04 · proposed
- **SF-SCOPE-004** Band fitting outputs. Band fits SHALL report position, width, area, baseline and
  overlapping components, each with its uncertainty and the ambiguity of mixed, overtone,
  combination or resonant assignments.
  *Source* Amd SF-SCOPE-004 (PI 2026-10-16) · Profile P7/P8 · T, V-VIB-04 · proposed

---

## 6. Non-functional requirements

Owner: all components.

- **SF-RUN-001a** Offline core. The core SHALL run the reference workflow of each accepted
  profile with network access disabled.
  *Source* SRS SF-RUN-001 (PI 2026-10-16, D-02) · Core · T, V-RUN-01 · proposed
- **SF-RUN-001b** Optional connected adapters. Connected import adapters SHALL be optional
  installation extras.
  *Source* SRS SF-RUN-001 · Core · I, V-RUN-01 · proposed
- **SF-RUN-001c** Traceable provisioning. Provisioned executables, data snapshots and dependencies
  SHALL carry a source URL and SHA-256.
  *Source* SRS SF-RUN-001 · Core · I, V-RUN-01 · proposed
- **SF-RUN-002a** CPU reference path. Each native capability SHALL have a validated CPU path.
  *Source* SRS SF-RUN-002 · Core · T, V-RUN-02 · proposed
- **SF-RUN-002b** Execution provenance. Each run SHALL record precision, backend, library
  versions, hardware and random seeds.
  *Source* SRS SF-RUN-002 · Core · T, V-RUN-02 · proposed
- **SF-RUN-002c** Accelerators match CPU. Accelerator output SHALL match the CPU path within
  TBD (D-SPYC-14).
  *Source* SRS SF-RUN-002; Amd §3.5 item 7 · Optional · T, V-RUN-02 · proposed
- **SF-RUN-003a** Resource limits. Core count, memory and wall-time limits SHALL be configurable
  per run.
  *Source* SRS SF-RUN-003 · Core · T, V-RUN-03 · proposed
- **SF-RUN-003b** Cache keys. Cache and chunk keys SHALL include model, data, state, code version,
  compiler, precision and hardware class.
  *Source* SRS SF-RUN-003 · Core · T, V-RUN-03 · proposed
- **SF-RUN-003c** Performance targets need workloads. A performance or memory target SHALL name
  its workload, host and the job that measures it.
  *Source* SRS SF-RUN-003 · Optional · I, V-PERF-01 · proposed
- **SF-RUN-004a** Distinct outcomes. Failure, cancellation, timeout and unsupported requests SHALL
  map to distinct outcomes from §0.4, each with a typed error that names the program or component.
  *Source* SRS SF-RUN-004 · Core · T, V-RUN-04 · proposed
- **SF-RUN-004b** Records survive failure. Inputs and run records SHALL be kept when a run fails.
  *Source* SRS SF-RUN-004 · Core · T, V-RUN-04 · proposed
- **SF-RUN-005a** Refinement request contents. A refinement request SHALL contain the diagnostic
  evidence, alternatives, intended observables, model revision, requested calculations and the
  user-approved resource policy, and list the SF-FIT-004 and SF-INF-003 diagnostics already run.
  *Source* SRS SF-RUN-005; R2 (unverifiable wording) · Profile P14 · T, V-RUN-05 · proposed;
  blocked on D-SPYC-03
- **SF-RUN-005b** New versions only. A returned refinement SHALL create a new version and leave
  accepted results unchanged.
  *Source* SRS SF-RUN-005 · Profile P14 · T, V-RUN-05 · proposed
- **SF-SEC-001** Subprocess safety. External programs SHALL be started without a shell, with an
  argument list, a fresh scratch directory, closed standard input, a timeout and a process-group
  boundary that never includes the caller.
  *Source* R5 §5; Phase 0 inventory (a cleanup routine killed the test runner's own process
  group) · Core · T, V-SEC-01 · proposed

The toolchain (Python version, numerical libraries, JAX or CuPy) is open decision D-SPYC-11.

---

## 7. Data records

Field lists are proposals for review with the architecture document. Every record carries
schema version, id, creation time and the provenance of its inputs.

| Record | Key fields | Requirements |
| :--- | :--- | :--- |
| Observation | id, kind, region, technique, observable, units, phase, conditions, calibration flags, uncertainty or absent reason, processing history, source SHA-256 | SF-ING-002, SF-IO-003, SF-SCOPE-003, SF-SCOPE-006 |
| AssignmentRevision | id, parent, observation ids, quantum-number mapping, hypothesis set, expert action (actor, time) | SF-HITL-001..003 |
| FitRecord | observation and assignment revisions, model and capability record, parameter states, objective, weights, priors, optimiser, statistics (SF-FIT-014), covariance, evidence label | SF-FIT, SF-INF |
| EngineRun | program, executable SHA-256, source manifest, build recipe, platform, inputs, argv, outcome, exit status, stdout, stderr, outputs, parser version | SF-CMP-011a, SF-CMP-012d |
| ComparisonRecord | stage, comparability record, engine runs, metrics, tier and tolerance, status, cause, coverage, expert review | SF-CMP |
| CapabilityRecord | backend, version, supported classes, conventions, limits, reference cases and their acceptance results | SF-MODEL-002, SF-PHYS-005 |
| ReferenceRecord | source, tag, version, dates, URL, retrieval time, Last-Modified, SHA-256, redistribution class, lineage key | SF-DATA-001, SF-IO-007 |
| RefinementRequest | evidence, alternatives, observables, model revision, requested calculations, resource policy | SF-RUN-005 |

Missing values use the absent states of §0.4 with a reason.

---

## 8. Verification

### 8.1 Evidence rules

Adopt TORQ SRS §15.1 (real engines with authentic outputs; malformed fixtures test robustness
only), §16 (a claimed agent audit cannot close a package) and V-RELEASE §15.2 (passed, failed,
skipped, blocked and unrun cases are all counted). A skipped or xfailed test never satisfies a
requirement (SF-VER-002, SF-VER-003).

### 8.2 Acceptance catalog

Pass conditions are summarised; each acceptance id becomes a test or a recorded demonstration
with its tolerance committed before the implementation PR (SF-VER-004).

| Id | Pass condition | Evidence class |
| :--- | :--- | :--- |
| V-PICK-01 | Every field in the coverage table parses and writes; unsupported fields raise typed errors naming field and line | software |
| V-PICK-02 | Writer output run through the pinned binary reproduces the original outputs on hash-verified inputs | external-engine |
| V-PICK-03 | SPCAT energies for each §1.7.2 row match an independent Watson Hamiltonian at J ≤ 6 within the `.egy` print quantum | external-engine, analytical |
| V-PICK-04 | Reference inputs are unchanged after any SPFIT or SPCAT run | software |
| V-PICK-05 | SPFIT/SPCAT on the HCO example: SPCAT reproduces `hco.cat` byte for byte; SPFIT's first iteration reproduces the last iteration of `hco.fit`. DPFIT/DPCAT on the NO example: listed differences recorded | external-engine |
| V-CMP-01..17 | One test per SF-CMP requirement, using genuine pinned engines on the §8.3 panel and convention, coverage and failure cases of amendment §3.5 | external-engine, software |
| V-FIT-01..13 | One test per fitting requirement; statistics checked against hand-computed values on small problems and against SPFIT's printed statistics | analytical, external-engine |
| V-FIT-08 | Linear additive and Ishigami indices reproduced within three bootstrap standard errors | analytical |
| V-CAL-01 | Coverage at k = 1, 2, 3 on lineage-held-out lines, with thresholds committed in advance | statistical, experimental |
| V-UNC-01..03, V-INV-01..02 | Uncertainty categories, absent states and claim rules enforced | software |
| V-ING-01..04, V-OBS-01..03, V-INT-01..03 | Observation schema, calibration flags, intensity registry and conversions (JPL ↔ HITRAN round trip for one species) | software, analytical |
| V-HAM-01..05 | Native terms match SPCAT within T1; label swaps halt; frame and constant-type rules | external-engine, analytical |
| V-LS-01..03 | Line-shape declarations; agreement with HAPI and analytic limits; chunked equals unchunked | analytical |
| V-VIB-01..04, V-NIR-01, V-ROVIB-01, V-PHASE-01, V-RAMAN-01..03, V-THZ-01 | Profile-specific checks listed in the capability matrix | per profile |
| V-HITL-01..03, V-UI-01..03 | Acceptance only by expert action; non-drag paths; stale marking; no older overwrite; views | software, demonstration |
| V-SET-01..04 | No prediction before temperature, window and J_max are set; editable in UI and API; edits mark results stale; values and their origin in provenance | software, demonstration |
| V-DATA-01..07 | Reference-record fields, redistribution class, fetch-and-verify, lineage partitions, extrapolation labels | software |
| V-OUT-01..06 | Coherent records, units, absent tokens, rounding reproduction check, provenance graph, SPFIT-first ordering | software |
| V-RUN-01..09, V-SEC-01 | Offline run, provenance, resource limits, outcomes, refinement versions, verified identity, use rights, builds only from manifest-verified JPL sources, subprocess safety | software |
| V-IF-01..08, V-API-01, V-PROV-01..03, V-ML-01..03 | Interfaces and providers, including direct input of ORCA and Pickett files, BASE as a required dependency and the GEOM payload | software |
| V-DEMO-1..5 | End-to-end demonstrations of the four workflows and of each capability-matrix row | demonstration, experimental |
| V-VER-01..05, V-PERF-01 | Evidence-class tags, no substitution, non-empty coverage, criteria committed first, improvement claims; performance workloads | inspection |

### 8.3 Benchmark manifest (panel accepted 2026-10-17; D-SPYC-14)

The first semirigid panel needs at least one linear, one symmetric-top and one asymmetric-top
closed-shell case, each compared only on transitions whose reference model has no hyperfine,
electron-spin, internal-rotation, inversion or interstate-coupling terms (R4-02). The PI accepted
this panel on 2026-10-17 (D-SPYC-14: "Yes, proceed."):

| Case | Rotor type | Reference records | Notes |
| :--- | :--- | :--- | :--- |
| CO | linear | CDMS 028503 and fit files; JPL 28001 | No hyperfine |
| OCS | linear | CDMS 060503; JPL 60001 | No published fit inputs: forward prediction only until a line list is compiled |
| SO₂ | asymmetric top | CDMS 064502 with A- and S-reduction fit files on the same data | Best reduction test |
| H₂CO | asymmetric top | CDMS 030501 and fit files; JPL 30004 | Exclude hyperfine-resolved lines |
| CH₃CCH (propyne) | symmetric top | CDMS 040502; ground-state fit files | The only semirigid symmetric top without hyperfine found |

HCl and CH₃CN belong to P2; NH₃, CH₃OH, CH₃CHO, dimethyl ether, ethanol, 1-propanol and H₂O to P4
or conformer work; CO₂ (no permanent dipole) to P6-P9; NO and OH to open-shell stress tests. The
calpgm HCO example is an I/O and provisioning regression (V-PICK-05), not semirigid evidence
(R4-06). The panel in D-SPYC-14 (propylene oxide, Ar-oxazole, H₂CO-HCl, HCO) and this panel
need one reconciled registry (R4-12). No data file from these sources is committed while its
redistribution class is "not redistributable" or "unknown" (SF-DATA-001d).

### 8.4 Release scenarios

Preregistered benchmark run; SPFIT/SPCAT provisioning acceptance (V-PICK-05); clean-wheel
install; UI journey on a reference case per deployment target.

---

## 9. Work packages

`docs/plan/CoChem-SpycFit_Plan.md` is authoritative for phases and work packages. Requirement
areas map to packages as follows: SF-IO-004, SF-PICK, SF-RUN-006..008 → WP-PICK; SF-ING, SF-IO-003,
SF-THZ, SF-RAMAN-001 → WP-EXP; SF-HAM, SF-PHYS-002..004 → WP-HAM; SF-FIT, SF-INF → WP-FIT;
SF-INT, SF-PHYS-007/008 → WP-INT; SF-CMP → WP-CMP (new); SF-HITL, SF-UI → WP-ASSIGN and WP-UI;
SF-IF → WP-IF; SF-SET → WP-UI; SF-OUT, SF-PUB → WP-OUT; SF-DATA, SF-IO-007 → WP-BENCH (new); vibrational profiles
→ WP-VIB (new).

---

## 10. Traceability

- Tests carry `@pytest.mark.req("SF-...")` and `@pytest.mark.evidence("<class>")`. The markers
  are registered with `--strict-markers` only in the Phase 2 hygiene PR (AGENTS.md).
- CI joins markers with JUnit results into a requirement → test → outcome matrix. A skipped,
  xfailed or absent test leaves its requirement unverified.
- Until the markers exist, test docstrings name the requirement ids they exercise. Tests added
  on 2026-10-16 for the amendment §5 defects name SF-CMP-012, SF-CMP-011, SF-CMP-013, SF-IO-004,
  SF-FIT-005, SF-OUT-001, SF-OUT-002 and SF-RUN-006. These are evidence for review, not a claim
  that any requirement is verified.

---

## 11. Decision register

`docs/decisions/D-SPYC-nn.md` files are the authoritative records; `docs/handoff/PI_DECISIONS.md`
lists every open question. The PI's 2026-10-16 decisions (`docs/HANDOFF.md`; amendment §7
D-01..D-03) bear on several entries. The PI confirmed them in conversation on 2026-10-16 and
answered further questions on 2026-10-17. The entries below are ANSWERED (records in
`docs/decisions/`) and become DECIDED when the PI approves the PR containing the records.

| Entry | PI answer | Remaining question |
| :--- | :--- | :--- |
| D-SPYC-01 Product identity, fitter of record | Broad assignment-and-fitting pipeline; native SpycFit and genuine SPCAT/SPFIT side by side; publication tables show SPFIT first (2026-10-16) | None |
| D-SPYC-06 IR/Raman/UV-Vis | IR, Raman, THz, near-IR and microwave first-class (2026-10-16); TORQ produces vibrational and Raman predictions (2026-10-17) | UV-Vis |
| D-SPYC-18 In-process model | A native model exists for compatible microwave models | Which terms, in which order |
| D-SPYC-04 Internal rotation | In scope after semirigid acceptance | Engines and order |
| D-SPYC-14 Benchmarks | Semirigid first (D-03); §8.3 panel accepted (2026-10-17) | Tolerance tiers and values; one registry with the earlier set (R4-12) |
| D-SPYC-03 Feedback to TORQ | PES refinement required, possibly in another module | Hand-off route |
| D-SPYC-17 SPFIT/SPCAT source | JPL sources (2026-10-17) | DPFIT/DPCAT; TORQ's pin; storage of verified copies; `pyckett`; use rights |
| D-SPYC-19 Dependencies | BASE is a hard requirement; TOPOS and TORQ are dependencies; ORCA and Pickett files are valid direct inputs (2026-10-17) | TOPOS/TORQ install-time or optional; ORCA quantities and parser |
| D-SPYC-20 Session parameters | Set before predictions; editable fields (2026-10-17) | Empty fields or per-technique presets, and their values |
| D-SPYC-10 Structure determination | Deferred to GEOM (2026-10-17) | GEOM payload schema |
| D-01, D-02, D-03 | Divergence review at export; offline-capable core; semirigid first | Recorded as D-SPYC entries |

Questions raised by this draft:

1. Confirm each 2026-10-16 direction above. Answered 2026-10-16 ("That is correct").
2. Keep the `SF-` id scheme (this draft) instead of the `SPYC-` scheme of Draft 0. Answered
   2026-10-17 ("Sure, go ahead."): the `SF-` scheme is kept.
3. Who owns vibrational and Raman predictions? Answered 2026-10-17: TORQ (D-SPYC-06).
4. Accept the semirigid panel of §8.3. Answered 2026-10-17 (D-SPYC-14); reconciliation with the
   earlier D-SPYC-14 set is open.
5. Pin SPFIT/SPCAT. Source answered 2026-10-17: JPL (D-SPYC-17). DPFIT/DPCAT is open.
6. Accept the four tolerance tiers of SF-CMP-008a and set their values. Open; the PI asked for
   suggestions from publications of the last five years.

## Appendices

**A. Disposition of review findings.** `docs/wiki/review/CoChem-SpycFit_SRS_2026-10-16_Literature_Review.md`
lists every R1-R5 finding with the requirement ids above that carry it, or the reason it was not
carried.

**B. Retired ids.** §0.6.

**C. Module map.** `docs/architecture/CoChem-SpycFit_Architecture.md` §2.
