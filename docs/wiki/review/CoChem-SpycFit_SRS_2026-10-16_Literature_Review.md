# Literature and practice review of the 2026-10-16 SpycFit SRS

| Field | Value |
| :--- | :--- |
| Reviewed | `docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md` and `docs/wiki/CoChem-SpycFit_SRS_2026-10-16_Amendment.md` at `488dc87` |
| Date | 2026-10-16 |
| Method | Five research passes, one per area (R1 Pickett conventions and the comparison contract; R2 fitting, uncertainty and identifiability; R3 vibrational, rovibrational, THz, NIR and Raman; R4 reference data, licences and the benchmark panel; R5 requirement quality), each checking the SRS text against primary sources, the pinned JPL calpgm source and runs of the genuine programs on scratch copies. |
| Result | `docs/srs/CoChem-SpycFit_SRS_v2.md` Draft 1 carries the findings as proposed requirements. This file records each finding's disposition and the evidence level. |
| Status | Review evidence for the PI. It approves nothing. |

## 1. Evidence levels

- **Checked directly in this session** (commands and outputs in `docs/evidence/phase0_inventory.md`
  or the commit `e87f6f4`):
  - SPFIT's `.fit` layout, RMS ERROR, MICROWAVE RMS, FRAC scaling and the `NEXT LINE NOT USED IN FIT`
    rule, read from the pinned `calfit.c` and confirmed on a genuine run of the HCO example.
  - SPFIT's first iteration on the shipped `hco.par`/`hco.lin` reproduces the last iteration of the
    shipped `hco.fit` exactly (all 76 residual rows and 21 parameter texts); SPCAT on the shipped
    `hco.var`/`hco.int` reproduces `hco.cat` byte for byte.
  - The JPL calpgm README says "the NO files are sample input for DPI". DPFIT built from the same
    sources reads every line of the hash-verified NO example and reaches MICROWAVE RMS 0.045678 MHz
    and RMS ERROR 1.15158, against 0.045684 MHz and 1.1534 in the shipped 1993 `no.fit`.
  - The Sobol total-effect defect: on the Ishigami function the code at `3f3c61a` returns
    S_T = (0.679, 0.549, 0.996) where the analytic values are (0.558, 0.442, 0.244), that is 1 − S_i.
- **Checked by a research pass** (scratch runs and source reading, not repeated here): the
  parameter sign table of SRS v2 §1.7.2, from SPCAT energies against an independent Watson
  Hamiltonian at J ≤ 6 (agreement to the `.egy` print limit, 4.9 × 10⁻⁷ cm⁻¹, and 0.27 cm⁻¹
  disagreement with the opposite d sign); the I^r behaviour of the prolate basis; forward
  agreement of an independent Watson model with SPCAT `.cat` to ≤ 49.5 Hz over about 110 lines.
  Acceptance test V-PICK-03 repeats the sign check.
- **Citations.** Every DOI or URL in SRS v2 §1.6 was resolved by a research pass. A pass marked a
  source "not read" when only its citation was checked, and UNVERIFIED when it could not confirm
  it. UNVERIFIED items are not relied on: the Albritton, Schmeltekopf and Zare 1976 chapter;
  Kisiel's correlation practice beyond the crib-sheet; Novick's SPFIT notes; the journal status of
  the 2025 mHT proposal; HITRAN, PNNL and AME2020 redistribution terms; a dataset for rotational
  Raman.

## 2. Main conclusions

1. **The comparison contract is sound but underspecified.** Amendment §3 states the right
   principles. It does not say what SPFIT's statistics mean (RMS ERROR divides by the number of
   fitted blends and includes prior terms; parameter errors depend on FRAC), how Pickett stores
   signs (200 holds −Δ_J; 40100 is −δ_J in A but +d_1 in S, contrary to JPL's `spins.nam` labels),
   that SPFIT's objective contains priors from ERPAR, or that a zero exit status does not mean
   success. SRS v2 §1.7.2, §1.7.4 and SF-CMP-012d, -016, -017 add these.
2. **No universal tolerance is right, but tiers are needed.** Four tiers make SF-CMP-008
   verifiable: forward implementation (half the print quantum of the compared output plus a
   numerical bound), fitted-parameter agreement in standard uncertainties, experimental adequacy
   from per-line uncertainties (about 5 kHz for CP-FTMW at 2-8 GHz; 5-50 kHz sub-millimetre;
   about 2.4 MHz far-IR [R-36]) and predictive coverage on held-out lines (SF-CMP-008a).
3. **"Experiment-only" must be decided from the engine input**, because in SPFIT an a priori
   error is a prior and a fixed computed constant contributes uncertainty that must be propagated
   (SF-FIT-012, SF-FIT-013, SF-INF-001a).
4. **Fit statistics, error scaling, rounding and calibration need definitions** consistent with
   the GUM and with Le Roy and Watson on rounding correlated parameters (SF-FIT-014..016,
   SF-OUT-005, SF-INF-003).
5. **The vibrational scope needs its physics stated.** A gas-phase band is a rotational contour,
   not a sum of peaks; intensity quantities differ between catalogs (JPL nm²·MHz at 300 K without
   isotopic weighting, HITRAN at 296 K with it); line shapes need HWHM/FWHM, normalisation and
   temperature laws; scale factors carry uncertainty and kinds; THz and NIR each split into
   gas-phase and condensed-phase work (SF-INT-001, SF-PHYS-008, SF-VIB-001/002, §1.2 matrix).
6. **The candidate panel mixes profiles.** Only CO, OCS, SO₂ and H₂CO are semirigid
   closed-shell cases; none is a semirigid symmetric top without hyperfine, so propyne is
   proposed. HCl and CH₃CN are hyperfine cases, NH₃, methanol, acetaldehyde, dimethyl ether,
   ethanol, 1-propanol and water are large-amplitude or conformer cases, and CO₂ has no rotational
   spectrum (SRS v2 §8.3).
7. **The reference sources forbid or do not grant redistribution** (NIST SRD; JPL; calpgm; CDMS
   unclear), so fixtures are fetched by URL and hash, never committed (SF-DATA-001d/e).
8. **Requirement form.** 65 of 73 ids were compound, 32 had no normative keyword and none had a
   verification method; nine duplicated an amendment id (SRS v2 §0.2-§0.6).

## 3. Disposition of findings

"Carried" names the SRS v2 requirement or section. Report numbers refer to the research notes
summarised here.

### R1. Pickett conventions and the comparison contract

| Finding | Carried |
| :--- | :--- |
| R1-01 Fitting and cross-prediction presume a native fitter | SF-CMP-020. The PI's 2026-10-16 direction (native beside SPFIT) settles that a native path exists; its terms remain D-SPYC-18. |
| R1-02 The NO example is DPFIT/DPCAT input | SF-RUN-007; handoff corrected; errata row 17; Phase 0 inventory §7 |
| R1-03 SPFIT statistics undefined | §1.7.4; SF-CMP-016; SF-FIT-014 |
| R1-04 Codes, signs, representation | §1.7.2; SF-PICK-001, SF-PICK-002; V-PICK-03 |
| R1-05 Exit status is not success | SF-CMP-012d |
| R1-06 Priors in SPFIT's objective | SF-CMP-017; SF-FIT-012 |
| R1-07 Intensities at the declared temperature | SF-CMP-003a; §1.7.3 |
| R1-08 State labelling and DIAG | SF-CMP-004a, SF-CMP-004b |
| R1-09 Tolerance tiers | SF-CMP-008a |
| R1-10 SPCAT truncations | SF-CMP-002a |
| R1-11 Executable identity and build drift | SF-CMP-011a; SF-RUN-006a |
| R1-12 Two meanings of `.cat` ERR | SF-CMP-018 |
| R1-13 Second reference engines | SF-CMP-019; §1.2 matrix |
| R1-14 Adapter field coverage | SF-IO-004d |
| R1-15 Undefined comparison wording | SF-CMP-014b; terms defined through SF-CMP-002, -016, -019 |
| R1-16 Use rights | SF-RUN-008 |

### R2. Fitting, uncertainty and computation-informed fits

| Finding | Carried |
| :--- | :--- |
| R2-01 Fit statistics | SF-FIT-014 |
| R2-02 Error scaling by σ_f | SF-FIT-015 |
| R2-03 Interval conventions | SF-FIT-016 |
| R2-04 Rounding correlated parameters | SF-OUT-005; SF-OUT-002c |
| R2-05 Conditioning beyond pairwise correlation | SF-FIT-004b |
| R2-06 Structural versus practical identifiability | SF-FIT-004c |
| R2-07 Sobol estimator and use | SF-FIT-005a-c; code corrected in `e87f6f4` |
| R2-08 Evidence for freezing parameters | SF-FIT-011 |
| R2-09 Experiment-only from engine input | SF-FIT-012; SF-INF-001a |
| R2-10 Fixed-parameter uncertainty; prior sensitivity | SF-FIT-013; SF-INF-001c |
| R2-11 GUM-consistent categories | SF-INF-002a, SF-INF-002c |
| R2-12 Predictive calibration | SF-INF-003 |
| R2-13 Double counting, exclusions, robust weights | SF-FIT-003c; SF-FIT-017 |
| R2-14 Nonlinear intervals | SF-FIT-018 |
| R2-15 ML calibration, domain, abstention | SF-ML-001a-c; SF-ML-003b |

### R3. Vibrational, rovibrational, THz, NIR and Raman

| Finding | Carried |
| :--- | :--- |
| R3-01 Scope decision and prediction ownership unrecorded | SRS v2 §11 open questions 1 and 3; `docs/handoff/PI_DECISIONS.md` D-SPYC-06 |
| R3-02 Gas-phase bands are contours | SF-VIB-001 |
| R3-03 THz and NIR each mix two kinds of work | §1.2 profiles P5, P11, P12 |
| R3-04 PGOPHER has a command-line program | `PI_DECISIONS.md` D-SPYC-01 option C evidence |
| R3-05 Line-shape contract | SF-PHYS-008a-c |
| R3-06 Intensity and abscissa conventions | SF-INT-001a-c; §1.7.3 |
| R3-07 Envelope identifiability | SF-VIB-002 |
| R3-08 Scale factors and resonance treatment | SF-PHYS-003c, SF-PHYS-003d |
| R3-09 Joint MW and IR fits | SF-FIT-009 |
| R3-10 Raman conventions | SF-RAMAN-001..003 |
| R3-11 THz-TDS observations | SF-THZ-001 |
| R3-12 Overtone models | SF-NIR-001 |
| R3-13 Rovibrational references | SF-ROVIB-001 |
| R3-14 Condensed-phase models | SF-PHASE-001 |
| R3-15 Untestable wording | SF-SCOPE-002 (capability matrix); SF-SCOPE-003c |
| R3-16 Duplicate ids | §0.6 |

### R4. Reference data, licences and the panel

| Finding | Carried |
| :--- | :--- |
| R4-01 CDMS and JPL publish fit inputs | SF-DATA-001c |
| R4-02 Panel mixes profiles | §8.3 |
| R4-03 Redistribution | SF-DATA-001d, SF-DATA-001e; `tests/pickett_reference.py` fetches nothing into the repository |
| R4-04 Version fields unreliable | SF-DATA-001a, SF-DATA-001b |
| R4-05 Published catalogs merge measured lines | SF-DATA-002 |
| R4-06 HCO is not semirigid evidence | §8.3; `tests/pickett_reference.py` docstring |
| R4-07 NO example is DPFIT/DPCAT | As R1-02 |
| R4-08 Extrapolation domain | SF-IO-007c |
| R4-09 Lineage and temporal holdouts | SF-IO-007b, SF-IO-007d |
| R4-10 Masses and constants editions | §1.7.5; SF-IO-005 |
| R4-11 Legacy fixture generator disagrees with JPL | SF-DATA-003; Phase 0 inventory §5 item 7 |
| R4-12 Two candidate panels | §8.3; §11 open question 4 |
| R4-13 Vibrational sources per observable | SF-DATA-004 |

### R5. Requirement quality

| Finding | Carried |
| :--- | :--- |
| Modal verbs undefined | §0.2 |
| Compound requirements | Every id split into single-SHALL parts with letter suffixes |
| No verification method or acceptance | §0.3 form; §8.2 acceptance catalog |
| Nine duplicate ids | §0.6 retired-id table |
| Three status vocabularies | §0.4 |
| Missing glossary, conventions, interfaces, data records, subprocess security | §1.5, §1.7, §4.3-§4.4, §7, SF-SEC-001 |
| Traceability | §10 |
| Baseline status and decision records inconsistent | `AGENTS.md`, `docs/handoff/AGENT_HANDOFF.md`, `PI_DECISIONS.md` updated to name the `488dc87` baseline; recording decisions remains the PI's step |
| Id scheme | SRS v2 keeps `SF-` ids; §11 open question 2 |

Not carried: the INCOSE rule-by-rule annotations of the R5 table (the rewrites carry their
effect), and suggestions that depend on UNVERIFIED sources.

## 4. Corrections to earlier agent statements

- `docs/handoff/AGENT_HANDOFF.md` and `REVIEW_ERRATA.md` row 9 said the shipped NO example "does
  not run" with SPFIT/SPCAT. It is DPFIT/DPCAT input (JPL README); DPFIT runs it (§1). The handoff
  is corrected; the errata file adds row 17, since only the PI changes an existing row.
- HCO was described as an exact regression case. It is, for file handling and provisioning; it
  is an open-shell radical and is not evidence for the semirigid profile (R4-06).
