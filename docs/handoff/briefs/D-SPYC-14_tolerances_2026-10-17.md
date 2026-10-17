### Question

D-SPYC-14 (tolerances): what starting value or rule should each SF-CMP-008a tier carry?
- T1: native vs SPCAT forward agreement.
- T2: native vs SPFIT parameters, in standard uncertainties u.
- T3: experimental adequacy.
- T4: held-out coverage.

Also: what search windows do computed constants justify? Evidence: six literature sweeps, each read a second time. Values taken only from abstracts, values checked by only one reader, and values published before 17 Oct 2021 are excluded.

### What recent papers treat as valid

| Technique (papers) | Line u | Fit rms | Dimensionless σ | Rejection (\|r\|/u) |
|---|---|---|---|---|
| Jet CP-FTMW 2-40 GHz (14) | ~10 kHz below 18 GHz (Gougoula 2022, Cabezas 2024, Sun 2026); 25 kHz in Ka band (Khemissi 2026 only) | 0.6-15 kHz CP-only, mostly 3-11 (Tsoi 2022, Steber 2025); 24-26 kHz with mm-wave lines (Sun 2026) | none reported | none stated |
| Jet cavity FTMW (10) | 2-4 kHz; 0.5 kHz best (Preston 2026) | 0.2-6 kHz, rms/u 0.1-2.3, usually ~1; 0.6-27 kHz per isotopologue (Changala 2023) | 1.083, mixed global fit (Preston 2026) | none stated |
| mm/sub-mm (17) | 2-5 kHz best, 10-50 typical, 100-200 weak (Müller 2022-2024); 30-60 kHz, weak 100 (Lille, MPE) | 9-71 kHz | 0.71-1.00 overall (13 papers); groups, mostly RAM36 torsional states, 0.36-1.34 | >5 (Bonah 2024); >10, removing 110 lines beside 269 kept in one state (Bonah 2025) |
| FTIR, mostly synchrotron (9) | 0.5-2.6×10⁻⁴ cm⁻¹ | 0.2-3.4×10⁻⁴ cm⁻¹ | 0.75-1.3, joint fits only (Tamassia 2022, Martin-Drumel 2023, Bonah 2026) | ≥3 plotted, not a rule (4 Woods/McMahon papers); >5 excluded, rotational lines (Bonah 2026) |
| Laser IR (4) | ~10⁻³ cm⁻¹ (wavemeter); 5-50 kHz (comb QCL; Tran 2025, preprint) | 2-4×10⁻⁴ cm⁻¹ (Silva 2024, Bast 2025); 8 kHz (Tran 2025, preprint) | none for IR lines | split lines omitted |

No paper states a pass threshold. A fit passes when its rms is close to the measurement accuracy or σ ≈ 1, and authors judge that against alternative fits:
- Müller 2022 published its σ = 1.162 fit, which was ">20% worse" than other reductions.
- Bonah 2025 (1.48-4.22) and Bunn 2025 (4.5, still used for an ALMA detection) kept fits that were "not at experimental accuracy".
- A fit that mixes line classes takes on the scatter of its poorer class (Khemissi 2026). One overall rms therefore misleads (Müller 2023).

**Cross-program agreement**
- **Same model.** Duo and PGOPHER differ by 40-50 Hz, which is 4-5 units of the last printed digit. A reduced model moved MgH energies by 42 MHz (Qu 2022).
- **Same Watson data.** WANG reproduces rounded published SPFIT parameters within 0.1 u. On new data SPFIT reached "a slightly less favourable minimum" (Albert 2025).
- **Internal-rotation programs.** B = 932.06085(19) (XIAM) vs 932.3075(47) MHz (BELGI-Cs) (Baweja 2023).
- **Convergence.** RAM36 stops when three conditions hold: the weighted rms changes by ~10⁻⁷ relative, corrections are <10⁻³ of their confidence intervals, and frequencies change by <1 kHz (Ilyushin 2023). SPFIT stops when its predicted RMS ERROR improves by ≤10⁻⁶ relative, or at NITR. It keeps the parameters from the start of that iteration (`calfit.c`).

**Computed vs experimental rotational constants** (Be vs B0 unless ΔBvib is named)
- **Hybrid DFT/MP2:** errors up to 2.1% (B3LYP-D3(BJ)/def2-TZVP) and 1.4% (MP2/aug-cc-pVTZ) (Sun 2026). Over 174 molecules, rRMSD is 2.2-3.1% (Tikhonov 2024).
- **Double hybrid:** maximum 0.75-0.99% (Barone & Lazzari 2023; Wenzel 2025); 1.3% with ΔBvib (Alessandrini 2026; Barone et al. 2023).
- **Composite or CCSD(T) + ΔBvib:** maximum 0.20% for semirigid molecules (Barone & Lazzari 2023). Pyridazine is off by 0.64% (CCSD(T)/cc-pCVTZ from Owen 2021, compared in Esselman 2024). One C5H2 isomer is off by >3% in Be (Thimmakondu & Karton 2023).
- **Complexes:** +2.3% (CCSD/6-311++G(2d,p); Macario 2022), +1.9% (B3LYP-D3(BJ)/def2-TZVP; Juanes 2022), −5.0% (B3LYP-D3BJ/6-311++G(2d,2p); Gilbert 2025). HCl(H₂O)n has a 7.9% rRMSD (PBE0-D3(BJ)/def2-TZVP; Tikhonov 2024). Which axis is worst varies.
- **Distortion:** quartic constants mostly within 10-20%, but ΔJK is off by −32% (pyridazine) and a small δK by a factor of 4 (Wenzel 2025). Sextic constants are off by −59% to +114% (pyridazine).

### Suggested values

**T1 (global).** |Δν| ≤ half the print quantum of the compared output + 1 Hz, with transitions matched by state identity.
- Half print quanta: `.cat` 50 Hz (500 Hz above 99,999,999 MHz; 5×10⁻⁷ cm⁻¹ in IR mode). `.fit` 5 Hz (5×10⁻⁸ cm⁻¹ for IR lines). `.egy` 5×10⁻⁷ cm⁻¹.
- The 1 Hz term and the `.fit` bound are SpycFit choices with no literature support.
- Evidence so far is the E4 pilot run (49.5 Hz). V-PICK-03 and V-HAM will supply the committed evidence.

**T2 (global).** Applies only when the model, data, weights, blend weighting, priors, ERRTST rejections and FRAC are all identical (SF-CMP-017). Never applied across Hamiltonians.
- Rule: |Δp| ≤ 0.1 u for every free parameter. Compare against SPFIT's `.var` values and unscaled u (FRAC=1, or divide out the printed factor).
- SPFIT's 10⁻⁶ stopping rule limits the rms change, not Δp. So SPFIT's last printed |Δp|/u must be ≤ 0.01; otherwise rerun SPFIT (SpycFit choice).
- The native fit stops when σ_f changes by <10⁻⁷ relative and max |Δp|/u is <10⁻³.
- Both fits use the same data, so Δp/u is a numerical scale, not a statistical test. Label it `approximation` (SF-CMP-009). Also report rms(Δν/u), because Δp/u is unreliable when |ρ|→1.

**T3 (global rule, default u per technique).**
- Per data set: √(Σ(r/u)²/N_k). A data set is source × instrument class, fixed before evaluation. Overall: σ_f = √(χ²/(N−M)) (SF-FIT-014), reported beside SPFIT's RMS ERROR (denominator N).
- Flag values outside the central 95% χ²_ν interval (SF-FIT-015), for example 0.41-1.60 at ν=5 and 0.86-1.14 at ν=100. The literature gives no fixed bound. Low values are informational only; accepted fits go as low as 0.36.
- Flag lines with |r|/u > 3; about 0.27% of lines exceed this by chance.
- Exclude a line only above 5, only after the user confirms, and only in a recorded scratch copy of the `.lin`. Comparison runs set ERRTST very large (SF-CMP-012d).
- Declare calibration offsets per data set.
- Default u values apply only after a PI record names them. They rank after the source's u and the user's u. Each is Type B with k=1 assumed, and any result that uses one is flagged.
  - Cavity 2 kHz (4 kHz with unresolved hyperfine; one paper).
  - CP-FTMW 10 kHz. Ka-band CP 25 kHz (one paper).
  - mm/sub-mm 50 kHz.
  - Synchrotron FTIR 2×10⁻⁴ cm⁻¹. Wavemeter IR 10⁻³ cm⁻¹.
  - Other techniques get no default: absent with a reason, never borrowed from another row.

**T4 (global rule, numbers per benchmark).**
- z = (ν_obs−ν_pred)/√(u_obs²+u_pred²). u_pred is the ERR that SPCAT computes from `.var`, at k=1, with its scaling declared (SF-CMP-018a). It is not a catalogue's 95% error.
- Report what SF-INF-003 requires: n, mean z, Var(z) with a bootstrap interval, and coverage at k=1-3 by extrapolation label. Use lineage partitions plus a temporal holdout.
- Set floors from binomial bounds at a declared false-alarm rate. Floors of 0.90 at k=2, 0.98 at k=3 and |mean z| ≤ 0.5 would fail about 12% of perfectly calibrated runs at n=30 (simulated).
- n < 30 gives INCOMPLETE. Extrapolated lines are reported only. No paper reports held-out coverage.

**Search windows (not a tier; needs its own decision line, not D-SPYC-20).** Half-widths q, applied as [B_pred/(1+q), B_pred/(1−q)]. They are labelled uncalibrated and never used as acceptance tolerances.
- Conflict: REVIEW_ERRATA item 4 lets windows default to the Method Matrix §3.1 bands: B₀ 0.3-0.5% semirigid, 1-2% floppy, quartic factor 2, and no double-hybrid row. The values below differ.
- Rule (SpycFit choice): q = 2 × rRMSD where a set statistic exists, otherwise 1.5 × the largest error seen.
- Hybrid DFT: B3LYP-D3(BJ)/def2-TZVP 4.5%, PBE0-D3(BJ)/def2-TZVP 5.1%, r2SCAN-3c 5.4%, B3LYP-D3(BJ)/def2-SVP 5.5%, PBEh-3c 6.2%. MP2 2.1% (one paper).
- Double hybrid 1.5%, or 2% with ΔBvib. Composite or CCSD(T) + ΔBvib 1%.
- Complexes 7.5%. Multi-water clusters 16% (one functional).
- Quartic constants [D/2, 2D], or a factor of 4 for constants below ~10% of their largest sibling. Sextic constants: order of magnitude only.

### Options

**A. Adopt the suggested values for every tier.**
- Pros: testing can start now. Every value has a source or is labelled a SpycFit choice.
- Cons: T2 rests on one comparison limited by print precision, and T4 on none.
- Risks: an inflated u can hide a poor fit. If the T4 floors are wrong, tests must be rewritten.

**B. Adopt the rules only, and set the numbers for each §8.3 benchmark from its CDMS fit file.**
- Pros: the numbers match each reference's own u. SpycFit claims no pass bar that the literature lacks.
- Cons: slower. Each benchmark needs a PI record. OCS has no fit inputs, so T2-T4 cannot be evaluated for it.
- Risks: the numbers may be chosen after seeing results unless they are committed before the run (SF-VER-004).

**C. Stricter or looser values.**
- Stricter: T1 against `.fit` only, T2 at 0.01 u, T3 at the 90% interval.
  - Pro: catches subtle convention errors.
  - Con and risk: false flags where correlated parameters converge slowly.
- Looser: T2 at 1 u, T3 at the 99.9% interval, T4 coverage ≥ 0.80 at k=2.
  - Pro: fewer flags.
  - Con and risk: a missing term, or a wrong sign on a small constant, goes unnoticed.

### Agent recommendation

A for T1 to T3; B for T4.
- T1 is deterministic.
- T2 rests on one published comparison.
- T3 uses the χ²_ν interval that SF-FIT-015 already requires.
- Commit the T4 floors for each benchmark before running it (SF-VER-004).
- T3 flags lines and fits but never rejects them. T3 alone never makes a fit accepted (SF-FIT-007).
- Each profile carries its own versioned values, and WITHIN_TOLERANCE names its tier (SF-CMP-008a).

### How to answer

"D-SPYC-14 (tolerances): A for T1-T3, B for T4." To choose a variant, name the change, for example "D-SPYC-14 (tolerances): A, except T2 at 0.01 u."
