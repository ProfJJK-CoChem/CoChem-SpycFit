# D-SPYC-14 Benchmark set, data supply and tolerances

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent). DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (panel; tolerances) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversations 2026-10-16 and 2026-10-17 (agent questions and PI replies quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`; see also amendment §7 D-03):

> "Establish independent semirigid comparison first; then separately validate hyperfine, excited-state, and large-amplitude-motion extensions. All remain in scope."

On 2026-10-17 the agent asked, as item 2: "**D-SPYC-14** first semirigid comparison panel
(proposed: CO, OCS, SO₂, H₂CO, propyne)". The PI replied: "2. Yes, proceed."

As item 3 the agent asked: "**D-SPYC-14** tolerance tiers and their starting values". The PI
replied: "3. Research what limits are considered valid in publications from the last 5 years and
give me suggestions."

### Tolerances (2026-10-17)

The agent sent a summary in the conversation, pointing to the full brief (committed as
`docs/handoff/briefs/D-SPYC-14_tolerances_2026-10-17.md`). The summary's suggested values were:

> - **T1, SpycFit vs SPCAT predictions:** |Δν| ≤ half the last printed digit of the compared
>   file, plus 1 Hz. That is 50 Hz for `.cat` and 5 Hz for `.fit`.
> - **T2, SpycFit vs SPFIT parameters:** each parameter within 0.1 u. Only for identical model,
>   data and weights, and reported as a numerical check, not a statistical test.
> - **T3, fit quality against data:** A fit is flagged when its normalised error lies outside the
>   95% χ² interval for its degrees of freedom (0.86–1.14 at 100 degrees of freedom). A line is
>   flagged above 3u, and excluded only above 5u after you confirm. If a line has no uncertainty
>   from its source or the user, the default is: cavity 2 kHz, CP-FTMW 10 kHz, mm/sub-mm 50 kHz,
>   synchrotron FTIR 2×10⁻⁴ cm⁻¹. Results using a default are flagged.
> - **T4, prediction of held-out lines:** report coverage at 1–3σ. The minimum acceptable
>   coverage is fixed for each benchmark before it is run.

Option A adopted these values for every tier; option B adopted the rules only, with the numbers
set for each benchmark from its CDMS fit file. The summary ended: "**To answer:** "D-SPYC-14
(tolerances): A for T1-T3, B for T4", or name a variant, for example "A, except T2 at 0.01 u"."

The PI replied: "D-SPYC-14 (tolerances): A for T1-T3, B for T4
Additionally ensure that SpycFit predictions are equal or closer than SPCAT and SPFIT."

#### Detail in the full brief that the summary did not show (needs PI confirmation)

- T1: the IR and high-frequency print quanta (`.cat` 500 Hz above 99,999,999 MHz and
  5×10⁻⁷ cm⁻¹ in IR mode; `.fit` 5×10⁻⁸ cm⁻¹ for IR lines; `.egy` 5×10⁻⁷ cm⁻¹); transitions
  matched by state identity.
- T2: the full list of conditions that must be identical (model, data, weights, blend weighting,
  priors, ERRTST rejections, FRAC); comparison against SPFIT's `.var` values and unscaled u;
  SPFIT is rerun unless its last printed |Δp|/u ≤ 0.01; the native fit stops when σ_f changes by
  less than 10⁻⁷ relative and max |Δp|/u < 10⁻³; rms(Δν/u) is reported beside Δp/u.
- T3: the per-data-set statistic √(Σ(r/u)²/N_k), a data set being source × instrument class;
  calibration offsets declared per data set; exclusion only in a recorded scratch copy of the
  `.lin`, with ERRTST set very large in comparison runs; further defaults of 4 kHz (cavity,
  unresolved hyperfine), 25 kHz (Ka-band CP) and 10⁻³ cm⁻¹ (wavemeter IR); defaults rank after
  the source's and the user's u, are Type B with k = 1, and no other technique gets one.
- T4: z = (ν_obs − ν_pred)/√(u_obs² + u_pred²) with SPCAT's ERR at k = 1; n, mean z and Var(z)
  with a bootstrap interval; floors from binomial bounds at a declared false-alarm rate; n < 30
  gives INCOMPLETE; extrapolated lines reported only.

#### The PI's additional instruction

"Additionally ensure that SpycFit predictions are equal or closer than SPCAT and SPFIT." The
agent reads this as: on every benchmark, SpycFit's native fit and predictions must be at least
as close to the measured data as SPFIT's fit and SPCAT's predictions from the same inputs, with
agreement within T1 and T2 counting as equal. This reading needs the PI's confirmation.

## Consequences

- The first SPCAT/SPFIT comparison panel is semirigid: CO, OCS, SO₂, H₂CO and propyne
  (CH₃CCH), as listed in SRS v2 §8.3.
- The earlier D-SPYC-14 set (propylene oxide, Ar-oxazole, H₂CO-HCl, HCO) and this panel still
  need one reconciled registry (literature review R4-12); HCO stays an I/O and provisioning
  regression (V-PICK-05).
- Tolerances: T1, T2 and T3 take the values of option A; T4 takes option B's rule, with its
  coverage floors and mean-z limit committed for each benchmark from its CDMS fit file before the
  benchmark runs (SF-VER-004). SRS v2 SF-CMP-008b to 008i carry them.
- The T3 default uncertainties are named defaults under AGENTS.md: they apply only when neither
  the source nor the user gives a line uncertainty, are recorded as Type B defaults with k = 1 and
  flag every result that uses them. The observation record still keeps the missing uncertainty
  as absent (SRS v2 SF-ING-002c); the default is a weight used by the fit and the T3 evaluation.
- The PI's additional instruction becomes SRS v2 SF-CMP-021a to 021c (proposed): native fits and
  predictions no farther from the measured data than SPFIT's and SPCAT's on each benchmark, a
  shortfall reported as DIVERGENT, and the native path enabled for a profile only when every
  benchmark of the profile meets it. A claim that SpycFit is closer is an improvement claim under
  SF-VER-005.
- Still open: confirmation of the full-brief detail above and of the reading of the additional
  instruction; the search-window widths for computed constants (raised in the brief; they
  conflict with Method Matrix §3.1 and need their own decision); the numerical tolerances outside
  the four tiers (SF-PHYS-008c, SF-PHYS-008e, SF-FIT-008a, SF-RUN-002c).

## Alternatives not chosen

For tolerances: option B for T1-T3, option A's fixed T4 floors, and the stricter or looser
variants of option C were not chosen.
