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
> - **T3, fit quality against data:**
>   - A fit is flagged when its normalised error lies outside the 95% χ² interval for its degrees
>     of freedom (0.86–1.14 at 100 degrees of freedom).
>   - A line is flagged above 3u, and excluded only above 5u after you confirm.
>   - If a line has no uncertainty from its source or the user, the default is: cavity 2 kHz,
>     CP-FTMW 10 kHz, mm/sub-mm 50 kHz, synchrotron FTIR 2×10⁻⁴ cm⁻¹. Results using a default
>     are flagged.
> - **T4, prediction of held-out lines:** report coverage at 1–3σ. The minimum acceptable
>   coverage is fixed for each benchmark before it is run. No paper reports held-out coverage, so
>   there is nothing to copy.

Option A adopted these values for every tier; option B adopted the rules only, with the numbers
set for each benchmark from its CDMS fit file. The summary ended: "**To answer:** "D-SPYC-14
(tolerances): A for T1-T3, B for T4", or name a variant, for example "A, except T2 at 0.01 u"."

The PI replied: "D-SPYC-14 (tolerances): A for T1-T3, B for T4
Additionally ensure that SpycFit predictions are equal or closer than SPCAT and SPFIT."

#### Detail in the full brief that the summary did not show (needs PI confirmation)

- T1: whether "plus 1 Hz" is inside the 50 Hz and 5 Hz the summary gave or added to them (51 Hz,
  6 Hz); the IR and high-frequency print quanta (`.cat` 500 Hz above 99,999,999 MHz and
  5×10⁻⁷ cm⁻¹ in IR mode; `.fit` 5×10⁻⁸ cm⁻¹ for IR lines; `.egy` 5×10⁻⁷ cm⁻¹); transitions
  matched by state identity.
- T2: the full list of conditions that must be identical (model, data, weights, blend weighting,
  priors, ERRTST rejections, FRAC); comparison against SPFIT's `.var` values and unscaled u; the
  label `approximation` (SF-CMP-009); rms(Δν/u) reported beside Δp/u; SPFIT is rerun unless its
  last printed |Δp|/u ≤ 0.01; the native fit stops when σ_f changes by less than 10⁻⁷ relative and
  max |Δp|/u < 10⁻³. Review found that rerunning SPFIT from its rewritten `.par` re-centres finite
  priors, so the agent now proposes: rerun with a larger NITR if SPFIT stopped at NITR, or from
  the rewritten `.par` only when no parameter has a finite a priori error; otherwise the T2
  comparison is INCOMPLETE.
- T3: the per-data-set statistic √(Σ(r/u)²/N_k), a data set being source × instrument class (review
  found its degrees of freedom must be corrected for leverage, ν_k = N_k − Σ h_ii); calibration
  offsets declared per data set; exclusion only in a recorded scratch copy of the `.lin`, with
  ERRTST set very large in comparison runs; whether lines may still be excluded below 5u for a
  recorded physical reason (for example a blend with another species); further defaults of 4 kHz
  (cavity, unresolved hyperfine), 25 kHz (Ka-band CP) and 10⁻³ cm⁻¹ (wavemeter IR); defaults rank
  after the source's and the user's u, are Type B with k = 1, and no other technique gets one; the
  agent's proposal that defaults weight fits only while the observation record keeps the
  uncertainty absent (SF-ING-002c), and that a line with no uncertainty and no default stays out of
  the fit, recorded with the reason.
- T4: z = (ν_obs − ν_pred)/√(u_obs² + u_pred²) with SPCAT's ERR at k = 1; n, mean z and Var(z)
  with a bootstrap interval; floors from binomial bounds at a declared false-alarm rate (review
  found these ignore the correlation of predictions from one parameter set; the agent now proposes
  floors simulated under the joint predictive covariance); a mean-z limit; whether over-coverage
  should also be bounded; n < 30 gives INCOMPLETE unless coverage is already below its floor;
  extrapolated lines reported only.

#### The PI's additional instruction

"Additionally ensure that SpycFit predictions are equal or closer than SPCAT and SPFIT." The
agent reads this as: on every benchmark, SpycFit's native fit and predictions must be at least
as close to the measured data as SPFIT's fit and SPCAT's predictions from the same inputs, with
fits that agree within T2 counting as equal. This reading, including its extension from
predictions to fits, needs the PI's confirmation. Review found that a strict "no farther" test on
held-out lines fails in about half of runs from measurement noise alone, so the prediction test
needs a margin or a significance rule, which the PI has to set; and whether the instruction also
covers native models that differ from SPFIT's is open.

The agent also proposed (SRS v2 SF-CMP-021c) that native fits be offered for a profile only after
its benchmarks meet this test. That gate is not in the PI's words. It departs from the
2026-10-16 decision "Divergence remains visible. Exploration continues, but publication export
requires explicit expert review" (`docs/HANDOFF.md`; amendment D-01) and needs the PI's
confirmation.

## Consequences

- The first SPCAT/SPFIT comparison panel is semirigid: CO, OCS, SO₂, H₂CO and propyne
  (CH₃CCH), as listed in SRS v2 §8.3.
- The earlier D-SPYC-14 set (propylene oxide, Ar-oxazole, H₂CO-HCl, HCO) and this panel still
  need one reconciled registry (literature review R4-12); HCO stays an I/O and provisioning
  regression (V-PICK-05).
- Tolerances: T1, T2 and T3 take the values of option A as stated in the summary; T4 takes option
  B's rule, with the minimum coverage at k = 1, 2, 3 committed for each benchmark from its CDMS fit
  file before the benchmark runs (SF-VER-004). SRS v2 SF-CMP-008b to 008l carry them; values the
  PI did not see are marked to confirm or TBD there.
- The T3 default uncertainties of the summary are named defaults under AGENTS.md: they apply only
  when neither the source nor the user gives a line uncertainty and flag every result that uses
  them. SF-ING-002c is scoped to the observation record, consistent with the baseline (missing
  uncertainty "never converted into zero"; SRS SF-IO-002: not "an invisible drop").
- The PI's additional instruction is drafted, on the agent's reading, as SRS v2 SF-CMP-021a, 021b
  and 021d, with the agent's proposed gate SF-CMP-021c; all four are blocked on the PI's
  confirmation. A claim that SpycFit is closer is an improvement claim under SF-VER-005.
- Still open: confirmation of the full-brief detail above, of the reading of the additional
  instruction, of the prediction margin and of the gate; the search-window widths for computed constants (raised in the brief; they
  conflict with Method Matrix §3.1 and need their own decision); the numerical tolerances outside
  the four tiers (SF-PHYS-008c, SF-PHYS-008e, SF-FIT-008a, SF-RUN-002c).

## Alternatives not chosen

For tolerances: option B for T1-T3, option A for T4 (minimum coverage fixed per benchmark but not
set from its CDMS fit file), and the stricter or looser variants of option C were not chosen.
