# D-SPYC-04 Internal rotation and tunnelling

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent): engines, reference build and BASE provisioning are answered; wrap-versus-native waits for research the PI asked for. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-16 (direction); 2026-10-17 (engines, build, provisioning; research request) |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-16 (agent question and PI reply quoted below); confirmed direction first recorded in `docs/HANDOFF.md` and amendment §7 on `main` at `488dc87`; pop-up answers of 2026-10-17 in `docs/handoff/pi_answers_2026-10-17_popups.md`; brief `docs/handoff/briefs/D-SPYC-04_2026-10-17.md` |

## Decision

The agent asked: "1. Confirm your October 16 decisions by id, for example "D-SPYC-06: IR, Raman, THz, NIR and MW are first-class". `PI_DECISIONS.md` quotes them under D-SPYC-01, 03, 04, 06, 14 and 18, but none is marked answered yet."

The PI replied: "1. That is correct."

The confirmed direction (`docs/HANDOFF.md`; see also amendment §7 D-03):

> "Establish independent semirigid comparison first; then separately validate hyperfine, excited-state, and large-amplitude-motion extensions. All remain in scope."

On 2026-10-17 the agent asked four questions in one pop-up.

1. "D-SPYC-04 (1 of 4) — internal rotation in profile P4: wrap external engines, build a native
   model, defer, or add only SPCAT-reproducible native terms?" with options A (wrap; the wrapped
   engine's fit is the P4 result of record), D (native terms SPCAT reproduces), B (full native
   model) and C (defer). The PI typed: "The main value component of SPYFY is that it can utilize
   GPU to parallel, array, and make fitting and calculations dynamic recalculate on the fly.
   Honestly I don't see how we could be useful just copying SPFIT and SPCAT. I need more research
   and analysis for this on what would give the better end result and fit the vision for CoChem"
2. "D-SPYC-04 (2 of 4) — which engines does SF-CMP-019 name for P4?" (multi-select). The PI
   selected all four: "XIAM first (Recommended)", "ERHAM second (Recommended)", "BELGI-Cs/C1 and
   RAM36 as later", "westerfit as later".
3. "D-SPYC-04 (3 of 4) — which XIAM variant is the reference build?" The PI selected: "XIAMALL.FOR
   build (the XIAM5 source)".
4. "D-SPYC-04 (4 of 4) — should BASE's build mechanism (task 11) also build the P4 engines, which
   needs gfortran (and LAPACK for RAM36) in BASE's image? That is a BASE change only you can
   request." The PI selected: "Yes: XIAM (and ERHAM when its terms allow) in the BASE build
   mechanism (Recommended)".

## Consequences

- Internal rotation and tunnelling are in scope and come after semirigid acceptance (SRS v2
  §1.2 profile P4).
- Engines named for P4 (SRS v2 SF-CMP-019): XIAM first, ERHAM second; BELGI-Cs/C1, RAM36 and
  westerfit listed as later engines. Only provisioned and accepted engines appear as available;
  the rest are listed as later.
- The XIAM reference build is the XIAMALL.FOR source (the source XIAM5 is built from), not the
  distributed XIAM 2.5e archive the agent recommended. XIAM's terms say that everybody can use and
  copy xiam for free and that a changed source must be marked, documented, renamed and reported;
  XIAMALL.FOR is the author's own variant, not an agent change, and its own terms have not been
  read. Its rights record (SF-RUN-008) and its acceptance case (the NH₃⋯HCOOH benchmark was fitted
  with XIAM5) are open items. ERHAM's terms were not found; rights stay PI-determined.
- BASE's build mechanism (task 11, the BASE-BUILD design) builds XIAM, and ERHAM when its terms
  allow; BASE's image needs gfortran. This is a BASE change the PI requests; the agent asks the PI
  to confirm the wording when the BASE build design is presented.
- The legacy defaults V3 = 0 and F = 160000 MHz in `cochem_jax_builder.py` become typed errors;
  V3, F and ρ come from the user or the direct-input path because TORQ defers large-amplitude
  motion.
- Still open: whether P4 is served by wrapped engines (A), a native model (B), SPCAT-reproducible
  native terms (D) or a combination. The PI asked for research and analysis on what gives the
  better end result and fits the CoChem vision, stating that SPYFY's main value is GPU-parallel,
  array-based fitting that recalculates on the fly. That research (R1 in
  `docs/handoff/pi_answers_2026-10-17_popups.md`) also bears on D-SPYC-11 (toolchain), D-SPYC-18
  (native model terms) and SRS v2 §1.1, and must state the conflict with the Method Matrix's note
  that JAX and the GPU are "not used to replace SPFIT" (`CoChem-BASE/Method_Matrix.md`, JAX note).

## Alternatives not chosen

None ruled out beyond the ordering. On 2026-10-17: XIAM 2.5e unchanged source (the agent's
recommendation) and the posted Windows XIAM5.EXE were not selected as the reference build;
user-provisioned P4 engines (no BASE build) were not selected.
