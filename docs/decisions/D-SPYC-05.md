# D-SPYC-05 User interface and hosting

| Field | Value |
| :--- | :--- |
| Status | ANSWERED in part (agent): hosting and the HPC journey are answered; the toolkit waits for research the PI asked for. DECIDED only after the PI approves the PR that contains this record. |
| Answered | 2026-10-17 |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | conversation 2026-10-17, pop-up questions (`docs/handoff/pi_answers_2026-10-17_popups.md`); brief `docs/handoff/briefs/D-SPYC-05_2026-10-17.md` |

## Decision

The agent asked three questions in one pop-up.

1. "D-SPYC-05 (1 of 3) — which user-interface stack should SRS v2 require for assignment
   interaction, given that BASE owns 'the unified GUI'?" with options C (Jupyter-widgets browser
   UI over the headless API, plot widget named after a measured comparison), A (ipywidgets +
   Plotly under Voila as a BASE tab) and D (desktop Qt), B typeable. The PI typed: "What is the
   best and most user friendly option? Research and come up with a low friction and best
   usability option suggestions."
2. "D-SPYC-05 (2 of 3) — how does the SpycFit UI appear inside BASE's unified GUI?" The PI
   selected: "H2: BASE's Voila renders a SpycFit notebook in a SpycFit-environment kernel; one
   port (Recommended)".
3. "D-SPYC-05 (3 of 3) — does the first release include a UI journey on HPC (browser via SSH
   tunnel or Open OnDemand; Qt via X11/VNC)?" The PI selected: "No: HPC is headless (API and CLI)
   in the first release (Recommended)".

## Consequences

- Hosting: the SpycFit UI is a notebook that BASE's Voila renders in a SpycFit-environment kernel
  on BASE's single forwarded port (SRS v2 SF-UI-006). BASE must register the notebook path and a
  kernelspec for the SpycFit environment; that is a BASE change the PI requests. Cross-environment
  widget front ends under Voila are unverified; if they fail, the fallbacks are H1 (SpycFit's own
  Voila server on a second port) or H3 (a BASE-written tab over a SpycFit adapter).
- HPC: the first release provides the API and CLI on HPC and requires no UI journey there (SRS v2
  SF-UI-007; plan Phase 4 and architecture §5 updated).
- Toolkit: open. The PI asked for research on the lowest-friction, best-usability option. The
  brief's comparison protocol (drag-to-refit latency, redraw time on a full spectrum, keyboard
  journey, offline load, accessible table) is the starting point; the PI's statement under
  D-SPYC-04 that SPYFY should "make fitting and calculations dynamic recalculate on the fly" makes
  the widget-to-fitter latency a design constraint.
- SF-API-001 (one headless API) and SF-HITL-002b (non-drag path) hold for any toolkit.

## Alternatives not chosen

For hosting, H1 and H3 were offered and not selected. For HPC, a UI journey as a release
requirement was not selected. For the toolkit, no option was selected; A, C and D remain open
until the research is presented.
