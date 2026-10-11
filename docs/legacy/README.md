# Historical material

Everything in this directory is kept for context and traceability. None of it is a requirement,
a design decision or evidence that a capability works. The current entry point is
`docs/handoff/AGENT_HANDOFF.md`.

| File | What it is | Status |
| :--- | :--- | :--- |
| `2026-08-07_spycfit_architectural_changes.md` | The PI-era plan that pivoted SpycFit from IR/Raman convolution to CP-FTMW Bayesian fitting with SPFIT handoff, LOIO cross-validation and next-best-transition suggestions. Restored from commit `21f6259` (`.old_plan_docs/`, deleted in `3e86ec1`). | Superseded. Useful for intent. Its "J up to 80, enormous matrices" premise and the penalty on negative distortion constants are wrong (review §3.4, CRITIC-05, PHYS-17). |
| `2026-08-07_spycfit_workflow.md` | The matching five-stage workflow (environment lock, prior assembly, interactive fitting, LOIO, Pickett export). Restored from the same commit. | Superseded. The predict, assign, fit, re-predict loop it describes is still the right shape. |
| `copilot-instructions_2026-08.md` | The swarm agent roster (CoChem-AUDIT, -TEST, -SCRIBE, -CODER, -SPEED) formerly at `.github/copilot-instructions.md`. | Superseded by `AGENTS.md`. Several rules in it ("output all 2,000 lines", "no mocking" with no evidence classes, "never alter scientific logic") caused the failures documented in the review (§2, GOV-20). |

Other historical artifacts remain where they were and are listed in the review §7.1: the swarm
SRS (`docs/wiki/CoChem-SpycFit_Comprehensive_SRS.md`), the copied CoChem-BASE Task 1
specification (`.docs/task1_subsystems_architectural_specification.md`), `swarm_state.json`
and the `*_payload.md` files. They will be removed or archived in the Phase 2 hygiene PR once
the PI approves.

Not in the repository: the prompt files that drove the old code
(`Perfected_SRS_Chunk_NN_prompts.md`, on the PI's machine). If the PI supplies them, store
them under `docs/legacy/prompts/`.
