# Pending-evidence register

Every provisional default, tolerance or threshold in use that still awaits a test is listed here
with the value in use, the test to run and the decision it serves (SRS v2 SF-VER-006). The PI asked
for this register on 2026-10-17: "Ensure a testing to be completed and published is flagged in a
file." An entry leaves this file only with the recorded result of its test, in
`docs/evidence/`, and the PI's revised value in the decision record.

| Id | Provisional value in use | Decision | Test to run | Where the result will be recorded | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| PT-01 | Swap test: largest-overlap map not the identity for a state an assigned line uses, two-state crossover 0.5 | D-SPYC-07 (07c), SRS v2 SF-HAM-002b | Analytical two-state crossing with known overlaps; HCO (JPL example, scratch copies) stepped across block 4; SO₂ and H₂CO §8.3 panel inputs stepped through ±3u in ten steps per fitted constant, counting swaps and near ties | `docs/evidence/V-HAM-03_label_swaps.md` | pending; PI instruction 2026-10-17: "use 1 as a provisional value that should be adjusted with further evidence" |
| PT-02 | Near-tie margin: largest overlap below 0.9 flags a near tie | D-SPYC-07 (07c), SRS v2 SF-HAM-002b | As PT-01; the margin is set from the distribution of largest overlaps on the stepped panel fits (HCO's block-4 pair at 0.796 is the known mixed case) | `docs/evidence/V-HAM-03_label_swaps.md` | pending |
