# Decision records

One file per PI decision, named `D-SPYC-nn.md`; these files are the authoritative records.
Open questions, options and recommendations live in `docs/handoff/PI_DECISIONS.md` until the PI
answers. An agent drafts a record only from the PI's verbatim words, with date and source, and
marks it ANSWERED. It becomes DECIDED only when the PI approves the PR that contains it. Never
invent, paraphrase or relay an approval.

Template:

```markdown
# D-SPYC-nn <title>

| Field | Value |
| :--- | :--- |
| Status | ANSWERED (agent) / DECIDED (only after the PI approves this PR) |
| Answered | YYYY-MM-DD |
| Answered by | PI (Dr. Joshua John Klaassen) |
| Source | PR / issue link, or "conversation YYYY-MM-DD" |

## Decision
<the PI's answer, quoted where possible>

## Consequences
<requirements, documents and code affected; follow-up work>

## Alternatives not chosen
<options from PI_DECISIONS.md and why they were not chosen, if the PI said>
```
