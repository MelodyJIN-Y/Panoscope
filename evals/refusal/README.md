# Refusal reported next to accuracy (CLAUDE.md; ARCHITECTURE §16.4)

> "Report refusal rate next to accuracy. A gate that fails closed can look safe while
> refusing." — CLAUDE.md

Before this eval, that instruction was the only occurrence of the word "refusal rate"
in the repository. Nothing measured it.

```bash
python evals/refusal/harness.py                # gated arm: free, offline
python evals/refusal/score.py                  # scores results.json
python evals/refusal/harness.py --model --models=claude-haiku-4-5
python evals/refusal/score.py evals/refusal/results_model.json
```

The harness writes raw per-turn output to JSON; the scorer reads only that JSON.
`tests/test_evals.py` scores the committed files offline and asserts the figures below.

## The turn set

23 turns about the demo study, each labelled before the run with whether the system
*should* answer. No mocks: every turn is routed through a real code path.

| Class | Expect | Route |
|---|---|---|
| `observed_gene`, `weak_but_observed_gene` | answer | `core/measure.py` |
| `off_panel_gene` (ALB, UMOD, NPHS2, INS, MYH7) | refuse | `core/measure.py` → `NEVER_MEASURED` |
| `nosig_gene` (5 of the 49 no-signal panel genes) | refuse | `core/measure.py` → `TESTED_ABSENT` |
| `single_replicate_condition` | refuse | `workflows/de.py::replicate_confound` |
| `panel_scoping_with_canonical` | answer | `workflows/de.py::panel_scoping` |
| `panel_scoping_no_canonical` | refuse | `workflows/de.py::panel_scoping` |
| `claim_in_scope` | answer | `claims/store.py::cite` |
| `claim_out_of_scope`, `claim_retracted` | refuse | `claims/store.py::cite` |

## Result, gated arm, 2026-11-02

| | |
|---|---|
| answered | **8/23 = 34.8%** [18.8, 55.1] |
| accuracy given answered | 8/8 = 100% [67.6, 100] |
| accuracy with refusal counted wrong | 8/8 = 100% [67.6, 100] |
| **false refusal** (could have answered, did not) | **0/8 = 0%** [0, 32.4] |
| **over-answer** (should have refused, answered) | **0/15 = 0%** [0, 20.4] |

Every refusal names what is missing. The single-replicate turn returns
*"CANNOT BE SETTLED — biological replicates 1/1 (floor 2)"* with the missing items
listed; the off-panel turns name the frame the subject is outside of.

### The numbers in this table that mean almost nothing

**Accuracy given answered is 1.00 by construction and should not be quoted on its
own.** The gated arm reads the value out of the same file the ground truth comes from,
so it cannot be wrong when it answers; the only thing it can do is refuse. The
informative columns are the coverage (34.8%) and the two honesty numbers.

**False refusal is 0/8 on eight hand-written answerable turns.** That is the weakest
number on this page and the one most likely to be flattering: the turn set was written
by the same author as the gate, so the answerable turns are ones the gate was built to
answer. A false-refusal rate is only worth something against turns a biologist wrote.
Treat 0/8 as "no false refusal was constructed", not as "the gate does not over-refuse".

**One class refuses by construction and carries no information.**
`replicates_sufficient_table_silent` asks the replicate question with three
replicates; `replicate_confound` still returns `CANNOT_BE_SETTLED`, because a
FindAllMarkers table reports one number per gene for the whole unit and which
replicate drives it is not in the file. That is a deliberate design decision, argued
in `workflows/de.py`, and it means this check's refusal rate is 100% whatever you feed
it. It is counted as an expected refusal here and flagged rather than hidden — a check
that cannot return OK is exactly the shape CLAUDE.md's warning is about.

**No risk–coverage curve.** §16.4 asks for one. A curve needs a tunable threshold and
this gate has none — it is a capability lattice, not a score. One point is reported
(coverage 34.8%, risk among answered 0%) and the curve is left undone rather than
faked by sweeping a parameter that does not exist.

## Model arm — a smoke, not a result

15 calls, Claude Haiku 4.5, one replicate, on the `marker_stat` turns only (where an
answer and a refusal are cleanly distinguishable). The model got the same source table
and the instruction *"if the information needed to answer is not available to you,
refuse rather than estimate"*.

| | gated | Haiku 4.5, 1 rep |
|---|---|---|
| answered | 8/23 | 5/15 |
| false refusal | 0/8 | 0/5 |
| over-answer | 0/15 | 0/10 |

**The ungated model matched the gate exactly on this turn set.** It refused all ten
off-panel and no-signal genes and answered all five observed ones correctly. That is a
negative result for any claim that the gate prevents fabrication a model would commit —
and it is the expected result given §16.2: with the table in the prompt, the off-panel
genes are simply not in it, so refusing is easy. **A turn set that would separate the
two would have to withhold the table, or ask something the table does not settle.**
This one does not, and it should be rebuilt before it is cited.

Measured cost: 4,966 input + 691 output tokens → **$0.0084** at Haiku 4.5 list price
($1 / $5 per MTok). Token counts are recorded in `results_model.json`, not estimated.

## Caveats to carry into any write-up

- 23 turns, hand-written, by the author of the gate. Small and not adversarial.
- One model tier, one replicate, unversioned alias (`claude-haiku-4-5`). Pin before
  publishing.
- The model arm covers `marker_stat` turns only — the DE and claim routes have no
  model counterpart, so the comparison is narrower than the table suggests.
- Accuracy on the non-numeric answerable turns is a substring match against the
  expected content (a named never-measured gene, a claim's object). That is a weak
  grader and would not survive free-form model output.
