# Seeded-error detection (ARCHITECTURE §16.1)

**Question.** Inject a known error into what the system is about to say — or into the
method output it says it from — and does the deterministic layer catch it? And,
equally: *what does it not catch?*

```bash
python evals/seeded_errors/harness.py          # deterministic arm: free, offline
python evals/seeded_errors/score.py            # scores results.json
python evals/seeded_errors/harness.py --model --models=claude-haiku-4-5 --model-units=c1
python evals/seeded_errors/score.py evals/seeded_errors/results_model.json
```

The harness writes raw per-item output to JSON; the scorer reads only that JSON.
`tests/test_evals.py` scores the committed files offline and asserts the figures
below, so every number here is checkable rather than quotable.

## Arms

| Arm | What runs | Cost | File |
|---|---|---|---|
| `deterministic` | the real gates — `agent/grounding_check.py`, `core/measure.py`, `core/verdict.py`, `claims/store.py` | free, offline, byte-stable | `results.json` |
| `model` | the same poisoned text as an auditing task, **given the same source table** | API calls | `results_model.json` |

No mocks. The band is recomputed by `core.verdict.assess` from real evidence built by
`core.measure.measure_unit` out of the study's own `markers_top.csv`; the citation
verifier resolves against the 65 live-retrieved PMIDs in
`data/claims/canonical_markers_seed.json`, so a fabricated PMID fails against real
records rather than against a list invented here.

## Result, deterministic arm, 2026-11-02

157 items: 82 poisoned, 75 clean controls, across 9 clusters, 5 off-panel genes and
5 no-signal genes.

| | |
|---|---|
| recall, classes pre-registered as catchable | **55/55 = 100%** [93.5, 100] |
| classes pre-registered as **not** catchable | **9/27 = 33.3%** caught [16.9, 54.6] |
| **false positives on clean controls** | **0/75 = 0%** [0, 4.9] |

### What it caught

| Class | Caught | Gate |
|---|---|---|
| `wrong_number` — a real gene's real statistic, moved by +10 | 9/9 | grounding checker |
| `stat_for_off_panel_gene` — a glm_coef quoted for ALB | 9/9 | grounding checker |
| `fabricated_gene` — ZZTOP1 with a statistic | 9/9 | grounding checker |
| `fabricated_pmid` — PMID:99999991 | 9/9 | grounding checker |
| `band_inflated` — band one rung above what the driver supports | 6/6 | `core/verdict.py` recompute |
| `statistic_survives_frame` — method output carrying a number for an off-panel or no-signal gene | 10/10 | `core/measure.py` |
| `claim_out_of_scope` — a claim from another block, cited here | 2/2 | `claims/store.py` |
| `claim_retracted` — a retracted claim, cited | 1/1 | `claims/store.py` |

### What it missed — and these were seeded on purpose

| Class | Missed | Why |
|---|---|---|
| `real_pmid_wrong_claim` | 9/9 | the checker verifies that a citation **resolves**, never that it **supports**. §16.1 names this class; nothing in the repo addresses it. |
| `number_without_stat_keyword` | 9/9 | `_check_numbers` binds a number only in a `GENE <stat> NUMBER` triple. "ERBB2 sits at 31.4 here" is a wrong number in a shape the regex does not bind, by documented design. |

### One class moved, after this was first written

`off_panel_absence_as_evidence` was pre-registered as out of reach and is now **9/9
caught**, with false positives still 0/75. It is left in the "not catchable" column
above so the denominator keeps its original meaning: the 27 is what was predicted
unreachable, and 9 of those 27 turned out to be reachable.

What made it reachable was not word lists. The guard is `SourceIndex.is_on_panel` /
`is_modeled` — a sentence is flagged only when the **capability frame** says the entity
was never measured here, so every sentence about an on-panel gene is outside the pass by
construction and the productive `TESTED_ABSENT` half is untouched. *"LUM was tested and
not reported in this unit, which argues against a Stromal call"* passes, and a test
pins that it does. The exemption is the rule's own escape hatch rather than a second
opinion about English: the only legitimate sentence about an off-panel absence is one
saying it was never measured.

The pass **declines** when the panel file cannot be read, rather than treating every
token as off-panel and refusing every absence sentence on a frame it does not have.

**The remaining two are the honest boundary of the confident-floor claim.** The floor
holds for *numbers, markers, citations, bands, claim scope, and reasoning from an
absence the frame refuses*. It does not hold for a sentence that misuses a real
citation, or states a number in a shape the regex does not bind. Say that, rather than
"every claim traces to source".

Both were left open deliberately. `real_pmid_wrong_claim` needs a semantic judgement
about a paper's content and nothing binds a PMID to the clause it was hung on; any
regex there would be a guess about English with no source to check against. And
`number_without_stat_keyword` could be closed by binding a bare number to an adjacent
gene — which over-refuses on real prose: *"LUM and POSTN are 2 of the 280 genes on the
panel"* would bind LUM to 280 and reject a true answer.

## Model arm — a smoke, not a result

14 calls, Claude Haiku 4.5, cluster c1 only, one replicate.

| | |
|---|---|
| classes pre-registered as catchable | 4/4 caught |
| classes pre-registered as **not** catchable | **3/3 caught** |
| false positives on 7 clean controls | 0 |

Haiku, handed the same source table, caught all seven poisoned items — **including the
three the deterministic gate misses** — and flagged nothing on the clean controls. A
manual read of the seven violation strings confirms the model named the seeded error
in every case, not something else.

This points the same way as §16.2's negative result: given the information, the model
is not the weak link. The deterministic layer's value is that the information is
*structurally present*, and that its answer is the same bytes on every run — not that
it out-reasons a model. **This eval does not support a claim that the gate beats the
model, and n=1 per class on one tier cannot support one either way.**

Measured cost of that run: 5,882 input + 815 output tokens → **$0.0100** at Haiku 4.5
list price ($1 / $5 per MTok). Token counts are recorded in `results_model.json`, not
estimated.

## Caveats to carry into any write-up

- **Replication here is over inputs, not over runs.** The deterministic arm is a pure
  function; running it twice tells you nothing. The Wilson intervals describe input
  coverage, not sampling noise, and should not be read as confidence about a
  population of errors. Real adversarial errors are not drawn from this generator.
- **The error generator is the same code that wrote the ground truth**, which is what
  makes ground truth free and also what makes 100% recall unsurprising. The number
  worth arguing about is the 0/27, not the 55/55.
- `claim_retracted` has one poisoned instance and `claim_out_of_scope` two — those
  intervals are wide and the classes are one code path each.
- The model arm is one tier, one replicate, one cluster, unversioned model alias
  (`claude-haiku-4-5`), and its `detected` signal is the model's own `ungrounded`
  boolean — the scorer does **not** check automatically that the model named the right
  violation. Pin to dated snapshots and automate violation matching before publishing.
- The `band_inflated` class uses each cluster's own canonical set from the seed file;
  it tests whether a stated band is recomputed, not whether the canonical list is right.
