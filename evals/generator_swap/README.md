# Generator swap and deterministic invariance (ARCHITECTURE §16.3)

**Question.** Swap the model that writes the prose. Do the band, the basis, `verify`, the
demotion ids, the driver and the measurement states move?

**The claim this eval makes, and the one it does not.** §16.2 retracted the framing that
the gate out-reasons the model, and `evals/seeded_errors` found a model catching three
error classes the gate misses. Nothing here argues the model does not matter. The claim
is narrow: *the deterministic fields do not move.* Where one does, that is the finding.

```bash
python evals/generator_swap/harness.py          # swap arm: free, offline
python evals/generator_swap/score.py            # scores results.json
python evals/generator_swap/harness.py --model --models=claude-haiku-4-5,claude-sonnet-5
python evals/generator_swap/score.py evals/generator_swap/results_model.json
```

The harness writes raw per-field output to JSON; the scorer reads only that JSON.
`tests/test_evals.py` scores the committed files offline and asserts the figures below.

## The two kinds of field, which is the whole result

| Kind | Fields | Why |
|---|---|---|
| `pure` | `cell_type`, `band`, `band_basis`, `demotions`, `driver`, `verify_record`, `measured_states` | computed by `pipeline/v1_verdicts.py::verdict_for` and `core/measure.py::measure`, whose arguments are a cluster id and the study files. No prose, no model, no history. |
| `seam` | `verify_response` | **Closed since 2026-11-04.** `agent/loop.py::_assemble` used to set `AgentResponse.verify` to `_text_asks_verify(text) or _cluster_verify_default(cluster)`, and `_text_asks_verify` was a **substring test over the model's prose**. It now calls `_verify_default(cluster, skill)`, which does not take prose as an argument. The field is kept so the seam cannot reopen unnoticed. |

The seam was found by reading `agent/loop.py:1246`, not by running anything. The harness
is what turns it into a number and keeps it from quietly closing or quietly widening.

**It has since closed.** `_text_asks_verify` was deleted and `_assemble` now calls
`_verify_default(cluster, skill)`. Re-scoring the `seam` column of the SAME 45 stored
observations with the current code gives **0/45**: the computation does not take prose,
so no generator can move it. No model was re-run to produce that number — the stored
prose is simply fed to the current function, which ignores it. The pre-fix measurement
below is kept as the record of what was there, not as a description of the code today.

## Result, swap arm, 2026-11-02 — the PRE-FIX measurement

Five prose styles standing in for five generators, over 9 clusters and 8 fields = 360
field observations. Every fact in the prose is read from the record at build time, so the
styles differ **in voice only** — anything that moves, moved on style alone.

| | |
|---|---|
| `pure` fields that moved | **0/315 = 0%** [0, 1.2] |
| `seam` fields that moved (pre-fix) | **16/45 = 35.6%** [23.2, 50.2] |
| `seam` fields that moved (same observations, current code) | **0/45** |

Per generator, on the seam:

| style | moved | what it wrote |
|---|---|---|
| `terse`, `verbose`, `clinical` | 0/9 each | plain statement of the record |
| `hedged` | **8/9** | "…though I would **re-check this** before acting on it." |
| `flagging` | **8/9** | "… **verify=TRUE**." |

8 and not 9 because c9 (Mast_Cells) already carries `verify=True` from the rubric, so
there is nothing for the seam to move.

### What that means, stated plainly

**`verify` on the object the biologist sees WAS a function of model prose.** CLAUDE.md:
*"`verify` comes from the deterministic rubric — never offer the model the choice."*
`agent/loop.py:1246` offers the choice. Two qualifications that matter:

- **It only moves `False → True`.** The seam is an `or`, so a generator can add a
  re-check flag and never remove one. The failure is conservative, not dangerous.
- **`hedged` is the worrying one, not `flagging`.** A model writing `verify=TRUE` is
  agreeing with a rubric decision. A model writing *"I would re-check this before acting
  on it"* is being polite, and the substring test cannot tell the two apart. Ordinary
  English trips it.

`core/verdict.py`'s own `verify` is not reachable this way — `tests/test_core_verdict.py`
`::test_verify_is_not_an_input_anywhere_in_the_module` pins that, and this eval confirms
it from the outside at 0/45. The seam is in the chat assembly layer only, which is why a
test scoped to `core/` never saw it.

## Model arm — 18 calls, 2 tiers, 2026-11-02

Claude Haiku 4.5 and Claude Sonnet 5 each wrote the opening paragraph for all 9 clusters
from the same deterministic record, and the same extraction ran over what they wrote.

| | swap (5 styles) | Haiku 4.5 | Sonnet 5 |
|---|---|---|---|
| `pure` fields moved | 0/315 | 0/63 | 0/63 |
| `seam` moved | 16/45 | **0/9** | **0/9** |

**Neither real generator tripped the seam.** That is a genuine result and it points the
opposite way from the swap arm: the seam is reachable by ordinary English, and on 18
openings neither tier wrote any. It does not close the seam — one differently-prompted
generator, or one asked to hedge, lands in the `hedged` row — but it does mean the
observed cross-model variance on today's prompt is zero.

**The `pure` 0/126 here is true by construction and should not be quoted on its own.**
The model never touches those functions; they take a cluster id. The number is included
because "no seam" is a claim about today's code, and re-measuring it is how it stays true
— not because it was hard won.

Measured cost: Haiku 2,086 in + 1,455 out = **$0.0094**; Sonnet 5 2,572 in + 2,499 out =
**$0.0301**; **$0.0395** over 18 calls, at list price ($1/$5 and $2/$10 per MTok). Token
counts are recorded in `results_model.json`, not estimated.

## What this eval does NOT show

- **It is not an ablation across model families.** §16.3 asks for three or four families
  including one open-weight. This is two tiers of one family. The "byte-identical across
  model families" claim in §16.3 is **not** supported by what has been run here, and
  nothing in this directory should be cited for it.
- **It does not test the full chat path.** The model arm prompts a generator directly
  with the record. It does not run `agent/loop.py`'s tool loop, so tool-call variation,
  repair rounds and fallback are all out of frame. A generator that behaves differently
  in the loop would not show up.
- **It says nothing about prose quality.** Only whether seven fields moved.

## Caveats to carry into any write-up

- **The swap arm's five styles are hand-written by the author of the seam.** They were
  chosen to include two that trip it. The 35.6% is therefore a statement about the
  constructed set, not a rate anyone will encounter; the useful content is *which* styles
  trip it and why, not the fraction.
- Two tiers, one family, one replicate, unversioned aliases (`claude-haiku-4-5`,
  `claude-sonnet-5`). Pin to dated snapshots before publishing.
- 9 clusters, one study, one panel. The second (kidney) study is not exercised here.
- The Wilson intervals on the swap arm describe input coverage, not sampling noise — the
  arm is a pure function and running it twice tells you nothing.
