# Panel-absence ablation (item 0.5)

**Question.** Does a model misstate the measurement conditions of a targeted-panel
assay, and does supplying the panel fix it?

Run `harness_v2.py` (the current design), score with `score_v2.py`.
`harness.py` / `score.py` are v1, kept because their failure is instructive.

## Result, 2026-09-22

405/405 calls parsed. Three Anthropic tiers (Opus 5, Sonnet 5, Haiku 4.5), 9 clusters,
3 arms, 5 replicates.

| Arm | Calls containing a false or unwarranted claim about the assay |
|---|---|
| bare (no panel) | **94.8%** [89.7, 97.5] |
| panel (280 genes in prompt) | **1.5%** [0.4, 5.2] |
| panel + rule | **0.0%** [0.0, 2.8] |

Per individual claim, the bare arm asserts measurement status it cannot know ~60% of
the time, consistently across tiers (Haiku 56.5%, Sonnet 59.1%, Opus 61.1%) — with
`cannot_tell` explicitly offered and one word away. Given the panel, 2 errors in 1,502
claims, both the inverse direction; zero fabricated absences.

## What this does and does not support

**Does not support** the hypothesis that motivated it. Supplying the panel essentially
solves this failure, so a deterministic gate is not justified by *this* mode. That is
a negative result, found before review rather than during it.

**Does support** a narrower, better-evidenced claim: the value is not out-reasoning
the model, it is *guaranteeing the measurement conditions are present and correct*.
That is a systems property. A prompt has it only when someone remembers; the
architecture has it structurally.

The bare-arm figure stands on its own as a measured fabrication rate in a domain where
fabrication is essentially never measured — and it is the condition most published LLM
annotation tools operate in, since none of them place the panel in the prompt. Whether
that is true of CASSIA / GPTCelltype / mLLMCelltype is the obvious follow-up; if so,
94.8% is a statement about the field rather than about models.

## Why v1 is kept

v1 scored "does an off-panel *canonical* marker appear in `evidence_against`" and
returned 0/216 across every arm including `bare`. The metric was too narrow: models do
not reason about a hand-picked canonical list, they reason about the genes they
themselves raise. A flat zero in the control arm should be read as a broken metric,
not a clean system.

v1's `unmeasured` field also conflated "not in this table" with "not measured in this
assay", which produced a 50% → 25% figure that did not survive a properly specified
task. v2 separates them and offers abstention, which is what makes the result
defensible.

## Caveats to carry into any write-up

- Three model *tiers*, one family — only `ANTHROPIC_API_KEY` was available. A
  cross-family claim needs OpenAI and Google keys.
- 9 clusters. Wide enough for a 60-point effect, not a subtle one. The panel-masking
  generalisation in ARCHITECTURE.md §16.2 is what scales this to hundreds of items.
- Models are not pinned to dated snapshots here. Pin them before any published run.
