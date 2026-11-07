# CLAUDE.md

Project: Panoscope — a human-in-the-loop interpretation layer for bioinformatics results.

Context for building this project with Claude Code. Read this before writing code.
Keep it tight. Every line here loads on every turn, so only load-bearing rules live here.

**Design of record:** `ARCHITECTURE.md` (what and why) and `IMPLEMENTATION.md` (contracts, migration).
Supporting: `docs/interface-design.md`, `docs/llm-access.md`. Superseded documents are in
`docs/archive/` — do not build from them; two contain examples that are actively wrong.

## What this is

A biologist has a result produced by someone else. Panoscope reads it faithfully, says what it supports
and what it does not, and lets the biologist argue with it. It **does not run the analysis** — method
output is a consumed input, never recomputed, which makes the confidence assessment an audit rather
than a self-assessment. The biologist makes the final call.

## Core principle (never violate)

**Confident floor.** Every marker, number and citation in an answer traces to method output, a
capability frame, or a stored claim. Never invent one.

**Open ceiling.** Literature and context reasoning is uncertain. Label it a direction, not a fact.

**The rule about reference data.** Reference data determines **which question is asked of the
measurement**. The measurement alone determines **the answer**. Reference data may select and order
evidence, generate rivals, supply canonical lists and carry citations; it may never supply a value,
count as agreement, or corroborate a measurement.

**What the knowledge apparatus buys.** The band is a function of one method's output; every other layer
can only subtract or flag. What the apparatus buys is the flags, the named rivals, the undecidables and
the blocked-with-reason — not the number. Say so plainly rather than implying otherwise.

## Structure: everything extensible is a registry

A **workflow** owns an interpretive question and its review surface. A **method** registers into a
workflow and declares its own statistics. Platform adapters and reference sources register the same way.

```
skills/<workflow>/SKILL.md        the interpretive question + its review surface
workflows/<workflow>.py           the workflow module
methods/<method>.py               a declaration; each names the workflow it joins

marker-interpretation      what IS this unit?            jazzpanda
enrichment-interpretation  what is RUNNING here?         jazzpanda_cgst
de-interpretation          does this result MAKE SENSE?  edger, seurat_de
```

A method module is **a machine-readable declaration of what a method's paper says it measures and does
not measure**. Adding a method is a declaration, not interpretation logic. `does_not_measure` and
`assumptions` are mandatory.

## Measurement status (the headline rule, generalised)

A negative observation is evidence only if the measurement was **capable** of producing a positive one.
Four states, never two:

| State | Absence is evidence? |
| --- | --- |
| `NEVER_MEASURED` — not in the capability frame | **no** |
| `UNDERPOWERED` — in frame, below the floor | **no** |
| `TESTED_ABSENT` — in frame, tested, not reported | **yes** — this is the productive half |
| `OBSERVED` | — |

One capability frame per entity kind: the panel for genes, the annotation label set for cell types, the
design's replicate counts for conditions, detectability for genes that produced no usable signal.
Composite subjects take the **meet** over required roles and name what blocked them.

`statistic is None` **iff** the state is not `OBSERVED`, so quoting a number for an unmeasured thing is
structurally unrepresentable. `UNDERPOWERED` withholds the number too: in frame but below the floor is a
state the measurement could not support a value from. Derived from `Measured.carries_number`, never from
a re-listed set of states — the second list is the one nobody updates.

## Hard rules for the agent

- Never fabricate. If it is not in source, do not state it.
- **Panel-absence rule.** The absence of an out-of-frame entity is not evidence against anything. Check
  the frame before down-weighting. Say when something was never measured.
- **But measured absence *is* evidence.** A marker that was measured and peaked elsewhere argues against
  a rival. Do not collapse the four states into "unknown".
- Cite everything. Every claim names its source.
- **Citations must be real, looked up from a real source at build or top-up, never from memory.**
  Never write a PMID or DOI from memory. A fabricated citation is the worst possible failure, worse than
  no citation. If a lookup returns nothing, say the literature is thin.
- When evidence is insufficient, set `verify = TRUE` and say re-check this. **`verify` comes from the
  deterministic rubric — never offer the model the choice.**
- No overclaiming. The tool accumulates the lab's knowledge; it does not learn on its own.

## Task kinds

Three, frozen. **Task kind belongs to the method, not to the use.**

| Kind | Question | Band? |
| --- | --- | --- |
| `ASSIGNMENT` | what *is* this? | yes — there is a label space |
| `CHARACTERISATION` | what is running here? | **no** |
| `COMPARISON` | does this result make sense? | **no** |

No label space means a unit-level band is *undefined*, not merely unhelpful. Header for those two is
per-item counts: *4 enriched, 3 untestable, top EMT z 4.2 on 18/200 panel genes*.

`unit_kind`, subject shape and method are open registries; only `TaskKind` is frozen.

## Memory: a reconciliation layer, not a memory of the user

Memory is where the biologist's judgment and the literature get reconciled into something the lab owns.

- **The value is in the disagreement.** When the biologist overrides, use the call AND cross-check the
  literature, reporting agreement and dissent with real citations, keeping the call with the tension
  visible. Never a bare "got it", never a silent overrule.
- **Capture at override**, asking scope and basis in two taps, not a form.
- **The biologist decides, always.**

Claims are one object type across measurement, lab, curated and literature, ordered by authority and
**never merged**. Scope is enforced fail-closed. The agent may propose; only the biologist commits. No
background consolidation, no LLM dedupe, no recency-based resolution.

## Study intake

The step that populates `Conditions` — every applicability check reads that object. Each field carries
provenance (`stated` / `read` / `inferred` / `unknown`); `unknown` is first-class and propagates.

Blocks the study: the measured set, clustering mode when multi-sample, tissue, platform.
Declared-unknown allowed: fixation, segmentation.

Interaction: read everything derivable from files first, then show a **checklist** the biologist ticks,
then chat only about ticked items and blocking gaps. Stop early.

## Interface

The review surface is **part of the interpretation contract**, declared by the workflow — not a UI
concern. See `docs/interface-design.md`.

- **Every plot carries its conditionality.** A volcano on a targeted panel is indistinguishable from a
  whole-transcriptome one unless the figure says so. A plot with no conditionality record refuses to draw.
- **Plots render, they do not derive.** No plot computes a statistic not already in the evidence.
- **Measurement status is visible**, not implied. A never-measured item is not a blank row and not a zero.
- **Click-back.** Every sentence resolves to its claim, with authority and conditions visible. A sentence
  with nothing to click should not have been rendered.
- Viewing controls (bin size, view toggle, sample switcher) change the picture, **never a value**.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,app,agent,export]"
pytest -m "not live"                             # grounding tests, must pass before every commit
panoscope run                                    # the app
panoscope build --study <id>                     # (re)build a study tree (deterministic)
panoscope explain <unit>                         # render the interpretation record
panoscope doctor                                 # resolved paths, studies, versions, credentials
```

## Grounding tests (the guarantee, not an afterthought)

`tests/` parses each agent answer and checks that every marker, number, claim and citation traces back
to source. The suite fails if the agent states anything not in source, uses a claim without citing it,
uses one outside its scope, or produces a citation that does not resolve.

**The golden-output test is the most valuable artifact in the repo**: it makes refactoring safe, and it
is the reproducibility claim. Keep it byte-stable.

Report refusal rate next to accuracy. A gate that fails closed can look safe while refusing.

## Data

- Demo: 10x Xenium FFPE human breast cancer, sample1 only. Analyzed panel = **280 genes**; panel size is
  **dynamic across studies — derive it from the file, never assert a constant.** 9 clusters.
  `glm_coef` (spatial GLM) is the primary statistic; `pearson` corroborates. Bands live in config and
  are tuned against the calibration set.
- Second study: human kidney, **immuno-oncology panel**, multiple samples, **joint clustering**. The
  panel was designed for a different question than the tissue, so kidney-resident types have almost no
  canonical markers on-panel — the panel-absence rule at maximum severity.
- **49 of 280 demo panel genes (17.5%) yield no usable spatial signal.** Detectability is a capability
  frame, not a footnote.

## Compliance

- New work only; the upstream method is a published dependency, not vendored.
- MIT. Everything shown must be open source. Fetch restrictively-licensed data, never bundle it.
- Precomputed method output only. Never run the analysis live.
- Deterministic fallback for every agent call, so a slow or failed request never breaks the demo.
- **Never use a consumer AI subscription programmatically.** Prohibited by every major provider's terms.
  See `docs/llm-access.md`.

## What not to do

- Do not invent markers, stats or confidence values.
- Do not treat out-of-frame absence as evidence against anything — **or** treat measured absence as
  uninformative.
- Do not let reference data supply a value or corroborate a measurement.
- Do not apply a claim without citing it, or outside its scope.
- Do not put a unit-level band on characterisation or comparison.
- Do not let a viewing control change a value, or a plot derive a statistic.
- Do not let the model decide `verify`.
- Do not claim the tool learns on its own.
- Do not report "% conditional" as a result — it is a measured constant.
- **Refused by decision:** a novelty verdict ("is this finding new?"); SVG interpretation; consumer
  subscription auth. Reasons are in `ARCHITECTURE.md` §15.
