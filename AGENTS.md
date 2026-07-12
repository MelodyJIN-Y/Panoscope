# AGENTS.md: how Panoscope was built and how it reasons

Panoscope was built with Claude Code used as an **orchestrated team of agents**, not a single chat.
This file names the roles so the design is legible. It draws one honest line:

> **The team of agents was the build methodology. The running product is deliberately simple:
> one grounding-gated agent loop over a deterministic engine, with specialized skill-driven stages.**
> We do not claim a swarm of live agents at runtime; we claim disciplined division of labor, enforced.

---

## Reviewing this project

Every claim here is runnable, and the fastest read is to run it. From a fresh clone (no API key needed):

- `pytest -m "not live"`: 230+ tests; a green run **is** the confident floor (no answer states anything not traced to source).
- The load-bearing evidence lives in a few files worth opening: the grounding gate ([`agent/grounding_check.py`](agent/grounding_check.py), asserted by [`tests/test_grounding.py`](tests/test_grounding.py)); the deterministic verdict engine ([`agent/verdict.py`](agent/verdict.py)); the skill-driven per-dataset annotation ([`pipeline/stages/annotate.py`](pipeline/stages/annotate.py), [`agent/annotation.py`](agent/annotation.py)); and the calibration table (`python scripts/calibration_table.py`).
- The interpretation contracts are the two skills under `skills/`; the product simply executes them.

The rest of this file is the honest map of who does what.

---

## 1. Build agents (how the software was made)

A planner produced a contract; parallel builders implemented against it; a referee kept them honest.
The blueprint and its build DAG are real artifacts in the repo.

| Role | What it produced | Artifact |
| --- | --- | --- |
| **Planner / architect** | File tree, the frozen shared dataclasses every module builds against, the verdict algorithm, and a **build DAG of parallel groups** (G0 contracts → G2 engine → G3 grounding → G4 loop → G5 UI) | [`BLUEPRINT.md`](BLUEPRINT.md), [`agent/types.py`](agent/types.py) |
| **Parallel builders** | Each DAG group implemented independently against the frozen contract: the deterministic engine, the memory layer, the MCP tools, the grounding checker, and the UI panes | `agent/`, `ui/`, `pipeline/` |
| **Reconciler** | Where independent agents made divergent choices (tidy-data layout, note format, citation convention), they were reconciled into one contract *before* merge; the blueprint's **"Key reconciliations"** table is the honest record | [`BLUEPRINT.md`](BLUEPRINT.md) §"Key reconciliations" |
| **Grounding-CI referee** | Every agent's output had to trace to source or be discarded; the deterministic grounding suite is wired into CI so parallel speed never cost the confident floor | [`.github/workflows/ci.yml`](.github/workflows/ci.yml), `tests/` |
| **Isolated worktrees** | Kept parallel streams from colliding (e.g. the Summary sign-off board) | `.claude/worktrees/` |

Because the builders shared one type contract, their work composed instead of drifting.

---

## 2. Runtime specialists (how the product reasons)

One grounding-gated loop, but the *work* is split into specialized stages, each with a single job and a
grounding contract. This is the honest "team" of the running product.

| Specialist | One job | Grounding contract | Module |
| --- | --- | --- | --- |
| **Annotator** | Assign each cluster's cell type, lineage/category, and canonical markers from its jazzPanda markers (the skill's Output 2) | The marker-gene skill reads only that cluster's markers; the call is persisted to `interp/annotation.json` and read thereafter | [`pipeline/stages/annotate.py`](pipeline/stages/annotate.py), `skills/jazzpanda-markers/` |
| **Verdict engine** | Score the annotated call's confidence band and attach panel-absence, from jazzPanda `glm_coef` | Deterministic; **owns every number**; the LLM is fenced out of statistic-generation | [`agent/verdict.py`](agent/verdict.py) |
| **Literature note-writer** | Write the per-marker / per-cluster biology, cited | One **real live PMID or none**, fetched via the PubMed MCP; never from memory | [`pipeline/stages/notes.py`](pipeline/stages/notes.py), `skills/jazzpanda-markers/` |
| **Discriminator** | "What would settle it": name the markers that separate the call from its alternative | Reads the cluster's *own* jazzPanda numbers; off-panel alternatives flagged never-measured | [`agent/discriminate.py`](agent/discriminate.py) |
| **Second opinion** (risk flag + live agent) | A deterministic *risk flag* screens every call from the numbers (thinness + localization + competing markers); on demand a *live agent* refutes the call as it stands, cited | Both trace to jazzPanda and clear the same grounding gate; the live agent falls back to the deterministic report | [`agent/skeptic.py`](agent/skeptic.py), `loop.pressure_test` |
| **Holistic reviewer** | A second opinion across *all* clusters (coherence + one refinement) | Refinements carry markers read from data + a real citation; numbers unchanged | [`agent/holistic.py`](agent/holistic.py) |
| **Enrichment interpreter** | Read gene-set programs per cluster (the Pathways workflow) | Panel-coverage rule (`K of N`, panel-scoped); cross-lineage flagged as tension, not re-typing | [`agent/enrichment.py`](agent/enrichment.py), `skills/geneset-enrichment/` |
| **Memory reconciler** | On an override, capture a scope-locked note and cross-check the literature; optionally save a portable, distilled copy to user memory | Keeps the biologist's call *with* the disagreement visible; cite-on-use; no note or memory rewrites a number | [`agent/memory.py`](agent/memory.py), [`agent/user_memory.py`](agent/user_memory.py) |
| **The grounding gate (referee)** | Veto any answer whose numbers/citations/notes don't trace to source | Prose-independent extraction, resolves against real data, **fails closed** to the deterministic fallback | [`agent/grounding_check.py`](agent/grounding_check.py) |
| **Deterministic fallback** | Guarantee a grounded answer even if a live call is slow or fails | Pre-baked, fully-grounded responses; the demo never breaks | [`agent/fallback.py`](agent/fallback.py) |

The orchestrator that runs the loop and applies the gate is [`agent/loop.py`](agent/loop.py).

The pipeline that builds a dataset's tree is **per-dataset and read-if-present, else generate**: a `prep`
stage ([`pipeline/stages/prep.py`](pipeline/stages/prep.py)) turns raw Seurat / jazzPanda `.Rds` into the
tidy inputs, the annotator assigns the cell types, and every downstream artifact is reused when already
present. The active dataset is selected with `PANOSCOPE_DATASET`; the bundled demo ships its artifacts and
rebuilds nothing.

Every system prompt also carries two grounded, open-ceiling context blocks: the dataset's **tissue**, read
from the manifest (e.g. human breast cancer, Xenium) so reasoning and citations are tissue-appropriate (a
search preference, not a filter, so a real cross-tissue paper is still valid), and the biologist's
**portable user memory** (prior saved decisions). Neither can change a jazzPanda number, a marker, or a
confidence band.

---

## 3. See the referee work

The grounding gate is the load-bearing idea: it is what lets an LLM be trusted where a fabricated
number is fatal. [`tests/test_grounding.py`](tests/test_grounding.py) exercises it on clean and poisoned
answers: it accepts a grounded answer and rejects an inflated `glm_coef`, a fabricated PMID, and a
jazzPanda number attached to an off-panel gene. The same checker ([`agent/grounding_check.py`](agent/grounding_check.py))
runs inside [`agent/loop.py`](agent/loop.py) on every turn, discarding any violating answer in favor of
the deterministic grounded fallback.
