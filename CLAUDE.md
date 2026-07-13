# CLAUDE.md

Project: Panoscope — an annotation-confidence layer for jazzPanda.

Context for building this project with Claude Code. Read this before writing code.
Keep it tight. Every line here loads on every turn, so only load-bearing rules live here.

## What this is

A grounded conversation with spatial data. A biologist asks about a cluster in plain language, and
Panoscope answers with a cell-type call, a confidence level, and the evidence behind it. It reads
jazzPanda's spatial marker output, never invents a marker or a number, and it remembers what the lab
tells it as context it can cite. jazzPanda is the engine. This is the interpretation layer. The chat is
the primary interface; the panels are the evidence it stands on.

## Core principle (never violate)

Confident floor: the agent reads jazzPanda's marker output and the panel gene list. It never invents a
marker, a number, or a confidence score. Every number in an answer must trace to jazzPanda's output,
the panel list, or a stored lab note.

Open ceiling: literature and context reasoning is uncertain. Label it as a direction, not a fact.

## Hard rules for the agent

- Never fabricate. If a marker, statistic, or value is not in the source, the agent does not state it.
- Panel-absence rule: the absence of an off-panel gene is not evidence against a cell type. Check the
  panel list before down-weighting a missing canonical marker. Say when a marker was never measured.
- Cite everything. Every claim names its source: a jazzPanda number, a panel fact, a paper, or a lab note.
- Citations must be real and looked up. Every interpretive claim carries a citation to a real paper,
  fetched live through the PubMed or bioRxiv connector. Never write a PMID or DOI from memory. A fabricated
  citation is the worst possible failure, worse than no citation. If a lookup returns nothing, say the
  literature is thin, do not invent a reference.
- When evidence is insufficient, set verify = TRUE and say re-check this. Do not guess to seem helpful.
- No overclaiming. The tool accumulates the lab's knowledge, it does not learn on its own. The feedback
  loop writes note files and re-reads them. It does not train anything.

## Memory: a reconciliation layer, not a memory of the user

Memory is where the biologist's judgment and the literature get reconciled into something the lab owns.
The biologist has depth (tissue-specific truth, lab convention, unpublished context). The agent has
breadth (the literature). The point is to combine them, with disagreement made visible, not to make the
agent agree faster.

- The value is in the disagreement. When the biologist overrides, the agent uses the call AND cross-checks
  the literature, reporting agreement and dissent with real citations, keeping the biologist's call with
  the tension visible. Never a bare "got it", never a silent overrule.
- Capture at override. That is where the biologist's knowledge diverges from the default. Ask scope and
  basis in two taps, not a form.
- The biologist decides, always. The tool informs and preserves the decision. It never makes the call.

A note is a structured object, not free text:
`{ claim, scope: cluster|dataset|lab, basis: paper|own_validation|convention, status: firm|tentative, tension: [citations] }`
Stored as versioned, scoped plain-text files under context/ that a lab version-controls as its own
knowledge (kept local in this public demo for privacy). Scope is enforced (a dataset or cluster note must
not fire elsewhere). The agent must cite a note to use it, and must show any attached tension when it does.
This is not a model that learns. It is a knowledge layer the lab owns, and it compounds because it
accumulates checkable, attributed knowledge.

A decision can also be saved to a portable user memory (agent/user_memory.py, local-only
context/user_memory.jsonl): a distilled, tissue-tagged copy that travels across projects and is loaded
back as prior lab knowledge on future datasets. Like the dataset tissue context, it is open-ceiling: it
sharpens reasoning and which real paper is cited, and can never change a jazzPanda number.

## Opening interpretation (first thing the biologist sees)

When a cluster opens, the agent posts its interpretation before any question: the call and confidence, the
driving markers with their jazzPanda numbers, and the literature basis for the cell-type assignment, each
claim carrying a real, clickable citation. Every citation here is fetched live, never written from memory.

## Spatial views (viewing controls, not analysis knobs)

One pinned marker drives three linked views the biologist switches between. Click a marker in the
evidence table to pin it, hover to preview.

- Transcript density: raw transcripts hex-binned, before cell calling. "Is the signal really there."
- Cell map: segmented cells at their tissue locations. "Did it land in the assigned cells." Default view.
- UMAP: expression space. "Is the cluster a clean island or bleeding into a neighbour."

The hex-bin bin size is a small preset control in µm (for example 25 / 50 / 100). The bin size and the
view toggle are viewing controls: they change the picture, never a value. Keep the density colour scale
area-normalized so coarser bins do not look hotter and mislead. Persist the pinned marker and bin size
across views.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py                             # the app
pytest -m "not live"                             # grounding tests, must pass before every commit
python -m pipeline.run --dataset <id>            # (re)build a dataset's tree: verdicts + manifest (deterministic)
python -m pipeline.stages.notes --dataset <id>   # live: skill-grounded gene + cell-type notes (PubMed)
```

## The pipeline (how a dataset is processed)

The skill is the interpretation contract; the **per-dataset pipeline is its executor and persistence layer**. One command per dataset builds a self-contained `data/datasets/<id>/` tree the UI reads with no live recomputation. Two tiers, both skill-driven:

- **Tier A — the skill's mechanical rules as deterministic code** (`agent/verdict.py` = Steps 3a/3b/3d): per-gene evaluation + confidence rubric + panel-absence. No LLM, no network.
- **Tier B — the skill's literature interpretation, live** (SKILL.md is in the agent's system prompt): the Output-4 per-marker biology note (grounded in that gene's Tier-A evidence) and the per-cluster cell-type note. One real live PMID or none.

jazzPanda is never run — its output is a consumed input. The UI falls back to legacy flat files when a tree artifact is absent, so the app works mid-migration.

## Repo layout

```
data/
  datasets/<id>/    # THE per-dataset pipeline output the UI reads (built by pipeline.run)
    inputs/         #   raw inputs copied + hashed (markers_top.csv, panel, cluster_key)
    interp/         #   verdicts.csv, clusters/c{n}.json, gene_notes.json, celltype_notes.json
    manifest.json   #   provenance + artifact sha256s + views_available
  jazzpanda/ panels/ cells/ transcripts/ embeddings/ density/   # inputs + precomputed viz frames
  gene_notes/       # legacy flat notes (fallback until the tree's gene_notes is complete)
pipeline/           # ONE command per dataset: validate -> verdicts -> notes -> manifest
  run.py            #   entrypoint; stages/ = validate (0), verdicts (4), notes (6, live), enrichment, viz
  serialize.py store.py paths.py manifest.py calibration.py
skills/
  jazzpanda-markers/SKILL.md    # the marker interpretation contract; the pipeline executes it
  geneset-enrichment/SKILL.md   # the Pathways workflow contract (gene-set enrichment)
agent/
  verdict.py        # the skill's Steps 3a/3b/3d as deterministic code (Tier A)
  loop.py tools.py  # tool-use loop + tools (panel/marker/literature MCP/memory). Never generates stats.
                    #   every prompt also carries grounded tissue context (from the manifest) + user memory; open-ceiling, never moves a number
  skeptic.py        # second opinion: deterministic risk flag + live pressure-test (loop.pressure_test); same grounding gate
  user_memory.py    # portable, distilled decisions kept locally across projects (context/user_memory.jsonl)
ui/                 # Streamlit surfaces; read via ui/data_access.py (tree first, legacy fallback)
context/            # lab notes (corrections/) + decision log (scoped, versioned; a lab tracks its own, local here) + user_memory.jsonl (local, portable)
app.py  tests/  README.md
```

## Data check  [LOCKED — verified against the real data 2026-07-08]

- Marker statistic + schema: `glm_coef` (spatial GLM Estimate, primary ranking) from jazzPanda
  `top_result`. Columns: `gene, top_cluster (c1–c9|NoSig), glm_coef, pearson, max_gg_corr, max_gc_corr`.
  `pearson` = gene↔assigned-cluster spatial specificity. Multivariate spatial GLM, not per-cluster DE.
- Confidence mapping (replaces mean-expression pct.1/pct.2): `glm_coef` DIRECT — bigger coef = higher
  band. Bands in `agent/config.py`: ≥10 Very High, ≥5 High, ≥2.5 Medium-High, ≥1 Medium, else Low.
  `pearson` corroborates (±1 band). Few-marker (1–2) clusters are capped and flagged `verify`;
  NoSig/near-cutoff = Low + verify. No within-cluster percentile.
- Demo dataset: 10x Xenium FFPE Human Breast Cancer Rep1 — **sample1 / biorep `hb1` only** (sample2
  dropped). Panel: analyzed set = **280 genes** (TSV Annotation≠Custom); Custom add-ons + neg-controls
  are off-panel. Panel size is DYNAMIC across datasets — derive `panel_contains` from the file, never
  assert a constant. 9 clusters: c1 Tumor, c2 Stromal, c3 Macrophages, c4 Myoepithelial, c5 T_Cells,
  c6 B_Cells, c7 Endothelial, c8 Dendritic, c9 Mast_Cells.
- Cell coordinates: yes — `cells.csv` (158,379 sample1 cells, x/y centroids). Cell map = default view.
- Transcript-level coordinates: yes — `transcripts.csv.gz`; hex-bins precomputed per demo marker,
  area-normalized, neg-controls + qv<20 filtered.
- UMAP embedding: yes — `umap.csv` (sample1, Idents c1–c9).
- Default hex bin size: 50µm (presets 25 / 50 / 100).

The cell map needs only cell coordinates. Density and UMAP each depend on the two coordinate/embedding
files above; if either is missing for a new dataset, that view is out — do not fake it. Precompute hex
bins for the demo markers rather than binning live.

## Skills

Each skill is a SKILL.md the agent loads. The jazzpanda-markers skill encodes the panel-absence rule,
the confidence rubric on jazzPanda's spatial signal, and the holistic review (re-check the whole set of
calls after annotating each cluster). It must install and run standalone, independent of the app. Adding
support for a new method means writing a new skill file — `skills/geneset-enrichment/` is the second such
skill (the Pathways workflow: jazzPanda-result-driven gene-set enrichment, executed by
`pipeline/stages/enrichment.py` and surfaced in the Pathways UI), grounded by the same no-fabrication /
real-citation discipline.

## Grounding tests (the guarantee, not an afterthought)

tests/ parses each agent answer and checks that every marker, number, note, and citation in it traces back
to jazzPanda's output, the panel list, a stored lab note, or a real literature record from an actual
connector lookup. The suite fails if the agent states anything not in source, uses a lab note without
citing it, or produces a citation that does not resolve to a real record. Run on every change and wire it
into CI so the badge is green. Also keep a small calibration set (a few well-supported calls, a few shaky
ones) with expected verdicts, reported as a table in the README.

## Output format

Per cluster: cell-type call, confidence level, the markers and numbers used, a short rationale, a verify
flag. Confidence labels: Very High, High, Medium-High, Medium, Low. Low or ambiguous clusters get
verify = TRUE.

CSV columns, in order:
`cluster, cell_type, cell_type_short, confidence, confidence_score, key_markers, notes, category, lineage, exclude, verify`

## Agent response style

Plain language for a wet-lab biologist. Every point cites a gene and its number. State the confidence
and name the caveat. Say plainly when a marker was off-panel and what that does and does not tell us.
When unsure, say re-check this and why. No hype, short over long.

## Compliance and demo safety

- New work only. jazzPanda is a published dependency. Everything in this repo is new work built fresh
  for this project. Build the interpretation skill fresh here; do not paste prior work in as-is.
- License: MIT. Everything shown must be open source.
- Precomputed jazzPanda output only. Do not run jazzPanda live in the demo.
- Deterministic fallback for every agent call so a slow or failed request never breaks the demo.

## What not to do

- Do not invent markers, stats, or confidence values.
- Do not treat off-panel absence as evidence against a cell type.
- Do not apply a lab note without citing it, or across a scope it was not saved for.
- Do not let a viewing control (bin size, view toggle) change any value.
- Do not claim the tool learns on its own.
- Do not run jazzPanda live in the demo.
- Do not add breadth (more platforms, more methods, more views) before the one marker path is solid
  end to end.
