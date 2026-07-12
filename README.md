<p align="center">
  <img src="assets/panoscope_logo_with_text.png" alt="Panoscope" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/MelodyJIN-Y/Panoscope/actions/workflows/ci.yml"><img src="https://github.com/MelodyJIN-Y/Panoscope/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
</p>

**An annotation-confidence layer for [jazzPanda](https://github.com/phipsonlab/jazzPanda).** A biologist asks about a cluster in plain language; Panoscope answers with a cell-type call, a confidence level, and the evidence behind it — every number traced back to jazzPanda's spatial marker output, every literature claim carrying a real, live-fetched PubMed citation.

jazzPanda is the engine. This is the interpretation layer.

## The honest framing

- **Confident floor.** Every marker and number in an answer comes from jazzPanda's precomputed output or the panel gene list. Every citation is looked up **live through PubMed**, never written from memory. Nothing is fabricated — a grounding gate discards any answer that states something not in source.
- **Open ceiling.** Reasoning from the literature about what a cluster *means* is uncertain. Panoscope labels it as a direction, not a fact, and shows agreement and dissent side by side.

## The headline catch

Panoscope catches cell-type calls you would get wrong **because a canonical marker was never on the panel.**

On a targeted panel, a general reader (or a general LLM) sees a missing marker and quietly counts it as evidence *against* a cell type. That is a mistake: an off-panel gene was never measured, so its absence says nothing. Cluster **c2** is a stromal population, but its textbook markers **COL1A1 and VIM are off-panel** in this 280-gene assay. Panoscope checks the panel list first, marks those markers "not measured" instead of "absent", and holds the call at **Very High** on the markers that *are* on the panel (LUM glm 18.00, POSTN glm 15.80). The absence of an off-panel gene never lowers confidence.

## Why this matters — and where it goes

Bioinformatics keeps getting more powerful and more opaque. A spatial method like jazzPanda produces statistically rigorous output that a wet-lab biologist still has to translate, by hand and under time pressure, into *"what is this cluster, and can I trust the call?"* That translation is where hours go and where mistakes hide: a missing marker misread as evidence against a cell type, a fair result over-interpreted into a story the data does not support.

Panoscope closes that gap for one method — but **the gap is general, and it is widening.** As analysis tools and agent-driven pipelines grow more capable, the distance between a sophisticated result and a confident, non-misleading reading only grows with them. Every such tool needs a layer that reads its output faithfully, cites real evidence, and refuses to overclaim. Here that layer is a **portable skill, not app code**: point any agent at a method's output plus its gene panel and it works — which makes this pattern a natural fit for Claude-driven science, not only this repository. jazzPanda is one instance; the same shape extends to the next spatial method, the next enrichment test, the next agentic analysis workflow.

It also **compounds across a lab's work.** The memory layer already reconciles a biologist's judgment with the literature into attributed, git-tracked notes the lab owns. Extended across a researcher's projects, that becomes a personal research assistant that consolidates *transferable, checkable* knowledge — how a marker behaves in this tissue, a lab convention that held up under scrutiny — instead of re-deriving it each time. The tool never trains and never learns on its own; it accumulates knowledge the lab can audit and carry forward, project to project.

**Who benefits, and how much.** Every wet-lab biologist now adopting spatial and single-cell methods faster than they can vet them. The cost of the interpretation gap is concrete: a wrong annotation propagates silently into every downstream analysis. A layer that is *confident where the data is clear and honest where it is not* is the missing piece that lets biologists use powerful methods without being overwhelmed by the machinery or misled by over-interpreted findings — exactly the kind of grounded, self-checking layer that agent-driven science will need at scale.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py                             # launch the app
pytest -m "not live"                             # run the grounding suite (deterministic, no network)

python -m pipeline.run --dataset <id>            # (re)build a dataset's tree: verdicts + manifest (offline)
python -m pipeline.stages.notes --dataset <id>   # live: skill-grounded per-marker + cell-type notes (PubMed)
```

The app runs entirely on precomputed jazzPanda output — no live jazzPanda run, no GPU. For live PubMed citations inside the app, add your NCBI credentials (see [MCP setup](#skill--mcp-setup)); without them the app still runs and every jazzPanda-grounded number is unaffected.

## Verify it (offline, no keys)

Every headline claim is runnable from a fresh clone, no API key required:

```bash
pytest -m "not live"                    # 230+ grounding tests, no network — the confident floor, checked
pytest tests/test_grounding.py -v       # the gate accepts grounded answers and rejects poisoned ones
pytest tests/test_calibration.py -v     # commits on clean calls (c1 Very High), flags shaky ones (c9 verify)
pytest tests/test_annotate.py -v        # the marker skill assigns a cell type + grounds panel-absence
python scripts/calibration_table.py     # print the commit-vs-flag calibration table
```

`pytest -m "not live"` fails if any answer ever states a marker, number, or citation that does not trace
to source, so a green run *is* the guarantee. The optional `-m live` suite adds a real PubMed lookup when
NCBI credentials are present. This is the fastest way to confirm the project does what it claims.

## The pipeline: one command per dataset

The skill is the interpretation contract; the **per-dataset pipeline is its executor**. `python -m pipeline.run --dataset <id>` turns a dataset's inputs (jazzPanda markers, the gene panel, an optional gene-set enrichment result, and viz sources) into a self-contained `data/datasets/<id>/` tree — `annotation.json`, `verdicts.csv`, per-cluster `interp/clusters/c{n}.json`, `gene_notes.json`, `celltype_notes.json`, and a `manifest.json` (provenance + artifact hashes) — which the UI reads with no live recomputation. It runs in two skill-driven tiers:

- **Tier A — the skill's mechanical rules as deterministic code** (`agent/verdict.py` = SKILL Steps 3a/3b/3d): per-gene evaluation, the `glm_coef`-anchored confidence rubric, and the panel-absence rule. Offline and reproducible.
- **Tier B — the skill's literature interpretation, live** (SKILL.md sits in the agent's system prompt): the Output-4 per-marker biology note is *the skill reading that gene's own Tier-A evidence* — role framed by canonical status, and a specificity caveat flagged only when the numbers show the gene marks another lineage. One real live PMID or none.

**Bring your own dataset.** Panoscope derives its interpretation entirely from the data you provide: the cluster set from its markers, the cell-type call from the marker-gene skill, and the panel-absence set from its own gene panel. Place the tidy inputs under `data/datasets/<id>/inputs/` (`markers_top.csv`, `panel.parquet`, and optionally `enrichment.csv`), then:

```bash
PANOSCOPE_DATASET=<id> python -m pipeline.run --dataset <id> --notes   # build the tree
PANOSCOPE_DATASET=<id> streamlit run app.py                            # review it
```

Each stage is **read-if-present, else generate**. A `prep` stage converts raw Seurat / jazzPanda `.Rds` objects into the tidy inputs (`scripts/prep_data.R`; skipped when they already exist). An **`annotate` stage applies the marker-gene skill to each cluster** — from that cluster's jazzPanda markers it produces the cell-type call, its lineage and category, and the canonical markers that ground the panel-absence rule (the skill's Output 2). The deterministic verdicts, the gene-set enrichment, and the live per-marker and per-pathway biology notes follow. Artifacts already present are read as they are, so a prepared dataset — including the bundled demo — rebuilds nothing. jazzPanda is never run; its output is a consumed input.

On a compact three-cluster subset of the demo panel, the skill assigns c1 → *HER2-positive tumor epithelial*, c2 → *cancer-associated fibroblast*, and c3 → *macrophage*, with panel-absence evaluated against that dataset's own panel — a full run in seconds.

## Calibration — commits on clean calls, flags the shaky ones

A confidence layer is only worth trusting if it does both jobs: it commits without hedging when the evidence is clean, and it raises `verify` when the evidence is thin. Below is the full 9-cluster verdict set on the demo dataset, produced by the deterministic engine (regenerate with `python scripts/calibration_table.py`):

| Cluster | Cell type | Confidence | Verify | Driving markers |
| --- | --- | --- | --- | --- |
| c1 | Tumor | Very High | FALSE | ERBB2 (glm 21.44), KRT7 (glm 13.48), EPCAM (glm 9.52) |
| c2 | Stromal | Very High | FALSE | LUM (glm 18.00), POSTN (glm 15.80), PDGFRB (glm 2.56) |
| c3 | Macrophages | Very High | FALSE | LYZ (glm 11.84), FCER1G (glm 4.73), CD68 (glm 4.49) |
| c4 | Myoepithelial | High | FALSE | MYLK (glm 5.90), ACTA2 (glm 5.07), KRT14 (glm 4.65) |
| c5 | T_Cells | Medium-High | FALSE | IL7R (glm 3.40), PTPRC (glm 3.26), TRAC (glm 2.81) |
| c6 | B_Cells | Medium-High | FALSE | MS4A1 (glm 2.55), BANK1 (glm 1.58), CD79A (glm 0.94) |
| c7 | Endothelial | High | FALSE | AQP1 (glm 7.37), PECAM1 (glm 4.92), VWF (glm 4.89) |
| c8 | Dendritic | Medium-High | FALSE | TCL1A (glm 1.88), LILRA4 (glm 1.75), SPIB (glm 1.48) |
| c9 | Mast_Cells | Medium | TRUE | CPA3 (glm 1.95) |

The clean calls commit (c1 Tumor rides ERBB2 at glm 21.44 → Very High, no verify). The fragile one is flagged: **c9 Mast_Cells** rests on a single marker (CPA3, glm 1.95, 2 markers total) that localizes better elsewhere, so it lands at **Medium with `verify = TRUE`** — re-check this. The calibration set is asserted in `tests/test_calibration.py`, which fails loudly if the rubric ever collapses into rubber-stamping everything or crying wolf on everything.

## How it works

1. **jazzPanda spatial markers → verdict.** jazzPanda measures whether a gene's transcripts fall where a cluster's cells are (a spatial GLM coefficient), not whether the gene is higher on average (mean-expression DE). `agent/verdict.py` reads that `glm_coef` directly: a larger coefficient on a cluster's driving canonical marker means a higher confidence band, corroborated by `pearson` spatial specificity. Small-n clusters (few assigned markers) are capped and flagged. The engine never invents or re-derives a statistic.
2. **The panel-absence rule.** Before down-weighting any missing canonical marker, the verdict checks the panel gene list (`panel_contains`). An off-panel gene is surfaced as "not measured" and never lowers confidence. This is the headline catch, made a hard invariant in `tests/test_panel_absence.py`.
3. **The grounding gate.** Every agent answer passes a checker (`agent/grounding_check.py`) that traces each marker, number, PMID, and lab note back to source. If anything doesn't resolve, the answer is discarded in favor of a deterministic, fully-grounded fallback — so the demo never shows an unverified claim and never breaks.
4. **Live PubMed citations.** Interpretive claims are looked up live via a PubMed MCP connector. A real, clickable PMID is fetched at answer time; a PMID is never written from memory. When the literature is thin, Panoscope says so rather than inventing a reference.
5. **Memory as scope-enforced lab notes.** When the biologist's judgment diverges from the default, Panoscope captures it as one of **eight typed, anchored notes** — `{ claim, type, scope, basis, status, tension }`, git-tracked. Not just a full cell-type override, but the everyday moments that used to evaporate: a marker re-read (*"POSTN here is tumor bleed, down-weight it"*), a re-interpreted pathway program, a validation (*"we confirmed c4 by p63 IHC"*), a confidence adjustment, an exclude, or a *"these two clusters are one population"*. Every type runs the **same two-tap capture** — cross-checked against the literature *before* it renders, keeping the biologist's call *with the disagreement visible* — from all three surfaces (the marker chat, the Pathways chat, and the holistic review). Notes render **anchored to their subject**: a marker note as a caveat row beneath that gene's driver, a program note beside its gene-set row, a confidence note as a dual band (`lab: High · computed Medium`, never overwriting the computed value). Scope is fail-closed (a cluster- or dataset-scoped note never fires elsewhere) and the agent must cite a note (`[note:id]`) to use it. An `exclude` note flips the exported flag at report-composition time **without mutating the deterministic verdict underneath**, and no note can ever rewrite a jazzPanda number. It is a knowledge layer the lab owns; nothing is trained, nothing learns on its own.
6. **"What would settle it."** When a call is ambiguous — the biologist asks *"could c1 be myoepithelial?"*, or a cluster is flagged `verify` — the agent names the concrete discriminating markers (`agent/discriminate.py`), grounded in that cluster's *own* jazzPanda numbers. Because jazzPanda assigns each gene to a single winning cluster, a marker's coefficient exists only there, so the discriminator re-reads a cluster's own markers across cell types: markers that support the call here (with numbers), alternative-type markers that are on the panel but localize to another cluster (measured, so they argue *against* the alternative — e.g. ACTA2/MYH11 localize to c4, not c1), and alternative-type markers that are off-panel (TP63) — **flagged** as never-measured, never a bench recommendation. Deterministic with a demo-safe fallback; the live agent adds one real citation for the distinguishing claim. Asserted in `tests/test_discriminate.py`.
7. **A downloadable interpretation summary.** The Summary page auto-assembles a per-dataset report from the durable, grounded artifacts — each cluster's call, confidence, driving numbers, the "what would settle it" line for shaky calls, the live-cited cell-type biology, and your saved lab notes with their tension — reviewable on the page and downloadable as Word (`.docx`) or PDF (`ui/report.py`). There is no live network at export time, so the report is a reproducible artifact. Asserted in `tests/test_report.py`.

8. **An adversarial second opinion.** Alongside each call, a skeptic agent (`agent/skeptic.py`) tries to *refute* it using only grounded facts — evidence thinness, a driver that localizes as well elsewhere, a rival cell type whose markers also peak here — and reports whether the call withstands the challenge or should be re-checked. It clears the same grounding gate as every other answer, so the engine that argues a call and the one that argues against it are held to the same standard. Asserted in `tests/test_skeptic.py`.

9. **Literature search tuned to the lab, kept private.** At setup the biologist can note their research focus in one line (e.g. breast-cancer CAF biology). It is stored locally and used only to make the agent's PubMed searches more precise and tissue-appropriate; it never leaves the machine and never changes a jazzPanda number, a marker, or a confidence — only which real paper is surfaced.

## A second workflow: gene-set enrichment (Pathways)

Markers say *what cell it is*; enrichment says *what program is running*. Panoscope's **Pathways** workflow interprets a per-cluster **MSigDB Hallmark** gene-set enrichment for clusters that are already cell-typed — a separate skill (`skills/geneset-enrichment/`), a separate pipeline stage (`pipeline/stages/enrichment.py`), a separate tab. It complements the cell-type call and never re-derives it.

The engine is the **jazzPanda competitive gene-set test**, re-scoped to the panel: for a set and a cluster it lasso-selects the set's genes whose spatial vectors track the cluster, then a one-sided z-test compares them against the mean of all *other* panel genes. The confident-floor discipline carries over intact:

- **The panel-coverage rule** — the enrichment analog of the panel-absence catch. A Hallmark set has ~200 genes, but only the genes *on the panel* were ever measured, so every program states `K of N` genes on panel and is labelled **panel-scoped, not genome-wide**. A set with too few panel genes, or a leading edge of 1–2 genes, is **untestable** and never surfaced as a program.
- **Two tiers with `verify`.** Programs are gated into *enriched* (`q < 0.05`, ≥3 driving genes on panel), *suggestive* (`q` in `[0.05, 0.25]`, carries `verify = TRUE`), and *untestable*. Confidence anchors on the top program's score.
- **Concordance vs. tension.** Each program is read against the cluster's cell-type call: a concordant program (proliferation in a tumor) reinforces the identity; a cross-lineage one (an immune program in a stromal cluster) is flagged as a **tension to check, never a re-typing**. A cross-cluster themes review reports which programs recur, so a signal enriched everywhere is down-weighted as panel/atmospheric rather than cluster-specific.
- **One real PMID or none**, live-fetched, exactly as on the marker path.

## Skill + MCP setup

The interpretation logic ships as two installable, standalone skills. The marker workflow is **[`skills/jazzpanda-markers/`](skills/jazzpanda-markers/SKILL.md)** — the panel-absence rule, the `glm_coef`-anchored confidence rubric, the holistic cross-cluster review, and the no-fabrication / real-citation discipline; point any agent at jazzPanda `top_result` output plus a panel list and it works. The Pathways workflow is a second skill, **[`skills/geneset-enrichment/`](skills/geneset-enrichment/SKILL.md)** — the panel-coverage rule, the two-tier enrichment report, and the cross-cluster themes review. Both install and run independently of this app.

Live citations come from a PubMed MCP server declared in [`.mcp.json`](.mcp.json) (`@cyanheads/pubmed-mcp-server`). To enable it, copy `.env.example` to `.env` and set `NCBI_API_KEY` and `NCBI_ADMIN_EMAIL` (the email registered to that key). Get a key at <https://www.ncbi.nlm.nih.gov/account/settings/>. The `pytest -m "not live"` suite needs none of this; one optional live test (`-m live`) exercises a real lookup when credentials are present.

## How we built it — a team of Claude Code agents

Panoscope was built the way it reasons: decomposed, contract-first, and verified. We used Claude Code as an orchestrated **team of agents**, not a single chat.

1. **A planner/architect agent** turned the goal into [`BLUEPRINT.md`](BLUEPRINT.md) — a file tree, the shared frozen dataclasses in [`agent/types.py`](agent/types.py) (the interface every module builds against), the verdict algorithm, and a **build DAG of parallel groups** (G0 contracts → G2 engine → G3 grounding → G4 loop → G5 UI).
2. **Parallel builder agents** implemented each group against those frozen contracts — the deterministic verdict engine, the memory layer, the MCP tools, the grounding checker, and the UI panes independently. Because they shared one type contract, their work composed. The blueprint's **"Key reconciliations"** section is the honest record of it: where independent agents made divergent choices (tidy-data layout, note format, citation convention), we reconciled them into one contract before merging.
3. **Isolated git worktrees** kept parallel streams from colliding (e.g. the Summary sign-off board).
4. **A grounding gate + CI was the shared referee** — every agent's output had to trace to source or be discarded, so parallel speed never cost the confident floor.

The result — two standalone skills, a per-dataset pipeline, three spatial views, and a grounding suite wired into CI — came together in days *because* the agents built in parallel against a contract, not in sequence. The product itself stays deliberately simple: one grounding-gated agent loop over a deterministic engine.

## Data provenance

The demo dataset is the public **10x Genomics Xenium FFPE Human Breast Cancer (Rep 1)** sample — the 280-gene analyzed panel, 9 clusters (c1–c9). Marker statistics come from **published jazzPanda** run offline; the app reads that **precomputed output** and never runs jazzPanda live. All spatial views (cell map, hex-bin density, UMAP) are built from the same public data. This keeps the demo deterministic and reproducible, and keeps the confident floor intact: every number on screen has a checkable source.

## License

MIT. Everything shown is open source. jazzPanda is a published dependency under its own license; all interpretation-layer code in this repository is new work built for this project.
