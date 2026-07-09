<p align="center">
  <img src="assets/panoscope_logo_with_text.png" alt="Panoscope" width="300"/>
</p>

<p align="center">
  <a href="https://github.com/MelodyJIN-Y/Panoscope/actions/workflows/ci.yml"><img src="https://github.com/MelodyJIN-Y/Panoscope/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
</p>

**An annotation-confidence layer for [jazzPanda](https://github.com/rwang-z/jazzPanda).** A biologist asks about a cluster in plain language; Panoscope answers with a cell-type call, a confidence level, and the evidence behind it — every number traced back to jazzPanda's spatial marker output, every literature claim carrying a real, live-fetched PubMed citation.

jazzPanda is the engine. This is the interpretation layer.

## The honest framing

- **Confident floor.** Every marker and number in an answer comes from jazzPanda's precomputed output or the panel gene list. Every citation is looked up **live through PubMed**, never written from memory. Nothing is fabricated — a grounding gate discards any answer that states something not in source.
- **Open ceiling.** Reasoning from the literature about what a cluster *means* is uncertain. Panoscope labels it as a direction, not a fact, and shows agreement and dissent side by side.

## The headline catch

Panoscope catches cell-type calls you would get wrong **because a canonical marker was never on the panel.**

On a targeted panel, a general reader (or a general LLM) sees a missing marker and quietly counts it as evidence *against* a cell type. That is a mistake: an off-panel gene was never measured, so its absence says nothing. Cluster **c2** is a stromal population, but its textbook markers **COL1A1 and VIM are off-panel** in this 280-gene assay. Panoscope checks the panel list first, marks those markers "not measured" instead of "absent", and holds the call at **Very High** on the markers that *are* on the panel (LUM glm 18.00, POSTN glm 15.80). The absence of an off-panel gene never lowers confidence.

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

## The pipeline: one command per dataset

The skill is the interpretation contract; the **per-dataset pipeline is its executor**. `python -m pipeline.run --dataset <id>` turns a dataset's raw inputs (jazzPanda markers, panel list, cluster key, + viz sources) into a self-contained `data/datasets/<id>/` tree — `verdicts.csv`, per-cluster `interp/clusters/c{n}.json`, `gene_notes.json`, `celltype_notes.json`, and a `manifest.json` (provenance + artifact hashes) — which the UI reads with no live recomputation. It runs in two skill-driven tiers:

- **Tier A — the skill's mechanical rules as deterministic code** (`agent/verdict.py` = SKILL Steps 3a/3b/3d): per-gene evaluation, the `glm_coef`-anchored confidence rubric, and the panel-absence rule. Offline and reproducible.
- **Tier B — the skill's literature interpretation, live** (SKILL.md sits in the agent's system prompt): the Output-4 per-marker biology note is *the skill reading that gene's own Tier-A evidence* — role framed by canonical status, and a specificity caveat flagged only when the numbers show the gene marks another lineage. One real live PMID or none.

A second dataset needs no code change: drop its raw inputs in `data/datasets/<id>/inputs/` and run. jazzPanda is never run — its output is a consumed input.

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
5. **Memory as scope-enforced lab notes.** When the biologist overrides a call, Panoscope captures a structured, git-tracked note — `{ claim, scope, basis, status, tension }` — cross-checks it against the literature, and keeps the biologist's call *with the disagreement visible*. Scope is enforced: a cluster- or dataset-scoped note never fires elsewhere, and the agent must cite a note to use it. It is a knowledge layer the lab owns; nothing is trained, nothing learns on its own.

## Skill + MCP setup

The interpretation logic ships as an installable, standalone skill: **[`skills/jazzpanda-markers/`](skills/jazzpanda-markers/SKILL.md)**. It encodes the panel-absence rule, the `glm_coef`-anchored confidence rubric, the holistic cross-cluster review, and the no-fabrication / real-citation discipline. It installs and runs independently of this app — point any agent at jazzPanda `top_result` output plus a panel list and it works.

Live citations come from a PubMed MCP server declared in [`.mcp.json`](.mcp.json) (`@cyanheads/pubmed-mcp-server`). To enable it, copy `.env.example` to `.env` and set `NCBI_API_KEY` and `NCBI_ADMIN_EMAIL` (the email registered to that key). Get a key at <https://www.ncbi.nlm.nih.gov/account/settings/>. The `pytest -m "not live"` suite needs none of this; one optional live test (`-m live`) exercises a real lookup when credentials are present.

## Data provenance

The demo dataset is the public **10x Genomics Xenium FFPE Human Breast Cancer (Rep 1)** sample — the 280-gene analyzed panel, 9 clusters (c1–c9). Marker statistics come from **published jazzPanda** run offline; the app reads that **precomputed output** and never runs jazzPanda live. All spatial views (cell map, hex-bin density, UMAP) are built from the same public data. This keeps the demo deterministic and reproducible, and keeps the confident floor intact: every number on screen has a checkable source.

## License

MIT. Everything shown is open source. jazzPanda is a published dependency under its own license; all interpretation-layer code in this repository is new work built for this project.
