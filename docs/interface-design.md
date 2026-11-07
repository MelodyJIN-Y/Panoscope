<!-- Generated 2026-09-11 by a verified multi-agent research pass.
     170 permission/price claims were extracted and adversarially re-checked; 21 were overturned.
     Items the verification could not confirm are flagged in the document's final section. -->

# Panoscope Interface Design Document

**Version 1.0 · target: Nature Methods interface contribution · repo `/Users/jin.m/Documents/Panoscope`**

---

## 0. The contribution, stated so a reviewer can test it

Panoscope's interface claim is **not** "a nicer spatial browser." It is:

> An interpretation layer can present *any* analysis method's output under one review contract, where every rendered pixel and every rendered sentence declares (a) what was measured, (b) what was measured and found absent, (c) what was never measured, and (d) who is responsible for the claim — and where the surface for a new method is **generated from a machine-readable declaration**, not hand-built.

Three consequences that drive every decision below:

1. **The review surface is a function of the task kind, not of the method.** ASSIGNMENT, CHARACTERISATION and COMPARISON get different headers because they answer different questions. A method (jazzPanda, limma-voom, ORA) registers *into* a task kind and contributes assumptions, columns and caveats — it does not get its own page.
2. **A surface that cannot declare its conditions does not render.** This is the visual analogue of the grounding gate. It is enforceable in code and testable in CI, which is what makes it a contribution rather than a style guide.
3. **The contract, not the widget toolkit, is the artifact.** Which is why the stack question resolves the way it does in §1.

Everything in this document is specified against code that exists. Line references are to the working tree at the time of writing.

---

## 1. Stack recommendation and migration path

### 1.1 The decision

**Target stack: FastAPI + React + TypeScript + Vite, with the compiled frontend shipped as static assets inside the wheel and served by FastAPI from `importlib.resources`.** `pip install panoscope && panoscope run` — no Node at install time, ever. Precedents that do exactly this: JupyterLab, Datasette, UCSC Cell Browser. Domain precedent for the *shape* (Python owns data and statistics, React/WebGL owns pixels): Vitessce, cellxgene.

**But the one-month window does not build it, and that is the point.**

The research streams disagree here, and the disagreement is real: stream 1 costs the React migration at four weeks; streams 2–5 specify roughly four weeks of *new interface* (status vocabulary, three-block tables, conditionality contract, testability plane, volcano gutter, concordance flag, decision ledger). Both do not fit. Ducking the question by "staying on Streamlit" concedes stream 1's strongest point — that a paper whose contribution is the interface cannot depend on `st-key-*` and `data-testid=` selectors it does not own (267 such hits across 21 files; `_kill_cross_page_bleed` at `app.py:219-278` reaches through `window.parent` from a zero-height iframe to mutate the host `<html>`).

The resolution is structural, not a compromise:

> **The UISpec (§3) *is* the migration path.** If the review surface is generated from a serializable declaration plus a resolved view-model, then porting to React is *writing a second renderer against a frozen contract* — not porting 8,610 lines of Streamlit. The one-month window builds the contract and its first renderer (Streamlit). The React renderer is a well-scoped, post-submission engineering task that the paper can describe as executed or in-flight without weakening the claim.

This is also the better *paper*. "We built a React app" is not a contribution. "We defined a review contract and demonstrated it is renderer-independent" is, and the second renderer is its proof. State it in the methods as a two-renderer design with one server-side view-model.

**Rejected, with reasons:**

- **Dash** — the callback graph is the slow part to write and Panoscope is callback-heavy (chat pane + draft-confirm card + three linked views + per-row selection + sign-off board). Worst choice under a deadline.
- **Panel/HoloViz** — best plotting story in the group (Datashader makes 158K points a non-question), but a four-layer abstraction tower is the opposite of what an agent-executed one-month build wants, and a Panel app is as pinned to its framework's serialization protocol as a Streamlit app is to its DOM.
- **Reflex** — downloads a Node/Bun toolchain on first run. Violates the packaging criterion outright.
- **Shiny for Python** — genuinely the runner-up: `ui.tags.div(class_="conv-title")` emits *your* HTML with *your* class names, so the ~110 `unsafe_allow_html=True` sites and `ui/theme.py`'s token block port with the least rework; `@reactive.calc` maps onto the cached readers almost one-to-one; `iSEE` is the exact domain precedent. If the React renderer slips twice, switch the second renderer to Shiny rather than abandoning the two-renderer claim. It loses the 158K-point rendering upgrade and leaves you inside a frontend you don't own.
- **Marimo** — wrong shape as the app, but **do ship a marimo notebook** that reproduces `data/studies/<id>/` and the paper's figures. One day's work, strong reproducibility appendix.

### 1.2 The API boundary, named

Two boundaries, not one. This matters: the second is what makes the renderer swappable.

**Boundary 1 — the data seam.**
`ui/data_access.py` (741 lines, 45 exports, 4 writes, cache shim at lines 65–95 that already degrades to an identity decorator headlessly) → **`panoscope/core/read.py`**, with `functools.lru_cache` replacing the Streamlit shim. Leave `ui/data_access.py` as a thin re-export so the 16 importing modules don't move on day one. `tests/test_data_access.py` already proves it runs with no server.

Fix the one leak while moving: `ui/summary.py:1142` and `:1328` call `pipeline.store.save_summary_edits` directly, bypassing the seam.

**Boundary 2 — the view contract (the load-bearing one).**

```
GET /api/view/{workflow_id}/{unit_id}  ->  { "spec": UISpec, "model": ViewModel }
```

`UISpec` is the workflow's static declaration (§3). `ViewModel` is the dataset-specific resolution: rows with their `MeasurementStatus` already assigned, numbers already formatted, `Conditionality` already composed, plot payload references already hashed. **No statistic, no band assignment, no binning, no normalization, and no rounding of a reported number happens in the renderer** — Streamlit or React. Add a CI check that greps the built bundle for arithmetic on `glm_coef` / `pearson` / `z` / `log2FC`. Cheap, real, and exactly the guarantee a Nature Methods reviewer will appreciate.

**Supporting tiers** (read-model, not an RPC mirror of the 45 `data_access` functions — chattiness is the failure mode):

| Tier | Endpoint | Payload |
|---|---|---|
| A bootstrap | `GET /api/dataset` | `manifest.json` (already carries `dataset_id, tissue, platform, panel_gene_count, n_cells, clusters, views_available, inputs{sha256}, artifacts`) + cluster key + `CLUSTER_ORDER` + `CONFIDENCE_BANDS` + cluster colour map. **`panel_contains` derives from the file — never hardcode 280 client-side.** (An earlier draft of this row cited `agent/config.py:PANEL_GENE_COUNT` as an existing informational constant. No such name exists, here or anywhere in the package; the rule stands, the example was wrong.) |
| B interpretation | `GET /api/view/{workflow}/{unit}` | the spec+model pair above. One call renders a whole page. |
| B′ set-level | `GET /api/units/{workflow}` | rail + summary board in one payload: `composed_verdicts()` + enrichment counts + review state + `override_info` + `pathway_themes`. |
| C viz frames | `GET /api/viz/{kind}/...` | **binary, not JSON** — Arrow IPC or raw little-endian Float32/Uint8 with a JSON header. 158K × (2×f32 + u8) ≈ 1.4 MB. Served with `ETag: <sha256 from manifest.artifacts>` and `Cache-Control: immutable`. |
| D agent (stateful) | `POST /api/chat/{unit}`, `POST /api/notes`, `POST /api/decisions`, `GET /api/export/*` | `PanoscopeAgent.chat` stays synchronous; declare the route `def` (not `async def`) so FastAPI threadpools it. The warm MCP session that `@st.cache_resource` hacks around becomes a proper `lifespan` singleton. `ui/report.py:build_report` is pure with no I/O and no Streamlit — it becomes three endpoints unchanged. |

One pipeline change is worth calling out as the highest-value item in tier C: **precompute a stable cell order once and record it in the manifest.** Then `GET /api/viz/expr/{gene}` is a Float32Array aligned by index to the cells array, and every feature-UMAP recolour is a ~600 KB buffer upload instead of a `merge` on `cell_id`.

### 1.3 Migration path

**Week 0 (2 days, non-negotiable, stack-independent).** This is the actual reproducibility fix and it must land first.
- Add `pyproject.toml` (hatchling) + a lockfile. Today there is **no packaging at all**: `requirements.txt` is 11 unpinned names, one of which is `streamlit>=1.59 # older Streamlit breaks the UI` — an unbounded lower pin on a package whose *undocumented DOM* the app depends on. This is a larger criterion-1 liability than the CSS.
- `ui/data_access.py` → `panoscope/core/read.py`, re-export shim left behind.
- `panoscope` console entry point (pointing at Streamlit for now).
- Surface `manifest.json` artifact sha256s in a UI footer and in the exported `.docx`/PDF: *"drawn from frame `sha256:ab3f…`"*. One line, stack-independent, materially strengthens the claim.
- Fix `ui/state.py` `DEFAULT_BIN_UM = 25` → `50`. See §4.3 — 25 µm renders at ~1 px/hex in the current 310 px panel, which is sub-legibility. (The constant is at line 46, not 53, and is still 25. An earlier draft attributed the 50 µm default to CLAUDE.md; CLAUDE.md says nothing about bin size, and the only claim it makes about the control is that it must never change a value.)

**Weeks 1–4 (the one-month window).** Build the UISpec contract and the Streamlit renderer, in the build order of §7. Streamlit stays the demo throughout.

**Weeks 5–8 (post-submission).** FastAPI over `core/read.py` (tiers A–D); OpenAPI → generated TS types so the boundary cannot drift; React renderer of the same `UISpec`; deck.gl for the two point clouds and the hex layer; then the cutover commit that deletes `app.py`, the Streamlit selectors in `ui/theme.py`, `_kill_cross_page_bleed`, and `streamlit` from the dependency list. Tag `streamlit-final` so any figures already in the manuscript stay reproducible.

**Plot migration, when it happens:**

| Current | Where | Migrates to |
|---|---|---|
| Cell map, 2× `Scattergl`, bg capped at `_MAX_BG_POINTS = 45_000` (`ui/spatial_stage.py:74`) | `_render_cell_map` | deck.gl `ScatterplotLayer` + `OrthographicView`; **`_MAX_BG_POINTS` is deleted** — "every cell is drawn" is an honesty win to state in the methods |
| Cluster/feature UMAP | `_render_umap` | same layer; recolour = swap one Float32Array |
| Hex density, `go.Heatmap` on a dense NaN grid (`_render_density_frame:626`) | `_render_density_frame` | deck.gl `PolygonLayer` at true µm width; the dense-array reconstruction is deleted; **binning stays precomputed in Python** |
| Dotplot (≤36 pts), volcano, heatmap | `_render_dotplot`, future | stay Plotly; Python keeps emitting `go.Figure`, ships `fig.to_dict()` |

Until then, the downsample is *stated* rather than deleted — see the conditionality line in §2.7.

---

## 2. The review surface per task kind

### 2.1 Eight invariants

These are the general claim; the three surfaces are its proof.

- **R1 — One shell, three fillings.** `unit rail │ evidence stage │ argument pane` (today `app.py:286`, `st.columns([216, 664, 474])`). Spatial memory transfers between workflows; only the contents change.
- **R2 — The header states the answer in the shape the task kind permits.** A label + band *only* where a label space exists. Otherwise a census with denominators. A band on a task with no label space is a category error, not a styling choice.
- **R3 — Every item carries a measurement status from one closed vocabulary**, redundantly encoded (glyph + cell texture + literal word) so it survives greyscale print and colour-vision deficiency.
- **R4 — No conditionality record, no plot.** The renderer refuses. This is the visual analogue of the grounding gate.
- **R5 — Plots render, they do not derive.** Every quantity in a figure exists in the view-model with file + sha256 + row provenance.
- **R6 — Every rendered sentence resolves to a claim with an authority and its formation conditions.** An inference with zero premises is not rendered.
- **R7 — Pin is a view-only verb on a typed subject**, governed by a declared capability matrix with two distinct honest failures: *unsupported kind* vs *missing artifact*.
- **R8 — Absence has a fixed shape.** Every evidence table is three blocks with counts in the block headers: detected / measured-and-absent / never measured.

### 2.2 The shared shell

```
┌─ UNIT RAIL ───────┬─ EVIDENCE STAGE ─────────────────────────┬─ ARGUMENT ──────┐
│ 216px             │ 664px                                     │ 474px           │
│                   │                                           │                 │
│ workflow tabs     │ ① HEADER      (R2: label+band | census |  │ ⑥ INTERPRETATION│
│  ▸ Markers        │                 confound checklist)       │   authority     │
│  ▸ Pathways       │ ─────────────────────────────────────────  │   spine + prose │
│  ▸ Contrasts      │ ② COVERAGE RIBBON  (the table's denoms)   │                 │
│                   │ ③ EVIDENCE TABLE   (R3/R8: 3 blocks)      │ ⑦ CLAIM CARD /  │
│ unit list         │ ─────────────────────────────────────────  │   PAPER DRAWER  │
│  ● c1 Tumor  VH   │ ④ PRIMARY PLOT     + conditionality line  │   (slide-over)  │
│  ● c2 Stromal H⚑  │ ⑤ DETAIL PLOTS     (driven by the PINNED  │                 │
│  …                │                     SUBJECT, small-mult.) │ ⑧ COMMIT BAR    │
│ ▸ cross-unit view │ ⑨ SAMPLE CONTEXT STRIP (multi-sample only)│   (sticky)      │
└───────────────────┴───────────────────────────────────────────┴─────────────────┘
```

The **unit** is per task kind: ASSIGNMENT unit = cluster, item = gene. CHARACTERISATION unit = cluster, item = gene set. COMPARISON unit = **contrast**, item = gene. The rail changes what it lists, never where it lives.

### 2.3 The measurement-status vocabulary (R3) — the headline claim, made visual

Five values, method-agnostic. Each registered method declares its mapping.

| glyph | cell texture | word | meaning | numbers shown? |
|---|---|---|---|---|
| ● solid | plain | `detected` | measured, tested, supports this unit | yes, full |
| ◐ half | plain, dimmed | `measured · not supporting` | measured and tested; did not support | **yes — this is weak evidence of absence** |
| ◌ dotted | plain, dimmed | `measured · untestable` | measured, test could not run at power | partial + the reason |
| ⊘ struck | **diagonal hatch** | `never measured` | not on the panel/assay | **none — hatched, never blank** |
| ⊗ | hatch + grey | `measured · QC-failed` | measured, discarded by QC | none + the filter |

**The one design decision that carries the scientific claim: a never-measured cell is hatched, not empty.** An empty cell reads as zero to every biologist who has ever looked at a heatmap. Hatching is the only encoding that reads as "no measurement exists here" in colour, in greyscale, and in print.

jazzPanda's mapping (extends `MarkerRole` at `agent/types.py:21`, which today has only `supports | expected_absent | off_panel`):

```
top_cluster == unit             → detected
top_cluster == another cluster  → measured · not supporting  ("assigned to c4, glm 2.1")
top_cluster == "NoSig"          → measured · not supporting  ("no cluster reached significance")
gene ∉ core.read.panel_names()  → never measured
neg-control / qv<20 probe       → measured · QC-failed
```

**R8, the three-block table**, identical across all three surfaces:

```
▾ SUPPORTING — 7 measured, detected
▾ MEASURED, NOT SUPPORTING — 3            ← weak evidence against
▾ NEVER MEASURED — 12 off panel           ← NOT evidence against
```

Ranking sorts *within* blocks. **A never-measured row can never be scrolled or filtered away by a top-N**, and it must be reachable by search — a biologist typing `COL1A1` or `HYPOXIA` must land on its row with its reason. Returning an empty search result is precisely the failure being fixed, and it is a testable interface requirement. Above the blocks sits the **coverage ribbon**: one 100%-width stacked bar with the four counts labelled — the table's own conditionality.

**Done.** When this was written, `ui/evidence_table.py` rendered no row for an off-panel gene and the headline claim was invisible in the headline surface — the single highest-value fix in this document. The surface rewrite shipped it: a never-measured row now renders with its gene, its reason and its frame, and on c1 the table carries 84 observed, 147 measured-not-here and 50 never-measured. The requirement above (never scrolled or filtered away, reachable by search) is what keeps it there.

### 2.4 Conditionality (R4)

Generated, never typed. One struct, one two-line mono caption under every plot, hover-expands to the full record.

```
transcript detections · CDK1 · 50 µm hex bins, area-normalised tx/µm², √ colour
280-gene panel · sample1/hb1 · qv<20 + controls removed · pre-segmentation, not per-cell   ⓘ
```

`render_panel(fig, cond)` **raises** if any of `drawn / denominator / transform / not_this` is empty. A panel that cannot say what it is conditioned on does not appear. This is a five-line function and it is the cheapest defensible thing in the whole design.

### 2.5 ASSIGNMENT — one label from a label space

**Header** — the current `ui/verdict_header.py` is right except for one omission: a label drawn from a label space is only meaningful relative to that space, so the header must name the runner-up and the item that separated them. `agent/discriminate.py` already computes this (`Discrimination`, `settleable_on_panel`, `settle_line`), so it is deterministic Tier-A content, not model output.

```
CLUSTER 1 · 4,812 cells                                          [agreed ✓]
Tumor                                    ⟨Very High⟩  ⌄
driven by  KRT8 17.99 · ERBB2 12.4 · EPCAM 9.8
next best  Myoepithelial — separated by KRT14 (2.1 here, assigned to c4)
band from  glm_coef ≥ 10 · corroborated by pearson 0.62 · 0 demotions   ⌄
```

`⟨Very High⟩ ⌄` opens the rubric card: the edges from `agent/config.py:CONFIDENCE_BANDS` (≥10 / ≥5 / ≥2.5 / ≥1), this unit's position on them, `ClusterVerdict.band_basis`, and the full `ClusterVerdict.demotions` audit trail. The band stops being an opaque adjective.

**Evidence table**

```
 st  GENE            glm_coef ▏band position   pearson  spec.   BIOLOGY · relevance (cited)
─────────────────────────────────────────────────────────────────────────────────────────
 ●   KRT8    canon   17.99  ▏ ▁▂▃▄█ ≥10        0.62     ▮▯      Luminal keratin, epithelial
                                                                identity …  📄 Baker 2019 ▸
 ●   ERBB2           12.40  ▏ ▁▂▃▄█            0.58     ▮▯      …
 ●   CDK1             6.10  ▏ ▁▂▃▮  ≥5         0.41     ▮▮ ⚠    also marks c3 (max_gc_corr
                                                                0.52 > pearson 0.41)
─── MEASURED, NOT SUPPORTING — 3 ─────────────────────────────────────────────────────────
 ◐   MKI67            4.80  ▏ assigned to c3   0.19     —       measured; jazzPanda assigns
                                                                its spatial signal to c3
 ◐   TP53             —     ▏ NoSig            —        —       measured; no cluster reached
                                                                significance
─── NEVER MEASURED — 12 off panel ────────────────────────────────────────────────────────
 ⊘   COL1A1   canon  ▨▨▨▨▨▨▨▨ not on panel ▨▨▨▨▨▨▨▨    canonical for Stromal; never measured
                                                        on this 280-gene panel. Its absence
                                                        is not evidence about c1.
 ⊘   PTPRC    canon  ▨▨▨▨▨▨▨▨ not on panel ▨▨▨▨▨▨▨▨    …
```

`▏▁▂▃▄█` is the **band-position micro-bar** — where this gene's `glm_coef` sits against the band edges. `▮▯` is the **specificity pair**: `pearson` (this cluster) vs `max_gc_corr` (best other cluster); the existing amber "also marks another cluster" chip promoted from a badge to a readable two-tick encoding.

**Primary plot** — cell map, with a napari-style layer stack (all cells / this unit / pinned subject), each layer contributing its fragment to the conditionality line. Always state `cell centroids, NOT segmentation boundaries` — every reader assumes boundaries.

**Detail plots (pinned gene)** — the existing density-hexbin │ feature-UMAP pair, plus a **discrimination lollipop** (`pearson` here vs `max_gc_corr` elsewhere), plus the cross-cluster dotplot.

**Cross-unit overview** — (1) marker × cluster matrix with **hatched cells for never-measured**, which a conventional heatmap cannot express; (2) a 9×9 **shared-evidence graph** (count of shared driving markers) — the skill's mandated holistic review made visual; (3) the decision ledger, exportable as the CSV contract.

### 2.6 CHARACTERISATION — no label space, therefore no unit band

This replaces the construction at `ui/enrichment_table.py:178-182`, which today does `from ui.verdict_header import _idline` and emits `<h1>{cell_type}</h1><span class="cf">{ce.confidence} enrichment</span>`. **That is a unit-level confidence band on a task with no label space.** It is the clearest existing instance of R2 being violated by inheritance.

```
CLUSTER 1 · 4,812 cells   ·   Tumor  (call imported from the marker workflow; not re-derived here)

PROGRAMS   4 enriched   3 suggestive   43 untestable      of 50 Hallmark sets scored
TOP        G2M Checkpoint   z 4.2   q 3e-04   on 18 of 200 set genes present (9% coverage)
SCOPE      panel-scoped, not genome-wide · competitive z against the 262 other panel genes
GATES      q<0.05 · ≥3 leading-edge genes · ≥3 set genes on panel                          ⌄
```

Four properties the assignment header does not have and must not lend: no unit-level band; every count carries its denominator; the cell-type call appears as *attributed imported context*; the gate that produced the counts is stated and clickable.

`ClusterEnrichment.confidence` / `.confidence_score` stay in `agent/types.py` (artifacts and tests read them) but are marked deprecated-for-display, and the census `HeaderSpec` (§3) simply **has no band slot**, so the field is structurally undrawable. Drop it from the artifact at the next `pipeline_version` bump.

**Evidence table** — three absences, distinguished:

```
 st  PROGRAM              z     q       COVERAGE K/N          LEADING EDGE   REL. TO CALL
──────────────────────────────────────────────────────────────────────────────────────────
 ●   G2M Checkpoint      4.20  3e-04   ▓▓▓░░░░░░ 18/200 9%    6 ·CDK1 CCNB1  concordant
                                                                 TOP2A …     proliferative 📄
 ●   Epithelial-Mesen.   3.10  8e-03   ▓▓▓▓░░░░░ 22/200 11%   5 ·VIM FN1 …   concordant
 ◐   Interferon Gamma    2.05  0.18    ▓▓░░░░░░░ 14/200 7%    4 ·…           ⚠ cross-lineage
      suggestive · re-check                                                   — a tension, never
                                                                              a re-typing; also
                                                                              enriched in 6 of 9
─── MEASURED, NOT ENRICHED — 3 ───────────────────────────────────────────────────────────
 ◐   Apoptosis           0.40  0.71    ▓▓▓░░░░░░ 19/200 10%   1              tested; did not
                                                                              clear the gate
─── UNTESTABLE ON THIS PANEL — 43   [search: ______]  ▾ ──────────────────────────────────
 ◌   Hypoxia            ▨▨▨▨  ▨▨▨▨    ▓░░░░░░░░  2/200  1%   —              only 2 of 200 set
                                                                              genes were ever
                                                                              measured. We cannot
                                                                              say whether hypoxia
                                                                              is active here.
```

`PathwayEvidence` already carries `set_size_full`, `panel_hits`, `panel_coverage`, `n_leading_edge`, `panel_scope_caveat` — everything needed. Today `K/N` is demoted to a `title=` tooltip on `cov 18/200`. **Promote it to a mark**: a coverage bar whose *unfilled* portion is the 182 of 200 genes never measured, with that sentence on hover.

**Primary plot — the testability plane, not a bar chart.** A `-log10(q)` bar chart hides `K` and cannot represent an untestable set at all.

```
coverage
 K/N %
  15 │                                        ● EMT 22/200
     │                          ● G2M 18/200
  10 │        ◐ IFN-γ 14/200          ○ Apoptosis 19/200
     │ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  K ≥ 3 gate
   1 │▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨▨
     │▨ 43 sets cannot be tested on this panel — click to list ▨
     └────────────┬──────────────┬──────────────┬────────────
       0          2              4              6     competitive z
                  ╎ q = 0.05                ● enriched  ◐ suggestive  ○ tested, not enriched
50 MSigDB Hallmark sets scored against 280 panel genes · jazzPanda competitive z vs the other
panel genes · BH-adjusted q · gates K≥3, LE≥3, q<0.05 drawn · panel-scoped, not genome-wide ⓘ
```

You can see at a glance which programs are excluded by *coverage* rather than by *biology*.

**Cross-unit overview — a categorical tier matrix**, glyph per cell (`● ◐ ○ ◌ ⊘`) with glyph *size* encoding z. Deliberately **not** a continuous colour heatmap: a continuous ramp makes an untestable cell look like a low-z cell, which is the exact conflation the workflow exists to prevent. Per-row recurrence count (`pathway_themes` / `themes.recurring` exists), because a program enriched in 6 of 9 units is a panel-design or shared-signal fact, not a cluster property.

### 2.7 COMPARISON — "does this result make sense?"

*No DE artifact exists in the repo. Numbers below are illustrative; the shapes are the spec.*

**Rail lists contrasts, each showing its design** — because the design determines whether the result is worth opening:

```
CONTRASTS
● c1 Tumor · treated ~ untreated        3 v 3 donors   ⚑ composition
○ c2 Stromal · treated ~ untreated      3 v 3 donors
○ Tumor ~ Stroma (within sample1)       1 section      ⚑ no replication
+ define contrast
```

**Header — a checklist of confounds, not a verdict.** These four checks *are* the interpretive question for a comparison, so they are the header rather than buried caveats.

```
CONTRAST   c1 Tumor · treated (n=3 donors, 12,404 cells) ~ untreated (n=3, 9,881 cells)

RESULTS    19 up · 7 down   at q<0.05        of 271 genes testable (280 panel − 9 zero-count)
SCOPE      280 of ~20,000 transcriptome genes · panel-scoped · limma-voom on donor pseudobulk

CHECKS   ▸ panel scoping   ⚑  37 canonical genes a reader would expect were never measured
         ▸ replication     ✓  3 v 3 biological donors; pseudobulk, not cell-level n
         ▸ composition     ⚑  Tumor fraction 41% → 57%; 6 of 19 up-genes track it
         ▸ cross-method    ✓  12 of 19 up-genes are also c1 jazzPanda spatial markers
```

Each chip has a state (`✓ ok` / `⚑ flagged` / `— not assessable`) and opens its claim card. The composition shift is computable deterministically from labels Panoscope already holds — it is measurement, not inference. Cross-method coherence costs nothing because both artifacts already exist.

**Evidence table** — four blocks here, because DE has a filter that assignment does not:

```
 st  GENE      log2FC    q        %cells A→B    CONF   SPATIAL (jazzPanda)   BIOLOGY (cited)
──────────────────────────────────────────────────────────────────────────────────────────
 ●▲  ERBB2     +2.10   4e-06    38% → 71%      ·      glm 12.4 in c1 ✓      …  📄
 ●▲  CDK1      +1.40   2e-03    22% → 39%      ⚑comp  glm 6.1 in c1 ✓       direction tracks
                                                                             the 41→57% shift
 ●▼  VIM       −1.80   9e-04    55% → 21%      ·      assigned to c2        ⚠ spatial signal
                                                                             sits in Stromal
─── MEASURED, NOT SIGNIFICANT — 245 ──────────────────────────────────────────────────────
 ◐   EPCAM     +0.10   0.88     66% → 67%      ·      glm 9.8 in c1         tested; no change
─── MEASURED, NOT TESTABLE — 9 ───────────────────────────────────────────────────────────
 ◌   IL2       ▨▨▨▨    ▨▨▨▨     0% → 4%        ·      —                     0 counts in
                                                                             untreated (n=3):
                                                                             filtered before
                                                                             testing. This is
                                                                             NOT "no change" —
                                                                             it may be the
                                                                             largest effect here
─── NEVER MEASURED — 37 off panel ────────────────────────────────────────────────────────
 ⊘   MKI67     ▨▨▨▨▨▨▨▨▨ not on panel ▨▨▨▨▨▨▨▨▨                             absence says
 ⊘   PTPRC     ▨▨▨▨▨▨▨▨▨ not on panel ▨▨▨▨▨▨▨▨▨                             nothing here
```

The `MEASURED, NOT TESTABLE` block is the block a volcano structurally *cannot* contain: a gene with zero counts in one arm is dropped by the filter before the plotting frame exists, and it is often the largest real effect. Surfacing it is a substantive contribution, not cosmetics.

**Primary plot — the volcano with a gutter:**

```
                    │ NOT PLOTTABLE │                -log10 q
                    ├───────────────┤        6 ┤              ● ERBB2
  measured, not     │      9        │          │        ● ●  ●
  testable  ◌       │  ▏▏▏▏▏▏▏▏▏    │        4 ┤     ●      ○ CDK1 (composition-linked)
                    │  IL2 IL10 …   │          │  ·   ·  ·  ·
                    ├───────────────┤ ─ ─ ─ ─ 2 ┤· ·· ·· ·· ·· ·· ·· ··  q = 0.05
  never measured ⊘  │     37        │          │ ·  ·   ·  ·   ·   ·
                    │  ▨▨▨▨▨▨▨▨▨    │        0 ┤
                    │  MKI67 PTPRC  │          └──┬─────┬─────┬─────┬─────┬──
                    │  COL1A1 …     │            -2    -1     0    +1    +2   log2 FC
                    └───────────────┘
271 of 280 panel genes tested; 280 of ~20,000 transcriptome genes on panel — panel-scoped,
NOT a whole-transcriptome volcano · limma-voom, donor pseudobulk, 3 v 3 · BH q · hollow points
are composition-linked · the left gutter holds the 46 genes these axes cannot represent    ⓘ
```

The gutter is part of the same figure, at the same scale, with counts and names.

**Detail plots** — (1) **replicate strip**: per-donor pseudobulk for the pinned gene, one dot per replicate, arms separated. The single most important and most commonly absent DE review plot: it shows whether 3v3 is consistent or one donor carries the effect. (2) **Paired arm tissue maps** on a *locked shared colour scale* — **never a difference map**, because a difference is a derived quantity (R5); a difference map renders only if the pipeline emitted a difference artifact with its own conditionality. (3) **Composition bar** — the confound drawn next to the effect it could explain. (4) **Cross-method strip** — the pinned gene's `glm_coef`/`pearson` in this unit.

### 2.8 The three existing spatial views, retrofitted

Keep all three; three changes each.

- **Density hexbin.** (a) The bin radio moves out of the free-floating control row (`_render_density_controls`, `ui/spatial_stage.py:835`) and **attaches to the panel's conditionality line**, so you cannot change the bin without watching the string change. (b) **Scale lock.** Today `cmax = float(dens_sqrt.max())` per panel (`:647`), so two genes side by side both saturate and look equally abundant. Add a lock that forces a shared `zmax` across small-multiples, and make the line read either `shared scale, max 0.084 tx/µm² from ERBB2` or `independent scale — panels are NOT comparable`. (c) Keep area-normalisation and √ colour exactly as they are; state both.
- **Cell map.** Layer stack; always state `centroids, not segmentation boundaries`; **and state the downsample**: `background thinned to 45,000 of 153,567 for rendering; the selected unit is drawn in full`. When deck.gl lands, that clause is deleted and the change is a methods-section sentence.
- **UMAP.** Standing caveat in its conditionality: `UMAP layout, 158,379 cells, parameters from the prep step — a layout, not a measurement; distances between clusters are not interpretable.` The "clean island or bleeding" read stays visual and is labelled visual, unless the pipeline stores a neighbour-purity number — then it becomes a claim with a number.

### 2.9 Click-back: every sentence resolves to a claim (R6)

Today provenance is per-turn (`GroundingSidecar{numbers, markers, pmids, notes_used}` plus one condensed `Sources` line) and PMIDs are spliced inline by `_PMID_RE`, putting `PMID:31142839` in the middle of prose.

**Five authorities, each with one hard rule:** `measurement` (resolves to a method-output cell with file + hash + row; the model can never author one) · `curated` (panel TSV, cluster key, MSigDB definition; has a version, not a p-value) · `lab` (resolves to a `Note` and **must render its attached `Tension`** whenever used) · `literature` (resolves to a `Citation` with `is_real=True` and `fetched_at`) · `inference` (**the only authority the model authors, and the only one required to declare non-empty `premises`; an inference with zero premises is not rendered**).

**Presentation — the authority spine.** Authority lives in a ~1.5ch left gutter, one mark per sentence, never inside the prose:

```
  ▌  KRT8 is the strongest spatial marker for this cluster (glm_coef 17.99),
     with ERBB2 and EPCAM behind it.
  ○  Luminal keratin expression of this kind is characteristic of ductal
     carcinoma epithelium.  Baker 2019 ▸
  △  Taken together the call is consistent, and the Very High band rests on
     three markers above the ≥10 edge rather than on one.
  ◆  Your note from 12 June — "we do not trust KRT8 in FFPE" — cuts against
     the top marker here; 2 papers agree with you, 1 dissents.  note:7a1c ▸

  ▌ measurement   ▏ curated   ○ literature   ◆ lab   △ inference
```

Prose reads as prose. The spine is a **scannable epistemic profile** — a paragraph of three `△` resting on one `○` is visible before you read a word, which is exactly the judgement a reviewer needs. Marks collapse over consecutive sentences sharing a claim. Numbers stay inline as value chips with a dotted underline; clicking one opens the measurement claim card. Hovering a sentence brushes its evidence rows and plot marks. Greyscale-safe.

**Enforcement is split, because dropping sentences silently changes meaning.** A falsified *measurement* claim poisons its paragraph → reject the whole turn and fall back deterministically (current behaviour in `agent/loop.py`). An *inference* with failed premises, or an unresolvable PMID → drop that sentence and render a visible stub:

```
⊘ 1 sentence removed — cited PMID did not resolve through the connector
```

The stub matters for the paper: a reader can **see the gate acting**, rather than trusting that it did.

---

## 3. The UISpec contract

`panoscope/ui_spec/spec.py` — Python 3.11+, all frozen. This is the declaration a **workflow** makes so its review surface is generated rather than hand-built.

```python
"""UISpec — the machine-readable declaration a workflow makes about its review surface.

A workflow owns the interpretive question AND the surface. A method registers into a
workflow and contributes its assumptions, its status rule, and extra columns. The
renderer (Streamlit today, React next) consumes UISpec + ViewModel and draws; it
never decides what a task kind is allowed to claim.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

# --------------------------------------------------------------------------- #
# Closed vocabularies
# --------------------------------------------------------------------------- #
TaskKind = Literal["assignment", "characterisation", "comparison", "unknown"]
UnitKind = Literal["cluster", "contrast", "region", "sample", "unknown"]
ItemKind = Literal["gene", "gene_set", "celltype_pair", "unknown"]

MeasurementStatus = Literal[
    "detected",            # ● measured, tested, supports this unit
    "not_supporting",      # ◐ measured and tested; did not support  (weak evidence of absence)
    "untestable",          # ◌ measured, but the test could not run at power
    "never_measured",      # ⊘ not on the panel/assay  (NOT evidence of absence)
    "qc_failed",           # ⊗ measured, discarded by QC
    "unknown",             # ? the method declared no rule  (degradation only)
]

HeaderKind = Literal["label_band", "census", "confound_checklist", "provenance_only"]
ColumnKind = Literal[
    "text", "number", "glyph", "band_microbar", "coverage_bar",
    "specificity_pair", "sparkstrip", "direction", "citation", "note_chip",
]
PlotKind = Literal[
    "cell_map", "density_hex", "umap", "feature_umap", "dotplot",
    "testability_plane", "volcano_gutter", "replicate_strip",
    "composition_bar", "paired_arm_maps", "tier_matrix", "shared_evidence_graph",
]
Authority = Literal["measurement", "curated", "lab", "literature", "inference"]


# --------------------------------------------------------------------------- #
# R4: no conditionality record, no plot. Enforced at construction.
# --------------------------------------------------------------------------- #
_REQUIRED_COND = ("drawn", "denominator", "transform", "not_this")


@dataclass(frozen=True)
class Conditionality:
    """What a figure is conditioned on. Generated from the manifest + view-model.

    Four fields are mandatory and non-empty. ``render_panel`` refuses to draw a
    figure whose Conditionality cannot be constructed, which is the visual
    analogue of the grounding gate discarding an ungrounded sentence.
    """
    drawn: str          # "transcript detections, pre-segmentation"
    denominator: str    # "280 of ~20,000 transcriptome genes on panel"
    transform: str      # "50 µm hex bins, area-normalised (tx/µm²), √ colour"
    not_this: str       # "not per-cell expression; not a statistic"
    unit_scope: str = ""    # "sample1/hb1 · 158,379 cells · c1 = 4,812"
    exclusions: str = ""    # "qv<20 and negative-control probes removed"
    provenance: str = ""    # "density/CDK1_50um.parquet sha256 9f3a… · run 2026-07-08"
    scale: str = ""         # "shared scale, max 0.084 tx/µm² from ERBB2" | "independent scale"
    rendering: str = ""     # "background thinned to 45,000 of 153,567 for rendering"

    def __post_init__(self) -> None:
        missing = [f for f in _REQUIRED_COND if not getattr(self, f).strip()]
        if missing:
            raise ValueError(
                f"Conditionality is incomplete: {', '.join(missing)}. "
                "A panel that cannot state what it is conditioned on does not render."
            )

    def caption(self) -> tuple[str, str]:
        """The two-line mono caption drawn under the panel."""
        line1 = " · ".join(p for p in (self.drawn, self.transform, self.scale) if p)
        line2 = " · ".join(
            p for p in (self.denominator, self.unit_scope, self.exclusions,
                        self.rendering, self.not_this) if p
        )
        return line1, line2


# --------------------------------------------------------------------------- #
# Header — R2. The variants are the whole point: a census header HAS NO BAND SLOT,
# so a band on a task with no label space is structurally undrawable.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelBandHeader:
    kind: Literal["label_band"] = "label_band"
    label_space: str = ""            # "cluster_key.json (9 cell types)"
    band_field: str = "confidence"
    band_score_field: str = "confidence_score"
    band_edges_ref: str = "panoscope.core.verdict.CONFIDENCE_BANDS"
    band_audit_fields: tuple[str, ...] = ("band_basis", "demotions")
    driver_field: str = "key_markers"
    verify_field: str = "verify"
    require_runner_up: bool = True   # the rejected alternative + its separating item
    runner_up_source: str = "agent.discriminate.discriminate"


@dataclass(frozen=True)
class CensusLine:
    label: str                       # "PROGRAMS"
    counts: tuple[tuple[str, str], ...]   # (("enriched","n_enriched"), ("untestable","n_untestable"))
    denominator_field: str           # "n_scored" -> "of 50 Hallmark sets scored"
    denominator_template: str = "of {value} scored"


@dataclass(frozen=True)
class CensusHeader:
    """No unit-level band. Every count carries its denominator."""
    kind: Literal["census"] = "census"
    lines: tuple[CensusLine, ...] = ()
    top_item_template: str = ""      # "{gene_set}  z {score}  q {q_value}  on {panel_hits} of {set_size_full} set genes present ({panel_coverage:.0%} coverage)"
    scope_line_field: str = "panel_scope_caveat"
    gates_field: str = "gates"       # rendered as a clickable, expandable record
    imported_context_fields: tuple[str, ...] = ("cell_type",)   # attributed, never re-derived
    imported_context_note: str = "call imported from the marker workflow; not re-derived here"


@dataclass(frozen=True)
class ConfoundCheck:
    key: str                         # "composition"
    label: str                       # "composition"
    state_field: str                 # "checks.composition.state" -> "ok"|"flagged"|"not_assessable"
    detail_field: str                # "checks.composition.detail"
    claim_id_field: str = ""         # opens this claim card


@dataclass(frozen=True)
class ConfoundChecklistHeader:
    kind: Literal["confound_checklist"] = "confound_checklist"
    design_template: str = ""        # "{arm_a} (n={n_a} donors, {cells_a} cells) ~ {arm_b} …"
    results_template: str = ""       # "{n_up} up · {n_down} down at q<{alpha}"
    testable_denominator_template: str = ""   # "of {n_testable} genes testable ({n_panel} panel − {n_filtered} zero-count)"
    scope_line_field: str = "scope_caveat"
    checks: tuple[ConfoundCheck, ...] = ()


@dataclass(frozen=True)
class ProvenanceOnlyHeader:
    """Degradation header for an unregistered analysis type. Makes no claim at all."""
    kind: Literal["provenance_only"] = "provenance_only"
    banner: str = (
        "Unregistered analysis type — Panoscope is showing this artifact as a table "
        "with its provenance. No confidence band, no status assignment and no "
        "interpretation are available for it."
    )


HeaderSpec = (
    LabelBandHeader | CensusHeader | ConfoundChecklistHeader | ProvenanceOnlyHeader
)


# --------------------------------------------------------------------------- #
# Table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ColumnSpec:
    key: str                         # field on the resolved item row
    label: str
    kind: ColumnKind
    unit: str = ""                   # "glm_coef", "z", "log2FC", "tx/µm²"
    precision: int = 2
    align: Literal["left", "right"] = "left"
    width: int = 0                   # 0 = renderer decides
    # R3/R8: when the row's status hides a number, the cell is HATCHED, never blank.
    hatch_when: tuple[MeasurementStatus, ...] = ("never_measured", "qc_failed")
    blank_is_forbidden: bool = True
    ref_fields: tuple[str, ...] = () # extra fields carried for the claim card
    sortable: bool = True


@dataclass(frozen=True)
class BlockSpec:
    """One of the three (or four) absence blocks. R8."""
    key: str
    label_template: str              # "NEVER MEASURED — {count} off panel"
    statuses: tuple[MeasurementStatus, ...]
    sort_by: str = ""                # sorting happens WITHIN a block, never across
    sort_desc: bool = True
    collapsed_by_default: bool = False
    searchable: bool = True          # a never-measured item MUST be findable by search
    never_truncate: bool = True      # a top-N can never hide an absence block
    empty_text: str = ""             # shown when count == 0; the block header still renders


@dataclass(frozen=True)
class CoverageRibbonSpec:
    """The table's own conditionality: one 100%-width stacked bar, counts labelled."""
    enabled: bool = True
    order: tuple[MeasurementStatus, ...] = (
        "detected", "not_supporting", "untestable", "qc_failed", "never_measured",
    )


# --------------------------------------------------------------------------- #
# Plots + the pin capability matrix (R5, R7)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlotSpec:
    key: str
    kind: PlotKind
    title: str
    accepts_subject_kinds: tuple[ItemKind, ...] = ()
    required_views: tuple[str, ...] = ()      # keys in manifest["views_available"]
    required_artifacts: tuple[str, ...] = ()  # relative paths hashed in manifest["artifacts"]
    conditionality_builder: str = ""          # dotted path to a fn -> Conditionality
    scale_lockable: bool = False
    derives_nothing: bool = True              # asserted; the grounding test checks it
    # Two DISTINCT honest failures, never one message:
    unsupported_kind_text: str = "{subject}: this view does not accept a {kind}"
    missing_artifact_text: str = "{subject}: {artifact} not precomputed for this dataset"


@dataclass(frozen=True)
class PinnedSubject:
    kind: ItemKind
    id: str                          # "SLC1A3" | "HALLMARK_G2M_CHECKPOINT" | "c1:treated~untreated"
    unit: str                        # where it was pinned from
    workflow: str
    members: tuple[str, ...] = ()    # leading edge, or the pair's two labels


# --------------------------------------------------------------------------- #
# The workflow declaration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UISpec:
    workflow_id: str                 # "markers" | "pathways" | "contrasts"
    title: str
    task_kind: TaskKind
    unit_kind: UnitKind
    item_kind: ItemKind
    interpretive_question: str       # the sentence the workflow exists to answer
    header: HeaderSpec
    columns: tuple[ColumnSpec, ...]
    blocks: tuple[BlockSpec, ...]
    coverage_ribbon: CoverageRibbonSpec = CoverageRibbonSpec()
    primary_plot: Optional[PlotSpec] = None
    detail_plots: tuple[PlotSpec, ...] = ()
    overview_plots: tuple[PlotSpec, ...] = ()
    pin_kinds: tuple[ItemKind, ...] = ()
    decision_states: tuple[str, ...] = ("open", "accepted", "overridden", "deferred")
    csv_columns: tuple[str, ...] = ()
    spec_version: str = "1.0"

    def __post_init__(self) -> None:
        # R2, enforced structurally rather than by review.
        if self.task_kind == "characterisation" and self.header.kind == "label_band":
            raise ValueError(
                "characterisation has no label space; a unit-level confidence band "
                "is a category error. Use CensusHeader."
            )
        if self.task_kind == "assignment" and self.header.kind != "label_band":
            raise ValueError("assignment must declare its label space and band rubric.")
        # R8, enforced: every surface must be able to say 'never measured'.
        if self.task_kind != "unknown":
            covered = {s for b in self.blocks for s in b.statuses}
            if "never_measured" not in covered:
                raise ValueError(
                    f"{self.workflow_id}: no block renders 'never_measured'. The "
                    "panel-absence rule must have somewhere to appear."
                )


# --------------------------------------------------------------------------- #
# The method declaration (registers INTO a workflow)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MethodSpec:
    method_id: str                   # "jazzpanda_spatial_glm"
    workflow_id: str
    display_name: str
    assumptions: tuple[str, ...]     # rendered in the header's expandable record
    outputs: tuple[str, ...]         # "glm_coef (spatial GLM Estimate)", "pearson", …
    caveats: tuple[str, ...]         # "multivariate spatial GLM, not per-cluster DE"
    # The status rule runs SERVER-SIDE only. The wire carries resolved statuses, so
    # no renderer ever decides what counts as measured.
    status_rule: Callable[[object, object], MeasurementStatus] = field(repr=False, default=None)  # type: ignore[assignment]
    extra_columns: tuple[ColumnSpec, ...] = ()
    citation: str = ""               # the method's own paper


# --------------------------------------------------------------------------- #
# What crosses the wire (the renderer's only input)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ItemRow:
    item_id: str
    status: MeasurementStatus
    status_reason: str               # "assigned to c3" | "not on panel" | "0 counts in untreated (n=3)"
    values: tuple[tuple[str, str], ...]   # (column key, ALREADY-FORMATTED string)
    raw: tuple[tuple[str, float], ...]    # (stat, value) for the claim card only
    claim_ids: tuple[str, ...] = ()
    note_ids: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()      # "spillover", "composition_linked", "cross_lineage"


@dataclass(frozen=True)
class ViewModel:
    workflow_id: str
    unit_id: str
    dataset_id: str
    header_values: tuple[tuple[str, str], ...]
    rows: tuple[ItemRow, ...]
    counts: tuple[tuple[MeasurementStatus, int], ...]
    conditionalities: tuple[tuple[str, Conditionality], ...]   # plot key -> record
    manifest_shas: tuple[tuple[str, str], ...]
    degraded: bool = False
    degradation_reason: str = ""
```

### 3.1 The registry, and graceful degradation

`panoscope/ui_spec/registry.py`:

```python
_WORKFLOWS: dict[str, UISpec] = {}
_METHODS: dict[str, MethodSpec] = {}


def register_workflow(spec: UISpec) -> None: ...
def register_method(spec: MethodSpec) -> None: ...


def resolve(workflow_id: str, artifact: "ArtifactHandle") -> UISpec:
    """Never raises. An unregistered analysis type degrades to LESS claim, never more."""
    spec = _WORKFLOWS.get(workflow_id)
    if spec is not None:
        return spec
    return degraded_spec(artifact)


def degraded_spec(artifact: "ArtifactHandle") -> UISpec:
    """The fallback surface for an analysis type nobody registered.

    Rules, in order of importance:
      1. No header claim.  ProvenanceOnlyHeader: a banner, the file, its sha256,
         its row count. No band. No census. No checks.
      2. No status assignment.  Every row is `unknown`, rendered `? status not
         declared by this method`. Nothing is called detected, and — critically —
         nothing is called absent, because an unregistered method has told us
         nothing about what it measured.
      3. NO PLOTS.  A plot needs a Conditionality; nobody declared one; R4 applies.
         The stage shows the empty-panel placeholder with the reason.
      4. No pin, no chat grounding on these rows, no sign-off, no CSV export.
         The commit bar renders disabled with "this analysis type is not registered;
         it cannot be signed off."
      5. Columns are read off the artifact's own schema as `ColumnKind.text`,
         verbatim, unrounded — the renderer must not reformat a number it does not
         understand.
    """
    cols = tuple(
        ColumnSpec(key=c, label=c, kind="text", blank_is_forbidden=False)
        for c in artifact.columns
    )
    return UISpec(
        workflow_id=f"unregistered:{artifact.name}",
        title=artifact.name,
        task_kind="unknown",
        unit_kind="unknown",
        item_kind="unknown",
        interpretive_question="(not declared)",
        header=ProvenanceOnlyHeader(),
        columns=cols,
        blocks=(BlockSpec(key="all", label_template="ROWS — {count}",
                          statuses=("unknown",), searchable=True),),
        coverage_ribbon=CoverageRibbonSpec(enabled=False),
        primary_plot=None,
        detail_plots=(),
        pin_kinds=(),
        decision_states=("open",),
        csv_columns=(),
    )
```

**The degradation principle, stated for the paper:** an unregistered analysis type degrades toward *making no claim*, never toward *reusing another task kind's claim*. The current `ui/enrichment_table.py` is the counter-example — it degraded upward by importing the assignment header and inheriting a confidence band it had no right to. `UISpec.__post_init__` now makes that construction impossible.

### 3.2 The registrations that exist

```python
# panoscope/workflows/markers.py
register_workflow(UISpec(
    workflow_id="markers", title="Marker genes", task_kind="assignment",
    unit_kind="cluster", item_kind="gene",
    interpretive_question="Which cell type is this cluster, and how sure can we be?",
    header=LabelBandHeader(label_space="cluster_key.json (9 cell types)"),
    columns=(
        ColumnSpec("gene", "GENE", "text"),
        ColumnSpec("glm_coef", "glm_coef", "number", unit="glm_coef", align="right"),
        ColumnSpec("band_position", "band", "band_microbar"),
        ColumnSpec("pearson", "pearson", "number", align="right"),
        ColumnSpec("specificity", "spec.", "specificity_pair",
                   ref_fields=("pearson", "max_gc_corr")),
        ColumnSpec("biology", "BIOLOGY · relevance (cited)", "citation"),
    ),
    blocks=(
        BlockSpec("supporting", "SUPPORTING — {count} measured, detected",
                  ("detected",), sort_by="glm_coef"),
        BlockSpec("not_supporting", "MEASURED, NOT SUPPORTING — {count}",
                  ("not_supporting",), sort_by="glm_coef"),
        BlockSpec("never", "NEVER MEASURED — {count} off panel",
                  ("never_measured", "qc_failed"), sort_by="gene", sort_desc=False),
    ),
    primary_plot=PlotSpec("cell_map", "cell_map", "Cell map",
                          accepts_subject_kinds=("gene",), required_views=("cell_map",)),
    detail_plots=(
        PlotSpec("density", "density_hex", "Transcript density",
                 accepts_subject_kinds=("gene", "gene_set"),
                 required_views=("density",), scale_lockable=True),
        PlotSpec("feature_umap", "feature_umap", "Feature UMAP",
                 accepts_subject_kinds=("gene", "gene_set"),
                 required_views=("umap", "expr"), scale_lockable=True),
        PlotSpec("dotplot", "dotplot", "Across clusters",
                 accepts_subject_kinds=("gene",), required_views=("expr",)),
    ),
    pin_kinds=("gene",),
    csv_columns=("cluster", "cell_type", "cell_type_short", "confidence",
                 "confidence_score", "key_markers", "notes", "category",
                 "lineage", "exclude", "verify"),
))

register_method(MethodSpec(
    method_id="jazzpanda_spatial_glm", workflow_id="markers",
    display_name="jazzPanda spatial GLM",
    assumptions=("multivariate spatial GLM across all panel genes, not per-cluster DE",
                 "one top_cluster per gene; NoSig means no cluster reached significance"),
    outputs=("glm_coef (spatial GLM Estimate, primary ranking)",
             "pearson (gene↔assigned-cluster spatial specificity)",
             "max_gg_corr", "max_gc_corr"),
    caveats=("panel-scoped: absence off panel is never evidence against a cell type",
             "1–2 marker clusters are capped and flagged verify"),
    status_rule=jazzpanda_status,   # the mapping table in §2.3
))
```

`pathways` registers a `CensusHeader`, `contrasts` a `ConfoundChecklistHeader`. **COMPARISON is built last deliberately**: if the shell, the status vocabulary, the conditionality contract and the spine carry a third task kind with no new mechanism, R1–R8 are load-bearing rather than descriptive — and that, not any single figure, is the interface contribution.

---

## 4. Multi-sample presentation

### 4.1 The decision: one primary + a context strip

Not equal-weight small multiples. Not a bare sample switcher.

The decisive constraint is rhetorical, not perceptual. *Divergence is a flag; agreement is not corroboration* is an **asymmetric** claim, so the layout must be asymmetric. An equal-weight grid of N sections is the single best design ever invented for making a viewer think "they all match, so this is solid" — exactly the inference the architecture forbids, since agreement between replicates is not an input to any Panoscope number.

The field is split on convention and the split is informative: interactive viewers do one-at-a-time (Loupe switches replicates via a dropdown; cellxgene has never shipped split-screen, citing screen space; 10x ships no multi-sample Xenium path at all and warns that merging needs large fixed centroid offsets), while scripted plotting does grids (`squidpy.pl.spatial_scatter(library_key=…, ncols=4)`). The one tool that solved it properly — Vitessce — solved it with **coordination**, not layout: each view bound to its own dataset scope, `linkViews()` sharing `spatialZoom`/`spatialTargetX`/`spatialTargetY`.

### 4.2 Why not the alternatives

- **Side-by-side small multiples** manufacture the forbidden inference, and accuracy declines *linearly* with frame count — driven by visual search, not a cognitive cliff — with highlighting the one intervention shown to mitigate it. They are also the weakest common layout for judging *magnitude* of difference (overlaid > adjacent > stacked).
- **A bare sample switcher** is the flicker paradigm: with a global transient between frames, large expected changes take on the order of 10 s to notice; remove the blank and it drops to under a second. Temporal juxtaposition also ranks last for analysis tasks.

The hybrid keeps the switcher (people expect it; it is how Loupe works) but stops *relying* on it to carry detection.

### 4.3 Screen-space arithmetic, measured on this shell

`app.py:286` gives a ~640 px centre column at 1440 px; `ui/spatial_stage.py` splits it two-up at `_TISSUE_HEIGHT = 300`, so **one density panel ≈ 310 × 300 px**. The demo tissue is 7,521 × 5,471 µm. A hex-bin ramp needs ≈2 px per bin edge to read as a field rather than noise:

```
panel_px  >=  2 × span_µm / bin_µm
```

| Layout in the 640 px column | px/panel | 25 µm | 50 µm | 100 µm |
|---|---|---|---|---|
| today (density + feature-UMAP pair) | 310 | ✗ 1.0 px | ~ 2.1 px | ✓ 4.1 px |
| 2 samples inside that 310 px slot | 149 | ✗ 0.5 | ✗ 1.0 | ~ 2.0 |
| 2 samples across the full column | 314 | ✗ | ~ 2.1 | ✓ 4.2 |
| 3 samples, full column | 205 | ✗ | ✗ 1.4 | ✓ 2.7 |
| 4 samples, full column | 151 | ✗ | ✗ 1.0 | ~ 2.0 |

**Equal-weight multiples break at N=2 at the default bin.** They are not on the table for a 4–8 sample study. Thumbnails, which only answer "does anything look different here?" — a detection task, not measurement — are legitimate at 150 px / 100 µm, which is exactly what the table says is affordable.

Two consequences: **(1)** fix `ui/state.py:53` `DEFAULT_BIN_UM` from 25 to 50 (today the default view renders at ~1 px/hex, sub-threshold, and CLAUDE.md already specifies 50 µm). **(2)** Never auto-coarsen the bin to make multiples fit. It changes no value, so it is not a grounding violation — but it lets layout drive the picture, which hollows out "the bin size is the biologist's control." If a thumbnail is forced to 100 µm, **label the thumbnail**.

### 4.4 Layout

```
┌─ EVIDENCE STAGE ─────────────────────────────────────────────────────┐
│  PRIMARY — sample1 / hb1                                   [layers ▾]│
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │   CD68 · 50 µm hex density   │  │   CD68 · feature UMAP        │  │
│  │            (310 px)          │  │            (310 px)          │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  │
│  transcript detections · CD68 · 50 µm bins, area-normalised tx/µm²,  │
│  √ colour · shared scale across all samples, max 0.084 from S1       │
│  280-gene panel · sample1/hb1 · qv<20 + controls removed · pre-       │
│  segmentation, not per-cell                                       ⓘ  │
│                                                                      │
│  other samples · context          (click to promote · same gene,     │
│                                    same bin, same colour scale)      │
│  ┌───────┐ ┌───────┐ ┌━━━━━━━┓ ┌───────┐                             │
│  │  S2   │ │  S3   │ ┃  S4   ┃ │  S5   │   ← S4 outlined: flagged    │
│  │ 150px │ │ 150px │ ┃ 150px ┃ │ 150px │                             │
│  └───────┘ └───────┘ ┗━━━━━━━┛ └───────┘                             │
│    glm 8.1   glm 7.6   glm 0.4    glm 7.9                            │
│                        ⚑ diverges                                    │
└──────────────────────────────────────────────────────────────────────┘
```

**The strip is the switcher.** No separate widget. The control sits on the thing it changes, and its state is legible from the pixels.

### 4.5 The concordance flag — one-directional, by construction

Computed deterministically (Tier A, alongside `agent/verdict.py`), written into `data/studies/<id>/interp/clusters/c{n}.json`, hashed in `manifest.json`. No LLM, reproducible.

```python
@dataclass(frozen=True)
class SampleObservation:
    sample_id: str
    glm_coef: Optional[float]
    pearson: Optional[float]
    status: MeasurementStatus          # a gene off-panel in one sample is never_measured there


@dataclass(frozen=True)
class ConcordanceRecord:
    """Divergence is a flag. Agreement is NOT recorded as corroboration.

    ``diverging`` is non-empty only when a marker's per-sample profile splits.
    When it is empty this record renders NOTHING — no green tick, no
    "concordant in 4/4" badge, no sample-count chip beside the confidence band
    (adjacency would imply the band is a function of n; it is not).
    """
    cluster: str
    gene: str
    observations: tuple[SampleObservation, ...]
    diverging: tuple[str, ...]         # sample_ids whose value splits from the rest
    rule: str                          # the deterministic rule, stated: e.g.
                                       # "one sample below the Medium edge (1.0) while
                                       #  the others are at or above Medium-High (2.5)"
    affects_call: bool                 # gene is in ClusterVerdict.key_markers
    source_trace: tuple[str, ...]
```

It surfaces in exactly three places, and **renders nothing when it does not fire**:

1. **Verdict header** — a caveat line at the same visual weight as the existing `verify` flag: *"CD68 profile diverges in S3 (glm_coef 8.1 / 7.6 / 0.4 / 7.9)."* Never a corroboration badge.
2. **The context strip** — the diverging thumbnail gets an outline and a one-line label. This is precisely the highlighting intervention that the small-multiples evidence shows actually mitigates the search cost.
3. **The evidence table** — a per-marker inline per-sample **sparkstrip** beside `glm_coef` (`ColumnKind.sparkstrip`). This is the explicit encoding, and it is where the *magnitude* of divergence lives, because juxtaposed maps are the worst layout for that judgement.

`verify = TRUE` when divergence fires on a marker the call depends on (`affects_call`) — consistent with the existing "evidence is insufficient" rule, and it needs no new CSV column.

**A grounding test encodes the asymmetry.** In `tests/`: the agent may never emit a corroboration claim sourced from replicate agreement ("consistent across samples", "confirmed in all four", "reproducible"); divergence claims must cite the per-sample numbers from the `ConcordanceRecord`. This turns a design principle into something CI enforces — the same move the repo already makes for fabricated PMIDs.

### 4.6 Coordination rules

The pin is a property of the **subject** (gene × cluster), not of the sample. `K_SELECTED_MARKERS` is `{cluster: [gene, …]}` and `K_BIN_UM` is global; **neither gains a sample dimension.** Promoting a different sample must not drop the pin. Four things stay coordinated:

1. **Pinned gene** — the same gene everywhere, always.
2. **Bin size** — the same preset everywhere, or the thumbnail states the coarser bin it used.
3. **Colour scale — the load-bearing one.** Fix the density domain across the whole sample set and state it in the conditionality line. Per-panel autoscaling on an area-normalised density would make two identical tissues look different and two different tissues look identical: it would fabricate and hide divergence in the same layout.
4. **Framing — normalised, not absolute.** Sync *fit-to-tissue-bbox plus zoom factor*, not absolute pan/zoom. Sections from different patients are unregistered; a shared coordinate space is a fiction. A rubber-band zoom on the primary maps to the same fractional region of each thumbnail's own bbox.

### 4.7 Where the sample roster lives

Split it: **the shell owns scope, the review surface owns focus.**

- **Study-level shell** — the roster: which samples are in the study, their metadata, which are in the joint clustering. This is static provenance text ("4 samples · joint clustering"), not a control. The one genuine shell control is **exclusion** (drop a failed section) — and it belongs there precisely because it changes what the concordance check is computed over, making it a pipeline input: it re-runs `panoscope build` and re-stamps the manifest, rather than live-re-rendering.
- **Review surface** — which sample is *primary right now*. A viewing control on the subject under review, exactly like the bin size. A shell-level dropdown would make it a **global mode**, and a global mode produces the worst failure a review tool can have: "I reviewed c4 in sample 2" — a per-cluster judgement whose evidence silently depended on a setting made ten minutes ago, on another screen, now invisible. That is how a reviewer records a wrong call in good faith.

---

## 5. The argue-then-commit flow

### 5.1 Three layers, deliberately non-interchangeable

| Layer | Object | Mutability | Where it lives |
|---|---|---|---|
| **Claim** | `ClusterVerdict` (Tier A) + `OpeningInterpretation` (Tier B) | immutable; rebuilt only by `panoscope build` | `data/studies/<id>/interp/clusters/c{n}.json`, hashed in `manifest.json` |
| **Argument** | chat thread, pinned views, skeptic pass, holistic refinements | mutable, disposable | `ui/chat_store.py`, session state |
| **Decision** | the biologist's committed call | **append-only, versioned** | `context/decisions/decision_log.jsonl`; `interp/review_state.json` becomes a projection |

Two invariants: **layer 2 never writes layer 3 implicitly** (no natural-language commit — "ok, call it CAF" is a *proposal*; the existing `_looks_like_override` → `NoteDraft` → two-tap confirm discipline extends to the decision, and **no LLM parse appears anywhere in the state machine**), and **layer 3 never mutates layer 1** (overrides are applied at composition, which `ui/data_access.py:composed_verdicts` / `override_info` already does correctly).

### 5.2 State diagram

```
                        ┌──────────────────────────────────────────┐
                        │                                          │
                        ▼                                          │
   ┌────────┐   accept (1 tap, unflagged)          ┌────────────┐  │
   │        │ ───────────────────────────────────► │  ACCEPTED  │  │
   │        │   accept a FLAGGED call (2 taps +    └────────────┘  │
   │        │   drafted validation note + live          │          │
   │  OPEN  │   literature reconciliation)              │          │
   │        │                                           │          │
   │        │   override (2 taps: scope, basis;     ┌──────────────┐
   │        │   +1 tap if scope ∈ {dataset, lab})   │ OVERRIDDEN   │
   │        │ ───────────────────────────────────► └──────────────┘
   │        │                                           │          │
   │        │   defer (1 tap + reason code           ┌────────────┐│
   │        │   auto-filled from discriminate.py)    │  DEFERRED  ││
   │        │ ─────────────────────────────────────► └────────────┘│
   └────────┘                                           │          │
        ▲                                               │ reopen   │
        │                                               │ (1 tap)  │
        └───────────────────────────────────────────────┴──────────┘

              any terminal state ──► [ EXPORT FREEZE ]  (1 heavy confirm)
                                            │
                                            ▼
                             reopen after export ──► ADDENDUM
                                   (+1 line reason; additive, attributed,
                                    never a silent edit)

   STALE is an overlay, not a state: claim_sha changed under a committed
   decision. It flags, it never auto-revokes. One-tap re-affirm is allowed
   ONLY when call, confidence and evidence are byte-identical; otherwise a
   forced diff ("what changed since you read it") gates the recommit.
```

Terminal states must be **exhaustive**: a report cannot export with a cluster in `OPEN`. `DEFERRED` is first-class — the BI-RADS 0 move, where "incomplete, needs additional imaging" is a legitimate final assessment, not a failure to report — and it splits deterministically using `agent/discriminate.py`: `settleable_on_panel == True` → *deferred: the panel can settle it* (name the discriminating gene, via `settle_line`); `False` → *deferred: needs another assay*.

Today `_triage_bucket` in `ui/summary.py` has only needs / review / signed. `DEFERRED` is new.

### 5.3 Numbered steps

1. **Open the unit.** The agent posts its interpretation before any question: call, band, driving markers with their jazzPanda numbers, and the literature basis, each claim carrying a real, live-fetched citation. *For flagged (`verify`) clusters, open evidence-first*: evidence and caveats rendered, the call one tap away. This preserves "the agent posts its interpretation first" while reducing anchoring — and it gives the paper a measurable: override rate and time-to-commit with vs without the call revealed.
2. **Argue.** The biologist questions; the agent answers under the grounding gate. Each turn is a `AgentResponse` with its `GroundingSidecar`. The authority spine (§2.9) makes the epistemic profile of each answer scannable.
3. **Pin evidence.** Clicking a marker pins it; views redraw. Nothing is recomputed — pinning is a viewing control.
4. **Attach an exhibit.** A view can be attached as a *receipt*: `{view, subject, bin_um, artifact_sha}` — a deterministic pointer that re-renders the same figure, **not a bitmap**. One tap, on the figure. This is how an exploration becomes evidence without exploration becoming state.
5. **(Optional) pressure test.** For flagged or contested clusters, one tap runs `agent/skeptic.py:pressure_test` — the double-reading analogue, cheaper than a committee.
6. **The agent may propose, never commit.** A natural-language override produces a `NoteDraft` with its `Tension` already computed. The commit control is **disabled while an unconfirmed draft is open** ("save or discard the note first"), so a decision can never cite knowledge that was never written.
7. **Commit, in the commit bar, on the unit page.** The identical component is mounted on the Summary board for the trivial cases, backed by one store — so a clean 9-cluster set can be closed without opening nine pages, while a contested one is decided where its evidence is.
8. **On override, the reconciliation fires.** The agent uses the biologist's call **and** cross-checks the literature, reporting agreement and dissent with real citations. The dissent is retained in the record and **prints in the report even when the biologist wins**. Never a bare "got it"; never a silent overrule.
9. **The decision is written as an event.** Append-only. `interp/review_state.json` (today `{"dataset", "reviewed": {}, "saved_at"}`) becomes a replayable projection of the ledger.
10. **Holistic re-check.** After each commit, the set-level board recomputes its reconciliation items (spillover, recurring programs, override dissent) — the skill's mandated holistic review, surfaced as unresolved count, contradictions, and staleness.
11. **Export.** One heavy confirm: the unresolved list, the version freeze, a report id, the dataset manifest sha. This is the only near-irreversible act.

### 5.4 The commit record

```python
@dataclass(frozen=True)
class DecisionRecord:
    id: str                            # "d-9f21"
    ts: str
    actor: str
    dataset: str
    unit: str                          # cluster id or contrast id
    workflow_id: str
    claim_ref: "ClaimRef"              # artifact path + sha256 + the call as read
    decision: Literal["accepted", "overridden", "deferred", "reopened"]
    final_call: Optional["FinalCall"]  # cell_type / lineage / category on override
    reason_codes: tuple[str, ...]
    note_ids: tuple[str, ...]
    exhibits: tuple["Exhibit", ...]    # {view, subject, bin_um, artifact_sha}
    argument_digest: "ArgumentDigest"
    supersedes: Optional[str] = None
```

The shape is deliberately PROV-O-mappable: the decision is an Entity `wasAttributedTo` an Agent, `wasGeneratedBy` the review Activity, `wasDerivedFrom` the claim artifact and the note ids, with `qualifiedRevision` for reopen→recommit.

**Reason codes** — a small closed vocabulary, each mapping to one fixed report sentence and each queryable for calibration:

`OFFPANEL_CANON` · `TISSUE_CONTEXT` · `OWN_VALIDATION` · `LAB_CONVENTION` · `SPILLOVER` · `LIT_CONTRADICTS` · `THIN_EVIDENCE` · `NEEDS_ASSAY` (auto-set when `settleable_on_panel == False`).

Codes are *tags on the note*, never a substitute for the note's claim text. They exist so that "why did this lab override the machine 14 times?" is a query rather than a read-through of chat logs.

### 5.5 Representing the argument without shipping a transcript

Print **positions and moves**, never turns.

A turn is **load-bearing** iff it (a) introduced a grounded artifact that the final rationale cites, (b) changed the proposed call, confidence or `verify`, or (c) is the biologist message that produced the saved `NoteDraft`. Each becomes a `Move{by: ai|you, kind: evidence|objection|concession|question, one_line, refs}`. The one-liner is LLM-written but passes the same grounding gate: **a move whose refs do not resolve is dropped, not printed.** The digest is **frozen into the decision record at commit**, so later chat never rewrites history.

Fixed-shape report block (≤5 lines), added to `ui/report.py:default_cluster_summary` as a `Decision —` movement beside the existing Identity / Programs / What-would-settle-it / Note movements:

> **c2 — Stromal → CAF (your call).** AI: Stromal, Medium-High (POSTN glm 6.3 · r 0.41). You: CAF · own_validation · firm.
> Why it differs: FAP, COL1A1, DCN off-panel — never measured, so their absence is not evidence \[OFFPANEL_CANON]; LUM/POSTN/PDGFRB consistent with CAF in breast TME \[TISSUE_CONTEXT]; lab IHC \[OWN_VALIDATION].
> Literature check at the time of the call: 2 agree (PMID …, …), 1 dissent (PMID …) — dissent retained.
> Exhibit: POSTN transcript density, 50 µm. Decision d-9f21, 3 exchanges; full record in appendix.

Three defensible properties: the disagreement is **structured, not narrative** (synoptic reporting is the one intervention with a large, replicated effect on documentation completeness); the **dissent survives the biologist winning**; and the full thread is **linked, not printed** — kept local, referenced by decision id.

### 5.6 When they agree

Agreement is an **attestation**, not a claim.

- **Record it.** Without it you cannot distinguish "a label no human ever read" from "a label a named human endorsed on this version at this time." That distinction is exactly what is missing from current annotation practice, where a cell type ships as a bare string in `obs` with no actor, date, evidence or dissent.
- **It writes no note.** No new claim → nothing enters the lab knowledge layer, nothing is citable as evidence later.
- **It never moves a number.** A grounding test, not a promise: no decision record may be an input to `agent/verdict.py`, and *N* accepts must leave `confidence_score` byte-identical. Never render agreement as an aggregate that reads like confidence ("8/9 confirmed").
- **The asymmetry rule.** *Accepting a call the system flagged is itself a claim* — "we looked and it holds despite the flag" — so it writes a validation note with a live literature reconciliation. Accepting an unflagged call is a tick. The line is principled: the note exists exactly when the human act contradicts the system's own caution, which is when it carries information.

### 5.7 Friction budget

| Act | Interaction | Justification |
|---|---|---|
| Accept unflagged call | **1 tap**, reversible | minimum viable commit; recorded with actor + claim_sha |
| Accept flagged (`verify`) call | 2 taps + read the drafted claim and live tension | contradicts the system's caution → note + literature check |
| Override, cluster scope | 2 taps (scope, basis) + confirm | tension precomputed at draft, so confirm is instant |
| Override, dataset/lab scope | +1 explicit scope confirm | scope is enforced and will fire elsewhere — blast radius pays |
| Defer | 1 tap + reason code (auto-filled from `discriminate.py`) | must state what would resolve it |
| Bulk-close remaining unflagged | 1 tap + 1 confirm listing the clusters | each recorded individually and reversibly; flagged/contested excluded by construction |
| Commit on a stale claim | forced diff | the artifact you judged is not the artifact on disk |
| Reopen after export | +1 line reason → addendum | the outward-facing record changed |
| **Export** | **1 heavy confirm** | the only near-irreversible act |

The shape of this budget is the clinical-decision-support lesson, not taste: when every action demands confirmation, confirmation becomes reflex — inpatient medication-alert override rates around 73%, with only ~60% of overrides judged appropriate. Spend friction where disagreement is informative. And the fix for rubber-stamping is **cheaper verification, not more clicks**: automation bias tracks verification complexity and cognitive load. Linked views, the explicit panel-absence statement, and "what would settle it" are the real anti-automation-bias mechanisms here.

### 5.8 Concrete deltas against the current repo

1. `ui/data_access.py:mark_signed_off` / `clear_signoff` never call `agent/memory.log_decision` — sign-off exists in review state but is invisible in `context/decisions/decision_log.jsonl`. **Log every transition.**
2. Review state is `{cluster: {at, note_id}}` — not version-pinned, not attributed. Add `actor`, `claim_sha`, `decision`, `reason_codes`, `exhibits`, `argument_digest`; make it a replayable projection.
3. No `deferred` state in `_triage_bucket`; wire `settleable_on_panel` into the reason code; add the export gate on non-terminal units.
4. `ui/chat_store.py` persists too little to build a digest (role, text, source *kinds*, verify). Build and **freeze** the digest at commit from in-session objects — freezing is what makes the report stable.
5. `ui/report.py:default_cluster_summary` has no decision movement. Add it.
6. Exports: append `decided_by, decided_at, decision, reason_codes, decision_id` **after** the locked CSV column order, so the contract is preserved; stamp reports with a report id + manifest sha.
7. New grounding tests: every exported unit has a terminal decision; every `claim_sha` resolves in the manifest; no decision record influences `confidence_score`; every reason code maps to a real note or a deterministic flag; an override's dissent citations appear in the report; every digest ref resolves (drop-not-print).

**Main risks and mitigations.** Ledger/projection drift → rebuild-on-read in tests. Digest generation is LLM-touched → the existing grounding gate plus drop-not-print. Reopen storms after a pipeline rebuild → `stale` is a flag, never an auto-revoke.

---

## 6. Build order for the one-month window

Governing rule from CLAUDE.md, applied to interface work: **one path solid end to end before breadth.**

| Week | Land |
|---|---|
| **0** (2 days) | `pyproject.toml` + lockfile + console entry point; `core/read.py` seam with the `ui/data_access.py` re-export shim; the `summary.py` store leak fixed; manifest shas in the footer and exports; `DEFAULT_BIN_UM = 50` |
| **1** | `Conditionality` + `render_panel()` that refuses; retrofit the three views in `ui/spatial_stage.py`; scale lock; `MeasurementStatus` + three-block evidence table + coverage ribbon **on ASSIGNMENT**, where the data already exists and is simply not rendered |
| **2** | `UISpec` / `MethodSpec` / registry + `degraded_spec`; port the assignment surface to be *generated* from its UISpec rather than hand-built; `PinnedSubject` typing + the capability matrix with its two distinct failure messages |
| **3** | CHARACTERISATION: census header replacing the borrowed band; K/N promoted from tooltip to coverage bar; the untestable block with search; the testability plane; the categorical tier matrix |
| **4** | Argue-then-commit: `DecisionRecord`, the ledger, `DEFERRED`, the commit bar on both mount points, the `Decision —` report movement, the seven new grounding tests; claim model + authority spine + claim card |

COMPARISON and multi-sample are **specified here and built after submission**, unless week 3 finishes early — see the cut list.

---

## 7. What to cut if the window is tight — ranked

Ranked *cut-first to cut-last*. Cut from the top.

**1. The React renderer (already scheduled post-submission).**
*Cost to the paper:* the two-renderer claim becomes "one renderer plus a frozen contract and a second renderer in progress." Reviewers will accept a contract demonstrated once if the contract is precise enough to re-implement — which is exactly what §3 is for. **Zero cost if the UISpec lands.** Catastrophic cost if it does not, because then "Streamlit app" is the only description available.

**2. The COMPARISON surface (build the `UISpec` registration and the fallback; skip the live DE artifact).**
*Cost:* R1–R8 lose their third proof point. The abstraction is *claimed* to generalise rather than *shown* to. Mitigation that recovers most of it for ~2 days: register the `contrasts` UISpec with its `ConfoundChecklistHeader` and render it against a **synthetic fixture** shipped in `tests/fixtures/`, labelled as a fixture in the figure caption. The generality claim then rests on a demonstrated declaration rather than on prose.

**3. The authority spine at per-sentence granularity (keep the claim card and the number chips).**
*Cost:* R6 degrades from "every sentence resolves" to "every number and every citation resolves." That is still stronger than any comparable tool, and the drop-not-print stub still shows the gate acting. The spine is the most striking figure in the paper, so cut it reluctantly — but the claim card carries the scientific content and the spine carries the aesthetics.

**4. Multi-sample entirely (single-sample demo only).**
*Cost:* the "divergence is a flag, agreement is not corroboration" argument becomes design rationale rather than an implemented feature, and one grounding test goes unwritten. Recoverable cheaply as a **written design principle + the `ConcordanceRecord` dataclass + the CI test that forbids corroboration language**, with no UI at all — about a day, and it keeps the sharpest epistemic claim in the paper alive.

**5. The testability plane (keep the census header and the untestable block with search).**
*Cost:* small. The header and the searchable untestable block carry the panel-scoping claim; the plane is the elegant version of a point already made three other ways. Fall back to a `-log10(q)` bar chart **only with the untestable count stated in the conditionality line** — never silently.

**6. The volcano gutter.**
*Cost:* only matters if COMPARISON ships at all. If it does, the gutter is the single most novel figure in the comparison surface and should not be cut before item 5.

**7. `DEFERRED` as a state + the export gate.**
*Cost:* the state machine loses exhaustiveness, which is one of the sharper prior-art connections (BI-RADS 0). Cheap enough to keep (one enum value, one bucket, one gate) that cutting it is almost never right.

**Never cut, in any scenario:**

- **Week 0 packaging.** If the month runs out here you have still fixed the loudest reproducibility hazard in the repo, and every criticism of `requirements.txt` is answered.
- **The three-block evidence table with hatched never-measured rows.** This is the headline claim made visible. It is currently *invisible* in the headline surface, the data model already supports it, and it is roughly two days of work. Shipping without it means the paper's core claim has no pixel that demonstrates it.
- **`Conditionality` + `render_panel()` refusing to draw.** Five lines of enforcement that convert a design principle into a mechanism. The cheapest defensible thing in the document.
- **`UISpec` + the registry + `degraded_spec`.** Cutting this cuts the interface contribution and leaves an app.

---

## 8. Summary of calls made where the research disagreed

| Tension | Call | Why |
|---|---|---|
| React migration (4 wks) vs new interface (4 wks), one month total | Build the **contract** now, the second renderer later; the UISpec *is* the migration path | "We built a React app" is not a contribution; "the surface is generated from a declaration" is, and it makes the port a scoped re-implementation rather than a rewrite |
| Fallback stack under deadline: Dash vs Shiny vs stay | **Shiny for Python** if React slips twice; never Dash | `ui.tags.*` emits your own class names so the ~110 HTML-string sites port cheaply; Dash's callback graph is the slow part and this app is callback-heavy |
| Equal-weight small multiples vs switcher vs hybrid | **One primary + context strip**, strip *is* the switcher | Asymmetric claim needs asymmetric layout; multiples break at N=2 at the default bin on this shell; a bare switcher is the flicker paradigm |
| Default hex bin 25 µm (`ui/state.py:53`) vs 50 µm (CLAUDE.md) | **50 µm** | `panel_px ≥ 2 × span/bin` puts 25 µm at ~1 px/hex in a 310 px panel — sub-legibility |
| Per-sentence authority spine vs per-turn `Sources` line | **Per-sentence model, collapsed rendering**; cut to claim-card-only under pressure | Marks collapse anyway; the claim card carries the science, the spine carries the scan |
| Commit on the cluster page vs the summary board | **One component, two mount points, one store** | Evidence locality for contested calls; set-level closing for clean ones; neither alone is right |
| `ClusterEnrichment.confidence` field | Keep in the artifact, **structurally undrawable** via `CensusHeader` having no band slot; deprecate at the next `pipeline_version` | Tests and artifacts read it; the fix is to make the category error impossible to render, not to break the schema |
| Difference maps for paired contrasts | **Refused by default**; two panels on a locked shared scale | A difference is a derived quantity (R5); it renders only when the pipeline emits a difference artifact with its own conditionality |

**Key files this document touches:** `app.py:286` · `ui/data_access.py` (→ `panoscope/core/read.py`) · `ui/spatial_stage.py:74,101,626,647,835` · `ui/evidence_table.py` · `ui/enrichment_table.py:171-186` · `ui/verdict_header.py` · `ui/state.py:47-56` · `ui/summary.py:1142,1328` · `ui/report.py` · `agent/types.py` · `agent/config.py:132-195` · `agent/discriminate.py` · `pipeline/store.py:209-251` · `pipeline/paths.py:107` · `data/studies/xenium_hbreast_sample1/manifest.json` · `requirements.txt`