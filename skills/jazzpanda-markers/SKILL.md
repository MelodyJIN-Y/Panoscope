---
name: jazzpanda-markers
description: Interpret jazzPanda spatial marker output to annotate cell clusters with a cell-type call, a confidence level, and the evidence behind it, for imaging-based spatial transcriptomics (Xenium, CosMx, MERSCOPE). Use whenever a user has jazzPanda marker results (a `glm_mg_result` / a `top_result` or `full_result` table with columns like gene, top_cluster, glm_coef, pearson, max_gg_corr, max_gc_corr, p_value) plus a panel gene list and wants cluster interpretation, cell-type annotation, lineage assignment, or a confidence/verify verdict. Also trigger on "annotate these clusters", "what cell type is cluster X", "jazzPanda markers", "spatial markers", "top_cluster", "glm_coef". Tissue- and species-agnostic; adapts naming and expected markers to the tissue context. Enforces the panel-absence rule, a confidence rubric grounded on jazzPanda's spatial signal (not mean-expression DE), a holistic cross-cluster review, and a no-fabrication / real-citation discipline.
---

# Interpreting jazzPanda Spatial Markers for Cell-Type Annotation

This skill turns jazzPanda's spatial marker output into a grounded, per-cluster cell-type call with a confidence level, a `verify` flag, and a rationale that names its evidence. It is the interpretation layer; jazzPanda is the engine. It is built for a wet-lab biologist reading a targeted spatial panel who has to decide what each cluster is and whether to trust the call.

It is designed for interpreting jazzPanda's spatial statistic. Every caveat below was found to matter in real annotation work: follow them strictly.

## Core principles (never violate)

- **Confident floor.** Every marker, number, and confidence score must trace to jazzPanda's output or the panel list. Never invent a marker, a statistic, a p-value, or a confidence number. If it is not in the source, do not state it.
- **Open ceiling.** Cell-type reasoning over literature and biological context is uncertain. Label it as a direction, not a fact.
- **Panel-absence rule (the headline catch).** The absence of a canonical marker is NOT evidence against a cell type if that gene was never on the panel. Always check the panel gene list before down-weighting a missing marker, and say plainly when a marker was off-panel: "not measured", not "not expressed".
- **Cite everything, with real references.** Every interpretive (literature) claim must carry a citation to a real paper, looked up live through a literature connector (PubMed / bioRxiv). Never write a PMID or DOI from memory: a fabricated citation is the worst possible failure, worse than no citation. If a lookup returns nothing, say the literature is thin; do not invent a reference.
- **Defer to the biologist, always.** It is their data and their name on the annotation. The tool informs and preserves the decision; it never makes the call and never silently overrules an override.
- **When evidence is insufficient, set `verify = TRUE` and say "re-check this."** Do not guess to seem helpful. A tool that knows when to stay cautious is more trustworthy than one that always answers.

## What jazzPanda measures (read this before interpreting any number)

jazzPanda does not ask "is this gene higher on average in this cluster's cells?" (mean-expression differential expression, e.g. Seurat `pct.1`/`pct.2`/`avg_log2FC`). It asks **"do this gene's transcripts fall in the same places on the tissue as this cluster's cells?"**

It lays a grid of tiles over the tissue and, for each gene, counts that gene's transcript detections in every tile: turning the gene's 2D spatial pattern into a one-dimensional **gene vector** of binned counts. It does the same for each cluster, counting how many of that cluster's cells fall in each tile, making a **cluster vector**. A gene is a marker for a cluster when its gene vector **rises linearly with that cluster's vector** across the tiles. (jazzPanda paper, Methods pp. 5, 29–30.)

jazzPanda offers two tests; the recommended default is the GLM:

- **jazzPanda-glm** (default, false-discovery controlled): a penalized (lasso) linear model where the **gene vector is the response** and the predictors are the **cluster vectors** plus **sample/batch vectors** and **negative-control background vectors**. Lasso selects the relevant clusters (per-gene λ by 10-fold CV); the gene is assigned to the cluster with the **maximum coefficient and minimum p-value**. Counts are used on their native scale (no normalization), to preserve the linear relationship.
- **jazzPanda-correlation**: a permutation-based Pearson correlation between the gene vector and each cluster vector; p-value from label permutation, BH-adjusted.

### The output columns: what each number means (grounded)

From the `glm_mg_result` object: `top_result` (one row per gene) and `full_result` (one row per gene × model term). Definitions are from the jazzPanda package docs (`get_top_mg.Rd`, `get_full_mg.Rd`) and source (`R/lasso_markers.R`):

| Column | Meaning | How to use it |
|---|---|---|
| `gene` | Gene symbol |: |
| `top_cluster` | Most relevant cluster after thresholding the GLM coefficient; **`"NoSig"`** if the top coefficient is below `coef_cutoff` (default 0.05) or its best match is a background term | `NoSig` → not a marker → contributes nothing; a cluster full of NoSig is weak |
| `glm_coef` | The lasso/GLM coefficient (Estimate) on the selected cluster vector: how strongly the gene's binned spatial counts rise with that cluster's binned cell counts | **Primary ranking statistic.** Bigger = stronger spatial marker |
| `pearson` | Pearson correlation between the gene vector and the **assigned** cluster vector | **Spatial specificity** (0–1): how tightly the gene's transcripts localize to the cluster. Corroborates `glm_coef` |
| `max_gc_corr` | Maximum Pearson correlation between this gene's vector and **any** cluster vector | If `max_gc_corr > pearson`, the gene localizes **better with a different cluster** than the one it was assigned → specificity caveat, flag it |
| `max_gg_corr` | Maximum Pearson correlation between this gene's vector and **any other gene's** vector | Very high (→1) = the gene's spatial pattern is nearly identical to another gene's → not spatially unique; down-weight as sole evidence |
| `p_value` (`full_result` only) | GLM coefficient significance (`Pr(>|t|)`) for that gene × cluster term | Significance filter; combine with `glm_coef` for ranking |

**Do not conflate jazzPanda's analysis grid with any viewing control.** jazzPanda's `bin_param` (e.g. 40×40 tiles for the Xenium breast data) is a fixed count of tiles chosen so the average cells-per-tile ≈ 1; it is baked into the precomputed result. A density-view bin size in the app (e.g. 50 µm) is a *viewing* control that changes the picture, never a value.

### Why this is stronger than mean-expression DE (say it accurately)

On sparse imaging data, mean-expression DE (Wilcoxon) flags most panel genes as "significant" and cannot adjust for sample/batch covariates, so it is too liberal. jazzPanda returns a smaller, spatially specific set and can adjust for batch and technical background. (Paper pp. 3, 18–19, 21, 27.) Keep this at the safe level: **spatial detection, distinct from DE.** Do not overstate the mechanism beyond the paper.

## Step 1: Gather context

Before examining any marker, collect (ask if not provided):

1. **Tissue** (e.g. breast cancer, kidney, brain) and **species** (human/mouse): sets the expected markers.
2. **Platform** (Xenium / CosMx / MERSCOPE) and **panel + approximate gene count**: this directly bounds confidence and is required for the panel-absence rule.
3. **Experimental design**: samples, conditions.
4. **Clustering**: method/resolution, and the cluster→label mapping if one exists.
5. **Goal**: broad lineage assignment vs. fine-grained typing.
6. **Expected cell types**: what the biologist expects in this tissue.

Targeted panels bias which markers appear. State the panel size; smaller panels (300–500 genes) yield more ambiguous clusters than larger ones (5k), and confidence must reflect that.

## Step 2: Read the jazzPanda output and the panel

- Load `top_result` (and `full_result` if per-term p-values are needed). Group by `top_cluster` to get each cluster's marker set; within a cluster, rank by `glm_coef` (descending).
- Count markers per cluster. Note which clusters are **small** (few assigned markers): jazzPanda has less power for small/rare clusters and their marker ranking is less stable (paper pp. 21, 24, 26, 28). This drives the small-n rule in Step 3d.
- Load the **panel gene list** (the source of truth for absence). If the panel carries a per-gene cell-type annotation column, it is useful supporting context: but it is not the call; jazzPanda's spatial signal and the literature drive the call.

## Step 3: Per-cluster annotation

For EACH cluster:

### 3a. Read the spatial marker signature

Look at the top markers by `glm_coef`, and for each read `pearson` alongside:

- **`glm_coef`** is the headline strength; **`pearson`** tells you whether that strength is spatially specific. A high `glm_coef` with high `pearson` is the strongest evidence: the gene's transcripts physically concentrate where the cluster's cells are.
- **Divergence is informative.** A marker can rank high on `glm_coef` but have modest `pearson` (or vice versa): surface both rather than collapsing them. If `max_gc_corr > pearson`, note that the gene localizes better with another cluster (possible bleed / shared domain). If `max_gg_corr` is near 1, the gene is not spatially unique.
- **Coherence.** Do the top markers tell one consistent biological story, or a mixed bag? Coherence raises confidence; a mixed bag lowers it and triggers the Step 3c branches.

### 3b. Apply the panel-absence rule (CRITICAL: do this before assigning or down-weighting)

For the cell type you are considering, list its canonical markers and check each against the **panel list**:

- If a canonical marker is **present on the panel** and jazzPanda gives it strong signal → supporting evidence.
- If a canonical marker is **present on the panel** but weak/absent in this cluster → genuine evidence *against* (a real down-weight).
- If a canonical marker is **NOT on the panel** → it is **uninformative**. Its absence says nothing. Never down-weight a cell type for a marker that was never measured. State this explicitly in the rationale: "CD3D is off-panel, so its absence is not evidence against T cells."

This single rule is the most common way both a human reader and a general LLM mis-call a targeted-panel cluster. Enforce it every time.

### 3c. Assign the cell type

1. **Strong canonical markers at the top of the cluster's `glm_coef` ranking, high `pearson`, coherent** → assign with high confidence.
2. **Multiple genes from one pathway but no single canonical marker** → assign with medium confidence; name the evidence.
3. **Mixed signals from multiple lineages** → consider: a **doublet** (markers from 2+ distinct lineages spatially co-mixed), a **transitional/EMT** state, an **injury/stress** signature overlaid on a real lineage, or **panel artifacts**.
4. **Stress/injury signatures** (FOS, JUN, HIF1A, VEGFA, S100A9, CXCL chemokines) still sit on an underlying lineage: look past them and label "Stressed [Lineage]".
5. **Proliferating clusters** (MKI67, TOP2A, CDK1, CCNB1): find the lineage markers alongside and label "Proliferating [Lineage]".

### 3d. Assign confidence: the rubric on jazzPanda's spatial signal

Confidence is **anchored on `glm_coef` directly: a larger `glm_coef` means a stronger, more confident marker; a smaller one means less confident.** jazzPanda's `top_result` is already thresholded (markers below the coefficient cutoff, ~0.2, are returned as `NoSig`), so every assigned marker is a real one and its `glm_coef` magnitude is its strength. Corroborate with `pearson` and canonical agreement. (Exact band edges live in `config.py` and are tuned against the calibration set, not hard-coded here.)

Set the cluster's confidence from the `glm_coef` of its **driving canonical marker(s)**: the on-panel canonical markers supporting the assigned type: larger coefficient → higher band:

| Confidence | Condition |
|---|---|
| **Very High** | A strong canonical marker with a large `glm_coef` and high `pearson`; multiple canonical markers agree; coherent story |
| **High** | Strong canonical `glm_coef` support, minor ambiguity |
| **Medium-High** | Moderate `glm_coef` canonical support, or fewer canonical markers |
| **Medium** | Weak-but-real `glm_coef` support, or the top genes are non-specific / mixed |
| **Low** | Driving genes are `NoSig` or near the cutoff, or no canonical marker supports the call: **set `verify = TRUE`** |

Because `glm_coef` partly reflects cluster size (larger clusters yield larger coefficients), small/rare clusters read as lower-confidence: that is intended, and such clusters should carry `verify = TRUE`.

Modifiers:
- **Corroboration:** low `pearson` on the driving markers, or `max_gc_corr > pearson` (localizes better elsewhere), or `max_gg_corr ≈ 1` (not unique) → demote one band. A high `pearson` on a canonical marker can support one band up.
- **Few markers → verify:** a cluster with only one or two assigned markers is fragile; cap its confidence and set `verify = TRUE` regardless of coefficient.
- **Panel-absence never lowers confidence.** A missing off-panel canonical marker is not a demotion.

Confidence labels (fixed set): **Very High, High, Medium-High, Medium, Low.** Low or ambiguous clusters get `verify = TRUE`.

## Step 4: Holistic review: does the whole set make sense?

After annotating every cluster individually, STOP and review the complete set together. Individual calls can look fine in isolation but tell an incoherent story as a whole. Revise as needed.

- **Expected composition.** Are the tissue's expected major lineages present? (A breast tumor should have tumor epithelial, immune infiltrate, stromal/CAF, endothelial.) If a major lineage is entirely missing, either clustering missed it (revisit ambiguous clusters) or the panel doesn't capture it (note as a limitation, not an absence).
- **Plausible proportions.** If most clusters land on one lineage, something is likely wrong.
- **Redundancy.** Several clusters with near-identical markers → are they real subtypes (M1/M2/tissue-resident) or over-clustering to merge? Make labels as specific as the evidence supports.
- **Split populations / misassigned lineage.** Two clusters sharing lineage markers where one adds stress/proliferation are the same cell type in different states: label consistently. A cluster that looks immune on its top markers may be an injured stromal/epithelial population expressing inflammatory genes: revisit uncertain calls now that all clusters are visible.
- **Revise and document.** Update any call that needs it and state what changed and why in a "Holistic Review Notes" section, e.g.: *"Revised c6 from Activated Stroma to Injured Stromal Progenitors: c0 already captures quiescent stroma, and c6's inflammatory markers fit an injury state in the same lineage."*

## Step 5: Literature grounding and citation

When you assert a marker→cell-type association or any interpretive claim:

- Look it up **live** through the PubMed/bioRxiv connector and cite a **real, resolvable** PMID/DOI the biologist can open. Never write an identifier from memory.
- Prefer tissue-specific evidence. When the literature is split, report the **consensus and the dissent**, both cited, rather than smoothing it over. When it is thin, say so.
- The connector's job is reconciliation, not decoration: give, for this marker in this tissue, what agrees, what dissents, how recent, and how it squares with jazzPanda's numbers.

## Step 6: Follow-up, override, and lab memory

The process is iterative and the biologist's judgment is authoritative.

- **On override** ("in our breast TME this signature is real", "PDGFRB marks CAFs here, not pericytes"): use the biologist's call AND cross-check the literature, reporting agreement and dissent with real citations, and keep the call **with the tension visible**. Never a bare "got it", never a silent overrule.
- **Capture at the override**, because that is where the lab's knowledge diverges from the default. Record it as a small structured note: `{ claim, scope: cluster|dataset|lab, basis: paper|own_validation|convention, status: firm|tentative, tension: [citations] }`: dated and attributed. Ask scope and basis in two taps, not a form.
- **Scope is enforced.** A cluster- or dataset-scoped note must not fire elsewhere. **Cite the note when you use it**, and show any attached tension. This is a knowledge layer the lab owns; it does not make the model learn.

## Step 7: Produce outputs

### Output 1: Annotation summary (Markdown)
Header (tissue, platform, panel + gene count, clustering, "Marker method: jazzPanda spatial markers"); a table (cluster | lineage | cell type | confidence | top markers | key evidence with `glm_coef`/`pearson`); detailed notes by lineage; a Flagged Clusters section (`verify = TRUE` and why); a Panel Limitations section; and Holistic Review Notes.

### Output 2: Annotation CSV
Columns, in this exact order (see `references/output_template.md`):
```
cluster,cell_type,cell_type_short,confidence,confidence_score,key_markers,notes,category,lineage,exclude,verify
```
- `confidence` = text label (e.g. "High"); `confidence_score` = decimal 0–1; `key_markers` = top 3–5 by `glm_coef`; `cell_type_short` = `Prefix_Descriptor`; `exclude` = TRUE for off-target/doublet; `verify` = TRUE for uncertain/low/small-n.

### Output 3: R integration
Ready-to-use R to map the CSV onto the object identities and metadata (see `references/output_template.md`).

### Output 4: Per-marker biology notes (cited)
For a dataset, generate a short **grounded biology note for every assigned marker gene** as an automated pipeline step (the per-dataset pipeline's notes stage writes the per-gene notes, each grounded in that gene's jazzPanda evidence, read by the evidence-table biology column). Each note states the gene's core biological role and its relevance to that cluster's cell-type identity, and flags a specificity caveat only when the gene's own evidence (`max_gc_corr > pearson`) shows it also clearly marks another lineage. Rules: cite exactly one **real, live-looked-up PubMed paper** per note (never from memory); do **not** restate any jazzPanda numbers (they are already shown in the table); keep it to one crisp sentence; never use an em dash. A note whose literature is thin says so honestly and carries no citation.

## Naming conventions

`cell_type_short` uses `Prefix_Descriptor`. Universal prefixes: `Str_` (stroma/mesenchymal), `Endo_` (endothelial), `Imm_` (immune; sub-prefix `Imm_Mac/Imm_T/Imm_B/Imm_NK/Imm_DC/Imm_Mast`), `Epi_` (epithelial, non-tumor), `Tum_` (tumor), `Prolif_` (proliferating, lineage unclear), `Offtarget_`. State suffixes: `_prolif`, `_stress`, `_hypox`, `_act`, `_SASP`. Define tissue-specific prefixes at the start (e.g. breast: `Lum_`, `Bas_`, `CAF_`, `Myoepi_`) and stay consistent. Follow the biologist's convention if they have one.

## jazzPanda-specific caveats (state these when relevant)

- **Linearity + spatial separability.** jazzPanda assumes a marker's spatial pattern rises linearly with its cluster's, and performs best when clusters are spatially separable. Spatially overlapping clusters lose power and may need merging. (Paper pp. 5, 28.)
- **Cluster-size effect.** Larger clusters yield more and stronger markers; small/rare clusters yield fewer and less stable ones: reflected in the small-n fallback (Step 3d).
- **Grid choice matters.** Very fine grids go near-binary and lose global pattern; very coarse grids lose spatial detail. The precomputed result already fixed this; do not re-bin.
- **`max_gg_corr`/`max_gc_corr` semantics** come from the package documentation/source, not the paper prose: treat them as the package defines them (above).

## Quality checklist

Before presenting:
- [ ] Every cluster annotated; none skipped.
- [ ] Every number traces to jazzPanda output or the panel: nothing invented.
- [ ] Panel-absence rule applied: no cell type down-weighted for an off-panel marker; off-panel markers named as "not measured".
- [ ] Confidence set by `glm_coef` magnitude (bigger = more confident), corroborated by `pearson`; fragile/few-marker clusters flagged `verify = TRUE`.
- [ ] Holistic review done; the full set is biologically coherent; revisions documented.
- [ ] Every interpretive claim carries a real, connector-fetched citation; no PMID from memory.
- [ ] Any lab note used is cited, within its scope, with its tension shown.
- [ ] Ambiguous/low/small-n clusters flagged `verify = TRUE`.
- [ ] CSV has the exact columns and is R-importable.
