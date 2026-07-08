# Output Templates — jazzPanda Cell-Type Annotation

Templates for annotating clusters from **jazzPanda spatial marker output** on imaging-based spatial transcriptomics (Xenium, CosMx, MERSCOPE). Evidence is expressed with jazzPanda's statistics (`glm_coef`, `pearson`) — never mean-expression `pct.1`/`pct.2`/`avg_log2FC`. All example values below are **illustrative**, not real calls.

## Template 1: Annotation summary (Markdown)

```markdown
# Cell Type Annotation Summary
[Tissue type] | [Platform] [Panel, N genes] | [Clustering method] [resolution]

## Dataset Overview
- **Tissue:** [tissue type]
- **Platform:** [Xenium/CosMx/MERSCOPE]
- **Panel:** [panel name, number of genes]
- **Samples:** [N samples, conditions]
- **Clustering:** [method] resolution [X], [N] clusters
- **Marker method:** jazzPanda spatial markers (glm), top markers by glm_coef

## Annotation Table

| Cluster | Lineage | Cell Type | Confidence | Top 5 Markers | Key Evidence |
|---------|---------|-----------|------------|---------------|--------------|
| c1 | Epithelial | [cell type] | [level] | [GENE1, ...] | [brief evidence: glm_coef + pearson, panel-absence note] |
| ... | ... | ... | ... | ... | ... |

## Detailed Notes by Lineage

### [Lineage 1] ([cluster list])
[For each cluster: the driving markers with their glm_coef and pearson, whether the signal is spatially specific, any max_gc_corr / max_gg_corr caveat, and — critically — which canonical markers are OFF-PANEL and therefore uninformative rather than absent.]

### Flagged Clusters
[Every cluster with verify=TRUE and why: NoSig-dominated, no canonical support, small-n (unstable ranking), or spatial-specificity caveat.]

### Panel Limitations
[Canonical markers that are not on this panel, per lineage. State that their absence is uninformative. This is central for targeted panels (~300–5000 genes).]

### Holistic Review Notes
[Any revisions made after reviewing all clusters together: original call, revised call, reason. If nothing changed, briefly say why the set is internally consistent.]
```

Save as: `cell_type_annotation_summary.md`

---

## Template 2: Annotation CSV

Header row (exact order):
```
cluster,cell_type,cell_type_short,confidence,confidence_score,key_markers,notes,category,lineage,exclude,verify
```

Example rows (illustrative — breast Xenium):
```csv
c1,Tumor epithelial (HER2+),Tum_HER2,Very High,0.96,"ERBB2, KRT7, EPCAM","ERBB2 top by glm_coef (21.4, pearson 0.91); epithelial signature coherent",Tumor,Epithelial,FALSE,FALSE
c2,Stromal / fibroblast,Str_Fib,High,0.88,"LUM, POSTN, PDGFRA","LUM/POSTN lead (glm 18.0/15.8); COL1A1 and VIM are OFF-PANEL, so their absence is not evidence against fibroblasts",Stromal,Mesenchymal,FALSE,FALSE
c9,Mast cells,Imm_Mast,Medium-High,0.72,"CPA3, CTSG","CPA3 canonical (glm 2.0) but only 2 markers and low pearson (0.42); small-n, ranking unstable",Immune,Myeloid,FALSE,TRUE
cX,Off-target / aberrant,Offtarget,Low,0.35,"GENE1, GENE2","Driving genes NoSig; no coherent lineage; likely artifact",Off-target,Unknown,TRUE,TRUE
```

Rules:
- Quote any field containing commas (especially `key_markers` and `notes`).
- `confidence` = one of Very High / High / Medium-High / Medium / Low; `confidence_score` = decimal 0–1.
- `key_markers` = top 3–5 genes by `glm_coef`.
- `cell_type_short` = `Prefix_Descriptor` (see SKILL.md Naming Conventions).
- `exclude` = TRUE for off-target/doublet clusters to drop downstream.
- `verify` = TRUE for uncertain, low-confidence, or small-n clusters.
- Cluster IDs match jazzPanda's `top_cluster` labels exactly (e.g. `c1`…`cN`).

Save as: `cell_type_annotations.csv`

---

## Template 3: R integration

```r
# ============================================================
# Panoscope / jazzPanda annotation integration
# [Tissue] | [Platform] [Panel] | [Date]
# ============================================================

annotations <- read.csv("cell_type_annotations.csv", stringsAsFactors = FALSE)

# jazzPanda clusters (c1..cN) map to Idents of the Seurat/SPE object.
# --- Option A: rename identities ---
new_ids <- setNames(annotations$cell_type_short, annotations$cluster)
seurat_obj <- RenameIdents(seurat_obj, new_ids)

# --- Option B: add as metadata columns (match on the cluster label column) ---
cluster_col <- "cluster"   # adjust to your object's cluster column
for (col in c("cell_type", "cell_type_short", "lineage", "category")) {
  seurat_obj[[col]] <- annotations[[col]][
    match(seurat_obj@meta.data[[cluster_col]], annotations$cluster)
  ]
}

# --- Exclude off-target / doublet clusters ---
drop <- annotations$cluster[annotations$exclude == TRUE]
seurat_obj_clean <- subset(seurat_obj,
  !(seurat_obj@meta.data[[cluster_col]] %in% drop))

# --- Verification plot ---
DimPlot(seurat_obj, group.by = "cell_type_short", label = TRUE, repel = TRUE) + NoLegend()
```

Save as: `cell_type_annotations.R`

---

## Iterative revision

When the biologist requests changes:
1. Update all three outputs (summary, CSV, R) consistently.
2. Note what changed and why in the summary's Holistic Review Notes.
3. Re-export the CSV with the same filename.
4. If the change came from an override, capture it as a scoped, dated, attributed lab note with any literature tension attached (see SKILL.md Step 6).
```
