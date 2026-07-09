#!/usr/bin/env Rscript
# =============================================================================
# prep_data.R  —  Panoscope R prep (cells + UMAP + marker expression)
#
# ONE-TIME: reads the precomputed jazzPanda .Rds inputs and writes tidy files
# under data/ for the Python loader. SAMPLE1 ONLY.
#
# Confident floor: this script only RESHAPES source data. It never invents a
# value. Every derived cell_id is asserted to be a positive integer taken from
# the "<int>_1" cell id. Row counts are asserted against the locked facts.
#
# Outputs (CSV, because arrow is not installed in this R; the Python loader
# adapts — the BLUEPRINT loader reads .parquet OR .csv):
#   data/cells/cells.csv                 cell_id, cluster, x, y      (158,379 rows)
#   data/embeddings/umap.csv             cell_id, umap_1, umap_2, cluster
#   data/embeddings/marker_expr.csv      cell_id + one col per DEMO_MARKER
#
# Run:  Rscript scripts/prep_data.R
# =============================================================================

suppressPackageStartupMessages(library(SeuratObject))

# --------------------------------------------------------------------------- #
# 0. Constants + fail-loud path asserts (P1: quote the space-containing paths)
# --------------------------------------------------------------------------- #
RAW_JZ  <- "jazzPanda output"                      # NOTE: contains a space
RAW_XEN <- "Raw_data_Xenium_hbreast_sample1"

CLUSTERS_RDS <- file.path(RAW_JZ, "xenium_hbreast_clusters.Rds")
SEURAT_RDS   <- file.path(RAW_JZ, "xenium_hbreast_seu.Rds")

OUT_CELLS      <- file.path("data", "cells", "cells.csv")
OUT_UMAP       <- file.path("data", "embeddings", "umap.csv")
OUT_MARKEREXPR <- file.path("data", "embeddings", "marker_expr.csv")

SAMPLE1_N_CELLS <- 158379L   # locked fact: clusters.Rds sample=="sample1" rows
SAMPLE1_LABEL   <- "sample1" # clusters.Rds sample column value
SAMPLE1_BIOREP  <- "hb1"     # seu.Rds biorep value for sample1

# DEMO_MARKERS — top-3 on-panel-per-cluster (c1..c9) by glm_coef, derived once
# in agent/config.py from the source files. Carried here verbatim (26 genes;
# c9 Mast contributes 2). If config's derivation changes, update this list.
DEMO_MARKERS <- c(
  "ERBB2", "KRT7", "SCD",          # c1 Tumor
  "LUM", "POSTN", "CCDC80",        # c2 Stromal
  "LYZ", "FCER1G", "CD68",         # c3 Macrophages
  "SERPINA3", "DST", "SFRP1",      # c4 Myoepithelial
  "IL7R", "PTPRC", "TRAC",         # c5 T_Cells
  "MS4A1", "BANK1", "MZB1",        # c6 B_Cells
  "AQP1", "PECAM1", "VWF",         # c7 Endothelial
  "TCL1A", "LILRA4", "SPIB",       # c8 Dendritic
  "CPA3", "CTSG"                   # c9 Mast_Cells (only 2 on-panel)
)

# --- fail-loud helpers ----------------------------------------------------- #
assert_exists <- function(path, name) {
  if (!file.exists(path)) {
    stop(sprintf("[prep_data] missing required input '%s': %s (cwd=%s)",
                 name, path, getwd()), call. = FALSE)
  }
  invisible(TRUE)
}

assert_true <- function(cond, msg) {
  if (!isTRUE(cond)) stop(sprintf("[prep_data] ASSERT FAILED: %s", msg), call. = FALSE)
  invisible(TRUE)
}

# cell_id := the integer BEFORE the first (non-leading) underscore of the
# "N_1" id. clusters.Rds ids look like "1_1"; seu.Rds names like "_1_1".
# Strip a single leading underscore, take everything before the first "_",
# assert it is a clean positive integer.
cell_id_from_name <- function(x) {
  x2  <- sub("^_", "", x)            # drop one leading underscore if present
  pre <- sub("_.*$", "", x2)         # integer part before the first underscore
  assert_true(all(grepl("^[0-9]+$", pre)),
              "some cell ids do not have a clean integer before '_'")
  id <- suppressWarnings(as.integer(pre))
  assert_true(all(!is.na(id)) && all(id > 0L),
              "some derived cell_ids are NA or non-positive")
  id
}

assert_exists(CLUSTERS_RDS, "clusters.Rds")
assert_exists(SEURAT_RDS,   "seu.Rds")

cat("== prep_data.R ==\n")
cat("cwd:", getwd(), "\n")
cat("arrow available:", requireNamespace("arrow", quietly = TRUE), "-> writing CSV\n\n")

# =========================================================================== #
# 1. CELLS  (clusters.Rds, sample1 only) -> data/cells/cells.csv
# =========================================================================== #
cat("[1] cells: reading", CLUSTERS_RDS, "\n")
clusters <- readRDS(CLUSTERS_RDS)
assert_true(is.data.frame(clusters), "clusters.Rds is not a data.frame")

needed_cols <- c("cluster", "x", "y", "cells", "sample")
assert_true(all(needed_cols %in% colnames(clusters)),
            paste("clusters.Rds missing columns; have:",
                  paste(colnames(clusters), collapse = ", ")))

cells_s1 <- clusters[clusters$sample == SAMPLE1_LABEL, , drop = FALSE]
cat("    sample1 rows:", nrow(cells_s1), "(expected", SAMPLE1_N_CELLS, ")\n")
assert_true(nrow(cells_s1) == SAMPLE1_N_CELLS,
            sprintf("sample1 row count %d != %d", nrow(cells_s1), SAMPLE1_N_CELLS))

# clusters.Rds IS the authoritative clustering (its `anno` is the CLUSTER_KEY
# source). Build the canonical cell-name -> cluster map here so the UMAP below
# labels cells with the SAME clustering as the cell map. The seu.Rds object
# carries its OWN, DIFFERENT Idents (~27% agreement) — those are NOT used for
# labeling; they are only reported as a cross-check.
# Canonical cell name string = "_" + clusters.Rds `cells` (matches seu colnames).
cells_s1$cell_name <- paste0("_", as.character(cells_s1$cells))
AUTH_CLUSTER <- setNames(as.character(cells_s1$cluster), cells_s1$cell_name)

cells_out <- data.frame(
  cell_id = cell_id_from_name(as.character(cells_s1$cells)),
  cluster = as.character(cells_s1$cluster),
  x       = as.numeric(cells_s1$x),
  y       = as.numeric(cells_s1$y),
  stringsAsFactors = FALSE
)

# integrity: unique cell_ids, no NA coords, clusters within c1..c9
assert_true(!any(is.na(cells_out$cell_id)), "cells: NA cell_id")
assert_true(!any(duplicated(cells_out$cell_id)), "cells: duplicate cell_id")
assert_true(all(!is.na(cells_out$x)) && all(!is.na(cells_out$y)), "cells: NA x/y")
assert_true(all(cells_out$cluster %in% paste0("c", 1:9)),
            "cells: cluster label outside c1..c9")

write.csv(cells_out, OUT_CELLS, row.names = FALSE)
cat("    wrote", OUT_CELLS, "rows:", nrow(cells_out), "\n")
cat("    head:\n"); print(utils::head(cells_out, 4))
cat("    cluster counts:\n"); print(table(cells_out$cluster))
cat("\n")

# =========================================================================== #
# 2. SEURAT (biorep hb1 = sample1): UMAP + marker expression
# =========================================================================== #
cat("[2] seurat: reading", SEURAT_RDS, "(large; ~344MB)\n")
seu <- readRDS(SEURAT_RDS)
assert_true(inherits(seu, "Seurat"), "seu.Rds is not a Seurat object")
assert_true("biorep" %in% colnames(seu[[]]), "seu.Rds meta has no 'biorep' column")
assert_true("umap" %in% SeuratObject::Reductions(seu), "seu.Rds has no 'umap' reduction")

# --- subset to sample1 (biorep hb1) --------------------------------------- #
hb1_cells <- colnames(seu)[seu$biorep == SAMPLE1_BIOREP]
cat("    biorep hb1 (sample1) cells:", length(hb1_cells), "\n")
assert_true(length(hb1_cells) > 0L, "no hb1 cells found in seu.Rds")
seu1 <- subset(seu, cells = hb1_cells)

# --------------------------------------------------------------------------- #
# 2a. UMAP  -> data/embeddings/umap.csv  (cell_id, umap_1, umap_2, cluster)
# --------------------------------------------------------------------------- #
um <- SeuratObject::Embeddings(seu1, "umap")
assert_true(ncol(um) == 2L, "umap embedding does not have 2 columns")
um_names <- rownames(um)  # seu cell names, format "_<int>_1" for hb1

# Authoritative cluster label = clusters.Rds label joined by cell-name string.
# Cells present in seu hb1 but dropped by the authoritative clustering get "" so
# the UMAP can show them as unassigned — never faked into a cluster.
auth_cluster <- unname(AUTH_CLUSTER[um_names])
n_labeled   <- sum(!is.na(auth_cluster))
n_unlabeled <- sum(is.na(auth_cluster))
auth_cluster[is.na(auth_cluster)] <- ""  # unassigned, honest blank

# seu Idents kept ONLY as a cross-check (a different clustering); NOT written.
seu_idents <- as.character(SeuratObject::Idents(seu1))
assert_true(length(seu_idents) == nrow(um), "seu idents length != umap rows")
shared <- !is.na(AUTH_CLUSTER[um_names])
agree_vs_seu <- mean(auth_cluster[shared] == seu_idents[shared])
cat(sprintf("    label source: clusters.Rds (authoritative). labeled=%d unassigned=%d\n",
            n_labeled, n_unlabeled))
cat(sprintf("    cross-check: authoritative vs seu Idents agreement = %.4f (seu is a separate clustering; NOT used)\n",
            agree_vs_seu))

umap_out <- data.frame(
  cell_id = cell_id_from_name(um_names),
  umap_1  = as.numeric(um[, 1]),
  umap_2  = as.numeric(um[, 2]),
  cluster = auth_cluster,
  stringsAsFactors = FALSE
)

assert_true(!any(is.na(umap_out$cell_id)), "umap: NA cell_id")
assert_true(all(!is.na(umap_out$umap_1)) && all(!is.na(umap_out$umap_2)),
            "umap: NA coordinates")
assert_true(all(umap_out$cluster %in% c(paste0("c", 1:9), "")),
            "umap: cluster label outside c1..c9 or blank")
assert_true(n_labeled == SAMPLE1_N_CELLS,
            sprintf("umap labeled-cell count %d != clusters.Rds sample1 %d",
                    n_labeled, SAMPLE1_N_CELLS))

write.csv(umap_out, OUT_UMAP, row.names = FALSE)
cat("    wrote", OUT_UMAP, "rows:", nrow(umap_out), "cols:",
    paste(colnames(umap_out), collapse = ", "), "\n")
cat("    head:\n"); print(utils::head(umap_out, 4))
cat("\n")

# --------------------------------------------------------------------------- #
# 2b. MARKER EXPRESSION -> data/embeddings/marker_expr.csv
#     normalized expression ('data' layer) per cell for DEMO_MARKERS.
#     Panel-absence discipline: only export markers actually in the assay;
#     a gene missing from the assay is reported, never faked with zeros.
# --------------------------------------------------------------------------- #
# ALL panel genes (280) so any pinned marker can drive the feature-UMAP + violin.
MARKERS_TOP_CSV <- file.path("data", "jazzpanda", "markers_top.csv")
assert_exists(MARKERS_TOP_CSV, "markers_top.csv")
PANEL_GENES <- unique(as.character(
  read.csv(MARKERS_TOP_CSV, check.names = FALSE)$gene))
assert_true(length(PANEL_GENES) > 0L, "no panel genes read from markers_top.csv")

assay <- SeuratObject::DefaultAssay(seu1)
gene_universe <- rownames(seu1)
present <- PANEL_GENES[PANEL_GENES %in% gene_universe]
missing <- PANEL_GENES[!PANEL_GENES %in% gene_universe]
if (length(missing) > 0L) {
  cat("    NOTE: panel genes not in assay (skipped, not faked):",
      length(missing), "\n")
}
assert_true(length(present) > 0L, "none of the panel genes present in assay")
cat("    panel genes present in assay:", length(present), "of",
    length(PANEL_GENES), "\n")

# 'data' = log-normalized layer. GetAssayData returns genes x cells; transpose.
expr <- SeuratObject::GetAssayData(seu1, assay = assay, layer = "data")[present, , drop = FALSE]
expr_t <- as.matrix(Matrix::t(expr))          # cells x genes
assert_true(nrow(expr_t) == length(hb1_cells),
            "marker_expr: cell count mismatch vs hb1 subset")

marker_full <- data.frame(
  cell_id = cell_id_from_name(rownames(expr_t)),
  stringsAsFactors = FALSE, check.names = FALSE
)
for (g in present) marker_full[[g]] <- as.numeric(expr_t[, g])
assert_true(!any(is.na(marker_full$cell_id)), "marker_expr: NA cell_id")
assert_true(nrow(marker_full) == nrow(umap_out),
            "marker_expr row count != umap row count (both = hb1 cells)")

# ---- cluster-STRATIFIED subsample (deterministic) so the committed matrix stays
# lean + fresh-clone-friendly; small clusters kept in full so every violin is
# legible; the feature-UMAP already downsamples the background -> visually lossless.
set.seed(0)
mk_cluster <- unname(AUTH_CLUSTER[rownames(expr_t)])
mk_cluster[is.na(mk_cluster)] <- ""            # unassigned cells kept as own group
CAP_PER_CLUSTER <- 5000L
keep_idx <- sort(unlist(
  lapply(split(seq_len(nrow(marker_full)), mk_cluster), function(idx) {
    if (length(idx) <= CAP_PER_CLUSTER) idx else sample(idx, CAP_PER_CLUSTER)
  }),
  use.names = FALSE))
marker_out <- marker_full[keep_idx, , drop = FALSE]
cat(sprintf("    stratified subsample: %d of %d hb1 cells (cap %d/cluster)\n",
            nrow(marker_out), nrow(marker_full), CAP_PER_CLUSTER))

write.csv(marker_out, OUT_MARKEREXPR, row.names = FALSE)
cat("    wrote", OUT_MARKEREXPR, "rows:", nrow(marker_out),
    "marker cols:", length(present), "\n")
cat("    head (first 6 cols):\n")
print(utils::head(marker_out[, 1:min(6, ncol(marker_out))], 4))
cat("\n")

# =========================================================================== #
# 3. Summary
# =========================================================================== #
cat("== DONE ==\n")
cat(sprintf("cells.csv        : %d rows  (assert %d) OK\n", nrow(cells_out), SAMPLE1_N_CELLS))
cat(sprintf("umap.csv         : %d rows,  %d cols  (%d authoritatively labeled, %d unassigned)\n",
            nrow(umap_out), ncol(umap_out), n_labeled, n_unlabeled))
cat(sprintf("marker_expr.csv  : %d rows (subsample),  %d marker cols (%d panel genes, %d missing from assay)\n",
            nrow(marker_out), length(present), length(PANEL_GENES), length(missing)))
