"""Panoscope constants and configuration.

Single source of truth for paths, the authoritative cluster->cell-type key,
model name, confidence bands, and the demo marker set.

Grounding discipline: DEMO_MARKERS is DERIVED at import time by reading the two
source files (the thresholded jazzPanda top-marker table and the Xenium panel
list). It is never a hand-typed literal, so it cannot drift from source. If a
required source file is missing, import fails loudly with a named error rather
than silently substituting a stale list.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Project root = parent of the agent/ package directory.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Raw input roots (these directory names contain spaces / underscores on disk).
RAW_JZ: str = "jazzPanda output"
RAW_XEN: str = "Raw_data_Xenium_hbreast_sample1"

RAW_JZ_DIR: Path = PROJECT_ROOT / RAW_JZ
RAW_XEN_DIR: Path = PROJECT_ROOT / RAW_XEN

# Exact raw file paths (constants — quoted on disk because of the space in RAW_JZ).
MARKERS_TOP_CSV: Path = RAW_JZ_DIR / "jazzPanda_top_marker.csv"
CLUSTERS_RDS: Path = RAW_JZ_DIR / "xenium_hbreast_clusters.Rds"
SEURAT_RDS: Path = RAW_JZ_DIR / "xenium_hbreast_seu.Rds"
RES_LST_RDS: Path = RAW_JZ_DIR / "xenium_hbreast_jazzPanda_res_lst.Rds"
PANEL_TSV: Path = RAW_XEN_DIR / "Xenium_FFPE_Human_Breast_Cancer_Rep1_panel.tsv"
TRANSCRIPTS_CSV_GZ: Path = RAW_XEN_DIR / "transcripts.csv.gz"

# Tidy output root (relative name per task; absolute helper for callers).
DATA_DIR: str = "data"
DATA_DIR_PATH: Path = PROJECT_ROOT / DATA_DIR

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
PRIMARY_MODEL: str = os.getenv("PANOSCOPE_MODEL", "claude-sonnet-4-6")

# --------------------------------------------------------------------------- #
# Dataset facts (asserted by the R prep; carried here for callers)
# --------------------------------------------------------------------------- #
DATASET_ID: str = "xenium_hbreast_sample1"
SAMPLE1_N_CELLS: int = 158_379  # clusters.Rds sample1 cell count (assert downstream)
PANEL_GENE_COUNT: int = 280  # informational only, NOT asserted — panel size is dataset-dependent (dynamic across spatial platforms). Analyzed panel here = 280 standard genes (Seurat/jazzPanda); the TSV lists 313 (280 + 33 "Custom" add-on), Custom excluded from the panel-absence set.

# --------------------------------------------------------------------------- #
# Authoritative cluster -> cell type key (from clusters.Rds `anno`)
# --------------------------------------------------------------------------- #
# Prefix convention (from the jazzpanda-markers SKILL naming section):
#   Tum_ tumor, Str_ stroma, Imm_Mac/Imm_T/Imm_B/Imm_DC/Imm_Mast immune subsets,
#   Myoepi_ myoepithelial, Endo_ endothelial.
CLUSTER_KEY: dict[str, dict[str, str]] = {
    "c1": {"cell_type": "Tumor",         "cell_type_short": "Tum_Epi",  "category": "Epithelial",  "lineage": "Epithelial"},
    "c2": {"cell_type": "Stromal",       "cell_type_short": "Str_Fib",  "category": "Stromal",     "lineage": "Mesenchymal"},
    "c3": {"cell_type": "Macrophages",   "cell_type_short": "Imm_Mac",  "category": "Immune",      "lineage": "Myeloid"},
    "c4": {"cell_type": "Myoepithelial", "cell_type_short": "Myoepi_",  "category": "Epithelial",  "lineage": "Epithelial"},
    "c5": {"cell_type": "T_Cells",       "cell_type_short": "Imm_T",    "category": "Immune",      "lineage": "Lymphoid"},
    "c6": {"cell_type": "B_Cells",       "cell_type_short": "Imm_B",    "category": "Immune",      "lineage": "Lymphoid"},
    "c7": {"cell_type": "Endothelial",   "cell_type_short": "Endo_",    "category": "Endothelial", "lineage": "Endothelial"},
    "c8": {"cell_type": "Dendritic",     "cell_type_short": "Imm_DC",   "category": "Immune",      "lineage": "Myeloid"},
    "c9": {"cell_type": "Mast_Cells",    "cell_type_short": "Imm_Mast", "category": "Immune",      "lineage": "Myeloid"},
}
KNOWN_CLUSTERS: frozenset[str] = frozenset(CLUSTER_KEY)
CLUSTER_ORDER: tuple[str, ...] = tuple(f"c{i}" for i in range(1, 10))

# --------------------------------------------------------------------------- #
# Confidence bands (TUNABLE) — glm_coef magnitude -> label.
# Ordered high->low. NoSig / None coef -> "Low". Calibrated against the
# calibration set in a later phase; these are the initial proposal.
# --------------------------------------------------------------------------- #
CONFIDENCE_BANDS: dict[str, float] = {
    "Very High": 10.0,
    "High": 5.0,
    "Medium-High": 2.5,
    "Medium": 1.0,
    "Low": 0.0,
}
# Fixed score anchor per band (matches the BLUEPRINT SCORE_MAP contract).
SCORE_MAP: dict[str, float] = {
    "Very High": 0.95,
    "High": 0.85,
    "Medium-High": 0.70,
    "Medium": 0.55,
    "Low": 0.30,
}


def band_for_coef(glm_coef: float | None, top_cluster: str | None = None) -> str:
    """Map a glm_coef magnitude to a confidence label.

    NoSig / None / non-positive -> "Low". Bigger coef -> higher band.
    Thresholds come from CONFIDENCE_BANDS (tunable). Pure; no source lookup.
    """
    if glm_coef is None or (top_cluster is not None and top_cluster == "NoSig"):
        return "Low"
    coef = abs(float(glm_coef))
    for label in ("Very High", "High", "Medium-High", "Medium"):
        if coef >= CONFIDENCE_BANDS[label]:
            return label
    return "Low"


# --------------------------------------------------------------------------- #
# DEMO_MARKERS — top 3 ON-PANEL markers per cluster by glm_coef.
# DERIVED from the source files (never a literal). ~26 genes (c9 Mast has only
# 2 on-panel assigned markers, so it contributes 2 rather than 3).
# --------------------------------------------------------------------------- #
DEMO_MARKERS_PER_CLUSTER: int = 3


def _load_panel_gene_set() -> frozenset[str]:
    """Read the panel TSV and return the frozenset of gene names.

    Raises FileNotFoundError (named) if the panel file is absent.
    """
    if not PANEL_TSV.exists():
        raise FileNotFoundError(
            f"[config] panel file missing: {PANEL_TSV} — cannot derive the panel "
            f"gene set or DEMO_MARKERS. Check RAW_XEN path."
        )
    panel = pd.read_csv(PANEL_TSV, sep="\t")
    for _col in ("Name", "Annotation"):
        if _col not in panel.columns:
            raise ValueError(
                f"[config] panel file {PANEL_TSV} has no {_col!r} column; got {list(panel.columns)}"
            )
    # The analyzed panel is the 280 standard genes. The 33 "Custom" add-on genes
    # were physically measured but excluded from the jazzPanda/Seurat analysis, so
    # they are NOT part of the panel-absence set (a gene jazzPanda never modeled
    # cannot be judged present or absent).
    # Panel size is dataset-dependent (dynamic across spatial platforms), so we do
    # NOT assert a fixed count — derive the set from whatever the file lists.
    panel = panel[panel["Annotation"].astype(str) != "Custom"]
    return frozenset(panel["Name"].astype(str))


def _derive_demo_markers() -> list[str]:
    """Top-3 on-panel markers per cluster (c1..c9) by glm_coef, in cluster order.

    Reads the ALREADY-thresholded jazzPanda top-marker CSV straight (index_col=0
    drops the blank index column). NoSig rows and off-panel genes are excluded.
    Order within a cluster is glm_coef descending; clusters are emitted c1..c9.
    """
    if not MARKERS_TOP_CSV.exists():
        raise FileNotFoundError(
            f"[config] jazzPanda top-marker file missing: {MARKERS_TOP_CSV} — "
            f"cannot derive DEMO_MARKERS. Check RAW_JZ path."
        )
    markers = pd.read_csv(MARKERS_TOP_CSV, index_col=0)
    expected_cols = {"gene", "top_cluster", "glm_coef", "pearson", "max_gg_corr", "max_gc_corr"}
    missing = expected_cols - set(markers.columns)
    if missing:
        raise ValueError(
            f"[config] top-marker CSV {MARKERS_TOP_CSV} missing columns {missing}; "
            f"got {list(markers.columns)}"
        )

    panel_genes = _load_panel_gene_set()

    on_panel = markers[
        (markers["top_cluster"] != "NoSig")
        & (markers["gene"].isin(panel_genes))
        & (markers["glm_coef"] > 0)
    ]

    demo: list[str] = []
    for cluster in CLUSTER_ORDER:
        sub = on_panel[on_panel["top_cluster"] == cluster].sort_values(
            "glm_coef", ascending=False
        )
        demo.extend(sub["gene"].head(DEMO_MARKERS_PER_CLUSTER).tolist())
    return demo


# Computed once at import.
DEMO_MARKERS: list[str] = _derive_demo_markers()


if __name__ == "__main__":
    print("DEMO_MARKERS =", DEMO_MARKERS)
    print("count =", len(DEMO_MARKERS))
