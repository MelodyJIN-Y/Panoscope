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

# Committed tidy files the app reads at runtime (incl. DEMO_MARKERS / panel-set
# derivation) — so a fresh clone or CI works WITHOUT the raw (gitignored) inputs.
TIDY_MARKERS_CSV: Path = DATA_DIR_PATH / "jazzpanda" / "markers_top.csv"
TIDY_PANEL_PARQUET: Path = DATA_DIR_PATH / "panels" / "panel.parquet"

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
PRIMARY_MODEL: str = os.getenv("PANOSCOPE_MODEL", "claude-sonnet-4-6")

# --------------------------------------------------------------------------- #
# Active dataset (selectable via PANOSCOPE_DATASET; one dataset per process).
# Everything below derives from THIS dataset's files, never a hardcoded literal.
# --------------------------------------------------------------------------- #
# The bundled demo dataset. Its identity is fixed (the bundled fallback maps /
# legacy files belong to it); any OTHER dataset derives everything from its own tree.
BUNDLED_DEMO_ID: str = "xenium_hbreast_sample1"
DATASET_ID: str = os.getenv("PANOSCOPE_DATASET", BUNDLED_DEMO_ID)
SAMPLE1_N_CELLS: int = 158_379  # demo cell count (informational only)
PANEL_GENE_COUNT: int = 280     # informational only; panel size is dataset-dependent

# Active-dataset input resolution: prefer the per-dataset tree, else the bundled
# legacy flat files (so the demo works with or without a built tree).
_DATASET_INPUTS: Path = DATA_DIR_PATH / "datasets" / DATASET_ID / "inputs"


def _active_input(name: str, legacy: Path) -> Path:
    cand = _DATASET_INPUTS / name
    return cand if cand.exists() else legacy


ACTIVE_MARKERS_CSV: Path = _active_input("markers_top.csv", TIDY_MARKERS_CSV)
ACTIVE_PANEL_PARQUET: Path = _active_input("panel.parquet", TIDY_PANEL_PARQUET)


def _derive_cluster_order() -> tuple[str, ...]:
    """Cluster ids in the active dataset's markers (top_cluster != NoSig), sorted
    naturally (c1, c2, ... c10). Derived from the data, never a hardcoded literal."""
    import re

    if not ACTIVE_MARKERS_CSV.exists():
        raise FileNotFoundError(
            f"[config] markers missing: {ACTIVE_MARKERS_CSV} (run the data prep)"
        )
    m = pd.read_csv(ACTIVE_MARKERS_CSV)
    labels = {str(x) for x in m["top_cluster"].unique() if str(x) != "NoSig"}

    def _key(c: str):
        mm = re.match(r"c(\d+)$", c)
        return (0, int(mm.group(1))) if mm else (1, c)

    return tuple(sorted(labels, key=_key))

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
# Derived from the active dataset's markers (never a hardcoded c1..c9 literal).
CLUSTER_ORDER: tuple[str, ...] = _derive_cluster_order()
KNOWN_CLUSTERS: frozenset[str] = frozenset(CLUSTER_ORDER)

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
# Gene-set enrichment rubric (second workflow; TUNABLE, calibrate on real data).
# Bands are keyed by score_kind (the schema stays method-agnostic); the jazzPanda
# competitive test_statistic (z-like, ~2–30) is NOT a glm_coef, so it has its own
# thresholds.
# --------------------------------------------------------------------------- #
ENRICHMENT_BANDS: dict[str, dict[str, float]] = {
    "jazzpanda_enrichment": {"Very High": 12.0, "High": 8.0, "Medium-High": 5.0, "Medium": 3.0, "Low": 0.0},
}
ENRICH_Q_MAX: float = 0.05      # q below this + gate -> "enriched"
SUGGESTIVE_Q_MAX: float = 0.25  # q in [ENRICH_Q_MAX, this] + gate -> "suggestive · verify"
MIN_PANEL_HITS: int = 3         # set genes on panel; below -> untestable
MIN_LEADING_EDGE: int = 3       # driving genes; below -> not defensible


def band_for_enrichment(score: float | None, score_kind: str) -> str:
    """Map an enrichment score to a confidence label, per method. Bigger -> higher.

    Unknown score_kind or None score -> "Low". Pure; thresholds from
    ENRICHMENT_BANDS (tunable). Never conflates the two methods' score scales.
    """
    bands = ENRICHMENT_BANDS.get(score_kind)
    if score is None or bands is None:
        return "Low"
    s = float(score)
    for label in ("Very High", "High", "Medium-High", "Medium"):
        if s >= bands[label]:
            return label
    return "Low"


# --------------------------------------------------------------------------- #
# DEMO_MARKERS — top 3 ON-PANEL markers per cluster by glm_coef.
# DERIVED from the source files (never a literal). ~26 genes (c9 Mast has only
# 2 on-panel assigned markers, so it contributes 2 rather than 3).
# --------------------------------------------------------------------------- #
DEMO_MARKERS_PER_CLUSTER: int = 3


def _load_panel_gene_set() -> frozenset[str]:
    """Return the analyzed-panel gene set from the committed tidy panel file.

    Reads data/panels/panel.parquet (committed) rather than the raw, gitignored
    TSV, so the app / DEMO_MARKERS derivation works from a fresh clone or in CI.
    The 33 "Custom" add-on genes are already excluded when the tidy panel is built,
    and panel size is dataset-dependent (dynamic), so no fixed count is asserted.
    """
    if not ACTIVE_PANEL_PARQUET.exists():
        raise FileNotFoundError(
            f"[config] panel missing: {ACTIVE_PANEL_PARQUET} — produced by the data "
            f"prep (inputs/panel.parquet). Run the prep before importing."
        )
    panel = pd.read_parquet(ACTIVE_PANEL_PARQUET)
    if "gene" not in panel.columns:
        raise ValueError(
            f"[config] panel {ACTIVE_PANEL_PARQUET} has no 'gene' column; got {list(panel.columns)}"
        )
    return frozenset(panel["gene"].astype(str))


def _derive_demo_markers() -> list[str]:
    """Top-3 on-panel markers per cluster (c1..c9) by glm_coef, in cluster order.

    Reads the committed tidy top-marker table (data/jazzpanda/markers_top.csv,
    already thresholded, proper columns) so DEMO_MARKERS derives from committed
    data on a fresh clone / CI. NoSig rows and off-panel genes are excluded.
    Order within a cluster is glm_coef descending; clusters are emitted c1..c9.
    """
    if not ACTIVE_MARKERS_CSV.exists():
        raise FileNotFoundError(
            f"[config] top-marker file missing: {ACTIVE_MARKERS_CSV} — produced by "
            f"the data prep (inputs/markers_top.csv). Run the prep before importing."
        )
    markers = pd.read_csv(ACTIVE_MARKERS_CSV)
    expected_cols = {"gene", "top_cluster", "glm_coef"}
    missing = expected_cols - set(markers.columns)
    if missing:
        raise ValueError(
            f"[config] markers CSV {ACTIVE_MARKERS_CSV} missing columns {missing}; "
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
