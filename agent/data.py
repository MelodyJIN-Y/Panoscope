"""Pure loader for Panoscope tidy data.

This module is the single interface every downstream module reads through. It
touches ONLY tidy files under ``data/`` (never the raw ``.Rds``/``.csv.gz``
sources) and it NEVER computes a new statistic — percentiles, counts, and
densities are all precomputed by sibling prep steps. Its job is to hand back
exactly what is on disk, plus the one grounding primitive the whole app leans
on: :func:`panel_contains` (the panel-absence source of truth).

Grounding discipline:
- ``panel_contains`` is the SOLE absence source; it derives a frozenset FROM the
  loaded panel file (never a literal), case-insensitively.
- A different ``bin_um`` returns a different precomputed frame, never a
  recomputed value.
- Loaders fail loudly (named ``FileNotFoundError``) when a required tidy file is
  missing, naming the prep step that produces it, rather than faking a view.
"""

from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from agent import config as cfg

# --------------------------------------------------------------------------- #
# Tidy file locations (all under data/)
# --------------------------------------------------------------------------- #
# Per-dataset pipeline tree (data/datasets/<id>/). Each artifact is resolved
# tree-first with a fall back to the legacy flat path, so the app reads the
# pipeline output where it exists and keeps working mid-migration. Resolved at
# import: the tree either exists (pipeline has run) or it does not.
_DATASET_DIR = cfg.DATA_DIR_PATH / "datasets" / cfg.DATASET_ID


def _resolved(tree_rel: str, legacy):
    """Return the per-dataset tree path if present, else the legacy flat path."""
    cand = _DATASET_DIR / tree_rel
    return cand if cand.exists() else legacy


_MARKERS_TOP = _resolved("inputs/markers_top.csv", cfg.DATA_DIR_PATH / "jazzpanda" / "markers_top.csv")
_PANEL_PARQUET = _resolved("inputs/panel.parquet", cfg.DATA_DIR_PATH / "panels" / "panel.parquet")
_CLUSTER_KEY_JSON = _resolved("inputs/cluster_key.json", cfg.DATA_DIR_PATH / "cluster_key.json")
_CELLS_PARQUET = _resolved("viz/cells.parquet", cfg.DATA_DIR_PATH / "cells" / "cells.parquet")
_CELLS_CSV = _resolved("viz/cells.csv", cfg.DATA_DIR_PATH / "cells" / "cells.csv")
_UMAP_PARQUET = _resolved("viz/umap.parquet", cfg.DATA_DIR_PATH / "embeddings" / "umap.parquet")
_UMAP_CSV = _resolved("viz/umap.csv", cfg.DATA_DIR_PATH / "embeddings" / "umap.csv")
_DENSITY_DIR = _resolved("viz/hexbin", cfg.DATA_DIR_PATH / "density")

_MARKERS_TOP_COLS = [
    "gene",
    "top_cluster",
    "glm_coef",
    "pearson",
    "max_gg_corr",
    "max_gc_corr",
    "cell_type",
]
_PANEL_COLS = ["gene", "ensembl_id", "annotation"]


def _require(path, produced_by: str):
    """Fail loudly if a tidy file is missing, naming the prep step."""
    if not path.exists():
        raise FileNotFoundError(
            f"[data] required tidy file missing: {path} — produced by {produced_by}. "
            f"Run the prep step before loading."
        )
    return path


def _read_table(parquet_path, csv_path, produced_by: str) -> pd.DataFrame:
    """Read a tidy table as parquet if present, else csv; fail loudly if neither.

    The R prep writes ``.csv`` (no ``arrow`` package in R). Preferring parquet
    lets a later optional conversion speed loads without changing callers.
    """
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"[data] required tidy file missing: {parquet_path} (or .csv) — "
        f"produced by {produced_by}. Run the prep step before loading."
    )


# --------------------------------------------------------------------------- #
# Markers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_markers() -> pd.DataFrame:
    """Return the full jazzPanda top-marker table (280 rows, incl. NoSig).

    Columns: gene, top_cluster, glm_coef, pearson, max_gg_corr, max_gc_corr,
    cell_type. Cached; callers must not mutate the returned frame.
    """
    path = _require(_MARKERS_TOP, "scripts prep (markers_top.csv)")
    df = pd.read_csv(path)
    missing = set(_MARKERS_TOP_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[data] markers_top.csv missing columns {missing}")
    return df[_MARKERS_TOP_COLS]


def get_cluster_markers(cluster: str, include_nosig: bool = False) -> pd.DataFrame:
    """Return one cluster's assigned markers, sorted by glm_coef descending.

    Raises KeyError for an unknown cluster id. NoSig rows never belong to a
    real cluster (NoSig is its own top_cluster label), so an assigned cluster's
    frame is always signal rows only. ``include_nosig`` is accepted for
    interface symmetry but does not mix NoSig into a real cluster's rows.
    """
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(
            f"[data] unknown cluster {cluster!r}; known clusters: "
            f"{sorted(cfg.KNOWN_CLUSTERS)}"
        )
    df = load_markers()
    sub = df[df["top_cluster"] == cluster]
    if not include_nosig:
        sub = sub[sub["top_cluster"] != "NoSig"]
    return sub.sort_values("glm_coef", ascending=False).reset_index(drop=True)


def get_marker(gene: str) -> pd.Series | None:
    """Return the marker row for ``gene`` (case-insensitive), or None."""
    df = load_markers()
    hit = df[df["gene"].astype(str).str.upper() == gene.upper()]
    if hit.empty:
        return None
    return hit.iloc[0]


# --------------------------------------------------------------------------- #
# Cluster key
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_cluster_key() -> dict[str, dict]:
    """Return the 9-entry cluster -> {cell_type, cell_type_short, category, lineage}."""
    path = _require(_CLUSTER_KEY_JSON, "scripts prep (cluster_key.json)")
    with open(path) as fh:
        key = json.load(fh)
    if len(key) != 9:
        raise ValueError(f"[data] cluster_key.json must have 9 entries, got {len(key)}")
    return key


def cell_type_for(cluster: str) -> str:
    """Return the cell type label for a cluster id. KeyError if unknown."""
    key = load_cluster_key()
    if cluster not in key:
        raise KeyError(f"[data] unknown cluster {cluster!r}")
    return key[cluster]["cell_type"]


# --------------------------------------------------------------------------- #
# Panel (the absence primitive)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_panel() -> pd.DataFrame:
    """Return the panel table (280 analyzed genes): gene, ensembl_id, annotation."""
    path = _require(_PANEL_PARQUET, "scripts prep (panel.parquet)")
    df = pd.read_parquet(path)
    missing = set(_PANEL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[data] panel.parquet missing columns {missing}")
    return df[_PANEL_COLS]


@lru_cache(maxsize=1)
def _panel_gene_set() -> frozenset[str]:
    """Case-insensitive frozenset of panel gene names, derived FROM the file."""
    return frozenset(load_panel()["gene"].astype(str).str.upper())


def panel_contains(gene: str) -> bool:
    """True iff ``gene`` is on the panel (case-insensitive). THE absence primitive.

    O(1) frozenset membership over a set derived FROM the loaded panel file. The
    absence of a gene NOT on the panel is never evidence against a cell type —
    callers use this to make exactly that call.
    """
    return gene.upper() in _panel_gene_set()


def panel_annotation(gene: str) -> str | None:
    """Return the panel Annotation for ``gene`` (case-insensitive), or None off-panel."""
    df = load_panel()
    hit = df[df["gene"].astype(str).str.upper() == gene.upper()]
    if hit.empty:
        return None
    return str(hit.iloc[0]["annotation"])


# --------------------------------------------------------------------------- #
# Spatial readers (files produced by sibling agents — pure readers only)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def load_cells() -> pd.DataFrame:
    """Return per-cell coordinates + cluster labels (cell_id, cluster, x, y).

    Reads data/cells/cells.parquet (produced by the R prep step). Fails loudly
    with a named FileNotFoundError if the file is missing.
    """
    return _read_table(_CELLS_PARQUET, _CELLS_CSV, "scripts/prep_data.R (cells.parquet/.csv)")


def get_cluster_cells(cluster: str) -> pd.DataFrame:
    """Return only the cells belonging to ``cluster``. KeyError if unknown id."""
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(
            f"[data] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
        )
    cells = load_cells()
    return cells[cells["cluster"] == cluster].reset_index(drop=True)


@lru_cache(maxsize=1)
def load_umap() -> pd.DataFrame:
    """Return UMAP coordinates per cell (cell_id, umap_1, umap_2, cluster).

    Reads data/embeddings/umap.parquet (produced by the R prep step).
    """
    return _read_table(_UMAP_PARQUET, _UMAP_CSV, "scripts/prep_data.R (umap.parquet/.csv)")


def get_density(gene: str, bin_um: int = 50) -> pd.DataFrame:
    """Return precomputed hex-bin density for one marker at one bin size.

    Reads data/density/{gene}_{bin_um}um.parquet (produced by
    scripts/precompute_density.py). A different ``bin_um`` reads a different
    precomputed frame — this reader NEVER re-bins. FileNotFoundError names the
    missing frame so the caller can fall back to the cell map.
    """
    path = _DENSITY_DIR / f"{gene}_{int(bin_um)}um.parquet"
    _require(path, "scripts/precompute_density.py (density hexbins)")
    return pd.read_parquet(path)
