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
from functools import lru_cache, wraps

import pandas as pd

from panoscope.agent import config as cfg

# --------------------------------------------------------------------------- #
# Tidy file locations (all under data/)
# --------------------------------------------------------------------------- #
# Per-dataset pipeline tree (data/datasets/<id>/). Each artifact is resolved
# tree-first with a fall back to the legacy flat path, so the app reads the
# pipeline output where it exists and keeps working mid-migration. Resolved at
# import: the tree either exists (pipeline has run) or it does not.
def _resolved(tree_rel: str, legacy_rel: str) -> Path:
    """The current study's tree path if it exists, else the legacy flat path.

    Resolved at CALL time against the current study. It used to be resolved at
    import, which froze every path to whichever study happened to be active when
    the module was first imported -- so opening a second study silently served the
    first study's frames to everything downstream.
    """
    study = cfg.current_study()
    cand = study.dir / tree_rel
    return cand if cand.exists() else cfg.DATA_DIR_PATH / legacy_rel


def _markers_top() -> Path:
    return _resolved("inputs/markers_top.csv", "jazzpanda/markers_top.csv")


def _panel_parquet() -> Path:
    return _resolved("inputs/panel.parquet", "panels/panel.parquet")


def _cluster_key_json() -> Path:
    return _resolved("inputs/cluster_key.json", "cluster_key.json")


def _cells_parquet() -> Path:
    return _resolved("viz/cells.parquet", "cells/cells.parquet")


def _cells_csv() -> Path:
    return _resolved("viz/cells.csv", "cells/cells.csv")


def _umap_parquet() -> Path:
    return _resolved("viz/umap.parquet", "embeddings/umap.parquet")


def _umap_csv() -> Path:
    return _resolved("viz/umap.csv", "embeddings/umap.csv")


def _density_dir() -> Path:
    return _resolved("viz/hexbin", "density")

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


def _per_study(fn):
    """Cache a zero-arg reader per study.

    The readers take no arguments, so a plain lru_cache keys everything on the empty
    tuple -- one slot, shared across studies. Keying on the current study is what
    makes cross-study staleness unrepresentable rather than merely tested for.
    """
    cached = lru_cache(maxsize=16)(lambda _study: fn())

    @wraps(fn)
    def wrapper():
        return cached(cfg.current_study())

    wrapper.cache_clear = cached.cache_clear
    return wrapper


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
@_per_study
def load_markers() -> pd.DataFrame:
    """Return the full jazzPanda top-marker table (280 rows, incl. NoSig).

    Columns: gene, top_cluster, glm_coef, pearson, max_gg_corr, max_gc_corr,
    cell_type. Cached; callers must not mutate the returned frame.
    """
    path = _require(_markers_top(), "scripts prep (markers_top.csv)")
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
@_per_study
def load_cluster_key() -> dict[str, dict]:
    """Return cluster -> {cell_type, cell_type_short, category, lineage}.

    Deliberately asserts no entry count. It used to require exactly 9, which is the
    bundled demo's cluster count and nothing more general -- a hard blocker for any
    second study, and one that pipeline/stages/validate.py swallowed as a bare
    Exception so a 3-cluster study silently skipped key validation entirely. What
    matters is that the entries are well-formed, not how many there are.
    """
    path = _require(_cluster_key_json(), "scripts prep (cluster_key.json)")
    with open(path) as fh:
        key = json.load(fh)
    if not isinstance(key, dict) or not key:
        raise ValueError(f"[data] cluster_key.json at {path} is empty or not an object")
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
@_per_study
def load_panel() -> pd.DataFrame:
    """Return the panel table (280 analyzed genes): gene, ensembl_id, annotation."""
    path = _require(_panel_parquet(), "scripts prep (panel.parquet)")
    df = pd.read_parquet(path)
    missing = set(_PANEL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[data] panel.parquet missing columns {missing}")
    return df[_PANEL_COLS]


@_per_study
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
@_per_study
def load_cells() -> pd.DataFrame:
    """Return per-cell coordinates + cluster labels (cell_id, cluster, x, y).

    Reads data/cells/cells.parquet (produced by the R prep step). Fails loudly
    with a named FileNotFoundError if the file is missing.
    """
    return _read_table(_cells_parquet(), _cells_csv(), "scripts/prep_data.R (cells.parquet/.csv)")


def get_cluster_cells(cluster: str) -> pd.DataFrame:
    """Return only the cells belonging to ``cluster``. KeyError if unknown id."""
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(
            f"[data] unknown cluster {cluster!r}; known: {sorted(cfg.KNOWN_CLUSTERS)}"
        )
    cells = load_cells()
    return cells[cells["cluster"] == cluster].reset_index(drop=True)


@_per_study
def load_umap() -> pd.DataFrame:
    """Return UMAP coordinates per cell (cell_id, umap_1, umap_2, cluster).

    Reads data/embeddings/umap.parquet (produced by the R prep step).
    """
    return _read_table(_umap_parquet(), _umap_csv(), "scripts/prep_data.R (umap.parquet/.csv)")


def get_density(gene: str, bin_um: int = 50) -> pd.DataFrame:
    """Return precomputed hex-bin density for one marker at one bin size.

    Reads data/density/{gene}_{bin_um}um.parquet (produced by
    scripts/precompute_density.py). A different ``bin_um`` reads a different
    precomputed frame — this reader NEVER re-bins. FileNotFoundError names the
    missing frame so the caller can fall back to the cell map.
    """
    path = _density_dir() / f"{gene}_{int(bin_um)}um.parquet"
    _require(path, "scripts/precompute_density.py (density hexbins)")
    return pd.read_parquet(path)
