"""Cached read layer between the UI and the pure agent modules.

Thin ``@st.cache_data`` wrappers over ``agent.data`` (markers, panel, cells,
umap, density, marker-expression) and over ``agent.verdict.verdict_for_cluster``
so a verdict is computed exactly once and every Streamlit rerun is a cache hit.
This is what lets viewing controls (pin, bin size, view toggle) rerun the script
without ever recomputing a value — they only re-read an already-cached frame.

``get_agent`` is a ``@st.cache_resource`` singleton around ``agent.loop`` so the
chat agent (and its warm MCP session) survives reruns.

``read_notes`` is deliberately NOT cached: notes mutate on save, so it must
re-read ``context/corrections/`` every call (never a stale drawer).

Streamlit's decorators are applied through a tiny shim so this module imports
with no server running (importing ``ui.data_access`` never needs a live app);
the underlying functions are plain and independently testable.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

import pandas as pd

from agent import config as cfg
from agent import data as agent_data
from agent import verdict as agent_verdict
from agent.config import CLUSTER_ORDER, DEMO_MARKERS
from agent.types import ClusterVerdict, Note

# --------------------------------------------------------------------------- #
# File paths this module reads directly (marker_expr only; everything else goes
# through agent.data). Field layout, per data check:
#   marker_expr.csv : cell_id (int) + one float column per demo marker
# --------------------------------------------------------------------------- #
_MARKER_EXPR_PARQUET = cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.parquet"
_MARKER_EXPR_CSV = cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.csv"
_DENSITY_INDEX = cfg.DATA_DIR_PATH / "density" / "_index.json"


# --------------------------------------------------------------------------- #
# Streamlit-cache shim
# --------------------------------------------------------------------------- #
# We want @st.cache_data / @st.cache_resource semantics in the running app, but
# a plain callable when imported outside Streamlit (tests, `python -c import`).
# These resolve the real decorator lazily; if Streamlit is unavailable they fall
# through to an identity decorator so the wrapped function still works (uncached).
# --------------------------------------------------------------------------- #
def _identity(*_dargs: Any, **_dkw: Any) -> Callable:
    """Return a decorator that leaves the function unchanged (no caching)."""

    def deco(fn: Callable) -> Callable:
        return fn

    return deco


def _cache_data(*dargs: Any, **dkw: Any) -> Callable:
    """``st.cache_data`` if Streamlit is importable, else a no-op decorator."""
    try:
        import streamlit as st

        return st.cache_data(*dargs, **dkw)
    except Exception:  # pragma: no cover - import-time fallback
        return _identity(*dargs, **dkw)


def _cache_resource(*dargs: Any, **dkw: Any) -> Callable:
    """``st.cache_resource`` if Streamlit is importable, else a no-op decorator."""
    try:
        import streamlit as st

        return st.cache_resource(*dargs, **dkw)
    except Exception:  # pragma: no cover - import-time fallback
        return _identity(*dargs, **dkw)


# --------------------------------------------------------------------------- #
# Marker / panel tables (small; cache forever within a session)
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def markers_df() -> pd.DataFrame:
    """The full jazzPanda top-marker table (280 rows). Cached copy; do not mutate."""
    return agent_data.load_markers()


@_cache_data(show_spinner=False)
def panel_df() -> pd.DataFrame:
    """The panel table (280 analyzed genes): gene, ensembl_id, annotation."""
    return agent_data.load_panel()


@_cache_data(show_spinner=False)
def panel_names() -> list[str]:
    """Sorted list of panel gene names (for autocomplete / search)."""
    return sorted(panel_df()["gene"].astype(str).tolist())


def panel_contains(gene: str) -> bool:
    """The panel-absence primitive (pass-through; O(1), already cached upstream)."""
    return agent_data.panel_contains(gene)


def panel_annotation(gene: str) -> Optional[str]:
    """Panel annotation for a gene, or None off-panel (pass-through)."""
    return agent_data.panel_annotation(gene)


# --------------------------------------------------------------------------- #
# Spatial frames (large; the whole point of caching)
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def cells_df() -> pd.DataFrame:
    """All cell coordinates + cluster labels (cell_id, cluster, x, y) — 158k rows."""
    return agent_data.load_cells()


@_cache_data(show_spinner=False)
def cluster_cells_df(cluster: str) -> pd.DataFrame:
    """Cells for one cluster (cell_id, cluster, x, y). Cached per cluster."""
    return agent_data.get_cluster_cells(cluster)


@_cache_data(show_spinner=False)
def umap_df() -> pd.DataFrame:
    """UMAP coordinates per cell (cell_id, umap_1, umap_2, cluster)."""
    return agent_data.load_umap()


@_cache_data(show_spinner=False)
def hexbins(gene: str, bin_um: int = 50) -> pd.DataFrame:
    """Precomputed hex-bin density (hx, hy, count, density) for one marker/bin.

    A different ``bin_um`` reads a DIFFERENT precomputed frame — it never
    re-bins. Cached per (gene, bin_um). Raises the loader's ``FileNotFoundError``
    if the frame was not precomputed, so the caller can fall back to the cell map.
    """
    return agent_data.get_density(gene, bin_um)


@_cache_data(show_spinner=False)
def available_density_markers() -> list[str]:
    """Markers that have precomputed density frames (from the density _index).

    Falls back to the demo marker set (each precomputed by the density step) if
    the index is missing or unparseable.
    """
    try:
        if _DENSITY_INDEX.exists():
            with open(_DENSITY_INDEX) as fh:
                idx = json.load(fh)
            genes = idx.get("genes") or idx.get("markers")
            if isinstance(genes, list):
                return [str(g) for g in genes]
    except Exception:
        pass
    return list(DEMO_MARKERS)


# --------------------------------------------------------------------------- #
# Marker expression (per-cell, demo markers only; for UMAP feature coloring)
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def marker_expr_df() -> pd.DataFrame:
    """Per-cell expression for ALL panel genes (cell_id + one column per gene),
    on a cluster-stratified cell subsample. Read once and cached; parquet-first
    (the committed artifact), CSV fallback for a raw prep output.
    """
    if _MARKER_EXPR_PARQUET.exists():
        return pd.read_parquet(_MARKER_EXPR_PARQUET)
    return pd.read_csv(_MARKER_EXPR_CSV)


@_cache_data(show_spinner=False)
def marker_expr_col(gene: str) -> Optional[pd.DataFrame]:
    """Return (cell_id, value) for one marker, or None if not exported.

    ``value`` is the log-normalized expression column named exactly ``gene``
    (case-insensitive match). Returns None rather than raising when the gene has
    no exported expression, so the feature panel can show its empty state.
    """
    df = marker_expr_df()
    col = None
    if gene in df.columns:
        col = gene
    else:
        upper = {c.upper(): c for c in df.columns}
        col = upper.get(gene.upper())
    if col is None or "cell_id" not in df.columns:
        return None
    return df[["cell_id", col]].rename(columns={col: "value"})


@_cache_data(show_spinner=False)
def available_expr_markers() -> list[str]:
    """Gene names with per-cell expression exported (feature-UMAP / violin) —
    all panel genes now, minus the ``cell_id`` key."""
    return [c for c in marker_expr_df().columns if c != "cell_id"]


@_cache_data(show_spinner=False)
def expr_by_cluster(gene: str) -> Optional[pd.DataFrame]:
    """Return (value, cluster) per cell for one gene, for the across-cluster
    violin. Joins the gene's expression onto the authoritative cluster labels
    (``cells_df`` on ``cell_id``). None if the gene has no exported expression.
    """
    col = marker_expr_col(gene)
    if col is None:
        return None
    cl = cells_df()[["cell_id", "cluster"]]
    return col.merge(cl, on="cell_id", how="inner")


# --------------------------------------------------------------------------- #
# Verdicts — computed ONCE, cached. Viewing controls never touch these.
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def verdict_for(cluster: str) -> ClusterVerdict:
    """Deterministic verdict for one cluster (cached; computed once per session).

    A pinned marker, a bin-size change, or a view toggle must NEVER call this
    with a changed argument — the verdict depends only on the cluster id.
    """
    return agent_verdict.verdict_for_cluster(cluster)


@_cache_data(show_spinner=False)
def all_verdicts() -> list[ClusterVerdict]:
    """Verdicts for all nine clusters (c1..c9), computed once and cached."""
    return [verdict_for(c) for c in CLUSTER_ORDER]


@_cache_data(show_spinner=False)
def verdict_csv() -> str:
    """The 11-column CSV export for all clusters (cached)."""
    return agent_verdict.to_csv(all_verdicts())


# --------------------------------------------------------------------------- #
# Agent (chat) — singleton resource, survives reruns.
# --------------------------------------------------------------------------- #
@_cache_resource(show_spinner=False)
def get_agent() -> Any:
    """Return the persistent chat agent (warm MCP session), cached as a resource.

    Wrapping ``agent.loop``'s module-level default agent keeps one instance (and
    its background MCP loop) alive across Streamlit reruns.
    """
    from agent import loop as agent_loop

    return agent_loop._default_agent()


# --------------------------------------------------------------------------- #
# Notes — NOT cached (mutate on save). Always a fresh read.
# --------------------------------------------------------------------------- #
def read_notes() -> list[Note]:
    """Read all lab notes fresh from disk (never cached — notes mutate on save)."""
    from agent import memory

    return memory.read_notes()


def notes_in_scope(cluster: Optional[str]) -> list[Note]:
    """Notes whose scope fires for the given cluster (fresh read; the scope gate)."""
    from agent import memory

    return memory.apply_notes(cluster)


__all__ = [
    "markers_df",
    "panel_df",
    "panel_names",
    "panel_contains",
    "panel_annotation",
    "cells_df",
    "cluster_cells_df",
    "umap_df",
    "hexbins",
    "available_density_markers",
    "marker_expr_df",
    "marker_expr_col",
    "available_expr_markers",
    "expr_by_cluster",
    "verdict_for",
    "all_verdicts",
    "verdict_csv",
    "get_agent",
    "read_notes",
    "notes_in_scope",
]
