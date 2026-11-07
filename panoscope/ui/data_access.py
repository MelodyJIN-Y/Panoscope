"""Cached read layer between the UI and the pure agent modules.

Viewing controls rerun the script and re-read cached frames; they never
recompute a value. Notes and sign-offs are never cached (they mutate at
runtime). The Streamlit cache decorators go through a shim so the module
imports, uncached, without a running app.
"""

from __future__ import annotations

from panoscope.agent import config as cfg
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from panoscope.agent import compose
from panoscope.agent import data as agent_data
from panoscope.agent.types import ClusterVerdict, Note
from panoscope.data import spatial

# Paths read directly; resolution (tree-first, legacy fallback) is agent.data.resolved.
def _marker_expr_parquet() -> Path:
    return agent_data.resolved("viz/expr.parquet", "embeddings/marker_expr.parquet")


def _marker_expr_csv() -> Path:
    return agent_data.resolved("viz/expr.csv", "embeddings/marker_expr.csv")


def _density_index() -> Path:
    return agent_data.resolved("viz/hexbin/_index.json", "density/_index.json")


# --------------------------------------------------------------------------- #
# Streamlit-cache shim: the real decorator in the app, identity outside it.
# --------------------------------------------------------------------------- #
def _identity(*_dargs: Any, **_dkw: Any) -> Callable:
    """Return a decorator that leaves the function unchanged (no caching)."""

    def deco(fn: Callable) -> Callable:
        return fn

    return deco


def _cache_data(*dargs: Any, **dkw: Any) -> Callable:
    """``st.cache_data`` keyed on the current study's tree path, else a no-op.

    The study key makes a cross-study cache hit unrepresentable (a path, not an
    id: two data roots may share an id). ``functools.wraps`` is load-bearing:
    Streamlit derives each function's cache namespace from it.
    """
    try:
        import streamlit as st
    except Exception:  # pragma: no cover - import-time fallback
        return _identity(*dargs, **dkw)

    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def keyed(_study: str, *args: Any, **kw: Any):
            return fn(*args, **kw)

        cached = st.cache_data(*dargs, **dkw)(keyed)

        @wraps(fn)
        def wrapper(*args: Any, **kw: Any):
            return cached(str(cfg.current_study().dir), *args, **kw)

        wrapper.clear = cached.clear  # keep st.cache_data's public API
        return wrapper

    return deco


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


def tissue_platform() -> tuple[str, str]:
    """(tissue, platform) off the active study's manifest (grounded, derived from the
    data); ``("", "")`` when it is unreadable."""
    try:
        m = cfg.current_study().manifest
        return str(m.get("tissue", "")).strip(), str(m.get("platform", "")).strip()
    except Exception:  # noqa: BLE001 - no manifest just means a plainer label
        return "", ""


@_cache_data(show_spinner=False)
def panel_names() -> list[str]:
    """Sorted list of panel gene names (for autocomplete / search)."""
    return sorted(agent_data.load_panel()["gene"].astype(str).tolist())


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
def hexbins(gene: str, bin_um: float = 50) -> pd.DataFrame:
    """Hex-bin density (hx, hy, count, density) for one marker at one bin size.

    SpatialData store first (any ``bin_um``), else the precomputed frames. Raises
    ``FileNotFoundError`` when neither has it, so the caller falls back to the cell map.
    """
    store = cfg.current_study().spatial
    if store.is_dir():
        try:
            return spatial.density(gene, float(bin_um), store)
        except spatial.StoreUnavailable:
            pass  # a directory that is not a readable store -- use the frames
    return agent_data.get_density(gene, int(bin_um))


# --------------------------------------------------------------------------- #
# Marker expression (per-cell, demo markers only; for UMAP feature coloring)
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def marker_expr_df() -> pd.DataFrame:
    """Per-cell expression for ALL panel genes (cell_id + one column per gene) on a
    cluster-stratified subsample. Parquet first, CSV fallback; a missing file raises
    ``agent.data``'s error naming the study path and the prep step that writes it.
    """
    return agent_data._read_table(
        _marker_expr_parquet(),
        _marker_expr_csv(),
        "scripts/prep_expr.py (expr.parquet/.csv)",
    )


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
# Gene-set aggregates (leading edge): a view of measured values, never a new statistic.
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def leading_edge_density(genes: tuple[str, ...], bin_um: int = 50) -> Optional[pd.DataFrame]:
    """Bin-by-bin sum of the set's per-gene density frames (hx, hy, count, density).

    Genes with no frame are skipped; None if none of them has one.
    """
    frames = []
    for g in genes:
        try:
            hb = hexbins(g, bin_um)
        except Exception:  # noqa: BLE001 - a missing frame is a skip, never a fake
            continue
        if hb is not None and not hb.empty:
            frames.append(hb[["hx", "hy", "count", "density"]])
    if not frames:
        return None
    allf = pd.concat(frames, ignore_index=True)
    return allf.groupby(["hx", "hy"], as_index=False).agg(
        count=("count", "sum"), density=("density", "sum")
    )


@_cache_data(show_spinner=False)
def leading_edge_expr(genes: tuple[str, ...]) -> Optional[pd.DataFrame]:
    """Per-cell (cell_id, value) for a gene set — the MEAN of its genes' exported
    expression (skipping genes with no exported column). None if none are exported."""
    cols = []
    for g in genes:
        c = marker_expr_col(g)
        if c is not None:
            cols.append(c.rename(columns={"value": g}))
    if not cols:
        return None
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c, on="cell_id", how="outer")
    gene_cols = [c for c in merged.columns if c != "cell_id"]
    out = merged[["cell_id"]].copy()
    out["value"] = merged[gene_cols].mean(axis=1)
    return out


# --------------------------------------------------------------------------- #
# Grounded per-gene biology notes, each live-cited. Read only; never generated here.
# --------------------------------------------------------------------------- #
_GENE_NOTES_JSON = cfg.DATA_DIR_PATH / "gene_notes" / "notes.json"


@_cache_data(show_spinner=False)
def gene_notes() -> dict:
    """All grounded gene notes as ``{cluster: {gene: note}}`` (``{}`` if none).

    Pipeline tree first, legacy ``data/gene_notes/notes.json`` fallback.
    """
    import json

    from panoscope.pipeline import store

    tree = store.load_gene_notes()
    if tree:
        return tree
    try:
        return json.loads(_GENE_NOTES_JSON.read_text())
    except (OSError, ValueError):
        return {}


def gene_note(cluster: str, gene: str) -> Optional[dict]:
    """The precomputed grounded note for ``(cluster, gene)``, or None if absent."""
    return gene_notes().get(cluster, {}).get(gene)


# --------------------------------------------------------------------------- #
# Verdicts — computed ONCE, cached. Viewing controls never touch these.
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def verdict_for(cluster: str) -> ClusterVerdict:
    """Verdict for one cluster: the persisted pipeline verdict, else built live
    through ``pipeline.v1_verdicts`` (the same engine)."""
    from panoscope.pipeline import store, v1_verdicts

    persisted = store.load_verdict(cluster)
    return persisted if persisted is not None else v1_verdicts.verdict_for(cluster)


@_cache_data(show_spinner=False)
def all_verdicts() -> list[ClusterVerdict]:
    """Verdicts for all nine clusters (c1..c9), computed once and cached."""
    return [verdict_for(c) for c in cfg.CLUSTER_ORDER]


def _excluded_clusters() -> set:
    """Clusters carrying a firing `exclude` note. Applied at composition only; the
    verdict on disk is never mutated. Study-gated: cluster ids repeat across studies."""
    from panoscope.agent import memory

    try:
        return {
            n.scope_ref.cluster
            for n in memory.notes_in_study()
            if getattr(n, "type", "") == "exclude" and n.scope == "cluster" and n.scope_ref.cluster
        }
    except Exception:  # noqa: BLE001 - a malformed note must never break the export
        return set()


def _signed_off_clusters() -> set:
    """Clusters signed off on the Summary board (fresh read, never cached). Their
    ``verify`` clears at composition only; the verdict on disk is never mutated."""
    from panoscope.pipeline import store

    try:
        return set(store.load_review_state().keys())
    except Exception:  # noqa: BLE001 - a missing/bad review file never breaks composition
        return set()


def composed_verdicts() -> list[ClusterVerdict]:
    """``all_verdicts()`` with exclude notes, cell-type overrides and sign-offs
    applied. Returns new objects (the cached verdicts are never mutated); not cached."""
    excluded = _excluded_clusters()
    overrides = compose.override_notes()
    signed_off = _signed_off_clusters()
    if not excluded and not overrides and not signed_off:
        return all_verdicts()
    import dataclasses

    out: list[ClusterVerdict] = []
    for v in all_verdicts():
        ov = overrides.get(v.cluster)
        if ov is not None:
            v = compose.compose_verdict(v, ov)
        changes: dict = {}
        if v.cluster in excluded:
            changes["exclude"] = True
        # Sign-off clears verify; applied last so it wins over an override's dissent flag.
        if v.cluster in signed_off:
            changes["verify"] = False
        out.append(dataclasses.replace(v, **changes) if changes else v)
    return out


def display_cell_type(cluster: str) -> str:
    """The cell type to SHOW for a cluster: the confirmed override if there is one, else
    the computed call. Reflects an override across the marker rail + Pathways tabs."""
    return compose.override_call(cluster) or verdict_for(cluster).cell_type


def display_verdict_for(cluster: str) -> ClusterVerdict:
    """``verdict_for`` with the confirmed override applied (cell type + lineage/category)
    for the marker header. Not cached (notes are runtime); the numbers are never mutated."""
    import dataclasses

    v = verdict_for(cluster)
    ov = compose.override_notes().get(cluster)
    call = (getattr(ov, "subject_cell_type", "") or "").strip() if ov else ""

    # Sign-off clears verify, as in ``composed_verdicts``.
    signed = cluster in _signed_off_clusters()
    if not call:
        return dataclasses.replace(v, verify=False) if (signed and v.verify) else v

    changes = {"cell_type": call, "cell_type_short": call}
    if signed:
        changes["verify"] = False
    if getattr(ov, "subject_lineage", "").strip():
        changes["lineage"] = ov.subject_lineage.strip()
    if getattr(ov, "subject_category", "").strip():
        changes["category"] = ov.subject_category.strip()
    return dataclasses.replace(v, **changes)


def anchored_notes(cluster: str) -> dict:
    """In-scope notes indexed by their anchor, for rendering next to the driver row:
    ``{"gene": {GENE: [note,...]}, "gene_set": {HALLMARK_X: [note,...]}}``. Marker and
    marker_convention notes index by gene; program_reinterpretation notes by gene set.
    Routes through ``apply_notes`` (the fail-closed scope gate), so a c2 note never
    shows on c3."""
    from panoscope.agent import memory

    genes: dict = {}
    sets: dict = {}
    try:
        for n in memory.apply_notes(cluster):
            t = getattr(n, "type", "")
            if t in ("marker_reinterpretation", "marker_convention"):
                for g in getattr(n, "subject_markers", ()):
                    genes.setdefault(g, []).append(n)
            elif t == "program_reinterpretation":
                for gs in getattr(n, "subject_gene_sets", ()):
                    sets.setdefault(gs, []).append(n)
    except Exception:  # noqa: BLE001 - a bad note never breaks a table
        pass
    return {"gene": genes, "gene_set": sets}


def override_info(cluster: str) -> "Optional[dict]":
    """For a cluster with a confirmed cell-type override: the new call, the computed
    call it replaces, the note id, and the literature agree/dissent counts — so the UI
    shows the override with the tension visible. None if there is no override."""
    ov = compose.override_notes().get(cluster)
    if ov is None:
        return None
    computed = verdict_for(cluster)
    return {
        "new_call": ov.subject_cell_type.strip(),
        "computed_call": computed.cell_type,
        "note_id": ov.id,
        "agree": len(ov.tension.agree),
        "dissent": len(ov.tension.dissent),
    }


def verdict_csv() -> str:
    """The 11-column CSV export of the composed verdicts, via ``core/export``'s
    ASSIGNMENT profile."""
    from panoscope.core.export import to_csv
    from panoscope.types.outputs import TaskKind

    return to_csv(TaskKind.ASSIGNMENT, composed_verdicts()).value


@_cache_data(show_spinner=False)
def celltype_notes() -> dict:
    """Per-cluster cell-type notes ``{cluster: {cell_type, summary, pmid, citation,
    verify}}`` (``{}`` if the notes stage has not run). Read one through
    :func:`celltype_note_for`."""
    from panoscope.pipeline import store

    return store.load_celltype_notes()


def celltype_note_for(verdict: Any) -> dict:
    """This verdict's cell-type note, or ``{}`` when the note describes another call.

    Matched against the verdict being rendered; the guard is
    ``report._celltype_note_for``.
    """
    from panoscope.ui import report

    return report._celltype_note_for(celltype_notes(), verdict)


def holistic():
    """The cross-cluster holistic review with confirmed call overrides applied.

    Not cached (overrides arrive at runtime); with no override it returns the
    cached computed review.
    """
    overrides = compose.override_notes()
    if not overrides:
        return _holistic_computed()
    from panoscope.agent import holistic as agent_holistic

    # Calls only: ``exclude`` is not applied, so cell counts and denominators hold.
    metas = {
        c: {"cell_type": (ov.subject_cell_type or "").strip(),
            "category": (ov.subject_category or "").strip()
            or annotation_category(c)}
        for c, ov in overrides.items()
        if (getattr(ov, "subject_cell_type", "") or "").strip()
    }
    return agent_holistic.holistic_review(metas)


def annotation_category(cluster: str) -> str:
    """This study's annotated category for ``cluster`` ("" if it names none), so a
    call-only override keeps its compartment."""
    from panoscope.agent import annotation

    try:
        return str(annotation.meta_for(cluster).get("category") or "")
    except Exception:  # noqa: BLE001 - an unknown cluster simply has no category
        return ""


@_cache_data(show_spinner=False)
def _holistic_computed():
    """The cross-cluster holistic review as computed: persisted first, else live."""
    from panoscope.agent import holistic as agent_holistic

    from panoscope.pipeline import store

    persisted = store.load_holistic()
    return persisted if persisted is not None else agent_holistic.holistic_review()


# --------------------------------------------------------------------------- #
# Gene-set enrichment — tree-first readers. No result means no Pathways slice.
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def enrichment_for(cluster: str):
    """Enrichment verdict for one cluster (persisted tree first, else live)."""
    from panoscope.agent import enrichment as agent_enrichment

    return agent_enrichment.load_or_compute(cluster)


@_cache_data(show_spinner=False)
def all_enrichments() -> list:
    """Enrichment verdicts for all nine clusters (cached)."""
    return [enrichment_for(c) for c in cfg.CLUSTER_ORDER]


@_cache_data(show_spinner=False)
def pathway_themes():
    """Cross-cluster pathway themes (persisted tree first, else live)."""
    from panoscope.agent import enrichment_themes

    from panoscope.pipeline import store

    persisted = store.load_pathway_themes()
    return persisted if persisted is not None else enrichment_themes.pathway_themes()


@_cache_data(show_spinner=False)
def enrichment_available() -> bool:
    """True iff this dataset has an enrichment slice (a result to interpret)."""
    try:
        return bool(all_enrichments())
    except Exception:  # noqa: BLE001 - no result -> no Pathways tab
        return False


@_cache_data(show_spinner=False)
def pathway_notes() -> dict:
    """Live-cited per-pathway biology notes ``{cluster: {gene_set: note}}`` ({} if none)."""
    from panoscope.pipeline import store

    return store.load_pathway_notes()


def pathway_note(cluster: str, gene_set: str) -> Optional[dict]:
    """The precomputed grounded note for ``(cluster, gene_set)``, or None if absent."""
    return pathway_notes().get(cluster, {}).get(gene_set)


# --------------------------------------------------------------------------- #
# Agent (chat) — singleton resource, survives reruns.
# --------------------------------------------------------------------------- #
@_cache_resource(show_spinner=False)
def get_agent() -> Any:
    """The persistent chat agent (warm MCP session), kept alive across reruns."""
    from panoscope.agent import loop as agent_loop

    return agent_loop._default_agent()


# --------------------------------------------------------------------------- #
# Notes — NOT cached (mutate on save). Always a fresh read.
# --------------------------------------------------------------------------- #
def read_notes() -> list[Note]:
    """This study's notes (plus lab-scoped ones), fresh from disk, never cached.

    Study-gated here so every caller gets the gate. Unscoped notes are surfaced by
    :func:`withheld_notes`."""
    from panoscope.agent import memory

    return memory.notes_in_study()


def withheld_notes() -> list[Note]:
    """Stored notes this study must NOT apply because they name no study (fail-closed).

    Returned so the UI can say so; each is a real Note, so it still clicks back."""
    from panoscope.agent import memory

    return [n for n in memory.read_notes() if not n.scope_ref.dataset and n.scope != "lab"]


def save_note_draft(draft: Any, trigger: str = "override") -> Note:
    """Persist a biologist-confirmed :class:`~agent.types.NoteDraft` as a note.

    Written to the same base dir the agent reads, so it is in scope at once.
    ``trigger`` records where the note was born (``override`` or ``holistic_review``).
    """
    from panoscope.agent import memory
    from panoscope.agent import tools

    return memory.save_draft(draft, trigger=trigger, base_dir=tools.memory_base_dir())


# --------------------------------------------------------------------------- #
# Sign-off state: the biologist's review record, never a computed value. Not cached.
# --------------------------------------------------------------------------- #
def signed_off() -> dict:
    """The full sign-off map ``{cluster: {at, note_id}}`` (fresh read)."""
    from panoscope.pipeline import store

    try:
        return store.load_review_state()
    except Exception:  # noqa: BLE001 - a missing/bad review file reads as none signed off
        return {}


def mark_signed_off(cluster: str, note_id: Optional[str] = None, at: str = "") -> None:
    """Record that the biologist reviewed and accepted ``cluster``'s call (idempotent).

    ``note_id`` links a contested sign-off's validation note; ``at`` is the caller's
    ISO timestamp.
    """
    from panoscope.pipeline import store

    try:
        reviewed = dict(store.load_review_state())
        reviewed[cluster] = {"at": at, "note_id": note_id}
        store.save_review_state(reviewed, saved_at=at)
    except Exception:  # noqa: BLE001 - a failed write must never crash the board
        pass


def clear_signoff(cluster: str, at: str = "") -> None:
    """Undo a sign-off. Any validation note it wrote is kept."""
    from panoscope.pipeline import store

    try:
        reviewed = dict(store.load_review_state())
        if cluster in reviewed:
            reviewed.pop(cluster, None)
            store.save_review_state(reviewed, saved_at=at)
    except Exception:  # noqa: BLE001 - a failed write must never crash the board
        pass
