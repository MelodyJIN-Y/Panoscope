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
# File paths this module reads directly (marker_expr + density index; everything
# else goes through agent.data). Both resolve to the per-dataset pipeline tree
# (data/datasets/<id>/viz/) when present, falling back to the legacy flat path.
#   expr.parquet : cell_id (int) + one float column per panel marker
# --------------------------------------------------------------------------- #
_DATASET_DIR = cfg.DATA_DIR_PATH / "datasets" / cfg.DATASET_ID


def _resolved(tree_rel: str, legacy):
    """Return the per-dataset tree path if present, else the legacy flat path."""
    cand = _DATASET_DIR / tree_rel
    return cand if cand.exists() else legacy


_MARKER_EXPR_PARQUET = _resolved(
    "viz/expr.parquet", cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.parquet"
)
_MARKER_EXPR_CSV = cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.csv"
_DENSITY_INDEX = _resolved(
    "viz/hexbin/_index.json", cfg.DATA_DIR_PATH / "density" / "_index.json"
)


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
# Gene-SET aggregates (enrichment leading edge) — the spatial view of a program.
# A gene set's tissue footprint is the SUM of its leading-edge genes' precomputed,
# area-normalized transcript densities (a view of measured values, never a new
# statistic); its per-cell activity is the MEAN of their exported expression.
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def leading_edge_density(genes: tuple[str, ...], bin_um: int = 50) -> Optional[pd.DataFrame]:
    """Aggregated hex-bin density (hx, hy, count, density) for a gene set.

    Sums the precomputed per-gene density frames bin-by-bin (each gene's density is
    tx/µm², already area-normalized, so the sum is the program's area-normalized
    footprint). Genes with no precomputed frame are honestly skipped. None if none
    of the set's genes have a density frame.
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
# Grounded per-gene biology notes: the skill's Output-4 notes produced by the
# pipeline (``pipeline/stages/notes.py`` -> ``interp/gene_notes.json``), each with
# a REAL live PubMed citation. The evidence table reads these; it never generates
# biology text (confident floor). Falls back to the legacy flat file below.
# --------------------------------------------------------------------------- #
_GENE_NOTES_JSON = cfg.DATA_DIR_PATH / "gene_notes" / "notes.json"


@_cache_data(show_spinner=False)
def gene_notes() -> dict:
    """All grounded gene notes as ``{cluster: {gene: note}}`` (``{}`` if none).

    Prefers the pipeline's SKILL-grounded notes (``interp/gene_notes.json``: the
    per-gene evaluation + the Output-4 biology note); falls back to the legacy flat
    ``data/gene_notes/notes.json`` during migration. Read-only cache; the column
    shows nothing for a gene without a note.
    """
    import json

    from pipeline import store

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
    """Verdict for one cluster (cached once per session).

    Prefers the verdict PERSISTED by the per-dataset pipeline
    (``data/datasets/<id>/interp/clusters/c{n}.json``); falls back to computing it
    live when the pipeline tree is absent. The persisted object is byte-faithful
    to the computed one (``tests/test_pipeline.py`` round-trip gate), so the call
    is identical either way — reading it just avoids recomputation and makes the
    verdict a portable dataset artifact. Depends only on the cluster id.
    """
    from pipeline import store

    persisted = store.load_verdict(cluster)
    return persisted if persisted is not None else agent_verdict.verdict_for_cluster(cluster)


@_cache_data(show_spinner=False)
def all_verdicts() -> list[ClusterVerdict]:
    """Verdicts for all nine clusters (c1..c9), computed once and cached."""
    return [verdict_for(c) for c in CLUSTER_ORDER]


def _excluded_clusters() -> set:
    """Clusters carrying a firing `exclude` note. Applied at COMPOSITION only — the
    deterministic jazzPanda verdict on disk is never mutated (docs/note-capture-design.md)."""
    from agent import memory

    try:
        return {
            n.scope_ref.cluster
            for n in memory.read_notes()
            if getattr(n, "type", "") == "exclude" and n.scope == "cluster" and n.scope_ref.cluster
        }
    except Exception:  # noqa: BLE001 - a malformed note must never break the export
        return set()


def _override_notes() -> dict:
    """The most-recent firing ``celltype_override`` note per cluster (one that carries a
    new call). Applied at COMPOSITION only — the deterministic verdict is never mutated."""
    from agent import memory

    out: dict = {}
    try:
        for n in memory.read_notes():  # sorted by created_at -> last write wins
            if (
                getattr(n, "type", "") == "celltype_override"
                and n.scope == "cluster"
                and n.scope_ref.cluster
                and (getattr(n, "subject_cell_type", "") or "").strip()
            ):
                out[n.scope_ref.cluster] = n
    except Exception:  # noqa: BLE001 - a malformed note never breaks the composition
        return {}
    return out


def _signed_off_clusters() -> set:
    """Clusters the biologist has reviewed and accepted on the Summary board.

    Read fresh from the runtime review-state file (never cached — sign-offs mutate
    as the biologist works). A signed-off call is treated as adjudicated: its
    ``verify`` flag is cleared at COMPOSITION only (the deterministic verdict on
    disk is never mutated)."""
    from pipeline import store

    try:
        return set(store.load_review_state().keys())
    except Exception:  # noqa: BLE001 - a missing/bad review file never breaks composition
        return set()


def composed_verdicts() -> list[ClusterVerdict]:
    """``all_verdicts()`` with your confirmed overlays applied — ``exclude`` notes,
    ``celltype_override`` notes (new cell type + lineage/category; verify flagged
    only when the literature dissents), and sign-offs (a reviewed call's ``verify``
    flag clears). Returns NEW objects; the cached deterministic verdicts are never
    mutated, so the computed output stays intact and only the composed view/export
    changes. NOT cached: notes and sign-offs are added at runtime."""
    excluded = _excluded_clusters()
    overrides = _override_notes()
    signed_off = _signed_off_clusters()
    if not excluded and not overrides and not signed_off:
        return all_verdicts()
    import dataclasses

    out: list[ClusterVerdict] = []
    for v in all_verdicts():
        changes: dict = {}
        if v.cluster in excluded:
            changes["exclude"] = True
        ov = overrides.get(v.cluster)
        if ov is not None:
            new_call = ov.subject_cell_type.strip()
            changes["cell_type"] = new_call
            changes["cell_type_short"] = new_call
            if ov.subject_lineage.strip():
                changes["lineage"] = ov.subject_lineage.strip()
            if ov.subject_category.strip():
                changes["category"] = ov.subject_category.strip()
            if ov.tension.dissent:  # flag for re-check ONLY when the literature dissents
                changes["verify"] = True
        # A signed-off call is adjudicated: clear its re-check flag (a contested
        # sign-off wrote a validation note recording the biologist's acceptance).
        # Applied last so it wins over an override's dissent flag.
        if v.cluster in signed_off:
            changes["verify"] = False
        out.append(dataclasses.replace(v, **changes) if changes else v)
    return out


def anchored_notes(cluster: str) -> dict:
    """In-scope notes indexed by their anchor, for rendering next to the driver row:
    ``{"gene": {GENE: [note,...]}, "gene_set": {HALLMARK_X: [note,...]}}``. Marker and
    marker_convention notes index by gene; program_reinterpretation notes by gene set.
    Routes through ``apply_notes`` (the fail-closed scope gate), so a c2 note never
    shows on c3."""
    from agent import memory

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
    ov = _override_notes().get(cluster)
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
    """The 11-column CSV export, with any `exclude` notes applied at composition."""
    return agent_verdict.to_csv(composed_verdicts())


@_cache_data(show_spinner=False)
def celltype_notes() -> dict:
    """Per-cluster cell-type summary notes from the pipeline tree (cached).

    Shape ``{cluster: {cell_type, summary, pmid, citation, verify}}``; ``{}`` when
    the pipeline notes stage has not run. Each summary is a short, grounded,
    live-cited description of the cell type (never fabricated)."""
    from pipeline import store

    return store.load_celltype_notes()


def celltype_summary(cluster: str) -> str:
    """The short grounded cell-type summary for a cluster, or '' if not available."""
    note = celltype_notes().get(cluster) or {}
    return str(note.get("summary") or "")


@_cache_data(show_spinner=False)
def holistic():
    """The cross-cluster holistic review (Step 4), cached once per session.

    Prefers the review PERSISTED by the pipeline (``interp/holistic.json``); falls
    back to computing it live when the tree is absent. The review is deterministic,
    so the persisted object is byte-faithful to the computed one
    (``tests/test_pipeline.py`` round-trip gate) — reading it just avoids
    recomputation and makes the review a portable dataset artifact. The one live
    piece (the refinement's citation) is still fetched at render time in
    ``ui.holistic``; this returns only the grounded coherence + refinement data.
    """
    from agent import holistic as agent_holistic

    from pipeline import store

    persisted = store.load_holistic()
    return persisted if persisted is not None else agent_holistic.holistic_review()


# --------------------------------------------------------------------------- #
# Gene-set enrichment (second workflow) — tree-first readers, mirroring the
# marker verdict readers. A dataset with no enrichment result simply has no
# Pathways slice (enrichment_available() is False).
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def enrichment_for(cluster: str):
    """Enrichment verdict for one cluster (persisted tree first, else live)."""
    from agent import enrichment as agent_enrichment

    from pipeline import store

    persisted = store.load_enrichment(cluster)
    return persisted if persisted is not None else agent_enrichment.enrichment_for_cluster(cluster)


@_cache_data(show_spinner=False)
def all_enrichments() -> list:
    """Enrichment verdicts for all nine clusters (cached)."""
    return [enrichment_for(c) for c in CLUSTER_ORDER]


@_cache_data(show_spinner=False)
def pathway_themes():
    """Cross-cluster pathway themes (persisted tree first, else live)."""
    from agent import enrichment_themes

    from pipeline import store

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
    from pipeline import store

    return store.load_pathway_notes()


def pathway_note(cluster: str, gene_set: str) -> Optional[dict]:
    """The precomputed grounded note for ``(cluster, gene_set)``, or None if absent."""
    return pathway_notes().get(cluster, {}).get(gene_set)


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
    """Read all notes fresh from disk (never cached — notes mutate on save)."""
    from agent import memory

    return memory.read_notes()


def notes_in_scope(cluster: Optional[str]) -> list[Note]:
    """Notes whose scope fires for the given cluster (fresh read; the scope gate)."""
    from agent import memory

    return memory.apply_notes(cluster)


def save_note_draft(draft: Any, trigger: str = "override") -> Note:
    """Persist a biologist-confirmed :class:`~agent.types.NoteDraft` as a note.

    The second half of capture-at-override: the confirm card produced a (possibly
    edited) draft, and this writes it via ``agent.memory.save_draft`` into the SAME
    base dir the agent reads notes from — so a just-saved note is immediately in scope
    for recall. No second literature lookup (the tension is already on the draft).
    ``trigger`` records where the note was born (``override`` from a chat, or
    ``holistic_review`` from a cross-cluster refinement). Returns the written Note.
    """
    from agent import memory
    from agent import tools

    return memory.save_draft(draft, trigger=trigger, base_dir=tools.memory_base_dir())


# --------------------------------------------------------------------------- #
# Sign-off state — the biologist's review checkmarks on the Summary board. NOT
# cached (mutates as the biologist works). A record of which calls were reviewed
# and accepted; never a computed value. A contested sign-off also carries the id
# of the validation note it wrote, so the board can link to the biologist's basis.
# --------------------------------------------------------------------------- #
def signed_off() -> dict:
    """The full sign-off map ``{cluster: {at, note_id}}`` (fresh read)."""
    from pipeline import store

    try:
        return store.load_review_state()
    except Exception:  # noqa: BLE001 - a missing/bad review file reads as none signed off
        return {}


def mark_signed_off(cluster: str, note_id: Optional[str] = None, at: str = "") -> None:
    """Record that the biologist reviewed and accepted ``cluster``'s call.

    ``note_id`` links the validation note a contested sign-off wrote (``None`` for a
    clean checkmark). ``at`` is an ISO timestamp passed in by the caller (this stays
    free of clock calls). Idempotent: re-signing a cluster just refreshes its record.
    """
    from pipeline import store

    try:
        reviewed = dict(store.load_review_state())
        reviewed[cluster] = {"at": at, "note_id": note_id}
        store.save_review_state(reviewed, saved_at=at)
    except Exception:  # noqa: BLE001 - a failed write must never crash the board
        pass


def clear_signoff(cluster: str, at: str = "") -> None:
    """Undo a sign-off: drop ``cluster`` from the review state (the call is unreviewed
    again). The validation note a contested sign-off wrote is left in My notes — undo
    reopens the review, it does not erase the biologist's recorded basis."""
    from pipeline import store

    try:
        reviewed = dict(store.load_review_state())
        if cluster in reviewed:
            reviewed.pop(cluster, None)
            store.save_review_state(reviewed, saved_at=at)
    except Exception:  # noqa: BLE001 - a failed write must never crash the board
        pass


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
    "gene_notes",
    "gene_note",
    "verdict_for",
    "all_verdicts",
    "verdict_csv",
    "celltype_notes",
    "celltype_summary",
    "holistic",
    "enrichment_for",
    "all_enrichments",
    "pathway_themes",
    "enrichment_available",
    "pathway_notes",
    "pathway_note",
    "get_agent",
    "read_notes",
    "notes_in_scope",
    "save_note_draft",
    "composed_verdicts",
    "override_info",
    "anchored_notes",
    "signed_off",
    "mark_signed_off",
    "clear_signoff",
]
