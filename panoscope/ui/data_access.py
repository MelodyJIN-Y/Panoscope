"""Cached read layer between the UI and the pure agent modules.

Thin ``@st.cache_data`` wrappers over ``agent.data`` (markers, panel, cells,
umap, density, marker-expression) and over the per-unit verdict so it is computed
exactly once and every Streamlit rerun is a cache hit.
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

from panoscope.agent import config as cfg
import json
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from panoscope.agent import data as agent_data
from panoscope.agent.types import ClusterVerdict, Note

# --------------------------------------------------------------------------- #
# File paths this module reads directly (marker_expr + density index; everything
# else goes through agent.data). Both resolve to the per-dataset pipeline tree
# (data/studies/<id>/viz/) when present, falling back to the legacy flat path.
#   expr.parquet : cell_id (int) + one float column per panel marker
# --------------------------------------------------------------------------- #
# Resolution (tree-first, demo-only legacy fallback) is agent.data.resolved -- this
# module is a thin wrapper over agent.data and must not own a second copy of it.
def _marker_expr_parquet() -> Path:
    return agent_data.resolved("viz/expr.parquet", "embeddings/marker_expr.parquet")


def _marker_expr_csv() -> Path:
    return agent_data.resolved("viz/expr.csv", "embeddings/marker_expr.csv")


def _density_index() -> Path:
    return agent_data.resolved("viz/hexbin/_index.json", "density/_index.json")


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
    """``st.cache_data``, KEYED ON THE CURRENT STUDY, else a no-op decorator.

    The readers below are either zero-arg or keyed on a cluster/gene, so a plain
    ``st.cache_data`` keys the demo's 280-gene panel and the second study's
    12-gene panel to the SAME cache entry: whichever study is opened first wins
    for the life of the session, and every later study silently reads its frames
    — a measured value from the wrong study reaching a plot header and the band.
    Not reachable today (there is no in-app study picker), but the kidney study is
    the second study on disk and the picker is what would make it reachable.

    The fix is ``agent.data._per_study``'s, applied at the one place every reader
    in this module routes through rather than as a second mechanism: the real
    function is cached behind a leading study key and the public wrapper supplies
    it, so cross-study staleness is unrepresentable instead of merely tested for.
    The key is the study's tree path, not its id, because two studies under
    different data roots may share an id (``tests`` open exactly that pair).

    ``functools.wraps`` is load-bearing here: Streamlit derives a function's cache
    namespace from ``__module__``/``__qualname__`` plus ``inspect.getsource``, and
    ``getsource`` unwraps. Without it every reader in this module would share one
    namespace — the same collision one level worse.
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
        if _density_index().exists():
            with open(_density_index()) as fh:
                idx = json.load(fh)
            genes = idx.get("genes") or idx.get("markers")
            if isinstance(genes, list):
                return [str(g) for g in genes]
    except Exception:
        pass
    return list(cfg.DEMO_MARKERS)


# --------------------------------------------------------------------------- #
# Marker expression (per-cell, demo markers only; for UMAP feature coloring)
# --------------------------------------------------------------------------- #
@_cache_data(show_spinner=False)
def marker_expr_df() -> pd.DataFrame:
    """Per-cell expression for ALL panel genes (cell_id + one column per gene),
    on a cluster-stratified cell subsample. Read once and cached; parquet-first
    (the committed artifact), CSV fallback for a raw prep output.

    Through ``agent.data``'s reader rather than a local parquet/CSV pair, for the
    sake of the error: the hand-rolled version ended in a bare ``pd.read_csv`` and so
    failed with ``[Errno 2] No such file or directory`` naming whichever CSV it tried
    last -- no producer clause, and for the demo a path under the LEGACY flat root
    rather than a study tree. Every other tidy table in the app names the study's own
    path and the prep step that writes it, and this one -- the file a half-prepped
    study is likeliest to be missing, because ``prep_expr.py`` is Panoscope's own
    extra step -- had the least informative message of the lot. Same resolution and
    same parquet-before-CSV preference; only the failure is now the common one.
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
    """Verdict for one cluster (cached once per session).

    Prefers the verdict PERSISTED by the per-dataset pipeline
    (``data/studies/<id>/interp/clusters/c{n}.json``); falls back to BUILDING it
    live when the pipeline tree is absent. Depends only on the cluster id.

    Both arms are now the same engine. The persisted file is written by
    ``pipeline/stages/verdicts.py`` through ``core/``, and the fallback goes through
    ``pipeline.v1_verdicts``, which is that stage's own projection. It used to call
    ``agent/verdict.py``, so an unbuilt study showed the v1 band in the table and a
    built one showed the v2 band — identical on the nine demo units
    (``tests/test_core_compat.py``) and pinned nowhere else.
    """
    from panoscope.pipeline import store, v1_verdicts

    persisted = store.load_verdict(cluster)
    return persisted if persisted is not None else v1_verdicts.verdict_for(cluster)


@_cache_data(show_spinner=False)
def all_verdicts() -> list[ClusterVerdict]:
    """Verdicts for all nine clusters (c1..c9), computed once and cached."""
    return [verdict_for(c) for c in cfg.CLUSTER_ORDER]


def _excluded_clusters() -> set:
    """Clusters carrying a firing `exclude` note. Applied at COMPOSITION only — the
    deterministic jazzPanda verdict on disk is never mutated (docs/note-capture-design.md).

    Reads through ``memory.notes_in_study`` rather than ``read_notes``: the note store
    is ONE directory for every study, and cluster ids are c1..cN in all of them, so an
    ungated read attributes one study's judgement to another study's cluster."""
    from panoscope.agent import memory

    try:
        return {
            n.scope_ref.cluster
            for n in memory.notes_in_study()
            if getattr(n, "type", "") == "exclude" and n.scope == "cluster" and n.scope_ref.cluster
        }
    except Exception:  # noqa: BLE001 - a malformed note must never break the export
        return set()


def _override_notes() -> dict:
    """The most-recent firing ``celltype_override`` note per cluster (one that carries a
    new call). Applied at COMPOSITION only — the deterministic verdict is never mutated.

    Study-gated for the reason in ``_excluded_clusters``, and this is the half that
    reached the Summary board, the exports and the sign-off: an override is the
    biologist's own call, so applying it to a study they never made it in attributes
    their judgement somewhere they never gave it, with click-back intact."""
    from panoscope.agent import memory

    out: dict = {}
    try:
        for n in memory.notes_in_study():  # sorted by created_at -> last write wins
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
    from panoscope.pipeline import store

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


def _override_call(cluster: str) -> Optional[str]:
    """The biologist's confirmed NEW cell-type call for a cluster (from a firing
    ``celltype_override`` note), or None. Display overlay only; never mutates the verdict."""
    ov = _override_notes().get(cluster)
    if ov is None:
        return None
    return (getattr(ov, "subject_cell_type", "") or "").strip() or None


def display_cell_type(cluster: str) -> str:
    """The cell type to SHOW for a cluster: the confirmed override if there is one, else
    the computed call. Reflects an override across the marker rail + Pathways tabs."""
    return _override_call(cluster) or verdict_for(cluster).cell_type


def display_verdict_for(cluster: str) -> ClusterVerdict:
    """``verdict_for`` with the confirmed override applied (cell type + lineage/category)
    for the marker header. Not cached (notes are runtime); the numbers are never mutated."""
    v = verdict_for(cluster)
    ov = _override_notes().get(cluster)
    call = (getattr(ov, "subject_cell_type", "") or "").strip() if ov else ""
    if not call:
        return v
    import dataclasses

    changes = {"cell_type": call, "cell_type_short": call}
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
    """The 11-column CSV export, with any `exclude` notes applied at composition.

    Rendered through ``core/export``'s ASSIGNMENT profile — the registry that pairs
    each column name with the getter that fills it — rather than ``agent/verdict``'s
    hand-written pair of tuples. Byte-identical: ``tests/test_core_export.py`` proves
    it by string equality against the function this replaces, including the
    un-tidied ``confidence_score`` (0.7, not 0.70).
    """
    from panoscope.core.export import to_csv
    from panoscope.types.outputs import TaskKind

    return to_csv(TaskKind.ASSIGNMENT, composed_verdicts()).value


@_cache_data(show_spinner=False)
def celltype_notes() -> dict:
    """Per-cluster cell-type summary notes from the pipeline tree (cached).

    Shape ``{cluster: {cell_type, summary, pmid, citation, verify}}``; ``{}`` when
    the pipeline notes stage has not run. Each summary is a short, grounded,
    live-cited description of the cell type (never fabricated)."""
    from panoscope.pipeline import store

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
    from panoscope.agent import holistic as agent_holistic

    from panoscope.pipeline import store

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
    from panoscope.agent import enrichment as agent_enrichment

    from panoscope.pipeline import store

    persisted = store.load_enrichment(cluster)
    return persisted if persisted is not None else agent_enrichment.enrichment_for_cluster(cluster)


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
    """Return the persistent chat agent (warm MCP session), cached as a resource.

    Wrapping ``agent.loop``'s module-level default agent keeps one instance (and
    its background MCP loop) alive across Streamlit reruns.
    """
    from panoscope.agent import loop as agent_loop

    return agent_loop._default_agent()


# --------------------------------------------------------------------------- #
# Notes — NOT cached (mutate on save). Always a fresh read.
# --------------------------------------------------------------------------- #
def read_notes() -> list[Note]:
    """This study's notes, fresh from disk (never cached — notes mutate on save).

    "All notes" is not what any caller wants: the note store is a single directory
    shared by every study, so the lab-note board and the exported report both asked
    for the current study's record and got the whole lab's. Study-gated here rather
    than in each caller, because the gate has to hold for the ones nobody remembers to
    change. Lab-scoped notes are kept — holding across studies is what that rung is.

    Notes belonging to OTHER studies are not withheld knowledge: they are shown in
    their own study. The ones that genuinely disappear are the unscoped ones, and
    :func:`withheld_notes` is where those are said out loud."""
    from panoscope.agent import memory

    return memory.notes_in_study()


def withheld_notes() -> list[Note]:
    """Stored notes this study must NOT apply because they name no study at all.

    The fail-closed gate is right — a note with no ``scope_ref.dataset`` cannot be
    assumed to belong to whichever study happens to be open — but a biologist's
    committed judgement vanishing with no sentence is its own failure. These are the
    notes to say that sentence about: each is a real :class:`~agent.types.Note`, so it
    carries its id and ``memory.cite_note`` resolves it, and the sentence that names it
    clicks back to the note like any other.

    Empty in the normal case: every note written since ``scope_ref`` existed names its
    dataset. Non-empty means a hand-edited or pre-schema file needs its study added."""
    from panoscope.agent import memory

    return [n for n in memory.read_notes() if not n.scope_ref.dataset and n.scope != "lab"]


def notes_in_scope(cluster: Optional[str]) -> list[Note]:
    """Notes whose scope fires for the given cluster (fresh read; the scope gate)."""
    from panoscope.agent import memory

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
    from panoscope.agent import memory
    from panoscope.agent import tools

    return memory.save_draft(draft, trigger=trigger, base_dir=tools.memory_base_dir())


# --------------------------------------------------------------------------- #
# Sign-off state — the biologist's review checkmarks on the Summary board. NOT
# cached (mutates as the biologist works). A record of which calls were reviewed
# and accepted; never a computed value. A contested sign-off also carries the id
# of the validation note it wrote, so the board can link to the biologist's basis.
# --------------------------------------------------------------------------- #
def signed_off() -> dict:
    """The full sign-off map ``{cluster: {at, note_id}}`` (fresh read)."""
    from panoscope.pipeline import store

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
    from panoscope.pipeline import store

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
    from panoscope.pipeline import store

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
    "withheld_notes",
    "notes_in_scope",
    "save_note_draft",
    "composed_verdicts",
    "display_cell_type",
    "display_verdict_for",
    "override_info",
    "anchored_notes",
    "signed_off",
    "mark_signed_off",
    "clear_signoff",
]
