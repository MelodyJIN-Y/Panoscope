"""Session-state schema + typed accessors for the Panoscope UI.

All mutable UI state lives in ``st.session_state`` behind these accessors so no
pane pokes a raw key by string. The one grounding-relevant guarantee here:
**selecting a marker only mutates ``selected_markers``** — it is a viewing
control, it never triggers a verdict recompute. Verdicts are computed once and
cached in ``ui.data_access``; nothing in this module recomputes a value.

Marker selection is a **multi-select**: the biologist toggles genes on/off via a
subtle borderless dot per evidence row, and the spatial grid draws small-multiples
for the selected set. ``selected_markers`` is an *ordered* list (selection order is
preserved), capped for legibility only when rendering small-multiples
(``active_markers(cap=…)``). The legacy single-pin API is kept as a thin alias over
``selected_markers[0]`` so callers that predate multi-select keep working.

Streamlit is imported lazily inside functions (``_ss()``) so this module imports
cleanly with no server running — importing ``ui.state`` never requires a live
session. The keys and their defaults are declared once in ``_DEFAULTS`` and
seeded idempotently by ``init_state()``.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.config import CLUSTER_ORDER

# --------------------------------------------------------------------------- #
# Key names (one place; accessors reference these, never bare strings)
# --------------------------------------------------------------------------- #
K_SELECTED_CLUSTER = "selected_cluster"
K_SELECTED_MARKERS = "selected_markers"
K_SELECTED_PATHWAYS = "selected_pathways"
K_BIN_UM = "bin_um"
K_CHAT_THREAD = "chat_thread"
K_PENDING_DRAFT = "pending_draft"
K_PAPER_OPEN = "paper_open"
K_ACTIVE_PMID = "active_pmid"
K_OPENING_POSTED = "opening_posted"
K_INITIALIZED = "_ui_initialized"

# --------------------------------------------------------------------------- #
# Closed vocabularies for the viewing/interaction controls
# --------------------------------------------------------------------------- #
BIN_SIZES_UM: tuple[int, ...] = (25, 50, 100)  # µm presets from the wireframe
DEFAULT_BIN_UM = 25

DEFAULT_CLUSTER = CLUSTER_ORDER[0]  # "c1"

# Small-multiples cap: how many selected genes get a density + feature-UMAP pair.
# Extras stay in ``selected_markers`` (and in the evidence table) but the spatial
# grid caps the panels so the small-multiples stay legible.
DEFAULT_MARKER_CAP = 4

# --------------------------------------------------------------------------- #
# Default seed values. Callables are invoked per-session so no mutable default
# is shared across reruns/sessions (thread/list must be a fresh object).
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, Any] = {
    K_SELECTED_CLUSTER: DEFAULT_CLUSTER,
    K_SELECTED_MARKERS: dict,   # {cluster: [gene, ...]} — per-cluster multi-select
    K_SELECTED_PATHWAYS: dict,  # {cluster: [gene_set, ...]} — per-cluster multi-select
    K_BIN_UM: DEFAULT_BIN_UM,
    K_CHAT_THREAD: dict,        # {cluster_id: [msg, ...]} — one thread per cluster
    K_PENDING_DRAFT: dict,      # {cluster_id: NoteDraft} — note awaiting confirm
    K_PAPER_OPEN: False,
    K_ACTIVE_PMID: None,
    K_OPENING_POSTED: dict,     # {cluster_id: True} — opening interp posted once
}


def _ss() -> Any:
    """Return ``st.session_state`` (import Streamlit lazily; import-safe module)."""
    import streamlit as st

    return st.session_state


def init_state() -> None:
    """Seed every session-state key to its default exactly once per session.

    Idempotent: existing keys are left untouched so a rerun never clobbers what
    the biologist changed. Safe to call at the top of every script run.
    """
    ss = _ss()
    if ss.get(K_INITIALIZED):
        return
    for key, default in _DEFAULTS.items():
        if key not in ss:
            ss[key] = default() if callable(default) else default
    _hydrate_threads(ss)  # restore chat transcripts from disk so a refresh keeps them
    ss[K_INITIALIZED] = True


def _hydrate_threads(ss: dict) -> None:
    """Restore persisted chat threads on a fresh session (a browser refresh). A
    non-empty thread marks its opening as already posted so the opening interpretation
    is not re-posted on top of the restored transcript."""
    try:
        from ui import chat_store

        threads = chat_store.load_all()
    except Exception:  # noqa: BLE001 - a bad transcript never blocks startup
        return
    if not threads:
        return
    ss[K_CHAT_THREAD] = threads
    posted = dict(ss.get(K_OPENING_POSTED, {}))
    for key, msgs in threads.items():
        if msgs:
            posted[key] = True
    ss[K_OPENING_POSTED] = posted


def _persist_threads() -> None:
    """Best-effort write of all chat threads to disk (so a refresh restores them)."""
    try:
        from ui import chat_store

        chat_store.save_all(_threads())
    except Exception:  # noqa: BLE001 - losing the transcript must not break the app
        pass


# --------------------------------------------------------------------------- #
# Selected cluster
# --------------------------------------------------------------------------- #
def get_selected_cluster() -> str:
    """Return the active cluster id (defaults to c1 before init)."""
    return _ss().get(K_SELECTED_CLUSTER, DEFAULT_CLUSTER)


def set_selected_cluster(cluster: str) -> None:
    """Select a cluster. Ignores an unknown id (fail-closed on bad input).

    Switching clusters keeps the selected markers and bin size — those persist
    across clusters per the viewing-control spec.
    """
    if cluster not in CLUSTER_ORDER:
        return
    _ss()[K_SELECTED_CLUSTER] = cluster


# The Marker-genes page hosts the per-cluster conversation. Mirrors app.py's page
# key/value so the Summary board can send the biologist there without importing
# app.py (which imports this module).
_K_ACTIVE_PAGE = "active_page"
_PAGE_WITH_CHAT = "markers"   # the Marker-genes page value (mirrors app.py)


def open_cluster_chat(cluster: str) -> None:
    """Select ``cluster`` and switch to the Marker-genes page (its conversation pane).

    The Summary board's "open chat" action: it lands the biologist in the cluster's
    live thread, where the full override/question flow lives. Ignores an unknown id
    (fail-closed), same as ``set_selected_cluster``."""
    if cluster not in CLUSTER_ORDER:
        return
    ss = _ss()
    ss[K_SELECTED_CLUSTER] = cluster
    ss[_K_ACTIVE_PAGE] = _PAGE_WITH_CHAT
    try:  # keep the URL in sync so the tab persists across a refresh (best-effort)
        import streamlit as st

        st.query_params["page"] = _PAGE_WITH_CHAT
        st.query_params["cluster"] = cluster
    except Exception:  # noqa: BLE001 - query-param sync is non-essential
        pass


# --------------------------------------------------------------------------- #
# Selected pathways — the enrichment analog of selected markers: a per-cluster
# MULTI-select of enriched gene sets. Each selected set drives one leading-edge
# small-multiple on the Pathways spatial stage. Selecting is a VIEWING control:
# it mutates one map key, nothing else. Order is preserved (selection order).
# --------------------------------------------------------------------------- #
def get_selected_pathways(cluster: str) -> list[str]:
    """Return a copy of the gene sets selected FOR ``cluster``, in selection order."""
    return list(_ss().get(K_SELECTED_PATHWAYS, {}).get(cluster, []))


def toggle_pathway(cluster: str, gene_set: str) -> None:
    """Add ``gene_set`` to ``cluster``'s selection if absent, else remove it (order kept)."""
    sel = dict(_ss().get(K_SELECTED_PATHWAYS, {}))
    cur = list(sel.get(cluster, []))
    if gene_set in cur:
        cur.remove(gene_set)
    else:
        cur.append(gene_set)
    sel[cluster] = cur
    _ss()[K_SELECTED_PATHWAYS] = sel


def is_pathway_selected(cluster: str, gene_set: str) -> bool:
    """True if ``gene_set`` is selected for ``cluster``."""
    return gene_set in _ss().get(K_SELECTED_PATHWAYS, {}).get(cluster, [])


def set_selected_pathways(cluster: str, gene_sets: list[str]) -> None:
    """Replace ``cluster``'s selected gene sets (used to restore from the URL)."""
    sel = dict(_ss().get(K_SELECTED_PATHWAYS, {}))
    sel[cluster] = list(gene_sets)
    _ss()[K_SELECTED_PATHWAYS] = sel


def active_pathways(cluster: str, cap: int = DEFAULT_MARKER_CAP) -> list[str]:
    """The selected pathways for ``cluster``, capped for small-multiples legibility."""
    return get_selected_pathways(cluster)[:cap]


# --------------------------------------------------------------------------- #
# Selected markers — the multi-select set that drives the spatial small-multiples.
# Selecting is a VIEWING control: it mutates one list key and nothing else. No
# recompute. Order is preserved (selection order) so the panels stay stable.
# --------------------------------------------------------------------------- #
def get_selected_markers(cluster: str) -> list[str]:
    """Return a copy of the markers selected FOR ``cluster``, in selection order.

    Selection is per-cluster: ``selected_markers`` is a ``{cluster: [gene, ...]}``
    map, so switching clusters never carries one cluster's picks onto another —
    each cluster shows only its own selected genes (or none). Empty list before
    anything is selected for that cluster.
    """
    return list(_ss().get(K_SELECTED_MARKERS, {}).get(cluster, []))


def set_selected_markers(cluster: str, genes: list[str]) -> None:
    """Replace ``cluster``'s selected markers (used to restore from the URL)."""
    sel = dict(_ss().get(K_SELECTED_MARKERS, {}))
    sel[cluster] = list(genes)
    _ss()[K_SELECTED_MARKERS] = sel


def toggle_marker(cluster: str, gene: str) -> None:
    """Add ``gene`` to ``cluster``'s selection if absent, else remove it (order kept).

    Viewing-control only: writes a new per-cluster list and returns. Never triggers
    a verdict recompute. The whole map is rewritten (immutable update).
    """
    ss = _ss()
    by = dict(ss.get(K_SELECTED_MARKERS, {}))
    current = list(by.get(cluster, []))
    if gene in current:
        by[cluster] = [g for g in current if g != gene]
    else:
        by[cluster] = current + [gene]
    ss[K_SELECTED_MARKERS] = by


def is_marker_selected(cluster: str, gene: str) -> bool:
    """True if ``gene`` is currently selected for ``cluster``."""
    return gene in _ss().get(K_SELECTED_MARKERS, {}).get(cluster, [])


def clear_markers(cluster: str) -> None:
    """Clear ``cluster``'s marker selection (fresh empty list)."""
    ss = _ss()
    by = dict(ss.get(K_SELECTED_MARKERS, {}))
    by[cluster] = []
    ss[K_SELECTED_MARKERS] = by


def active_markers(cluster: str, cap: int = DEFAULT_MARKER_CAP) -> list[str]:
    """Markers selected for ``cluster``, capped for small-multiples legibility.

    The full selection stays in the evidence table; this only limits how many
    density + feature-UMAP pairs the spatial grid draws. ``cap <= 0`` is uncapped.
    """
    selected = get_selected_markers(cluster)
    if cap is not None and cap > 0:
        return selected[:cap]
    return selected


def active_marker(cluster: str) -> Optional[str]:
    """First marker selected for ``cluster`` (or None) — single-marker convenience."""
    selected = get_selected_markers(cluster)
    return selected[0] if selected else None


# --------------------------------------------------------------------------- #
# Legacy pin API — per-cluster aliases for callers that pin a single marker (e.g.
# the agent may pin a marker to back a chat answer). Still a pure viewing control.
# --------------------------------------------------------------------------- #
def get_pinned_marker(cluster: str) -> Optional[str]:
    """Back-compat: the first marker selected for ``cluster`` (alias of active_marker)."""
    return active_marker(cluster)


def set_pinned_marker(cluster: str, gene: Optional[str]) -> None:
    """Pin ``gene`` for ``cluster`` (append if not already selected), or clear on None.

    Appends rather than replacing, so an agent pinning a marker during chat adds it
    to the current cluster's selection without wiping the biologist's picks. Pure
    viewing control — writes ``selected_markers`` and returns, never a recompute.
    """
    ss = _ss()
    by = dict(ss.get(K_SELECTED_MARKERS, {}))
    if gene is None:
        by[cluster] = []
    elif gene not in by.get(cluster, []):
        by[cluster] = list(by.get(cluster, [])) + [gene]
    ss[K_SELECTED_MARKERS] = by


def toggle_pin(cluster: str, gene: str) -> None:
    """Back-compat alias of :func:`toggle_marker` for one cluster."""
    toggle_marker(cluster, gene)


def unpin_marker(cluster: str) -> None:
    """Back-compat: clear ``cluster``'s marker selection (alias of clear_markers)."""
    clear_markers(cluster)


# --------------------------------------------------------------------------- #
# Bin size — a pure viewing control (change the picture only, never a value)
# --------------------------------------------------------------------------- #
def get_bin_um() -> int:
    """Return the hex-bin size in µm (25 / 50 / 100)."""
    return int(_ss().get(K_BIN_UM, DEFAULT_BIN_UM))


def set_bin_um(bin_um: int) -> None:
    """Set the hex-bin size. Rejects any value not in the presets (no recompute).

    A bin size only selects which *precomputed* density frame is shown; it never
    re-bins or changes a value. Invalid sizes are ignored.
    """
    if int(bin_um) in BIN_SIZES_UM:
        _ss()[K_BIN_UM] = int(bin_um)


# --------------------------------------------------------------------------- #
# Chat thread — PER-CLUSTER. Each cluster owns its own conversation so switching
# clusters swaps threads and never bleeds one cluster's chat into another. The
# agent's *knowledge* of the whole annotation is carried separately (grounded
# global-interpretation context in the system prompt), not through this log.
# Store shape: {cluster_id: [ {role, text, ...}, ... ]}.
# --------------------------------------------------------------------------- #
def _threads() -> dict:
    """Return the {cluster: [msg,...]} store (created lazily; live object)."""
    ss = _ss()
    if K_CHAT_THREAD not in ss or not isinstance(ss[K_CHAT_THREAD], dict):
        ss[K_CHAT_THREAD] = {}
    return ss[K_CHAT_THREAD]


def get_chat_thread(cluster: str) -> list[dict]:
    """Return ``cluster``'s chat thread (the live list, appended in place)."""
    return _threads().setdefault(cluster, [])


def append_message(cluster: str, message: dict) -> None:
    """Append one message dict to ``cluster``'s chat thread (persisted to disk)."""
    get_chat_thread(cluster).append(message)
    _persist_threads()


def clear_chat_thread(cluster: str) -> None:
    """Reset ``cluster``'s chat thread to empty (other clusters untouched; persisted)."""
    _threads()[cluster] = []
    _persist_threads()


def opening_was_posted(cluster: str) -> bool:
    """True if the opening interpretation for ``cluster`` was already posted."""
    return bool(_ss().get(K_OPENING_POSTED, {}).get(cluster))


def mark_opening_posted(cluster: str) -> None:
    """Record that ``cluster``'s opening interpretation has been posted once."""
    ss = _ss()
    posted = dict(ss.get(K_OPENING_POSTED, {}))  # new dict (immutable update)
    posted[cluster] = True
    ss[K_OPENING_POSTED] = posted


def reset_opening_posted(cluster: str) -> None:
    """Forget that ``cluster``'s opening was posted, so it re-posts on next render
    (used by the conversation's 'clear' control to restart a fresh thread)."""
    ss = _ss()
    posted = dict(ss.get(K_OPENING_POSTED, {}))
    posted.pop(cluster, None)
    ss[K_OPENING_POSTED] = posted


# --------------------------------------------------------------------------- #
# Pending note draft — a proposed note awaiting the biologist's two-tap confirm.
# Per cluster (the draft belongs to the cluster it was raised on). Held here as a
# frozen NoteDraft; the confirm card edits scope/basis/status before saving.
# --------------------------------------------------------------------------- #
def _drafts() -> dict:
    ss = _ss()
    if K_PENDING_DRAFT not in ss or not isinstance(ss[K_PENDING_DRAFT], dict):
        ss[K_PENDING_DRAFT] = {}
    return ss[K_PENDING_DRAFT]


def set_pending_draft(cluster: str, draft: Any) -> None:
    """Stash a proposed NoteDraft for ``cluster`` (renders the confirm card)."""
    _drafts()[cluster] = draft


def get_pending_draft(cluster: str) -> Any:
    """The proposed NoteDraft awaiting confirm for ``cluster``, or None."""
    return _drafts().get(cluster)


def clear_pending_draft(cluster: str) -> None:
    """Drop ``cluster``'s pending draft (after Save or Discard)."""
    _drafts().pop(cluster, None)


# --------------------------------------------------------------------------- #
# Citation paper drawer (opened from the holistic-review / opening citations)
# --------------------------------------------------------------------------- #
def is_paper_open() -> bool:
    """True if the citation paper drawer is open."""
    return bool(_ss().get(K_PAPER_OPEN, False))


def get_active_pmid() -> Optional[str]:
    """Return the PMID whose paper the drawer is showing, or None."""
    return _ss().get(K_ACTIVE_PMID)


def open_paper(pmid: str) -> None:
    """Open the citation paper drawer on a given PMID."""
    ss = _ss()
    ss[K_ACTIVE_PMID] = pmid
    ss[K_PAPER_OPEN] = True


def close_paper() -> None:
    """Close the citation paper drawer (leaves the last pmid for re-open)."""
    _ss()[K_PAPER_OPEN] = False


__all__ = [
    # keys / vocabularies
    "BIN_SIZES_UM",
    "DEFAULT_CLUSTER",
    "DEFAULT_BIN_UM",
    "DEFAULT_MARKER_CAP",
    # lifecycle
    "init_state",
    # cluster
    "get_selected_cluster",
    "set_selected_cluster",
    # selected pathways (enrichment multi-select viewing control — no recompute)
    "get_selected_pathways",
    "toggle_pathway",
    "is_pathway_selected",
    "set_selected_pathways",
    "active_pathways",
    # marker multi-select (viewing controls — no recompute)
    "get_selected_markers",
    "set_selected_markers",
    "toggle_marker",
    "is_marker_selected",
    "clear_markers",
    "active_markers",
    "active_marker",
    # legacy pin aliases (back-compat over selected_markers — no recompute)
    "get_pinned_marker",
    "set_pinned_marker",
    "toggle_pin",
    "unpin_marker",
    # bin (viewing control — no recompute)
    "get_bin_um",
    "set_bin_um",
    # chat
    "get_chat_thread",
    "append_message",
    "clear_chat_thread",
    "opening_was_posted",
    "mark_opening_posted",
    "reset_opening_posted",
    "set_pending_draft",
    "get_pending_draft",
    "clear_pending_draft",
    # citation paper drawer
    "is_paper_open",
    "get_active_pmid",
    "open_paper",
    "close_paper",
]
