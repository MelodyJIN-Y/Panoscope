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
K_BIN_UM = "bin_um"
K_CHAT_THREAD = "chat_thread"
K_SCOPE = "scope"
K_CAPTURE_OPEN = "capture_open"
K_LK_OPEN = "lab_knowledge_open"
K_PAPER_OPEN = "paper_open"
K_ACTIVE_PMID = "active_pmid"
K_OPENING_POSTED = "opening_posted"
K_INITIALIZED = "_ui_initialized"

# --------------------------------------------------------------------------- #
# Closed vocabularies for the viewing/interaction controls
# --------------------------------------------------------------------------- #
BIN_SIZES_UM: tuple[int, ...] = (25, 50, 100)  # µm presets from the wireframe
DEFAULT_BIN_UM = 50

SCOPES: tuple[str, ...] = ("cluster", "dataset", "lab")
DEFAULT_SCOPE = "cluster"

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
    K_SELECTED_MARKERS: list,   # fresh [] per session — ordered multi-select
    K_BIN_UM: DEFAULT_BIN_UM,
    K_CHAT_THREAD: list,        # fresh [] per session
    K_SCOPE: DEFAULT_SCOPE,
    K_CAPTURE_OPEN: False,
    K_LK_OPEN: False,
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
    ss[K_INITIALIZED] = True


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


# --------------------------------------------------------------------------- #
# Selected markers — the multi-select set that drives the spatial small-multiples.
# Selecting is a VIEWING control: it mutates one list key and nothing else. No
# recompute. Order is preserved (selection order) so the panels stay stable.
# --------------------------------------------------------------------------- #
def get_selected_markers() -> list[str]:
    """Return a copy of the selected marker genes, in selection order.

    A copy is returned so callers cannot mutate the stored list by side effect
    (immutable-in, immutable-out). Empty list before anything is selected.
    """
    return list(_ss().get(K_SELECTED_MARKERS, []))


def toggle_marker(gene: str) -> None:
    """Add ``gene`` to the selection if absent, else remove it. Order preserved.

    Viewing-control only: writes a new list to ``selected_markers`` and returns.
    Never triggers a verdict recompute. A new list is stored (immutable update)
    so nothing else holding the old reference is surprised.
    """
    ss = _ss()
    current = list(ss.get(K_SELECTED_MARKERS, []))
    if gene in current:
        ss[K_SELECTED_MARKERS] = [g for g in current if g != gene]
    else:
        ss[K_SELECTED_MARKERS] = current + [gene]


def is_marker_selected(gene: str) -> bool:
    """True if ``gene`` is currently in the selection."""
    return gene in _ss().get(K_SELECTED_MARKERS, [])


def clear_markers() -> None:
    """Clear the whole marker selection (fresh empty list)."""
    _ss()[K_SELECTED_MARKERS] = []


def active_markers(cap: int = DEFAULT_MARKER_CAP) -> list[str]:
    """Return the selected markers, capped to ``cap`` for small-multiples legibility.

    The full selection stays in ``selected_markers`` (and in the evidence table);
    this only limits how many density + feature-UMAP pairs the spatial grid draws.
    ``cap <= 0`` returns the full selection uncapped.
    """
    selected = get_selected_markers()
    if cap is not None and cap > 0:
        return selected[:cap]
    return selected


def active_marker() -> Optional[str]:
    """Return the first selected marker (or None) — back-compat single-marker view.

    Kept so callers that predate multi-select (e.g. a single density/UMAP draw)
    keep working. Equivalent to ``selected_markers[0]`` when anything is selected.
    """
    selected = _ss().get(K_SELECTED_MARKERS, [])
    return selected[0] if selected else None


# --------------------------------------------------------------------------- #
# Legacy pin API — thin aliases over ``selected_markers`` for pre-multi-select
# callers (e.g. the agent may request a marker be pinned). ``pinned_marker`` is
# ``selected_markers[0]``; setting it selects that single gene; clearing empties
# the selection. Still a pure viewing control — no recompute.
# --------------------------------------------------------------------------- #
def get_pinned_marker() -> Optional[str]:
    """Back-compat: the first selected marker, or None (alias of ``active_marker``)."""
    return active_marker()


def set_pinned_marker(gene: Optional[str]) -> None:
    """Back-compat: select ``gene`` as the sole marker (or clear with None).

    Only acts when the gene is not already selected, so an agent re-requesting an
    already-selected marker does not collapse a multi-gene selection. Viewing-
    control only — writes ``selected_markers`` and returns, never a recompute.
    """
    ss = _ss()
    if gene is None:
        ss[K_SELECTED_MARKERS] = []
    elif gene not in ss.get(K_SELECTED_MARKERS, []):
        ss[K_SELECTED_MARKERS] = [gene]


def toggle_pin(gene: str) -> None:
    """Back-compat alias of :func:`toggle_marker` (add if absent else remove)."""
    toggle_marker(gene)


def unpin_marker() -> None:
    """Back-compat: clear the whole marker selection (alias of :func:`clear_markers`)."""
    clear_markers()


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
# Chat thread — list of message dicts {role, text, ...} appended by conversation
# --------------------------------------------------------------------------- #
def get_chat_thread() -> list[dict]:
    """Return the chat thread list (the live object, appended in place)."""
    ss = _ss()
    if K_CHAT_THREAD not in ss:
        ss[K_CHAT_THREAD] = []
    return ss[K_CHAT_THREAD]


def append_message(message: dict) -> None:
    """Append one message dict to the chat thread."""
    get_chat_thread().append(message)


def clear_chat_thread() -> None:
    """Reset the chat thread to empty."""
    _ss()[K_CHAT_THREAD] = []


def opening_was_posted(cluster: str) -> bool:
    """True if the opening interpretation for ``cluster`` was already posted."""
    return bool(_ss().get(K_OPENING_POSTED, {}).get(cluster))


def mark_opening_posted(cluster: str) -> None:
    """Record that ``cluster``'s opening interpretation has been posted once."""
    ss = _ss()
    posted = dict(ss.get(K_OPENING_POSTED, {}))  # new dict (immutable update)
    posted[cluster] = True
    ss[K_OPENING_POSTED] = posted


# --------------------------------------------------------------------------- #
# Note scope (for capture-at-override)
# --------------------------------------------------------------------------- #
def get_scope() -> str:
    """Return the active memory scope (cluster / dataset / lab)."""
    return _ss().get(K_SCOPE, DEFAULT_SCOPE)


def set_scope(scope: str) -> None:
    """Set the active memory scope. Rejects unknown scopes."""
    if scope in SCOPES:
        _ss()[K_SCOPE] = scope


# --------------------------------------------------------------------------- #
# Capture / drawer open flags
# --------------------------------------------------------------------------- #
def is_capture_open() -> bool:
    """True if the capture-at-override note panel is open."""
    return bool(_ss().get(K_CAPTURE_OPEN, False))


def open_capture() -> None:
    """Open the capture-at-override note panel."""
    _ss()[K_CAPTURE_OPEN] = True


def close_capture() -> None:
    """Close the capture-at-override note panel."""
    _ss()[K_CAPTURE_OPEN] = False


def is_lab_knowledge_open() -> bool:
    """True if the lab-knowledge drawer is open."""
    return bool(_ss().get(K_LK_OPEN, False))


def set_lab_knowledge_open(is_open: bool) -> None:
    """Open/close the lab-knowledge drawer."""
    _ss()[K_LK_OPEN] = bool(is_open)


def toggle_lab_knowledge() -> None:
    """Toggle the lab-knowledge drawer."""
    ss = _ss()
    ss[K_LK_OPEN] = not bool(ss.get(K_LK_OPEN, False))


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
    "SCOPES",
    "DEFAULT_CLUSTER",
    "DEFAULT_BIN_UM",
    "DEFAULT_SCOPE",
    "DEFAULT_MARKER_CAP",
    # lifecycle
    "init_state",
    # cluster
    "get_selected_cluster",
    "set_selected_cluster",
    # marker multi-select (viewing controls — no recompute)
    "get_selected_markers",
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
    # scope
    "get_scope",
    "set_scope",
    # capture / drawers
    "is_capture_open",
    "open_capture",
    "close_capture",
    "is_lab_knowledge_open",
    "set_lab_knowledge_open",
    "toggle_lab_knowledge",
    "is_paper_open",
    "get_active_pmid",
    "open_paper",
    "close_paper",
]
