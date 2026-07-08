"""Session-state schema + typed accessors for the Panoscope UI.

All mutable UI state lives in ``st.session_state`` behind these accessors so no
pane pokes a raw key by string. The one grounding-relevant guarantee here:
**pinning a marker only sets ``pinned_marker``** — it is a viewing control, it
never triggers a verdict recompute. Verdicts are computed once and cached in
``ui.data_access``; nothing in this module recomputes a value.

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
K_PINNED_MARKER = "pinned_marker"
K_HOVER_MARKER = "hover_marker"
K_BIN_UM = "bin_um"
K_SPATIAL_VIEW = "spatial_view"
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
SPATIAL_VIEWS: tuple[str, ...] = ("cell_map", "umap", "density")
DEFAULT_SPATIAL_VIEW = "cell_map"

BIN_SIZES_UM: tuple[int, ...] = (25, 50, 100)  # µm presets from the wireframe
DEFAULT_BIN_UM = 50

SCOPES: tuple[str, ...] = ("cluster", "dataset", "lab")
DEFAULT_SCOPE = "cluster"

DEFAULT_CLUSTER = CLUSTER_ORDER[0]  # "c1"

# --------------------------------------------------------------------------- #
# Default seed values. Callables are invoked per-session so no mutable default
# is shared across reruns/sessions (thread/list must be a fresh object).
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, Any] = {
    K_SELECTED_CLUSTER: DEFAULT_CLUSTER,
    K_PINNED_MARKER: None,
    K_HOVER_MARKER: None,
    K_BIN_UM: DEFAULT_BIN_UM,
    K_SPATIAL_VIEW: DEFAULT_SPATIAL_VIEW,
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

    Switching clusters clears the transient hover but keeps the pinned marker
    and bin size — those persist across clusters per the viewing-control spec.
    """
    if cluster not in CLUSTER_ORDER:
        return
    ss = _ss()
    ss[K_SELECTED_CLUSTER] = cluster
    ss[K_HOVER_MARKER] = None


# --------------------------------------------------------------------------- #
# Pinned marker — the one pin that drives all three linked spatial views.
# Pinning is a VIEWING control: it sets a key and nothing else. No recompute.
# --------------------------------------------------------------------------- #
def get_pinned_marker() -> Optional[str]:
    """Return the pinned marker gene, or None."""
    return _ss().get(K_PINNED_MARKER)


def set_pinned_marker(gene: Optional[str]) -> None:
    """Pin ``gene`` (or clear with None). Sets ``pinned_marker`` ONLY.

    This must never trigger a verdict recompute — it changes the picture, not a
    value. It writes a single session-state key and returns.
    """
    _ss()[K_PINNED_MARKER] = gene


def toggle_pin(gene: str) -> None:
    """Pin ``gene`` if not pinned, else unpin. Viewing-control only, no recompute."""
    ss = _ss()
    ss[K_PINNED_MARKER] = None if ss.get(K_PINNED_MARKER) == gene else gene


def unpin_marker() -> None:
    """Clear the pinned marker."""
    _ss()[K_PINNED_MARKER] = None


def get_hover_marker() -> Optional[str]:
    """Return the hover-preview marker (transient), or None."""
    return _ss().get(K_HOVER_MARKER)


def set_hover_marker(gene: Optional[str]) -> None:
    """Set the hover-preview marker. Viewing-control only, no recompute."""
    _ss()[K_HOVER_MARKER] = gene


def active_marker() -> Optional[str]:
    """Return hover if set (preview), else the pinned marker — what to draw now."""
    ss = _ss()
    return ss.get(K_HOVER_MARKER) or ss.get(K_PINNED_MARKER)


# --------------------------------------------------------------------------- #
# Bin size + spatial view — both pure viewing controls (change the picture only)
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


def get_spatial_view() -> str:
    """Return the active spatial view (cell_map / umap / density)."""
    return _ss().get(K_SPATIAL_VIEW, DEFAULT_SPATIAL_VIEW)


def set_spatial_view(view: str) -> None:
    """Switch the spatial view. Rejects unknown views. Viewing-control only."""
    if view in SPATIAL_VIEWS:
        _ss()[K_SPATIAL_VIEW] = view


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
    "SPATIAL_VIEWS",
    "BIN_SIZES_UM",
    "SCOPES",
    "DEFAULT_CLUSTER",
    "DEFAULT_BIN_UM",
    "DEFAULT_SPATIAL_VIEW",
    "DEFAULT_SCOPE",
    # lifecycle
    "init_state",
    # cluster
    "get_selected_cluster",
    "set_selected_cluster",
    # pin / hover (viewing controls — no recompute)
    "get_pinned_marker",
    "set_pinned_marker",
    "toggle_pin",
    "unpin_marker",
    "get_hover_marker",
    "set_hover_marker",
    "active_marker",
    # bin / view (viewing controls — no recompute)
    "get_bin_um",
    "set_bin_um",
    "get_spatial_view",
    "set_spatial_view",
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
