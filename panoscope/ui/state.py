"""Session-state schema + typed accessors for the Panoscope UI.

All mutable UI state lives in ``st.session_state`` behind these accessors. Every
selection here is a viewing control: it mutates one key and never recomputes a
value. Streamlit is imported lazily so the module imports without a server.
"""


from __future__ import annotations

from panoscope.agent import config as cfg
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Key names (one place; accessors reference these, never bare strings)
# --------------------------------------------------------------------------- #
K_SELECTED_CLUSTER = "selected_cluster"
K_SELECTED_MARKERS = "selected_markers"
K_SELECTED_PATHWAYS = "selected_pathways"
K_BIN_UM = "bin_um"
K_CHAT_THREAD = "chat_thread"
K_PENDING_DRAFT = "pending_draft"
K_OPENING_POSTED = "opening_posted"
K_INITIALIZED = "_ui_initialized"
K_STUDY = "_study"  # (data root, study id) this session opened; None = the default

# --------------------------------------------------------------------------- #
# Closed vocabularies for the viewing/interaction controls
# --------------------------------------------------------------------------- #
BIN_SIZES_UM: tuple[int, ...] = (25, 50, 100)  # µm presets from the wireframe
DEFAULT_BIN_UM = 25

def default_unit() -> str:
    """First unit of the current study, or ``""`` when it has none (no data is
    rendered as unavailable, never a traceback)."""
    try:
        order = cfg.CLUSTER_ORDER
    except Exception:  # noqa: BLE001 - no readable method output is "no units"
        return ""
    return order[0] if order else ""

# How many selected genes get a small-multiple panel; the selection itself is uncapped.
DEFAULT_MARKER_CAP = 4

# --------------------------------------------------------------------------- #
# Default seed values; callables give each session a fresh object.
# --------------------------------------------------------------------------- #
_DEFAULTS: dict[str, Any] = {
    K_SELECTED_CLUSTER: None,  # resolved lazily via default_unit()
    K_SELECTED_MARKERS: dict,   # {cluster: [gene, ...]} — per-cluster multi-select
    K_SELECTED_PATHWAYS: dict,  # {cluster: [gene_set, ...]} — per-cluster multi-select
    K_BIN_UM: DEFAULT_BIN_UM,
    K_CHAT_THREAD: dict,        # {cluster_id: [msg, ...]} — one thread per cluster
    K_PENDING_DRAFT: dict,      # {cluster_id: NoteDraft} — note awaiting confirm
    K_OPENING_POSTED: dict,     # {cluster_id: True} — opening interp posted once
}


def _ss() -> Any:
    """Return ``st.session_state`` (import Streamlit lazily; import-safe module)."""
    import streamlit as st

    return st.session_state


def apply_session_study() -> None:
    """Point this rerun's context at the session's study, or back at the default.

    Unconditional: Streamlit may reuse a script thread across reruns, so a context
    left by another run must not survive. A fresh session restores the study from
    ``?study=&root=`` (the demo when absent or invalid), and the URL follows the session.
    """
    from pathlib import Path

    import streamlit as st

    from panoscope.data import layout
    from panoscope.data.study import Study

    ss, qp = _ss(), st.query_params
    if K_STUDY not in ss:  # a fresh session: a refresh keeps the study the URL names
        sid, root = qp.get("study"), qp.get("root")
        ok = bool(sid and root) and Path(sid).name == sid and Study(id=sid, data_root=Path(root)).exists
        ss[K_STUDY] = (root, sid) if ok else None
    root, sid = ss[K_STUDY] or (None, None)
    for key, value in (("study", sid), ("root", root)):
        if value and qp.get(key) != value:
            qp[key] = value
        elif not value and key in qp:
            del qp[key]
    layout.set_session_root(Path(root) if root else None)
    cfg.set_study(Study(id=sid, data_root=Path(root)) if sid
                  else layout.default_study_id() or cfg.BUNDLED_DEMO_ID)


def open_study(store: Any) -> None:
    """Make ``store`` (``None``: the default) this session's study; per-study UI state starts over."""
    ss = _ss()
    for key in (*_DEFAULTS, K_INITIALIZED, K_STUDY):
        ss.pop(key, None)
    ss[K_STUDY] = (str(store.data_root), store.id) if store is not None else None
    apply_session_study()


def init_state() -> None:
    """Seed every session-state key to its default once per session (idempotent)."""
    ss = _ss()
    if ss.get(K_INITIALIZED):
        return
    for key, default in _DEFAULTS.items():
        if key not in ss:
            ss[key] = default() if callable(default) else default
    _hydrate_threads(ss)  # restore chat transcripts from disk so a refresh keeps them
    ss[K_INITIALIZED] = True


def _hydrate_threads(ss: dict) -> None:
    """Restore persisted chat threads on a fresh session; a restored thread counts
    as having its opening posted."""
    try:
        from panoscope.ui import chat_store

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
        from panoscope.ui import chat_store

        chat_store.save_all(_threads())
    except Exception:  # noqa: BLE001 - losing the transcript must not break the app
        pass


# --------------------------------------------------------------------------- #
# Selected cluster
# --------------------------------------------------------------------------- #
def get_selected_cluster() -> str:
    """Return the active cluster id (defaults to c1 before init)."""
    return _ss().get(K_SELECTED_CLUSTER) or default_unit()


def set_selected_cluster(cluster: str) -> None:
    """Select a cluster. Ignores an unknown id (fail-closed on bad input)."""
    if cluster not in cfg.CLUSTER_ORDER:
        return
    _ss()[K_SELECTED_CLUSTER] = cluster


# Mirrors app.py's page key/value (app.py imports this module, not the reverse).
_K_ACTIVE_PAGE = "active_page"
_PAGE_WITH_CHAT = "markers"   # the Marker-genes page value (mirrors app.py)


def open_cluster_chat(cluster: str) -> None:
    """Select ``cluster`` and switch to the Marker-genes page (its conversation pane).
    Ignores an unknown id (fail-closed)."""
    if cluster not in cfg.CLUSTER_ORDER:
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
# Selected pathways: per-cluster ordered multi-select of gene sets (viewing control).
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
    """Replace ``cluster``'s selected gene sets (URL restore); de-duplicated."""
    sel = dict(_ss().get(K_SELECTED_PATHWAYS, {}))
    sel[cluster] = list(dict.fromkeys(gene_sets))
    _ss()[K_SELECTED_PATHWAYS] = sel


def active_pathways(cluster: str, cap: int = DEFAULT_MARKER_CAP) -> list[str]:
    """The selected pathways for ``cluster``, capped for small-multiples legibility."""
    return get_selected_pathways(cluster)[:cap]


# --------------------------------------------------------------------------- #
# Selected markers: per-cluster ordered multi-select (viewing control, no recompute).
# --------------------------------------------------------------------------- #
def get_selected_markers(cluster: str) -> list[str]:
    """Return a copy of the markers selected FOR ``cluster``, in selection order."""
    return list(_ss().get(K_SELECTED_MARKERS, {}).get(cluster, []))


def set_selected_markers(cluster: str, genes: list[str]) -> None:
    """Replace ``cluster``'s selected markers (URL restore).

    De-duplicated, first occurrence wins: two identical charts raise
    ``StreamlitDuplicateElementId`` and abort the script.
    """
    sel = dict(_ss().get(K_SELECTED_MARKERS, {}))
    sel[cluster] = list(dict.fromkeys(genes))
    _ss()[K_SELECTED_MARKERS] = sel


def toggle_marker(cluster: str, gene: str) -> None:
    """Add ``gene`` to ``cluster``'s selection if absent, else remove it (order kept)."""
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


def active_markers(cluster: str, cap: int = DEFAULT_MARKER_CAP) -> list[str]:
    """Markers selected for ``cluster``, capped for small-multiples; ``cap <= 0`` is
    uncapped."""
    selected = get_selected_markers(cluster)
    if cap is not None and cap > 0:
        return selected[:cap]
    return selected


def set_pinned_marker(cluster: str, gene: Optional[str]) -> None:
    """Pin ``gene`` for ``cluster`` (append, keeping existing picks), or clear on None."""
    ss = _ss()
    by = dict(ss.get(K_SELECTED_MARKERS, {}))
    if gene is None:
        by[cluster] = []
    elif gene not in by.get(cluster, []):
        by[cluster] = list(by.get(cluster, [])) + [gene]
    ss[K_SELECTED_MARKERS] = by


# --------------------------------------------------------------------------- #
# Bin size — a pure viewing control (change the picture only, never a value)
# --------------------------------------------------------------------------- #
def get_bin_um() -> int:
    """Return the hex-bin size in µm (25 / 50 / 100)."""
    return int(_ss().get(K_BIN_UM, DEFAULT_BIN_UM))


def set_bin_um(bin_um: int) -> None:
    """Set the hex-bin size; values outside the presets are ignored. Never a value change."""
    if int(bin_um) in BIN_SIZES_UM:
        _ss()[K_BIN_UM] = int(bin_um)


# --------------------------------------------------------------------------- #
# Chat thread, per cluster: {cluster_id: [ {role, text, ...}, ... ]}.
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


def mark_opening_posted(cluster: str, call: str = "") -> None:
    """Record that ``cluster``'s opening interpretation has been posted once.

    ``call`` is the call it was written for, so a later override is noticed;
    ``True`` means "posted, call unknown" (a restored transcript).
    """
    ss = _ss()
    posted = dict(ss.get(K_OPENING_POSTED, {}))  # new dict (immutable update)
    posted[cluster] = call or True
    ss[K_OPENING_POSTED] = posted


def opening_posted_call(cluster: str):
    """The call ``cluster``'s posted opening was written for, or None if unknown
    (a restored transcript)."""
    v = _ss().get(K_OPENING_POSTED, {}).get(cluster)
    return v if isinstance(v, str) else None


def opening_needed(key: str, call: str) -> bool:
    """True unless an opening was already posted for this ``call``. A restored
    thread (call unknown) is left alone."""
    if not opening_was_posted(key):
        return True
    return opening_posted_call(key) not in (None, call)


def reset_opening_posted(cluster: str) -> None:
    """Forget that ``cluster``'s opening was posted, so it re-posts on next render
    (used by the conversation's 'clear' control to restart a fresh thread)."""
    ss = _ss()
    posted = dict(ss.get(K_OPENING_POSTED, {}))
    posted.pop(cluster, None)
    ss[K_OPENING_POSTED] = posted


# --------------------------------------------------------------------------- #
# Pending note draft, per cluster, awaiting the biologist's two-tap confirm.
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
