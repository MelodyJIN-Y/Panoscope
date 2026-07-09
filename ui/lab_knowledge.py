"""Lab-knowledge drawer — "what this tool knows about your lab".

Renders the stored :class:`~agent.types.Note` objects the agent is allowed to
cite. Each note shows its claim, its scope / basis / status chips, who wrote it
and when, and any literature tension (agreeing vs dissenting PMIDs) captured at
reconciliation. Every field here comes straight off the persisted Note — this
panel never computes a value, never invents a citation, and never smooths a
disagreement over: if a note carries dissent, the dissent is shown.

Reads are always fresh via ``ui.data_access.read_notes`` (notes mutate on save,
so the drawer must never show a stale list). The delete control removes the
note's git-tracked JSON file through ``agent.memory``'s own path resolver and
appends a ``note_deleted`` event to the decision log, so a deletion leaves an
attributed trace in the same append-only ledger every other memory mutation does.

Streamlit is imported lazily inside every function, so ``import ui.lab_knowledge``
works with no server running (the module touches no ``st.*`` at import time).
"""

from __future__ import annotations

from typing import Optional

from agent.types import Citation, Note

from ui import data_access as da
from ui import format as fmt
from ui import state

# --------------------------------------------------------------------------- #
# Human-readable labels for the closed vocabularies (Scope / Basis / Status).
# These only *relabel* stored enum values; they never change one.
# --------------------------------------------------------------------------- #
_SCOPE_LABEL: dict[str, str] = {
    "cluster": "cluster",
    "dataset": "this dataset",
    "lab": "lab-wide",
}
_BASIS_LABEL: dict[str, str] = {
    "paper": "a paper",
    "own_validation": "our own data",
    "convention": "convention",
}
_STATUS_LABEL: dict[str, str] = {
    "firm": "firm rule",
    "tentative": "tentative",
}

_EMPTY_TEXT = (
    "No notes yet. Override or confirm something in the chat and it is saved "
    "here with its basis and any literature tension."
)
_DRAWER_TITLE = "What this tool knows about your lab"
_DRAWER_SUB = (
    "Notes the agent can cite. Each carries its basis and any literature tension."
)

# Drawer-local styles for the note cards. These reference CSS variables that
# ``ui.theme.inject_css`` already defines (--hair, --accent, --faint, --absent,
# etc.), so they inherit the design tokens without duplicating them. Class names
# and values mirror the wireframe (.lki / .lke / .sco). ``.tension`` lives in
# ``ui.theme`` already; we do not redefine it here. Re-injecting is harmless.
_LK_CSS = """
.lki { border:1px solid var(--hair); border-radius:9px; padding:11px 12px;
       margin-bottom:2px; font-size:13px; background:var(--paper); }
.lki-claim { line-height:1.45; color:var(--ink); }
.lki .m { font-family:var(--mono); font-size:10px; color:var(--faint);
          margin-top:8px; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.lki .sco { background:var(--accent-soft); color:var(--accent);
            padding:2px 7px; border-radius:5px; }
.lke { font-size:12px; color:var(--muted); border:1px dashed var(--hair);
       border-radius:9px; padding:16px; text-align:center; }
"""


def _inject_local_css() -> None:
    """Inject the note-card styles once per rerun (guarded; import-safe)."""
    import streamlit as st

    st.markdown(f"<style>{_LK_CSS}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small helpers (read the Note only; never compute a value)
# --------------------------------------------------------------------------- #
def _scope_text(note: Note) -> str:
    """Human scope label, naming the cluster for cluster-scoped notes."""
    if note.scope == "cluster" and note.scope_ref.cluster:
        return f"cluster {fmt.short_cluster_id(note.scope_ref.cluster)}"
    return _SCOPE_LABEL.get(note.scope, note.scope)


def _basis_text(note: Note) -> str:
    return _BASIS_LABEL.get(note.basis, note.basis)


def _status_text(note: Note) -> str:
    return _STATUS_LABEL.get(note.status, note.status)


def _created_date(note: Note) -> str:
    """The date portion of the ISO created_at stamp (whole stamp if unparseable)."""
    if not note.created_at:
        return "—"  # em dash
    return note.created_at.split("T", 1)[0]


def _pmids(cites: tuple[Citation, ...]) -> str:
    """Comma-joined ``PMID:xxxx`` list for a citation tuple (real PMIDs only)."""
    return ", ".join(f"PMID:{c.pmid}" for c in cites if c.pmid)


def _tension_line(note: Note) -> Optional[str]:
    """Render the literature-tension summary for a note, or None when there is none.

    Uses only what reconciliation already stored on the note. Thin literature is
    reported honestly ("literature thin") rather than hidden; dissent is always
    surfaced alongside agreement.
    """
    t = note.tension
    if t.agree or t.dissent:
        bits: list[str] = []
        if t.agree:
            bits.append(f"agrees ({len(t.agree)}): {_pmids(t.agree)}")
        if t.dissent:
            bits.append(f"dissents ({len(t.dissent)}): {_pmids(t.dissent)}")
        return "literature tension — " + " · ".join(bits)
    if t.thin:
        return "literature thin — no supporting reference found on file"
    return None


# --------------------------------------------------------------------------- #
# Delete — remove the note file via agent.memory's own path resolver, then log.
# We never re-implement the memory layout; we ask memory where the file is.
# --------------------------------------------------------------------------- #
def _delete_note(note: Note) -> None:
    """Delete one note's JSON file and append a ``note_deleted`` decision event.

    Routed through ``agent.memory._note_path`` so the on-disk convention stays
    owned by the memory module. A missing file is a no-op (already gone). The
    deletion is logged to the append-only decision ledger for the audit trail.
    """
    from agent import memory

    path = memory._note_path(note.id, None)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Fail-soft: a filesystem hiccup must not crash the drawer.
        return
    memory.log_decision(
        kind="note_deleted",
        cluster=note.scope_ref.cluster,
        note_id=note.id,
        actor=note.author or "melody.xyjin@gmail.com",
        detail=f"scope={note.scope} basis={note.basis} status={note.status}",
    )


# --------------------------------------------------------------------------- #
# One note card
# --------------------------------------------------------------------------- #
def _render_note_card(note: Note) -> None:
    """Render a single note: claim, meta chips, tension, and a delete control."""
    import streamlit as st

    scope = _scope_text(note)
    basis = _basis_text(note)
    status = _status_text(note)
    author = note.author or "you"
    date = _created_date(note)
    tension = _tension_line(note)

    tension_html = f'<div class="tension">{tension}</div>' if tension else ""
    st.markdown(
        f"""<div class="lki">
  <div class="lki-claim">{note.claim}</div>
  <div class="m">
    <span class="sco">{scope}</span>
    <span>basis: {basis}</span>
    <span>· {status}</span>
    <span>· {author}</span>
    <span>· {date}</span>
  </div>
  {tension_html}
</div>""",
        unsafe_allow_html=True,
    )
    # Native button for the delete action (HTML onclick can't call Python).
    if st.button("delete", key=f"lk_del_{note.id}", help="Remove this note"):
        _delete_note(note)
        st.rerun()


# --------------------------------------------------------------------------- #
# Public: the drawer
# --------------------------------------------------------------------------- #
def render_lab_panel(*, expanded: Optional[bool] = None) -> None:
    """Render the lab-knowledge drawer: every stored note the agent can cite.

    Reads notes FRESH each call (``ui.data_access.read_notes``) so the list is
    never stale after a save or delete. Open/closed state comes from
    ``ui.state.is_lab_knowledge_open()`` unless ``expanded`` overrides it, so the
    header "Lab knowledge N" button and this drawer stay in sync.

    Renders, per note: the claim, a scope / basis / status meta line, the author
    and date, and any literature tension (agree / dissent PMIDs) — all read
    directly off the persisted Note. Each note has a delete control that removes
    its file and reruns. Nothing here is fabricated or recomputed.
    """
    import streamlit as st

    _inject_local_css()
    notes = da.read_notes()
    is_open = state.is_lab_knowledge_open() if expanded is None else bool(expanded)

    with st.expander(f"{_DRAWER_TITLE}  ·  {len(notes)}", expanded=is_open):
        st.caption(_DRAWER_SUB)
        if not notes:
            st.markdown(
                f'<div class="lke">{_EMPTY_TEXT}</div>', unsafe_allow_html=True
            )
            return
        # Newest first — the most recent lab judgment reads at the top.
        for note in sorted(notes, key=lambda n: (n.created_at, n.id), reverse=True):
            _render_note_card(note)


def render_lab_page() -> None:
    """Full-page lab-knowledge view (its own top tab): every stored note the
    agent can cite, newest first, each with its scope / basis / status, author,
    date, and any literature tension. Reads notes FRESH so the page never shows a
    stale list after the agent captures or the user deletes one. Nothing here is
    fabricated or recomputed — every field is read off the persisted Note.
    """
    import streamlit as st

    _inject_local_css()
    notes = da.read_notes()

    st.markdown(
        f'<p class="pano-eyebrow">Lab knowledge · {len(notes)} notes</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="pano-rat" style="max-width:70ch">{_DRAWER_SUB} '
        "They are captured by telling the agent in the chat — an override or a "
        "confirmation — and stay here, git-tracked and attributed.</div>",
        unsafe_allow_html=True,
    )

    if not notes:
        st.markdown(f'<div class="lke">{_EMPTY_TEXT}</div>', unsafe_allow_html=True)
        return

    # Newest first — the most recent lab judgment reads at the top. A modest
    # max-width keeps the note cards readable on a wide page.
    left, _spacer = st.columns([0.62, 0.38])
    with left:
        for note in sorted(notes, key=lambda n: (n.created_at, n.id), reverse=True):
            _render_note_card(note)


def lab_knowledge_button(*, key: str = "lk_open_btn") -> None:
    """Render the header "Lab knowledge N" pill that toggles the drawer open.

    A thin companion to :func:`render_lab_panel`: it shows the live note count
    and flips ``ui.state``'s lab-knowledge-open flag. Count is a fresh read so it
    tracks saves and deletes immediately.
    """
    import streamlit as st

    n = len(da.read_notes())
    if st.button(f"Lab knowledge  {n}", key=key):
        state.toggle_lab_knowledge()
        st.rerun()


__all__ = ["render_lab_panel", "lab_knowledge_button"]
