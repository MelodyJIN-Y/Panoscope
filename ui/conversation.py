"""Conversation pane — the primary interface. The chat is the product.

``render_conversation(cluster)`` is the whole third pane of the shell. On a fresh
cluster it posts the grounded opening interpretation (``agent.loop.opening_interpretation``)
BEFORE any question, then renders the running thread, then an ask box that calls
``agent.loop.chat``. Every agent turn shows its prose, one condensed Sources
line naming only the provenance kinds present (jazzPanda / panel / PubMed / lab
note), and its inline ``PMID:xxx`` citations rendered as clickable buttons that
open the paper drawer via session state.

Capture-at-override lives here too: a note editor with scope / basis / status
chips and a Save that calls ``agent.memory.create_note``. Saving posts the note's
reconciliation (agreement + dissent) back into the thread, never a bare "got it".

Grounding discipline: this module renders values the engine produced and never
computes one. It reads the agent's ``AgentResponse`` verbatim. Pinning a marker
that the agent chose (``resp.pin_marker``) is a viewing control — it sets one
session key and never recomputes a verdict.

Streamlit is imported lazily inside the render functions so importing
``ui.conversation`` needs no running server.
"""

from __future__ import annotations

import html
import re
from typing import Any, Optional

from agent.types import AgentResponse, Citation, Note, Source

from ui import data_access as dax
from ui import format as fmt  # noqa: F401 - shared infra per task contract
from ui import state

# --------------------------------------------------------------------------- #
# Thread message dicts stored in session_state (via ui.state.append_message):
#   {"role": "agent"|"user"|"system", "text": str, "resp": Optional[AgentResponse]}
# The AgentResponse is frozen and picklable, safe to hold across reruns.
# --------------------------------------------------------------------------- #
_ROLE_AGENT = "agent"
_ROLE_USER = "user"
_ROLE_SYSTEM = "system"

# Source.kind -> display name for the condensed "Sources:" line under a turn.
# Only the KINDS present are shown (deduped), in this fixed provenance order —
# no per-number chips. Numbers stay inline in the prose; PMIDs stay clickable via
# the citation buttons below the bubble.
_SRC_NAME: dict[str, str] = {
    "jz": "jazzPanda",
    "panel": "panel",
    "lit": "PubMed",
    "mem": "lab note",
}
# Fixed left-to-right order for the condensed line (grounded floor first).
_SRC_ORDER: tuple[str, ...] = ("jz", "panel", "lit", "mem")

# Basis chip labels (wireframe wording) -> the memory.create_note Basis literal.
_BASIS_CHOICES: tuple[tuple[str, str], ...] = (
    ("a paper", "paper"),
    ("our own data", "own_validation"),
    ("convention", "convention"),
)
_STATUS_CHOICES: tuple[tuple[str, str], ...] = (
    ("firm rule", "firm"),
    ("tentative", "tentative"),
)
# Scope options come from ui.state.SCOPES = ("cluster","dataset","lab").
_SCOPE_LABEL: dict[str, str] = {
    "cluster": "this cluster",
    "dataset": "this dataset",
    "lab": "lab-wide",
}

# Session keys owned by this pane (namespaced so no other pane collides).
_K_CAPTURE_CLAIM = "conv_capture_claim"
_K_CAPTURE_BASIS = "conv_capture_basis"
_K_CAPTURE_STATUS = "conv_capture_status"
_K_ASK = "conv_ask_input"

# Matches PMID:12345678 (case-insensitive, optional space) so citations in the
# agent's prose become clickable. Group 1 is the numeric id.
_PMID_RE = re.compile(r"PMID:\s*(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_conversation(cluster: str) -> None:
    """Render the conversation pane for ``cluster`` (opening + thread + ask box).

    Reads all UI state through ``ui.state`` (selected cluster, chat thread,
    capture flags, active PMID). Posts the opening interpretation exactly once
    per cluster (guarded by ``state.opening_was_posted``). Never raises into the
    caller: the agent layer already returns a grounded fallback on any failure.
    """
    import streamlit as st

    _ensure_opening(cluster)

    st.markdown(
        '<div class="pano-sect">Conversation'
        f'<span class="r mono">{html.escape(cluster)}</span></div>',
        unsafe_allow_html=True,
    )

    _render_thread(cluster)
    _render_ask_box(cluster)


# --------------------------------------------------------------------------- #
# Opening interpretation — posted before any question, once per cluster.
# --------------------------------------------------------------------------- #
def _ensure_opening(cluster: str) -> None:
    """Post the grounded opening interpretation the first time a cluster opens.

    Guarded by ``state.opening_was_posted`` so a rerun never double-posts. The
    agent's opening is auto-pinned to its leading driver (a viewing control:
    sets ``pinned_marker`` only, no verdict recompute).
    """
    if state.opening_was_posted(cluster):
        return

    resp = _safe_opening(cluster)
    state.append_message({"role": _ROLE_AGENT, "text": resp.text, "resp": resp})
    # Auto-pin the opening's leading driver — viewing control only.
    if resp.pin_marker and state.get_pinned_marker() is None:
        state.set_pinned_marker(resp.pin_marker)
    state.mark_opening_posted(cluster)


def _safe_opening(cluster: str) -> AgentResponse:
    """Fetch the opening interpretation; never raise into the render path."""
    try:
        return dax.get_agent().opening_interpretation(cluster)
    except Exception:  # noqa: BLE001 - the pane must never crash the app
        from agent import loop as agent_loop

        return agent_loop.opening_interpretation(cluster)


# --------------------------------------------------------------------------- #
# Thread rendering
# --------------------------------------------------------------------------- #
def _render_thread(cluster: str) -> None:
    """Render each message in the thread as a bubble with its sources."""
    import streamlit as st

    thread = state.get_chat_thread()
    if not thread:
        st.markdown(
            '<div class="who">agent</div>'
            '<div class="bubble a">Opening interpretation loads when a cluster is selected.</div>',
            unsafe_allow_html=True,
        )
        return

    for idx, msg in enumerate(thread):
        role = msg.get("role", _ROLE_AGENT)
        if role == _ROLE_USER:
            _render_user_bubble(msg.get("text", ""))
        elif role == _ROLE_SYSTEM:
            _render_system_bubble(msg.get("text", ""))
        else:
            _render_agent_bubble(msg, idx)


def _render_user_bubble(text: str) -> None:
    import streamlit as st

    st.markdown(
        '<div class="who">you</div>'
        f'<div class="bubble u">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _render_system_bubble(text: str) -> None:
    import streamlit as st

    st.markdown(
        f'<div class="bubble sys">{html.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def _render_agent_bubble(msg: dict, idx: int) -> None:
    """Render one agent turn: prose (with inline citations) + source chips.

    Citations in the prose (``PMID:xxx``) are rendered as small buttons so a
    click opens the paper drawer (``state.open_paper``). The chip row uses the
    ``src-*`` classes from ``ui.theme``.
    """
    import streamlit as st

    resp: Optional[AgentResponse] = msg.get("resp")
    text = msg.get("text", "")

    prose_html, cited_pmids = _linkify_citations(text)
    sources_html = _sources_line_html(resp.sources if resp else ())
    verify_html = _verify_line_html(resp)
    st.markdown(
        f'<div class="who">agent</div>'
        f'<div class="bubble a">{prose_html}{verify_html}{sources_html}</div>',
        unsafe_allow_html=True,
    )

    # Clickable citation buttons open the paper drawer. Streamlit cannot embed a
    # callback inside markdown HTML, so the inline dotted-underline text shows
    # WHERE the citation is, and these compact buttons are the actual affordance.
    citations = list(resp.citations) if resp else []
    _render_citation_buttons(citations, cited_pmids, idx)


def _verify_line_html(resp: Optional[AgentResponse]) -> str:
    """Return a small 're-check this' line when the turn is flagged verify."""
    if resp is None or not resp.verify:
        return ""
    return (
        '<div class="tension" style="border-left-color:var(--absent)">'
        "&#9873; re-check this &mdash; evidence is thin; confirm before relying on it."
        "</div>"
    )


def _sources_line_html(sources: tuple[Source, ...]) -> str:
    """Build ONE condensed provenance line for a turn (empty when no sources).

    Collapses the per-number chip stack into a single muted line naming only the
    source KINDS present (deduped), e.g. ``Sources jazzPanda · PubMed · lab note``.
    The numbers themselves stay inline in the prose and any cited PMIDs stay
    clickable via the citation buttons rendered below the bubble — this line is
    provenance, not the values. Uses the shared ``.pano-sources`` theme class.
    """
    kinds_present = {s.kind for s in sources}
    names = [_SRC_NAME[k] for k in _SRC_ORDER if k in kinds_present]
    if not names:
        return ""

    spans = f'<span class="sep">{chr(0x00B7)}</span>'.join(
        f'<span class="src">{html.escape(n)}</span>' for n in names
    )
    return f'<div class="pano-sources"><span class="lbl">Sources</span>{spans}</div>'


# --------------------------------------------------------------------------- #
# Inline citation linkification
# --------------------------------------------------------------------------- #
def _linkify_citations(text: str) -> tuple[str, list[str]]:
    """Escape prose and turn ``PMID:xxx`` mentions into dotted-underline spans.

    Returns ``(html, pmids)`` where ``pmids`` is the ordered, de-duplicated list
    of PMIDs found — the caller renders a clickable button per PMID (Streamlit
    cannot bind a callback inside a markdown string).
    """
    pmids: list[str] = []
    parts: list[str] = []
    last = 0
    for m in _PMID_RE.finditer(text):
        parts.append(html.escape(text[last : m.start()]))
        pmid = m.group(1)
        if pmid not in pmids:
            pmids.append(pmid)
        parts.append(
            f'<span class="pcite">\U0001f4c4 PMID:{html.escape(pmid)}</span>'
        )
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts), pmids


def _render_citation_buttons(
    citations: list[Citation], cited_pmids: list[str], idx: int
) -> None:
    """Render one compact button per citation that opens the paper drawer.

    Prefers the ``Citation`` objects on the response (they carry title/authors);
    falls back to a bare button for any PMID mentioned in prose without a matching
    citation object, so no cited paper is unreachable.
    """
    import streamlit as st

    by_pmid = {str(c.pmid): c for c in citations}
    # Order: citations first (rich), then any prose-only PMIDs.
    ordered: list[str] = [str(c.pmid) for c in citations]
    for p in cited_pmids:
        if p not in ordered:
            ordered.append(p)
    if not ordered:
        return

    cols = st.columns(min(len(ordered), 3))
    for i, pmid in enumerate(ordered):
        cite = by_pmid.get(pmid)
        label = _citation_button_label(pmid, cite)
        with cols[i % len(cols)]:
            if st.button(
                label,
                key=f"cite_{idx}_{i}_{pmid}",
                use_container_width=True,
                help="Open the paper in the citation drawer",
            ):
                state.open_paper(pmid)
                _rerun(st)


def _citation_button_label(pmid: str, cite: Optional[Citation]) -> str:
    """Short button label like ``[doc] Rivera 2022`` or ``[doc] PMID:123``."""
    if cite is not None:
        first = _first_author(cite.authors)
        year = f" {cite.year}" if cite.year else ""
        if first:
            return f"\U0001f4c4 {first}{year}"
    return f"\U0001f4c4 PMID:{pmid}"


def _first_author(authors: str) -> str:
    """Return the first author's surname-ish token from an authors string."""
    a = (authors or "").strip()
    if not a:
        return ""
    first = a.split(",")[0].split(";")[0].strip()
    # "Rivera et al." -> "Rivera"; "Smith J" -> "Smith".
    token = first.split(" ")[0].strip()
    return token or first


# --------------------------------------------------------------------------- #
# Ask box
# --------------------------------------------------------------------------- #
def _render_ask_box(cluster: str) -> None:
    """Render the ask input + Ask button + the 'capture a note' toggle.

    On submit, appends the user turn, calls ``agent.loop.chat`` through the
    cached agent, appends the grounded answer, and — if the answer chose a pin —
    applies it as a viewing control. Never raises: the agent returns a fallback.
    """
    import streamlit as st

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    with st.form(key=f"ask_form_{cluster}", clear_on_submit=True):
        query = st.text_input(
            "Ask",
            key=_K_ASK,
            placeholder=f"Ask, or override the call (e.g. 'this is CAF, our own data')…",
            label_visibility="collapsed",
        )
        asked = st.form_submit_button("Ask", use_container_width=True, type="primary")

    if asked and query and query.strip():
        _submit_query(cluster, query.strip())
        _rerun(st)


def _submit_query(cluster: str, query: str) -> None:
    """Append the user turn, run the agent, append its grounded answer."""
    state.append_message({"role": _ROLE_USER, "text": query, "resp": None})
    resp = _safe_chat(cluster, query)
    state.append_message({"role": _ROLE_AGENT, "text": resp.text, "resp": resp})
    # If the agent pinned a marker to back its answer, honor it (viewing control).
    if resp.pin_marker:
        state.set_pinned_marker(resp.pin_marker)


def _safe_chat(cluster: str, query: str) -> AgentResponse:
    """Run one chat turn; never raise into the render path."""
    try:
        history = _history_for_agent()
        return dax.get_agent().chat(query, cluster=cluster, history=history)
    except Exception:  # noqa: BLE001 - agent has its own fallback; belt + braces
        from agent import loop as agent_loop

        return agent_loop.chat(query, cluster=cluster)


def _history_for_agent() -> list[dict]:
    """Flatten the thread into a minimal role/text history for the agent loop."""
    out: list[dict] = []
    for msg in state.get_chat_thread():
        role = msg.get("role")
        if role == _ROLE_USER:
            out.append({"role": "user", "content": msg.get("text", "")})
        elif role == _ROLE_AGENT:
            out.append({"role": "assistant", "content": msg.get("text", "")})
    return out


# --------------------------------------------------------------------------- #
# Capture-at-override — scope / basis / status chips + Save.
# --------------------------------------------------------------------------- #
def _render_capture(cluster: str) -> None:
    """Render the note-capture editor when open (scope/basis/status + Save)."""
    import streamlit as st

    if not state.is_capture_open():
        return

    st.markdown(
        '<div class="bubble sys" style="background:#FCF7EC;color:#7a5b1e">'
        "Your call is kept and cross-checked &mdash; agreement and dissent stay visible."
        "</div>",
        unsafe_allow_html=True,
    )

    claim = st.text_input(
        "Note",
        key=_K_CAPTURE_CLAIM,
        placeholder="e.g. In our breast panels, this reads as CAF, not pericyte.",
        label_visibility="collapsed",
    )

    # scope chips (cluster / dataset / lab) — reuse the shared scope state.
    st.caption("scope")
    _scope_chip_row(cluster)

    # basis chips (a paper / our own data / convention)
    st.caption("basis")
    basis = _radio_chip_row(key=_K_CAPTURE_BASIS, choices=_BASIS_CHOICES, default="paper")

    # status chips (firm rule / tentative)
    st.caption("status")
    status = _radio_chip_row(
        key=_K_CAPTURE_STATUS, choices=_STATUS_CHOICES, default="firm"
    )

    c_save, c_cancel = st.columns([2, 1])
    with c_save:
        save = st.button(
            "Save note", use_container_width=True, type="primary", key="cap_save"
        )
    with c_cancel:
        cancel = st.button("Cancel", use_container_width=True, key="cap_cancel")

    if cancel:
        state.close_capture()
        _rerun(st)

    if save:
        _save_note(cluster, claim, basis, status)


def _scope_chip_row(cluster: str) -> None:
    """Render scope selection as a segmented radio bound to the shared scope."""
    import streamlit as st

    scopes = list(state.SCOPES)
    current = state.get_scope()
    labels = {s: _SCOPE_LABEL.get(s, s) for s in scopes}
    picked = st.radio(
        "scope",
        options=scopes,
        index=scopes.index(current) if current in scopes else 0,
        format_func=lambda s: labels[s],
        horizontal=True,
        label_visibility="collapsed",
        key="conv_scope_radio",
    )
    if picked != current:
        state.set_scope(picked)


def _radio_chip_row(
    *, key: str, choices: tuple[tuple[str, str], ...], default: str
) -> str:
    """Render a horizontal radio of (label, value) choices; return the value."""
    import streamlit as st

    values = [v for _, v in choices]
    label_by_value = {v: lbl for lbl, v in choices}
    idx = values.index(default) if default in values else 0
    picked = st.radio(
        key,
        options=values,
        index=idx,
        format_func=lambda v: label_by_value.get(v, v),
        horizontal=True,
        label_visibility="collapsed",
        key=f"{key}_radio",
    )
    return picked


def _save_note(cluster: str, claim: str, basis: str, status: str) -> None:
    """Persist the note via ``agent.memory.create_note`` and report reconciliation.

    The note is saved with the shared scope. On success we post a system line
    (Saved · scope · basis · status) AND an agent line summarizing the literature
    reconciliation carried on the note's tension — agreement and dissent stay
    visible, never a bare acknowledgement. A save error posts a plain-language
    message and keeps the editor open so the biologist can retry.
    """
    import streamlit as st

    text = (claim or "").strip()
    if not text:
        _toast(st, "Write the note first — nothing to save.")
        return

    scope = state.get_scope()
    verdict = _verdict_or_none(cluster)
    subject_cell_type = verdict.cell_type if verdict else None
    subject_markers = tuple(verdict.key_markers) if verdict else ()

    try:
        from agent import memory

        saved: Note = memory.create_note(
            claim=text,
            scope=scope,  # type: ignore[arg-type]
            basis=basis,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            cluster=cluster if scope == "cluster" else None,
            subject_cell_type=subject_cell_type,
            subject_markers=subject_markers,
        )
    except Exception as exc:  # noqa: BLE001 - surface a clean message, keep editor open
        _toast(st, f"Could not save the note: {exc}")
        return

    # System confirmation line (scope · basis · status).
    basis_label = _label_for(basis, _BASIS_CHOICES)
    status_label = _label_for(status, _STATUS_CHOICES)
    scope_label = _SCOPE_LABEL.get(scope, scope)
    state.append_message(
        {
            "role": _ROLE_SYSTEM,
            "text": f"Saved · {scope_label} · basis: {basis_label} · {status_label}.",
            "resp": None,
        }
    )
    # Agent reconciliation line — surfaces the tension the note was born with.
    recon_resp = _reconciliation_response(saved)
    state.append_message(
        {"role": _ROLE_AGENT, "text": recon_resp.text, "resp": recon_resp}
    )

    state.close_capture()
    _reset_capture_fields()
    _rerun(st)


def _reconciliation_response(note: Note) -> AgentResponse:
    """Build the agent's reconciliation turn for a freshly saved note.

    Reads the note's tension (agreement + dissent citations, thinness) and states
    it plainly, keeping the biologist's call. Citations here are the real ones the
    note was reconciled against; they render as clickable buttons. No number or
    citation is invented — everything comes off the stored note.
    """
    agree = tuple(note.tension.agree)
    dissent = tuple(note.tension.dissent)
    citations = agree + dissent

    if note.tension.thin and not citations:
        text = (
            "I kept your call and checked the literature. It is thin here — "
            "I did not find a clear paper either way, so I am recording your note "
            "as-is and will re-check when more shows up."
        )
    else:
        parts = ["I kept your call and cross-checked the literature."]
        if agree:
            parts.append(
                f"{len(agree)} paper{'s' if len(agree) != 1 else ''} agree "
                + " ".join(f"PMID:{c.pmid}" for c in agree)
                + "."
            )
        if dissent:
            parts.append(
                f"{len(dissent)} report{'s' if len(dissent) != 1 else ''} the "
                "opposite context "
                + " ".join(f"PMID:{c.pmid}" for c in dissent)
                + " — a tension I am keeping visible with the note, not "
                "smoothing over."
            )
        if not agree and not dissent:
            parts.append("The literature was thin; I am recording your note as-is.")
        text = " ".join(parts)

    sources = (
        Source(
            kind="mem",
            ref=note.id,
            value=note.claim,
            detail="reconciled with lab note",
        ),
    )
    return AgentResponse(
        text=text,
        sources=sources,
        verify=False,
        grounding=_note_grounding(note),
        citations=citations,
        note_written=note,
    )


def _note_grounding(note: Note):
    """Minimal grounding sidecar for a reconciliation turn (PMIDs + note id)."""
    from agent.types import GroundingSidecar

    pmids = tuple(str(c.pmid) for c in (note.tension.agree + note.tension.dissent))
    return GroundingSidecar(numbers=(), markers=(), pmids=pmids, notes_used=(note.id,))


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _verdict_or_none(cluster: str) -> Optional[Any]:
    """Cached verdict for a cluster, or None if unavailable (never raises)."""
    try:
        return dax.verdict_for(cluster)
    except Exception:  # noqa: BLE001
        return None


def _label_for(value: str, choices: tuple[tuple[str, str], ...]) -> str:
    """Return the display label for a stored value from a choices table."""
    for lbl, val in choices:
        if val == value:
            return lbl
    return value


def _reset_capture_fields() -> None:
    """Clear the capture editor's claim field after a successful save."""
    ss = _ss()
    if _K_CAPTURE_CLAIM in ss:
        ss[_K_CAPTURE_CLAIM] = ""


def _ss() -> Any:
    import streamlit as st

    return st.session_state


def _toast(st: Any, message: str) -> None:
    """Show a transient message (falls back to a warning banner on old Streamlit)."""
    toast = getattr(st, "toast", None)
    if callable(toast):
        toast(message)
    else:  # pragma: no cover - older Streamlit
        st.warning(message)


def _rerun(st: Any) -> None:
    """Rerun the script (supports both the new and legacy Streamlit API)."""
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


__all__ = ["render_conversation"]
