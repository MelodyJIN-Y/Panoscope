"""Conversation pane — the primary interface. The chat is the product.

``render_conversation(cluster)`` is the whole third pane of the shell. It is
**scoped per cluster**: each cluster owns its own thread, so switching clusters
swaps conversations and never bleeds one cluster's chat into another. On a fresh
cluster it posts the grounded opening interpretation
(``agent.loop.opening_interpretation``) as the first turn, then renders the
running thread inside a fixed-height scroll area, then a pinned ask box that
calls ``agent.loop.chat``.

The agent's *knowledge* of the whole annotation is NOT carried by this log — it
comes from the grounded global-interpretation block in the system prompt
(``agent.loop._cluster_context``). So a per-cluster thread costs the agent no
global awareness: it still knows every cluster's call and confidence, and durable
overrides persist across clusters through scope-enforced lab notes, not the chat.

Override-at-chat: the biologist just tells the agent ("this is CAF, our own
data"); the agent's ``memory_write`` tool persists a scope-enforced note and its
prose reports the agreement/dissent. This pane confirms the save inline (detected
by the notes-count delta) — never a bare "got it".

Grounding discipline: this module renders values the engine produced and never
computes one. It reads the agent's ``AgentResponse`` verbatim. Pinning a marker
the agent chose (``resp.pin_marker``) is a viewing control — one session key, no
recompute. Streamlit is imported lazily so importing ``ui.conversation`` needs no
running server.
"""

from __future__ import annotations

import html
import re
from typing import Any, Optional

from agent.types import AgentResponse, Citation, Source

from ui import data_access as dax
from ui import state

# --------------------------------------------------------------------------- #
# Thread message dicts (per cluster, via ui.state.append_message):
#   {"role": "agent"|"user"|"system", "text": str, "resp": Optional[AgentResponse]}
# The AgentResponse is frozen and picklable, safe to hold across reruns.
# --------------------------------------------------------------------------- #
_ROLE_AGENT = "agent"
_ROLE_USER = "user"
_ROLE_SYSTEM = "system"

# Source.kind -> display name for the condensed "Sources:" line under a turn.
_SRC_NAME: dict[str, str] = {
    "jz": "jazzPanda",
    "panel": "panel",
    "lit": "PubMed",
    "mem": "lab note",
}
# Fixed left-to-right order for the condensed line (grounded floor first).
_SRC_ORDER: tuple[str, ...] = ("jz", "panel", "lit", "mem")

# Session key for the ask input (namespaced so no other pane collides).
_K_ASK = "conv_ask_input"

# Confirm-card toggles: (display label, stored enum value). Scope's cluster label
# is filled with the active cluster id at render time.
_BASIS_OPTS: tuple[tuple[str, str], ...] = (
    ("our own data", "own_validation"),
    ("a paper", "paper"),
    ("convention", "convention"),
)
_STATUS_OPTS: tuple[tuple[str, str], ...] = (
    ("firm", "firm"),
    ("tentative", "tentative"),
)
_SCOPE_SAVED_LABEL: dict[str, str] = {
    "dataset": "this dataset",
    "lab": "lab-wide",
}
_BASIS_SAVED_LABEL: dict[str, str] = {
    "own_validation": "our own data",
    "paper": "a paper",
    "convention": "convention",
}

# Fixed-height scroll area for the thread; the ask box sits pinned just below it.
_THREAD_HEIGHT = 440

# Matches PMID:12345678 (case-insensitive, optional space) so citations in the
# agent's prose become clickable. Group 1 is the numeric id.
_PMID_RE = re.compile(r"PMID:\s*(\d+)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Scoped CSS for the new layout (header + scroll thread + speaker-aligned turns).
# Structural only — colours/tokens come from ui.theme.
# --------------------------------------------------------------------------- #
_CONVO_CSS = """
<style>
/* Header: title + active cluster + a light ghost 'Clear'. */
.conv-title {
  font-family: var(--mono); font-size: 11px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 500;
  display: flex; align-items: baseline; gap: 9px;
}
.conv-title .conv-sub {
  text-transform: none; letter-spacing: 0; color: var(--muted);
  font-size: 12.5px; font-weight: 600;
}
.st-key-conv_head { border-bottom: 1px solid var(--hair); padding-bottom: 7px; margin-bottom: 2px; }
.st-key-conv_head div[data-testid="stButton"] button {
  background: transparent !important; border: 1px solid var(--hair) !important;
  color: var(--faint) !important; box-shadow: none !important; min-height: 0 !important;
  padding: 3px 0 !important; border-radius: 7px !important; font-family: var(--mono) !important;
  font-size: 10px !important; letter-spacing: .08em; text-transform: uppercase;
}
.st-key-conv_head div[data-testid="stButton"] button:hover {
  color: var(--absent) !important; border-color: var(--absent) !important;
  background: var(--absent-bg) !important;
}
/* Thread scroll area: no box, a little breathing room for the scrollbar. */
.st-key-conv_thread { padding-right: 8px; }
/* Turn wrappers: agent to the left, you to the right — reads as a chat. */
.turn { display: flex; flex-direction: column; margin: 0 0 13px; }
.turn.a { align-items: flex-start; }
.turn.u { align-items: flex-end; }
.turn.u .bubble.u { max-width: 88%; }
.turn.a .bubble.a { max-width: 97%; }
.turn.sys .bubble.sys { width: 100%; }
/* Compact, ghosted citation buttons under an agent turn. */
.st-key-conv_thread div[data-testid="stButton"] button {
  min-height: 0 !important; padding: 3px 9px !important; border-radius: 7px !important;
  font-family: var(--mono) !important; font-size: 10px !important;
  background: var(--paper) !important; border: 1px solid var(--hair) !important;
  color: var(--accent) !important; box-shadow: none !important; font-weight: 500 !important;
}
.st-key-conv_thread div[data-testid="stButton"] button:hover {
  border-color: var(--accent) !important; background: var(--accent-soft) !important;
}
/* Ask row: input + a compact send, aligned at the bottom. */
.st-key-conv_ask [data-testid="stFormSubmitButton"] button {
  min-height: 38px !important; border-radius: 9px !important;
}
/* Confirm-to-save card: an accent-tinted panel between thread and ask box. */
.st-key-conv_draft {
  border: 1px solid var(--accent); border-radius: 12px;
  background: var(--accent-soft); padding: 12px 13px; margin: 4px 0 10px;
}
.draft-eyebrow {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--accent); font-weight: 600; margin-bottom: 6px;
}
.draft-claim { font-size: 13px; line-height: 1.5; color: var(--ink); font-weight: 500; margin-bottom: 8px; }
.draft-lbl {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .06em; color: var(--faint); margin: 6px 0 1px;
}
.draft-tension { font-family: var(--mono); font-size: 11px; color: var(--muted); margin: 10px 0 6px; }
.draft-tension .lbl { text-transform: uppercase; letter-spacing: .06em; color: var(--faint); margin-right: 6px; }
.draft-tension .agree { color: var(--accent); font-weight: 600; }
.draft-tension .dissent { color: var(--absent); font-weight: 600; }
.draft-tension .thin { color: var(--faint); }
.draft-tension a { color: var(--accent); text-decoration: none; }
.draft-tension a:hover { text-decoration: underline; }
.st-key-conv_draft [data-testid="stButton"] button { min-height: 34px !important; border-radius: 8px !important; }
</style>
"""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_conversation(cluster: str) -> None:
    """Render the conversation pane for ``cluster`` (header + thread + ask box).

    Per-cluster: reads/writes only ``cluster``'s thread. Posts the opening
    interpretation exactly once per cluster (guarded by ``opening_was_posted``).
    Never raises into the caller: the agent layer returns a grounded fallback.
    """
    import streamlit as st

    st.markdown(_CONVO_CSS, unsafe_allow_html=True)
    _ensure_opening(cluster)
    _render_header(cluster)

    with st.container(height=_THREAD_HEIGHT, border=False, key="conv_thread"):
        _render_thread(cluster)

    _render_draft_card(cluster)
    _render_ask_box(cluster)


def _render_header(cluster: str) -> None:
    """Title + active cluster/cell-type + a per-cluster 'Clear' control."""
    import streamlit as st

    verdict = _verdict_or_none(cluster)
    cell_type = html.escape(verdict.cell_type.replace("_", " ")) if verdict else ""
    with st.container(key="conv_head"):
        c_title, c_clear = st.columns([0.74, 0.26], vertical_alignment="center")
        with c_title:
            st.markdown(
                f'<div class="conv-title">Conversation'
                f'<span class="conv-sub">{html.escape(cluster)} · {cell_type}</span></div>',
                unsafe_allow_html=True,
            )
        with c_clear:
            st.button(
                "Clear",
                key=f"conv_clear_{cluster}",
                use_container_width=True,
                on_click=_clear_conversation,
                args=(cluster,),
                help="Clear this cluster's conversation and re-post its opening interpretation",
            )


def _clear_conversation(cluster: str) -> None:
    """on_click: wipe THIS cluster's thread and let the opening re-post next run."""
    state.clear_chat_thread(cluster)
    state.reset_opening_posted(cluster)


# --------------------------------------------------------------------------- #
# Opening interpretation — the first turn of a cluster's thread, once per cluster.
# --------------------------------------------------------------------------- #
def _ensure_opening(cluster: str) -> None:
    """Post the grounded opening interpretation the first time a cluster opens.

    Guarded by ``opening_was_posted`` so a rerun never double-posts. No auto-pin:
    a cluster opens with NO genes selected (marker selection is per-cluster and
    driven by the evidence table).
    """
    if state.opening_was_posted(cluster):
        return
    resp = _safe_opening(cluster)
    state.append_message(cluster, {"role": _ROLE_AGENT, "text": resp.text, "resp": resp})
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
    """Render each message in ``cluster``'s thread as a speaker-aligned bubble."""
    import streamlit as st

    thread = state.get_chat_thread(cluster)
    if not thread:
        st.markdown(
            '<div class="turn a"><div class="bubble a">Opening interpretation loads '
            "when a cluster is selected.</div></div>",
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
        '<div class="turn u"><div class="who">you</div>'
        f'<div class="bubble u">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_system_bubble(text: str) -> None:
    import streamlit as st

    st.markdown(
        f'<div class="turn sys"><div class="bubble sys">{html.escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def _render_agent_bubble(msg: dict, idx: int) -> None:
    """Render one agent turn: prose (inline citations) + verify + sources line.

    Inline ``PMID:xxx`` mentions get a dotted underline (they show WHERE the
    citation is); the actual affordance is the compact button row below, which
    opens the paper drawer (Streamlit can't bind a callback inside markdown HTML).
    """
    import streamlit as st

    resp: Optional[AgentResponse] = msg.get("resp")
    text = msg.get("text", "")

    prose_html, cited_pmids = _linkify_citations(text)
    sources_html = _sources_line_html(resp.sources if resp else ())
    verify_html = _verify_line_html(resp)
    st.markdown(
        f'<div class="turn a"><div class="who">agent</div>'
        f'<div class="bubble a">{prose_html}{verify_html}{sources_html}</div></div>',
        unsafe_allow_html=True,
    )
    citations = list(resp.citations) if resp else []
    _render_citation_buttons(citations, cited_pmids, idx)


def _verify_line_html(resp: Optional[AgentResponse]) -> str:
    """A small 're-check this' line when the turn is flagged verify."""
    if resp is None or not resp.verify:
        return ""
    return (
        '<div class="tension" style="border-left-color:var(--absent)">'
        "&#9873; re-check this: evidence is thin; confirm before relying on it."
        "</div>"
    )


def _sources_line_html(sources: tuple[Source, ...]) -> str:
    """ONE condensed provenance line for a turn (empty when no sources).

    Names only the source KINDS present (deduped), e.g.
    ``Sources jazzPanda · PubMed · lab note`` — the numbers stay inline in the
    prose and cited PMIDs stay clickable via the buttons below.
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
    of PMIDs found — the caller renders a clickable button per PMID.
    """
    pmids: list[str] = []
    parts: list[str] = []
    last = 0
    for m in _PMID_RE.finditer(text):
        parts.append(html.escape(text[last : m.start()]))
        pmid = m.group(1)
        if pmid not in pmids:
            pmids.append(pmid)
        parts.append(f'<span class="pcite">\U0001f4c4 PMID:{html.escape(pmid)}</span>')
        last = m.end()
    parts.append(html.escape(text[last:]))
    return "".join(parts), pmids


def _render_citation_buttons(
    citations: list[Citation], cited_pmids: list[str], idx: int
) -> None:
    """One compact button per citation that opens the paper drawer.

    Prefers the ``Citation`` objects (they carry title/authors); falls back to a
    bare button for any prose-only PMID so no cited paper is unreachable.
    """
    import streamlit as st

    by_pmid = {str(c.pmid): c for c in citations}
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
    token = first.split(" ")[0].strip()
    return token or first


# --------------------------------------------------------------------------- #
# Ask box — pinned just below the scroll area.
# --------------------------------------------------------------------------- #
def _render_ask_box(cluster: str) -> None:
    """Render the ask input + send. On submit, run one grounded chat turn.

    Appends the user turn to THIS cluster's thread, calls ``agent.loop.chat`` with
    only this cluster's history, appends the grounded answer, applies any pin the
    agent chose, and — if the turn wrote a lab note — confirms the save inline.
    """
    import streamlit as st

    with st.container(key="conv_ask"):
        with st.form(key=f"ask_form_{cluster}", clear_on_submit=True):
            c_in, c_send = st.columns([0.78, 0.22], vertical_alignment="bottom")
            with c_in:
                query = st.text_input(
                    "Ask",
                    key=_K_ASK,
                    placeholder="Ask, or override the call (e.g. 'this is CAF, our own data')…",
                    label_visibility="collapsed",
                )
            with c_send:
                asked = st.form_submit_button(
                    "Ask", use_container_width=True, type="primary"
                )

    if asked and query and query.strip():
        _submit_query(cluster, query.strip())
        _rerun(st)


def _submit_query(cluster: str, query: str) -> None:
    """Append the user turn, run the agent, append its grounded answer.

    A chat-driven override does NOT persist a note on the model's word: the agent
    proposes one (``resp.note_draft``) and we stash it so the confirm card renders.
    Nothing is written until the biologist saves — see ``_render_draft_card``.
    """
    state.append_message(cluster, {"role": _ROLE_USER, "text": query, "resp": None})

    resp = _safe_chat(cluster, query)
    state.append_message(cluster, {"role": _ROLE_AGENT, "text": resp.text, "resp": resp})

    # If the agent pinned a marker to back its answer, honour it for THIS cluster.
    if resp.pin_marker:
        state.set_pinned_marker(cluster, resp.pin_marker)

    # A proposed override note awaits the biologist's two-tap confirm.
    if resp.note_draft is not None:
        state.set_pending_draft(cluster, resp.note_draft)


# --------------------------------------------------------------------------- #
# Confirm-to-save card — the biologist's two taps (scope + basis), then Save.
# Nothing is persisted until Save; Discard drops the draft. Sits between the
# thread and the ask box so a pending decision is always in view.
# --------------------------------------------------------------------------- #
def _render_draft_card(cluster: str) -> None:
    """Render the pending note draft (if any) as a confirm card with toggles."""
    import dataclasses

    import streamlit as st

    draft = state.get_pending_draft(cluster)
    if draft is None:
        return

    nonce = _draft_nonce(draft)
    scope_opts = (
        (f"this cluster ({cluster})", "cluster"),
        ("this dataset", "dataset"),
        ("lab-wide", "lab"),
    )
    with st.container(key="conv_draft"):
        st.markdown(
            '<div class="draft-eyebrow">Draft to save · confirm scope &amp; basis</div>'
            f'<div class="draft-claim">{html.escape(draft.claim)}</div>',
            unsafe_allow_html=True,
        )
        scope = _seg(st, "scope", scope_opts, draft.scope, f"dscope_{cluster}_{nonce}")
        basis = _seg(st, "basis", _BASIS_OPTS, draft.basis, f"dbasis_{cluster}_{nonce}")
        status = _seg(st, "status", _STATUS_OPTS, draft.status, f"dstatus_{cluster}_{nonce}")
        st.markdown(_draft_tension_html(draft), unsafe_allow_html=True)
        c_save, c_disc = st.columns([0.6, 0.4])
        save = c_save.button(
            "Save note", key=f"dsave_{cluster}_{nonce}", type="primary", use_container_width=True
        )
        disc = c_disc.button("Discard", key=f"ddisc_{cluster}_{nonce}", use_container_width=True)

    if disc:
        state.clear_pending_draft(cluster)
        state.append_message(
            cluster,
            {"role": _ROLE_SYSTEM, "text": "Draft discarded — nothing was saved.", "resp": None},
        )
        _rerun(st)
    elif save:
        edited = dataclasses.replace(
            draft,
            scope=scope,
            basis=basis,
            status=status,
            cluster=cluster if scope == "cluster" else None,
        )
        _save_pending_draft(cluster, edited)
        _rerun(st)


def _seg(st: Any, label: str, opts, default_value: str, key: str) -> str:
    """A segmented-control (or radio fallback) over (label, value) options.

    Seeds from ``default_value`` (the agent's proposal); the widget key carries the
    draft nonce so a NEW draft starts fresh. Returns the chosen enum value.
    """
    labels = [lab for lab, _ in opts]
    val_by_label = {lab: val for lab, val in opts}
    label_by_val = {val: lab for lab, val in opts}
    default_label = label_by_val.get(default_value, labels[0])
    st.markdown(f'<div class="draft-lbl">{html.escape(label)}</div>', unsafe_allow_html=True)
    seg = getattr(st, "segmented_control", None)
    if callable(seg):
        picked = seg(
            label, labels, default=default_label, key=key, label_visibility="collapsed"
        )
    else:  # pragma: no cover - older Streamlit without segmented_control
        picked = st.radio(
            label,
            labels,
            index=labels.index(default_label),
            horizontal=True,
            key=key,
            label_visibility="collapsed",
        )
    return val_by_label.get(picked, default_value)


def _draft_nonce(draft: Any) -> int:
    """A per-draft key suffix so a new draft's toggles start from its proposal
    (stable across reruns within a session; derived from the claim)."""
    return abs(hash(draft.claim)) % 100000


def _draft_tension_html(draft: Any) -> str:
    """The live literature-check line for the draft (agree/dissent PMIDs or thin)."""
    t = draft.tension
    if t.agree or t.dissent:
        parts: list[str] = []
        if t.agree:
            parts.append(f'<span class="agree">{len(t.agree)} agree</span> {_pmid_links(t.agree)}')
        if t.dissent:
            parts.append(
                f'<span class="dissent">{len(t.dissent)} dissent</span> {_pmid_links(t.dissent)}'
            )
        body = " · ".join(parts)
    else:
        body = '<span class="thin">literature thin — no clear paper either way</span>'
    return f'<div class="draft-tension"><span class="lbl">literature check</span>{body}</div>'


def _pmid_links(cites) -> str:
    """Clickable external PubMed links for a tuple of citations (real PMIDs)."""
    return " ".join(
        f'<a href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(str(c.pmid))}/" '
        f'target="_blank">PMID:{html.escape(str(c.pmid))}</a>'
        for c in cites
        if c.pmid
    )


def _save_pending_draft(cluster: str, edited: Any) -> None:
    """Persist the confirmed draft and post an inline saved confirmation."""
    try:
        note = dax.save_note_draft(edited)
    except Exception:  # noqa: BLE001 - surface a clean message, never crash the pane
        state.clear_pending_draft(cluster)
        state.append_message(
            cluster,
            {"role": _ROLE_SYSTEM, "text": "Could not save the note — please try again.", "resp": None},
        )
        return
    state.clear_pending_draft(cluster)
    state.append_message(cluster, {"role": _ROLE_SYSTEM, "text": _saved_line(note), "resp": None})


def _saved_line(note: Any) -> str:
    """The 'Saved to Lab knowledge · scope · basis · status' confirmation line."""
    if note.scope == "cluster" and note.scope_ref.cluster:
        scope_txt = f"cluster {note.scope_ref.cluster}"
    else:
        scope_txt = _SCOPE_SAVED_LABEL.get(note.scope, note.scope)
    basis_txt = _BASIS_SAVED_LABEL.get(note.basis, note.basis)
    t = note.tension
    if t.agree or t.dissent:
        tension = f"{len(t.agree)} agree / {len(t.dissent)} dissent kept visible"
    else:
        tension = "literature thin, recorded as-is"
    return (
        f"Saved to Lab knowledge · {scope_txt} · {basis_txt} · {note.status}. "
        f"Literature: {tension}. It is cited whenever it applies."
    )


def _safe_chat(cluster: str, query: str) -> AgentResponse:
    """Run one chat turn with THIS cluster's history; never raise into render."""
    try:
        history = _history_for_agent(cluster)
        return dax.get_agent().chat(query, cluster=cluster, history=history)
    except Exception:  # noqa: BLE001 - agent has its own fallback; belt + braces
        from agent import loop as agent_loop

        return agent_loop.chat(query, cluster=cluster)


def _history_for_agent(cluster: str) -> list[dict]:
    """Flatten ONLY ``cluster``'s thread into role/text history for the agent.

    Cross-cluster history is deliberately excluded: the agent's awareness of the
    other clusters comes from the grounded global-interpretation context in the
    system prompt, not from another cluster's chat log.
    """
    out: list[dict] = []
    for msg in state.get_chat_thread(cluster):
        role = msg.get("role")
        if role == _ROLE_USER:
            out.append({"role": "user", "content": msg.get("text", "")})
        elif role == _ROLE_AGENT:
            out.append({"role": "assistant", "content": msg.get("text", "")})
    return out


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _verdict_or_none(cluster: str) -> Optional[Any]:
    """Cached verdict for a cluster, or None if unavailable (never raises)."""
    try:
        return dax.verdict_for(cluster)
    except Exception:  # noqa: BLE001
        return None


def _rerun(st: Any) -> None:
    """Rerun the script (supports both the new and legacy Streamlit API)."""
    rerun = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun):
        rerun()


__all__ = ["render_conversation"]
