"""Pathway conversation pane — the Pathways page's third column.

The enrichment analog of ``ui.conversation``: a per-cluster chat whose target is
that cluster's enriched PROGRAMS, grounded by live literature. It mirrors the
marker conversation's design and flow (header + opening + scroll thread + pinned
ask box) and reuses its chrome, but:

* the thread is namespaced ``pw::{cluster}`` so it never mixes with the marker
  chat for the same cluster;
* the agent runs under the ``geneset-enrichment`` skill with the cluster's
  enrichment records injected (``agent.loop._enrichment_context``), so answers are
  grounded in the real programs + scores + leading-edge genes;
* the opening is a deterministic, cited summary built from the enrichment records
  and the persisted per-pathway notes (no live call to open a cluster).

Grounding: this pane renders the agent's ``AgentResponse`` verbatim (inline
clickable PMIDs, a condensed Sources line) and computes nothing. Streamlit is
imported lazily so importing the module needs no server.
"""

from __future__ import annotations

import html
from typing import Any, Optional

from ui import conversation as convo
from ui import data_access as da
from ui import state

_SKILL = "geneset-enrichment"
_K_ASK = "pwconv_ask_input"


def _thread_key(cluster: str) -> str:
    return f"pw::{cluster}"


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _ce(cluster: str) -> Optional[Any]:
    try:
        return da.enrichment_for(cluster)
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_pathway_conversation(cluster: str) -> None:
    """Render the pathway conversation for ``cluster`` (header + thread + ask)."""
    import streamlit as st

    st.markdown(convo._CONVO_CSS, unsafe_allow_html=True)
    _ensure_opening(cluster)
    _render_header(cluster)
    st.markdown(
        '<div class="conv-hint">Ask about this cluster\'s enriched programs — what they '
        "mean, whether they fit the cell type, or which look cross-lineage. Reinterpret a "
        "program (e.g. co-infiltration, not the cluster's own) and I draft a note to save. "
        "Answers cite live literature.</div>",
        unsafe_allow_html=True,
    )
    with st.container(key="conv_thread"):
        _render_thread(cluster)
    # Same two-tap confirm card as the marker chat, on this cluster's pathway thread —
    # so a program_reinterpretation captured here never collides with the marker draft.
    convo._render_draft_card(cluster, thread_key=_thread_key(cluster))
    _render_ask_box(cluster)


def _render_header(cluster: str) -> None:
    import streamlit as st

    ce = _ce(cluster)
    cell_type = html.escape(ce.cell_type.replace("_", " ")) if ce else ""
    with st.container(key="conv_head"):
        c_title, c_clear = st.columns([0.72, 0.28], vertical_alignment="center")
        with c_title:
            st.markdown(
                f'<div class="conv-title">Pathway chat'
                f'<span class="conv-sub">{html.escape(cluster)} · {cell_type}</span></div>',
                unsafe_allow_html=True,
            )
        with c_clear:
            st.button(
                "Clear",
                key=f"pwconv_clear_{cluster}",
                use_container_width=True,
                on_click=_clear,
                args=(cluster,),
                help="Clear this cluster's pathway chat and re-post its opening summary",
            )


def _clear(cluster: str) -> None:
    key = _thread_key(cluster)
    state.clear_chat_thread(key)
    state.reset_opening_posted(key)


# --------------------------------------------------------------------------- #
# Opening — a deterministic, cited summary of the cluster's enriched programs.
# --------------------------------------------------------------------------- #
def _ensure_opening(cluster: str) -> None:
    key = _thread_key(cluster)
    if state.opening_was_posted(key):
        return
    text, pmids, verify = _opening_text(cluster)
    state.append_message(key, {"role": "agent", "text": text, "resp": _opening_resp(pmids, verify)})
    state.mark_opening_posted(key)


def _opening_text(cluster: str) -> tuple[str, list[str], bool]:
    """The deterministic, cited opening PROSE plus the PMIDs it cites and the
    enrichment verify flag. Returned so the bubble renders the same clickable
    citations + 'Sources' line + re-check note as the marker opening (parity) — with
    NO live call (the PMIDs are the precomputed real ones from pathway_notes)."""
    ce = _ce(cluster)
    ct = ce.cell_type.replace("_", " ") if ce else cluster
    verify = bool(getattr(ce, "verify", False))
    if ce is None or (not ce.enriched and not ce.suggestive):
        return (
            f"No gene-set program clears the enrichment gate for {cluster} {ct}. "
            "Ask me why, or about any suggestive program.", [], verify,
        )
    parts = [
        f"{cluster} {ct} shows {ce.confidence} enrichment, panel-scoped: only the set genes "
        "on the 280-gene panel are measured, never genome-wide."
    ]
    progs: list[str] = []
    pmids: list[str] = []
    for p in ce.enriched[:3]:
        note = da.pathway_note(cluster, p.gene_set) or {}
        summ = str(note.get("summary") or "").strip()
        pmid = note.get("pmid")
        if pmid and str(pmid).strip().isdigit():
            pmids.append(str(pmid).strip())
        cite = f" PMID:{pmid}" if pmid else ""
        if summ:
            progs.append(f"{_short(p.gene_set)}: {summ}{cite}")
        else:
            progs.append(f"{_short(p.gene_set)} (leading edge {', '.join(p.leading_edge[:4])})")
    if progs:
        parts.append("Top programs — " + "  ".join(progs))
    parts.append(
        f"Ask me what any program means for a {ct} cluster, whether it fits the call, or "
        "which look like co-infiltration rather than this cell type's own program."
    )
    return "\n\n".join(parts), pmids, verify


def _opening_resp(pmids: list[str], verify: bool):
    """A lightweight cited response for the opening bubble so it renders like the
    marker opening: sources = jazzPanda (the enrichment scores) + PubMed (the cited
    notes). No live call — the PMIDs are the precomputed, real ones from pathway_notes."""
    from agent.types import AgentResponse, GroundingSidecar, Source

    seen = list(dict.fromkeys(pmids))
    sources = [Source(kind="jz", ref="enrichment", value="jazzPanda enrichment")]
    sources += [Source(kind="lit", ref=pm, value=None) for pm in seen]
    grounding = GroundingSidecar(numbers=(), markers=(), pmids=tuple(seen), notes_used=())
    return AgentResponse(text="", sources=tuple(sources), verify=verify, grounding=grounding, opening=True)


# --------------------------------------------------------------------------- #
# Thread + ask box (reuse the marker conversation's bubble chrome)
# --------------------------------------------------------------------------- #
def _render_thread(cluster: str) -> None:
    import streamlit as st

    thread = state.get_chat_thread(_thread_key(cluster))
    if not thread:
        st.markdown(
            '<div class="turn a"><div class="bubble a">Enrichment summary loads when a '
            "cluster is selected.</div></div>",
            unsafe_allow_html=True,
        )
        return
    for msg in thread:
        role = msg.get("role", "agent")
        if role == "user":
            convo._render_user_bubble(msg.get("text", ""))
        elif role == "system":
            convo._render_system_bubble(msg.get("text", ""))
        else:
            convo._render_agent_bubble(msg)


def _render_ask_box(cluster: str) -> None:
    import streamlit as st

    with st.container(key="conv_ask"):
        with st.form(key=f"pwask_form_{cluster}", clear_on_submit=True):
            c_in, c_send = st.columns([0.78, 0.22], vertical_alignment="bottom")
            with c_in:
                query = st.text_input(
                    "Ask",
                    key=_K_ASK,
                    placeholder="Ask about these pathways…",
                    label_visibility="collapsed",
                )
            with c_send:
                asked = st.form_submit_button("Ask", use_container_width=True, type="primary")

    if asked and query and query.strip():
        _submit(cluster, query.strip())
        convo._rerun(st)


_FALLBACK_TEXT = (
    "I couldn't ground a confident, literature-backed answer for that just now. Ask about a "
    "specific enriched program above (what it means for this cell type, whether it fits, or "
    "whether it looks like co-infiltration) and I'll cite the literature."
)


def _submit(cluster: str, query: str) -> None:
    key = _thread_key(cluster)
    state.append_message(key, {"role": "user", "text": query, "resp": None})
    resp = _safe_chat(cluster, query)
    # The shared agent's fallback is marker-flavored (a cell-type verdict); for a
    # pathway question that is off-topic, so substitute an honest enrichment reply.
    text = _FALLBACK_TEXT if getattr(resp, "used_fallback", False) else resp.text
    keep = None if getattr(resp, "used_fallback", False) else resp
    state.append_message(key, {"role": "agent", "text": text, "resp": keep})
    # If the agent proposed a note (e.g. a program_reinterpretation), stash it under
    # this pathway thread so the confirm card renders; nothing saves until confirmed.
    if keep is not None and getattr(resp, "note_draft", None) is not None:
        state.set_pending_draft(key, resp.note_draft)


def _safe_chat(cluster: str, query: str):
    history = _history(cluster)
    try:
        return da.get_agent().chat(query, cluster=cluster, history=history, skill=_SKILL)
    except Exception:  # noqa: BLE001 - the agent has its own fallback; belt + braces
        from agent import loop as agent_loop

        return agent_loop.chat(query, cluster=cluster, skill=_SKILL)


def _history(cluster: str) -> list[dict]:
    out: list[dict] = []
    for msg in state.get_chat_thread(_thread_key(cluster)):
        role = msg.get("role")
        if role == "user":
            out.append({"role": "user", "content": msg.get("text", "")})
        elif role == "agent":
            out.append({"role": "assistant", "content": msg.get("text", "")})
    return out


__all__ = ["render_pathway_conversation"]
