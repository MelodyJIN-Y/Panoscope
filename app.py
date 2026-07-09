"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint. The chat is the product; the panels are the evidence it
stands on. Every number shown here comes from the agent layer (jazzPanda output,
the panel list, or a cited lab note) — this file only wires panels together.

Two pages, sharing session state natively via ``st.navigation``:

* **Examine cluster** (default): the 3-pane review surface —
  ``cluster rail | verdict + evidence + spatial stage | conversation``.
* **Summary**: the whole annotation set — table + holistic pass + CSV download.

``_shared_init`` runs once per script run at the top of each page (theme + state
+ header/brand). It is idempotent: ``st.set_page_config`` and ``inject_css`` are
called once each, and ``state.init_state`` is itself idempotent.
"""
from __future__ import annotations

import streamlit as st

from ui import (
    cluster_rail,
    conversation,
    evidence_table,
    lab_knowledge,
    paper_drawer,
    spatial_stage,
    state,
    summary,
    theme,
    verdict_header,
)

# Guard so ``st.set_page_config`` (must be the first Streamlit call, once only)
# and the header run exactly once per script run even though every page calls
# ``_shared_init`` at its top.
_K_SHARED_INIT = "_shared_init_done"


def _shared_init() -> None:
    """Idempotent per-run setup: page config, theme, state, and the header/brand.

    Runs at the top of every page. Guarded by a session flag that is reset each
    script run (see below), so config/CSS/header render exactly once per run —
    never twice, even though both page functions call it.
    """
    if st.session_state.get(_K_SHARED_INIT):
        return

    st.set_page_config(page_title="Panoscope", page_icon="🔬", layout="wide")
    theme.inject_css()
    state.init_state()

    # ── Header ────────────────────────────────────────────────────────────
    head_left, head_right = st.columns([0.78, 0.22])
    with head_left:
        st.markdown(
            '<div class="pano-brand">Pano<span class="d">·</span>scope</div>'
            '<div class="pano-ctx">Xenium breast · 280 genes · 9 clusters · sample 1</div>',
            unsafe_allow_html=True,
        )
    with head_right:
        lab_knowledge.lab_knowledge_button()

    st.session_state[_K_SHARED_INIT] = True


def examine_page() -> None:
    """The 3-pane review surface: rail | verdict+evidence+spatial | chat.

    Lab-knowledge and citation-paper drawers live here (they hang off the
    examine surface's session flags). The Summary page owns the holistic pass.
    """
    _shared_init()

    # ── Drawers (render only when opened via session state) ───────────────
    if state.is_lab_knowledge_open():
        lab_knowledge.render_lab_panel(expanded=True)
    if state.is_paper_open():
        paper_drawer.render_paper_drawer()

    # ── 3-pane shell ──────────────────────────────────────────────────────
    rail_col, center_col, chat_col = st.columns([222, 760, 372], gap="small")

    with rail_col:
        cluster_rail.render_rail()

    cluster = state.get_selected_cluster()

    with center_col:
        verdict_header.render_verdict(cluster)
        evidence_table.render_evidence_table(cluster)
        spatial_stage.render_spatial_stage(cluster)

    with chat_col:
        conversation.render_conversation(cluster)


def summary_page() -> None:
    """The Summary page: annotation table + holistic pass + CSV download."""
    _shared_init()
    summary.render_summary_page()


# ── Multipage nav — state is shared natively across pages ──────────────────
# ``_shared_init`` is guarded by a session flag; reset it at the top of every
# script run so config/CSS/header render once per run (not once per session).
st.session_state[_K_SHARED_INIT] = False

pg = st.navigation(
    [
        st.Page(examine_page, title="Examine cluster", icon="🔬", default=True),
        st.Page(summary_page, title="Summary", icon="📋"),
    ]
)
pg.run()
