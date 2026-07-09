"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint. The chat is the product; the panels are the evidence it
stands on. Every number shown here comes from the agent layer (jazzPanda output,
the panel list, or a cited lab note) — this file only wires panels together.

Layout (3-pane, matching dashboard_wireframe_panels.html):
    cluster rail  |  verdict + evidence + spatial stage  |  conversation
"""
from __future__ import annotations

import streamlit as st

from ui import (
    cluster_rail,
    conversation,
    evidence_table,
    holistic,
    lab_knowledge,
    paper_drawer,
    spatial_stage,
    state,
    theme,
    verdict_header,
)

# Direct session_state key for the holistic ("Review all clusters") panel toggle.
K_HOLISTIC_OPEN = "holistic_open"

st.set_page_config(page_title="Panoscope", page_icon="🔬", layout="wide")
theme.inject_css()
state.init_state()

# ── Header ──────────────────────────────────────────────────────────────────
head_left, head_mid, head_right = st.columns([0.60, 0.20, 0.20])
with head_left:
    st.markdown(
        '<div class="pano-brand">Pano<span class="d">·</span>scope</div>'
        '<div class="pano-ctx">Xenium breast · 280 genes · 9 clusters · sample 1</div>',
        unsafe_allow_html=True,
    )
with head_mid:
    # "Review all clusters" — toggles the holistic cross-cluster panel (Step 4).
    if st.button("Review all clusters", key="holistic_toggle_btn"):
        st.session_state[K_HOLISTIC_OPEN] = not st.session_state.get(K_HOLISTIC_OPEN, False)
        st.rerun()
with head_right:
    lab_knowledge.lab_knowledge_button()

# ── Drawers / panels (render only when opened via session state) ────────────
if st.session_state.get(K_HOLISTIC_OPEN, False):
    with st.expander("Holistic review — all 9 clusters together", expanded=True):
        holistic.render_holistic_review()
if state.is_lab_knowledge_open():
    lab_knowledge.render_lab_panel(expanded=True)
if state.is_paper_open():
    paper_drawer.render_paper_drawer()

# ── 3-pane shell ────────────────────────────────────────────────────────────
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
