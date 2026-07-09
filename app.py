"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint. The chat is the product; the panels are the evidence it
stands on. Every number shown here comes from the agent layer (jazzPanda output,
the panel list, or a cited lab note) — this file only wires panels together.

Navigation lives in a single TOP bar (not a sidebar): the brand and three tabs —
``Examine cluster | Summary | Lab knowledge`` — in one row. Session state is
shared across pages natively (one script run per rerun), so the selected cluster
/ markers / chat / notes carry over between tabs.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

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

# Brand assets (the Panoscope logo mark + the with-text logo live in assets/).
_ASSETS = Path(__file__).resolve().parent / "assets"
_LOGO_MARK = _ASSETS / "panoscope_logo.png"


@lru_cache(maxsize=2)
def _logo_data_uri(name: str) -> str:
    """Base64 data URI for a brand PNG so it embeds inline in the header HTML
    (Streamlit's strict CSP blocks external asset requests). '' if missing."""
    path = _ASSETS / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# Active top-tab page. A plain session-state key (nav state, not domain state).
_K_PAGE = "active_page"
_PAGE_EXAMINE = "examine"
_PAGE_SUMMARY = "summary"
_PAGE_LAB = "lab"


def _set_page(page: str) -> None:
    """on_click handler for a top tab — fires before the rerun renders."""
    st.session_state[_K_PAGE] = page


def _top_bar(page: str) -> None:
    """The single top bar: brand · Examine / Summary / Lab knowledge tabs.

    The tabs are chromeless buttons styled (theme ``.st-key-pano_topnav``) into a
    top tab strip — the active one (``type="primary"``) gets an accent underline.
    """
    with st.container(key="pano_appbar"):
        brand_col, tabs_col, ctx_col = st.columns(
            [0.27, 0.46, 0.27], vertical_alignment="center"
        )
        with brand_col:
            logo = _logo_data_uri("panoscope_logo.png")
            mark = (
                f'<img class="pano-logo" src="{logo}" alt="Panoscope logo"/>'
                if logo
                else '<span class="pano-mark"></span>'
            )
            st.markdown(
                f'<div class="pano-brand">{mark}Panoscope</div>',
                unsafe_allow_html=True,
            )
        with tabs_col:
            with st.container(key="pano_topnav"):
                # Order: Summary · Examine cluster · Lab knowledge.
                t_summary, t_examine, t_lab = st.columns(3)
                with t_summary:
                    st.button(
                        "Summary",
                        key="nav_summary",
                        type="primary" if page == _PAGE_SUMMARY else "secondary",
                        use_container_width=True,
                        on_click=_set_page,
                        args=(_PAGE_SUMMARY,),
                    )
                with t_examine:
                    st.button(
                        "Examine cluster",
                        key="nav_examine",
                        type="primary" if page == _PAGE_EXAMINE else "secondary",
                        use_container_width=True,
                        on_click=_set_page,
                        args=(_PAGE_EXAMINE,),
                    )
                with t_lab:
                    st.button(
                        "Lab knowledge",
                        key="nav_lab",
                        type="primary" if page == _PAGE_LAB else "secondary",
                        use_container_width=True,
                        on_click=_set_page,
                        args=(_PAGE_LAB,),
                    )
        with ctx_col:
            st.markdown(
                '<div class="pano-ctx-wrap"><div class="pano-ctx-chip">'
                '<div class="pano-ctx-text">'
                '<span class="pano-ctx-main">Xenium human breast · sample 1</span>'
                '<span class="pano-ctx-sub">280 genes · 9 clusters · jazzPanda markers</span>'
                "</div></div></div>",
                unsafe_allow_html=True,
            )


def _examine_body() -> None:
    """The 3-pane review surface: rail | verdict + evidence + spatial | chat."""
    if state.is_paper_open():
        paper_drawer.render_paper_drawer()
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


# ── One script run per rerun (no st.navigation / sidebar) ──────────────────
st.set_page_config(
    page_title="Panoscope",
    page_icon=str(_LOGO_MARK) if _LOGO_MARK.exists() else "🔬",
    layout="wide",
)
theme.inject_css()
state.init_state()

# on_click tab handlers have already fired, so this reads the fresh page.
page = st.session_state.get(_K_PAGE, _PAGE_EXAMINE)
_top_bar(page)

if page == _PAGE_SUMMARY:
    summary.render_summary_page()
elif page == _PAGE_LAB:
    lab_knowledge.render_lab_page()
else:
    _examine_body()
