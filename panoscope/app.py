"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint; this file only wires panels together. Every number shown
comes from the agent layer. The top-bar tabs are built by ``ui/nav.py`` from the
workflow registry; this file supplies only :data:`_SURFACES` (which renderer draws a
surface id, and whether this study has a result for it).
"""
from __future__ import annotations

import html

import streamlit as st

from panoscope.ui import (
    cluster_rail,
    conversation,
    data_access as da,
    enrichment_table,
    evidence_table,
    nav,
    onboarding,
    review,
    spatial_stage,
    state,
    summary,
    theme,
)

_LOGO_MARK = nav.ASSETS / "panoscope_logo.png"


# Active page: session state, mirrored to ``?page=`` so a refresh restores it.
_K_PAGE = nav.PAGE_KEY
# The onboarding pages are not workflows, so they are named here.
_PAGE_WELCOME = nav.ONBOARDING_PAGES[0]   # landing
_PAGE_UPLOAD = nav.ONBOARDING_PAGES[1]    # load-your-data


def _resolve_page(strip: tuple[nav.Tab, ...]) -> str:
    """The active tab, restored from ``?page=`` on a fresh session. An unknown name
    falls back to the landing page (fail-closed)."""
    valid = nav.pages(strip)
    if _K_PAGE not in st.session_state:
        url_page = st.query_params.get("page")
        if url_page in valid:
            default = url_page
        elif any(k in st.query_params for k in ("sign", "undo", "drill", "confirm", "accept")):
            # A Summary-table action link (?drill=cN etc.) drops the page param; the
            # user is on the Summary dashboard, not making a fresh visit -> not welcome.
            default = nav.SUMMARY_PAGE
        else:
            default = _PAGE_WELCOME
        st.session_state[_K_PAGE] = default
    page = st.session_state[_K_PAGE]
    if page not in valid:
        page = st.session_state[_K_PAGE] = _PAGE_WELCOME
    if st.query_params.get("page") != page:
        st.query_params["page"] = page
    return page


def _restore_selection_from_url() -> None:
    """Once per session, restore the selected cluster and its markers + gene sets
    from the URL."""
    if st.session_state.get("_sel_restored"):
        return
    st.session_state["_sel_restored"] = True
    qp = st.query_params
    c = qp.get("cluster")
    if c:
        state.set_selected_cluster(c)  # ignores an unknown id (fail-closed)
    cluster = state.get_selected_cluster()
    markers = qp.get("m")
    if markers:
        state.set_selected_markers(cluster, [g for g in markers.split(",") if g])
    pathways = qp.get("pw")
    if pathways:
        state.set_selected_pathways(cluster, [s for s in pathways.split(",") if s])


def _sync_selection_to_url(page: str) -> None:
    """Mirror the current cluster + its selections to the URL (only on change). The
    Summary page carries no cluster/marker params."""
    qp = st.query_params
    if page == nav.SUMMARY_PAGE:
        for key in ("cluster", "m", "pw"):
            if key in qp:
                del qp[key]
        return
    cluster = state.get_selected_cluster()
    if qp.get("cluster") != cluster:
        qp["cluster"] = cluster
    for key, values in (("m", state.get_selected_markers(cluster)),
                        ("pw", state.get_selected_pathways(cluster))):
        joined = ",".join(values)
        if joined:
            if qp.get(key) != joined:
                qp[key] = joined
        elif key in qp:
            del qp[key]


def _study_chip() -> tuple[str, str]:
    """The top-bar context chip, derived from the open study (panel size from the
    file, never a constant). A field that cannot be read is omitted, never guessed."""
    from panoscope import methods as _methods
    from panoscope.agent import config as _cfg
    from panoscope.workflows.marker import WORKFLOW_ID as _MARKER_ID

    study = _cfg.current_study()
    bits = [b for b in (study.platform, study.tissue) if b]
    samples = study.manifest.get("samples") or []
    if len(samples) > 1:
        bits.append(f"{len(samples)} samples")
    elif len(samples) == 1:
        bits.append(str(samples[0]))

    try:
        panel_n = len(da.panel_names())
    except Exception:  # noqa: BLE001 - a study with no panel file still gets a chip
        panel_n = 0
    facts = []
    if panel_n:
        facts.append(f"{panel_n} genes")
    facts.append(f"{len(study.units)} clusters")
    reg = _methods.for_workflow(_MARKER_ID)
    if reg:
        facts.append(f"{getattr(reg[0], 'display_name', reg[0].name)} markers")
    from panoscope.corpus.build import header as _corpus_header

    facts.append(_corpus_header(study.id))
    return " · ".join(bits), " · ".join(facts)


def _top_bar(page: str, strip: tuple[nav.Tab, ...]) -> None:
    """The single top bar: brand · one tab per registered workflow · study chip."""
    with st.container(key="pano_appbar"):
        brand_col, tabs_col, ctx_col = st.columns(
            [0.27, 0.46, 0.27], vertical_alignment="center"
        )
        with brand_col:
            logo = nav.logo_data_uri("panoscope_logo.png")
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
                for col, tab in zip(st.columns(len(strip)), strip):
                    with col:
                        # A workflow with no result keeps its tab and says so on hover.
                        st.button(
                            tab.title,
                            key=tab.key,
                            help=None if tab.available()
                            else f"No {tab.title.lower()} result for this study",
                            type="primary" if page == tab.page else "secondary",
                            width="stretch",
                            on_click=nav.set_page,
                            args=(tab.page,),
                        )
        with ctx_col:
            main, sub = _study_chip()
            st.markdown(
                '<div class="pano-ctx-wrap"><div class="pano-ctx-chip">'
                '<div class="pano-ctx-text">'
                f'<span class="pano-ctx-main">{html.escape(main)}</span>'
                f'<span class="pano-ctx-sub">{html.escape(sub)}</span>'
                "</div></div></div>",
                unsafe_allow_html=True,
            )


@st.fragment
def _chat_pane_fragment(cluster: str) -> None:
    """The marker chat as a fragment, so a chat turn re-renders only this pane."""
    state.apply_session_study()  # a fragment rerun skips the top of this file
    conversation.render_conversation(cluster)


def _kill_cross_page_bleed() -> None:
    """Hide the previous page's stale leftover DOM after a tab switch.

    A parent-document watcher adds ``pano-killbleed`` to <html> when ``?page=``
    changes; the CSS hides ``data-stale`` elements inside a page body until none
    remain (4s hard cap).
    """
    import streamlit.components.v1 as components

    st.markdown(
        "<style>"
        '.pano-killbleed [class*="st-key-page_body_"] [data-stale="true"]'
        "{display:none !important;}"
        "</style>",
        unsafe_allow_html=True,
    )
    # Not migrated to st.iframe: it takes a src URL, and a data: URL would be
    # cross-origin and blocked from window.parent.document.
    components.html(
        """
<script>
(function() {
  var win = window.parent, doc = win && win.document;
  if (!doc) return;
  function curPage() {
    var m = (win.location.search || "").match(/[?&]page=([^&]+)/);
    return m ? decodeURIComponent(m[1]) : null;
  }
  function bodyHasStale() {
    var body = doc.querySelector('[class*="st-key-page_body_"]');
    return !!(body && body.querySelector('[data-stale="true"]'));
  }
  if (win.__panoBleed) return;  // watcher already installed
  win.__panoBleed = { last: curPage(), at: 0 };
  var root = doc.documentElement;
  var obs = new MutationObserver(function() {
    var p = curPage();
    if (p && p !== win.__panoBleed.last) {         // page just changed -> start hiding leftovers
      win.__panoBleed.last = p;
      win.__panoBleed.at = Date.now();
      root.classList.add("pano-killbleed");
    }
    if (root.classList.contains("pano-killbleed")) {
      // lift once the leftover is physically gone, or after a 4s hard cap (never hide forever)
      if (!bodyHasStale() || Date.now() - win.__panoBleed.at > 4000) {
        root.classList.remove("pano-killbleed");
      }
    }
  });
  obs.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["data-stale"] });
})();
</script>
""",
        height=0,
    )


def _examine_body() -> None:
    """The 3-pane review surface: rail | verdict + evidence + spatial | chat."""
    rail_col, center_col, chat_col = st.columns([216, 664, 474], gap="small")
    with rail_col:
        cluster_rail.render_rail()
    cluster = state.get_selected_cluster()
    with center_col:
        # Evidence first; the proposal's header is one tap away (ui/review.py).
        evidence_table.render_evidence_table(cluster)
        review.render_review(cluster)
        spatial_stage.render_spatial_stage(cluster)
    with chat_col:
        _chat_pane_fragment(cluster)


def _markers_available() -> bool:
    """Does this study have a marker result to draw?"""
    return not da.markers_df().empty


# ``UISpec.workflow_id`` -> (renderer, has-a-result check). A registered workflow
# with no entry keeps its tab and renders nav's stated-absence page.
_SURFACES: nav.Wiring = {
    "markers": (_examine_body, _markers_available),
    "pathways": (enrichment_table.render_pathways_page, da.enrichment_available),
}


# ── One script run per rerun (no st.navigation / sidebar) ──────────────────
st.set_page_config(
    page_title="Panoscope",
    page_icon=str(_LOGO_MARK) if _LOGO_MARK.exists() else "🔬",
    layout="wide",
)
state.apply_session_study()  # before anything reads the study
theme.inject_css()
state.init_state()

# Page routing: onboarding, summary, then one ?page= name per registered workflow.
TABS = nav.tabs(_SURFACES, summary.render_summary_page)
page = _resolve_page(TABS)

if page == _PAGE_WELCOME:
    with st.container(key="pg_welcome"):
        onboarding.render_landing()
    st.stop()
if page == _PAGE_UPLOAD:
    with st.container(key="pg_upload"):
        onboarding.render_upload()
    st.stop()

# Dashboard. on_click tab handlers have already fired, so this reads the fresh
# page. On a hard refresh the tab + selections are restored from the URL.
_restore_selection_from_url()
_top_bar(page, TABS)

# The body renders the tab's own renderer, or its stated-absence page.
active = nav.find(TABS, page)
with st.container(key=f"page_body_{page}"):
    if active is None:
        # Only reachable under a bare import (no ScriptRunContext). Nothing to draw.
        pass
    elif active.render is None or not active.available():
        nav.render_unavailable(active)
    else:
        active.render()

_kill_cross_page_bleed()

# Mirror the current cluster + its selections to the URL so a refresh restores them.
_sync_selection_to_url(page)
