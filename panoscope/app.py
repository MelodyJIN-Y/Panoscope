"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint. The chat is the product; the panels are the evidence it
stands on. Every number shown here comes from the agent layer (jazzPanda output,
the panel list, or a cited lab note) — this file only wires panels together.

Navigation lives in a single TOP bar (not a sidebar): the brand and three tabs —
``Marker genes | Pathways | Summary`` — in one row, left-to-right in workflow
order (the Summary is the final, integrated step). Lab notes are folded into the
Summary page. Session state is shared across pages natively (one script run per
rerun), so the selected cluster / markers / chat / notes carry over between tabs.
"""
from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

from ui import (
    cluster_rail,
    conversation,
    enrichment_table,
    evidence_table,
    onboarding,
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


# Active top-tab page. Held in session state for snappy in-session switching AND
# mirrored to the URL query param (?page=) so a full browser refresh restores the
# tab instead of dropping back to the default — the tab is also shareable now.
_K_PAGE = "active_page"
# Every page has a dedicated ?page= URL name.
_PAGE_WELCOME = "welcome"      # landing
_PAGE_UPLOAD = "upload"        # load-your-data
_PAGE_EXAMINE = "markers"      # marker-genes dashboard
_PAGE_SUMMARY = "summary"
_PAGE_PATHWAYS = "pathways"
_ONBOARDING_PAGES = (_PAGE_WELCOME, _PAGE_UPLOAD)
_VALID_PAGES = (_PAGE_WELCOME, _PAGE_UPLOAD, _PAGE_EXAMINE, _PAGE_SUMMARY, _PAGE_PATHWAYS)


def _set_page(page: str) -> None:
    """on_click handler for a top tab — fires before the rerun renders."""
    st.session_state[_K_PAGE] = page
    st.query_params["page"] = page  # keep the URL in sync so a refresh persists the tab


def _resolve_page() -> str:
    """The active tab, restoring it from the URL after a browser refresh.

    session_state is wiped on a hard refresh but the ``?page=`` query param
    survives, so on a fresh session we seed the tab from the URL (validated), then
    keep the URL mirrored to the current tab.
    """
    if _K_PAGE not in st.session_state:
        url_page = st.query_params.get("page")
        if url_page in _VALID_PAGES:
            default = url_page
        elif any(k in st.query_params for k in ("sign", "undo", "drill", "confirm", "accept")):
            # A Summary-table action link (?drill=cN etc.) drops the page param; the
            # user is on the Summary dashboard, not making a fresh visit -> not welcome.
            default = _PAGE_SUMMARY
        else:
            default = _PAGE_WELCOME
        st.session_state[_K_PAGE] = default
    page = st.session_state[_K_PAGE]
    if st.query_params.get("page") != page:
        st.query_params["page"] = page
    return page


def _restore_selection_from_url() -> None:
    """On a fresh session (a browser refresh), restore the selected cluster and that
    cluster's selected markers + gene sets from the URL, so a refresh keeps what the
    biologist selected instead of resetting to defaults. Runs once per session.
    """
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
    """Mirror the current cluster + its selections to the URL (write only on change,
    so a refresh restores them without triggering redundant reruns).

    The Summary page is dataset-wide, so it carries no cluster/marker params — the
    URL stays a clean ``?page=summary``. Marker/Pathways pages keep the selection.
    """
    qp = st.query_params
    if page == _PAGE_SUMMARY:
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


def _top_bar(page: str) -> None:
    """The single top bar: brand · Marker genes / Pathways / Summary tabs.

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
                # Order: Summary · Marker genes · Pathways. Summary leads — it is the
                # review surface the biologist lands on and signs off from; the marker
                # and pathway panes are the evidence it stands on.
                t_summary, t_examine, t_pathways = st.columns(3)
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
                        "Marker genes",
                        key="nav_examine",
                        type="primary" if page == _PAGE_EXAMINE else "secondary",
                        use_container_width=True,
                        on_click=_set_page,
                        args=(_PAGE_EXAMINE,),
                    )
                with t_pathways:
                    st.button(
                        "Pathways",
                        key="nav_pathways",
                        type="primary" if page == _PAGE_PATHWAYS else "secondary",
                        use_container_width=True,
                        on_click=_set_page,
                        args=(_PAGE_PATHWAYS,),
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


@st.fragment
def _chat_pane_fragment(cluster: str) -> None:
    """The marker chat as a fragment: sending a message re-renders ONLY the chat pane
    (no whole-app refresh, no spatial-figure flicker). The chat's own reruns are scoped
    to this fragment (ui.conversation uses ``st.rerun(scope='fragment')``)."""
    conversation.render_conversation(cluster)


def _kill_cross_page_bleed() -> None:
    """Hide the "cross-page bleed" — the previous page's faded leftover after a tab switch.

    On a tab switch Streamlit reuses the body container's DOM node and diffs its children in
    place; going from a taller page (Summary) to a shorter one leaves the taller page's
    trailing children mounted, marked ``data-stale="true"`` (rendered at opacity 0.33), and
    only PHYSICALLY removed by a later frontend pass — which on some machines/flows lags long
    enough that the faded block just sits below the new page. No server-side trick clears it:
    st.rerun discards the current frame, a programmatic query-param write does not rerun, and
    a rerun re-renders the CURRENT tree without touching orphaned stale DOM.

    So we hide it at the source. A singleton parent-document watcher adds ``pano-killbleed``
    to <html> the moment ``?page=`` changes; the CSS below then hides any ``data-stale``
    element INSIDE a page body (the leftover), scoped so normal same-page reruns still just
    fade as Streamlit intends. The class lifts as soon as the body has no stale leftover
    (residue physically gone), with a 4s hard cap so it can never hide content indefinitely.
    Same ``window.parent`` bridge the chat auto-scroll already uses.
    """
    import streamlit.components.v1 as components

    st.markdown(
        "<style>"
        '.pano-killbleed [class*="st-key-page_body_"] [data-stale="true"]'
        "{display:none !important;}"
        "</style>",
        unsafe_allow_html=True,
    )
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
    if state.is_paper_open():
        paper_drawer.render_paper_drawer()
    # Narrower center (the spatial figures letterbox less horizontally) so the chat
    # pane gets more room; the rail stays fixed.
    rail_col, center_col, chat_col = st.columns([216, 664, 474], gap="small")
    with rail_col:
        cluster_rail.render_rail()
    cluster = state.get_selected_cluster()
    with center_col:
        verdict_header.render_verdict(cluster)
        evidence_table.render_evidence_table(cluster)
        spatial_stage.render_spatial_stage(cluster)
    with chat_col:
        _chat_pane_fragment(cluster)


# ── One script run per rerun (no st.navigation / sidebar) ──────────────────
st.set_page_config(
    page_title="Panoscope",
    page_icon=str(_LOGO_MARK) if _LOGO_MARK.exists() else "🔬",
    layout="wide",
)
theme.inject_css()
state.init_state()

# Page routing — each page has a dedicated ?page= URL name:
#   welcome (landing) · upload · markers · pathways · summary.
# welcome + upload are the onboarding front door; the rest are the dashboard.
page = _resolve_page()

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
_top_bar(page)

# The dashboard body renders into a page-keyed container.
with st.container(key=f"page_body_{page}"):
    if page == _PAGE_SUMMARY:
        summary.render_summary_page()
    elif page == _PAGE_PATHWAYS:
        enrichment_table.render_pathways_page()
    else:
        _examine_body()

_kill_cross_page_bleed()

# Mirror the current cluster + its selections to the URL so a refresh restores them.
_sync_selection_to_url(page)
