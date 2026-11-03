"""Panoscope — a grounded conversation with spatial data.

Streamlit entrypoint. The chat is the product; the panels are the evidence it
stands on. Every number shown here comes from the agent layer (jazzPanda output,
the panel list, or a cited lab note) — this file only wires panels together.

Navigation lives in a single TOP bar (not a sidebar): the brand, then one tab per
registered workflow, in one row. The strip is **not listed here** — ``ui/nav.py``
builds it from the workflow registry, so registering a workflow is the only step
needed to make its declared surface reachable (ARCHITECTURE §13: the review surface
is declared by the workflow, not a UI concern). This file supplies the wiring only
— which renderer draws a surface id, and how to tell whether this study has a
result for it — in :data:`_SURFACES`. Lab notes are folded into the Summary page.
Session state is shared across pages natively (one script run per rerun), so the
selected cluster / markers / chat / notes carry over between tabs.
"""
from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

from panoscope.ui import (
    cluster_rail,
    conversation,
    data_access as da,
    enrichment_table,
    evidence_table,
    nav,
    onboarding,
    paper_drawer,
    spatial_stage,
    state,
    summary,
    theme,
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
# The onboarding front door. These two are NOT workflows (no study is open yet), so
# they are named here; every other ?page= name is a registered workflow's own
# ``UISpec.workflow_id``, read off the registry by ui/nav.py.
_PAGE_WELCOME = nav.ONBOARDING_PAGES[0]   # landing
_PAGE_UPLOAD = nav.ONBOARDING_PAGES[1]    # load-your-data


def _set_page(page: str) -> None:
    """on_click handler for a top tab — fires before the rerun renders."""
    st.session_state[_K_PAGE] = page
    st.query_params["page"] = page  # keep the URL in sync so a refresh persists the tab


def _resolve_page(strip: tuple[nav.Tab, ...]) -> str:
    """The active tab, restoring it from the URL after a browser refresh.

    session_state is wiped on a hard refresh but the ``?page=`` query param
    survives, so on a fresh session we seed the tab from the URL (validated against
    what this build actually registers), then keep the URL mirrored to the current
    tab. A name nobody answers to falls back to the landing page rather than to
    whichever page happens to be first — the same fail-closed rule the registries
    use for an unknown id.
    """
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
    """The top-bar context chip, DERIVED from the open study.

    Every value here used to be a literal reading "Xenium human breast · sample 1 /
    280 genes · 9 clusters". It was correct for the demo and wrong for every other
    study: ``demo_mini`` is a 9-gene panel over 3 clusters, and the multi-sample
    fixture is 12 genes over 3 clusters of no tissue at all. CLAUDE.md is explicit
    that panel size is dynamic across studies and derived from the file, never
    asserted -- and a chip that names the wrong panel size sits two inches above a
    table whose denominators come from the right one.

    Each field degrades to silence rather than to a guess: an unstated tissue drops
    out of the line instead of defaulting to the demo's.
    """
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
    return " · ".join(bits), " · ".join(facts)


def _top_bar(page: str, strip: tuple[nav.Tab, ...]) -> None:
    """The single top bar: brand · one tab per registered workflow · study chip.

    The tabs are chromeless buttons styled (theme ``.st-key-pano_topnav``) into a
    top tab strip — the active one (``type="primary"``) gets an accent underline.
    Which tabs exist, what they are called and what order they sit in all come from
    ``ui/nav.py`` reading the workflow registry; nothing is named here.
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
                # Summary leads (it is the surface the biologist signs off from),
                # then the workflows in task-kind order — all decided by nav.tabs().
                for col, tab in zip(st.columns(len(strip)), strip):
                    with col:
                        # A workflow with no result in this study keeps its tab and
                        # says so on hover, rather than vanishing: a missing tab and
                        # a missing capability look identical, and they are not.
                        st.button(
                            tab.title,
                            key=tab.key,
                            help=None if tab.available()
                            else f"No {tab.title.lower()} result for this study",
                            type="primary" if page == tab.page else "secondary",
                            width="stretch",
                            on_click=_set_page,
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
    # DEPRECATED BY STREAMLIT, AND NOT MIGRATED ON PURPOSE. The warning says
    # "replace st.components.v1.html with st.iframe", but st.iframe takes
    # src= (a URL or Path) and this takes an HTML STRING. The script below
    # reaches window.parent.document; served from a data: URL it would be
    # cross-origin and the browser would block exactly that access. So the
    # advice does not apply here -- migrating would break the feature, not
    # fix it. Revisit when Streamlit offers a srcdoc-shaped replacement.
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
        # The declared LabelBandHeader is part of the ASSIGNMENT surface and is drawn
        # by ui/evidence_table.py. ui/verdict_header.py hand-built the same header
        # above it and would now render it twice.
        evidence_table.render_evidence_table(cluster)
        spatial_stage.render_spatial_stage(cluster)
    with chat_col:
        _chat_pane_fragment(cluster)


def _markers_available() -> bool:
    """Does this study have a marker result to draw? Mirrors ``enrichment_available``
    — the question is *is there a result*, not *is the method any good*."""
    return not da.markers_df().empty


# The wiring, and the only place this file names a surface: which renderer draws a
# ``UISpec.workflow_id``, and how to tell whether THIS study has a result for it.
# Registering a workflow is what puts a tab in the strip; adding a line here is what
# gives that tab a page. A workflow with no line renders nav's stated-absence page —
# it keeps its tab, because a hidden tab and an unsupported question read the same.
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
theme.inject_css()
state.init_state()

# Page routing — each page has a dedicated ?page= URL name: welcome (landing) and
# upload are the onboarding front door; summary is the integrated review surface;
# every other name is a registered workflow's own surface id (markers · pathways
# today, whatever is registered tomorrow).
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

# The dashboard body renders into a page-keyed container. Which renderer runs is the
# tab's own, from the registry-built strip — there is no page branch here to forget
# to extend. A tab with no result in this study states the absence instead.
active = nav.find(TABS, page)
with st.container(key=f"page_body_{page}"):
    if active is None:
        # Not a dashboard page at all. Only reachable with no ScriptRunContext —
        # under a bare ``import panoscope.app`` the ``st.stop()`` above is a no-op,
        # so the onboarding pages fall through to here. Nothing to draw.
        pass
    elif active.render is None or not active.available():
        nav.render_unavailable(active)
    else:
        active.render()

_kill_cross_page_bleed()

# Mirror the current cluster + its selections to the URL so a refresh restores them.
_sync_selection_to_url(page)
