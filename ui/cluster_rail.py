"""Left pane — the cluster rail (c1..c9).

One row per cluster: the numeric id, the cell-type call, its top markers, a
confidence dot, and a verify flag. Selecting a row sets
``state.selected_cluster`` and nothing else — the rail is navigation, never a
computation. Every value shown (cell type, markers, confidence band, verify
flag) is read straight off the cached ``ClusterVerdict`` produced by
``agent.verdict`` via ``ui.data_access``; this module invents nothing.

Rendering strategy (Streamlit): each row is a real ``st.button`` (so selection
is a normal rerun-safe click) whose *label area* is drawn by a sibling HTML
card matching the wireframe ``.clu`` markup. The button spans the row and is
made transparent by a scoped ``.pano-rail-row`` style so the visible surface is
the HTML card, while the click target stays a genuine Streamlit control (no
brittle JS bridge). The confidence dot uses the per-band confidence color; the
left id keeps the stable per-cluster color as a spine so the rail reads as a
legend for the spatial views.

Import-safe: Streamlit is imported lazily inside ``render_rail`` so
``import ui.cluster_rail`` needs no running server.
"""

from __future__ import annotations

import html
from typing import Optional

from agent.config import CLUSTER_KEY, CLUSTER_ORDER
from agent.types import ClusterVerdict

from ui import data_access, format as fmt, state

# --------------------------------------------------------------------------- #
# Confidence-band dot colors — mirror the theme.py confidence tokens so the dot
# reads on the same scale as the header chip (Very-High deepest teal -> Low
# faint grey). Keyed by band label, not by index.
# --------------------------------------------------------------------------- #
_CONFIDENCE_DOT: dict[str, str] = {
    "Very High": "#0F5B65",
    "High": "#2E8C97",
    "Medium-High": "#5FA7AE",
    "Medium": "#A9C7CB",
    "Low": "#E2E6E7",
}
_DEFAULT_DOT = "#E2E6E7"

_MAX_MARKERS_IN_RAIL = 3  # keep the subline short; the table shows the full set


def _dot_color(confidence: str) -> str:
    """Confidence-band color for the rail dot (faint-grey fallback)."""
    return _CONFIDENCE_DOT.get(confidence, _DEFAULT_DOT)


def _sub_markers(verdict: ClusterVerdict) -> str:
    """Space-joined top markers for the row subline (never fabricates a name).

    Reads the verdict's already-computed ``key_markers`` (top by glm_coef) and
    shows at most three. Empty string when a cluster has no assigned markers, so
    the row degrades gracefully rather than printing a stray value.
    """
    return " ".join(verdict.key_markers[:_MAX_MARKERS_IN_RAIL])


def _cluster_label(cluster: str) -> str:
    """Human cell-type name for a cluster from the authoritative key (fallback: id)."""
    entry = CLUSTER_KEY.get(cluster)
    if entry:
        return entry["cell_type"].replace("_", " ")
    return cluster


def _row_html(verdict: ClusterVerdict, *, selected: bool) -> str:
    """Return the ``.clu`` card markup for one cluster row (pure, escaped).

    Matches the wireframe rail row: id spine · (name + markers) · confidence dot,
    with a verify flag appended to the name when the verdict asks for a re-check.
    All dynamic text is HTML-escaped — cell types and marker names are trusted
    source values, but escaping keeps this XSS-safe by construction.
    """
    cid = html.escape(fmt.short_cluster_id(verdict.cluster))
    name = html.escape(verdict.cell_type.replace("_", " "))
    sub = html.escape(_sub_markers(verdict)) or "&mdash;"
    id_color = fmt.cluster_color(verdict.cluster)
    dot_color = _dot_color(verdict.confidence)
    sel_cls = " sel" if selected else ""
    flag = (
        '<span class="pano-clu-flag" title="re-check this call">&#9873;</span>'
        if verdict.verify
        else ""
    )
    return (
        f'<div class="pano-clu{sel_cls}">'
        f'<span class="pano-clu-id" style="color:{id_color}">{cid}</span>'
        f'<div class="pano-clu-body">'
        f'<div class="pano-clu-name">{name}{flag}</div>'
        f'<div class="pano-clu-sub">{sub}</div>'
        f"</div>"
        f'<span class="pano-clu-dot" style="background:{dot_color}"'
        f' title="{html.escape(verdict.confidence)} confidence"></span>'
        f"</div>"
    )


def _missing_row_html(cluster: str, *, selected: bool) -> str:
    """Defensive card for a cluster whose verdict failed to load (still navigable)."""
    sel_cls = " sel" if selected else ""
    return (
        f'<div class="pano-clu{sel_cls}">'
        f'<span class="pano-clu-id" style="color:{fmt.cluster_color(cluster)}">'
        f"{html.escape(fmt.short_cluster_id(cluster))}</span>"
        f'<div class="pano-clu-body"><div class="pano-clu-name">'
        f"{html.escape(_cluster_label(cluster))}</div>"
        f'<div class="pano-clu-sub">&mdash;</div></div>'
        f'<span class="pano-clu-dot" style="background:{_DEFAULT_DOT}"></span></div>'
    )


# --------------------------------------------------------------------------- #
# Scoped CSS — turns each row's Streamlit button into a transparent full-width
# click target so the visible surface is the HTML card above it. Injected once
# per run (idempotent; harmless to repeat).
# --------------------------------------------------------------------------- #
_RAIL_CSS = """
<style>
.pano-rail-title{
  font-family:var(--mono,monospace);font-size:10px;text-transform:uppercase;
  letter-spacing:.1em;color:var(--faint,#9AA3AB);font-weight:500;margin:0 4px 10px;
}
.pano-clu{
  display:grid;grid-template-columns:22px 1fr auto;gap:9px;align-items:center;
  padding:9px 8px;border-radius:8px;border:1px solid transparent;
}
.pano-clu.sel{background:var(--accent-soft,#EBF5F6);border-color:#D3E9EB;}
.pano-clu-id{font-family:var(--mono,monospace);font-size:12px;font-weight:600;}
.pano-clu-name{font-weight:600;font-size:13px;color:var(--ink,#161B20);line-height:1.25;}
.pano-clu-sub{font-size:11px;color:var(--muted,#606A73);font-family:var(--mono,monospace);
  margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.pano-clu-dot{width:9px;height:9px;border-radius:50%;box-shadow:0 0 0 3px rgba(0,0,0,.02);}
.pano-clu-flag{color:var(--absent,#BE7A1E);font-size:10px;margin-left:5px;}

/* Each row: card sits above a transparent, negative-margin button overlay so
   the visible surface is the HTML card while the click target is a real button. */
.pano-rail-row{position:relative;}
.pano-rail-row div[data-testid="stButton"]{margin-top:-62px;margin-bottom:4px;}
.pano-rail-row div[data-testid="stButton"] > button{
  width:100%;min-height:54px;background:transparent;border:1px solid transparent;
  color:transparent;box-shadow:none;padding:0;
}
.pano-rail-row div[data-testid="stButton"] > button:hover{
  background:rgba(15,123,135,.05);border-color:transparent;
}
.pano-rail-row div[data-testid="stButton"] > button:focus-visible{
  outline:2px solid var(--accent,#0F7B87);outline-offset:-2px;
}
</style>
"""


def render_rail() -> Optional[str]:
    """Render the cluster rail and return the currently selected cluster id.

    Reads all nine verdicts from ``ui.data_access.all_verdicts`` (cached, so this
    never recomputes a value) and the current selection from
    ``ui.state.get_selected_cluster``. Clicking a row calls
    ``ui.state.set_selected_cluster`` (which only mutates ``selected_cluster``)
    and triggers a rerun; the freshly selected id is returned for the caller.

    Selection is fail-closed: ``set_selected_cluster`` ignores ids outside
    ``CLUSTER_ORDER``, so a stray click can never point the app at a phantom
    cluster.
    """
    import streamlit as st

    st.markdown(_RAIL_CSS, unsafe_allow_html=True)
    st.markdown('<p class="pano-rail-title">Clusters</p>', unsafe_allow_html=True)

    verdicts = {v.cluster: v for v in data_access.all_verdicts()}
    selected = state.get_selected_cluster()

    for cluster in CLUSTER_ORDER:
        verdict = verdicts.get(cluster)
        is_sel = cluster == selected

        st.markdown('<div class="pano-rail-row">', unsafe_allow_html=True)
        if verdict is not None:
            st.markdown(_row_html(verdict, selected=is_sel), unsafe_allow_html=True)
            aria = f"Select cluster {fmt.short_cluster_id(cluster)}: {verdict.cell_type}"
        else:
            st.markdown(_missing_row_html(cluster, selected=is_sel), unsafe_allow_html=True)
            aria = f"Select cluster {fmt.short_cluster_id(cluster)}"

        if st.button(aria, key=f"rail_{cluster}", use_container_width=True):
            state.set_selected_cluster(cluster)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    return state.get_selected_cluster()


__all__ = ["render_rail"]
