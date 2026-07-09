"""Left pane — the cluster rail (c1..c9).

A light, borderless clickable **list**: one row per cluster is a small confidence
dot beside a chromeless cell-type button. The selected row is accent-highlighted
(an accent left-bar + bold name). Selecting a row sets ``state.selected_cluster``
and nothing else — the rail is navigation, never a computation. Every value shown
(cell type, confidence band, verify flag) is read straight off the cached
``ClusterVerdict`` produced by ``agent.verdict`` via ``ui.data_access``; this
module invents nothing.

Rendering strategy (Streamlit): the whole list lives inside a real
``st.container(key="pano_rail")`` — which emits a genuine wrapper element
(``.st-key-pano_rail``) that actually contains its child buttons, unlike a
``st.markdown('<div>')`` block (which auto-closes empty and cannot wrap later
widgets). The theme's ``.st-key-pano_rail`` rules then de-chrome those buttons
into light list rows and give the selected (``type="primary"``) row an accent —
no brittle absolute-overlay, no heavy button box. The confidence dot is plain
HTML in the row's left column.

All borderless styling lives in ``ui.theme``; this module emits only markup and
the ``pano-raildot`` span. Streamlit is imported lazily inside ``render_rail`` so
``import ui.cluster_rail`` needs no running server.
"""

from __future__ import annotations

import html
from typing import Optional

from agent.config import CLUSTER_KEY, CLUSTER_ORDER
from agent.types import ClusterVerdict

from ui import data_access, state

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


def _dot_color(confidence: str) -> str:
    """Confidence-band color for the rail dot (faint-grey fallback)."""
    return _CONFIDENCE_DOT.get(confidence, _DEFAULT_DOT)


def _cluster_label(cluster: str) -> str:
    """Human cell-type name for a cluster from the authoritative key (fallback: id)."""
    entry = CLUSTER_KEY.get(cluster)
    if entry:
        return entry["cell_type"].replace("_", " ")
    return cluster


def _dot_html(color: str, title: str) -> str:
    """The small confidence dot for a rail row (pure, escaped title)."""
    return (
        f'<div class="pano-raildot" style="background:{color}" '
        f'title="{html.escape(title)}"></div>'
    )


def render_rail() -> Optional[str]:
    """Render the cluster rail and return the currently selected cluster id.

    Reads all nine verdicts from ``ui.data_access.all_verdicts`` (cached, so this
    never recomputes a value) and the current selection from
    ``ui.state.get_selected_cluster``. Each row is a confidence dot + a chromeless
    cell-type button; clicking one calls ``ui.state.set_selected_cluster`` (which
    only mutates ``selected_cluster``) via ``on_click`` and Streamlit reruns.

    Selection is fail-closed: ``set_selected_cluster`` ignores ids outside
    ``CLUSTER_ORDER``, so a stray click can never point the app at a phantom
    cluster.
    """
    import streamlit as st

    st.markdown('<p class="pano-eyebrow">Clusters</p>', unsafe_allow_html=True)

    verdicts = {v.cluster: v for v in data_access.all_verdicts()}
    selected = state.get_selected_cluster()

    with st.container(key="pano_rail"):
        for cluster in CLUSTER_ORDER:
            verdict: Optional[ClusterVerdict] = verdicts.get(cluster)
            is_sel = cluster == selected

            if verdict is not None:
                name = verdict.cell_type.replace("_", " ")
                label = f"{name}  ⚑" if verdict.verify else name
                dot = _dot_html(
                    _dot_color(verdict.confidence), f"{verdict.confidence} confidence"
                )
            else:
                label = _cluster_label(cluster)
                dot = _dot_html(_DEFAULT_DOT, "confidence unavailable")

            dot_col, btn_col = st.columns([0.14, 0.86], vertical_alignment="center")
            with dot_col:
                st.markdown(dot, unsafe_allow_html=True)
            with btn_col:
                st.button(
                    label,
                    key=f"rail_{cluster}",
                    type="primary" if is_sel else "secondary",
                    use_container_width=True,
                    on_click=state.set_selected_cluster,
                    args=(cluster,),
                )

    return state.get_selected_cluster()


__all__ = ["render_rail"]
