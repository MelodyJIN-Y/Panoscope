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

from typing import Optional

from agent.config import CLUSTER_KEY, CLUSTER_ORDER
from agent.types import ClusterVerdict

from ui import data_access, state

# The rail dot is the cluster button's CSS ``::before`` (coloured per cluster in
# ui/theme.py to match that cluster's colour in the cell map, UMAP, and dot plot),
# so the dot beside a cluster name is a single visual key across the whole app.


def _cluster_label(cluster: str) -> str:
    """Human cell-type name for a cluster from the authoritative key (fallback: id)."""
    entry = CLUSTER_KEY.get(cluster)
    if entry:
        return entry["cell_type"].replace("_", " ")
    return cluster


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

            # Show the cluster id alongside the cell type (e.g. "c1 Tumor") — the
            # id is how every other surface (spatial views, chat, notes) refers to
            # the cluster, so the rail names it the same way.
            if verdict is not None:
                # display_cell_type reflects a confirmed override (else the computed call).
                name = f"{cluster} {data_access.display_cell_type(cluster).replace('_', ' ')}"
                label = f"{name}  ⚑" if verdict.verify else name
            else:
                label = f"{cluster} {_cluster_label(cluster)}"

            # One chromeless button per cluster. The colour dot is the button's
            # ``::before`` (theme, coloured per cluster via the ``st-key-rail_cN``
            # class), so it sits right next to the left-aligned name — perfectly
            # aligned. The selected row is a simple accent tint (like a selected
            # gene row), no left bar. Button key ``rail_<cluster>`` drives the dot
            # colour rule.
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
