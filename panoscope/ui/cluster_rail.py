"""Left pane — the cluster rail: one chromeless button per cluster, read off the
composed verdicts. Selecting a row sets ``state.selected_cluster`` and nothing else."""


from __future__ import annotations

from panoscope.agent import config as cfg
from typing import Optional

from panoscope.ui import data_access, state

# The rail dot is the cluster button's CSS ``::before`` (coloured per cluster in
# ui/theme.py to match that cluster's colour in the cell map, UMAP, and dot plot),
# so the dot beside a cluster name is a single visual key across the whole app.


def render_rail() -> Optional[str]:
    """Render the cluster rail and return the currently selected cluster id.

    Reads all nine verdicts from ``ui.data_access.all_verdicts`` (cached, so this
    never recomputes a value) and the current selection from
    ``ui.state.get_selected_cluster``. Each row is a confidence dot + a chromeless
    cell-type button; clicking one calls ``ui.state.set_selected_cluster`` (which
    only mutates ``selected_cluster``) via ``on_click`` and Streamlit reruns.

    Selection is fail-closed: ``set_selected_cluster`` ignores ids outside
    ``cfg.CLUSTER_ORDER``, so a stray click can never point the app at a phantom
    cluster.
    """
    import streamlit as st

    st.markdown('<p class="pano-eyebrow">Clusters</p>', unsafe_allow_html=True)

    # COMPOSED, not raw. ``composed_verdicts`` clears ``verify`` on a signed-off
    # unit -- it is adjudicated, and the board, the exports and the Summary header
    # all consume that. The rail read the raw verdict, so the moment a call was
    # signed off the board said done and the rail went on showing ⚑ beside it.
    # Two surfaces in one app disagreeing about the same unit's review state is
    # the defect this project keeps finding, and the rail already reaches for the
    # override-aware ``display_cell_type`` a few lines below for the same reason.
    verdicts = {v.cluster: v for v in data_access.composed_verdicts()}
    selected = state.get_selected_cluster()

    with st.container(key="pano_rail"):
        for cluster in cfg.CLUSTER_ORDER:
            is_sel = cluster == selected

            # Show the cluster id alongside the cell type (e.g. "c1 Tumor") — the
            # id is how every other surface (spatial views, chat, notes) refers to
            # the cluster, so the rail names it the same way.
            # display_cell_type reflects a confirmed override (else the computed call).
            name = f"{cluster} {data_access.display_cell_type(cluster).replace('_', ' ')}"
            label = f"{name}  ⚑" if verdicts[cluster].verify else name

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
                width="stretch",
                on_click=state.set_selected_cluster,
                args=(cluster,),
            )

    return state.get_selected_cluster()
