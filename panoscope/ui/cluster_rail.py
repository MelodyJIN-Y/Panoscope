"""Left pane — the cluster rail: one chromeless button per cluster, read off the
composed verdicts. Selecting a row sets ``state.selected_cluster`` and nothing else."""


from __future__ import annotations

from panoscope.agent import config as cfg
from typing import Optional

from panoscope.ui import data_access, state

# The rail dot is the button's CSS ``::before``, coloured per cluster in ui/theme.py.


def render_rail() -> Optional[str]:
    """Render the cluster rail and return the currently selected cluster id.
    Selection is fail-closed (``set_selected_cluster`` ignores unknown ids)."""
    import streamlit as st

    st.markdown('<p class="pano-eyebrow">Clusters</p>', unsafe_allow_html=True)

    # Composed, not raw: a sign-off clears ⚑ here as on the board.
    verdicts = {v.cluster: v for v in data_access.composed_verdicts()}
    selected = state.get_selected_cluster()

    with st.container(key="pano_rail"):
        for cluster in cfg.CLUSTER_ORDER:
            is_sel = cluster == selected

            # "c1 Tumor": id plus the override-aware cell type.
            name = f"{cluster} {data_access.display_cell_type(cluster).replace('_', ' ')}"
            label = f"{name}  ⚑" if verdicts[cluster].verify else name

            # Key ``rail_<cluster>`` drives the theme's dot colour rule.
            st.button(
                label,
                key=f"rail_{cluster}",
                type="primary" if is_sel else "secondary",
                width="stretch",
                on_click=state.set_selected_cluster,
                args=(cluster,),
            )

    return state.get_selected_cluster()
