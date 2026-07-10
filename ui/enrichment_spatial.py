"""The Pathways spatial stage — a pinned pathway's LEADING-EDGE gene set on tissue.

The enrichment analog of ``ui.spatial_stage``, reusing its renderers so the
Pathways page reads like the Marker-genes page. For the pinned pathway it shows,
top to bottom:

  * Row 1 (context) — the cluster **cell map** beside the cluster **UMAP** (the
    same context row the marker page uses).
  * Row 2 (the program) — the pathway's **leading-edge transcript density** on
    tissue (the SUM of its driving genes' precomputed, area-normalized hex
    densities) beside the **leading-edge activity UMAP** (the MEAN of their
    per-cell expression, plasma low->high).
  * A dot plot of the leading-edge genes across all nine clusters.

This is the view that answers what the score/q cannot: *is the enriched program
spatially in this cluster's cells, or bleeding in from a neighbour?* Grounding
invariant: the set density is a SUM of measured per-gene densities (a viewing
representation), area-normalized, and is NOT the enrichment statistic — the bin
size is a viewing control that changes the picture, never a value.
"""

from __future__ import annotations

from typing import Any

from ui import data_access as da
from ui import state
from ui.spatial_stage import (
    _MAX_BG_POINTS,
    _base_layout,
    _downsample_bg,
    _empty_panel,
    _go,
    _inject_stage_css,
    _legend_line,
    _panel_title,
    _plot_config,
    _render_cell_map,
    _render_density_controls,
    _render_density_frame,
    _render_dotplot,
    _render_umap,
    _st,
    _umap_feature,
)


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _render_set_density(genes: tuple[str, ...]) -> None:
    """The leading-edge set's transcript density (sum of the driving genes'
    precomputed hex densities), drawn as a ratio-locked tissue image."""
    bin_um = state.get_bin_um()
    hb = da.leading_edge_density(tuple(genes), bin_um)
    _render_density_frame(hb, bin_um, empty="no precomputed density for the leading-edge genes")


def _render_set_feature_umap(cluster: str, genes: tuple[str, ...]) -> None:
    """UMAP recoloured by the leading-edge set's mean per-cell expression."""
    st = _st()
    go = _go()
    expr = da.leading_edge_expr(tuple(genes))
    umap = da.umap_df()
    if umap.empty or expr is None:
        _empty_panel("no per-cell expression for the leading-edge genes")
        return
    sel_mask = umap["cluster"] == cluster
    view = _downsample_bg(umap, sel_mask, _MAX_BG_POINTS)
    fig = go.Figure(layout=_base_layout(go, showlegend=False))
    _umap_feature(fig, go, view, expr, cluster)
    st.plotly_chart(fig, use_container_width=True, config=_plot_config())


def render_pathway_spatial(cluster: str, pathway: Any) -> None:
    """Render the spatial stage for ``pathway`` (a PathwayEvidence) in ``cluster``.

    ``pathway.leading_edge`` are the driving genes; the views aggregate them so the
    biologist can see where the program lives on the tissue relative to the cluster.
    """
    st = _st()
    _inject_stage_css()
    genes = tuple(pathway.leading_edge)
    short = _short(pathway.gene_set)

    # Row 1 — cluster context (same as the marker page).
    r1_left, r1_right = st.columns(2, gap="medium")
    with r1_left:
        _panel_title("Cell map <span style='color:var(--faint)'>· tissue</span>")
        _render_cell_map(cluster)
    with r1_right:
        _panel_title("UMAP <span style='color:var(--faint)'>· clusters</span>")
        _render_umap(cluster, feature=False)

    if not genes:
        _empty_panel("this pathway has no leading-edge genes to map", height=140)
        return

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    _render_density_controls()

    # Row 2 — the pinned program's leading edge on tissue + in expression space.
    r2_left, r2_right = st.columns(2, gap="medium")
    with r2_left:
        _panel_title(f"<b>{short}</b> leading-edge transcript density")
        _render_set_density(genes)
    with r2_right:
        _panel_title(f"<b>{short}</b> leading-edge activity <span style='color:var(--faint)'>· UMAP</span>")
        _render_set_feature_umap(cluster, genes)
    _legend_line(
        "leading-edge = the driving genes (" + ", ".join(genes)
        + ") · density is the summed, area-normalized transcript signal (a view, not the enrichment score)"
    )

    # Dot plot: the leading-edge genes' expression across all nine clusters.
    _panel_title("Leading-edge genes across clusters")
    _render_dotplot(list(genes), cluster)


__all__ = ["render_pathway_spatial"]
