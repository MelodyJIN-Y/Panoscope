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

from panoscope.ui import conditionality as cnd
from panoscope.ui import data_access as da
from panoscope.ui import state
from panoscope.ui.spatial_stage import (
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


def _n_with_density(genes: tuple[str, ...]) -> int:
    """How many driving genes actually contributed a frame to the sum.

    ``leading_edge_density`` skips a gene with no precomputed frame, and the
    summed picture cannot show which ones are missing — so the count goes in the
    conditionality. Read from the density index (the same list the stage uses).
    """
    have = set(da.available_density_markers())
    return sum(1 for g in genes if g in have)


def _n_with_expr(genes: tuple[str, ...]) -> int:
    """How many driving genes contributed a column to the mean (same reason)."""
    have = {c.upper() for c in da.available_expr_markers()}
    return sum(1 for g in genes if g.upper() in have)


def _render_set_density(program: str, genes: tuple[str, ...]) -> None:
    """The leading-edge set's transcript density (sum of the driving genes'
    precomputed hex densities), drawn as a ratio-locked tissue image."""
    bin_um = state.get_bin_um()
    hb = da.leading_edge_density(tuple(genes), bin_um)
    _render_density_frame(
        hb,
        bin_um,
        cond=cnd.set_density_cond(
            program=program,
            hb=hb,
            bin_um=bin_um,
            n_genes=len(genes),
            n_drawn=_n_with_density(genes),
        ),
        empty="no precomputed density for the leading-edge genes",
    )


def _render_set_feature_umap(program: str, cluster: str, genes: tuple[str, ...]) -> None:
    """UMAP recoloured by the leading-edge set's mean per-cell expression."""
    go = _go()
    expr = da.leading_edge_expr(tuple(genes))
    umap = da.umap_df()
    if umap.empty or expr is None:
        _empty_panel("no per-cell expression for the leading-edge genes")
        return
    sel_mask = umap["cluster"] == cluster
    view = _downsample_bg(umap, sel_mask, _MAX_BG_POINTS)
    fig = go.Figure(layout=_base_layout(go, showlegend=False, equal_aspect=True))
    _umap_feature(fig, go, view, expr, cluster)
    cnd.render_panel(
        fig,
        cnd.leading_edge_umap_cond(
            program=program,
            n_genes=len(genes),
            n_drawn=_n_with_expr(genes),
            n_expr_cells=len(expr),
            n_total=len(umap),
            n_bg_drawn=len(view) - int(sel_mask.sum()),
            n_bg_total=int((~sel_mask).sum()),
        ),
        config=_plot_config(),
    )


def render_pathways_spatial(cluster: str, pathways: list) -> None:
    """Render the spatial stage for the SELECTED ``pathways`` in ``cluster``.

    Mirrors the marker small-multiples: Row 1 is the cluster context (cell map +
    cluster UMAP), then one leading-edge row per selected pathway (its driving genes
    summed on tissue + in expression space), then a dot plot of the union of the
    selected programs' leading-edge genes across all nine clusters.
    """
    st = _st()
    _inject_stage_css()

    # Row 1 — cluster context (same as the marker page).
    r1_left, r1_right = st.columns(2, gap="medium")
    with r1_left:
        _panel_title("Cell map <span style='color:var(--faint)'>· tissue</span>")
        _render_cell_map(cluster)
    with r1_right:
        _panel_title("UMAP <span style='color:var(--faint)'>· clusters</span>")
        _render_umap(cluster, feature=False)

    if not pathways:
        return

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    _render_density_controls()

    # One small-multiple row per selected program: leading-edge density | activity UMAP.
    all_genes: list[str] = []
    for p in pathways:
        genes = tuple(p.leading_edge)
        short = _short(p.gene_set)
        r_left, r_right = st.columns(2, gap="medium")
        with r_left:
            _panel_title(f"<b>{short}</b> leading-edge transcript density")
            _render_set_density(short, genes)
        with r_right:
            _panel_title(
                f"<b>{short}</b> leading-edge activity <span style='color:var(--faint)'>· UMAP</span>"
            )
            _render_set_feature_umap(short, cluster, genes)
        for g in genes:
            if g not in all_genes:
                all_genes.append(g)

    _legend_line(
        "leading-edge = each program's driving genes · density is the summed, "
        "area-normalized transcript signal (a view, not the enrichment score)"
    )

    # Dot plot: the union of the selected programs' leading-edge genes across clusters.
    if all_genes:
        _panel_title("Leading-edge genes across clusters")
        _render_dotplot(all_genes[:14], cluster)


__all__ = ["render_pathways_spatial"]
