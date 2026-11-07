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

It answers what the score cannot: is the program spatially in this cluster's
cells? The set density is a viewing sum of measured densities, not the enrichment
statistic; the bin size changes the picture, never a value.
"""

from __future__ import annotations

from typing import Any

from panoscope.agent.enrichment_themes import pretty_gene_set
from panoscope.ui import conditionality as cnd
from panoscope.ui import data_access as da
from panoscope.ui import state
from panoscope.ui.spatial_stage import (
    _MAX_BG_POINTS,
    _PLOT_CONFIG,
    _TISSUE_HEIGHT,
    _base_layout,
    _downsample_bg,
    _empty_panel,
    _go,
    _inject_stage_css,
    _legend_line,
    _panel_title,
    _render_cell_map,
    _render_density_controls,
    _render_density_frame,
    _render_dotplot,
    _render_umap,
    _st,
    _umap_feature,
    _unavailable,
    _unavailable_panel,
)


def _n_with_density(genes: tuple[str, ...], bin_um: int) -> int:
    """How many driving genes actually contributed a frame to the sum at this bin
    size, counted the way the sum reads them (cache hits), never from a declaration."""
    n = 0
    for g in genes:
        try:
            hb = da.hexbins(g, bin_um)
        except Exception:  # noqa: BLE001 - a missing frame is a skip, never a fake
            continue
        if hb is not None and not hb.empty:
            n += 1
    return n


def _n_with_expr(genes: tuple[str, ...]) -> int:
    """How many driving genes contributed a column to the mean (same reason)."""
    have = {c.upper() for c in da.available_expr_markers()}
    return sum(1 for g in genes if g.upper() in have)


def _render_set_density(program: str, genes: tuple[str, ...]) -> None:
    """The leading-edge set's transcript density (sum of the driving genes'
    precomputed hex densities), drawn as a ratio-locked tissue image."""
    # An absent view belongs to the study, not to the genes.
    missing = _unavailable("density")
    if missing:
        _unavailable_panel(missing, height=_TISSUE_HEIGHT)
        return

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
            n_drawn=_n_with_density(genes, bin_um),
        ),
        key=f"pano_density_set_{program}_{bin_um}",
        empty="no precomputed density for the leading-edge genes",
    )


def _render_set_feature_umap(program: str, cluster: str, genes: tuple[str, ...]) -> None:
    """UMAP recoloured by the leading-edge set's mean per-cell expression."""
    # Needs the embedding (layout) and the exported expression (colour).
    missing = _unavailable("umap", "expr")
    if missing:
        _unavailable_panel(missing)
        return

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
        config=_PLOT_CONFIG,
        key=f"pano_le_umap_{cluster}_{program}",
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
        short = pretty_gene_set(p.gene_set)
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
        # The uncapped count, so "N of M" names what was left out.
        _render_dotplot(all_genes[:14], cluster, n_selected=len(all_genes))
