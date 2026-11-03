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
    _TISSUE_HEIGHT,
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
    _unavailable,
    _unavailable_panel,
)


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _n_with_density(genes: tuple[str, ...], bin_um: int) -> int:
    """How many driving genes actually contributed a frame to the sum.

    ``leading_edge_density`` skips a gene with no precomputed frame, and the summed
    picture cannot show which ones are missing — so the count goes in the
    conditionality. Counted HERE the way the sum sums: by asking for each gene's
    frame at this bin size and counting the ones that come back non-empty.

    It used to count against ``available_density_markers()``, which is the density
    index's DECLARATION of what was precomputed (and, with no index at all, the
    study's demo marker set). That is a claim about the run, not a reading of what
    was read: on a study declaring ``density: true`` with a partial precompute it
    counted three of three driving genes for a figure summed from one frame, so the
    caption stated a count larger than the evidence behind the picture. R5 — plots
    render, they do not derive — and a count in a caption is a derived statistic: it
    has to come from what was drawn. The bin size is part of that, because a frame
    exists per (gene, bin_um) and the viewing control picks which.

    ``hexbins`` is cached per (gene, bin_um) and the sum has just asked for the same
    frames, so this costs cache hits rather than a second read.
    """
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
    # Guarded for the same reason as the panel below, though this one never raised
    # (``leading_edge_density`` skips a gene with no frame and returns None): without
    # the guard the empty state read "no precomputed density for the leading-edge
    # genes", which blames the genes for an absence that belongs to the whole study.
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
    # This stage reuses the marker page's renderers, each of which carries its own
    # view guard -- but the two panels it owns itself had none, so on a study whose
    # viz stage never ran the Pathways page still took the loader's traceback here:
    # ``leading_edge_expr`` and ``umap_df`` are both read before any guarded call.
    # Two views, for the same reason the marker feature UMAP needs two -- the
    # embedding is the layout, the exported expression is the colour.
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
        config=_plot_config(),
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
