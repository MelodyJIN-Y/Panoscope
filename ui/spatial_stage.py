"""The spatial stage: a fixed grid of linked Plotly views for the selected markers.

Public entry point: ``render_spatial_stage(cluster)``.

The stage is a fixed grid — there is no view toggle. It always shows, top to
bottom:

  * **Row 1 (context, always on)** — the cluster **cell map** (segmented cells at
    tissue coordinates, selected cluster brought forward) beside the cluster
    **UMAP** (expression space, coloured by cluster). "Did the call land in the
    right cells, and is the cluster a clean island."
  * **One small-multiple row per selected marker** — that gene's transcript
    **density** (precomputed, area-normalized hex-bins, before cell calling)
    beside its **feature UMAP** (the same UMAP recoloured by that gene's per-cell
    expression). "Is the raw signal really there, and where does it sit in
    expression space." The 25 / 50 / 100 µm bin control lives on the density
    panel and nowhere else.
  * **A full-width grouped violin** — each selected gene's per-cell expression
    distribution across all nine clusters, one violin per cluster in cluster
    colour. "How cluster-specific is this marker, holistically."

When no marker is selected only Row 1 renders, followed by an honest placeholder
inviting a pick — nothing is faked.

Grounding invariant (the load-bearing one): every control on this stage is a
**viewing** control. The bin size and the selected-marker set change *the
picture*, never a value. The bin control selects which *precomputed* density
frame to draw (``ui.data_access.hexbins`` reads a different file per bin) — it
never re-bins, and the colour scale is the frame's already-area-normalized
``density`` column so a coarser bin cannot look hotter than a finer one. The
violin draws the per-cell expression the prep step exported and joins it onto the
authoritative cluster labels; nothing in this module computes a statistic or a
confidence.

Tissue aspect ratio: the cell map and the density map are true tissue images, so
they lock a 1:1 data aspect ratio (``_tissue_layout``: ``scaleanchor="x"``,
``scaleratio=1``, y reversed) — a coarse bin or a wide container can never stretch
the slide. The UMAP and the violin are abstract spaces and keep the free
``_base_layout``.

State is read through ``ui.state`` (never a raw session-state key): the selected
markers (``active_markers()``, capped for small-multiples legibility) and the bin
size (``get_bin_um()``). When a selected marker has no precomputed density or
expression frame, that panel shows an honest "not precomputed for this marker"
placeholder — it never fakes a value.

Streamlit and Plotly are imported lazily inside the render functions so the
module imports cleanly with no server running.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from agent.config import CLUSTER_ORDER

from ui import data_access as da
from ui import format as fmt
from ui import state

# --------------------------------------------------------------------------- #
# Constants — all purely cosmetic (viewing only)
# --------------------------------------------------------------------------- #
# Background-point ceiling per scattergl trace. 158k cells render fine in WebGL,
# but a hard ceiling on the *background* keeps the first paint snappy and browser
# memory bounded. This is a *display* downsample: it thins how many context dots
# we paint, it never changes a value and never touches the selected cluster
# (always drawn in full). The budget applies to the background only, so the
# selected cluster can exceed it and the background still never collapses to zero
# — the largest cluster (Tumor ≈ 62.7k) always keeps surrounding tissue context.
_MAX_BG_POINTS = 45_000

_FADE_COLOR = "#D3D8DC"        # non-selected cells on the cell map (muted grey)
_HIGHLIGHT_LINE = "#161B20"    # thin outline on selected-cluster cells (ink)
_DENSITY_COLORSCALE = [        # matches the wireframe teal ramp (#EAF3F4 -> #0F5B65)
    [0.0, "#EAF3F4"],
    [0.25, "#B7CDD0"],
    [0.5, "#7FB0B6"],
    [0.75, "#3C8F99"],
    [1.0, "#0F5B65"],
]
_EXPR_COLORSCALE = _DENSITY_COLORSCALE  # same low->high teal ramp for feature UMAP

# Panel heights. The tissue panels (cell map / density) are square-locked images
# (7520×5470 µm ⇒ W:H ≈ 1.375); the ratio lock does the shaping, this is only the
# panel's drawing height. The UMAP + violin are abstract and use the default.
# One shared height for the paired panels so the two columns of a row line up
# (the tissue image letterboxes within it; the UMAP fills it). The violin is a
# full-width chart of its own.
_PLOT_HEIGHT = 300          # UMAP (abstract space — free ratio)
_TISSUE_HEIGHT = 300        # cell map / density (ratio-locked tissue image)
_VIOLIN_HEIGHT = 340        # full-width grouped violin
_PLOT_BG = "#FFFFFF"

# Density tile size in px per bin — purely cosmetic legibility; COLOUR is the
# only thing that encodes signal.
_DENSITY_TILE_PX: dict[int, int] = {25: 5, 50: 7, 100: 11}


# --------------------------------------------------------------------------- #
# Small lazy-import helpers (keep the module import-safe with no server)
# --------------------------------------------------------------------------- #
def _st() -> Any:
    import streamlit as st

    return st


# --------------------------------------------------------------------------- #
# Stage-local styling. These classes (.pempty / .plegend / .ptitle) are used
# only by this stage and are not part of the shared theme, so the stage ships
# them itself. Injected once per session (idempotent flag) so reruns never
# restyle. Colours reference the theme's CSS custom properties so it stays in
# the design system.
# --------------------------------------------------------------------------- #
_STAGE_CSS = """
.plegend {
  font-family: var(--mono, ui-monospace, monospace); font-size: 11px;
  color: var(--muted, #606A73); margin-top: 4px; line-height: 1.5;
}
.ptitle {
  font-family: var(--mono, ui-monospace, monospace); font-size: 11px;
  color: var(--faint, #9AA3AB); margin: 2px 0 2px; letter-spacing: .02em;
}
.ptitle b { color: var(--accent, #0F7B87); font-weight: 600; }
.pempty {
  display: flex; align-items: center; justify-content: center; text-align: center;
  font-family: var(--mono, ui-monospace, monospace); font-size: 12px;
  color: var(--faint, #9AA3AB); background: var(--hair2, #EEF0F2);
  border: 1px dashed var(--hair, #E4E7EA); border-radius: 10px; padding: 0 24px;
}
"""


def _inject_stage_css() -> None:
    """Inject stage-local CSS once per session (idempotent; reruns re-inject
    harmlessly but the flag skips it). Import-safe: Streamlit imported lazily.
    """
    st = _st()
    if st.session_state.get("_stage_css_done"):
        return
    st.markdown(f"<style>{_STAGE_CSS}</style>", unsafe_allow_html=True)
    st.session_state["_stage_css_done"] = True


def _go() -> Any:
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------- #
# Shared Plotly layouts
# --------------------------------------------------------------------------- #
def _base_layout(go_mod: Any, *, showlegend: bool, height: int = _PLOT_HEIGHT) -> Any:
    """A shared, chrome-light Plotly layout (no axis ticks — this is a map).

    Used by the abstract views (UMAP, and the violin after it re-enables its
    axes): axes are hidden and the aspect ratio is free, so the point cloud fills
    the panel.
    """
    axis = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        visible=False,
        fixedrange=True,   # view locked — no zoom/pan (hover still works)
    )
    return go_mod.Layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=_PLOT_BG,
        plot_bgcolor=_PLOT_BG,
        xaxis=axis,
        yaxis=axis,
        showlegend=showlegend,
        font=dict(family="Space Grotesk, system-ui, sans-serif"),
        dragmode=False,
    )


def _tissue_layout(go_mod: Any, *, height: int = _TISSUE_HEIGHT) -> Any:
    """Layout for a **true tissue image** (cell map / density): 1:1 data aspect.

    ``yaxis.scaleanchor="x"`` + ``scaleratio=1`` lock one micron on x to one
    micron on y, so the slide keeps its real shape (W:H ≈ 1.375) no matter how
    wide the container is or how coarse the density bin is — a viewing control can
    never stretch the tissue. ``autorange="reversed"`` flips y so the map reads
    top-down like the physical slide. Axes stay hidden (a map, not a chart), and
    ``autoScale2d`` is removed from the toolbar (see ``_plot_config``) so the user
    cannot unlock the ratio.
    """
    axis_x = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        visible=False,
        fixedrange=True,       # view locked — no zoom/pan (hover still works)
    )
    axis_y = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        visible=False,
        scaleanchor="x",       # lock 1 µm on y to 1 µm on x — never stretch tissue
        scaleratio=1,
        autorange="reversed",  # tissue reads top-down like the slide
        fixedrange=True,       # view locked — no zoom/pan
    )
    return go_mod.Layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=_PLOT_BG,
        plot_bgcolor=_PLOT_BG,
        xaxis=axis_x,
        yaxis=axis_y,
        showlegend=False,
        font=dict(family="Space Grotesk, system-ui, sans-serif"),
        dragmode=False,
    )


def _plot_config() -> dict[str, Any]:
    """Plotly display config: the view is LOCKED — no zoom, no pan, no toolbar.

    A biologist reads the signal in place; dragging or scroll-zoom could push the
    tissue out of frame, which is only annoying here. Zoom/pan are disabled at the
    axis level (``fixedrange``) and via ``dragmode=False``; ``scrollZoom`` and
    ``doubleClick`` are off and the mode bar is hidden. Hover tooltips stay on, so
    per-cell / per-bin values remain readable.
    """
    return {
        "displaylogo": False,
        "displayModeBar": False,
        "scrollZoom": False,
        "doubleClick": False,
        "staticPlot": False,   # keep hover; only zoom/pan are disabled
    }


def _panel_title(html: str) -> None:
    """A small mono title above a panel (e.g. the gene name for a small-multiple)."""
    st = _st()
    st.markdown(f'<div class="ptitle">{html}</div>', unsafe_allow_html=True)


def _empty_panel(message: str, *, height: int = _PLOT_HEIGHT) -> None:
    """Render the honest empty state (never a faked plot)."""
    st = _st()
    st.markdown(
        f'<div class="pempty" style="height:{height}px">{message}</div>',
        unsafe_allow_html=True,
    )


def _legend_line(html: str) -> None:
    """A small mono caption under a view (the plegend row from the wireframe)."""
    st = _st()
    st.markdown(f'<div class="plegend">{html}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Display-only downsample of a background frame (never touches selected cluster)
# --------------------------------------------------------------------------- #
def _downsample_bg(df: pd.DataFrame, keep_mask: pd.Series, bg_cap: int) -> pd.DataFrame:
    """Return ``df`` with every ``keep_mask`` row kept in full (the selected
    cluster) plus at most ``bg_cap`` background rows.

    ``bg_cap`` bounds the BACKGROUND only, not the total — so a large selected
    cluster (Tumor ≈ 62.7k > the cap) is still drawn in full while the surrounding
    tissue keeps a bounded, non-zero sample for context. Deterministic (fixed
    seed) so the picture is stable across reruns. This is a rendering optimisation
    only: it changes how many context dots we paint, never a value.
    """
    keep = df[keep_mask]
    rest = df[~keep_mask]
    if len(rest) > bg_cap:
        rest = rest.sample(n=bg_cap, random_state=0)
    return pd.concat([keep, rest], axis=0)


# --------------------------------------------------------------------------- #
# View 1 — Cell map (tissue image; ratio-locked)
# --------------------------------------------------------------------------- #
def _render_cell_map(cluster: str) -> None:
    """Scattergl of all cells at tissue x/y, coloured by cluster; selected cluster
    brought forward (full colour + thin outline), the rest faded to grey.

    Pure viewing surface: it draws ``cells_df()`` (already cached) and the
    selected cluster comes straight from ``cluster``. No value is computed. The
    tissue aspect ratio is locked by ``_tissue_layout`` so the slide is never
    stretched.
    """
    st = _st()
    go = _go()

    cells = da.cells_df()
    if cells.empty:
        _empty_panel("no cell coordinates available", height=_TISSUE_HEIGHT)
        return

    sel_mask = cells["cluster"] == cluster
    view = _downsample_bg(cells, sel_mask, _MAX_BG_POINTS)
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]

    fig = go.Figure(layout=_tissue_layout(go))

    # Background: all other clusters, muted.
    fig.add_trace(
        go.Scattergl(
            x=bg["x"],
            y=bg["y"],
            mode="markers",
            marker=dict(size=2.2, color=_FADE_COLOR, opacity=0.55),
            hoverinfo="skip",
            showlegend=False,
            name="other clusters",
        )
    )
    # Foreground: the selected cluster filled with its own palette colour (no
    # outline — just the filled points).
    sel_color = fmt.cluster_color(cluster)
    fig.add_trace(
        go.Scattergl(
            x=sel["x"],
            y=sel["y"],
            mode="markers",
            marker=dict(size=3.4, color=sel_color, opacity=0.95),
            customdata=sel["cell_id"],
            hovertemplate="cell %{customdata}<extra></extra>",
            showlegend=False,
            name=cluster,
        )
    )

    st.plotly_chart(fig, use_container_width=True, config=_plot_config())


# --------------------------------------------------------------------------- #
# View 2 — UMAP (cluster colour in Row 1; per-gene feature colour in gene rows)
# --------------------------------------------------------------------------- #
def _render_umap(cluster: str, *, feature: bool, gene: Optional[str] = None) -> None:
    """Scattergl in UMAP space (abstract — free aspect ratio via ``_base_layout``).

    ``feature=False`` (Row 1): always cluster-coloured (``_umap_by_cluster``),
    selected cluster outlined.
    ``feature=True`` (a gene row): recolour by ``gene``'s per-cell expression
    (``marker_expr_col``) if exported (``_umap_feature``); if the gene has no
    exported expression, fall back to the cluster-coloured UMAP with an honest
    note. ``gene`` is required when ``feature`` is True.
    """
    st = _st()
    go = _go()

    umap = da.umap_df()
    if umap.empty:
        _empty_panel("no UMAP embedding available")
        return

    expr = da.marker_expr_col(gene) if (feature and gene) else None

    sel_mask = umap["cluster"] == cluster
    view = _downsample_bg(umap, sel_mask, _MAX_BG_POINTS)

    fig = go.Figure(layout=_base_layout(go, showlegend=False))

    if feature and gene and expr is not None:
        _umap_feature(fig, go, view, expr, cluster)
        st.plotly_chart(fig, use_container_width=True, config=_plot_config())
        _legend_line('low <span class="grad"></span> high')
        return

    # Cluster-coloured UMAP: Row 1 default, or a gene row whose expression is
    # not exported (honest fallback).
    _umap_by_cluster(fig, go, view, cluster)
    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    if feature and gene and expr is None:
        _legend_line(
            f'<span style="color:var(--absent)">{gene}: per-cell expression not '
            f"precomputed</span> · showing cluster colour instead"
        )
    # Row-1 cluster UMAP: no caption (the label was removed for a cleaner grid).


def _umap_by_cluster(fig: Any, go: Any, view: pd.DataFrame, cluster: str) -> None:
    """Grey out every other cluster; fill the selected cluster with its own
    palette colour (no outline). Mirrors the cell map's selected-vs-rest
    treatment so the two Row-1 panels read the same way."""
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]
    fig.add_trace(
        go.Scattergl(
            x=bg["umap_1"],
            y=bg["umap_2"],
            mode="markers",
            marker=dict(size=2.2, color=_FADE_COLOR, opacity=0.5),
            hoverinfo="skip",
            showlegend=False,
            name="other clusters",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=sel["umap_1"],
            y=sel["umap_2"],
            mode="markers",
            marker=dict(size=3.2, color=fmt.cluster_color(cluster), opacity=0.95),
            hovertemplate=f"{cluster}<extra></extra>",
            showlegend=False,
            name=cluster,
        )
    )


def _umap_feature(
    fig: Any, go: Any, view: pd.DataFrame, expr: pd.DataFrame, cluster: str
) -> None:
    """Colour the UMAP by per-cell marker expression (teal low->high ramp).

    ``expr`` is (cell_id, value). Merged onto the view by cell_id so only cells
    with an exported value are coloured; the selected cluster is outlined on top.
    """
    merged = view.merge(expr, on="cell_id", how="left")
    vals = merged["value"]
    has_vals = bool(vals.notna().any())
    fig.add_trace(
        go.Scattergl(
            x=merged["umap_1"],
            y=merged["umap_2"],
            mode="markers",
            marker=dict(
                size=2.8,
                color=vals,
                colorscale=_EXPR_COLORSCALE,
                cmin=float(vals.min()) if has_vals else 0.0,
                cmax=float(vals.max()) if has_vals else 1.0,
                opacity=0.85,
                showscale=False,
            ),
            hoverinfo="skip",
            showlegend=False,
            name="expression",
        )
    )
    # Outline the selected cluster so it stays legible over the feature colouring.
    sel = merged[merged["cluster"] == cluster]
    fig.add_trace(
        go.Scattergl(
            x=sel["umap_1"],
            y=sel["umap_2"],
            mode="markers",
            marker=dict(
                size=3.4,
                color="rgba(0,0,0,0)",
                line=dict(width=0.7, color=_HIGHLIGHT_LINE),
            ),
            hoverinfo="skip",
            showlegend=False,
            name=f"{cluster} outline",
        )
    )


# --------------------------------------------------------------------------- #
# View 3 — Density (precomputed, area-normalized hex-bins; tissue image)
# --------------------------------------------------------------------------- #
def _render_density(gene: str) -> None:
    """Precomputed transcript density for ``gene`` as an area-normalized map.

    The global 25 / 50 / 100 µm bin control (``_render_bin_control``) picks which
    *precomputed* frame this reads; the colour is that frame's ``density``
    (transcripts / µm²) so bins are comparable across sizes. If the gene has no
    precomputed density frame, show an honest placeholder — never a synthesized
    field. The tissue aspect ratio is locked by ``_tissue_layout`` so a coarse
    bin never stretches the slide.
    """
    st = _st()
    go = _go()

    # Bin size is a single GLOBAL viewing control (``_render_bin_control``, drawn
    # once above the gene rows), so this panel just reads it — no per-panel control
    # to offset the density plot from its feature-UMAP neighbour.
    bin_um = state.get_bin_um()
    try:
        hb = da.hexbins(gene, bin_um)
    except FileNotFoundError:
        _empty_panel(f"{gene}: density not precomputed", height=_TISSUE_HEIGHT)
        return
    except Exception:
        _empty_panel(f"{gene}: density unavailable", height=_TISSUE_HEIGHT)
        return

    if hb is None or hb.empty:
        _empty_panel(f"{gene}: density not precomputed", height=_TISSUE_HEIGHT)
        return

    dens = hb["density"]
    fig = go.Figure(layout=_tissue_layout(go))
    # Square markers on the precomputed bin centres. Marker size scales with the
    # bin (bigger bins -> bigger tiles) purely for legibility; the COLOUR is the
    # area-normalized density and is the only thing that encodes signal.
    tile = _DENSITY_TILE_PX.get(int(bin_um), 7)
    fig.add_trace(
        go.Scattergl(
            x=hb["hx"],
            y=hb["hy"],
            mode="markers",
            marker=dict(
                size=tile,
                symbol="square",
                color=dens,
                colorscale=_DENSITY_COLORSCALE,
                cmin=0.0,
                cmax=float(dens.max()) if dens.notna().any() else 1.0,
                opacity=0.92,
                showscale=False,
            ),
            customdata=hb[["count", "density"]].to_numpy(),
            hovertemplate=(
                "count %{customdata[0]}<br>"
                "density %{customdata[1]:.4f} tx/µm²<extra></extra>"
            ),
            showlegend=False,
            name="density",
        )
    )

    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    _legend_line(
        f'{bin_um} µm bins · tx/µm² · low <span class="grad"></span> high'
    )


# --------------------------------------------------------------------------- #
# View 4 — Grouped violin (per-cell expression across all nine clusters)
# --------------------------------------------------------------------------- #
def _render_violin(genes: list[str], cluster: str) -> None:
    """Per-cell expression distribution of each selected gene across all nine
    clusters, one violin per cluster in cluster colour.

    Abstract chart (a distribution needs its axes), so it starts from
    ``_base_layout`` and then *re-enables* the axes. Data is exactly what the prep
    step exported: for each gene, ``expr_by_cluster`` joins the gene's per-cell
    value onto the authoritative cluster labels — this module derives nothing.
    ``points=False`` keeps 40k-cell violins light; ``meanline_visible`` shows the
    mean Plotly draws from those same values (no invented statistic). Multiple
    genes are grouped side by side per cluster (``violinmode="group"``); a single
    gene reads as one violin per cluster in that cluster's colour.
    """
    st = _st()
    go = _go()

    # Only plot genes that actually have exported per-cell expression; the rest
    # are honestly omitted (never faked).
    frames: list[tuple[str, pd.DataFrame]] = []
    for g in genes:
        by = da.expr_by_cluster(g)
        if by is not None and not by.empty:
            frames.append((g, by))

    if not frames:
        _empty_panel(
            "no per-cell expression precomputed for the selected marker(s)",
            height=_VIOLIN_HEIGHT,
        )
        return

    single = len(frames) == 1
    fig = go.Figure(layout=_base_layout(go, showlegend=not single, height=_VIOLIN_HEIGHT))
    # Re-enable axes: a distribution is a chart, not a map. x = clusters (c1..c9),
    # y = expression. Keep it light — a hairline y grid, category x in c1..c9 order.
    fig.update_layout(
        xaxis=dict(
            visible=True,
            showticklabels=True,   # cluster ids on the x-axis so each violin reads
            ticks="outside",
            showgrid=False,
            zeroline=False,
            type="category",
            categoryorder="array",
            categoryarray=list(CLUSTER_ORDER),
            tickfont=dict(family="IBM Plex Mono, monospace", size=10),
        ),
        yaxis=dict(
            visible=True,
            showgrid=True,
            gridcolor="#EEF0F2",
            zeroline=False,
            title=dict(
                text="expression",
                font=dict(family="IBM Plex Mono, monospace", size=10),
            ),
            tickfont=dict(family="IBM Plex Mono, monospace", size=10),
        ),
        violinmode="group",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(family="IBM Plex Mono, monospace", size=10),
        ),
    )

    # Fixed per-gene colours for the grouped (multi-gene) case so each gene keeps
    # one legend colour across every cluster. For a single gene we colour each
    # violin by its own cluster (the familiar cluster palette).
    for gi, (gene, by) in enumerate(frames):
        present = [c for c in CLUSTER_ORDER if c in set(by["cluster"].unique())]
        gene_color = fmt.cluster_color(CLUSTER_ORDER[gi % len(CLUSTER_ORDER)])
        first_present = present[0] if present else None
        for c in present:
            vals = by.loc[by["cluster"] == c, "value"]
            color = fmt.cluster_color(c) if single else gene_color
            fig.add_trace(
                go.Violin(
                    x=[c] * len(vals),
                    y=vals,
                    name=gene,
                    legendgroup=gene,
                    scalegroup=gene,
                    # one legend entry per gene (only on its first cluster trace)
                    showlegend=(not single) and (c == first_present),
                    line=dict(color=color, width=1),
                    fillcolor=color,
                    opacity=0.75 if single else 0.55,
                    points=False,
                    meanline_visible=True,
                    hovertemplate=f"{gene} · {c}<extra></extra>",
                )
            )

    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    if single:
        _legend_line(
            f'selected cluster <b style="color:{fmt.cluster_color(cluster)}">'
            f"{cluster}</b> highlighted · one violin per cluster"
        )
    else:
        _legend_line("violins grouped by gene · one per cluster")


# --------------------------------------------------------------------------- #
# Global density bin control (one control for every density panel)
# --------------------------------------------------------------------------- #
def _render_bin_control() -> None:
    """One 25 / 50 / 100 µm density-bin selector for the whole grid.

    Bin size is a global viewing control (it picks which precomputed frame every
    density panel reads — it never re-bins a value). Rendered once, above the gene
    rows, it reads fresh in the same run, so all density panels update together and
    each stays aligned with its feature-UMAP neighbour.
    """
    st = _st()
    sizes = list(state.BIN_SIZES_UM)
    current = state.get_bin_um()
    idx = sizes.index(current) if current in sizes else sizes.index(state.DEFAULT_BIN_UM)

    label_col, radio_col = st.columns([0.6, 0.4], vertical_alignment="center")
    with label_col:
        st.markdown(
            '<div class="ptitle">Transcript density · bin size</div>',
            unsafe_allow_html=True,
        )
    with radio_col:
        picked = st.radio(
            "Density bin size (µm)",
            options=sizes,
            index=idx,
            format_func=lambda b: f"{b} µm",
            horizontal=True,
            label_visibility="collapsed",
            key="bin_um_global",
        )
    # Read fresh in the same run — the density panels below pick it up immediately.
    state.set_bin_um(picked)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_spatial_stage(cluster: str) -> None:
    """Render the fixed spatial grid for ``cluster``.

    The grid is:
      * Row 1 — cluster cell map | cluster UMAP (always on, the context row).
      * One small-multiple row per selected marker — density | feature UMAP.
      * A full-width grouped violin across all nine clusters (selected markers).
      * A placeholder inviting a pick when no marker is selected.

    Reads state via ``ui.state``:
      * ``active_markers()`` — the selected markers, capped for small-multiples.
      * ``get_bin_um()``     — which precomputed density frame (density panels).

    Every control here is a viewing control. Selecting markers or changing the
    bin changes the picture only — this function never calls a verdict and never
    derives a statistic.
    """
    st = _st()
    _inject_stage_css()

    # Row 1 — cluster context: cell map (tissue) beside cluster UMAP (abstract).
    r1_left, r1_right = st.columns(2, gap="medium")
    with r1_left:
        _panel_title("Cell map <span style='color:var(--faint)'>· tissue</span>")
        _render_cell_map(cluster)
    with r1_right:
        _render_umap(cluster, feature=False)

    markers = state.active_markers()

    if not markers:
        _empty_panel(
            "select a marker in the evidence table to see its transcript density, "
            "feature UMAP, and across-cluster distribution",
            height=140,
        )
        return

    # Breathing room between the cell-level row and the transcript hex-bin rows.
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

    # Single global density bin control (one row, above all density panels).
    _render_bin_control()

    # One small-multiple row per selected marker: density (tissue) | feature UMAP.
    for gene in markers:
        g_left, g_right = st.columns(2, gap="medium")
        with g_left:
            _panel_title(f"<b>{gene}</b> density")
            _render_density(gene)
        with g_right:
            _panel_title(f"<b>{gene}</b> feature UMAP")
            _render_umap(cluster, feature=True, gene=gene)

    # Full-width grouped violin across all nine clusters for the selected set.
    _panel_title("Expression across clusters")
    _render_violin(markers, cluster)


__all__ = ["render_spatial_stage"]
