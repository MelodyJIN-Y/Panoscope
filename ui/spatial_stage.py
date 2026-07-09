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

import numpy as np
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
# Plasma (matplotlib) — HIGHER = BRIGHTER, LOWER = DARKER: low = deep purple,
# high = bright yellow (the viridis "plasma" ramp; hot spots read bright). Shared
# by the transcript-density hex bins, the feature-UMAP, and the dotplot so the
# spatial value plots read on one perceptual ramp. A sqrt transform on the value
# (applied where each is drawn, matching trans="sqrt") lifts the low end so sparse
# signal stays visible.
_PLASMA = [
    [0.0, "#0D0887"],
    [0.2, "#7E03A8"],
    [0.4, "#CC4778"],
    [0.6, "#F1605D"],
    [0.8, "#FCA636"],
    [1.0, "#F0F921"],
]
_DENSITY_COLORSCALE = _PLASMA
_EXPR_COLORSCALE = _PLASMA    # feature UMAP + dotplot share the same ramp

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

# Density is drawn as data-coordinate cells (go.Heatmap): each bin is exactly
# bin_um wide on the tissue, so tile size tracks the µm control instead of a fixed
# pixel size that could never line up with the selected bin.


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
.ctrl-lbl {
  font-family: var(--mono, ui-monospace, monospace); font-size: 11px;
  color: var(--faint, #9AA3AB); text-transform: uppercase; letter-spacing: .06em;
  margin: 0; line-height: 1; transform: translateY(-5px);
}
/* Bin control: put the label and the radio on one baseline. Center the row and
   drop the radio's reserved (collapsed) label space so nothing pushes it down. */
div[class*="st-key-pano_binctl"] [data-testid="stHorizontalBlock"] { align-items: center; }
div[class*="st-key-pano_binctl"] [data-testid="stRadio"] > label { display: none; }
div[class*="st-key-pano_binctl"] [data-testid="stRadio"] { margin: 0; }
div[class*="st-key-pano_binctl"] [data-testid="stRadio"] [role="radiogroup"] { align-items: center; }
.stage-legend {
  font-family: var(--mono, ui-monospace, monospace); font-size: 10.5px;
  color: var(--faint, #9AA3AB); line-height: 1.5;
  display: flex; align-items: center; justify-content: flex-end; gap: 8px;
}
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


def _fmt_val(v: float) -> str:
    """Format a colorbar tick value with magnitude-appropriate precision."""
    if v == 0:
        return "0"
    av = abs(v)
    if av >= 100:
        return f"{v:.0f}"
    if av >= 1:
        return f"{v:.1f}"
    if av >= 0.01:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _colorbar(tickvals: list, ticktext: list, title: str) -> dict:
    """A compact right-edge colorbar with explicit tick positions + labels."""
    return dict(
        title=dict(
            text=title, side="right",
            font=dict(family="IBM Plex Mono, monospace", size=9),
        ),
        thickness=7,
        len=0.62,
        x=1.0,
        xpad=3,
        outlinewidth=0,
        tickvals=tickvals,
        ticktext=ticktext,
        tickfont=dict(family="IBM Plex Mono, monospace", size=8),
    )


def _sqrt_colorbar(vmax: float, title: str) -> dict:
    """Colorbar for a sqrt-mapped value: tick POSITIONS on the sqrt scale, LABELS
    in real units. Three ticks (0, mid, max) keep it legible in a small panel."""
    vmax = float(vmax) if vmax and vmax > 0 else 1.0
    orig = [0.0, vmax * 0.5, vmax]
    return _colorbar([v ** 0.5 for v in orig], [_fmt_val(v) for v in orig], title)


def _count_colorbar(dmax: float, count_max: float) -> dict:
    """Colorbar for the density panel, LABELLED in bin counts.

    The colour stays the sqrt of the area-normalized density (so switching bin
    size never makes coarser bins look uniformly hotter — a confident-floor
    invariant), but the ticks read the intuitive per-bin transcript COUNT. Within
    one bin size count is proportional to density (uniform bin area), so tick
    positions on the sqrt-density axis map exactly to count labels.
    """
    dmax = float(dmax) if dmax and dmax > 0 else 1.0
    dens_ticks = [0.0, dmax * 0.5, dmax]
    count_ticks = [0.0, count_max * 0.5, count_max]
    return _colorbar(
        [d ** 0.5 for d in dens_ticks],
        [f"{int(round(c)):,}" for c in count_ticks],
        "count",
    )


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
# Cell-type labels for hover (read the authoritative key; never computes a value)
# --------------------------------------------------------------------------- #
def _cluster_celltype_map() -> dict:
    """Map each cluster id to its human cell-type label for hover tooltips.

    Read straight from the authoritative ``CLUSTER_KEY``; unassigned cells (blank
    cluster) read as ``unassigned``. Nothing is computed.
    """
    from agent.config import CLUSTER_KEY, CLUSTER_ORDER

    labels = {
        c: CLUSTER_KEY.get(c, {}).get("cell_type", c).replace("_", " ")
        for c in CLUSTER_ORDER
    }
    labels[""] = "unassigned"
    return labels


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
    view = view.assign(
        cell_type=view["cluster"].map(_cluster_celltype_map()).fillna("unassigned")
    )
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]

    fig = go.Figure(layout=_tissue_layout(go))

    # Background: all other clusters, muted. Hovering any cell shows its cell type.
    fig.add_trace(
        go.Scattergl(
            x=bg["x"],
            y=bg["y"],
            mode="markers",
            marker=dict(size=1.6, color=_FADE_COLOR, opacity=0.55),
            customdata=bg["cell_type"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="other clusters",
        )
    )
    # Foreground: the selected cluster filled with its own palette colour (no
    # outline — just the filled points). Hover shows the cell type.
    sel_color = fmt.cluster_color(cluster)
    fig.add_trace(
        go.Scattergl(
            x=sel["x"],
            y=sel["y"],
            mode="markers",
            marker=dict(size=2.4, color=sel_color, opacity=0.95),
            customdata=sel["cell_type"],
            hovertemplate="%{customdata}<extra></extra>",
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
        # Legend is shared once in the control row above (no per-panel legend).
        st.plotly_chart(fig, use_container_width=True, config=_plot_config())
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
    treatment so the two Row-1 panels read the same way. Hovering any cell shows
    its cell type (read from the authoritative key), not the opaque cell id."""
    view = view.assign(
        cell_type=view["cluster"].map(_cluster_celltype_map()).fillna("unassigned")
    )
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]
    fig.add_trace(
        go.Scattergl(
            x=bg["umap_1"],
            y=bg["umap_2"],
            mode="markers",
            marker=dict(size=1.6, color=_FADE_COLOR, opacity=0.5),
            customdata=bg["cell_type"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="other clusters",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=sel["umap_1"],
            y=sel["umap_2"],
            mode="markers",
            marker=dict(size=2.4, color=fmt.cluster_color(cluster), opacity=0.95),
            customdata=sel["cell_type"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name=cluster,
        )
    )


def _umap_feature(
    fig: Any, go: Any, view: pd.DataFrame, expr: pd.DataFrame, cluster: str
) -> None:
    """Colour the UMAP by per-cell marker expression (plasma low->high ramp).

    ``expr`` is (cell_id, value). A feature plot in two layers so higher=brighter
    reads cleanly on a light panel and there is no dark blob:

    * NON-expressing cells (value <= 0 or no exported value) are a light-grey base
      that recedes — not dark purple, which would swamp the plot.
    * EXPRESSING cells (value > 0) are drawn ON TOP, sqrt-transformed and coloured
      plasma (dark = low, bright = high), in ascending order so the brightest
      cells land last and stay visible.

    No cluster outline: the Row-1 UMAP already shows where the cluster sits, and a
    dark outline over a large cluster collapses into a black mass.
    """
    merged = view.merge(expr, on="cell_id", how="left")
    vals = merged["value"]
    is_pos = vals > 0
    off = merged[~is_pos]
    pos = merged[is_pos].sort_values("value", ascending=True)

    # Base: non-expressing cells, muted grey so they recede into the panel.
    fig.add_trace(
        go.Scattergl(
            x=off["umap_1"],
            y=off["umap_2"],
            mode="markers",
            marker=dict(size=1.8, color=_FADE_COLOR, opacity=0.5),
            hoverinfo="skip",
            showlegend=False,
            name="not expressing",
        )
    )
    # Expressing cells on top, plasma sqrt (higher = brighter) with a numeric colorbar.
    if not pos.empty:
        pos_sqrt = pos["value"].clip(lower=0) ** 0.5
        fig.add_trace(
            go.Scattergl(
                x=pos["umap_1"],
                y=pos["umap_2"],
                mode="markers",
                marker=dict(
                    size=2.4,
                    color=pos_sqrt,
                    colorscale=_EXPR_COLORSCALE,
                    cmin=0.0,
                    cmax=float(pos_sqrt.max()),
                    opacity=0.92,
                    showscale=True,
                    colorbar=_sqrt_colorbar(float(pos["value"].max()), "expr"),
                ),
                hoverinfo="skip",
                showlegend=False,
                name="expression",
            )
        )


# --------------------------------------------------------------------------- #
# View 3 — Density (precomputed, area-normalized hex-bins; tissue image)
# --------------------------------------------------------------------------- #
def _render_density(gene: str) -> None:
    """Precomputed transcript density for ``gene`` as an area-normalized map.

    The global 25 / 50 / 100 µm bin control (``_render_density_controls``) picks which
    *precomputed* frame this reads; the colour is that frame's ``density``
    (transcripts / µm²) so bins are comparable across sizes. Each bin is rendered as
    a data-coordinate ``go.Heatmap`` cell exactly ``bin_um`` wide, so the tiles tile
    the slide and their size visibly matches the µm the biologist selected (a fixed
    pixel marker never could). If the gene has no precomputed density frame, show an
    honest placeholder — never a synthesized field. The tissue aspect ratio is locked
    by ``_tissue_layout`` so a coarse bin never stretches the slide.
    """
    st = _st()
    go = _go()

    # Bin size is a single GLOBAL viewing control (``_render_density_controls``, drawn
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
    # sqrt transform on the density for the colour mapping (trans="sqrt"): it lifts
    # the low end so sparse bins stay visible. The hover still shows the REAL count
    # and density — the transform is display-only and never changes a value.
    dens_sqrt = (dens.clip(lower=0) ** 0.5).to_numpy(dtype="float64")
    cmax = float(dens_sqrt.max()) if dens_sqrt.size and np.isfinite(dens_sqrt).any() else 1.0

    # Draw each bin as a DATA-COORDINATE cell (go.Heatmap), not a fixed-pixel marker:
    # a "50 µm" bin is exactly 50 µm wide on the tissue, so the tiles tile the slide
    # and the picture visibly scales with the 25/50/100 µm control (a pixel-sized
    # marker could never line up with the µm the biologist picked). The sparse
    # precomputed bins are placed onto their regular grid; absent bins stay NaN and
    # render transparent (hoverongaps=False) so the slide reads through the holes.
    hx = hb["hx"].to_numpy(dtype="float64")
    hy = hb["hy"].to_numpy(dtype="float64")
    x0, y0 = hx.min(), hy.min()
    nx = int(round((hx.max() - x0) / bin_um)) + 1
    ny = int(round((hy.max() - y0) / bin_um)) + 1
    gx = x0 + np.arange(nx) * bin_um           # column centres, µm
    gy = y0 + np.arange(ny) * bin_um           # row centres, µm
    ix = np.round((hx - x0) / bin_um).astype(int)
    iy = np.round((hy - y0) / bin_um).astype(int)

    z = np.full((ny, nx), np.nan)              # sqrt-density per cell (colour)
    z[iy, ix] = dens_sqrt
    cd = np.full((ny, nx, 2), np.nan)          # hover: real count + real density
    cd[iy, ix, 0] = hb["count"].to_numpy(dtype="float64")
    cd[iy, ix, 1] = dens.to_numpy(dtype="float64")

    fig = go.Figure(layout=_tissue_layout(go))
    fig.add_trace(
        go.Heatmap(
            x=gx,
            y=gy,
            z=z,
            customdata=cd,
            colorscale=_DENSITY_COLORSCALE,
            zmin=0.0,
            zmax=cmax,
            zsmooth=False,
            xgap=0,
            ygap=0,
            hoverongaps=False,
            hovertemplate=(
                "count %{customdata[0]:.0f}<br>"
                "density %{customdata[1]:.4f} tx/µm²<extra></extra>"
            ),
            colorbar=_count_colorbar(
                float(dens.max()) if dens.notna().any() else 1.0,
                float(hb["count"].max()) if hb["count"].notna().any() else 0.0,
            ),
            name="density",
        )
    )

    # No per-panel legend: the shared control+legend row above every gene row
    # carries the ramp and the units once (see _render_density_controls).
    st.plotly_chart(fig, use_container_width=True, config=_plot_config())


# --------------------------------------------------------------------------- #
# View 4 — Grouped violin (per-cell expression across all nine clusters)
# --------------------------------------------------------------------------- #
def _render_dotplot(genes: list[str], cluster: str) -> None:
    """Dot plot of each selected gene's expression across the nine clusters.

    The standard single-cell summary: for every (gene, cluster) a dot whose SIZE
    is the fraction of that cluster's cells expressing the gene (value > 0) and
    whose COLOUR is the mean per-cell expression. Both come straight from
    ``expr_by_cluster`` (the exported per-cell values joined to the authoritative
    cluster labels) — this derives nothing beyond a mean and a fraction of source
    values. x = clusters (c1..c9), y = the selected genes.
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

    # (gene, cluster) -> mean expression + % of cells expressing (value > 0),
    # straight from the exported per-cell values (a mean and a fraction; nothing
    # else is derived).
    rows: list[tuple[str, str, float, float]] = []
    for gene, by in frames:
        for c in CLUSTER_ORDER:
            vals = by.loc[by["cluster"] == c, "value"]
            if len(vals) == 0:
                continue
            rows.append(
                (gene, c, float(vals.mean()), float((vals > 0).mean()) * 100.0)
            )

    df = pd.DataFrame(rows, columns=["gene", "cluster", "mean", "pct"])
    gene_order = [g for g, _ in frames]
    y_order = list(reversed(gene_order))  # first-selected gene reads at the top

    max_px = 26.0
    pct_max = max(1.0, float(df["pct"].max()))
    sizeref = 2.0 * pct_max / (max_px ** 2)
    mean_max = float(df["mean"].max())
    # sqrt colour transform (trans="sqrt"): colour the dots by sqrt(mean), but keep
    # the colourbar labelled in REAL mean units so the axis stays honest.
    mean_sqrt = df["mean"].clip(lower=0) ** 0.5
    cmax_sqrt = float(mean_sqrt.max()) if mean_sqrt.notna().any() and mean_sqrt.max() > 0 else 1.0
    _bar_orig = [0.0, mean_max * 0.25, mean_max * 0.5, mean_max] if mean_max > 0 else [0.0, 1.0]
    _bar_tickvals = [v ** 0.5 for v in _bar_orig]
    _bar_ticktext = [f"{v:.1f}" for v in _bar_orig]
    height = max(160, 74 + len(gene_order) * 46)

    fig = go.Figure(layout=_base_layout(go, showlegend=False, height=height))
    fig.add_trace(
        go.Scatter(
            x=df["cluster"],
            y=df["gene"],
            mode="markers",
            marker=dict(
                size=df["pct"],
                sizemode="area",
                sizeref=sizeref,
                sizemin=3,
                color=mean_sqrt,
                colorscale=_EXPR_COLORSCALE,
                cmin=0.0,
                cmax=cmax_sqrt,
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="mean",
                        side="right",
                        font=dict(family="IBM Plex Mono, monospace", size=9),
                    ),
                    thickness=8,
                    len=0.7,
                    tickvals=_bar_tickvals,
                    ticktext=_bar_ticktext,
                    tickfont=dict(family="IBM Plex Mono, monospace", size=9),
                ),
                line=dict(width=0),
            ),
            customdata=df[["pct", "mean"]].to_numpy(),
            hovertemplate=(
                "%{y} · %{x}<br>%{customdata[0]:.0f}% expressing<br>"
                "mean %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        margin=dict(l=8, r=40, t=8, b=8),
        xaxis=dict(
            visible=True,
            showticklabels=True,
            ticks="outside",
            automargin=True,
            showgrid=True,
            gridcolor="#F1F3F4",
            zeroline=False,
            type="category",
            categoryorder="array",
            categoryarray=list(CLUSTER_ORDER),
            tickfont=dict(family="IBM Plex Mono, monospace", size=10),
        ),
        yaxis=dict(
            visible=True,
            showticklabels=True,
            automargin=True,
            showgrid=True,
            gridcolor="#F1F3F4",
            zeroline=False,
            type="category",
            categoryorder="array",
            categoryarray=y_order,
            tickfont=dict(family="IBM Plex Mono, monospace", size=11),
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    _legend_line("dot size = % of cells expressing · colour = mean expression")


# --------------------------------------------------------------------------- #
# Global density bin control (one control for every density panel)
# --------------------------------------------------------------------------- #
def _render_density_controls() -> None:
    """One tidy row for the whole density section: the bin-size selector beside a
    single shared colour legend.

    Bin size is ONE global viewing control — the same size is used for every
    selected gene (it picks which precomputed frame each density panel reads, never
    re-binning a value). Rendering it once here, with the legend, means the gene
    rows below carry no repeated control and no per-panel legend.
    """
    st = _st()
    sizes = list(state.BIN_SIZES_UM)
    current = state.get_bin_um()
    idx = sizes.index(current) if current in sizes else sizes.index(state.DEFAULT_BIN_UM)

    with st.container(key="pano_binctl"):
        lbl_col, radio_col, _spacer = st.columns(
            [0.13, 0.5, 0.37], vertical_alignment="center"
        )
        with lbl_col:
            st.markdown('<div class="ctrl-lbl">Bin size</div>', unsafe_allow_html=True)
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
    # Each panel carries its own numeric colorbar, so no shared text legend here.
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
        _panel_title("UMAP <span style='color:var(--faint)'>· clusters</span>")
        _render_umap(cluster, feature=False)

    markers = state.active_markers(cluster)

    if not markers:
        _empty_panel(
            "select a marker in the evidence table to see its transcript density, "
            "feature UMAP, and across-cluster distribution",
            height=140,
        )
        return

    # Breathing room between the cell-level row and the transcript hex-bin rows.
    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

    # Single tidy control+legend row (one bin size for every gene, shared legend).
    _render_density_controls()

    # One small-multiple row per selected marker: density (tissue) | feature UMAP.
    for gene in markers:
        g_left, g_right = st.columns(2, gap="medium")
        with g_left:
            _panel_title(f"<b>{gene}</b> transcript detections")
            _render_density(gene)
        with g_right:
            _panel_title(f"<b>{gene}</b> feature UMAP")
            _render_umap(cluster, feature=True, gene=gene)

    # Dot plot: expression of the selected genes across all nine clusters.
    _panel_title("Expression across clusters")
    _render_dotplot(markers, cluster)


__all__ = ["render_spatial_stage"]
