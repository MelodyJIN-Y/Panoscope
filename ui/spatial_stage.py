"""The spatial stage: three linked Plotly views driven by one pinned marker.

Public entry point: ``render_spatial_stage(cluster)``.

The stage renders exactly one of three linked views, chosen by a view toggle:

  * ``cell_map`` (default) — every segmented cell at its tissue coordinate,
    coloured by cluster, with the selected cluster brought forward. "Did the
    call land in the right cells."
  * ``umap`` — every cell in expression space, coloured by cluster; when a
    marker is pinned (or hovered) it recolours by that marker's per-cell
    expression instead. "Is the cluster a clean island or bleeding into a
    neighbour."
  * ``density`` — the precomputed, area-normalized transcript hex-bins for the
    pinned marker, before cell calling. "Is the raw signal really there." The
    25 / 50 / 100 µm bin control lives here and here only.

Grounding invariant (the load-bearing one): every control on this stage is a
**viewing** control. The view toggle, the bin size, and the pinned marker change
*the picture*, never a value. The bin control selects which *precomputed* density
frame to draw (``ui.data_access.hexbins`` reads a different file per bin) — it
never re-bins, and the colour scale is the frame's already-area-normalized
``density`` column so a coarser bin cannot look hotter than a finer one. Nothing
in this module computes a statistic or a confidence.

State is read through ``ui.state`` (never a raw session-state key): the active
view, the pinned/hovered marker (``active_marker()`` = hover preview, else pin),
and the bin size. When a pinned marker has no precomputed density or expression
frame, the relevant view shows an honest "not precomputed for this marker"
placeholder — it never fakes a value.

Streamlit and Plotly are imported lazily inside the render functions so the
module imports cleanly with no server running.
"""

from __future__ import annotations

from typing import Any

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

_PLOT_HEIGHT = 460
_PLOT_BG = "#FFFFFF"

_VIEW_LABELS: dict[str, str] = {
    "cell_map": "Cell map",
    "umap": "UMAP",
    "density": "Density",
}
_VIEW_CAPTION: dict[str, str] = {
    "cell_map": "segmented cells at tissue coordinates · selected cluster brought forward",
    "umap": "expression space · coloured by cluster, or by pinned-marker expression",
    "density": "raw transcripts · hex-binned before cell calling · area-normalized",
}

# Density tile size in px per bin — purely cosmetic legibility; COLOUR is the
# only thing that encodes signal.
_DENSITY_TILE_PX: dict[int, int] = {25: 6, 50: 9, 100: 14}


# --------------------------------------------------------------------------- #
# Small lazy-import helpers (keep the module import-safe with no server)
# --------------------------------------------------------------------------- #
def _st() -> Any:
    import streamlit as st

    return st


# --------------------------------------------------------------------------- #
# Stage-local styling. These classes (.pinbar / .pempty / .plegend / .g) are
# used only by this stage and are not part of the shared theme, so the stage
# ships them itself. Injected once per session (idempotent flag) so reruns and
# viewing-control toggles never restyle. Colours reference the theme's CSS
# custom properties (--accent, --faint, …) so it stays in the design system.
# --------------------------------------------------------------------------- #
_STAGE_CSS = """
.pinbar {
  font-family: var(--mono, ui-monospace, monospace); font-size: 11px;
  color: var(--muted, #606A73); margin: 6px 0 4px;
}
.pinbar .g {
  font-weight: 600; color: var(--accent, #0F7B87);
  background: var(--accent-soft, #EBF5F6); padding: 1px 6px; border-radius: 5px;
}
.plegend {
  font-family: var(--mono, ui-monospace, monospace); font-size: 11px;
  color: var(--muted, #606A73); margin-top: 6px; line-height: 1.5;
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


def _base_layout(go_mod: Any, *, showlegend: bool) -> Any:
    """A shared, chrome-light Plotly layout (no axis ticks — this is a map)."""
    axis = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        visible=False,
    )
    return go_mod.Layout(
        height=_PLOT_HEIGHT,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=_PLOT_BG,
        plot_bgcolor=_PLOT_BG,
        xaxis=axis,
        yaxis=axis,
        showlegend=showlegend,
        font=dict(family="Space Grotesk, system-ui, sans-serif"),
        dragmode="pan",
    )


def _plot_config() -> dict[str, Any]:
    """Plotly display config: keep the toolbar minimal."""
    return {
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "toggleSpikelines",
        ],
    }


# --------------------------------------------------------------------------- #
# View toolbar — the toggle + (density-only) bin control. Viewing controls only.
# --------------------------------------------------------------------------- #
def _render_view_toolbar(active_view: str) -> None:
    """Render the view toggle and, for density, the µm bin control.

    Both write only their viewing-control key (``spatial_view`` / ``bin_um``)
    via ``ui.state`` setters. Neither ever calls a verdict.
    """
    st = _st()
    left, right = st.columns([3, 2])

    with left:
        options = list(state.SPATIAL_VIEWS)
        try:
            idx = options.index(active_view)
        except ValueError:
            idx = 0
        picked = st.radio(
            "Spatial view",
            options=options,
            index=idx,
            format_func=lambda v: _VIEW_LABELS.get(v, v),
            horizontal=True,
            label_visibility="collapsed",
            key="spatial_view_toggle",
        )
        if picked != active_view:
            state.set_spatial_view(picked)  # picture only, no recompute
            st.rerun()

    with right:
        if active_view == "density":
            current = state.get_bin_um()
            sizes = list(state.BIN_SIZES_UM)
            try:
                bidx = sizes.index(current)
            except ValueError:
                bidx = sizes.index(state.DEFAULT_BIN_UM)
            picked_bin = st.radio(
                "Bin size (µm)",
                options=sizes,
                index=bidx,
                format_func=lambda b: f"{b} µm",
                horizontal=True,
                label_visibility="collapsed",
                key="bin_um_toggle",
            )
            if picked_bin != current:
                state.set_bin_um(picked_bin)  # selects a precomputed frame; no re-bin
                st.rerun()


def _render_pinbar(active_view: str) -> None:
    """One mono line naming what is pinned/previewed (mirrors the wireframe pinbar)."""
    st = _st()
    pinned = state.get_pinned_marker()
    hover = state.get_hover_marker()
    if pinned:
        st.markdown(
            f'<div class="pinbar">showing <span class="g">{pinned}</span> '
            f"across the spatial views · click the pin again in the evidence "
            f"table to unpin</div>",
            unsafe_allow_html=True,
        )
    elif hover:
        st.markdown(
            f'<div class="pinbar">preview <span class="g">{hover}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        hint = (
            "pin a marker in the evidence table to colour these views"
            if active_view != "cell_map"
            else "cells coloured by cluster · pin a marker to switch to its signal"
        )
        st.markdown(
            f'<div class="pinbar"><span style="color:var(--faint)">{hint}</span></div>',
            unsafe_allow_html=True,
        )


def _empty_panel(message: str) -> None:
    """Render the honest empty state (never a faked plot)."""
    st = _st()
    st.markdown(
        f'<div class="pempty" style="height:{_PLOT_HEIGHT}px">{message}</div>',
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
# View 1 — Cell map (default)
# --------------------------------------------------------------------------- #
def _render_cell_map(cluster: str) -> None:
    """Scattergl of all cells at tissue x/y, coloured by cluster; selected cluster
    brought forward (full colour + thin outline), the rest faded to grey.

    Pure viewing surface: it draws ``cells_df()`` (already cached) and the
    selected cluster comes straight from ``cluster``. No value is computed.
    """
    st = _st()
    go = _go()

    cells = da.cells_df()
    if cells.empty:
        _empty_panel("no cell coordinates available")
        return

    sel_mask = cells["cluster"] == cluster
    view = _downsample_bg(cells, sel_mask, _MAX_BG_POINTS)
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]

    fig = go.Figure(layout=_base_layout(go, showlegend=False))

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
    # Foreground: the selected cluster in its own colour, brought forward.
    sel_color = fmt.cluster_color(cluster)
    fig.add_trace(
        go.Scattergl(
            x=sel["x"],
            y=sel["y"],
            mode="markers",
            marker=dict(
                size=3.4,
                color=sel_color,
                opacity=0.95,
                line=dict(width=0.4, color=_HIGHLIGHT_LINE),
            ),
            customdata=sel["cell_id"],
            hovertemplate="cell %{customdata}<extra></extra>",
            showlegend=False,
            name=cluster,
        )
    )
    # A tissue image reads top-down; flip y so the map matches the slide.
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    ct = _cell_type_label(cluster)
    _legend_line(
        f'cells coloured by cluster · <b style="color:{sel_color}">{cluster} '
        f"{ct}</b> brought forward · {len(sel):,} of {len(cells):,} cells shown"
    )


# --------------------------------------------------------------------------- #
# View 2 — UMAP (cluster colour, or pinned-marker expression)
# --------------------------------------------------------------------------- #
def _render_umap(cluster: str) -> None:
    """Scattergl in UMAP space. Coloured by cluster when nothing is pinned; when a
    marker is pinned/hovered it recolours by that marker's per-cell expression
    (``marker_expr_col``) if exported, else falls back to cluster colour with a
    note. Selected cluster is always outlined.
    """
    st = _st()
    go = _go()

    umap = da.umap_df()
    if umap.empty:
        _empty_panel("no UMAP embedding available")
        return

    gene = state.active_marker()  # hover preview, else pinned
    expr = da.marker_expr_col(gene) if gene else None

    sel_mask = umap["cluster"] == cluster
    view = _downsample_bg(umap, sel_mask, _MAX_BG_POINTS)

    fig = go.Figure(layout=_base_layout(go, showlegend=False))

    if gene and expr is not None:
        _umap_feature(fig, go, view, expr, cluster)
        st.plotly_chart(fig, use_container_width=True, config=_plot_config())
        _legend_line(
            f'<b>{gene}</b> expression · low <span class="grad"></span> high · '
            f"selected cluster {cluster} outlined"
        )
        return

    # Cluster-coloured UMAP (default, or marker pinned but not exported)
    _umap_by_cluster(fig, go, view, cluster)
    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    if gene and expr is None:
        _legend_line(
            f'<span style="color:var(--absent)">{gene}: per-cell expression not '
            f"precomputed</span> · showing cluster colour instead"
        )
    else:
        sel_color = fmt.cluster_color(cluster)
        _legend_line(
            f"cells coloured by cluster · "
            f'<b style="color:{sel_color}">{cluster}</b> outlined'
        )


def _umap_by_cluster(fig: Any, go: Any, view: pd.DataFrame, cluster: str) -> None:
    """Add per-cluster scattergl traces (one per cluster, in c1..c9 order)."""
    present = set(view["cluster"].unique())
    for c in [c for c in CLUSTER_ORDER if c in present]:
        sub = view[view["cluster"] == c]
        is_sel = c == cluster
        fig.add_trace(
            go.Scattergl(
                x=sub["umap_1"],
                y=sub["umap_2"],
                mode="markers",
                marker=dict(
                    size=3.2 if is_sel else 2.2,
                    color=fmt.cluster_color(c),
                    opacity=0.9 if is_sel else 0.5,
                    line=dict(width=0.5, color=_HIGHLIGHT_LINE)
                    if is_sel
                    else dict(width=0),
                ),
                name=c,
                hovertemplate=f"{c}<extra></extra>",
                showlegend=False,
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
# View 3 — Density (precomputed, area-normalized hex-bins for the pinned marker)
# --------------------------------------------------------------------------- #
def _render_density(cluster: str) -> None:
    """Precomputed transcript density for the pinned marker as an area-normalized
    map. The bin control picks which precomputed frame to read; the colour is the
    frame's ``density`` (transcripts / µm²) so bins are comparable across sizes.

    If no marker is pinned, or the pinned marker has no precomputed density frame,
    show an honest placeholder — never a synthesized field.
    """
    st = _st()
    go = _go()

    gene = state.active_marker()
    if not gene:
        _empty_panel("pin a marker to see its transcript density")
        return

    bin_um = state.get_bin_um()
    try:
        hb = da.hexbins(gene, bin_um)
    except FileNotFoundError:
        _empty_panel(f"{gene}: density not precomputed for this marker")
        return
    except Exception:
        _empty_panel(f"{gene}: density unavailable")
        return

    if hb is None or hb.empty:
        _empty_panel(f"{gene}: density not precomputed for this marker")
        return

    dens = hb["density"]
    fig = go.Figure(layout=_base_layout(go, showlegend=False))
    # Square markers on the precomputed bin centres. Marker size scales with the
    # bin (bigger bins -> bigger tiles) purely for legibility; the COLOUR is the
    # area-normalized density and is the only thing that encodes signal.
    tile = _DENSITY_TILE_PX.get(int(bin_um), 9)
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
    fig.update_yaxes(autorange="reversed")

    st.plotly_chart(fig, use_container_width=True, config=_plot_config())
    _legend_line(
        f"<b>{gene}</b> transcripts · {bin_um} µm bins · area-normalized "
        f'(tx/µm²) · low <span class="grad"></span> high'
    )


# --------------------------------------------------------------------------- #
# Small display helper
# --------------------------------------------------------------------------- #
def _cell_type_label(cluster: str) -> str:
    """Best-effort cell-type label for a caption (never computes; reads the key).

    Falls back to an empty string if the key is unavailable, so a caption never
    crashes the stage.
    """
    try:
        from agent.config import CLUSTER_KEY

        return CLUSTER_KEY.get(cluster, {}).get("cell_type", "")
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_spatial_stage(cluster: str) -> None:
    """Render the spatial stage for ``cluster``: the view toggle + pin readout +
    the one active linked view (cell map / UMAP / density).

    Reads state via ``ui.state``:
      * ``get_spatial_view()`` — which of the three views to draw (default cell_map)
      * ``active_marker()``    — hover preview else pinned marker (drives UMAP/density)
      * ``get_bin_um()``       — which precomputed density frame (density view only)

    Every control here is a viewing control. Switching view, changing the bin, or
    pinning a marker changes the picture only — this function never calls a verdict
    and never derives a statistic.
    """
    st = _st()
    _inject_stage_css()

    st.markdown(
        '<p class="pano-sect">Spatial evidence '
        '<span class="r">a pinned marker drives all three views</span></p>',
        unsafe_allow_html=True,
    )

    active_view = state.get_spatial_view()
    _render_view_toolbar(active_view)
    _render_pinbar(active_view)

    st.caption(_VIEW_CAPTION.get(active_view, ""))

    if active_view == "cell_map":
        _render_cell_map(cluster)
    elif active_view == "umap":
        _render_umap(cluster)
    elif active_view == "density":
        _render_density(cluster)
    else:  # defensive: unknown view falls back to the default map
        _render_cell_map(cluster)


__all__ = ["render_spatial_stage"]
