"""The spatial stage: a fixed grid of linked Plotly views for the selected markers.

Public entry point: ``render_spatial_stage(cluster)``. Row 1 is the cluster cell
map beside the cluster UMAP; then one density | feature-UMAP row per selected
marker; then a dot plot across clusters.

Every control here is a viewing control: the bin size picks a precomputed,
area-normalized frame and the selection picks genes; neither changes a value.
Tissue panels lock a 1:1 aspect. A missing frame renders a placeholder, never a
faked plot. Streamlit and Plotly are imported lazily.
"""


from __future__ import annotations

from panoscope.agent import config as cfg
from typing import Any, Optional

import numpy as np
import pandas as pd

from panoscope.ui import conditionality as cnd
from panoscope.ui import data_access as da
from panoscope.ui import format as fmt
from panoscope.ui import state

# --------------------------------------------------------------------------- #
# Constants — all purely cosmetic (viewing only)
# --------------------------------------------------------------------------- #
# Display-only cap on BACKGROUND points; the selected cluster is always drawn in full.
_MAX_BG_POINTS = 45_000

_FADE_COLOR = "#D3D8DC"        # non-selected cells on the cell map (muted grey)
# Plasma ramp (higher = brighter), shared by density, feature UMAP and dot plot.
# Values are sqrt-transformed where drawn so sparse signal stays visible.
_PLASMA = [
    [0.0, "#0D0887"],
    [0.2, "#7E03A8"],
    [0.4, "#CC4778"],
    [0.6, "#F1605D"],
    [0.8, "#FCA636"],
    [1.0, "#F0F921"],
]

# Panel heights; paired panels share one so a row's two columns line up.
_PLOT_HEIGHT = 300          # UMAP (abstract space — free ratio)
_TISSUE_HEIGHT = 300        # cell map / density (ratio-locked tissue image)
_VIOLIN_HEIGHT = 340        # full-width grouped violin
_PLOT_BG = "#FFFFFF"


# --------------------------------------------------------------------------- #
# Small lazy-import helpers (keep the module import-safe with no server)
# --------------------------------------------------------------------------- #
def _st() -> Any:
    import streamlit as st

    return st


# --------------------------------------------------------------------------- #
# Stage-local styling (theme custom properties).
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
.pempty {
  display: flex; align-items: center; justify-content: center; text-align: center;
  font-family: var(--mono, ui-monospace, monospace); font-size: 12px;
  color: var(--faint, #9AA3AB); background: var(--hair2, #EEF0F2);
  border: 1px dashed var(--hair, #E4E7EA); border-radius: 10px; padding: 0 24px;
}
"""


def _inject_stage_css() -> None:
    """Inject stage-local CSS on EVERY run: Streamlit drops elements not re-emitted,
    and the conditionality caption would lose its styling."""
    _st().markdown(f"<style>{_STAGE_CSS}</style>", unsafe_allow_html=True)


def _go() -> Any:
    import plotly.graph_objects as go

    return go


# --------------------------------------------------------------------------- #
# Shared Plotly layouts
# --------------------------------------------------------------------------- #
def _base_layout(
    go_mod: Any, *, showlegend: bool, height: int = _PLOT_HEIGHT, equal_aspect: bool = False
) -> Any:
    """A shared, chrome-light Plotly layout with hidden axes. ``equal_aspect`` locks
    1:1 scaling so an embedding keeps its shape."""
    axis = dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
        visible=False,
        fixedrange=True,   # view locked — no zoom/pan (hover still works)
    )
    yaxis = dict(axis)
    if equal_aspect:
        yaxis.update(scaleanchor="x", scaleratio=1)
    return go_mod.Layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=_PLOT_BG,
        plot_bgcolor=_PLOT_BG,
        xaxis=axis,
        yaxis=yaxis,
        showlegend=showlegend,
        font=dict(family="Space Grotesk, system-ui, sans-serif"),
        dragmode=False,
    )


def _tissue_layout(go_mod: Any, *, height: int = _TISSUE_HEIGHT) -> Any:
    """Layout for a true tissue image (cell map / density): 1:1 micron aspect, y
    reversed so the map reads like the slide."""
    lay = _base_layout(go_mod, showlegend=False, height=height, equal_aspect=True)
    lay.yaxis.autorange = "reversed"  # tissue reads top-down like the slide
    return lay


# Plotly display config: the view is LOCKED (no zoom, pan or toolbar); hover stays on.
_PLOT_CONFIG: dict[str, Any] = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


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
    """Colorbar for the density panel, labelled in bin counts.

    Colour stays sqrt(area-normalized density) so a coarser bin never looks hotter;
    within one bin size count is proportional to density, so the labels map exactly.
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
    """Return ``df`` with every ``keep_mask`` row kept in full plus at most
    ``bg_cap`` background rows (fixed seed). Display only, never a value."""
    keep = df[keep_mask]
    rest = df[~keep_mask]
    if len(rest) > bg_cap:
        rest = rest.sample(n=bg_cap, random_state=0)
    return pd.concat([keep, rest], axis=0)


# --------------------------------------------------------------------------- #
# Cell-type labels for hover (read the authoritative key; never computes a value)
# --------------------------------------------------------------------------- #
def _cluster_celltype_map() -> dict:
    """Map each cluster id to its displayed cell-type label (override-aware) for
    hover; blank reads as ``unassigned``."""
    from panoscope.ui import data_access as _da

    labels = {
        c: _da.display_cell_type(c).replace("_", " ")
        for c in cfg.CLUSTER_ORDER
    }
    labels[""] = "unassigned"
    return labels


# --------------------------------------------------------------------------- #
# Declared views — what this study's manifest says was never prepped
# --------------------------------------------------------------------------- #
# A view the manifest declares unavailable refuses to draw and names the prep step
# that would produce it. Guarded in the renderers, which enrichment_spatial reuses.
_VIEW_SOURCE: dict[str, str] = {
    "cell_map": "viz/cells.parquet · scripts/prep_data.R",
    "umap": "viz/umap.parquet · scripts/prep_data.R",
    "density": "viz/hexbin/ · scripts/precompute_density.py",
    "expr": "viz/expr.parquet · scripts/prep_expr.py",
}
_VIEW_LABEL: dict[str, str] = {
    "cell_map": "cell map",
    "umap": "UMAP",
    "density": "transcript density",
    "expr": "per-cell expression",
}


def _unavailable(*views: str) -> Optional[str]:
    """The first of ``views`` this study declares unavailable, or None to draw.

    The manifest may only veto a view; unknown or unreadable reads as available and
    the loader stays the backstop.
    """
    try:
        declared = cfg.current_study().views_available
    except Exception:  # noqa: BLE001 - no readable manifest is not a veto
        return None
    return next((v for v in views if not declared.get(v, True)), None)


def _unavailable_panel(view: str, *, height: int = _PLOT_HEIGHT) -> None:
    """State the absent view plainly: what is missing, and what would produce it."""
    _empty_panel(
        f"{_VIEW_LABEL.get(view, view)} not prepared for this study"
        f'<br><span style="color:var(--muted,#606A73)">'
        f"would come from {_VIEW_SOURCE.get(view, view)}</span>",
        height=height,
    )


# --------------------------------------------------------------------------- #
# View 1 — Cell map (tissue image; ratio-locked)
# --------------------------------------------------------------------------- #
def _render_cell_map(cluster: str) -> None:
    """Scattergl of all cells at tissue x/y; the selected cluster in colour, the
    rest faded to grey."""
    missing = _unavailable("cell_map")
    if missing:
        _unavailable_panel(missing, height=_TISSUE_HEIGHT)
        return

    go = _go()

    cells = da.cells_df()
    if cells.empty:
        _empty_panel("no cell coordinates available", height=_TISSUE_HEIGHT)
        return

    sel_mask = cells["cluster"] == cluster
    view = _downsample_bg(cells, sel_mask, _MAX_BG_POINTS)

    fig = go.Figure(layout=_tissue_layout(go))
    _selected_vs_rest(fig, go, view, cluster, "x", "y", bg_opacity=0.55)

    cnd.render_panel(
        fig,
        cnd.cell_map_cond(
            cluster=cluster,
            n_total=len(cells),
            n_selected=int(sel_mask.sum()),
            n_bg_drawn=int((view["cluster"] != cluster).sum()),
            n_bg_total=int((~sel_mask).sum()),
        ),
        config=_PLOT_CONFIG,
        key=f"pano_cellmap_{cluster}",
    )


# --------------------------------------------------------------------------- #
# View 2 — UMAP (cluster colour in Row 1; per-gene feature colour in gene rows)
# --------------------------------------------------------------------------- #
def _render_umap(cluster: str, *, feature: bool, gene: Optional[str] = None) -> None:
    """Scattergl in UMAP space.

    ``feature=False``: cluster-coloured. ``feature=True``: recoloured by ``gene``'s
    per-cell expression, falling back to cluster colour with a note when the gene
    has none exported.
    """
    # The feature recolour also needs the expression matrix, not just the embedding.
    missing = _unavailable("umap", "expr") if feature else _unavailable("umap")
    if missing:
        _unavailable_panel(missing)
        return

    go = _go()

    umap = da.umap_df()
    if umap.empty:
        _empty_panel("no UMAP embedding available")
        return

    expr = da.marker_expr_col(gene) if (feature and gene) else None

    sel_mask = umap["cluster"] == cluster
    view = _downsample_bg(umap, sel_mask, _MAX_BG_POINTS)
    n_sel = int(sel_mask.sum())
    n_bg_total = int((~sel_mask).sum())
    n_bg_drawn = len(view) - n_sel

    fig = go.Figure(layout=_base_layout(go, showlegend=False, equal_aspect=True))

    if feature and gene and expr is not None:
        _umap_feature(fig, go, view, expr, cluster)
        cnd.render_panel(
            fig,
            cnd.feature_umap_cond(
                gene=gene,
                n_expr_cells=len(expr),
                n_total=len(umap),
                n_bg_drawn=n_bg_drawn,
                n_bg_total=n_bg_total,
            ),
            config=_PLOT_CONFIG,
            key=f"pano_umap_feature_{cluster}_{gene}",
        )
        return

    # Cluster-coloured UMAP: Row 1, or a gene with no exported expression.
    _selected_vs_rest(fig, go, view, cluster)
    cnd.render_panel(
        fig,
        cnd.cluster_umap_cond(
            cluster=cluster,
            n_total=len(umap),
            n_selected=n_sel,
            n_bg_drawn=n_bg_drawn,
            n_bg_total=n_bg_total,
        ),
        config=_PLOT_CONFIG,
        # Distinct keys: Row 1 and a gene's fallback draw the same figure.
        key=(f"pano_umap_nofeature_{cluster}_{gene}" if feature and gene
             else f"pano_umap_clusters_{cluster}"),
    )
    if feature and gene and expr is None:
        _legend_line(
            f'<span style="color:var(--absent)">{gene}: per-cell expression not '
            f"precomputed</span> · showing cluster colour instead"
        )


def _selected_vs_rest(
    fig: Any, go: Any, view: pd.DataFrame, cluster: str,
    x: str = "umap_1", y: str = "umap_2", *, bg_opacity: float = 0.5,
) -> None:
    """Grey out every other cluster and fill the selected one with its palette
    colour; hover shows the cell type. Shared by the cell map and cluster UMAP."""
    view = view.assign(
        cell_type=view["cluster"].map(_cluster_celltype_map()).fillna("unassigned")
    )
    sel = view[view["cluster"] == cluster]
    bg = view[view["cluster"] != cluster]
    fig.add_trace(
        go.Scattergl(
            x=bg[x],
            y=bg[y],
            mode="markers",
            marker=dict(size=1.6, color=_FADE_COLOR, opacity=bg_opacity),
            customdata=bg["cell_type"],
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="other clusters",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=sel[x],
            y=sel[y],
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
    """Colour the UMAP by per-cell expression ``expr`` (cell_id, value): non-expressing
    cells as a grey base, expressing cells on top in sqrt plasma, brightest last."""
    merged = view.merge(expr, on="cell_id", how="left")
    vals = merged["value"]
    is_pos = vals > 0
    off = merged[~is_pos]
    pos = merged[is_pos].sort_values("value", ascending=True)

    # The gene's own maximum, not the thinned view's: a viewing control (which
    # cluster is selected) must never change the colourbar's value.
    gene_max = float(expr["value"].max()) if len(expr) else 0.0

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
                    colorscale=_PLASMA,
                    cmin=0.0,
                    cmax=float(gene_max ** 0.5) or float(pos_sqrt.max()),
                    opacity=0.92,
                    showscale=True,
                    colorbar=_sqrt_colorbar(gene_max or float(pos["value"].max()), "expr"),
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
    """Precomputed transcript density for ``gene`` at the global bin size, or a
    placeholder when no frame exists (never a synthesized field)."""
    missing = _unavailable("density")
    if missing:
        _unavailable_panel(missing, height=_TISSUE_HEIGHT)
        return

    bin_um = state.get_bin_um()
    try:
        hb = da.hexbins(gene, bin_um)
    except FileNotFoundError:
        _empty_panel(f"{gene}: density not precomputed", height=_TISSUE_HEIGHT)
        return
    except Exception:
        _empty_panel(f"{gene}: density unavailable", height=_TISSUE_HEIGHT)
        return
    _render_density_frame(
        hb,
        bin_um,
        cond=cnd.gene_density_cond(gene=gene, hb=hb, bin_um=bin_um),
        key=f"pano_density_gene_{gene}_{bin_um}",
        empty=f"{gene}: density not precomputed",
    )


def _render_density_frame(
    hb: Any, bin_um: int, *, cond: Any, empty: str = "density not precomputed",
    key: str | None = None,
) -> None:
    """Draw a hex-density frame (hx, hy, count, density) as a ratio-locked tissue
    image: colour sqrt(density), hover the real values, absent bins transparent.

    ``cond`` is required: only the caller knows what the frame is (one gene, or a
    sum over a gene set).
    """
    go = _go()

    if hb is None or hb.empty:
        _empty_panel(empty, height=_TISSUE_HEIGHT)
        return

    dens = hb["density"]
    # Display-only sqrt for colour; hover shows the real values.
    dens_sqrt = (dens.clip(lower=0) ** 0.5).to_numpy(dtype="float64")
    cmax = float(dens_sqrt.max()) if dens_sqrt.size and np.isfinite(dens_sqrt).any() else 1.0

    # Data-coordinate cells: each tile is exactly bin_um wide; absent bins stay NaN.
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
            colorscale=_PLASMA,
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

    cnd.render_panel(fig, cond, config=_PLOT_CONFIG, key=key)


# --------------------------------------------------------------------------- #
# View 4 — Dot plot (per-cell expression across clusters)
# --------------------------------------------------------------------------- #
def _render_dotplot(genes: list[str], cluster: str,
                    n_selected: int | None = None) -> None:
    """Dot plot of each selected gene across clusters: size = % of cells expressing,
    colour = mean per-cell expression, both from ``expr_by_cluster``."""
    # Needs both the expression matrix and the cell table's cluster labels.
    missing = _unavailable("expr", "cell_map")
    if missing:
        _unavailable_panel(missing, height=_VIOLIN_HEIGHT)
        return

    go = _go()

    # Genes with no exported expression are omitted, never faked.
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

    # (gene, cluster) -> mean expression + % of cells expressing (value > 0).
    rows: list[tuple[str, str, float, float]] = []
    for gene, by in frames:
        for c in cfg.CLUSTER_ORDER:
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
    # Colour by sqrt(mean); the colourbar is labelled in real mean units.
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
                colorscale=_PLASMA,
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
            categoryarray=list(cfg.CLUSTER_ORDER),
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
    cnd.render_panel(
        fig,
        cnd.dotplot_cond(
            genes_drawn=len(gene_order),
            # The uncapped selection, so "N of M" names what the figure left out.
            genes_selected=(len(genes) if n_selected is None else n_selected),
            n_clusters=len(cfg.CLUSTER_ORDER),
            n_expr_cells=len(da.marker_expr_df()),
            n_total=len(da.cells_df()),
        ),
        config=_PLOT_CONFIG,
        key=f"pano_dotplot_{cluster}",
    )
    _legend_line("dot size = % of cells expressing · colour = mean expression")


# --------------------------------------------------------------------------- #
# Global density bin control (one control for every density panel)
# --------------------------------------------------------------------------- #
def _render_density_controls() -> None:
    """The one global bin-size control for every density panel (viewing only)."""
    # No density frames -> no control; the panels below say what is missing.
    if _unavailable("density"):
        return

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
    state.set_bin_um(picked)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def render_spatial_stage(cluster: str) -> None:
    """Render the fixed spatial grid for ``cluster`` (see the module docstring)."""
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

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)

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
    _render_dotplot(markers, cluster,
                    n_selected=len(state.get_selected_markers(cluster)))
