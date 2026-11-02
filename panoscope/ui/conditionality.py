"""R4, with pixels: no conditionality record, no plot.

``panoscope/types/uispec.py`` already refuses to *declare* a figure without a
conditionality. This module is the other half — the renderer that refuses to
*draw* one. Every figure on the spatial stage goes through :func:`render_panel`,
which raises :class:`UndrawableFigure` for a panel that cannot say what it drew,
over what denominator, under what transform, and what it is **not**.

The builders below are the conditionality records for the five existing figures.
They are pure functions of numbers the caller already has in hand (row counts,
the bin size, the colour maximum) plus two grounded files — the study manifest
(sha256 per artifact) and the density ``_index.json`` (qv floor, control-probe
prefixes, normalisation). Nothing here computes a statistic, and nothing here
asserts a constant the data can answer: the panel denominator is
``len(panel_names())``, never 280.

Three things the audit found reaching no pixel, now stated:

* the ``_MAX_BG_POINTS`` display downsample (``rendering``) — "the downsample is
  stated rather than deleted" (docs/interface-design.md §1.3);
* the Row-1 cluster UMAP caption, deleted for a cleaner grid and restored here as
  a record rather than a label;
* the panel denominator, the hex-bin µm and the area normalisation, which lived
  in module docstrings for developers and nowhere for the biologist.
"""

from __future__ import annotations

import html
import json
from functools import lru_cache
from typing import Any, Optional

from panoscope.agent import config as cfg
from panoscope.types.uispec import Conditionality, UndrawableFigure
from panoscope.ui import data_access as da

# Approximate size of the human protein-coding gene set — the denominator a
# targeted panel is a fraction OF, and the number a reader silently assumes when
# a figure does not say otherwise. Written with a "~" because it is a round
# external fact, not a measurement. The panel side of the ratio is always read
# from the panel file.
_TRANSCRIPTOME_APPROX = "~20,000"


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def render_panel(fig: Any, cond: Conditionality, *, config: Optional[dict] = None) -> None:
    """Draw ``fig`` with its two-line conditionality caption, or refuse.

    The visual analogue of the grounding gate discarding an ungrounded sentence.
    A figure arriving without a :class:`Conditionality` does not render; it
    raises. (A ``Conditionality`` missing one of its four mandatory fields cannot
    exist in the first place — ``Conditionality.__post_init__`` raises the same
    error at construction, so both halves of R4 fail closed.)

    The caption is drawn from :meth:`Conditionality.caption`; the full record,
    every field labelled, is the panel's hover title.
    """
    if not isinstance(cond, Conditionality):
        raise UndrawableFigure(
            f"figure passed to render_panel with {type(cond).__name__} instead of a "
            "Conditionality. A plot with no conditionality record does not draw."
        )

    import streamlit as st

    st.plotly_chart(fig, use_container_width=True, config=config)
    line1, line2 = cond.caption()
    st.markdown(
        f'<div class="plegend" title="{html.escape(_full_record(cond))}">'
        f"{html.escape(line1)}<br>{html.escape(line2)}</div>",
        unsafe_allow_html=True,
    )


def _full_record(cond: Conditionality) -> str:
    """Every populated field, labelled — the hover expansion of the caption."""
    fields = ("drawn", "denominator", "transform", "not_this", "unit_scope",
              "exclusions", "scale", "rendering", "provenance")
    return "\n".join(
        f"{name.replace('_', ' ')}: {getattr(cond, name)}"
        for name in fields
        if getattr(cond, name)
    )


# --------------------------------------------------------------------------- #
# Grounded fragments (read from files; never asserted)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _manifest() -> dict:
    """The study manifest, for artifact sha256s. ``{}`` if unreadable."""
    try:
        from panoscope.pipeline import paths

        p = paths.dataset_dir(cfg.DATASET_ID) / "manifest.json"
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - no manifest just means no sha fragment
        return {}


@lru_cache(maxsize=1)
def _density_index() -> dict:
    """The density precompute index: qv floor, control prefixes, normalisation."""
    try:
        return json.loads(da._density_index().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - absent index just means no exclusions line
        return {}


def _sha(artifact: str) -> str:
    """``viz/cells.csv sha256 219a4588`` — provenance, or '' if unknown."""
    rec = (_manifest().get("artifacts") or {}).get(artifact) or {}
    digest = str(rec.get("sha256", ""))
    return f"{artifact} sha256 {digest[:8]}" if digest else ""


def _panel_scope() -> str:
    """The denominator always left out: how many genes could have been drawn.

    Read from the panel file every time (CLAUDE.md: panel size is dynamic across
    studies — derive it, never assert a constant).
    """
    try:
        n = len(da.panel_names())
    except Exception:  # noqa: BLE001 - an unreadable panel is still a denominator
        return "a targeted panel of unknown size; NOT whole-transcriptome"
    return (
        f"a {n}-gene targeted panel (of {_TRANSCRIPTOME_APPROX} protein-coding "
        "genes; NOT whole-transcriptome)"
    )


def _density_exclusions() -> str:
    idx = _density_index()
    qv, prefixes = idx.get("qv_min"), idx.get("control_prefixes")
    if qv is None or not prefixes:
        return ""
    return f"qv >= {qv:g} · {'/'.join(prefixes)} probes removed"


def _thinning(n_bg_drawn: int, n_bg_total: int, unit: str) -> str:
    """The display downsample, stated. '' when nothing was thinned."""
    if n_bg_drawn >= n_bg_total:
        return ""
    return (
        f"background thinned to {n_bg_drawn:,} of {n_bg_total:,} for rendering; "
        f"{unit} drawn in full"
    )


def _unit_scope(cluster: str, n_selected: int, n_total: int) -> str:
    return (
        f"{cfg.DATASET_ID} · {cluster} = {n_selected:,} of {n_total:,} cells "
        f"· {len(cfg.CLUSTER_ORDER)} clusters"
    )


def _all_cells_scope(n_total: int) -> str:
    """Scope for a panel coloured by a value rather than by the selected unit."""
    return (
        f"{cfg.DATASET_ID} · all {n_total:,} cells drawn, every cluster included"
    )


# --------------------------------------------------------------------------- #
# The five figures
# --------------------------------------------------------------------------- #
def cell_map_cond(
    *, cluster: str, n_total: int, n_selected: int, n_bg_drawn: int, n_bg_total: int
) -> Conditionality:
    """Row 1 left — cell centroids at tissue coordinates, coloured by cluster."""
    return Conditionality(
        drawn="cell centroids at tissue coordinates, coloured by cluster",
        denominator=(
            f"cluster labels derived from {_panel_scope()}; "
            f"{n_total:,} segmented cells"
        ),
        transform="no transform · tissue um, 1:1 aspect lock, y reversed",
        not_this=(
            "centroids, NOT segmentation boundaries; colour is the cluster label, "
            "not a measured value"
        ),
        unit_scope=_unit_scope(cluster, n_selected, n_total),
        rendering=_thinning(n_bg_drawn, n_bg_total, cluster),
        provenance=_sha("viz/cells.csv"),
    )


def cluster_umap_cond(
    *, cluster: str, n_total: int, n_selected: int, n_bg_drawn: int, n_bg_total: int
) -> Conditionality:
    """Row 1 right — the cluster UMAP. The caption deleted "for a cleaner grid".

    Its standing caveat is the one a reader cannot get from the pixels and
    reliably gets wrong: a UMAP is a layout, so the gaps between clusters are not
    distances and "clean island" is a visual impression, not a number.
    """
    return Conditionality(
        drawn="UMAP embedding of the same cells, coloured by cluster",
        denominator=(
            f"{n_total:,} embedded cells; layout computed from {_panel_scope()}"
        ),
        transform="UMAP layout from the prep step, 1:1 unit lock",
        not_this=(
            "a layout, NOT a measurement · between-cluster distances are not "
            "interpretable and no statistic is drawn"
        ),
        unit_scope=_unit_scope(cluster, n_selected, n_total),
        rendering=_thinning(n_bg_drawn, n_bg_total, cluster),
        provenance=_sha("viz/umap.csv"),
    )


def feature_umap_cond(
    *, gene: str, n_expr_cells: int, n_total: int, n_bg_drawn: int, n_bg_total: int
) -> Conditionality:
    """A marker row, right — the UMAP recoloured by one gene's expression."""
    return Conditionality(
        drawn=f"{gene} log-normalised per-cell expression on the UMAP",
        denominator=(
            f"expression exported for {n_expr_cells:,} of {n_total:,} cells "
            f"(cluster-stratified subsample); {gene} is 1 gene on {_panel_scope()}"
        ),
        transform=(
            "sqrt colour on expression; cells at or below zero drawn grey beneath"
        ),
        not_this=(
            "not glm_coef, not pearson, not a test · this panel carries no "
            "statistic and no confidence"
        ),
        unit_scope=_all_cells_scope(n_total),
        scale="independent scale · this gene's own maximum; panels are NOT comparable",
        rendering=_thinning(n_bg_drawn, n_bg_total, "expressing cells"),
        provenance=_sha("viz/expr.parquet"),
    )


def _density_shape(hb: Any, bin_um: int) -> tuple[str, str, str]:
    """(transform, scale, rendering) shared by both density panels."""
    n_bins = 0 if hb is None or hb.empty else len(hb)
    dmax = 0.0 if not n_bins else float(hb["density"].max())
    norm = _density_index().get("normalization", "density = count / bin area")
    return (
        f"{bin_um} um square bins, area-normalised ({norm}), sqrt colour; "
        "colourbar ticks labelled in per-bin transcript count",
        f"independent scale · max {dmax:.4g} tx/um2 in this panel; "
        "panels are NOT comparable",
        f"{n_bins:,} occupied bins drawn; empty bins transparent",
    )


def gene_density_cond(*, gene: str, hb: Any, bin_um: int) -> Conditionality:
    """A marker row, left — one gene's precomputed transcript density."""
    transform, scale, rendering = _density_shape(hb, bin_um)
    return Conditionality(
        drawn=f"{gene} transcript detections, pre-segmentation",
        denominator=_panel_scope(),
        transform=transform,
        not_this="not per-cell expression, not a statistic · no test and no q-value",
        exclusions=_density_exclusions(),
        scale=scale,
        rendering=rendering,
        provenance=f"viz/hexbin/{gene}_{bin_um}um.parquet",
    )


def set_density_cond(
    *, program: str, hb: Any, bin_um: int, n_genes: int, n_drawn: int
) -> Conditionality:
    """Pathways row — the leading-edge set's summed transcript density.

    ``n_drawn`` of ``n_genes`` is the fact the sum hides: ``leading_edge_density``
    skips a driving gene with no precomputed frame, and the picture cannot show
    which ones are missing.
    """
    transform, scale, rendering = _density_shape(hb, bin_um)
    return Conditionality(
        drawn=(
            f"{program} leading-edge transcript detections · the SUM of "
            f"{n_drawn} of {n_genes} driving genes' densities, pre-segmentation"
        ),
        denominator=_panel_scope(),
        transform=transform,
        not_this=(
            "not the enrichment score and not its q-value · a sum of measured "
            "densities, not a test"
        ),
        exclusions=_density_exclusions(),
        scale=scale,
        rendering=rendering,
        provenance=f"viz/hexbin/*_{bin_um}um.parquet ({n_drawn} frames summed)",
    )


def leading_edge_umap_cond(
    *, program: str, n_genes: int, n_drawn: int, n_expr_cells: int,
    n_total: int, n_bg_drawn: int, n_bg_total: int,
) -> Conditionality:
    """Pathways row, right — leading-edge activity on the UMAP."""
    return Conditionality(
        drawn=(
            f"{program} leading-edge activity · the MEAN per-cell expression of "
            f"{n_drawn} of {n_genes} driving genes, on the UMAP"
        ),
        denominator=(
            f"expression exported for {n_expr_cells:,} of {n_total:,} cells "
            f"(cluster-stratified subsample); driving genes drawn from {_panel_scope()}"
        ),
        transform=(
            "mean across the drawn genes, then sqrt colour; cells at or below "
            "zero drawn grey beneath"
        ),
        not_this=(
            "not the enrichment score, not its q-value, and not a per-cell "
            "activity test"
        ),
        unit_scope=_all_cells_scope(n_total),
        scale="independent scale · this program's own maximum; panels are NOT comparable",
        rendering=_thinning(n_bg_drawn, n_bg_total, "expressing cells"),
        provenance=_sha("viz/expr.parquet"),
    )


def dotplot_cond(
    *, genes_drawn: int, genes_selected: int, n_clusters: int,
    n_expr_cells: int, n_total: int,
) -> Conditionality:
    """The across-cluster dot plot.

    Its ``transform`` says the mean and the percentage are computed here, in the
    renderer, from the exported per-cell values. That is the one place on this
    stage where a figure derives rather than renders, and the caption states it
    rather than hiding it.
    """
    return Conditionality(
        drawn=(
            f"mean per-cell expression (colour) and % of cells expressing (dot "
            f"area) for {genes_drawn} of {genes_selected} selected genes "
            f"across {n_clusters} clusters"
        ),
        denominator=(
            f"expression exported for {n_expr_cells:,} of {n_total:,} cells "
            f"(cluster-stratified subsample); genes from {_panel_scope()}"
        ),
        transform=(
            "mean and % expressing (value > 0) computed in the renderer from the "
            "exported per-cell values; sqrt colour on the mean, colourbar "
            "labelled in real mean units"
        ),
        not_this=(
            "not the marker statistic · no glm_coef, no pearson, no significance "
            "test; a gene with no exported expression has no row here"
        ),
        provenance=_sha("viz/expr.parquet"),
    )


__all__ = [
    "render_panel",
    "cell_map_cond",
    "cluster_umap_cond",
    "feature_umap_cond",
    "gene_density_cond",
    "set_density_cond",
    "leading_edge_umap_cond",
    "dotplot_cond",
]
