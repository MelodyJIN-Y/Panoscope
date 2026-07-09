"""Pure UI formatters — no Streamlit, no I/O, importable anywhere.

Everything here is a total function over already-loaded values. Colors and CSS
class names mirror the design tokens in ``theme.py`` and the wireframe
(``dashboard_wireframe_panels.html``). Nothing in this module computes a spatial
statistic or a confidence value — it only *labels* values the engine produced.

The marker-role classifier is the one place the panel-absence rule is rendered
visible in the table: a canonical gene that is off-panel is ``off_panel`` (never
measured, uninformative), a canonical on-panel gene that is present is
``supports``, and a canonical on-panel gene that reads weak is ``expected_absent``
(a real down-weight). It reads the role the verdict already assigned when the
gene is in the evidence set, and falls back to ``panel_contains`` for genes the
biologist adds by hand.
"""

from __future__ import annotations

from typing import Optional

from agent.config import CLUSTER_ORDER
from agent.types import ClusterVerdict, MarkerEvidence, MarkerRole

# --------------------------------------------------------------------------- #
# Cluster colors — one stable color per cluster c1..c9.
# Distinct, print-safe hues; teal (c1) matches the brand accent so the primary
# cluster reads as "home". Keyed by cluster id, never by index, so a reorder
# never reassigns a color.
# --------------------------------------------------------------------------- #
CLUSTER_COLORS: dict[str, str] = {
    "c1": "#FC8D62",  # orange (Tumor)
    "c2": "#66C2A5",  # teal-green (Stromal)
    "c3": "#8DA0CB",  # blue-violet (Macrophages)
    "c4": "#E78AC3",  # pink (Myoepithelial)
    "c5": "#A6D854",  # green (T cells)
    "c6": "#87CEEB",  # skyblue (B cells)
    "c7": "#7D26CD",  # purple3 (Endothelial)
    "c8": "#E5C498",  # tan (Dendritic)
    "c9": "#0000FF",  # blue (Mast cells)
}
_FALLBACK_CLUSTER_COLOR = "#9AA3AB"


def cluster_color(cluster: str) -> str:
    """Return the stable hex color for a cluster id (grey fallback if unknown)."""
    return CLUSTER_COLORS.get(cluster, _FALLBACK_CLUSTER_COLOR)


def cluster_palette() -> dict[str, str]:
    """Return the cluster->color map in c1..c9 order (fresh dict, safe to mutate)."""
    return {c: cluster_color(c) for c in CLUSTER_ORDER}


# --------------------------------------------------------------------------- #
# Confidence chips — label -> (css class, display text).
# Class names line up with .cf-* rules injected by theme.inject_css().
# --------------------------------------------------------------------------- #
_CONFIDENCE_CLASS: dict[str, str] = {
    "Very High": "cf-vh",
    "High": "cf-h",
    "Medium-High": "cf-mh",
    "Medium": "cf-m",
    "Low": "cf-l",
}


def confidence_chip(label: str) -> tuple[str, str]:
    """Return ``(css_class, text)`` for a confidence band label.

    Unknown labels degrade to the "Low" styling with the raw label as text, so a
    stray value renders visibly rather than crashing the header.
    """
    css = _CONFIDENCE_CLASS.get(label, "cf-l")
    return css, f"{label} confidence"


# --------------------------------------------------------------------------- #
# Role chips — the panel-absence rule made visible in the evidence table.
# --------------------------------------------------------------------------- #
_ROLE_CLASS: dict[str, str] = {
    "supports": "role-sup",
    "expected_absent": "role-abs",
    "off_panel": "role-off",
}
_ROLE_TEXT: dict[str, str] = {
    "supports": "● supports",              # ● supports
    "expected_absent": "▲ expected, absent",  # ▲ expected, absent
    "off_panel": "⊘ off-panel",            # ⊘ off-panel
}


def role_chip(role: str) -> tuple[str, str]:
    """Return ``(css_class, text)`` for a marker role.

    Accepts the three ``MarkerRole`` values (``supports`` / ``expected_absent``
    / ``off_panel``). Unknown roles fall back to the off-panel (neutral grey)
    styling so nothing renders as a false supporter.
    """
    css = _ROLE_CLASS.get(role, "role-off")
    text = _ROLE_TEXT.get(role, "⊘ off-panel")
    return css, text


# --------------------------------------------------------------------------- #
# Marker role classifier
# --------------------------------------------------------------------------- #
def marker_role(
    gene: str,
    verdict: ClusterVerdict,
    *,
    panel_contains,
) -> MarkerRole:
    """Classify a gene's role for a cluster, deferring to the verdict's evidence.

    Priority:
      1. If the gene is already in the verdict's evidence set, use the role the
         engine assigned (single source of truth for on-panel markers).
      2. Otherwise the gene is one the biologist typed in. If it is off the
         panel it is ``off_panel`` (never measured — absence is uninformative,
         the headline invariant). If it is on-panel but not a modeled marker for
         this cluster, it reads ``expected_absent`` (measured here, not a
         supporter of this call).

    ``panel_contains`` is injected (the ``agent.data.panel_contains`` primitive)
    so this stays a pure function and never imports the loader directly.
    """
    up = gene.upper()
    for ev in verdict.evidence:
        if ev.gene.upper() == up:
            return ev.role
    if not panel_contains(gene):
        return "off_panel"
    return "expected_absent"


def evidence_role(ev: MarkerEvidence) -> MarkerRole:
    """Return the role already carried on an evidence row (trivial accessor)."""
    return ev.role


# --------------------------------------------------------------------------- #
# Number formatting — mono, honest, never invents precision.
# --------------------------------------------------------------------------- #
def num_fmt(value: Optional[float], digits: int = 2) -> str:
    """Format a number for the mono columns; ``None``/NaN -> an em dash.

    Keeps a fixed number of decimals so columns align. Does not round-then-lie:
    it shows exactly ``digits`` places of the stored value.
    """
    if value is None:
        return "n/a"  # no value
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:  # NaN
        return "n/a"
    return f"{f:.{digits}f}"


def pct_fmt(value: Optional[float]) -> str:
    """Format a 0..1 within-cluster percentile as a rank string (e.g. ``top 4%``).

    ``within_cluster_pctile`` is 1.0 for the strongest marker in a cluster, so
    the displayed "top X%" is ``(1 - pctile) * 100`` — a stronger marker reads
    as a smaller top-percent.
    """
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if f != f:
        return "n/a"
    top = max(0.0, min(1.0, 1.0 - f))
    return f"top {top * 100:.0f}%"


def verify_badge(verify: bool) -> str:
    """Return a short verify marker for rails/headers (empty when not flagged)."""
    return "⚑ re-check" if verify else ""  # ⚑ re-check


def short_cluster_id(cluster: str) -> str:
    """Return the numeric label shown in the rail (``c3`` -> ``3``)."""
    if cluster.startswith("c") and cluster[1:].isdigit():
        return cluster[1:]
    return cluster


__all__ = [
    "CLUSTER_COLORS",
    "cluster_color",
    "cluster_palette",
    "confidence_chip",
    "role_chip",
    "marker_role",
    "evidence_role",
    "num_fmt",
    "pct_fmt",
    "verify_badge",
    "short_cluster_id",
]
