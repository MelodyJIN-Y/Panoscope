"""Pure UI formatters — no Streamlit, no I/O, importable anywhere.

Everything here is a total function over already-loaded values. Colors and CSS
class names mirror the design tokens in ``theme.py`` and the wireframe
(``docs/archive/dashboard_wireframe_panels.html``). Nothing in this module computes a spatial
statistic or a confidence value — it only *labels* values the engine produced.
"""


from __future__ import annotations

from panoscope.agent import config as cfg
from typing import Optional

# --------------------------------------------------------------------------- #
# Cluster colors — one stable color per cluster c1..c9.
# Distinct, print-safe hues; teal (c1) matches the brand accent so the primary
# cluster reads as "home". Keyed by cluster id, never by index, so a reorder
# never reassigns a color.
# --------------------------------------------------------------------------- #
# The bundled demo's hand-assigned hues, kept so the demo's figures are stable and
# each colour still matches its cell type.
_DEMO_CLUSTER_COLORS: dict[str, str] = {
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

# A qualitative scale for any other study. 24 entries, so a study with more units
# than the demo does not render most of them identical grey -- which is what a
# nine-entry literal did.
_QUALITATIVE: tuple[str, ...] = (
    "#FC8D62", "#66C2A5", "#8DA0CB", "#E78AC3", "#A6D854", "#87CEEB",
    "#7D26CD", "#E5C498", "#0000FF", "#FFD92F", "#B3B3B3", "#1B9E77",
    "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D",
    "#666666", "#1F78B4", "#B2DF8A", "#FB9A99", "#CAB2D6", "#FF7F00",
)
_FALLBACK_CLUSTER_COLOR = "#9AA3AB"


def cluster_color(cluster: str) -> str:
    """Stable hex colour for a unit id, for ANY study.

    The demo keeps its hand-assigned hues. Any other study indexes the qualitative
    scale by the unit's position in that study's own order, so the assignment is
    stable per study and never collapses to grey past nine units.
    """
    if cfg.DATASET_ID == cfg.BUNDLED_DEMO_ID and cluster in _DEMO_CLUSTER_COLORS:
        return _DEMO_CLUSTER_COLORS[cluster]
    order = cfg.CLUSTER_ORDER
    if cluster in order:
        return _QUALITATIVE[order.index(cluster) % len(_QUALITATIVE)]
    return _FALLBACK_CLUSTER_COLOR


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


def short_cluster_id(cluster: str) -> str:
    """Return the numeric label shown in the rail (``c3`` -> ``3``)."""
    if cluster.startswith("c") and cluster[1:].isdigit():
        return cluster[1:]
    return cluster
