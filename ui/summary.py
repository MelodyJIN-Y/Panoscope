"""Summary page — the whole annotation set at a glance.

One page that answers "how do all nine calls look together": the full annotation
table, the holistic cross-cluster coherence pass (and the one grounded c8 -> pDC
refinement), and a CSV download of the exact export the biologist ships.

Grounding is preserved end-to-end — nothing here computes a value:

* the table is a straight projection of :func:`ui.data_access.all_verdicts`
  (each ``ClusterVerdict`` was computed once, deterministically, in the agent
  layer). This module only *selects and renames columns*; it never derives a
  cell type, a confidence, or a marker.
* the holistic pass is :func:`ui.holistic.render_holistic_review` reused
  verbatim (its numbers are computed from source, its refinement citation is
  fetched live or degrades honestly — see that module's contract).
* the download is ``da.verdict_csv()`` — the same 11-column CSV the grounding
  tests check, byte-for-byte.

Streamlit is imported lazily inside the render function so ``import ui.summary``
works with no server running (the module touches no ``st.*`` at import time).
"""

from __future__ import annotations

import pandas as pd

from agent.types import ClusterVerdict

from ui import data_access as da
from ui import holistic

# --------------------------------------------------------------------------- #
# Copy — page headings. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_PAGE_TITLE = "All clusters — annotation summary"
_PAGE_SUB = (
    "Every call side by side. The table below is the exact export; the holistic "
    "pass re-reads the whole set for coherence. Numbers come from jazzPanda — "
    "nothing on this page recomputes a value."
)
_DOWNLOAD_LABEL = "Download annotations (CSV)"
_DOWNLOAD_NAME = "panoscope_annotations.csv"
_DOWNLOAD_MIME = "text/csv"

# Column order + display labels for the summary table. The keys are attributes
# on ``ClusterVerdict`` (``key_markers`` is ";"-joined for a flat cell); the
# values are the header labels shown to the biologist.
_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cluster", "Cluster"),
    ("cell_type", "Cell type"),
    ("cell_type_short", "Short"),
    ("confidence", "Confidence"),
    ("verify", "Re-check"),
    ("key_markers", "Key markers"),
    ("category", "Category"),
    ("lineage", "Lineage"),
)


def _verdicts_to_frame(verdicts: list[ClusterVerdict]) -> pd.DataFrame:
    """Project the verdict list into the summary table (select + rename only).

    Pure projection — reads attributes off each frozen ``ClusterVerdict`` and
    ";"-joins ``key_markers`` into one flat cell. Computes nothing.
    """
    rows = [
        {
            "cluster": v.cluster,
            "cell_type": v.cell_type,
            "cell_type_short": v.cell_type_short,
            "confidence": v.confidence,
            "verify": bool(v.verify),
            "key_markers": "; ".join(v.key_markers),
            "category": v.category,
            "lineage": v.lineage,
        }
        for v in verdicts
    ]
    frame = pd.DataFrame(rows, columns=[key for key, _ in _COLUMNS])
    return frame.rename(columns=dict(_COLUMNS))


def render_summary_page() -> None:
    """Render the Summary page: annotation table + holistic pass + CSV download.

    Order: page heading, the full-width annotation ``st.dataframe`` (all nine
    verdicts, ``hide_index`` / ``use_container_width``), the holistic
    cross-cluster review reused verbatim, then the CSV download button.
    """
    import streamlit as st

    st.markdown(
        f'<div class="pano-eyebrow">{_PAGE_TITLE}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="hol-sub">{_PAGE_SUB}</div>',
        unsafe_allow_html=True,
    )

    frame = _verdicts_to_frame(da.all_verdicts())
    st.dataframe(
        frame,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Re-check": st.column_config.CheckboxColumn(
                "Re-check", help="Flagged for re-checking (verify)"
            ),
        },
    )

    st.download_button(
        _DOWNLOAD_LABEL,
        da.verdict_csv(),
        _DOWNLOAD_NAME,
        _DOWNLOAD_MIME,
    )

    # The cross-cluster coherence pass + the one grounded c8 -> pDC refinement,
    # reused verbatim (its numbers are computed from source, its citation is
    # fetched live or degrades honestly).
    holistic.render_holistic_review()


__all__ = ["render_summary_page"]
