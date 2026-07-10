"""Summary page — the whole annotation set at a glance.

One page that answers "how do all nine calls look together": a clean page header
with at-a-glance counts, the full annotation table (every call side by side, with
the grounded cell-type summary wrapping in full), the cross-cluster coherence pass
(and the one grounded c8 -> pDC refinement), and a CSV download of the exact
export the biologist ships.

Grounding is preserved end-to-end — nothing here computes a value:

* the table is a straight projection of :func:`ui.data_access.all_verdicts`
  (each ``ClusterVerdict`` was computed once, deterministically). This module only
  *selects, orders, and styles* columns; it never derives a cell type, a
  confidence, or a marker. The cell-type summary comes from the pipeline's
  live-cited notes (``ui.data_access.celltype_summary``).
* the holistic pass is :func:`ui.holistic.render_holistic_review` reused verbatim.
* the download is ``da.verdict_csv()`` — the same 11-column CSV the grounding
  tests check, byte-for-byte.

Streamlit is imported lazily inside the render function so ``import ui.summary``
works with no server running.
"""

from __future__ import annotations

import html

from agent.types import ClusterVerdict

from ui import data_access as da
from ui import format as fmt
from ui import holistic

# --------------------------------------------------------------------------- #
# Copy — page headings. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Annotation summary"
_SUB = (
    "Every call side by side — the table is the exact export. Numbers come from "
    "jazzPanda; the cell-type summaries are live-cited; nothing here recomputes a value."
)
_DOWNLOAD_LABEL = "⬇  Download annotations (CSV)"
_DOWNLOAD_NAME = "panoscope_annotations.csv"
_DOWNLOAD_MIME = "text/csv"

# Column order the biologist asked for: Cluster, Lineage, Cell type, Key markers,
# Cell-type summary, Confidence, Re-check. (key, header, width%)
_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("cluster", "Cluster", 7),
    ("lineage", "Lineage", 9),
    ("cell_type", "Cell type", 12),
    ("key_markers", "Key markers", 19),
    ("summary", "Cell-type summary", 34),
    ("confidence", "Confidence", 12),
    ("verify", "Re-check", 7),
)

# --------------------------------------------------------------------------- #
# Page styling. Reuses the theme tokens + the global .cf-* confidence pills. A
# custom table (not st.dataframe) so the cell-type summary wraps in full and the
# whole page reads as one designed surface. All-around hairline card, no accents.
# --------------------------------------------------------------------------- #
_SUMMARY_CSS = """
.pano-sum-title { font-family: var(--sans); font-size: 24px; font-weight: 700;
                  letter-spacing: -.02em; color: var(--ink); margin: 2px 0 5px; }
.pano-sum-sub { font-size: 13px; color: var(--muted); line-height: 1.55;
                max-width: 82ch; margin: 0 0 12px; }
.pano-sum-meta { font-family: var(--mono); font-size: 11px; color: var(--faint);
                 display: flex; gap: 9px; align-items: center; margin: 0 0 18px; }
.pano-sum-meta .n { color: var(--ink); font-weight: 600; }
.pano-sum-meta .sep { color: var(--hair); }

.pano-sum-tablewrap { background: var(--paper); border: 1px solid var(--hair);
                      border-radius: 14px; overflow: hidden; }
.pano-sum-table { width: 100%; border-collapse: collapse; table-layout: fixed;
                  font-family: var(--sans); }
.pano-sum-table thead th {
  text-align: left; font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .07em; color: var(--faint); font-weight: 500; padding: 13px 15px;
  border-bottom: 1px solid var(--hair); background: #FAFBFB;
}
.pano-sum-table tbody td { padding: 15px; border-bottom: 1px solid var(--hair2);
                           vertical-align: top; }
.pano-sum-table tbody tr:last-child td { border-bottom: 0; }
.pano-sum-table tbody tr:hover td { background: #FAFCFC; }

.pano-sum-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                margin-right: 8px; vertical-align: middle; }
.pano-sum-cid { font-family: var(--mono); font-size: 12px; font-weight: 600;
                color: var(--ink); vertical-align: middle; }
.pano-sum-lin { font-size: 12.5px; color: var(--muted); }
.pano-sum-ct { font-size: 14px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.pano-sum-km { font-family: var(--mono); font-size: 11.5px; color: var(--muted);
               line-height: 1.65; word-break: break-word; }
.pano-sum-desc { font-size: 12.5px; color: var(--muted); line-height: 1.55; }
.pano-sum-dash { color: var(--faint); }
.pano-sum-flag { font-family: var(--mono); font-size: 10px; font-weight: 600;
                 color: var(--absent); background: var(--absent-bg); padding: 3px 8px;
                 border-radius: 6px; white-space: nowrap; }
/* Confidence pill: reuse the global .cf/.cf-* tokens; nudge size for the table. */
.pano-sum-table .cf { font-size: 10.5px; padding: 3px 9px; }
/* Right-align the CSV download under the table. */
.st-key-pano_dl { display: flex; justify-content: flex-end; margin-top: 14px; }
.st-key-pano_dl button { border-radius: 9px !important; }
"""


def _markers_cell(v: ClusterVerdict) -> str:
    """Key markers as a mono, middot-separated, wrapping list."""
    if not v.key_markers:
        return '<span class="pano-sum-dash">—</span>'
    genes = " · ".join(html.escape(str(g)) for g in v.key_markers)
    return f'<span class="pano-sum-km">{genes}</span>'


def _summary_cell(cluster: str) -> str:
    """The grounded, live-cited cell-type summary (wraps in full), or a dash."""
    text = da.celltype_summary(cluster)
    if not text:
        return '<span class="pano-sum-dash">—</span>'
    return f'<div class="pano-sum-desc">{html.escape(text)}</div>'


def _confidence_cell(v: ClusterVerdict) -> str:
    """A semantic confidence pill (reuses the global .cf-* band colors)."""
    css, _ = fmt.confidence_chip(v.confidence)
    return f'<span class="cf {css}">{html.escape(v.confidence)}</span>'


def _recheck_cell(v: ClusterVerdict) -> str:
    """An amber re-check flag when verify, else a quiet dash."""
    if v.verify:
        return '<span class="pano-sum-flag" title="flagged for re-check">⚑ re-check</span>'
    return '<span class="pano-sum-dash">—</span>'


def _row_html(v: ClusterVerdict) -> str:
    color = fmt.cluster_color(v.cluster)
    cluster_cell = (
        f'<span class="pano-sum-dot" style="background:{color}"></span>'
        f'<span class="pano-sum-cid">{html.escape(v.cluster)}</span>'
    )
    cells = {
        "cluster": cluster_cell,
        "lineage": f'<span class="pano-sum-lin">{html.escape(v.lineage)}</span>',
        "cell_type": f'<span class="pano-sum-ct">{html.escape(v.cell_type)}</span>',
        "key_markers": _markers_cell(v),
        "summary": _summary_cell(v.cluster),
        "confidence": _confidence_cell(v),
        "verify": _recheck_cell(v),
    }
    tds = "".join(f"<td>{cells[key]}</td>" for key, _, _ in _COLUMNS)
    return f"<tr>{tds}</tr>"


def _table_html(verdicts: list[ClusterVerdict]) -> str:
    """The full annotation table as one wrapping, styled HTML table."""
    cols = "".join(f'<col style="width:{w}%">' for _, _, w in _COLUMNS)
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label, _ in _COLUMNS)
    body = "".join(_row_html(v) for v in verdicts)
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{cols}</colgroup>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody>"
        "</table></div>"
    )


def render_summary_page() -> None:
    """Render the Summary page: header + annotation table + holistic pass + download.

    Order: page title + one-line intent + at-a-glance counts, the full-width
    annotation table (the cell-type summary wraps in full), a right-aligned CSV
    download, then the cross-cluster review reused verbatim.
    """
    import streamlit as st

    st.markdown(f"<style>{_SUMMARY_CSS}</style>", unsafe_allow_html=True)

    verdicts = da.all_verdicts()
    n_flagged = sum(1 for v in verdicts if v.verify)
    try:
        n_panel = len(da.panel_names())
    except Exception:  # noqa: BLE001 - a missing panel count is not fatal to the page
        n_panel = 0

    flagged_txt = (
        f'<span class="n">{n_flagged}</span> flagged for re-check'
        if n_flagged
        else "none flagged for re-check"
    )
    st.markdown(
        f'<div class="pano-sum-title">{html.escape(_TITLE)}</div>'
        f'<div class="pano-sum-sub">{html.escape(_SUB)}</div>'
        '<div class="pano-sum-meta">'
        f'<span class="n">{len(verdicts)}</span> clusters'
        '<span class="sep">·</span>'
        f"{flagged_txt}"
        '<span class="sep">·</span>'
        f'<span class="n">{n_panel}</span>-gene panel'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(_table_html(verdicts), unsafe_allow_html=True)

    with st.container(key="pano_dl"):
        st.download_button(
            _DOWNLOAD_LABEL,
            da.verdict_csv(),
            _DOWNLOAD_NAME,
            _DOWNLOAD_MIME,
        )

    holistic.render_holistic_review()


__all__ = ["render_summary_page"]
