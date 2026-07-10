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

import datetime
import html

from agent.types import ClusterVerdict

from ui import data_access as da
from ui import format as fmt
from ui import report

# --------------------------------------------------------------------------- #
# Copy — page headings. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Interpretation summary"
_SUB = (
    "The final step: both workflows in one place — the marker-gene calls and the gene-set "
    "programs — drafted into a per-cluster write-up you edit and export. Numbers come from "
    "jazzPanda; biology is live-cited; nothing here recomputes a value."
)
_DOWNLOAD_LABEL = "⬇  Download annotations (CSV)"
_DOWNLOAD_NAME = "panoscope_annotations.csv"
_DOWNLOAD_MIME = "text/csv"

# The working-space export — the edited interpretation, downloadable as Word/PDF.
_DOCX_LABEL = "⬇  Download report (Word)"
_DOCX_NAME = "panoscope_interpretation.docx"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_LABEL = "⬇  Download report (PDF)"
_PDF_NAME = "panoscope_interpretation.pdf"
_PDF_MIME = "application/pdf"

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
/* The interpretation-summary region header + its two report downloads. */
.pano-report-head { font-family: var(--sans); font-size: 19px; font-weight: 700;
                    color: var(--ink); letter-spacing: -.01em; margin: 30px 0 4px; }
.pano-report-sub { font-size: 12.5px; color: var(--muted); line-height: 1.55;
                   max-width: 82ch; margin: 0 0 14px; }
.st-key-pano_report_dl { display: flex; justify-content: flex-end; gap: 10px; margin-top: 14px; }
.st-key-pano_report_dl button { border-radius: 9px !important; }

/* Stacked-section headers + the editable working space (modern, professional). */
.pano-sum-h2 { font-family: var(--sans); font-size: 18px; font-weight: 700; color: var(--ink);
  letter-spacing: -.01em; margin: 42px 0 3px; }
.pano-sum-h2 .k { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--accent); font-weight: 600; display: block; margin-bottom: 5px; }
.pano-sum-note { font-size: 12.5px; color: var(--muted); line-height: 1.55; margin: 2px 0 14px; max-width: 84ch; }
.pano-enr-le { font-family: var(--mono); font-size: 11px; color: var(--muted); line-height: 1.6; word-break: break-word; }
.pano-ws-head { display: flex; align-items: baseline; gap: 10px; padding: 6px 0 3px; flex-wrap: wrap; }
.pano-ws-head .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.pano-ws-head .cid { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.pano-ws-head .ct { font-size: 15px; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.pano-ws-head .cf { font-size: 10px; padding: 2px 8px; }
div[class*="st-key-ws_"] textarea { font-family: var(--sans) !important; font-size: 13px !important;
  line-height: 1.6 !important; border-radius: 11px !important; border: 1px solid var(--hair) !important;
  background: #FCFCFD !important; color: var(--ink) !important; }
div[class*="st-key-ws_"] textarea:focus { border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important; background: var(--paper) !important; }
.st-key-pano_ws_refresh { display: flex; justify-content: flex-end; margin: -8px 0 6px; }
.st-key-pano_ws_refresh button { background: transparent !important; border: 1px solid var(--hair) !important;
  color: var(--faint) !important; box-shadow: none !important; min-height: 0 !important; padding: 4px 12px !important;
  border-radius: 7px !important; font-family: var(--mono) !important; font-size: 10.5px !important; }
.st-key-pano_ws_refresh button:hover { color: var(--accent) !important; border-color: var(--accent) !important; background: var(--accent-soft) !important; }
.st-key-pano_ws_dl { display: flex; gap: 10px; justify-content: flex-end; margin-top: 12px; }
.st-key-pano_ws_dl button { border-radius: 9px !important; }
.pano-lk-card { border: 1px solid var(--hair); border-radius: 11px; padding: 11px 13px; margin: 8px 0; background: var(--paper); max-width: 84ch; }
.pano-lk-claim { font-size: 13px; color: var(--ink); line-height: 1.5; }
.pano-lk-meta { font-family: var(--mono); font-size: 10px; color: var(--faint); margin-top: 7px; }
.pano-lk-meta .sco { background: var(--accent-soft); color: var(--accent); padding: 2px 7px; border-radius: 5px; }
.pano-lk-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); border: 1px dashed var(--hair);
  border-radius: 10px; padding: 16px; text-align: center; max-width: 84ch; }
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


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _enrichment_table_html(enrichments: list) -> str:
    """Per-cluster gene-set enrichment overview: cell type, enrichment confidence,
    top enriched programs, and their leading-edge genes."""
    ecols = (
        ("Cluster", 8), ("Cell type", 13), ("Enrichment", 13),
        ("Top programs", 36), ("Leading-edge genes", 30),
    )
    cols = "".join(f'<col style="width:{w}%">' for _, w in ecols)
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in ecols)
    rows = []
    for ce in enrichments:
        color = fmt.cluster_color(ce.cluster)
        cl = (f'<span class="pano-sum-dot" style="background:{color}"></span>'
              f'<span class="pano-sum-cid">{html.escape(ce.cluster)}</span>')
        ct = f'<span class="pano-sum-ct">{html.escape(ce.cell_type)}</span>'
        if ce.enriched:
            css, _ = fmt.confidence_chip(ce.confidence)
            conf = f'<span class="cf {css}">{html.escape(ce.confidence)}</span>'
            progs = f'<span class="pano-sum-desc">{" · ".join(_short(p.gene_set) for p in ce.enriched[:4])}</span>'
            le: list[str] = []
            for p in ce.enriched[:3]:
                for g in p.leading_edge:
                    if g not in le:
                        le.append(g)
            le_html = f'<span class="pano-enr-le">{" · ".join(html.escape(g) for g in le[:10])}</span>'
        else:
            conf = '<span class="pano-sum-dash">—</span>'
            progs = '<span class="pano-sum-dash">no program clears the gate</span>'
            le_html = '<span class="pano-sum-dash">—</span>'
        rows.append(f"<tr><td>{cl}</td><td>{ct}</td><td>{conf}</td><td>{progs}</td><td>{le_html}</td></tr>")
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _reset_ws_all(defaults: dict) -> None:
    """on_click: reset EVERY working-space region to its freshest auto-seed at once.

    ``defaults`` is the {session_key: latest-draft} map computed this render, so one
    button re-drafts the whole write-up (e.g. after chatting saved a new lab note)
    without touching regions the biologist has not edited differently."""
    import streamlit as st

    for key, default in defaults.items():
        st.session_state[key] = default


def _editable(st, key: str, default: str, height: int) -> None:
    """An auto-seeded, directly editable region. Seeds from ``default`` once; the
    biologist's edits then persist (edits win). The single working-space refresh
    button re-seeds every region from the latest draft."""
    if key not in st.session_state:
        st.session_state[key] = default
    st.text_area("edit", key=key, height=height, label_visibility="collapsed")


def _sec_head(st, kicker: str, title: str, note: str = "") -> None:
    note_html = f'<div class="pano-sum-note">{html.escape(note)}</div>' if note else ""
    st.markdown(
        f'<div class="pano-sum-h2"><span class="k">{html.escape(kicker)}</span>{html.escape(title)}</div>{note_html}',
        unsafe_allow_html=True,
    )


def _render_lab_notes(st, notes: list) -> None:
    _sec_head(st, "Lab knowledge", "Your notes",
              "Captured when you override or confirm a call in the chat — each carries its basis "
              "and any literature tension, and is compiled into the working space above.")
    if not notes:
        st.markdown('<div class="pano-lk-empty">No notes yet. Override or confirm a call in the chat '
                    "and it is saved here with its basis and any literature tension.</div>",
                    unsafe_allow_html=True)
        return
    for n in sorted(notes, key=lambda x: (x.created_at, x.id), reverse=True):
        scope = (f"cluster {n.scope_ref.cluster}" if n.scope == "cluster" and n.scope_ref.cluster
                 else {"dataset": "this dataset", "lab": "lab-wide"}.get(n.scope, n.scope))
        date = n.created_at.split("T", 1)[0] if n.created_at else "n/a"
        st.markdown(
            f'<div class="pano-lk-card"><div class="pano-lk-claim">{html.escape(n.claim)}</div>'
            f'<div class="pano-lk-meta"><span class="sco">{html.escape(scope)}</span> · '
            f'basis: {html.escape(n.basis)} · {html.escape(n.status)} · {html.escape(n.author or "you")} · {html.escape(date)}</div></div>',
            unsafe_allow_html=True,
        )


def render_summary_page() -> None:
    """The final, integrated step: marker + enrichment result tables, then an
    editable per-cluster working space (identity + programs), a cross-cluster
    global check, the caveats, and .docx/PDF export of the edited write-up."""
    import streamlit as st

    st.markdown(f"<style>{_SUMMARY_CSS}</style>", unsafe_allow_html=True)

    verdicts = da.all_verdicts()
    n_flagged = sum(1 for v in verdicts if v.verify)
    try:
        n_panel = len(da.panel_names())
    except Exception:  # noqa: BLE001 - a missing panel count is not fatal to the page
        n_panel = 0
    try:
        enrichments = da.all_enrichments()
    except Exception:  # noqa: BLE001 - no enrichment slice -> marker-only summary
        enrichments = []

    flagged_txt = (f'<span class="n">{n_flagged}</span> flagged for re-check'
                   if n_flagged else "none flagged for re-check")
    st.markdown(
        f'<div class="pano-sum-title">{html.escape(_TITLE)}</div>'
        f'<div class="pano-sum-sub">{html.escape(_SUB)}</div>'
        '<div class="pano-sum-meta">'
        f'<span class="n">{len(verdicts)}</span> clusters<span class="sep">·</span>{flagged_txt}'
        f'<span class="sep">·</span><span class="n">{n_panel}</span>-gene panel</div>',
        unsafe_allow_html=True,
    )

    # 1) Marker genes result table
    _sec_head(st, "Result · 1 of 2", "Marker genes — cell-type annotation")
    st.markdown(_table_html(verdicts), unsafe_allow_html=True)
    with st.container(key="pano_dl"):
        st.download_button(_DOWNLOAD_LABEL, da.verdict_csv(), _DOWNLOAD_NAME, _DOWNLOAD_MIME)

    # 2) Enrichment result table
    _sec_head(st, "Result · 2 of 2", "Gene-set enrichment — active programs (panel-scoped)")
    if enrichments:
        st.markdown(_enrichment_table_html(enrichments), unsafe_allow_html=True)
    else:
        st.markdown('<div class="pano-lk-empty">No enrichment result for this dataset yet.</div>',
                    unsafe_allow_html=True)

    # 3) Working space — editable per-cluster synthesis + global check + caveats.
    # Compute every region's latest auto-seed up front so ONE refresh button can
    # re-draft the whole write-up at once.
    rep = report.build_report_from_sources(generated_at=datetime.date.today().isoformat())
    try:
        themes = da.pathway_themes()
    except Exception:  # noqa: BLE001
        themes = None
    enr_map = {ce.cluster: ce for ce in enrichments}
    ws_defaults: dict[str, str] = {
        f"ws_{s.cluster}": report.default_cluster_summary(s) for s in rep.sections
    }
    ws_defaults["ws_global"] = report.global_check_text(da.holistic(), themes)
    ws_defaults["ws_caveats"] = report.caveats_text(verdicts, enr_map, n_panel)

    _sec_head(st, "Working space · editable", "Final interpretation",
              "Auto-drafted per cluster from both workflows (identity from markers · programs from "
              "enrichment · live-cited biology) and your lab notes. Edit any region directly; your "
              "edits persist. This is exactly what exports below.")
    with st.container(key="pano_ws_refresh"):
        st.button("↻ refresh all from latest", key="btn_ws_refresh",
                  on_click=_reset_ws_all, args=(ws_defaults,),
                  help="Re-draft every region below from the latest calls, programs, and lab notes "
                       "(e.g. after chatting on the Marker genes / Pathways pages).")

    for s in rep.sections:
        color = fmt.cluster_color(s.cluster)
        css, _ = fmt.confidence_chip(s.confidence)
        flag = ' <span class="pano-sum-flag">⚑ re-check</span>' if s.verify else ""
        st.markdown(
            f'<div class="pano-ws-head"><span class="dot" style="background:{color}"></span>'
            f'<span class="cid">{html.escape(s.cluster)}</span>'
            f'<span class="ct">{html.escape(s.cell_type)}</span>'
            f'<span class="cf {css}">{html.escape(s.confidence)}</span>{flag}</div>',
            unsafe_allow_html=True,
        )
        _editable(st, f"ws_{s.cluster}", ws_defaults[f"ws_{s.cluster}"], height=200)

    _sec_head(st, "Working space", "Cross-cluster global check")
    _editable(st, "ws_global", ws_defaults["ws_global"], height=180)

    _sec_head(st, "Working space", "Caveats")
    _editable(st, "ws_caveats", ws_defaults["ws_caveats"], height=200)

    # Export the EDITED working space (per-cluster + global check + caveats + lab notes).
    export_sections = [
        (s.cluster, s.cell_type, s.confidence, s.verify, st.session_state.get(f"ws_{s.cluster}", ""))
        for s in rep.sections
    ]
    ga = datetime.date.today().isoformat()
    kw = dict(
        dataset=rep.dataset, generated_at=ga,
        global_check=st.session_state.get("ws_global", ""),
        caveats=st.session_state.get("ws_caveats", ""),
        lab_notes=rep.dataset_notes,
    )
    with st.container(key="pano_ws_dl"):
        st.download_button(_DOCX_LABEL, report.working_docx(export_sections, **kw), _DOCX_NAME, _DOCX_MIME)
        st.download_button(_PDF_LABEL, report.working_pdf(export_sections, **kw), _PDF_NAME, _PDF_MIME)

    # 4) Lab knowledge — integrated here (its own tab is retired)
    _render_lab_notes(st, da.read_notes())


__all__ = ["render_summary_page"]
