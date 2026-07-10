"""Summary page — the two-pane report editor (the final, integrated step).

Both workflows land here. The page reads like a manuscript you finalise:

* a **contents rail** on the left — ``Overall`` (the whole-dataset map + the
  cross-cluster check), then every cluster, then ``Caveats`` and ``Lab notes``;
* a **right pane** that shows exactly one section at a time: the Overall page is
  the merged overview table + the editable global check; a cluster page is that
  cluster's editable identity+programs write-up; the dataset pages are their own
  editors / note cards.

Grounding is preserved end to end — nothing here computes a value. The overview is
a straight projection of :func:`ui.data_access.all_verdicts` +
:func:`~ui.data_access.all_enrichments`; each draft is
:func:`ui.report.default_cluster_summary` / ``global_check_text`` / ``caveats_text``
(live-cited, PMID-carrying); the exports are byte-for-byte what you edit.

"Auto-seed, edits win": each editable region seeds once from the latest draft and
then keeps your edits (held in plain ``wsval_*`` session keys, so switching sections
never drops one). One "refresh all" re-drafts every region from the freshest calls,
programs, and lab notes.

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
# Copy + export metadata. Constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Interpretation summary"
_SUB = (
    "Both workflows in one report — the marker-gene calls and the gene-set programs, "
    "drafted per cluster for you to edit and export. Every number is jazzPanda's, every "
    "biology claim is live-cited, nothing here recomputes a value."
)

_CSV_LABEL = "⬇  Annotations (CSV)"
_CSV_NAME = "panoscope_annotations.csv"
_CSV_MIME = "text/csv"
_DOCX_LABEL = "⬇  Word"
_DOCX_NAME = "panoscope_interpretation.docx"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_PDF_LABEL = "⬇  PDF"
_PDF_NAME = "panoscope_interpretation.pdf"
_PDF_MIME = "application/pdf"

# Rail sections. "overall" hosts the overview + the (editable) cross-cluster check;
# then one entry per cluster; then the two dataset-level editors/cards.
_SEC_OVERALL = "overall"
_SEC_CAVEATS = "caveats"
_SEC_NOTES = "labnotes"
_DATASET_SECTIONS = ((_SEC_CAVEATS, "Caveats"), (_SEC_NOTES, "Lab notes"))
_ED_GLOBAL = "global"   # editor name for the cross-cluster check (lives on Overall)
_K_ACTIVE = "sum_active_section"  # which rail item is focused

# Overview table columns (merged marker + enrichment). (label, width%)
_OVERVIEW_COLS = (
    ("Cluster", 9), ("Cell type", 13), ("Conf.", 10), ("Re-check", 8),
    ("Key markers", 26), ("Enriched programs", 34),
)

# --------------------------------------------------------------------------- #
# Styling. Reuses the theme tokens (var(--sans) etc.) + the global .cf-* pills.
# --------------------------------------------------------------------------- #
_SUMMARY_CSS = """
.pano-sum-title { font-family: var(--sans); font-size: 23px; font-weight: 700;
                  letter-spacing: -.02em; color: var(--ink); margin: 2px 0 5px; }
.pano-sum-sub { font-size: 12.5px; color: var(--muted); line-height: 1.5;
                max-width: 74ch; margin: 0 0 10px; }
.pano-sum-meta { font-family: var(--mono); font-size: 11px; color: var(--faint);
                 display: flex; gap: 9px; align-items: center; margin: 0; }
.pano-sum-meta .n { color: var(--ink); font-weight: 600; }
.pano-sum-meta .sep { color: var(--hair); }

/* Top action buttons (refresh + Word + PDF) — laid out in three columns. */
.st-key-btn_ws_refresh button, .st-key-dl_docx button, .st-key-dl_pdf button {
  border-radius: 9px !important; font-size: 12px !important; white-space: nowrap;
  min-height: 0 !important; padding: 7px 10px !important; }
.st-key-btn_ws_refresh button { background: transparent !important; border: 1px solid var(--hair) !important;
  color: var(--faint) !important; box-shadow: none !important; }
.st-key-btn_ws_refresh button:hover { color: var(--accent) !important; border-color: var(--accent) !important;
  background: var(--accent-soft) !important; }

/* Left contents rail — a sticky table of contents of buttons. */
.st-key-pano_rail { position: sticky; top: 68px; align-self: start;
  border: 1px solid var(--hair); border-radius: 14px; background: var(--paper);
  padding: 10px 10px 12px; }
.pano-rail-lbl { font-family: var(--mono); font-size: 9.5px; text-transform: uppercase;
  letter-spacing: .12em; color: var(--faint); font-weight: 600; margin: 12px 4px 4px; }
.pano-rail-lbl:first-child { margin-top: 2px; }
.st-key-pano_rail button { justify-content: flex-start !important; text-align: left !important;
  border: 0 !important; box-shadow: none !important; background: transparent !important;
  color: var(--muted) !important; font-family: var(--sans) !important; font-size: 12.5px !important;
  font-weight: 500 !important; padding: 6px 9px !important; min-height: 0 !important;
  border-radius: 8px !important; }
.st-key-pano_rail button:hover { background: #F4F7F8 !important; color: var(--ink) !important; }
.st-key-pano_rail button[kind="primary"] { background: var(--accent-soft) !important;
  color: var(--accent) !important; font-weight: 700 !important; }

/* Overview table (marker + programs merged). */
.pano-ov-cap { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); font-weight: 600; margin: 2px 0 8px; }
.pano-sum-tablewrap { background: var(--paper); border: 1px solid var(--hair);
                      border-radius: 14px; overflow: hidden; }
.pano-sum-table { width: 100%; border-collapse: collapse; table-layout: fixed; font-family: var(--sans); }
.pano-sum-table thead th { text-align: left; font-family: var(--mono); font-size: 9.5px;
  text-transform: uppercase; letter-spacing: .06em; color: var(--faint); font-weight: 500;
  padding: 11px 13px; border-bottom: 1px solid var(--hair); background: #FAFBFB; }
.pano-sum-table tbody td { padding: 12px 13px; border-bottom: 1px solid var(--hair2); vertical-align: top; }
.pano-sum-table tbody tr:last-child td { border-bottom: 0; }
.pano-sum-table tbody tr:hover td { background: #FAFCFC; }
.pano-sum-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%;
                margin-right: 8px; vertical-align: middle; }
.pano-sum-cid { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink); vertical-align: middle; }
.pano-sum-ct { font-size: 13.5px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; }
.pano-sum-km { font-family: var(--mono); font-size: 11px; color: var(--muted); line-height: 1.6; word-break: break-word; }
.pano-sum-desc { font-size: 12px; color: var(--muted); line-height: 1.5; }
.pano-enr-le { font-family: var(--mono); font-size: 10px; color: var(--faint); line-height: 1.55;
  word-break: break-word; display: block; margin-top: 3px; }
.pano-sum-dash { color: var(--faint); }
.pano-sum-flag { font-family: var(--mono); font-size: 9.5px; font-weight: 600; color: var(--absent);
  background: var(--absent-bg); padding: 2px 7px; border-radius: 6px; white-space: nowrap; }
.pano-sum-table .cf { font-size: 10px; padding: 2px 8px; }
.st-key-pano_csv { display: flex; justify-content: flex-end; margin-top: 10px; }
.st-key-pano_csv button { background: transparent !important; border: 1px solid var(--hair) !important;
  color: var(--faint) !important; box-shadow: none !important; border-radius: 8px !important;
  font-size: 11px !important; padding: 4px 11px !important; min-height: 0 !important; }
.st-key-pano_csv button:hover { color: var(--accent) !important; border-color: var(--accent) !important; }

/* Focused editor. */
.pano-ed-hr { border: 0; border-top: 1px solid var(--hair); margin: 24px 0 4px; }
.pano-ed-eyebrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .11em; color: var(--accent); font-weight: 600; margin: 4px 0 10px; }
.pano-ws-head { display: flex; align-items: baseline; gap: 10px; padding: 0 0 10px; flex-wrap: wrap; }
.pano-ws-head .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.pano-ws-head .cid { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.pano-ws-head .ct { font-size: 18px; font-weight: 700; color: var(--ink); letter-spacing: -.01em; }
.pano-ws-head .cf { font-size: 10px; padding: 2px 8px; }
.pano-ed-hint { font-size: 11.5px; color: var(--faint); margin: 6px 0 0; }
div[class*="st-key-wsw_"] textarea { font-family: var(--sans) !important; font-size: 13.5px !important;
  line-height: 1.65 !important; border-radius: 12px !important; border: 1px solid var(--hair) !important;
  background: #FCFCFD !important; color: var(--ink) !important; }
div[class*="st-key-wsw_"] textarea:focus { border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important; background: var(--paper) !important; }

/* Lab-note cards (read-only, inside the Lab notes section). */
.pano-lk-card { border: 1px solid var(--hair); border-radius: 11px; padding: 11px 13px;
  margin: 8px 0; background: var(--paper); }
.pano-lk-claim { font-size: 13px; color: var(--ink); line-height: 1.5; }
.pano-lk-meta { font-family: var(--mono); font-size: 10px; color: var(--faint); margin-top: 7px; }
.pano-lk-meta .sco { background: var(--accent-soft); color: var(--accent); padding: 2px 7px; border-radius: 5px; }
.pano-lk-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); border: 1px dashed var(--hair);
  border-radius: 10px; padding: 16px; text-align: center; }
"""


# --------------------------------------------------------------------------- #
# Small pure HTML cell builders (grounded projections — no value invented).
# --------------------------------------------------------------------------- #
def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


def _dot_id(cluster: str) -> str:
    color = fmt.cluster_color(cluster)
    return (f'<span class="pano-sum-dot" style="background:{color}"></span>'
            f'<span class="pano-sum-cid">{html.escape(cluster)}</span>')


def _conf_pill(confidence: str) -> str:
    css, _ = fmt.confidence_chip(confidence)
    return f'<span class="cf {css}">{html.escape(confidence)}</span>'


def _markers_cell(v: ClusterVerdict) -> str:
    if not v.key_markers:
        return '<span class="pano-sum-dash">—</span>'
    return f'<span class="pano-sum-km">{" · ".join(html.escape(str(g)) for g in v.key_markers)}</span>'


def _recheck_cell(verify: bool) -> str:
    if verify:
        return '<span class="pano-sum-flag" title="flagged for re-check">⚑ re-check</span>'
    return '<span class="pano-sum-dash">—</span>'


def _programs_cell(ce) -> str:
    """Top enriched programs + their leading-edge genes, compact. Dash if none."""
    if ce is None or not ce.enriched:
        return '<span class="pano-sum-dash">no program clears the gate</span>'
    progs = " · ".join(_short(p.gene_set) for p in ce.enriched[:4])
    le: list[str] = []
    for p in ce.enriched[:3]:
        for g in p.leading_edge:
            if g not in le:
                le.append(g)
    le_html = f'<span class="pano-enr-le">{" · ".join(html.escape(g) for g in le[:9])}</span>' if le else ""
    return f'<span class="pano-sum-desc">{html.escape(progs)}</span>{le_html}'


def _overview_table_html(verdicts: list[ClusterVerdict], enr_map: dict) -> str:
    """The one scannable map: per cluster, the marker call and the enriched programs."""
    cols = "".join(f'<col style="width:{w}%">' for _, w in _OVERVIEW_COLS)
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in _OVERVIEW_COLS)
    rows = []
    for v in verdicts:
        ce = enr_map.get(v.cluster)
        rows.append(
            "<tr>"
            f"<td>{_dot_id(v.cluster)}</td>"
            f'<td><span class="pano-sum-ct">{html.escape(v.cell_type)}</span></td>'
            f"<td>{_conf_pill(v.confidence)}</td>"
            f"<td>{_recheck_cell(v.verify)}</td>"
            f"<td>{_markers_cell(v)}</td>"
            f"<td>{_programs_cell(ce)}</td>"
            "</tr>"
        )
    return (
        '<div class="pano-sum-tablewrap"><table class="pano-sum-table">'
        f"<colgroup>{cols}</colgroup><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _ws_head_html(cluster: str, cell_type: str, confidence: str, verify: bool) -> str:
    color = fmt.cluster_color(cluster)
    css, _ = fmt.confidence_chip(confidence)
    flag = ' <span class="pano-sum-flag">⚑ re-check</span>' if verify else ""
    return (
        f'<div class="pano-ws-head"><span class="dot" style="background:{color}"></span>'
        f'<span class="cid">{html.escape(cluster)}</span>'
        f'<span class="ct">{html.escape(cell_type)}</span>'
        f'<span class="cf {css}">{html.escape(confidence)}</span>{flag}</div>'
    )


# --------------------------------------------------------------------------- #
# Editable working space — one section rendered at a time; edits held in plain
# ``wsval_*`` keys so navigating away never drops them.
# --------------------------------------------------------------------------- #
def _reset_ws_all(defaults: dict) -> None:
    """on_click: re-draft EVERY region from its freshest auto-seed. Pops the mounted
    widget key so the visible editor re-seeds from the new default."""
    import streamlit as st

    for name, default in defaults.items():
        st.session_state[f"wsval_{name}"] = default
        st.session_state.pop(f"wsw_{name}", None)


def _set_active(section: str) -> None:
    """on_click: focus a rail section."""
    import streamlit as st

    st.session_state[_K_ACTIVE] = section


def _editor(st, name: str, default: str, height: int) -> None:
    """Render one editable region. Canonical text lives in ``wsval_{name}`` (a plain
    key that survives when the widget is unmounted); the ``wsw_{name}`` widget seeds
    from it. The caller reconciles ``wsval`` from ``wsw`` before this runs."""
    vkey = f"wsval_{name}"
    if vkey not in st.session_state:
        st.session_state[vkey] = default
    val = st.text_area("edit", value=st.session_state[vkey], key=f"wsw_{name}",
                       height=height, label_visibility="collapsed")
    st.session_state[vkey] = val


def _eyebrow(st, text: str) -> None:
    st.markdown(f'<div class="pano-ed-eyebrow">{html.escape(text)}</div>', unsafe_allow_html=True)


def _render_lab_note_cards(st, notes: list) -> None:
    if not notes:
        st.markdown('<div class="pano-lk-empty">No notes yet. Override or confirm a call in the '
                    "chat and it is saved here with its basis and any literature tension.</div>",
                    unsafe_allow_html=True)
        return
    for n in sorted(notes, key=lambda x: (x.created_at, x.id), reverse=True):
        scope = (f"cluster {n.scope_ref.cluster}" if n.scope == "cluster" and n.scope_ref.cluster
                 else {"dataset": "this dataset", "lab": "lab-wide"}.get(n.scope, n.scope))
        date = n.created_at.split("T", 1)[0] if n.created_at else "n/a"
        st.markdown(
            f'<div class="pano-lk-card"><div class="pano-lk-claim">{html.escape(n.claim)}</div>'
            f'<div class="pano-lk-meta"><span class="sco">{html.escape(scope)}</span> · '
            f'basis: {html.escape(n.basis)} · {html.escape(n.status)} · '
            f'{html.escape(n.author or "you")} · {html.escape(date)}</div></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def render_summary_page() -> None:
    """Two-pane report editor: contents rail (Overall + clusters + dataset) on the
    left, one focused section on the right."""
    import streamlit as st

    st.markdown(f"<style>{_SUMMARY_CSS}</style>", unsafe_allow_html=True)

    verdicts = da.all_verdicts()
    sec_by_id = {v.cluster: v for v in verdicts}
    n_flagged = sum(1 for v in verdicts if v.verify)
    try:
        n_panel = len(da.panel_names())
    except Exception:  # noqa: BLE001 - a missing panel count is not fatal to the page
        n_panel = 0
    try:
        enrichments = da.all_enrichments()
    except Exception:  # noqa: BLE001 - no enrichment slice -> marker-only summary
        enrichments = []
    enr_map = {ce.cluster: ce for ce in enrichments}

    # Latest per-section drafts (grounded). Computed up front so one button can
    # re-draft everything and the exports read the freshest text.
    rep = report.build_report_from_sources(generated_at=datetime.date.today().isoformat())
    try:
        themes = da.pathway_themes()
    except Exception:  # noqa: BLE001
        themes = None
    ws_defaults: dict[str, str] = {
        s.cluster: report.default_cluster_summary(s) for s in rep.sections
    }
    ws_defaults[_ED_GLOBAL] = report.global_check_text(da.holistic(), themes)
    ws_defaults[_SEC_CAVEATS] = report.caveats_text(verdicts, enr_map, n_panel)

    # Reconcile canonical text from the (possibly just-edited) mounted widget BEFORE
    # exports are built, so a download always reflects the latest keystrokes.
    for name in ws_defaults:
        wkey = f"wsw_{name}"
        if wkey in st.session_state:
            st.session_state[f"wsval_{name}"] = st.session_state[wkey]

    def _val(name: str) -> str:
        return st.session_state.get(f"wsval_{name}", ws_defaults[name])

    active = st.session_state.get(_K_ACTIVE, _SEC_OVERALL)

    export_sections = [
        (s.cluster, s.cell_type, s.confidence, s.verify, _val(s.cluster))
        for s in rep.sections
    ]
    ga = datetime.date.today().isoformat()
    exp_kw = dict(dataset=rep.dataset, generated_at=ga,
                  global_check=_val(_ED_GLOBAL), caveats=_val(_SEC_CAVEATS),
                  lab_notes=rep.dataset_notes)

    # ---- Top bar: title + meta | [refresh] [Word] [PDF] ------------------ #
    head_col, act_col = st.columns([0.58, 0.42], vertical_alignment="center")
    with head_col:
        flagged_txt = (f'<span class="n">{n_flagged}</span> flagged for re-check'
                       if n_flagged else "none flagged")
        st.markdown(
            f'<div class="pano-sum-title">{html.escape(_TITLE)}</div>'
            f'<div class="pano-sum-sub">{html.escape(_SUB)}</div>'
            '<div class="pano-sum-meta">'
            f'<span class="n">{len(verdicts)}</span> clusters<span class="sep">·</span>{flagged_txt}'
            f'<span class="sep">·</span><span class="n">{n_panel}</span>-gene panel</div>',
            unsafe_allow_html=True,
        )
    with act_col:
        b_ref, b_doc, b_pdf = st.columns(3)
        with b_ref:
            st.button("↻ refresh", key="btn_ws_refresh", use_container_width=True,
                      on_click=_reset_ws_all, args=(ws_defaults,),
                      help="Re-draft every region from the latest calls, programs, and lab notes.")
        with b_doc:
            st.download_button(_DOCX_LABEL, report.working_docx(export_sections, **exp_kw),
                               _DOCX_NAME, _DOCX_MIME, key="dl_docx", use_container_width=True)
        with b_pdf:
            st.download_button(_PDF_LABEL, report.working_pdf(export_sections, **exp_kw),
                               _PDF_NAME, _PDF_MIME, key="dl_pdf", use_container_width=True)

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)

    # ---- Two panes: contents rail | focused section ---------------------- #
    rail_col, pane_col = st.columns([0.24, 0.76], gap="large")

    with rail_col:
        with st.container(key="pano_rail"):
            st.markdown('<div class="pano-rail-lbl">Report</div>', unsafe_allow_html=True)
            st.button("Overall", key="nav_overall", use_container_width=True,
                      type="primary" if active == _SEC_OVERALL else "secondary",
                      on_click=_set_active, args=(_SEC_OVERALL,))
            st.markdown('<div class="pano-rail-lbl">Clusters</div>', unsafe_allow_html=True)
            for s in rep.sections:
                label = f"{s.cluster} · {s.cell_type}" + ("  ⚑" if s.verify else "")
                st.button(label, key=f"nav_{s.cluster}", use_container_width=True,
                          type="primary" if active == s.cluster else "secondary",
                          on_click=_set_active, args=(s.cluster,))
            st.markdown('<div class="pano-rail-lbl">Dataset</div>', unsafe_allow_html=True)
            for sec, label in _DATASET_SECTIONS:
                st.button(label, key=f"nav_{sec}", use_container_width=True,
                          type="primary" if active == sec else "secondary",
                          on_click=_set_active, args=(sec,))

    with pane_col:
        if active == _SEC_OVERALL:
            st.markdown('<div class="pano-ov-cap">Overview · marker call + enriched programs</div>',
                        unsafe_allow_html=True)
            st.markdown(_overview_table_html(verdicts, enr_map), unsafe_allow_html=True)
            with st.container(key="pano_csv"):
                st.download_button(_CSV_LABEL, da.verdict_csv(), _CSV_NAME, _CSV_MIME, key="dl_csv")
            st.markdown('<hr class="pano-ed-hr"/>', unsafe_allow_html=True)
            _eyebrow(st, "Dataset · cross-cluster global check")
            _editor(st, _ED_GLOBAL, ws_defaults[_ED_GLOBAL], height=240)

        elif active in sec_by_id:
            s = sec_by_id[active]
            _eyebrow(st, "Cluster interpretation · identity + programs")
            st.markdown(_ws_head_html(s.cluster, s.cell_type, s.confidence, s.verify),
                        unsafe_allow_html=True)
            _editor(st, s.cluster, ws_defaults[s.cluster], height=320)
            st.markdown('<div class="pano-ed-hint">Edit freely — this is exactly what exports for '
                        f"{html.escape(s.cluster)}.</div>", unsafe_allow_html=True)

        elif active == _SEC_CAVEATS:
            _eyebrow(st, "Dataset · caveats")
            _editor(st, _SEC_CAVEATS, ws_defaults[_SEC_CAVEATS], height=300)

        else:  # _SEC_NOTES (or any stale value) -> lab notes
            _eyebrow(st, "Lab knowledge · your saved notes")
            st.markdown('<div class="pano-ed-hint" style="margin:0 0 10px">Captured when you override '
                        "or confirm a call in the chat; compiled into the write-up and the exported "
                        "report.</div>", unsafe_allow_html=True)
            _render_lab_note_cards(st, da.read_notes())


__all__ = ["render_summary_page"]
