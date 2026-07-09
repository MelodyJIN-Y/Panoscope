"""Evidence table — the marker rows a cluster's call rests on.

``render_evidence_table(cluster)`` draws one cluster's marker evidence. Each row is:

* a **gene button** — the gene name with a small select dot as its ``::before``
  (a hollow ring when unselected, a filled teal disc when selected, exactly like
  the cluster-rail dot). Clicking it toggles the gene in this cluster's
  multi-select set (``ui.state.toggle_marker``) — a viewing control that drives
  the linked spatial small-multiples, never a recompute.
* its **jazzPanda numbers** (glm_coef + pearson) and a ``canonical`` tag when the
  gene is a canonical marker for the cell type.
* a **grounded biology note** — a short summary of the gene's role and its
  relevance to this cluster's identity, WITH a real PubMed citation. This text is
  read from precomputed, cited notes (``ui.data_access.gene_note`` /
  ``scripts/precompute_gene_notes.py``); this module never generates biology
  (confident floor). Genes without a note show an em dash.

Every value comes straight off the cached ``ClusterVerdict`` / the precomputed
notes — this module computes nothing. Streamlit is imported lazily so importing
``ui.evidence_table`` needs no running server.
"""

from __future__ import annotations

import html
from typing import Optional

from agent.types import ClusterVerdict, MarkerEvidence

from ui import data_access, format as fmt, state

# --------------------------------------------------------------------------- #
# How many rows to show. A cluster can carry dozens of assigned markers (c1 has
# 84); showing all buries the drivers. Keep every canonical marker (the drivers
# the call rests on), then top up with the strongest non-canonical supporters by
# glm_coef until the cap. This only *selects* rows the verdict already produced.
# --------------------------------------------------------------------------- #
MAX_SUPPORT_ROWS: int = 8


def _rows_to_show(verdict: ClusterVerdict) -> list[MarkerEvidence]:
    """Canonical markers (always) + strongest non-canonical supporters, capped."""
    canonical = [e for e in verdict.evidence if e.is_canonical]
    non_canonical = [e for e in verdict.evidence if not e.is_canonical]
    budget = max(0, MAX_SUPPORT_ROWS - len(canonical))
    kept = {id(e) for e in canonical} | {id(e) for e in non_canonical[:budget]}
    return [e for e in verdict.evidence if id(e) in kept]


# --------------------------------------------------------------------------- #
# Pure HTML fragments (escape all source-derived text)
# --------------------------------------------------------------------------- #
def _num_html(ev: MarkerEvidence) -> str:
    """The numbers cell: a canonical tag (if any), glm_coef, and pearson."""
    glm = fmt.num_fmt(ev.glm_coef, 2)
    pear = fmt.num_fmt(ev.pearson, 2)
    canon = (
        '<span class="pano-canon" title="canonical marker for this cell type">canonical</span>'
        if ev.is_canonical
        else ""
    )
    return (
        f'<div class="pano-ev-num">{canon}'
        f'<span class="num">{glm}</span>'
        f'<span class="num dim pano-ev-sub">pearson {pear}</span></div>'
    )


def _bio_html(cluster: str, gene: str) -> str:
    """The biology cell: the precomputed grounded note + its real citation.

    Reads ``data_access.gene_note`` (precomputed, cited). Never generates text: a
    gene without a note shows an em dash. A thin-literature note is labelled
    honestly; a verify-flagged note carries a small check marker.
    """
    note = data_access.gene_note(cluster, gene)
    if not note or not note.get("summary"):
        return '<div class="pano-ev-bio empty"></div>'

    summary = html.escape(str(note["summary"]))
    pmid = note.get("pmid")
    if pmid:
        cite = (
            f'<a class="pano-bio-cite" '
            f'href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(str(pmid))}/" '
            f'target="_blank">&#128196; PMID:{html.escape(str(pmid))}</a>'
        )
    else:
        cite = '<span class="pano-bio-thin">literature thin</span>'
    verify = (
        ' <span class="pano-bio-verify" title="re-check this">&#9873; check</span>'
        if note.get("verify")
        else ""
    )
    return f'<div class="pano-ev-bio">{summary} {cite}{verify}</div>'


# --------------------------------------------------------------------------- #
# One-time CSS. The gene button's dot is a ::before ring/disc (scoped to the
# per-row keyed container st-key-generow_*), matching the cluster-rail dot.
# --------------------------------------------------------------------------- #
_EVIDENCE_CSS = """
<style>
.pano-ev-hlabel {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--faint); font-weight: 500;
}
.pano-ev-num { display: flex; flex-direction: column; line-height: 1.3; gap: 2px; }
.pano-ev-sub { font-size: 10px; }
.pano-canon {
  font-family: var(--mono); font-size: 9px; color: var(--accent);
  background: var(--accent-soft); padding: 1px 6px; border-radius: 4px;
  align-self: flex-start; margin-bottom: 2px;
}
.pano-ev-bio {
  font-size: 12px; color: var(--muted); line-height: 1.5;
}
.pano-ev-bio.empty { color: var(--faint); }
.pano-bio-cite {
  font-family: var(--mono); font-size: 10px; color: var(--accent);
  white-space: nowrap; text-decoration: none;
}
.pano-bio-cite:hover { text-decoration: underline; }
.pano-bio-thin { font-family: var(--mono); font-size: 10px; color: var(--faint); }
.pano-bio-verify { font-family: var(--mono); font-size: 10px; color: var(--absent); }
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
}
/* Per-row keyed container: tighten spacing + render the gene button as a
   chromeless, left-aligned name with a ::before select dot (like the rail). */
div[class*="st-key-evrows"] [data-testid="stHorizontalBlock"] { gap: 8px !important; }
div[class*="st-key-generow_"] [data-testid="stColumn"] { padding: 0 !important; }
div[class*="st-key-generow_"] div[data-testid="stButton"] > button {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  min-height: 0 !important; padding: 4px 4px !important; border-radius: 6px !important;
  display: flex !important; align-items: center !important; justify-content: flex-start !important;
  font-family: var(--mono) !important; font-weight: 600 !important; font-size: 14px !important;
  color: var(--ink) !important;
}
div[class*="st-key-generow_"] div[data-testid="stButton"] > button > div {
  flex: 0 1 auto !important; width: auto !important;
}
div[class*="st-key-generow_"] div[data-testid="stButton"] > button::before {
  content: ''; width: 10px; height: 10px; border-radius: 50%; flex: none;
  margin-right: 9px; box-sizing: border-box;
  border: 1.5px solid var(--faint); background: transparent;
  transition: background 120ms ease, border-color 120ms ease;
}
div[class*="st-key-generow_"] button[kind="primary"]::before,
div[class*="st-key-generow_"] button[data-testid="stBaseButton-primary"]::before {
  background: var(--accent) !important; border-color: var(--accent) !important;
}
div[class*="st-key-generow_"] button[kind="primary"],
div[class*="st-key-generow_"] button[data-testid="stBaseButton-primary"] {
  color: var(--accent) !important; background: transparent !important;
}
div[class*="st-key-generow_"] div[data-testid="stButton"] > button:hover { background: var(--hair2) !important; }
</style>
"""

# Column proportions shared by the header and every row so they line up.
_COLS = [0.24, 0.20, 0.56]


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
def render_evidence_table(cluster: str) -> None:
    """Render the marker-evidence table for ``cluster`` into the current column."""
    import streamlit as st

    verdict: ClusterVerdict = data_access.verdict_for(cluster)

    st.markdown(_EVIDENCE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="pano-eyebrow">Marker evidence · '
        f"{html.escape(verdict.cell_type.replace('_', ' '))} · "
        f"{len(verdict.evidence)} assigned markers</p>",
        unsafe_allow_html=True,
    )

    with st.container(key="evhead"):
        h_gene, h_num, h_bio = st.columns(_COLS)
        h_gene.markdown('<span class="pano-ev-hlabel">Gene</span>', unsafe_allow_html=True)
        h_num.markdown(
            '<span class="pano-ev-hlabel">glm / pearson</span>', unsafe_allow_html=True
        )
        h_bio.markdown(
            '<span class="pano-ev-hlabel">Biology · relevance (cited)</span>',
            unsafe_allow_html=True,
        )

    rows = _rows_to_show(verdict)
    if not rows:
        st.markdown(
            '<div class="pano-ev-empty">No assigned markers for this cluster.</div>',
            unsafe_allow_html=True,
        )

    _render_marker_rows(st, rows, cluster, key="evrows_shown")

    # The long tail (non-driver markers beyond the default view) is tucked behind
    # an expander. Shown and hidden rows are disjoint, so per-gene keys stay unique.
    shown_ids = {id(e) for e in rows}
    hidden = [e for e in verdict.evidence if id(e) not in shown_ids]
    if hidden:
        with st.expander(
            f"Show all {len(verdict.evidence)} assigned markers", expanded=False
        ):
            _render_marker_rows(st, hidden, cluster, key="evrows_hidden")


def _render_marker_rows(st, rows, cluster: str, *, key: str) -> None:
    """Render each marker as ``[gene button | numbers | cited biology note]``.

    The gene button (dot ::before + name) toggles this ``cluster``'s marker set
    (per-cluster selection) in-line — the evidence table renders BEFORE the spatial
    stage in the same run, so the grid sees the change immediately. No recompute.
    """
    with st.container(key=key):
        for ev in rows:
            is_selected = state.is_marker_selected(cluster, ev.gene)
            with st.container(key=f"generow_{ev.gene}"):
                g_col, n_col, b_col = st.columns(_COLS, vertical_alignment="top")
                with g_col:
                    if st.button(
                        str(ev.gene),
                        key=f"select_{ev.gene}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True,
                    ):
                        state.toggle_marker(cluster, ev.gene)
                        st.rerun()
                with n_col:
                    st.markdown(_num_html(ev), unsafe_allow_html=True)
                with b_col:
                    st.markdown(_bio_html(cluster, ev.gene), unsafe_allow_html=True)


__all__ = ["render_evidence_table"]
