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
  read from the pipeline's Output-4 notes (``ui.data_access.gene_note`` /
  ``pipeline/stages/notes.py``); this module never generates biology
  (confident floor). Genes without a note show nothing.

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
# Pure HTML fragments (escape all source-derived text)
# --------------------------------------------------------------------------- #
def _num_html(ev: MarkerEvidence) -> str:
    """The numbers cell: glm_coef and pearson. (The canonical tag now sits next to
    the gene name in the gene column, via the button's ::after, not here.)"""
    glm = fmt.num_fmt(ev.glm_coef, 2)
    pear = fmt.num_fmt(ev.pearson, 2)
    return (
        f'<div class="pano-ev-num"><span class="num">{glm}</span>'
        f'<span class="num dim pano-ev-sub">pearson {pear}</span></div>'
    )


# The one jazzPanda-derived caveat we surface as a specificity badge: the gene's
# transcripts correlate more with ANOTHER cluster's spatial pattern than with this
# one (max_gc_corr > pearson). The skill treats this as a specificity caveat — the
# marker also marks another lineage — so we flag it at a glance.
_SPECIFICITY_CAVEAT = "localizes better with another cluster"


def _caveat_badge(ev: MarkerEvidence) -> str:
    """A small specificity badge when jazzPanda says this marker localizes better
    with another cluster (Tier-A evidence, not the note). Empty otherwise."""
    if _SPECIFICITY_CAVEAT in ev.caveats:
        return (
            ' <span class="pano-bio-caveat" '
            'title="jazzPanda: transcripts localize better with another cluster '
            '(max_gc_corr &gt; pearson) — also marks another lineage">'
            "also marks another cluster</span>"
        )
    return ""


def _bio_html(cluster: str, ev: MarkerEvidence) -> str:
    """The biology cell: the precomputed grounded note + its real citation + a
    jazzPanda-derived specificity badge.

    Reads ``data_access.gene_note`` (precomputed, cited) for the prose; never
    generates text. The specificity badge is driven off the verdict's own
    ``ev.caveats`` (Tier A), so it shows even when a gene has no biology note. A
    thin-literature note is labelled honestly; a verify-flagged note carries a
    small check marker.
    """
    caveat = _caveat_badge(ev)
    note = data_access.gene_note(cluster, ev.gene)
    if not note or not note.get("summary"):
        # No prose, but a grounded caveat still belongs here if present.
        return f'<div class="pano-ev-bio empty">{caveat}</div>'

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
    return f'<div class="pano-ev-bio">{summary} {cite}{verify}{caveat}</div>'


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
/* Specificity badge: this marker localizes better with another cluster
   (jazzPanda max_gc_corr > pearson). A small outlined amber chip — cautionary,
   not alarming, and visually distinct from the teal canonical tag. */
.pano-bio-caveat {
  display: inline-block; font-family: var(--mono); font-size: 9px;
  letter-spacing: .02em; color: var(--absent); border: 1px solid var(--absent);
  border-radius: 4px; padding: 0 5px; margin-left: 2px; white-space: nowrap;
  vertical-align: baseline; opacity: .9;
}
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
}
/* Scrollable rows body: no boxed border (border=False on the container); just a
   single top hairline so it reads as the table body under the header row. */
div[class*="st-key-evrows"] {
  border-top: 1px solid var(--hair) !important;
  padding-top: 2px;
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
/* Canonical marker: a small tag right after the gene name (canonical rows only). */
div[class*="st-key-generow_canon_"] div[data-testid="stButton"] > button::after {
  content: 'canonical'; align-self: center; flex: none; margin-left: 8px;
  font-family: var(--mono); font-size: 9px; font-weight: 500; line-height: 1.5;
  color: var(--accent); background: var(--accent-soft);
  padding: 1px 6px; border-radius: 4px;
}
</style>
"""

# Column proportions shared by the header and every row so they line up.
_COLS = [0.24, 0.20, 0.56]

# The marker rows live in one fixed-height, scrollable panel (all markers at once,
# scroll for the long tail) rather than a top-N view plus a "show all" expander.
_EV_SCROLL_HEIGHT = 360


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
def render_evidence_table(cluster: str) -> None:
    """Render the marker-evidence table for ``cluster`` into the current column."""
    import streamlit as st

    verdict: ClusterVerdict = data_access.verdict_for(cluster)

    st.markdown(_EVIDENCE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="pano-eyebrow">Assigned markers '
        f"({len(verdict.evidence)} markers)</p>",
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

    all_rows = list(verdict.evidence)
    if not all_rows:
        st.markdown(
            '<div class="pano-ev-empty">No assigned markers for this cluster.</div>',
            unsafe_allow_html=True,
        )
        return

    # Every assigned marker in one fixed-height, scrollable panel — scroll for the
    # long tail rather than an expander that pushes the tissue down the page.
    # border=False drops Streamlit's heavy default box; a single top hairline (CSS)
    # ties the scrolling body to the header above it, table-style.
    with st.container(height=_EV_SCROLL_HEIGHT, border=False, key="evrows"):
        _render_marker_rows(st, all_rows, cluster)


def _render_marker_rows(st, rows, cluster: str) -> None:
    """Render each marker as ``[gene button | numbers | cited biology note]``.

    Called inside the scrollable rows container. The gene button (dot ::before +
    name) toggles this ``cluster``'s marker set (per-cluster selection) in-line; the
    evidence table renders BEFORE the spatial stage in the same run, so the grid
    sees the change immediately. No recompute.
    """
    for ev in rows:
        is_selected = state.is_marker_selected(cluster, ev.gene)
        # Canonical markers get a `canonical` tag after the gene name via CSS
        # (::after on the button), keyed by the container name below.
        row_key = f"generow_canon_{ev.gene}" if ev.is_canonical else f"generow_{ev.gene}"
        with st.container(key=row_key):
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
                st.markdown(_bio_html(cluster, ev), unsafe_allow_html=True)


__all__ = ["render_evidence_table"]
