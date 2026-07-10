"""Pathways page — gene-set enrichment interpretation (second workflow).

Laid out like the Marker-genes page: a 3-pane shell — the cluster **rail** (shared
selection), the **center stage** (the cluster's enrichment header + a pinnable
pathway table + the pinned pathway's leading-edge spatial views), and a **side
column** with the cross-cluster pathway themes.

Selecting pathways is a viewing control (``ui.state.toggle_pathway``, a per-cluster
multi-select): it drives the spatial stage (``ui.enrichment_spatial``) small-multiples
and nothing else. Every value
is read off the enrichment records (``ui.data_access``); this module computes
nothing and never generates biology text (the per-pathway note is the pipeline's,
live-cited). Each pathway carries its panel-scope caveat; only gated sets show.

Streamlit is imported lazily so ``import ui.enrichment_table`` needs no server.
"""

from __future__ import annotations

import html

from agent.types import ClusterEnrichment, PathwayEvidence

from ui import cluster_rail
from ui import data_access as da
from ui import enrichment_conversation
from ui import enrichment_spatial
from ui import format as fmt
from ui import state

_METHOD_LABEL = {"jazzpanda_enrichment": "jazzPanda competitive test"}


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Styling (reuses theme tokens + the global .cf-* confidence pills)
# --------------------------------------------------------------------------- #
_ENR_CSS = """
.pano-enr-eyebrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); margin: 0 0 10px; }
.pano-enr-head { display: flex; align-items: baseline; gap: 11px; flex-wrap: wrap; margin: 2px 0 3px; }
.pano-enr-head .dot { font-size: 12px; }
.pano-enr-head .cid { font-family: var(--mono); font-size: 12px; color: var(--faint); }
.pano-enr-head .ct { font-size: 22px; font-weight: 700; letter-spacing: -.02em; color: var(--ink); }
.pano-enr-sub { font-family: var(--mono); font-size: 11px; color: var(--muted); margin: 0 0 14px; }
.pano-enr-sub .lbl { color: var(--faint); }

/* Pathway table header + rows */
.pano-enr-hrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--faint); }
.pano-enr-num { display: flex; flex-direction: column; line-height: 1.35; gap: 1px; }
.pano-enr-num .s { font-family: var(--mono); font-size: 13px; color: var(--ink); }
.pano-enr-num .q { font-family: var(--mono); font-size: 10px; color: var(--faint); }
.pano-enr-num .cov { font-family: var(--mono); font-size: 10px; color: var(--muted); cursor: help; }
.pano-enr-bio { font-size: 12px; color: var(--muted); line-height: 1.5; }
.pano-enr-bio .dim { color: var(--faint); }
.pano-bio-cite { font-family: var(--mono); font-size: 10px; color: var(--accent); white-space: nowrap; text-decoration: none; }
.pano-bio-cite:hover { text-decoration: underline; }
.pano-enr-sug { font-family: var(--mono); font-size: 9px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--absent); background: var(--absent-bg); border-radius: 5px; padding: 3px 9px;
  display: inline-block; margin: 10px 0 2px; }
.pano-enr-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
  border: 1px dashed var(--hair); border-radius: 10px; text-align: center; }

/* Pinnable pathway button: chromeless, a ●/○ pin dot as ::before (mirrors the
   marker evidence-table select dot). Keyed container st-key-pwrow_. */
div[class*="st-key-pwrow_"] [data-testid="stColumn"] { padding: 0 !important; }
div[class*="st-key-pwrow_"] div[data-testid="stButton"] > button {
  background: transparent !important; border: 0 !important; box-shadow: none !important;
  min-height: 0 !important; padding: 5px 4px !important; border-radius: 6px !important;
  display: flex !important; align-items: center !important; justify-content: flex-start !important;
  font-family: var(--sans) !important; font-weight: 600 !important; font-size: 13px !important;
  color: var(--ink) !important; text-align: left !important;
}
div[class*="st-key-pwrow_"] div[data-testid="stButton"] > button > div { flex: 0 1 auto !important; width: auto !important; }
div[class*="st-key-pwrow_"] div[data-testid="stButton"] > button::before {
  content: ''; width: 10px; height: 10px; border-radius: 50%; flex: none; margin-right: 9px;
  box-sizing: border-box; border: 1.5px solid var(--faint); background: transparent;
  transition: background 120ms ease, border-color 120ms ease;
}
div[class*="st-key-pwrow_"] button[kind="primary"]::before,
div[class*="st-key-pwrow_"] button[data-testid="stBaseButton-primary"]::before {
  background: var(--accent) !important; border-color: var(--accent) !important;
}
div[class*="st-key-pwrow_"] button[kind="primary"],
div[class*="st-key-pwrow_"] button[data-testid="stBaseButton-primary"] { color: var(--accent) !important; background: transparent !important; }
div[class*="st-key-pwrow_"] div[data-testid="stButton"] > button:hover { background: var(--hair2) !important; }
div[class*="st-key-pwrow_sug_"] div[data-testid="stButton"] > button::after {
  content: 'suggestive'; margin-left: 8px; font-family: var(--mono); font-size: 8px; font-weight: 500;
  color: var(--absent); background: var(--absent-bg); padding: 1px 6px; border-radius: 4px; }

/* Cross-cluster themes (side column) */
.pano-th-h { font-family: var(--sans); font-size: 15px; font-weight: 600; color: var(--ink); margin: 2px 0 10px; }
.pano-th-card { background: var(--paper); border: 1px solid var(--hair); border-radius: 12px; padding: 2px 15px; margin-bottom: 14px; }
.pano-th-item { display: grid; grid-template-columns: 16px 1fr; gap: 9px; padding: 11px 0; border-bottom: 1px solid var(--hair2); }
.pano-th-item:last-child { border-bottom: 0; }
.pano-th-check { color: var(--accent); font-weight: 700; font-size: 12px; }
.pano-th-body { font-size: 12px; color: var(--muted); line-height: 1.55; }
.pano-recur { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 0; }
.pano-recur-chip { font-family: var(--mono); font-size: 10.5px; color: var(--ink); background: var(--hair2);
  border-radius: 6px; padding: 3px 8px; }
.pano-recur-chip .c { color: var(--accent); font-weight: 600; }
.pano-enr-pinhint { font-family: var(--mono); font-size: 11px; color: var(--faint);
  border: 1px dashed var(--hair); border-radius: 10px; text-align: center; padding: 22px 16px; margin-top: 14px; }
"""

_ROW_COLS = [0.36, 0.22, 0.42]


# --------------------------------------------------------------------------- #
# Center: header + pinnable pathway table + spatial stage
# --------------------------------------------------------------------------- #
def _cite_html(note: dict | None) -> str:
    if not note or not note.get("summary"):
        return '<span class="dim">—</span>'
    summary = html.escape(str(note["summary"]))
    pmid = note.get("pmid")
    if pmid:
        cite = (
            f'<a class="pano-bio-cite" href="https://pubmed.ncbi.nlm.nih.gov/{html.escape(str(pmid))}/" '
            f'target="_blank">&#128196; PMID:{html.escape(str(pmid))}</a>'
        )
    else:
        cite = '<span class="dim">literature thin</span>'
    return f"{summary} {cite}"


def _num_html(p: PathwayEvidence) -> str:
    q = "" if p.q_value is None else f'<span class="q">q {p.q_value:.1e}</span>'
    cov = (
        f'<span class="cov" title="{html.escape(p.panel_scope_caveat)}">'
        f"cov {p.panel_hits}/{p.set_size_full}</span>"
    )
    return f'<div class="pano-enr-num"><span class="s">{p.score:.2f}</span>{q}{cov}</div>'


def _render_pathway_rows(st, cluster: str, pathways, suggestive: bool) -> None:
    for p in pathways:
        selected = state.is_pathway_selected(cluster, p.gene_set)
        key = f"pwrow_{'sug_' if suggestive else ''}{'on' if selected else 'off'}_{p.gene_set}"
        with st.container(key=key):
            c_name, c_num, c_bio = st.columns(_ROW_COLS, vertical_alignment="top")
            with c_name:
                if st.button(
                    _short(p.gene_set),
                    key=f"pwsel_{p.gene_set}",
                    type="primary" if selected else "secondary",
                    use_container_width=True,
                ):
                    state.toggle_pathway(cluster, p.gene_set)
                    st.rerun()
            with c_num:
                st.markdown(_num_html(p), unsafe_allow_html=True)
            with c_bio:
                note = da.pathway_note(cluster, p.gene_set)
                st.markdown(f'<div class="pano-enr-bio">{_cite_html(note)}</div>', unsafe_allow_html=True)


def _render_center(st, ce: ClusterEnrichment) -> None:
    from ui.verdict_header import _idline  # reuse the marker id line, verbatim

    css, _ = fmt.confidence_chip(ce.confidence)
    verify = f' <span class="pano-verify">{html.escape(fmt.verify_badge(True))}</span>' if ce.verify else ""
    tops = [_short(p.gene_set) for p in ce.enriched[:3]]
    rationale = ("Enriched for " + ", ".join(tops) + ".") if tops else "No program clears the enrichment gate."
    # Same simple header as the marker verdict: id line, big call + a pill, a one-liner.
    st.markdown(
        f'<div class="pano-idline">{html.escape(_idline(ce.cluster))}</div>'
        f'<div class="pano-verdict"><h1>{html.escape(ce.cell_type)}</h1>'
        f'<span class="cf {css}">{html.escape(ce.confidence)} enrichment</span>{verify}</div>'
        f'<div class="pano-rat">{html.escape(rationale)}</div>',
        unsafe_allow_html=True,
    )

    if not ce.enriched and not ce.suggestive:
        st.markdown('<div class="pano-enr-empty">No pathway clears the enrichment gate for this cluster.</div>',
                    unsafe_allow_html=True)
        return

    with st.container(key="pwhead"):
        h1, h2, h3 = st.columns(_ROW_COLS)
        h1.markdown('<span class="pano-enr-hrow">Program · select to map</span>', unsafe_allow_html=True)
        h2.markdown('<span class="pano-enr-hrow">score · q · cov</span>', unsafe_allow_html=True)
        h3.markdown('<span class="pano-enr-hrow">biology · relevance (cited)</span>', unsafe_allow_html=True)

    _render_pathway_rows(st, ce.cluster, ce.enriched, suggestive=False)
    if ce.suggestive:
        st.markdown('<div class="pano-enr-sug">suggestive · re-check (below the strict bar)</div>',
                    unsafe_allow_html=True)
        _render_pathway_rows(st, ce.cluster, ce.suggestive, suggestive=True)

    # The selected programs' leading edge on tissue (spatial stage — small-multiples).
    selected = state.active_pathways(ce.cluster)
    chosen = [p for p in (*ce.enriched, *ce.suggestive) if p.gene_set in selected]
    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    enrichment_spatial.render_pathways_spatial(ce.cluster, chosen)


# --------------------------------------------------------------------------- #
# Side column: cross-cluster themes
# --------------------------------------------------------------------------- #
def _render_themes(st, themes) -> None:
    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="pano-th-h">Cross-cluster themes</div>', unsafe_allow_html=True)
    notes = "".join(
        f'<div class="pano-th-item"><div class="pano-th-check">&#10003;</div>'
        f'<div class="pano-th-body">{html.escape(n)}</div></div>'
        for n in themes.coherence_notes
    )
    st.markdown(f'<div class="pano-th-card">{notes}</div>', unsafe_allow_html=True)
    recur = "".join(
        f'<span class="pano-recur-chip">{html.escape(_short(t.gene_set))} '
        f'<span class="c">&times;{t.n_clusters}</span></span>'
        for t in themes.recurring[:12]
    )
    if recur:
        st.markdown(f'<div class="pano-th-card"><div class="pano-recur">{recur}</div></div>',
                    unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Public: the page
# --------------------------------------------------------------------------- #
def render_pathways_page() -> None:
    """Render the Pathways page in the 3-pane marker-genes shell."""
    import streamlit as st

    st.markdown(f"<style>{_ENR_CSS}</style>", unsafe_allow_html=True)

    if not da.enrichment_available():
        st.markdown(
            '<div class="pano-enr-empty" style="margin-top:24px">No enrichment result for this '
            "dataset yet. Add a per-cluster gene-set enrichment result to build the Pathways slice.</div>",
            unsafe_allow_html=True,
        )
        return

    rail_col, center_col, chat_col = st.columns([222, 760, 372], gap="small")
    with rail_col:
        cluster_rail.render_rail()
    cluster = state.get_selected_cluster()
    with center_col:
        _render_center(st, da.enrichment_for(cluster))
        _render_themes(st, da.pathway_themes())
    with chat_col:
        enrichment_conversation.render_pathway_conversation(cluster)


__all__ = ["render_pathways_page"]
