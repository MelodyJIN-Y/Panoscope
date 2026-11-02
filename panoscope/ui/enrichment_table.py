"""Pathways page — gene-set enrichment interpretation (second workflow).

**The center stage is no longer hand-built.** It is rendered by ``ui/surface.py``
from ``workflows/enrichment.py``'s ``UISpec`` — the header variant, every column,
every absence block, the sort key and the coverage ribbon are read off the
declaration, and this module's job is reduced to the one thing a declaration cannot
do: adapt this method's rows (``PathwayEvidence``) into resolved ``SurfaceRow``\\s
with a measurement state on each. That is the shape ARCHITECTURE §13 asks for —
*the review surface is part of the interpretation contract, declared by the
workflow, not a UI concern.*

Two consequences are visible on screen:

* **The confidence band is gone.** This page used to import the assignment header's
  id line and render ``ce.confidence`` as a ``cf-*`` pill — the counter-example
  ``docs/interface-design.md`` §3.1 names: a CHARACTERISATION surface inheriting a
  unit-level band it had no right to, on a task with no label space. The declared
  header is a ``CensusHeader``, which has no band slot, and
  ``surface.label_band_html`` refuses a non-ASSIGNMENT spec, so the old shape is
  unreachable from here rather than merely discouraged.
* **Untestable programs are rows now.** The page showed ``enriched`` and
  ``suggestive`` only; every set the panel could not test simply did not render,
  which is indistinguishable from a set that was tested and found nothing. All of
  ``ce.all_tested`` is adapted, and the declared blocks put each set under the
  state that produced it, with its number hatched where there is none.

Laid out like the Marker-genes page: a 3-pane shell — the cluster **rail** (shared
selection), the **center stage** (the declared surface + the pinned pathway's
leading-edge spatial views), and a **side column** with the cross-cluster themes.

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

import streamlit as st  # module-level for the @st.fragment chat wrapper

from panoscope.agent import config as cfg
from panoscope.agent.types import ClusterEnrichment, PathwayEvidence
from panoscope.types.measured import MeasuredState
from panoscope.types.subject import gene_set, subject
from panoscope.workflows import enrichment as enrichment_workflow

from panoscope.ui import cluster_rail
from panoscope.ui import data_access as da
from panoscope.ui import enrichment_conversation
from panoscope.ui import enrichment_spatial
from panoscope.ui import format as fmt
from panoscope.ui import state
from panoscope.ui import surface

_METHOD_LABEL = {"jazzpanda_enrichment": "jazzPanda competitive test"}


@st.fragment
def _pathway_chat_fragment(cluster: str) -> None:
    """The Pathways chat as a fragment: a chat turn re-renders only this pane, not the
    whole Pathways page (its reruns are scoped via ``convo._rerun(scope='fragment')``)."""
    enrichment_conversation.render_pathway_conversation(cluster)


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Styling (reuses theme tokens + the global .cf-* confidence pills)
# --------------------------------------------------------------------------- #
_ENR_CSS = """
.pano-enr-eyebrow { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .1em; color: var(--faint); margin: 0 0 10px; }
.pano-enr-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
  border: 1px dashed var(--hair); border-radius: 10px; text-align: center; }

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

# --------------------------------------------------------------------------- #
# The adapter: this method's rows -> resolved SurfaceRows.
#
# The ONLY thing left that is specific to jazzPanda's enrichment output. Everything
# downstream of it is read off the declaration. Nothing here computes a statistic:
# the score, the q and the K of N are read off `PathwayEvidence`, and the state is
# read off the same gates `agent/enrichment.py` already applied to produce the tier.
# --------------------------------------------------------------------------- #
def _state_of(p: PathwayEvidence) -> MeasuredState:
    """Which of the four states this set is in, from the gates that made its tier.

    ``enriched`` and ``suggestive`` both carry a number, so both are OBSERVED -- the
    enriched/suggestive split is a question about how cautious this lab is, not about
    whether the set was measured, and the surface keeps those separate concerns
    separate (the split shows in the header census, the state shows in the block).

    The two ways a set can fail are NOT the same fact and do not share a row:
    below the capability floor is UNDERPOWERED and argues nothing; scored, reported
    and above the q bar is TESTED_ABSENT and argues against the program running here.
    """
    if p.tier in ("enriched", "suggestive"):
        return MeasuredState.OBSERVED
    if p.panel_hits < cfg.MIN_PANEL_HITS or p.n_leading_edge < cfg.MIN_LEADING_EDGE:
        return MeasuredState.UNDERPOWERED
    return MeasuredState.TESTED_ABSENT


def _reason(p: PathwayEvidence, mstate: MeasuredState) -> str:
    """Why this row is in this state -- the hover on the hatch, named not implied.

    Both counts say *usably measured*, not *on panel*. Neither number is a panel
    fact any more: ``panel_hits`` is the set genes every gene capability frame
    admits, and ``n_leading_edge`` is the selected genes those same frames admit,
    so a sentence that called either one "on panel" would describe the panel design
    where it is reporting what the apparatus could actually see -- and on exactly
    the rows (a set below the leading-edge floor only because the detectability
    frame rejected its drivers) where the difference is the whole explanation.
    """
    if mstate is MeasuredState.UNDERPOWERED:
        return (f"{p.panel_hits} of {p.set_size_full} set genes usably measured "
                f"(floor {cfg.MIN_PANEL_HITS}), {p.n_leading_edge} driving gene(s) "
                f"the frames admit (floor {cfg.MIN_LEADING_EDGE}): below the "
                "capability floor, so its absence here argues nothing.")
    if mstate is MeasuredState.TESTED_ABSENT:
        return (f"scored on {p.panel_hits} of {p.set_size_full} usably measured set "
                f"genes and not reported for this unit (q at or above "
                f"{cfg.SUGGESTIVE_Q_MAX}); absence IS evidence here.")
    return p.panel_scope_caveat


def _key_of(p: PathwayEvidence) -> str:
    """The click-back anchor: the same ``Subject.key`` reconciliation joins on."""
    return subject("gene_set", set=gene_set(p.gene_set)).key


def _cite_value(note: dict | None):
    """The CITATION cell: ``(prose, url)``. The renderer escapes and links it.

    Never generated here -- the prose is the pipeline's precomputed, live-cited note
    and a set with no note gets no sentence rather than an invented one.
    """
    if not note or not note.get("summary"):
        return ""
    pmid = note.get("pmid")
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    summary = str(note["summary"]) + ("" if pmid else "  (literature thin)")
    return (summary, url)


def _row(cluster: str, p: PathwayEvidence) -> surface.SurfaceRow:
    """One ``PathwayEvidence`` as a resolved row.

    ``statistic`` and ``p_adj_bh`` are withheld on every state that does not carry a
    number, so those cells hatch. That is not politeness: ``cell_html`` refuses a
    number on such a row outright, which is the structural form of *statistic is
    None iff the state is TESTED_ABSENT or NEVER_MEASURED*.
    """
    mstate = _state_of(p)
    observed = mstate is MeasuredState.OBSERVED
    return surface.SurfaceRow(
        key=_key_of(p),
        state=mstate,
        reason=_reason(p, mstate),
        values={
            "gene_set": _short(p.gene_set),
            "statistic": p.score if observed else None,
            "p_adj_bh": p.q_value if observed else None,
            # TEXT, not a numeric kind, by declaration: the K of N is the REASON a
            # row has no statistic, so hatching it would hide the panel-coverage
            # rule on exactly the rows it decided.
            "coverage": f"{p.panel_hits}/{p.set_size_full}",
            "leading_edge": p.leading_edge[:4],
            "citation": _cite_value(da.pathway_note(cluster, p.gene_set)),
        },
    )


def _unit_scope(cluster: str) -> str:
    """``CLUSTER 3 · 4,812 cells`` — provenance, read, never computed."""
    label = f"CLUSTER {fmt.short_cluster_id(cluster)}"
    try:
        return f"{label} · {len(da.cluster_cells_df(cluster)):,} cells"
    except Exception:  # noqa: BLE001 - an unknown cell count still has an id line
        return label


def _header_values(ce: ClusterEnrichment,
                   rows: list[surface.SurfaceRow]) -> dict:
    """The fields the declared ``CensusHeader`` names. Counts, denominators, gates.

    No band, and nowhere to put one. The cell type is passed as imported context:
    the header attributes it to the marker workflow and does not re-derive it.
    """
    top = ce.enriched[0] if ce.enriched else None
    values = {
        "unit_scope": _unit_scope(ce.cluster),
        "cell_type": da.display_cell_type(ce.cluster),
        "n_enriched": len(ce.enriched),
        "n_suggestive": len(ce.suggestive),
        "n_not_supported": sum(1 for r in rows
                               if r.state is MeasuredState.TESTED_ABSENT),
        "n_untestable": sum(1 for r in rows
                            if r.state is MeasuredState.UNDERPOWERED),
        "n_scored": len(rows),
        "panel_scope_caveat": (
            top.panel_scope_caveat if top else
            "Panel-scoped: over-representation among the genes this panel measured, "
            "not genome-wide."
        ),
        "gates": (
            f"enriched: q < {cfg.ENRICH_Q_MAX}",
            f"suggestive: q <= {cfg.SUGGESTIVE_Q_MAX}",
            f"testable: >= {cfg.MIN_PANEL_HITS} set genes on panel",
            f"defensible: >= {cfg.MIN_LEADING_EDGE} driving genes",
        ),
    }
    if top is not None:
        values |= {
            "gene_set": _short(top.gene_set),
            "statistic_name": "test_statistic",
            "statistic": f"{top.score:.2f}",
            "panel_hits": top.panel_hits,
            "set_size": top.set_size_full,
            "top_key": _key_of(top),      # the header sentence's click-back anchor
        }
    return values


# --------------------------------------------------------------------------- #
# Center: the declared surface + the pinned programs' spatial stage
# --------------------------------------------------------------------------- #
def _render_center(st, ce: ClusterEnrichment) -> None:
    """Render the declaration, then the figures the pinned rows select.

    This function no longer decides what the page looks like. It adapts rows, hands
    them to the renderer with the workflow's own ``UISpec``, and maps a pin back to
    the gene set id the spatial stage takes.
    """
    spec = enrichment_workflow.WORKFLOW.uispec()
    rows = [_row(ce.cluster, p) for p in ce.all_tested]
    by_key = {r.key: p for r, p in zip(rows, ce.all_tested)}

    surface.render_surface(
        spec,
        header=_header_values(ce, rows),
        rows=rows,
        on_select=lambda r: state.toggle_pathway(ce.cluster, by_key[r.key].gene_set),
        selected=lambda r: state.is_pathway_selected(ce.cluster,
                                                     by_key[r.key].gene_set),
    )

    # The selected programs' leading edge on tissue (spatial stage — small-multiples).
    selected = state.active_pathways(ce.cluster)
    chosen = [p for p in ce.all_tested if p.gene_set in selected]
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

    rail_col, center_col, chat_col = st.columns([216, 664, 474], gap="small")
    with rail_col:
        cluster_rail.render_rail()
    cluster = state.get_selected_cluster()
    with center_col:
        _render_center(st, da.enrichment_for(cluster))
    with chat_col:
        _pathway_chat_fragment(cluster)


__all__ = ["render_pathways_page"]
