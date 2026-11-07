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

import streamlit as st  # module-level for the @st.fragment chat wrapper

from panoscope.agent import config as cfg
from panoscope.agent.enrichment_themes import pretty_gene_set
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

@st.fragment
def _pathway_chat_fragment(cluster: str) -> None:
    """The Pathways chat as a fragment: a chat turn re-renders only this pane, not the
    whole Pathways page (its reruns are scoped via ``convo._rerun(scope='fragment')``)."""
    enrichment_conversation.render_pathway_conversation(cluster)


# --------------------------------------------------------------------------- #
# Styling (reuses theme tokens + the global .cf-* confidence pills)
# --------------------------------------------------------------------------- #
_ENR_CSS = """
.pano-enr-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
  border: 1px dashed var(--hair); border-radius: 10px; text-align: center; }
"""

# --------------------------------------------------------------------------- #
# The adapter: this method's rows -> resolved SurfaceRows.
#
# The ONLY thing left that is specific to jazzPanda's enrichment output. Everything
# downstream of it is read off the declaration. Nothing here computes a statistic:
# the score, the q and the K of N are read off `PathwayEvidence`, and the state is
# read off the same gates `agent/enrichment.py` already applied to produce the tier.
# --------------------------------------------------------------------------- #
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
    None iff the state is not OBSERVED* -- read off ``Measured.carries_number``,
    so UNDERPOWERED withholds its number too.
    """
    mstate = p.measured_state  # the four-state split lives on the evidence object
    observed = mstate is MeasuredState.OBSERVED
    return surface.SurfaceRow(
        key=_key_of(p),
        state=mstate,
        reason=_reason(p, mstate),
        values={
            "gene_set": pretty_gene_set(p.gene_set),
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
            "gene_set": pretty_gene_set(top.gene_set),
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
# Public: the page
# --------------------------------------------------------------------------- #
def render_pathways_page() -> None:
    """Render the Pathways page in the 3-pane marker-genes shell."""
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
