"""Pathways page — gene-set enrichment interpretation (second workflow).

The center stage is ``ui/surface.py`` rendering ``workflows/enrichment.py``'s
``UISpec``: a ``CensusHeader`` with no band slot (CHARACTERISATION has no label
space), and every tested set as a row under the state that produced it, so an
untestable program is visible rather than missing. This module only adapts
``PathwayEvidence`` into ``SurfaceRow``\\s. Selecting pathways is a viewing control;
nothing here computes a value or generates biology text.
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
    """The Pathways chat as a fragment, so a chat turn re-renders only this pane."""
    state.apply_session_study()  # a fragment rerun skips app.py's top
    enrichment_conversation.render_pathway_conversation(cluster)


# --------------------------------------------------------------------------- #
# Styling (reuses theme tokens + the global .cf-* confidence pills)
# --------------------------------------------------------------------------- #
_ENR_CSS = """
.pano-enr-empty { font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
  border: 1px dashed var(--hair); border-radius: 10px; text-align: center; }
"""

# --------------------------------------------------------------------------- #
# The adapter: PathwayEvidence -> SurfaceRows. Values and state are read, never computed.
# --------------------------------------------------------------------------- #
def _reason(p: PathwayEvidence, mstate: MeasuredState) -> str:
    """Why this row is in this state (the hatch hover). Counts are of *usably
    measured* genes (every capability frame applied), never "on panel"."""
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
    """The CITATION cell: ``(prose, url)`` from the pipeline's live-cited note; no
    note, no sentence."""
    if not note or not note.get("summary"):
        return ""
    pmid = note.get("pmid")
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    summary = str(note["summary"]) + ("" if pmid else "  (literature thin)")
    return (summary, url)


def _row(cluster: str, p: PathwayEvidence) -> surface.SurfaceRow:
    """One ``PathwayEvidence`` as a resolved row; numbers are withheld wherever the
    state forbids one (``cell_html`` refuses them)."""
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
            # TEXT by declaration: K of N explains a missing statistic, so never hatch it.
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
    """The fields the declared ``CensusHeader`` names: counts, denominators, gates.
    No band. The cell type is imported context from the marker workflow."""
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
    """Render the declared surface, then the figures the pinned rows select."""
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
