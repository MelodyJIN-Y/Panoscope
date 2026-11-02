"""Marker genes — the ASSIGNMENT surface, rendered from its declaration.

**The center stage is no longer hand-built.** ``workflows/marker.py``'s ``UISpec``
owns the header variant, every column, every absence block, the sort key and the
coverage ribbon; ``ui/surface.py`` draws them. What is left here is the one thing a
declaration cannot do: adapt this method's rows into resolved ``SurfaceRow``\\s with
a measurement state on each. That is ARCHITECTURE §13's shape — *the review surface
is part of the interpretation contract, declared by the workflow, not a UI concern* —
and it is the same conversion ``ui/enrichment_table.py`` already did for Pathways.

Two consequences are visible on screen:

* **Never-measured markers are rows now.** The page rendered ``ClusterVerdict.evidence``,
  which is the v1 *compatibility view* and carries only the rows that were measured
  (c2: 28 of 33). A canonical marker the panel never carried lived in a sibling field
  (``offpanel_notes``) that this table never read, so it was indistinguishable from a
  gene that was measured and found nothing — the exact failure ``CLAUDE.md`` names:
  *"A never-measured item is not a blank row and not a zero."* The rows now come from
  the v2 record's own evidence (``pipeline/records.build_all``, the composition root
  ``panoscope build`` and ``panoscope explain`` share), which carries a
  ``MeasuredState`` and the frame that settled it on every item. Each of the four
  states has a declared block, and a block renders its header at zero.
* **The band moved into the declared header.** ``ui/verdict_header.py`` hand-built the
  id line, the call, the confidence chip and the driver line. ``LabelBandHeader`` is
  the declaration of exactly that, and ASSIGNMENT is the one task kind allowed one —
  ``surface.label_band_html`` refuses any other, which is the asymmetry with Pathways
  made structural. The band, ``verify`` and the call are still read off the v1
  ``ClusterVerdict`` (``display_verdict_for``, so a confirmed override still shows),
  so no number on screen moved; only who draws it did.

Selecting a gene is a viewing control (``ui.state.toggle_marker``, a per-cluster
multi-select): it drives the spatial small-multiples and nothing else. The renderer
offers a pin only on an OBSERVED row — a gene the panel never carried has no map to
draw. This module computes nothing and never generates biology text: the per-gene
note is the pipeline's, live-cited (``ui.data_access.gene_note``).

Streamlit is imported lazily so ``import ui.evidence_table`` needs no server.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from panoscope.agent import config as cfg
from panoscope.methods.jazzpanda import BAND_EDGES
from panoscope.types.measured import MeasuredState
from panoscope.workflows import marker as marker_workflow

from panoscope.ui import data_access as da
from panoscope.ui import format as fmt
from panoscope.ui import state
from panoscope.ui import surface

# --------------------------------------------------------------------------- #
# Flag copy. The ONE home for the English of a registered evidence flag
# (``types.evidence.FLAGS``). Copy belongs in the UI layer; ids belong everywhere
# else. The specificity badge used to be decided by matching a literal English
# phrase, so a copy edit in the analysis layer silently stopped it rendering and
# its test stayed green. The chips below are keyed on the ids the record carries.
# --------------------------------------------------------------------------- #
FLAG_COPY: dict[str, str] = {
    "specificity:localises_elsewhere": "localizes better with another cluster",
    "specificity:not_unique": "spatial pattern not unique",
    "specificity:low": "low spatial specificity",
}

#: How to READ each state — the second half of the hatch hover, after the frame.
#: Four entries because there are four states; the two-entry version of this table
#: is the collapse into "unknown" the whole lattice exists to stop.
_READS: dict[MeasuredState, str] = {
    MeasuredState.OBSERVED: "reported with a statistic",
    MeasuredState.TESTED_ABSENT:
        "on panel, tested and not reported for this unit — absence IS evidence here",
    MeasuredState.UNDERPOWERED:
        "in frame but below its floor, so its absence argues nothing",
    MeasuredState.NEVER_MEASURED:
        "absent from the capability frame, so its absence argues nothing",
}

#: The bars a band had to clear, read off the ref the declaration names
#: (``LabelBandHeader.band_edges_ref``) rather than restated here.
_GATES: tuple[str, ...] = tuple(
    f"{band}: glm_coef >= {edge:g}" for band, edge in BAND_EDGES
)


# --------------------------------------------------------------------------- #
# The record's own evidence. Built through the composition root, never re-derived:
# ``panoscope build`` writes the record from these same functions, so the table and
# the file on disk cannot disagree about a state. ``lru_cache`` because Streamlit
# re-executes the script, not the module, on every rerun.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _builds(study_id: str) -> dict[str, Any]:
    from panoscope.pipeline import records

    return records.build_all(cfg.current_study())


def unit_build(cluster: str) -> Any | None:
    """This unit's ``UnitBuild``, or ``None`` if the study cannot produce one.

    Returns rather than raises: a study with no method output on disk must render a
    page that says so, not a traceback — and never a table of rows whose states were
    guessed off some other artifact.
    """
    try:
        return _builds(cfg.current_study().id).get(cluster)
    except Exception:  # noqa: BLE001 - a missing/broken input is a message, not a crash
        return None


# --------------------------------------------------------------------------- #
# The adapter: this method's items -> resolved SurfaceRows.
#
# The ONLY thing left that is specific to jazzPanda's marker output. Nothing here
# computes a statistic or decides a state: both are read off the ``EvidenceItem``
# the record was assembled from.
# --------------------------------------------------------------------------- #
def _reason(item: Any) -> str:
    """Why this row is in this state — the hover on the hatch, naming the frame.

    The frame is named rather than implied: "never measured" without *which frame
    refused it* is the blank row with extra words.
    """
    m = item.measured
    extra = [b for b in m.because if m.frame not in b]
    return (f"{m.state.value} in frame {m.frame}: {_READS[m.state]}"
            + (f" ({'; '.join(extra)})" if extra else ""))


def _pair(item: Any) -> tuple[float, ...] | None:
    """The SPECIFICITY_PAIR cell: pearson here / the best correlation elsewhere.

    ``None`` on any row that carries no number — ``core/measure.py`` withholds the
    corroborators with the statistic, so there is nothing to put here and
    ``cell_html`` hatches the cell.
    """
    if item.statistic is None:
        return None
    corr = dict(item.corroborating)
    vals = tuple(corr[k] for k in ("pearson", "max_gc_corr") if k in corr)
    return vals or None


def _chips(item: Any, canonical: frozenset[str]) -> tuple[str, ...]:
    """The NOTE_CHIP cell: canonical membership, then every registered flag as prose.

    ``canonical`` is reference data doing the one thing reference data may do — it
    selects which rows matter. It supplies no value and corroborates nothing.
    """
    gene = item.subject.display()
    canon = ("canonical",) if gene.upper() in canonical else ()
    return canon + tuple(FLAG_COPY.get(f, f) for f in item.flags)


def _cite_value(cluster: str, gene: str, anchored: dict) -> tuple[str, str] | str:
    """The CITATION cell: ``(prose, url)``. The renderer escapes and links it.

    Never generated here — the prose is the pipeline's precomputed, live-cited note,
    and a gene with no note gets no sentence rather than an invented one. A note the
    biologist anchored to this gene rides along with its literature tension, because
    the biologist's correction belongs beside the marker it corrects.
    """
    note = da.gene_note(cluster, gene) or {}
    summary = str(note.get("summary") or "").strip()
    pmid = note.get("pmid")
    bits: list[str] = []
    if summary:
        bits.append(summary + ("" if pmid else "  (literature thin)"))
    for n in (anchored.get(gene) or [])[:2]:
        t = n.tension
        tension = (f" · {len(t.dissent)} lit. dissent" if t.dissent
                   else f" · {len(t.agree)} lit. agree" if t.agree else "")
        bits.append(f"⚑ note: {n.claim} [note:{n.id[:6]}]{tension}")
    text = "  ".join(bits)
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
    return (text, url) if text else ""


def _row(cluster: str, item: Any, canonical: frozenset[str],
         anchored: dict) -> surface.SurfaceRow:
    """One ``EvidenceItem`` as a resolved row.

    ``statistic`` and ``pearson`` are whatever the item carries, which is ``None`` on
    every state that forbids a number. That is not politeness: ``cell_html`` refuses
    a number on such a row outright, which is the structural form of *statistic is
    None iff the state is TESTED_ABSENT or NEVER_MEASURED*.
    """
    gene = item.subject.display()
    return surface.SurfaceRow(
        key=item.subject.key,          # the click-back anchor reconciliation joins on
        state=item.measured.state,
        reason=_reason(item),
        values={
            "gene": gene,
            "statistic": item.statistic,
            "pearson": _pair(item),
            "flags": _chips(item, canonical),
            "citation": _cite_value(cluster, gene, anchored),
        },
    )


# --------------------------------------------------------------------------- #
# The declared header's fields
# --------------------------------------------------------------------------- #
def _unit_scope(cluster: str) -> str:
    """``CLUSTER 3 · 4,812 cells`` — provenance, read, never computed."""
    label = f"CLUSTER {fmt.short_cluster_id(cluster)}"
    try:
        return f"{label} · {len(da.cluster_cells_df(cluster)):,} cells"
    except Exception:  # noqa: BLE001 - an unknown cell count still has an id line
        return label


def _runner_up(cluster: str) -> str:
    """The rejected alternative and what separates it (``require_runner_up``).

    Read off ``agent.discriminate`` — the source the declaration names — and empty
    when no competing hypothesis is evident in the measured markers, which is a
    different statement from a rival nobody looked for.
    """
    try:
        from panoscope.agent import discriminate

        d = discriminate.discriminate(cluster)
        if d.alt_B is None:
            return ""
        return f"{d.alt_B.replace('_', ' ')} —{discriminate.settle_line(d)}"
    except Exception:  # noqa: BLE001 - no rival line is better than a broken page
        return ""


def _header_values(cluster: str) -> dict:
    """The fields the declared ``LabelBandHeader`` names.

    The band, ``verify`` and the call come off the v1 ``ClusterVerdict``
    (``display_verdict_for``, so a confirmed override shows) — the same values this
    page showed before the conversion, so nothing on screen moved. ``verify`` is the
    deterministic rubric's and is read, never recomputed and never offered.
    """
    v = da.display_verdict_for(cluster)
    return {
        "unit_scope": _unit_scope(cluster),
        "label": v.cell_type,
        "confidence": v.confidence,
        "verify": v.verify,
        "band_basis": v.band_basis,
        "demotions": v.demotions,
        "key_markers": v.key_markers,
        "runner_up": _runner_up(cluster),
        "gates": _GATES,
    }


def rows_for(cluster: str, built: Any) -> list[surface.SurfaceRow]:
    """Every item in this unit's record, adapted. Nothing is filtered out here.

    Filtering would put the decision about which absences are visible in the
    renderer, which is where it least belongs: the declared blocks decide where a
    row goes, and ``never_truncate`` decides that it goes somewhere.
    """
    anchored = da.anchored_notes(cluster).get("gene", {})
    return [_row(cluster, i, built.canonical, anchored)
            for i in built.evidence.items]


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
_EVIDENCE_CSS = """
<style>
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
}
</style>
"""


def render_evidence_table(cluster: str) -> None:
    """Render ``cluster``'s declared ASSIGNMENT surface into the current column."""
    import streamlit as st

    st.markdown(_EVIDENCE_CSS, unsafe_allow_html=True)

    built = unit_build(cluster)
    if built is None:
        st.markdown(
            '<div class="pano-ev-empty">No interpretation record for this unit. '
            "Run <code>panoscope build</code> for this study to produce one.</div>",
            unsafe_allow_html=True,
        )
        return

    spec = marker_workflow.WORKFLOW.uispec()
    rows = rows_for(cluster, built)
    by_key = {r.key: r.values["gene"] for r in rows}

    surface.render_surface(
        spec,
        header=_header_values(cluster),
        rows=rows,
        on_select=lambda r: state.toggle_marker(cluster, by_key[r.key]),
        selected=lambda r: state.is_marker_selected(cluster, by_key[r.key]),
    )


__all__ = ["render_evidence_table", "rows_for", "unit_build", "FLAG_COPY"]
