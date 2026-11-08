"""Marker genes — the ASSIGNMENT surface, rendered from its declaration.

``workflows/marker.py``'s ``UISpec`` declares the header, columns, absence blocks
and sort; ``ui/surface.py`` draws them. This module only adapts the v2 record's
evidence into ``SurfaceRow``\\s, each carrying its measurement state, so a
never-measured marker is a row, never a blank. The band, ``verify`` and the call
are read off ``display_verdict_for``. Selecting a gene is a viewing control. Nothing
here computes a statistic or generates biology text.
"""

from __future__ import annotations

import html
from functools import lru_cache
from typing import Any, Container, NamedTuple, Sequence

from panoscope.agent import config as cfg
from panoscope.data.study import NOT_YET_CALLED
from panoscope.methods.jazzpanda import BAND_EDGES
from panoscope.types.measured import MeasuredState
from panoscope.types.subject import Entity, EntityKind, subject
from panoscope.workflows import marker as marker_workflow

from panoscope.ui import data_access as da
from panoscope.ui import format as fmt
from panoscope.ui import state
from panoscope.ui import surface

# The one home for the English of a registered evidence flag, keyed on flag ids.
FLAG_COPY: dict[str, str] = {
    "specificity:localises_elsewhere": "localizes better with another cluster",
    "specificity:not_unique": "spatial pattern not unique",
    "specificity:low": "low spatial specificity",
}

#: How to read each of the four states (hatch hover). Never collapse to two.
_READS: dict[MeasuredState, str] = {
    MeasuredState.OBSERVED: "reported with a statistic",
    MeasuredState.TESTED_ABSENT:
        "on panel, tested and not reported for this unit — absence IS evidence here",
    MeasuredState.UNDERPOWERED:
        "in frame but below its floor, so its absence argues nothing",
    MeasuredState.NEVER_MEASURED:
        "absent from the capability frame, so its absence argues nothing",
}

#: The bars a band had to clear, read off the declared ref, not restated.
_GATES: tuple[str, ...] = tuple(
    f"{band}: glm_coef >= {edge:g}" for band, edge in BAND_EDGES
)


# --------------------------------------------------------------------------- #
# The record's own evidence, via the same composition root ``panoscope build`` uses.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _builds(study_dir: str) -> dict[str, Any]:
    """Every unit's build for ONE study tree, keyed on its path (ids can collide)."""
    from panoscope.pipeline import records

    return records.build_all(cfg.current_study())


def unit_build(cluster: str) -> Any | None:
    """This unit's ``UnitBuild``, or ``None`` (rendered as a message) if the study
    cannot produce one."""
    try:
        return _builds(str(cfg.current_study().dir)).get(cluster)
    # SystemExit too: the build refuses a study with no marker output that way (a CLI
    # idiom), and uncaught in a page it kills the script thread (enrichment/DE-only).
    except (Exception, SystemExit):  # noqa: BLE001 - a missing input is a message, not a crash
        return None


# --------------------------------------------------------------------------- #
# The adapter: items -> SurfaceRows. Statistic and state are read, never decided.
# --------------------------------------------------------------------------- #
def _reason(item: Any) -> str:
    """Why this row is in this state, naming the frame (the hatch hover)."""
    m = item.measured
    extra = [b for b in m.because if m.frame not in b]
    return (f"{m.state.value} in frame {m.frame}: {_READS[m.state]}"
            + (f" ({'; '.join(extra)})" if extra else ""))


def _pair(item: Any) -> tuple[float, ...] | None:
    """The SPECIFICITY_PAIR cell: pearson here / the best correlation elsewhere.
    ``None`` on any row with no statistic."""
    if item.statistic is None:
        return None
    corr = dict(item.corroborating)
    vals = tuple(corr[k] for k in ("pearson", "max_gc_corr") if k in corr)
    return vals or None


def _chips(item: Any, canonical: frozenset[str]) -> tuple[str, ...]:
    """The NOTE_CHIP cell: canonical membership, then every registered flag as prose.
    ``canonical`` is reference data: it selects, never supplies a value."""
    gene = item.subject.display()
    canon = ("canonical",) if gene.upper() in canonical else ()
    return canon + tuple(FLAG_COPY.get(f, f) for f in item.flags)


def _cite_value(cluster: str, gene: str, anchored: dict) -> tuple[str, str] | str:
    """The CITATION cell: ``(prose, url)`` from the pipeline's live-cited note, plus
    any note anchored to this gene with its literature tension. No note, no sentence.
    """
    note = da.gene_note(cluster, gene) or {}
    summary = str(note.get("summary") or "").strip()
    pmid = note.get("pmid")
    bits: list[str] = []
    if summary:
        bits.append(summary + ("" if pmid or "card" in note else "  (literature thin)"))
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
    """One ``EvidenceItem`` as a resolved row; numbers are ``None`` wherever the state
    forbids one (``cell_html`` refuses them)."""
    gene = item.subject.display()
    return surface.SurfaceRow(
        key=item.subject.key,          # the click-back anchor reconciliation joins on
        state=item.measured.state,
        frame=item.measured.frame,     # which frame refused it; the block selects on this
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
    """The rejected alternative and what separates it, from ``agent.discriminate``;
    empty when no rival is evident in the measured markers."""
    try:
        from panoscope.agent import discriminate

        d = discriminate.discriminate(cluster)
        if d.alt_B is None:
            return ""
        # Withheld when an override replaced the call the discrimination was for.
        shown = da.display_cell_type(cluster)
        if shown and d.call_A and shown != d.call_A:
            return ""
        return f"{d.alt_B.replace('_', ' ')} —{discriminate.settle_line(d)}"
    except Exception:  # noqa: BLE001 - no rival line is better than a broken page
        return ""


def _header_values(cluster: str) -> dict:
    """The fields the declared ``LabelBandHeader`` names, off ``display_verdict_for``.
    ``verify`` is the rubric's, read and never recomputed."""
    v = da.display_verdict_for(cluster)
    return {
        "unit_scope": _unit_scope(cluster),
        "label": v.cell_type,
        # No label, nothing to assess: an uncalled unit shows no band.
        "confidence": "" if v.cell_type == NOT_YET_CALLED else v.confidence,
        "verify": v.verify,
        "band_basis": v.band_basis,
        "demotions": v.demotions,
        "key_markers": v.key_markers,
        "runner_up": _runner_up(cluster),
        "gates": _GATES,
    }


def spec_for(built: Any) -> Any:
    """The declared surface; with no canonical list (no label, or none curated) the
    off-panel block must not claim every canonical marker is on panel."""
    import dataclasses

    spec = marker_workflow.WORKFLOW.uispec()
    if built.canonical:
        return spec
    label = built.label.cell_type
    why = ("no label yet" if label in ("", NOT_YET_CALLED)
           else f"{label} is not in reference data for this tissue")
    return dataclasses.replace(spec, blocks=tuple(
        dataclasses.replace(b, empty_text=f"no canonical marker list for this unit "
                                          f"({why}), so none to look for off panel")
        if b.key == "never_measured" else b for b in spec.blocks))


def rows_for(cluster: str, built: Any) -> list[surface.SurfaceRow]:
    """Every item in this unit's record, adapted; the declared blocks decide placement,
    so nothing is filtered here."""
    anchored = da.anchored_notes(cluster).get("gene", {})
    return [_row(cluster, i, built.canonical, anchored)
            for i in built.evidence.items]


# --------------------------------------------------------------------------- #
# The replicate read (ARCHITECTURE §9.10), under the evidence it is about.
#
# A divergence is a flag about the evidence, never a value. "Could not be compared"
# and "disagree" keep separate glyphs and sentences. Every line is read off the
# record's decisions and ``undecidable`` sentences, the same fields the CLI prints.
# --------------------------------------------------------------------------- #
class ConcLine(NamedTuple):
    """One line of the replicate read, resolved.

    ``claim`` is the click-back anchor and must resolve to something on the page:
    the subject's own row, else the unit. :func:`concordance_html` raises otherwise.
    """

    cls: str
    glyph: str
    text: str
    claim: str
    why: str = ""


#: ``concordance:*`` rule id -> (css class, glyph, English after the subject name).
#: ``concordance:consistent`` is absent: it is summarised, never listed per subject.
#: No glyph here may come from ``surface._GLYPH`` (the measurement lattice): a
#: not-comparable line carries no measurement status, and the record does not say
#: which of its three causes applies, so the visible sentence names all three.
_CONC_COPY: dict[str, tuple[str, str, str]] = {
    "concordance:divergent": (
        "diverg", "&#9888;", "read differently across the samples",
    ),
    "concordance:not_comparable": (
        "blocked", "&#63;", "could not be compared across the samples — a sample "
        "never measured it, was underpowered for it, or reported no row at all, "
        "and the record does not say which",
    ),
}


def _unit_key(cluster: str) -> str:
    """This unit's own ``Subject.key`` -- the anchor for a line about the whole unit."""
    return subject("unit", u=Entity(EntityKind.UNIT, cluster)).key


def _subject_name(key: str) -> str:
    """A display name for a subject no row on this page carries: the entity ids
    inside its ``Subject.key``, never the join string itself."""
    ids = [part.split("=", 1)[-1].split(":", 1)[-1]
           for part in key.split("|")[1:]]
    return ", ".join(i for i in ids if i) or key


def concordance_lines(cluster: str, built: Any) -> list[ConcLine]:
    """This unit's replicate read, as lines. Pure: no Streamlit, nothing derived.

    Divergence first, then blocked, then the record's ``undecidable`` sentences
    verbatim. Agreement renders nothing (interface-design §4.5): a count beside the
    band would read as corroboration, and replicates are not independent witnesses.
    """
    rec = built.record
    unit_key = _unit_key(cluster)
    # Display names off the record's own evidence; absent keys have no row above.
    display = {i.subject.key: i.subject.display() for i in built.evidence.items}

    subjects: list[ConcLine] = []
    for d in rec.decisions:
        if d.rule in _CONC_COPY:
            cls, glyph, sentence = _CONC_COPY[d.rule]
            # No row on the page: anchor on the unit and say so.
            on_page = d.outcome in display
            subjects.append(ConcLine(
                cls, glyph,
                f"{display.get(d.outcome) or _subject_name(d.outcome)} {sentence}"
                + ("" if on_page else
                   " — no row for it in the evidence above, so this line resolves "
                   "to the unit"),
                d.outcome if on_page else unit_key, "; ".join(d.because),
            ))
    subjects.sort(key=lambda ln: ln.cls != "diverg")   # stable: divergence first

    subjects.extend(ConcLine("undecidable", "&#8226;", u, unit_key)
                    for u in rec.undecidable)
    return subjects


def concordance_html(lines: Sequence[ConcLine],
                     resolvable: Container[str]) -> str:
    """The lines as HTML, or ``""`` when the record carries no replicate read.

    ``resolvable`` is every anchor this page carries; a line anchored elsewhere raises.
    """
    if not lines:
        return ""
    out = ['<div class="pano-sf-block">replicate read &middot; '
           'ARCHITECTURE &sect;9.10</div>']
    for ln in lines:
        claim = surface.anchored(ln.claim, resolvable,
                                 f"replicate-read line ({ln.text!r})")
        title = html.escape(ln.why or ln.text, quote=True)
        out.append(f'<div class="pano-ev-conc c-{ln.cls}" '
                   f'data-claim="{html.escape(claim, quote=True)}" '
                   f'title="{title}">{ln.glyph} '
                   f'{html.escape(ln.text, quote=True)}</div>')
    return "".join(out)


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
_EVIDENCE_CSS = """
<style>
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint); padding: 16px 4px;
}
.pano-ev-conc { font-size: 12px; line-height: 1.5; padding: 4px 0; cursor: help;
  color: var(--faint); }
.pano-ev-conc.c-diverg { color: var(--absent); }
.pano-ev-conc.c-undecidable { color: var(--muted); }
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

    spec = spec_for(built)
    rows = rows_for(cluster, built)
    by_key = {r.key: r.values["gene"] for r in rows}

    # Evidence first (interface §5.3): the proposal's header is one tap away, in
    # ``ui/review.py``; only the unit's provenance line leads.
    st.markdown(f'<div class="pano-idline">{html.escape(_unit_scope(cluster))}</div>',
                unsafe_allow_html=True)
    surface.render_surface(
        spec,
        header=None,
        rows=rows,
        on_select=lambda r: state.toggle_marker(cluster, by_key[r.key]),
        selected=lambda r: state.is_marker_selected(cluster, by_key[r.key]),
    )

    # Anchors on this page: one per row, plus the unit.
    conc = concordance_html(concordance_lines(cluster, built),
                            {r.key for r in rows} | {_unit_key(cluster)})
    if conc:
        st.markdown(conc, unsafe_allow_html=True)
