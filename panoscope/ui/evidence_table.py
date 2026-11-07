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

import html
from functools import lru_cache
from typing import Any, Container, NamedTuple, Sequence

from panoscope.agent import config as cfg
from panoscope.methods.jazzpanda import BAND_EDGES
from panoscope.types.measured import MeasuredState
from panoscope.types.subject import Entity, EntityKind, subject
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
def _builds(study_dir: str) -> dict[str, Any]:
    """Every unit's build for ONE study tree, keyed on that tree's PATH.

    The path, not the study id: two data roots may hold the same id (the pipeline's
    copy-to-tmp tests and the scratch rehearsals open exactly that pair), so an
    id-keyed entry merges them and serves one tree's measurement states — the state
    a row's hatch and its "absence IS evidence here" reading come from — under the
    other tree's rows. The same key ``ui.data_access`` uses, for the same reason.
    """
    from panoscope.pipeline import records

    return records.build_all(cfg.current_study())


def unit_build(cluster: str) -> Any | None:
    """This unit's ``UnitBuild``, or ``None`` if the study cannot produce one.

    Returns rather than raises: a study with no method output on disk must render a
    page that says so, not a traceback — and never a table of rows whose states were
    guessed off some other artifact.
    """
    try:
        return _builds(str(cfg.current_study().dir)).get(cluster)
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
    None iff the state is not OBSERVED* -- read off ``Measured.carries_number``,
    so UNDERPOWERED withholds its number too.
    """
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
        # Withheld when the header's call is not the one the discrimination was
        # computed for. ``discriminate`` reads the deterministic table, never an
        # override, so after the biologist overrode c3 to Dendritic this chip
        # rendered "To settle Macrophages vs T Cells: ... CD4 (glm_coef 4.02)"
        # under <h1>Dendritic</h1> -- a rival test argued for a call they had
        # rejected, carrying real statistics. ``ui/report._settle_for`` already
        # withholds exactly this in the document; the screen did not.
        shown = da.display_cell_type(cluster)
        if shown and d.call_A and shown != d.call_A:
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
# The replicate read (ARCHITECTURE §9.10)
#
# The concordance wiring produces exactly one thing a biologist acts on — *sampleA
# and sampleB read this gene differently* — and it rendered in ``panoscope explain``
# and nowhere else (``cli.py``'s ``undecidable`` section). Searching the rendered
# markdown of all three registered pages on a two-sample study with a genuine
# divergence found no "read differently", no sample name and no "undecidable"
# anywhere. The kidney study is multi-sample, so on the surface a biologist actually
# uses, the main product of the multi-sample half was a pooled number with no caveat
# on it — the confident wrong answer §9.10 exists to refuse, one layer up.
#
# **It renders here, under the evidence, not in the header.** A divergence is a
# statement about the EVIDENCE — two readings of one subject disagree — not about
# the call. It sits between the table it is about and the spatial stage below, which
# is where its own sentence sends the reader ("compare the diverging subject across
# sections before reading the pooled number"). Nothing in it is a number a band
# could be read off and nothing here moves a value: a divergence is a flag, and
# §9.10 does not claim whether it is batch effect, section quality or biology.
#
# **The outcomes stay apart on screen, and no further apart than the record keeps
# them.** "they could not be compared" and "they genuinely disagree" are different
# facts; rendering them the same way would be the four-states-into-two collapse
# arriving in the renderer, after ``core/concordance.py`` went to the trouble of
# keeping them apart. Each has its own glyph and its own sentence, and the
# decision's own ``because`` — which names the state and the frame that produced it
# — is the hover. But the record carries ONE id for not-comparable, covering a
# frame that refused the subject, an underpowered sample AND a sample that emitted
# no row at all, and this renderer has no honest way to split them: the only thing
# that differs is the English in ``because``, and matching on that prose is the
# exact mistake ``FLAG_COPY`` was rebuilt to stop. So the line states all three
# causes rather than picking one, and splitting it properly is a change to the rule
# ids in ``core/concordance.py``, not to the copy here.
#
# Nothing is computed here. Every line is read off the record's own decisions and
# ``undecidable`` sentences, the same two fields the CLI prints, so the terminal and
# the app say the same thing about the same object.
# --------------------------------------------------------------------------- #
class ConcLine(NamedTuple):
    """One line of the replicate read, resolved.

    ``claim`` is the click-back anchor and must resolve to something ON THE PAGE: a
    subject line resolves to that subject's own row in the table above — the same
    ``Subject.key`` reconciliation joins on — and a line about this unit as a whole,
    or about a subject the table above has no row for, resolves to the unit subject.
    :func:`concordance_html` raises on a line that resolves to nothing rather than
    printing it, because a sentence with nothing to click should not have been
    rendered, and an anchor nobody can follow is nothing to click.
    """

    cls: str
    glyph: str
    text: str
    claim: str
    why: str = ""


#: ``concordance:*`` rule -> (css class, glyph, the English after the subject name).
#: Keyed on the rule ids the record carries, not on matching their prose — the
#: mistake ``FLAG_COPY`` above was rebuilt to stop. ``concordance:consistent`` is
#: deliberately absent: it is summarised, never listed per subject, because a screen
#: of green "agrees" lines is the reassurance §9.10 forbids.
#:
#: **No entry here may use a glyph from ``surface._GLYPH``.** Those four are the
#: measurement lattice, registered, and the rows immediately above this block use
#: them with exactly that meaning. ``&#8856;`` -- NEVER_MEASURED -- sat on
#: ``concordance:not_comparable`` and said *off panel, nothing to see* about a gene
#: that is on the panel and was run: a fabricated measurement status, in the one
#: direction an off-target IO panel makes invisible, because "never measured" is
#: the kidney study's background hypothesis for almost every marker. The rule id
#: carries no such status. It says the comparison could not be made, and the three
#: causes it covers -- a frame refused the subject in one sample, a sample was
#: underpowered for it, or a sample emitted no row at all -- are not separated in
#: the record (``core/concordance.py`` says in its own docstring that it cannot
#: tell the last one from a reader that dropped the row). So the line says all
#: three, in the VISIBLE sentence: the hedge was in a ``title=`` hover, and
#: "measurement status is visible, not implied" is not satisfied by a tooltip.
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
    """A display name for a subject NO row on this page carries.

    ``Decision.outcome`` is a ``Subject.key``, not a ``Subject``: the record stores
    identity, and the only subjects that arrive with a display name are the ones
    the evidence table above also carries. The previous fallback printed the key,
    so ``gene|g=gene:SYNG13`` -- a join string, the same one reconciliation matches
    on -- rendered where a gene name belongs. The entity ids inside it are the
    name; everything around them is the join.
    """
    ids = [part.split("=", 1)[-1].split(":", 1)[-1]
           for part in key.split("|")[1:]]
    return ", ".join(i for i in ids if i) or key


def concordance_lines(cluster: str, built: Any) -> list[ConcLine]:
    """This unit's replicate read, as lines. Pure: no Streamlit, nothing derived.

    Divergence first, then what was blocked, because the productive half is the one
    a biologist acts on. The agreement count is one summarised line carrying the
    asymmetry with it: without it, a check that compared one subject and a check
    that compared forty both render as *no divergence*, which is the
    denominator-thrown-away failure ``Concordance.coverage`` was added for.

    The record's ``undecidable`` sentences come last and verbatim. They are the
    whole content on a single-sample study, where there is no comparison to report
    and the honest line is *why there is none* — a never-measured thing is not a
    blank row here either.
    """
    rec = built.record
    unit_key = _unit_key(cluster)
    # Display names off the record's own evidence. A subject only the per-sample
    # tables carried has no row above to name it -- and, critically, nothing above
    # for its line to resolve to either.
    display = {i.subject.key: i.subject.display() for i in built.evidence.items}

    subjects: list[ConcLine] = []
    consistent = 0
    for d in rec.decisions:
        if d.rule == "concordance:consistent":
            consistent += 1
        elif d.rule in _CONC_COPY:
            cls, glyph, sentence = _CONC_COPY[d.rule]
            # Two different facts, so two different anchors. A subject the table
            # above carries resolves to its own row; a subject only a per-sample
            # table reported has no row, and anchoring the line on it rendered a
            # sentence whose click-back resolved to nothing at all. It resolves to
            # the unit instead, and SAYS SO -- a line that quietly retargets is a
            # click-back that lies about where it goes.
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

    if consistent:
        subjects.append(ConcLine(
            "same", "&middot;",
            f"{consistent} subject(s) read the same way across the samples — "
            "agreement NEVER raises confidence: replicates are not independent "
            "witnesses (§9.10)",
            unit_key,
        ))
    subjects.extend(ConcLine("undecidable", "&#8226;", u, unit_key)
                    for u in rec.undecidable)
    return subjects


def concordance_html(lines: Sequence[ConcLine],
                     resolvable: Container[str]) -> str:
    """The lines as HTML, or ``""`` when the record carries no replicate read.

    ``resolvable`` is every anchor THIS PAGE carries -- the table's row keys plus
    the unit -- and it is required, not optional. The guard here checked the claim
    for the empty string, which no line has ever had; what it let through was a
    line anchored on a subject the page does not carry, whose ``data-claim``
    resolved to none of the anchors on it. ``uispec.anchored`` is now the one home
    for both halves of that rule, so a second renderer adopts the widened half by
    calling it -- ``render/text.py`` still carries its own empty-string-only
    ``_anchored`` and has not, which is a one-line change in a file this stream
    does not own rather than a second copy of the rule to write here.
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

    # Under the evidence it is about, and directly above the spatial stage its own
    # sentence sends the reader to.
    # The anchors this page actually carries: one per rendered row, plus the unit
    # itself, which is what a line about the whole unit resolves to.
    conc = concordance_html(concordance_lines(cluster, built),
                            {r.key for r in rows} | {_unit_key(cluster)})
    if conc:
        st.markdown(conc, unsafe_allow_html=True)
