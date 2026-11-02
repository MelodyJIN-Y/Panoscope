"""The enrichment workflow: *what is RUNNING here?* -- and the surface that answers it.

The second interpretive question, and the one that makes the no-band rule visible
rather than merely stated. Enrichment is a CHARACTERISATION: there is no label
space -- the programs are not alternatives competing for one slot, a cluster may be
running four of them at once -- so a unit-level band is **undefined**, not merely
unhelpful. There is no "instead" for the remaining confidence to be confidence *in*.

That single fact shapes everything below:

* :meth:`EnrichmentInterpretation.interpret` returns a ``Finding``, which has no
  ``band`` field to fill. An ``Assignment`` routed through it raises.
* :meth:`uispec` declares a ``CensusHeader``. ``UISpec.__post_init__`` refuses a
  ``LabelBandHeader`` on anything that is not an ASSIGNMENT, so the bug this
  replaces -- ``ui/enrichment_table.py`` importing the assignment header and
  inheriting a confidence band it had no right to -- is now unconstructable.
* ``methods/jazzpanda_cgst.py`` declares no band ladder, and its ``rubric`` raises.

The header is per-item counts: *4 enriched, 3 untestable, top EMT z 4.2 on 18/200
panel genes*. A count survives having no denominator for the thing being decided,
because it is a statement about the items rather than about the unit.

**What this module does NOT own.** Not the reporting bar (``jazzpanda_cgst.py``
declares ``Q_REPORTED`` on its own statistic, and a set that fails it is
TESTED_ABSENT at load), not the capability floors (same place, as ``Coverage``, so
the lattice demotes on them), not the measurement states. What lives here is
:data:`Q_ENRICHED` -- the *enriched versus suggestive* split, which is a question
about how cautious this lab has decided to be and not about what the statistic
means. That is the same split jazzPanda's ``BAND_EDGES`` and ``core/verdict.py``
make, drawn in the same place for the same reason.

**The cell-type call is imported, never re-derived.** ``EnrichmentReference`` carries
it as display context and the census header attributes it. Recomputing it here would
give one number two homes and let them disagree; arguing with it from an enrichment
result would be the method speaking outside what it measures. A cross-lineage program
is reported as a tension, never as a re-typing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from panoscope.core.measure import resolve as _resolve
from panoscope.types.contracts import Check, RivalResult
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Coverage, MeasuredState, MeasuredWorld
from panoscope.types.outputs import Finding, FindingItem, TaskKind, Tier
from panoscope.types.subject import gene_set, subject
from panoscope.types.trace import Trace, Traced, fmt_num, ref
from panoscope.types.uispec import (
    BlockSpec, CensusHeader, CensusLine, ColumnKind, ColumnSpec,
    CoverageRibbonSpec, UISpec,
)

WORKFLOW_ID: str = "enrichment-interpretation"

#: The short surface id the shell routes on, deliberately not the skill directory:
#: renaming a tab must not silently re-point a skill.
SURFACE_ID: str = "pathways"

#: The lab's caution, applied to whatever q a registered method hands up. Sets
#: between this and the method's own reporting bar are real-but-weak and carry
#: ``verify``; they are never silently promoted and never silently dropped.
Q_ENRICHED: float = 0.05

#: A footprint this thin clears the capability floor but is not something to act on
#: unexamined, so it sets ``verify`` rather than changing the tier. It cannot change
#: the tier: that would be the panel deciding the result instead of conditioning it.
THIN_COVERAGE: float = 0.05

#: Where the K of N that matters lives, by name. One lookup, not a second copy.
PANEL_COVERAGE: str = "panel_hits"


@dataclass(frozen=True, slots=True)
class EnrichmentReference:
    """The reference side of one characterisation. Selects; never supplies a value.

    ``cell_type`` is IMPORTED from the marker workflow and is display context only.
    Nothing here can become a number in the output: the counts are over
    ``ev.items``, the z and the q come from the method, and the K of N comes from
    the capability frames.

    ``world`` is here for the same reason ``MarkerReference.world`` is, and under
    the same protest: ``Workflow.rival_test`` receives only
    ``(evidence, alternative, reference)``, and deciding whether a rival program was
    *never scored* or *scored and not reported here* needs frame membership, which
    ``NormalisedEvidence`` carries only as digests.
    """

    cell_type: str = ""
    world: MeasuredWorld = field(default_factory=MeasuredWorld)


@dataclass(frozen=True, slots=True)
class EnrichmentInterpretation:
    """The registered workflow object. Declaration plus composition, no statistics."""

    name: str = WORKFLOW_ID
    task_kind: TaskKind = TaskKind.CHARACTERISATION
    question: str = (
        "What biological programs are running in this unit, within what the panel "
        "was able to measure?"
    )
    skill_path: str = "panoscope/skills/enrichment-interpretation/SKILL.md"

    # ----------------------------------------------------------------------- #
    def interpret(self, ev: NormalisedEvidence,
                  reference: Any) -> Traced[Finding]:
        """One unit's gene-set evidence -> the finding, traced. No band anywhere.

        The tier is almost entirely a function of the measurement: ``permitted_tiers``
        leaves exactly one choice open, SUPPORTED versus SUGGESTIVE, and that is a
        question about the size of a number that exists. Everything else -- measured
        absence, below a floor, never scored -- was settled by the lattice at load,
        and re-deciding it here would be a second implementation of the thing
        ``types/`` already owns.
        """
        t = Trace()
        ref_: EnrichmentReference = reference
        items: list[FindingItem] = []
        for i in ev.items:
            tier, verify, rule, because = _tier(i)
            t.note(rule, f"{tier.value}{' + verify' if verify else ''}", because,
                   (ref(i.subject, i.statistic_name, i.statistic),))
            items.append(FindingItem(evidence=i, tier=tier, verify=verify,
                                     because=because))
        return t.done(Finding(
            unit=ev.unit,
            unit_kind=ev.unit_kind,
            task_kind=self.task_kind,
            items=tuple(items),
            summary=_summary(ev, ref_, tuple(items)),
        ))

    def rival_test(self, ev: NormalisedEvidence, alternative: str,
                   reference: Any) -> Traced[RivalResult]:
        """Could another PROGRAM account for this cluster's signal instead?

        The alternative is a gene set, so the rival space is the collection -- the
        same shape as the marker workflow's rival test with cell types swapped out,
        which is why ``RivalResult`` needed nothing added to carry it.

        ``core.rival.rival_test`` is deliberately not reused: it walks a canonical
        marker LIST and builds ``gene`` subjects, and a program is one subject, not
        a list of them. What is reused is the part that matters -- ``core.measure``
        adjudicating the subject against the frames -- so the four states are
        decided in one place here too.
        """
        t = Trace()
        ref_: EnrichmentReference = reference
        subj = subject("gene_set", set=gene_set(alternative))
        item = next((i for i in ev.items if i.subject.key == subj.key), None)

        if item is None:
            # Not among this unit's rows. The frames decide what that means: scored
            # elsewhere in the collection is a measured absence, outside it is not.
            reported = frozenset(
                e.id for i in ev.items if i.measured.carries_number
                for e in i.subject.entities()
            )
            stat_name = ev.items[0].statistic_name if ev.items else ""
            m = t.absorb(_resolve(subj, ref_.world, reported))
            item = EvidenceItem(subject=subj, statistic=None,
                                statistic_name=stat_name, measured=m)

        cov = Coverage("rival_set_in_frame", int(item.measured.testable), 1, 1)
        because = (f"{alternative}: {cov}",)
        refs = (ref(ev.unit_subject), ref(item.subject))

        if item.measured.carries_number:
            t.note("rival:ambiguous", alternative,
                   because + (f"{alternative} is itself enriched in {ev.unit}; "
                              "genuine rival signal, which separates nothing",), refs)
            return t.done(RivalResult(
                alternative=alternative,
                undecidable=(f"{item.subject.display()} is enriched here too "
                             f"({item.statistic_name}="
                             f"{fmt_num(item.statistic)}); both programs are "
                             "running, so the measurement does not choose between "
                             "them",),
            ))
        if item.measured.absence_is_evidence:
            t.note("rival:argued_against", alternative, because, refs)
            return t.done(RivalResult(alternative=alternative,
                                      separated_by=(item,)))
        t.note("rival:undecidable", alternative,
               because + (f"{item.subject.display()} "
                          f"{item.measured.state.value} in {item.measured.frame}; "
                          "it argues nothing",), refs)
        return t.done(RivalResult(
            alternative=alternative,
            undecidable=(f"{item.subject.display()}: "
                         f"{item.measured.state.value} in {item.measured.frame}",
                         f"{cov}"),
        ))

    def checks(self) -> tuple[Check, ...]:
        """What this workflow CAN ask -- declared, so a check that could not be
        settled is visible next to the ones that fired."""
        return (
            Check("panel_coverage",
                  "How many of this program's genes were on the panel at all?"),
            Check("leading_edge",
                  "How many genes actually drive the enrichment, and which?"),
            Check("cross_lineage",
                  "Does this program belong to a different lineage than the "
                  "cluster's imported cell-type call?"),
            Check("program_separation",
                  "Was a named alternative program ruled out by a measured absence, "
                  "or merely unasked?"),
        )

    def uispec(self) -> UISpec:
        """The review surface this question requires.

        A ``CensusHeader``, which has no band slot at all -- the structural form of
        the rule. Four blocks, one per measurement state, so an untestable program
        is a row that says why and not a row that quietly did not render.

        No plots are declared, for the reason the marker surface declares none: a
        ``PlotSpec`` is a promise to draw and R4 makes it conditional on a
        conditionality builder that can state the denominator. Those land with the
        view layer.
        """
        return UISpec(
            workflow_id=SURFACE_ID,
            title="Pathways",
            task_kind=TaskKind.CHARACTERISATION,
            unit_kind="cluster",
            item_kind="gene_set",
            interpretive_question=self.question,
            header=CensusHeader(
                lines=(
                    CensusLine(
                        "PROGRAMS",
                        counts=(("enriched", "n_enriched"),
                                ("suggestive", "n_suggestive"),
                                ("not enriched", "n_not_supported"),
                                ("untestable", "n_untestable")),
                        denominator_field="n_scored",
                        denominator_template="of {value} sets scored",
                    ),
                ),
                top_item_template=(
                    "top {gene_set} {statistic_name} {statistic} on "
                    "{panel_hits}/{set_size} panel genes"
                ),
            ),
            columns=(
                ColumnSpec("gene_set", "Program", ColumnKind.TEXT),
                ColumnSpec("statistic", "cgst z", ColumnKind.NUMBER,
                           unit="cgst_z", align="right"),
                ColumnSpec("p_adj_bh", "q", ColumnKind.NUMBER,
                           unit="q", precision=3, align="right"),
                # TEXT, not COVERAGE_BAR, on purpose. A numeric kind must hatch
                # every row with no statistic -- and those are exactly the rows
                # whose K of N is the REASON there is no statistic. Hatching it
                # would hide the panel-coverage rule on the only rows it decided.
                ColumnSpec("coverage", "On panel (K of N)", ColumnKind.TEXT),
                ColumnSpec("leading_edge", "Driving genes", ColumnKind.NOTE_CHIP),
                ColumnSpec("status", "Status", ColumnKind.GLYPH),
                ColumnSpec("citation", "Biology", ColumnKind.CITATION,
                           sortable=False),
            ),
            blocks=(
                BlockSpec("observed", "PROGRAMS - {count} with a measurement",
                          statuses=(MeasuredState.OBSERVED,),
                          sort_by="statistic"),
                BlockSpec("tested_absent",
                          "SCORED, NOT ENRICHED HERE - {count} below the reporting bar",
                          statuses=(MeasuredState.TESTED_ABSENT,),
                          empty_text="every scored set cleared the reporting bar"),
                BlockSpec("underpowered",
                          "TOO LITTLE OF THE SET ON PANEL - {count}",
                          statuses=(MeasuredState.UNDERPOWERED,),
                          empty_text="every scored set cleared the capability floors"),
                BlockSpec("never_measured",
                          "NEVER SCORED - {count} outside this collection",
                          statuses=(MeasuredState.NEVER_MEASURED,),
                          empty_text="every set asked about was in the collection"),
            ),
            coverage_ribbon=CoverageRibbonSpec(),
            pin_kinds=("gene_set", "cell_type"),
            csv_columns=("unit", "cell_type", "gene_set", "tier", "verify",
                         "statistic", "p_adj_bh", "coverage", "status"),
        )


# --------------------------------------------------------------------------- #
# Deterministic composition. No statistic is computed here; every number is read
# off an EvidenceItem or a Coverage the method already carried.
# --------------------------------------------------------------------------- #
def _q_of(item: EvidenceItem) -> float | None:
    return dict(item.corroborating).get("p_adj_bh")


def _coverage(item: EvidenceItem, what: str = PANEL_COVERAGE) -> Coverage | None:
    return next((c for c in item.measured.coverage if c.what == what), None)


def _tier(item: EvidenceItem) -> tuple[Tier, bool, str, tuple[str, ...]]:
    """``(tier, verify, rule id, because)`` for one item. Deterministic, total.

    ``verify`` is produced HERE, by rubric, and is never offered to a caller or a
    model as an argument. It is the flag that says a human must look again, so a
    layer that could set it could also clear it.

    An untestable item does not carry ``verify``: it is not a call that a second
    look could settle, it is a question this panel could not ask. Saying "re-check"
    about it would read as doubt over a result when there is no result.
    """
    m = item.measured
    if not m.carries_number:
        if m.absence_is_evidence:
            return (Tier.NOT_SUPPORTED, False, "enrichment:not_enriched",
                    (f"scored and not reported for this unit; absence IS evidence "
                     f"here, within {m.frame}",))
        failing = tuple(str(c) for c in m.coverage if not c.ok)
        return (Tier.UNTESTABLE, False, "enrichment:untestable",
                (failing or (f"{m.state.value} in {m.frame}",))
                + ("absence argues nothing",))

    q = _q_of(item)
    cov = _coverage(item)
    thin = cov is not None and cov.fraction < THIN_COVERAGE
    if q is not None and q < Q_ENRICHED:
        because = (f"q={fmt_num(q)} below {fmt_num(Q_ENRICHED)}",) + (
            (f"{cov} is a thin footprint; re-check",) if thin and cov else ())
        return (Tier.SUPPORTED, thin, "enrichment:enriched", because)
    return (Tier.SUGGESTIVE, True, "enrichment:suggestive",
            (f"q={fmt_num(q)} is at or above {fmt_num(Q_ENRICHED)}: real but weak; "
             "re-check this",))


def _summary(ev: NormalisedEvidence, reference: EnrichmentReference,
             items: tuple[FindingItem, ...]) -> str:
    """Per-item counts, the top program, and the panel scope. NOT the narration layer.

    It must render identically with the model offline, so it carries no adjective
    the numbers do not license. There is no band in it and nowhere to put one: the
    counts ARE the header, which is the whole shape of a characterisation.

    The panel-scope sentence is mandatory and deterministic. A reader handed "c2 is
    enriched for interferon gamma response" forms a genome-wide claim unless the
    denominator is said out loud, and the figure cannot say it for them.
    """
    n = {t: sum(1 for i in items if i.tier is t) for t in Tier}
    head = (f"{ev.unit}: {n[Tier.SUPPORTED]} enriched, "
            f"{n[Tier.SUGGESTIVE]} suggestive, "
            f"{n[Tier.NOT_SUPPORTED]} not enriched, "
            f"{n[Tier.UNTESTABLE]} untestable of {len(items)} set(s) scored")
    if reference.cell_type:
        head += (f", read against the {reference.cell_type} call imported from the "
                 "marker workflow")

    top = next((i for i in items if i.evidence.measured.carries_number), None)
    if top is not None:
        cov = _coverage(top.evidence)
        head += (f"; top {top.subject.display()} {top.evidence.statistic_name} "
                 f"{fmt_num(top.evidence.statistic)}")
        if cov is not None:
            head += f" on {cov.k}/{cov.n} panel genes"
    if n[Tier.SUPPORTED] == 0:
        head += ". Nothing clears the enriched bar -- re-check this"
    return (head + ". Panel-scoped: over-representation among the genes this panel "
            "measured, not genome-wide.")


#: The registered instance.
WORKFLOW = EnrichmentInterpretation()
