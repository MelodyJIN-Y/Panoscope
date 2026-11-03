"""The marker workflow: *what IS this unit?* -- and the surface that answers it.

One interpretive question, one task kind, one review surface. The question is an
ASSIGNMENT: there is a label space (the study's annotation label set), so probability
mass has somewhere to sit and a unit-level band is meaningful. That is the whole
reason this workflow may declare a ``LabelBandHeader`` and the enrichment workflow
may not -- ``UISpec.__post_init__`` refuses the other combination, so the rule is
structural rather than reviewed.

**Task kind belongs to the method, not to the use.** ``task_kind`` is declared here
because a workflow *is* a question, and every method registering into it is a
different way of producing evidence that this same question can be asked of. A reader
who points the marker surface at an enrichment table does not thereby turn an
enrichment into an assignment; they get a refusal from the registry
(``panoscope/methods/__init__.py``), which is the honest outcome.

**What this module does NOT own.** Not the band ladder (``methods/jazzpanda.py``
declares the edges for its own statistic), not the promote/demote/cap/floor
(``core/verdict.py`` owns the confidence policy), not the rival bucketing
(``core/rival.py``). ``interpret`` and ``rival_test`` here are composition: they say
which reference list selects which evidence, and they assemble the declared output
shape. Every number in the result came up through ``core/`` from the method's table.
If this file ever computes a statistic, the layering has inverted.

**Reference data selects; it never supplies.** :class:`MarkerReference` carries a
curated canonical-marker list and the rival lists. Those decide *which* measurement
is asked for -- which gene is allowed to drive the band, which absences are worth
reporting -- and nothing else. No field on it can become a value in the output: the
band is read off the driver's ``glm_coef``, the drivers are ``EvidenceItem``\\s from
the method, and the caveats name entities the frame refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from panoscope.core.rival import rival_test as _core_rival_test
from panoscope.core.verdict import REPLICATE_COVERAGE_WHAT, assess as _assess
from panoscope.types.contracts import Check, RivalResult
from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.measured import MeasuredState, MeasuredWorld
from panoscope.types.outputs import Assignment, TaskKind
from panoscope.types.trace import Trace, Traced, fmt_num
from panoscope.types.uispec import (
    BlockSpec, ColumnKind, ColumnSpec, CoverageRibbonSpec, LabelBandHeader, UISpec,
)

WORKFLOW_ID: str = "marker-interpretation"

#: ``UISpec.workflow_id`` uses the short surface id the shell routes on; the
#: workflow's own ``name`` is the skill directory. They are deliberately different
#: strings so that renaming a tab cannot silently re-point a skill.
SURFACE_ID: str = "markers"


@dataclass(frozen=True, slots=True)
class MarkerReference:
    """The reference side of one assignment: a candidate label and its marker lists.

    Passed in rather than looked up, because this layer does no IO and because a
    curated marker list is reference data -- it may select and order evidence, and
    it may generate rivals, but it may never supply a value or count as agreement.

    ``world`` is the awkward field and it is here under protest.
    ``Workflow.rival_test`` receives only ``(evidence, alternative, reference)``, and
    deciding whether a rival's canonical marker was *never measured* or *tested and
    absent* needs the capability frames -- which ``NormalisedEvidence`` carries only
    as digests (``conditions.frames``), not as membership. Deriving the frames from
    ``ev.reported`` instead would be wrong in the one direction that matters: every
    unreported canonical marker would come back NEVER_MEASURED, collapsing the
    productive half of the lattice into "we don't know".
    """

    label: str
    canonical: tuple[str, ...]
    label_id: str | None = None
    rivals: tuple[tuple[str, tuple[str, ...]], ...] = ()   # alternative -> its markers
    world: MeasuredWorld = field(default_factory=MeasuredWorld)
    rival_floor: int = 1

    def canonical_for(self, alternative: str) -> tuple[str, ...] | None:
        """The rival's marker list, or ``None`` if nobody curated one.

        ``None`` and ``()`` are different answers and are kept apart: an empty list
        means the rival has no canonical markers in this reference release, and a
        missing one means the rival was never looked up. Only the second is a gap in
        the apparatus, and the biologist can act on the difference.
        """
        return next((m for name, m in self.rivals if name == alternative), None)


@dataclass(frozen=True, slots=True)
class MarkerInterpretation:
    """The registered workflow object. Declaration plus composition, no statistics."""

    name: str = WORKFLOW_ID
    task_kind: TaskKind = TaskKind.ASSIGNMENT
    question: str = (
        "What cell type is this unit, and how much does the measurement support it?"
    )
    skill_path: str = "panoscope/skills/marker-interpretation/SKILL.md"

    # ----------------------------------------------------------------------- #
    def interpret(self, ev: NormalisedEvidence,
                  reference: Any) -> Traced[Assignment]:
        """One unit's evidence plus one candidate label -> the assignment, traced.

        The partition into ``drivers`` and ``unmeasured`` is by whether a number
        exists, which ``Assignment.__post_init__`` re-checks: a driver with no
        statistic would render as the blank row the interface rule forbids, and an
        "unmeasured" item carrying a number is a caveat about something that was in
        fact measured.

        ``verify`` is taken from ``core/verdict.py`` and is never recomputed, never
        widened and never offered as an argument. It is the flag that says a human
        must look again, so a layer that could set it could also clear it.

        **The §9.10 replicate coverage reaches the band through the unit, not
        through the reference.** :func:`_replicate_coverage` reads it off
        ``ev.unit_measured``, which is where ``Measured.coverage`` already says
        every K-of-N lives and where ``workflows/de.py`` already reads one. The
        alternative was a field on :class:`MarkerReference`, and it would have been
        wrong twice over: a replicate count is a measurement fact rather than
        reference data, and a reference object that can lower a band is exactly the
        thing the rule about reference data forbids. Routing it through the unit
        also means a composition root cannot forget to pass it -- the coverage
        travels with the evidence it describes.
        """
        t = Trace()
        ref_: MarkerReference = reference
        verdict = t.absorb(_assess(ev.items, frozenset(ref_.canonical),
                                   _replicate_coverage(ev)))

        drivers = tuple(i for i in ev.items if i.measured.carries_number)
        unmeasured = tuple(i for i in ev.items if not i.measured.carries_number)

        return t.done(Assignment(
            unit=ev.unit,
            unit_kind=ev.unit_kind,
            label=ref_.label,
            label_id=ref_.label_id,
            band=verdict.band,
            verify=verdict.verify,
            drivers=drivers,
            unmeasured=unmeasured,
            rationale=_rationale(ev, ref_, verdict),
        ))

    def rival_test(self, ev: NormalisedEvidence, alternative: str,
                   reference: Any) -> Traced[RivalResult]:
        """Does the measurement separate this call from ``alternative``?

        ``undecidable`` is not a failure field. "The deciding marker is off panel" is
        the productive output of the panel-absence rule at full strength -- the
        kidney study's whole shape -- and it must be reported as a named gap rather
        than folded into a lower band, where the biologist could not tell it from a
        weak measurement.
        """
        t = Trace()
        ref_: MarkerReference = reference
        markers = ref_.canonical_for(alternative)
        if markers is None:
            # Nobody curated this rival. Say so; do not let an empty list stand in
            # for a lookup that never happened.
            return t.done(RivalResult(
                alternative=alternative,
                undecidable=(f"no canonical marker list is registered for "
                             f"{alternative!r}; the measurement was not asked",),
            ))

        r = t.absorb(_core_rival_test(
            ev, call=ref_.label, alternative=alternative,
            canonical_call=ref_.canonical, canonical_rival=markers,
            world=ref_.world, floor=ref_.rival_floor,
        ))
        return t.done(RivalResult(
            alternative=alternative,
            # Only the measured absences separate: a rival marker that was also
            # observed here is genuine rival signal and separates nothing.
            separated_by=r.against_rival,
            undecidable=tuple(
                f"{i.subject.display()}: {i.measured.state.value} in "
                f"{i.measured.frame}" for i in r.undecidable
            ) + ((f"{r.coverage}",) if not r.settleable else ()),
        ))

    def checks(self) -> tuple[Check, ...]:
        """What this workflow CAN ask -- declared, so a check that could not be
        settled is visible next to the ones that fired.

        A check the biologist cannot see was skipped is a check they will assume
        passed.
        """
        return (
            Check("canonical_support",
                  "Does a canonical marker for this label carry the call?"),
            Check("driver_specificity",
                  "Does the driving marker localise better with another cluster?"),
            Check("panel_coverage",
                  "How many of this label's canonical markers were on the panel at all?"),
            Check("rival_separation",
                  "Was a named alternative ruled out by a measured absence, or merely "
                  "unasked?"),
        )

    def uispec(self) -> UISpec:
        """The review surface this question requires.

        Four blocks, one per measurement state, because the panel-absence rule needs
        somewhere to appear and a never-measured gene must be a *row that says so* --
        not a missing row, which is indistinguishable from a gene that was measured
        and found nothing.

        No plots are declared. A ``PlotSpec`` is a promise to draw, and R4 makes that
        promise conditional on a conditionality builder that can state the cell count,
        the artifact sha and the shared colour maximum. Those are the view layer's and
        are landing separately; declaring a dotted path to a builder that does not yet
        exist would be a surface promising a figure it cannot produce.
        """
        return UISpec(
            workflow_id=SURFACE_ID,
            title="Markers",
            task_kind=TaskKind.ASSIGNMENT,
            unit_kind="cluster",
            item_kind="gene",
            interpretive_question=self.question,
            header=LabelBandHeader(
                label_space="the study's annotation label set",
                band_edges_ref="panoscope.methods.jazzpanda.BAND_EDGES",
            ),
            columns=(
                ColumnSpec("gene", "Gene", ColumnKind.TEXT),
                ColumnSpec("statistic", "glm_coef", ColumnKind.NUMBER,
                           unit="glm_coef", align="right"),
                ColumnSpec("pearson", "Spatial specificity",
                           ColumnKind.SPECIFICITY_PAIR, unit="pearson", align="right"),
                ColumnSpec("status", "Status", ColumnKind.GLYPH),
                ColumnSpec("flags", "Caveats", ColumnKind.NOTE_CHIP),
                ColumnSpec("citation", "Biology", ColumnKind.CITATION, sortable=False),
            ),
            blocks=(
                BlockSpec("observed", "SUPPORTING - {count} with a measurement",
                          statuses=(MeasuredState.OBSERVED,),
                          sort_by="statistic"),
                BlockSpec("tested_absent",
                          "MEASURED, NOT HERE - {count} on panel, localised elsewhere",
                          statuses=(MeasuredState.TESTED_ABSENT,),
                          empty_text="nothing canonical was tested and found absent"),
                BlockSpec("underpowered", "TOO LITTLE SIGNAL - {count}",
                          statuses=(MeasuredState.UNDERPOWERED,),
                          empty_text="every on-panel marker cleared the floor"),
                # NEVER_MEASURED splits on the frame that refused the gene, because
                # the two readings are not interchangeable and the biologist acts on
                # the difference. Off panel: never probed, so nothing can be said of
                # it. No usable signal: it WAS probed, and the reason there is no
                # number is this sample's detectability, not the panel design. One
                # header covering both is a false statement about whichever group it
                # does not describe -- on this demo, about 49 genes out of 49.
                # EXCLUDES rather than matching "panel:", so the two blocks partition
                # the state instead of both selecting from it -- a row carrying an
                # unrecognised frame falls here rather than off the table entirely.
                BlockSpec("never_measured", "NEVER MEASURED - {count} off panel",
                          statuses=(MeasuredState.NEVER_MEASURED,),
                          frame_excludes=("detect:",),
                          empty_text="every canonical marker for this label is on panel"),
                BlockSpec("undetected",
                          "NEVER MEASURED - {count} on panel, no usable signal",
                          statuses=(MeasuredState.NEVER_MEASURED,),
                          frame_prefix="detect:",
                          empty_text="every on-panel gene produced usable signal"),
            ),
            coverage_ribbon=CoverageRibbonSpec(),
            pin_kinds=("gene", "cell_type"),
            csv_columns=("unit", "label", "band", "verify", "gene", "statistic",
                         "pearson", "status"),
        )


def _replicate_coverage(ev: NormalisedEvidence) -> Any:
    """The §9.10 K of N this unit's band must see, or ``None`` if it was never asked.

    ``None`` and ``Coverage(k=0, n=N)`` are different answers and the difference is
    the whole reason this is a lookup rather than a flag. Absent means the question
    was never askable -- one sample, no per-sample tables, independent clustering --
    and the band is untouched, because demoting for the absence of a question would
    punish the single-sample demo on every unit. Present with ``k=0`` means the
    comparison WAS asked and compared nothing, which is a genuinely unsupported
    profile and demotes.

    Matched on :data:`~panoscope.core.verdict.REPLICATE_COVERAGE_WHAT` rather than
    by position, because ``Measured.coverage`` is one home for every K-of-N in the
    system -- panel hits, leading-edge size, cells in a niche -- and taking the
    first would hand ``assess`` whichever one happened to be written first.
    """
    return next((c for c in ev.unit_measured.coverage
                 if c.what == REPLICATE_COVERAGE_WHAT), None)


def _rationale(ev: NormalisedEvidence, reference: MarkerReference,
               verdict: Any) -> str:
    """Deterministic prose, built from the evidence. NOT the narration layer.

    It must render identically with the model offline, so it carries no adjective the
    numbers do not license and no sentence a template cannot produce. The counts are
    the honest headline: a screen of three supporting markers reads as three out of
    three unless the denominator is said out loud.
    """
    observed = sum(1 for i in ev.items if i.measured.carries_number)
    blocked = tuple(i for i in ev.items
                    if i.measured.state is MeasuredState.NEVER_MEASURED)
    driver = verdict.driver
    head = (
        f"{ev.unit} reads as {reference.label} at {verdict.band} "
        f"({verdict.basis}), on {observed} of {len(ev.items)} item(s) carrying a "
        "measurement"
    )
    if driver is not None:
        head += (f"; driven by {driver.subject.display()} "
                 f"{driver.statistic_name}={fmt_num(driver.statistic)}")
    if blocked:
        names = ", ".join(i.subject.display() for i in blocked)
        head += (f". Never measured here: {names} -- absent from the capability "
                 "frame, so their absence argues nothing")
    if verdict.verify:
        head += ". Flagged for re-check"
    return head + "."


#: The registered instance.
WORKFLOW = MarkerInterpretation()
