"""The marker workflow: *what IS this unit?* -- and the surface that answers it.

An ASSIGNMENT: there is a label space (the study's annotation label set), so a
unit-level band is meaningful and this workflow may declare a ``LabelBandHeader``
(``UISpec.__post_init__`` refuses it for the other task kinds).

**What this module does NOT own:** the band ladder (``methods/jazzpanda.py``), the
promote/demote/cap/floor (``core/verdict.py``), the rival bucketing (``core/rival.py``).
``interpret`` and ``rival_test`` are composition; every number comes up through
``core/`` from the method's table.

**Reference data selects; it never supplies.** :class:`MarkerReference`'s lists decide
*which* measurement is asked for, and no field on it can become a value in the output.
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

    Passed in (this layer does no IO). ``world`` is here because ``rival_test``
    needs the capability frames to tell *never measured* from *tested absent*, and
    ``NormalisedEvidence`` carries them only as digests.
    """

    label: str
    canonical: tuple[str, ...]
    label_id: str | None = None
    rivals: tuple[tuple[str, tuple[str, ...]], ...] = ()   # alternative -> its markers
    world: MeasuredWorld = field(default_factory=MeasuredWorld)
    rival_floor: int = 1
    called: bool = True     # False: the unit has no label yet, so no band

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

        ``drivers`` vs ``unmeasured`` partitions on whether a number exists
        (re-checked by ``Assignment.__post_init__``). ``verify`` comes from
        ``core/verdict.py``, never recomputed or passed in.

        The §9.10 replicate coverage reaches the band through the unit
        (:func:`_replicate_coverage`), not the reference: it is a measurement fact,
        and a reference that could lower a band is what the reference-data rule
        forbids.
        """
        t = Trace()
        ref_: MarkerReference = reference
        verdict = t.absorb(_assess(ev.items, frozenset(ref_.canonical),
                                   _replicate_coverage(ev), called=ref_.called))

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

        ``undecidable`` is a named gap (e.g. the deciding marker is off panel), not a
        failure and not a lower band.
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

        Four blocks, one per measurement state: a never-measured gene is a *row that
        says so*, not a missing row. No plots are declared yet: a ``PlotSpec`` needs
        a conditionality builder (R4) that the view layer does not have.
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
                # NEVER_MEASURED splits on the refusing frame: off panel (never
                # probed) vs no usable signal (probed, this sample's detectability).
                # EXCLUDES "panel:" so the two blocks partition the state and an
                # unrecognised frame falls here.
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

    ``None`` (never askable: band untouched) differs from ``Coverage(k=0, n=N)``
    (asked, compared nothing: demotes). Matched on
    :data:`~panoscope.core.verdict.REPLICATE_COVERAGE_WHAT`, not position:
    ``Measured.coverage`` holds every K-of-N.
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
    if not reference.called:
        return (f"{ev.unit} is not yet called: no label, so no band, on {observed} "
                f"of {len(ev.items)} item(s) carrying a measurement.")
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
