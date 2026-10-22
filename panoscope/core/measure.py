"""The measure step: what the apparatus was CAPABLE of seeing, recorded as decisions.

This is where "the panel said no" stops being a value and becomes a click-backable
decision. The split with ``types/measured.py`` is deliberate and is the layering of
the whole ``core/`` package:

- ``types.measured.resolve`` is the **lattice** -- a meet over required roles,
  demoted by coverage. Pure data in, pure data out, no tracing. It is the thing a
  property test can hammer.
- this module **wraps** it: it calls the lattice, records which of the four states
  came out and why, binds (or withholds) the statistic, and hands back
  ``Traced[...]``.

Keeping the computation out of the traced layer is what stops the record from
becoming the implementation. A reader can delete every ``note`` below and the
measurement status is unchanged; delete the lattice and nothing works. That is the
right dependency direction for something whose output is audited.

**The failure this module exists to prevent.** ``EvidenceItem`` raises when a
statistic and a measurement status disagree, so a fabricated number is already
unrepresentable. The remaining failure is quieter and worse: a number silently
dropped. An item whose statistic vanished without a decision is indistinguishable
from a method that reported nothing at all -- so the biologist cannot tell a real
refusal from a lost measurement, and the panel-absence rule reads as paralysis
rather than as a finding. Every withholding here is therefore *recorded*, naming
the number that was withheld and the state that forbade it.

**Corroborators are withheld with the statistic.** ``EvidenceItem`` guards the
``statistic`` field only, but ``corroborating`` is a second channel for numbers
about the same subject: a pearson carried alongside a withheld glm_coef would put
a measurement back into an item the frame says was never measured. They travel
together or not at all.

One decision per subject, not one per lattice step: the lattice returns a single
settled state, and re-deriving the pre-demotion state here would be a second
implementation of the thing ``types/`` already owns -- the duplicate nobody updates.
``Measured.because`` already carries both halves ("what blocked the meet" and
"which coverage failed"), so it is quoted rather than re-written.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoscope.types.conditions import Conditions
from panoscope.types.evidence import Direction, EvidenceItem, NormalisedEvidence
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.measured import resolve as _lattice
from panoscope.types.subject import Subject
from panoscope.types.trace import Trace, Traced, fmt_num, ref


@dataclass(frozen=True, slots=True)
class Reading:
    """One row of method output, BEFORE capability is applied.

    Deliberately not an ``EvidenceItem``: a reading is what the method claims it
    found, and an item is what survives the frame. The difference between the two
    is exactly what this module records, so collapsing them would leave nothing to
    audit -- the method would be grading its own homework, which is the arrangement
    the whole project exists to replace.

    ``statistic`` is whatever the method reported, including a number for a subject
    the frame will refuse. That is not an error to construct; it is the input the
    measure step is here to adjudicate.
    """

    subject: Subject
    statistic: float | None
    statistic_name: str
    coverage: tuple[Coverage, ...] = ()
    direction: Direction | None = None
    flags: tuple[str, ...] = ()                        # REGISTERED ids, never prose
    corroborating: tuple[tuple[str, float], ...] = ()  # sorted at the EvidenceItem
    shared_with: tuple[str, ...] = ()


def resolve(subject: Subject, world: MeasuredWorld, reported: frozenset[str],
            coverage: tuple[Coverage, ...] = ()) -> Traced[Measured]:
    """Measurement status for one subject, with the decision that produced it.

    The rule ids are written as literals, one branch each, rather than looked up in
    a ``{state: rule}`` table. The table reads better and is wrong:
    ``test_every_decision_rule_is_registered`` scans for *literal* ids passed to
    ``note``, so a computed id is a decision the static check cannot see -- the one
    place an unregistered rule could re-enter after all this.

    ``refs`` carries the bare subject anchor and never a number here: this function
    settles capability, and a ref with a statistic in it would assert a measurement
    for a subject that may have none.
    """
    t = Trace()
    m = _lattice(subject, world, reported, coverage)
    anchor = ref(subject)

    if m.state is MeasuredState.OBSERVED:
        t.note("measure:observed", m.state.value,
               m.because or (f"in frame {m.frame} and reported",), (anchor,))
    elif m.state is MeasuredState.TESTED_ABSENT:
        t.note("measure:tested_absent", m.state.value,
               m.because or (f"in frame {m.frame}, tested, not reported",), (anchor,))
    elif m.state is MeasuredState.UNDERPOWERED:
        t.note("measure:coverage_floor", m.state.value,
               m.because or (f"below a declared floor in {m.frame}",), (anchor,))
    else:
        t.note("measure:never_measured", m.state.value,
               m.because or (f"outside frame {m.frame}",), (anchor,))
    return t.done(m)


def measure(reading: Reading, world: MeasuredWorld,
            reported: frozenset[str]) -> Traced[EvidenceItem]:
    """One reading, adjudicated against the frame, as a validated ``EvidenceItem``.

    A number survives only where the subject was OBSERVED. Where it does not, the
    withholding gets its own decision quoting the number that was refused -- the
    audit is worth more than the tidiness of dropping it, because "a statistic was
    withheld" without the value cannot tell a reviewer whether the method
    contradicted the frame badly or trivially.
    """
    t = Trace()
    m = t.absorb(resolve(reading.subject, world, reported, reading.coverage))
    anchor = ref(reading.subject)
    grounded = m.carries_number

    if reading.statistic is not None and not grounded:
        dropped = ", ".join(f"{k}={fmt_num(v)}" for k, v in sorted(reading.corroborating))
        because = (
            f"{reading.statistic_name}={fmt_num(reading.statistic)} was reported but "
            f"is withheld: the subject is {m.state.value} in {m.frame}",
        ) + ((f"corroborating {dropped} withheld with it",) if dropped else ())
        if m.state is MeasuredState.NEVER_MEASURED:
            t.note("measure:never_measured", "statistic withheld", because, (anchor,))
        elif m.state is MeasuredState.TESTED_ABSENT:
            t.note("measure:tested_absent", "statistic withheld", because, (anchor,))
        else:
            t.note("measure:coverage_floor", "statistic withheld", because, (anchor,))

    return t.done(EvidenceItem(
        subject=reading.subject,
        statistic=reading.statistic if grounded else None,
        statistic_name=reading.statistic_name,
        measured=m,
        corroborating=reading.corroborating if grounded else (),
        direction=reading.direction,
        flags=reading.flags,
        shared_with=reading.shared_with,
    ))


def measure_unit(unit: str, unit_kind: str, unit_subject: Subject, module: str,
                 readings: tuple[Reading, ...], world: MeasuredWorld,
                 reported: tuple[Frame, ...], conditions: Conditions,
                 unit_coverage: tuple[Coverage, ...] = ()) -> Traced[NormalisedEvidence]:
    """Every reading for ONE unit, reduced to the one shape the interpreters consume.

    ``reported`` arrives as frames rather than as a flat id set because that is what
    ``NormalisedEvidence`` carries, and re-deriving it here from a flat set would
    mean guessing which kind each id belonged to. The lattice wants ids, so they are
    flattened on the way in.

    The unit's own capability verdict is resolved FIRST, and its decision therefore
    precedes every item's. That ordering is load-bearing: "this unit has four cells"
    explains every untestable item under it, and a record that states it afterwards
    reads as an afterthought rather than as the reason.
    """
    t = Trace()
    # ponytail: flat union, as the lattice itself keys on bare ids. A gene id equal
    # to a cell-type label would cross-report; frame membership is checked first, so
    # the collision can only ever promote TESTED_ABSENT to OBSERVED within a frame
    # that already admits the entity. Kind-aware reporting is a lattice change.
    ids: frozenset[str] = frozenset().union(*(f.members for f in reported)) if reported \
        else frozenset()

    unit_measured = t.absorb(resolve(unit_subject, world, ids, unit_coverage))
    items = tuple(t.absorb(measure(r, world, ids)) for r in readings)

    return t.done(NormalisedEvidence(
        unit=unit,
        unit_kind=unit_kind,
        unit_subject=unit_subject,
        unit_measured=unit_measured,
        module=module,
        items=items,
        reported=reported,
        conditions=conditions,
    ))
