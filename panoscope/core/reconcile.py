"""Reconciliation: the join across task kinds on ``Subject.key``.

Disagreement and silence are not the same, and silence has two kinds:

- B **never measured** the subject: no conflict; *could not be compared*, naming B and
  the frame that refused it;
- B measured it and did not report it (``TESTED_ABSENT``): that is evidence, and the two
  genuinely conflict.

The gate is ``carries_number or absence_is_evidence``, the same pair of properties
``outputs.permitted_tiers`` derives from. A subject one side never emitted is
adjudicated against that side's own frames (as ``core/rival.py`` does); the synthesised
item carries no statistic.

- Same question or different question (ARCHITECTURE §9.11): only results answering the
  same question can agree; others are coherent or in tension.
- Reference data never counts as agreement: ``Side.authority`` has no default and the
  authority gate runs first; a non-``MEASUREMENT`` side is ``NOT_COMPARABLE`` with
  ``corroborates=False``.
- Dependency structure is mandatory (§9.11): ``Relationship.DERIVED`` clears
  ``corroborates`` while keeping the outcome.
- No band, and no field one could go in (pinned by a field-name guard in the tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.core.measure import resolve
from panoscope.types.claim import Authority
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import MeasuredWorld
from panoscope.types.outputs import TaskKind
from panoscope.types.subject import Subject
from panoscope.types.trace import Trace, Traced, fmt_num, ref


class Relationship(StrEnum):
    """How much of a second witness the other side actually is (§9.11).

    Transcribed from ``IMPLEMENTATION.md`` §6's ``DEPENDENCY`` table. It is the
    caller's declaration, not something inferable here: whether an enrichment run was
    computed *from* the marker result is a fact about the pipeline, and guessing it
    from the outputs is how a derived agreement gets sold as corroboration.
    """

    INDEPENDENT = "independent"                # lab validation, IHC, orthogonal assay
    PARTLY_INDEPENDENT = "partly_independent"  # different questions, same data
    DERIVED = "derived"                        # one result computed from the other


class Agreement(StrEnum):
    """The §9.11 outcome table, both rows, in one enum.

    ``AGREE``/``DISAGREE`` are reachable only for a same-question pair;
    ``COHERENT``/``TENSION`` only for a different-question one. One enum rather than
    two because every consumer wants a single field to branch on, and the
    unreachability is enforced where the mapping happens rather than by a type the
    caller has to pick correctly.
    """

    AGREE = "agree"
    DISAGREE = "disagree"
    COHERENT = "coherent"
    TENSION = "tension"
    NOT_COMPARABLE = "not_comparable"


class Comparison(StrEnum):
    """What the *measurements* say, before the question-sameness mapping.

    Separated from :class:`Agreement` because ``concordance.py`` needs exactly this
    and none of the question machinery: replicates of one unit are by construction
    the same question, so mapping CONSISTENT onto "agree" there would import a word
    that means something it must not mean (§9.10: agreement never raises confidence).
    """

    CONSISTENT = "consistent"
    CONFLICT = "conflict"
    NOT_COMPARABLE = "not_comparable"


class DifferentUnits(ValueError):
    """Raised when two sides describe different units.

    A join is only meaningful within one unit. Reconciling c3's markers against c7's
    DE result joins happily on ``Subject.key`` and produces confident nonsense, and
    nothing downstream could tell -- every pair would look well-formed.
    """


@dataclass(frozen=True, slots=True)
class Side:
    """One method's output for one unit, plus what it was capable of seeing.

    ``authority`` has no default on purpose. It is the field that decides whether this
    side is allowed to corroborate anything, and a default would be the one place the
    rule about reference data could be forgotten silently.

    ``world`` is this side's own capability frames, used to adjudicate subjects the
    *other* side raised and this one has no item for. An empty ``MeasuredWorld`` is
    the honest default: with no frame declared, every such subject resolves to
    ``NEVER_MEASURED``, which reads as "this side cannot speak to it" rather than as
    a tested absence it never earned.
    """

    module: str
    workflow: str          # the interpretive question; a workflow OWNS one
    task_kind: TaskKind
    unit: str
    authority: Authority
    items: tuple[EvidenceItem, ...]
    world: MeasuredWorld = MeasuredWorld()

    @property
    def measurement(self) -> bool:
        return self.authority is Authority.MEASUREMENT


@dataclass(frozen=True, slots=True)
class Reconciliation:
    """What two methods said about ONE subject, and whether that is a second witness.

    ``corroborates`` is a separate field from ``outcome`` because they answer
    different questions and the project loses if they are conflated: the outcome says
    whether the two readings fit, and ``corroborates`` says whether the fit is worth
    anything. A derived agreement is a real agreement and a fake witness.

    ``blocked`` names the side that could not be compared, so the caveat is
    actionable -- "the DE panel never had ERBB2" is something a biologist can act on,
    "not comparable" is not.
    """

    subject: Subject
    outcome: Agreement
    left: EvidenceItem
    right: EvidenceItem
    corroborates: bool
    because: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()   # module ids that could not speak to this subject


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    """The join, for one unit. No band -- see the module docstring.

    ``pairs`` covers the UNION of both sides' subjects, so "could not be compared"
    appears as a row rather than as an absent one. A reconciliation that silently
    drops the subjects one side could not see reports the half of the join that
    happened to work, which is the quiet under-report ``Subject.key`` exists to
    prevent.
    """

    unit: str
    left: str               # module id
    right: str              # module id
    relationship: Relationship
    same_question: bool
    pairs: tuple[Reconciliation, ...]

    def with_outcome(self, outcome: Agreement) -> tuple[Reconciliation, ...]:
        return tuple(p for p in self.pairs if p.outcome is outcome)

    @property
    def corroborated(self) -> tuple[Reconciliation, ...]:
        """Pairs that are a genuine second witness. Never reference, never derived."""
        return tuple(p for p in self.pairs if p.corroborates)


# --------------------------------------------------------------------------- #
# The pure comparison. No tracing, no questions, no authority -- hammerable, and
# shared with concordance.py, which needs this and none of the rest.
# --------------------------------------------------------------------------- #
def _sign(x: float | None) -> int:
    return 0 if x is None else (1 if x > 0 else (-1 if x < 0 else 0))


def _polarity(item: EvidenceItem) -> int:
    """Which way the item points, as -1/0/+1.

    ``direction`` first, falling back to the sign of the statistic. Magnitudes are
    NOT compared: a spatial coefficient and a log fold-change are not on one scale,
    and "these two numbers are close" across methods is a claim neither measurement
    supports. Sign is the most two methods can be asked to agree about.
    """
    if item.direction is not None:
        return 1 if item.direction.value == "up" else -1
    return _sign(item.statistic)


def _compare(a: EvidenceItem, a_name: str,
             b: EvidenceItem, b_name: str) -> tuple[Comparison, tuple[str, ...],
                                                    tuple[str, ...]]:
    """Two readings of one subject -> (comparison, because, blocked side names).

    The lattice gate comes first: a ``NEVER_MEASURED`` or ``UNDERPOWERED`` side has made no
    observation and can neither agree nor conflict. Both sides are checked, so a pair
    blocked on both says so. Past the gate:

    - both observed    -> polarity;
    - one observed, one ``TESTED_ABSENT`` -> **conflict** (the productive half);
    - both ``TESTED_ABSENT`` -> consistent.
    """
    blocked: list[str] = []
    because: list[str] = []
    for item, name in ((a, a_name), (b, b_name)):
        m = item.measured
        if not (m.carries_number or m.absence_is_evidence):
            blocked.append(name)
            because.append(
                f"{name}: {item.subject.key} is {m.state.value} in {m.frame}; "
                "absence is not evidence there, so it cannot be compared"
            )
    if blocked:
        return Comparison.NOT_COMPARABLE, tuple(because), tuple(blocked)

    if a.measured.carries_number and b.measured.carries_number:
        if _polarity(a) == _polarity(b):
            return Comparison.CONSISTENT, (
                f"{a_name} {a.statistic_name}={fmt_num(a.statistic)} and "
                f"{b_name} {b.statistic_name}={fmt_num(b.statistic)} point the same way",
            ), ()
        return Comparison.CONFLICT, (
            f"{a_name} {a.statistic_name}={fmt_num(a.statistic)} and "
            f"{b_name} {b.statistic_name}={fmt_num(b.statistic)} point opposite ways",
        ), ()

    if a.measured.carries_number or b.measured.carries_number:
        seen, absent = (a, b) if a.measured.carries_number else (b, a)
        seen_n, absent_n = ((a_name, b_name) if a.measured.carries_number
                            else (b_name, a_name))
        return Comparison.CONFLICT, (
            f"{seen_n} reports {seen.statistic_name}={fmt_num(seen.statistic)}; "
            f"{absent_n} had it in frame {absent.measured.frame}, tested it, and did "
            "not report it -- a measured absence, so they genuinely conflict",
        ), ()

    return Comparison.CONSISTENT, (
        f"{a_name} and {b_name} both had it in frame, tested it, and neither "
        "reported it",
    ), ()


# The §9.11 table, as data. Same question agrees or disagrees; a different question
# may only be coherent or in tension, and NEVER agrees.
_SAME: dict[Comparison, Agreement] = {
    Comparison.CONSISTENT: Agreement.AGREE,
    Comparison.CONFLICT: Agreement.DISAGREE,
    Comparison.NOT_COMPARABLE: Agreement.NOT_COMPARABLE,
}
_DIFFERENT: dict[Comparison, Agreement] = {
    Comparison.CONSISTENT: Agreement.COHERENT,
    Comparison.CONFLICT: Agreement.TENSION,
    Comparison.NOT_COMPARABLE: Agreement.NOT_COMPARABLE,
}


def _reported(side: Side) -> frozenset[str]:
    # ponytail: flat ids, as the lattice itself keys on bare ids -- the same trade
    # core/measure.py and core/rival.py make, and for the same reason.
    return frozenset(
        e.id for i in side.items if i.measured.carries_number
        for e in i.subject.entities()
    )


def _silent(subj: Subject, side: Side, reported: frozenset[str]) -> Traced[EvidenceItem]:
    """A subject this side emitted no item for, adjudicated against its own frames.

    It carries no statistic by construction, which is what makes the two kinds of
    silence separable: resolving against ``side.world`` is the difference between
    "never in this panel" and "in the panel, tested, not reported", and a missing row
    on its own cannot tell them apart.
    """
    t = Trace()
    m = t.absorb(resolve(subj, side.world, reported))
    name = side.items[0].statistic_name if side.items else ""
    return t.done(EvidenceItem(subject=subj, statistic=None, statistic_name=name,
                               measured=m))


def _note_outcome(t: Trace, outcome: Agreement, subj: Subject,
                  because: tuple[str, ...]) -> None:
    """One decision per pair, under a LITERAL rule id.

    The ids are written out one branch each rather than looked up in a
    ``{outcome: rule}`` table, for the reason ``core/measure.py`` gives:
    ``test_every_decision_rule_is_registered`` scans for literal ids passed to
    ``note``, so a computed id is a decision the static check cannot see.
    """
    anchor = (ref(subj),)
    if outcome is Agreement.AGREE:
        t.note("reconcile:agree", subj.key, because, anchor)
    elif outcome is Agreement.DISAGREE:
        t.note("reconcile:disagree", subj.key, because, anchor)
    elif outcome is Agreement.COHERENT:
        t.note("reconcile:coherent", subj.key, because, anchor)
    elif outcome is Agreement.TENSION:
        t.note("reconcile:tension", subj.key, because, anchor)
    else:
        t.note("reconcile:not_comparable", subj.key, because, anchor)


def reconcile(left: Side, right: Side,
              relationship: Relationship) -> Traced[ReconcileResult]:
    """Join two methods' outputs for one unit on ``Subject.key``.

    Gate order is the audit and it is code order: **authority first** (a filter, not a
    downgrade -- reference data is refused before anything looks at a number), then
    the measurement-status lattice, then the comparison, then the question-sameness
    mapping. ``relationship`` clears ``corroborates`` last, because a derived pair
    still has a real outcome worth showing and only the witness claim is false.

    Pair order is left's item order, then right's extras in right's order. Neither is
    re-sorted: left's order is the method's own (glm_coef descending for jazzPanda),
    and sorting a join whose rows mostly have no number would invent a ranking the
    data does not support -- and this order is hashed into the record.
    """
    if left.unit != right.unit:
        raise DifferentUnits(
            f"cannot reconcile {left.module} on {left.unit!r} against "
            f"{right.module} on {right.unit!r}: a join is only meaningful within "
            "one unit, and across units it produces well-formed nonsense"
        )
    t = Trace()
    same_question = (left.workflow == right.workflow
                     and left.task_kind is right.task_kind)
    table = _SAME if same_question else _DIFFERENT

    if relationship is Relationship.DERIVED:
        t.note("reconcile:derived_is_not_corroboration", relationship.value,
               (f"{right.module} is derived from {left.module}; agreement here is "
                "the same result twice, so no pair may count as a second witness",),
               (ref(f"module:{left.module}"), ref(f"module:{right.module}")))

    reference = tuple(s.module for s in (left, right) if not s.measurement)
    if reference:
        t.note("reconcile:reference_cannot_corroborate", ", ".join(reference),
               tuple(f"{s.module} carries authority {s.authority.name}, not "
                     "MEASUREMENT; reference data may select and order evidence but "
                     "may never supply a value or count as agreement"
                     for s in (left, right) if not s.measurement),
               tuple(ref(f"module:{m}") for m in reference))

    rep_l, rep_r = _reported(left), _reported(right)
    by_l = {i.subject.key: i for i in left.items}
    by_r = {i.subject.key: i for i in right.items}
    order = [i.subject for i in left.items] + [
        i.subject for i in right.items if i.subject.key not in by_l
    ]

    pairs: list[Reconciliation] = []
    for subj in order:
        a = by_l.get(subj.key) or t.absorb(_silent(subj, left, rep_l))
        b = by_r.get(subj.key) or t.absorb(_silent(subj, right, rep_r))

        if reference:
            outcome = Agreement.NOT_COMPARABLE
            because: tuple[str, ...] = tuple(
                f"{m}: reference authority, which may never corroborate a measurement"
                for m in reference
            )
            blocked: tuple[str, ...] = reference
        else:
            cmp_, because, blocked = _compare(a, left.module, b, right.module)
            outcome = table[cmp_]

        corroborates = (outcome in (Agreement.AGREE, Agreement.COHERENT)
                        and relationship is not Relationship.DERIVED
                        and not reference)
        _note_outcome(t, outcome, subj, because)
        pairs.append(Reconciliation(subject=subj, outcome=outcome, left=a, right=b,
                                    corroborates=corroborates, because=because,
                                    blocked=blocked))

    return t.done(ReconcileResult(
        unit=left.unit, left=left.module, right=right.module,
        relationship=relationship, same_question=same_question, pairs=tuple(pairs),
    ))
