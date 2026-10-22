"""Outputs: what an interpretation says, and the two shapes it is allowed to say it in.

``EvidenceItem`` is the input side (``evidence.py``); everything here is the output
side -- tier, ``verify``, band, summary. The split is what stops a method grading its
own homework, so nothing in this module may be constructed by a method module.

**Two output types, not one, because of the label space.** ``ASSIGNMENT`` asks "what
*is* this?", so there is a set of alternatives and probability mass has somewhere to
sit: a band is meaningful. ``CHARACTERISATION`` and ``COMPARISON`` have no label
space, so a unit-level band is *undefined*, not merely unhelpful -- there is no
"instead" for the remaining confidence to be confidence *in*. That asymmetry is the
entire boundary between :class:`Assignment` and :class:`Finding`, and it is why
``Finding`` is a separate type rather than an ``Assignment`` with ``band=None``: an
optional field invites a caller to fill it, a missing field cannot be filled.

**Four tiers, not the three the spec lists.** ``IMPLEMENTATION.md`` §2.4 gives
``SUPPORTED | SUGGESTIVE | UNTESTABLE``, which collapses two different answers into
one word for exactly the reason ``measured.py`` refuses to collapse four states into
a boolean. ``UNTESTABLE`` means *we could not ask the question*. A subject that was
in frame, was tested and came back absent HAS been asked and HAS been answered, and
the answer argues **against** -- that is the productive half of the panel-absence
rule, the half that lets a discriminator rule a rival out because the rival's markers
were measured and peaked elsewhere. Filed under ``UNTESTABLE`` it reads to the
biologist as "we don't know", and the discriminator silently stops discriminating
while every test stays green. So: ``NOT_SUPPORTED``.

**The tier/measured correspondence is derived, not re-listed.** ``permitted_tiers``
reads ``Measured.carries_number`` and ``Measured.absence_is_evidence`` -- the same two
properties ``evidence.py`` derives its IFF from -- rather than enumerating states a
second time. A hand-maintained second list of states is how the §2.2 spec bug got
written down, and the copy nobody updates is always the one that governs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.types.evidence import Direction, EvidenceItem
from panoscope.types.measured import Measured
from panoscope.types.subject import Subject


class TaskKind(StrEnum):
    """Frozen. Task kind belongs to the METHOD, not to the use a reader puts it to."""

    ASSIGNMENT = "assignment"              # what IS this?         -> has a label space
    CHARACTERISATION = "characterisation"  # what is RUNNING here? -> no label space
    COMPARISON = "comparison"              # does this MAKE SENSE? -> no label space


class Tier(StrEnum):
    """How one item stands, once the measurement status has been taken into account.

    ``NOT_SUPPORTED`` and ``UNTESTABLE`` are different answers and must stay apart:
    "we measured it and it argues against" versus "we could not look".
    """

    SUPPORTED = "supported"          # observed, and it carries the call
    SUGGESTIVE = "suggestive"        # observed, but weakly
    NOT_SUPPORTED = "not_supported"  # tested, absent -> argues AGAINST
    UNTESTABLE = "untestable"        # never measured or underpowered -> no evidence


class MisTieredItem(ValueError):
    """Raised when a tier contradicts the measurement status underneath it.

    Both directions are failures. A ``SUPPORTED`` item over a never-measured subject
    claims support that no number exists for; an ``UNTESTABLE`` item over a tested
    absence throws away the half of the lattice that argues against a rival, and
    reports a refusal the biologist cannot distinguish from a real one.
    """


class AssignmentIsNotAFinding(ValueError):
    """Raised when a ``Finding`` is given ``task_kind=ASSIGNMENT``.

    ``Finding`` is the no-band shape. An assignment routed through it is an
    assignment whose band has been dropped on the floor rather than computed -- the
    unit reads as interpreted while the one number the biologist acts on is gone.
    """


class MisfiledEvidence(ValueError):
    """Raised when ``Assignment.drivers``/``unmeasured`` are filled the wrong way.

    The two tuples partition the evidence by whether a number exists. A driver with
    no statistic renders as a blank cell in the drivers table -- the blank row the
    interface rule forbids -- and an "unmeasured" item carrying a number is a caveat
    about something that was, in fact, measured.
    """


def permitted_tiers(measured: Measured) -> frozenset[Tier]:
    """Which tiers the measurement status allows. The whole reason for four states.

    Derived from ``carries_number``/``absence_is_evidence`` so that adding a fifth
    ``MeasuredState``, or widening either property, moves this with it instead of
    leaving a stale second list behind.

    The tier is therefore almost entirely a function of the measurement: the only
    judgement left to the rubric is ``SUPPORTED`` versus ``SUGGESTIVE``, which is a
    question about the size of a number that exists. That is the intended shape --
    "the band is a function of one method's output; every other layer can only
    subtract or flag".
    """
    if measured.carries_number:
        return frozenset({Tier.SUPPORTED, Tier.SUGGESTIVE})
    if measured.absence_is_evidence:
        return frozenset({Tier.NOT_SUPPORTED})
    return frozenset({Tier.UNTESTABLE})


@dataclass(frozen=True, slots=True)
class FindingItem:
    """One interpreted item. WRAPS its evidence rather than copying it.

    ``subject`` and ``direction`` delegate. Copying ``statistic`` onto this type
    would give one number two homes, and the statistic/measured IFF would then need
    enforcing in both -- which means it would end up enforced in one.

    ``verify`` arrives from the deterministic rubric. It is a plain field because
    this type stores a decision someone else made; it must never be computed here,
    and never from anything the model said.
    """

    evidence: EvidenceItem
    tier: Tier
    verify: bool
    because: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        allowed = permitted_tiers(self.evidence.measured)
        if self.tier not in allowed:
            raise MisTieredItem(
                f"{self.subject.key}: tier {self.tier.value!r} with measured state "
                f"{self.evidence.measured.state.value!r} "
                f"(frame {self.evidence.measured.frame!r}); permitted here: "
                f"{sorted(t.value for t in allowed)}"
            )

    @property
    def subject(self) -> Subject:
        return self.evidence.subject

    @property
    def direction(self) -> Direction | None:
        return self.evidence.direction


@dataclass(frozen=True, slots=True)
class Assignment:
    """ASSIGNMENT only. Has a label space, therefore has a band.

    ``unmeasured`` carries whole ``EvidenceItem``s, not bare names, because the
    caveat has to say WHICH constituent blocked the call -- "CXCR4 was never
    measured", not "something was unmeasured". ``Measured.blocked_by`` already holds
    the entities and the frame that excluded them; a tuple of strings throws that
    away and leaves the biologist a caveat they cannot act on.

    ``rationale`` is deterministic prose built in core from the evidence. It is NOT
    the gated narration layer: it must render identically with the model offline.
    """

    unit: str
    unit_kind: str          # open vocabulary; only TaskKind is frozen
    label: str
    label_id: str | None
    band: str
    verify: bool
    drivers: tuple[EvidenceItem, ...]
    unmeasured: tuple[EvidenceItem, ...]
    rationale: str

    def __post_init__(self) -> None:
        for d in self.drivers:
            if not d.measured.carries_number:
                raise MisfiledEvidence(
                    f"{self.unit}: driver {d.subject.key} has no statistic "
                    f"({d.measured.state.value}); it belongs in unmeasured"
                )
        for u in self.unmeasured:
            if u.measured.carries_number:
                raise MisfiledEvidence(
                    f"{self.unit}: unmeasured {u.subject.key} carries "
                    f"{u.statistic_name}={u.statistic!r}; it belongs in drivers"
                )


@dataclass(frozen=True, slots=True)
class Finding:
    """CHARACTERISATION and COMPARISON. No band. No confidence. Ever.

    There is no label space, so there is no alternative for the remaining
    probability mass to sit on and a unit-level band is undefined. The header for
    these is per-item counts -- "4 enriched, 3 untestable, top EMT z 4.2 on 18/200
    panel genes" -- which is a statement about the items and survives having no
    denominator.

    The absence of a band is enforced as a FIELD-NAME guard in the tests rather than
    by checking that nothing populates one: a behavioural test only catches a band
    that is filled in, while the field guard fails the moment the field is
    *declared*, which is when the mistake costs nothing to undo.
    """

    unit: str
    unit_kind: str          # open vocabulary; only TaskKind is frozen
    task_kind: TaskKind     # never ASSIGNMENT
    items: tuple[FindingItem, ...]
    summary: str

    def __post_init__(self) -> None:
        if self.task_kind is TaskKind.ASSIGNMENT:
            raise AssignmentIsNotAFinding(
                f"{self.unit}: ASSIGNMENT has a label space and therefore a band; "
                "it must be an Assignment, not a Finding"
            )

    @property
    def untestable_count(self) -> int:
        """Items the measurement could not ask about. Deliberately NOT a count of
        everything without a number: ``NOT_SUPPORTED`` items were asked and
        answered, and counting them here would re-collapse the distinction this
        module exists to keep."""
        return sum(1 for i in self.items if i.tier is Tier.UNTESTABLE)
