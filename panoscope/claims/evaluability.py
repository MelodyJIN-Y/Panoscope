"""Can the instrument say anything about this claim? The capability half of §7.5.

ARCHITECTURE §7.4 found the single conditional-validity check constant (0.000 bits over
280 curated edges); §7.5 splits it:

* **this module** -- a *capability* gate on every claim: are the entities it rests on
  measured here? A fact about the instrument, not a judgement about the claim.
* :mod:`panoscope.claims.transfer` -- condition deltas, manifest-sourced claims only.

Off-panel and undetectable are kept apart (49 of 280 demo panel genes yield no usable
spatial signal): a refusing frame is a detectability frame iff its ``name`` starts
``detect:`` (``measured.py``'s naming convention).

Not ``measured.resolve``: a claim rests on more than its subject (``depends_on``), and
the point is to name *which* frame refused. ``Frame.contains`` stays the authority.

Pure: no IO, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.types.claim import Claim
from panoscope.types.measured import MeasuredWorld
from panoscope.types.subject import Entity, Gate
from panoscope.types.trace import Trace, Traced, ref

DETECTABILITY = "detect:"
"""Frame-name prefix for a detectability frame (``measured.py``'s stated convention).

A plain membership frame says what the apparatus was *aimed* at; a ``detect:`` frame
says what it actually returned usable signal for. Both refuse entities, and the two
refusals mean opposite things to a reader, so the prefix is load-bearing."""


class Evaluable(StrEnum):
    """Four states, ordered worst-first by :data:`_RANK`.

    Only ``EVALUABLE`` licenses using the claim as support. The other three are all
    *withheld with a reason*: §10.3 is explicit that silent dropping is the failure
    mode, not a safe default.
    """

    NO_FRAME = "no_frame"                # nothing declared for that entity kind
    NEVER_MEASURED = "never_measured"    # outside the frame -- panel-absence
    UNDETECTABLE = "undetectable"        # in frame, no usable signal -- §7.5
    EVALUABLE = "evaluable"


_RANK: dict[Evaluable, int] = {
    Evaluable.NO_FRAME: 0,
    Evaluable.NEVER_MEASURED: 1,
    Evaluable.UNDETECTABLE: 2,
    Evaluable.EVALUABLE: 3,
}


@dataclass(frozen=True, slots=True)
class Evaluability:
    """What the instrument can do about one claim here, and what blocked it."""

    state: Evaluable
    frame: str = ""                       # the frame that refused; "" when none did
    blocked_by: tuple[Entity, ...] = ()   # the entities it refused

    @property
    def usable(self) -> bool:
        return self.state is Evaluable.EVALUABLE


def rests_on(claim: Claim) -> tuple[Entity, ...]:
    """The entities a claim's truth depends on, in a stable order, deduplicated.

    The subject's REQUIRED roles (MEMBER roles are coverage, not identity -- the same
    exclusion ``Subject.key`` and ``measured.resolve`` already make) plus
    ``depends_on``. ``object`` is deliberately NOT included: it is a bare string whose
    entity kind the predicate declares but the claim does not carry, and guessing a
    kind here is how a cell-type label would end up checked against the gene panel.
    A writer that wants the object gated puts it in ``depends_on``, which is the field
    that exists for exactly that.
    """
    out: list[Entity] = [e for _r, e in claim.subject.roles_with_gate(Gate.REQUIRED)]
    out.extend(e for e in claim.depends_on if e not in out)
    return tuple(out)


def _state_for(entity: Entity, world: MeasuredWorld) -> tuple[Evaluable, str]:
    """One entity against every frame for its kind. Worst answer wins, frame named."""
    frames = world.for_kind(entity.kind)
    if not frames:
        return Evaluable.NO_FRAME, f"no frame declared for {entity.kind.value}"
    for f in frames:
        if not f.contains(entity):
            return (
                (Evaluable.UNDETECTABLE if f.name.startswith(DETECTABILITY)
                 else Evaluable.NEVER_MEASURED),
                f.name,
            )
    return Evaluable.EVALUABLE, frames[0].name


def evaluability(claim: Claim, world: MeasuredWorld) -> Traced[Evaluability]:
    """Are the entities this claim rests on measured here? (IMPLEMENTATION §5.)

    The meet over every entity, naming the frame that pulled it down. One decision,
    not one per entity: a reader needs the settled answer and the reason, and a
    per-entity stream would bury the reason in the common case where a claim rests on
    one gene and one cell type.

    Applies to every claim in every authority class, which is the half of §7.5 that
    makes a curated edge adjudicable at all -- by the measurement, never by its own
    formation conditions (§7.6).
    """
    t = Trace()
    anchor = ref(f"claim:{claim.id}")
    entities = rests_on(claim)

    if not entities:
        # Nothing to gate. Honest rather than convenient: a claim resting on no
        # entity cannot be refused by any instrument, so it is evaluable everywhere,
        # and saying otherwise would invent a capability failure out of an empty set.
        t.note("claims:evaluable", Evaluable.EVALUABLE.value,
               ("the claim rests on no entity; no frame can refuse it",), (anchor,))
        return t.done(Evaluability(Evaluable.EVALUABLE))

    per = tuple((e,) + _state_for(e, world) for e in entities)
    worst = min(_RANK[s] for _e, s, _f in per)
    state = next(s for _e, s, _f in per if _RANK[s] == worst)
    blocked = tuple(e for e, s, _f in per if _RANK[s] == worst) \
        if state is not Evaluable.EVALUABLE else ()
    frame = next(f for _e, s, f in per if _RANK[s] == worst)

    because = (
        f"{', '.join(str(e) for e in blocked)} {state.value} in {frame}",
    ) if blocked else (f"all of {', '.join(str(e) for e in entities)} in {frame}",)

    # Literal rule ids, one branch each: test_every_decision_rule_is_registered
    # scans for literals passed to .note(), so a computed id is a decision the
    # static check cannot see. Same reason core/measure.py spells its four out.
    if state is Evaluable.EVALUABLE:
        t.note("claims:evaluable", state.value, because, (anchor,))
    elif state is Evaluable.UNDETECTABLE:
        t.note("claims:undetectable", state.value, because, (anchor,))
    elif state is Evaluable.NEVER_MEASURED:
        t.note("claims:never_measured", state.value, because, (anchor,))
    else:
        t.note("claims:no_frame", state.value, because, (anchor,))

    return t.done(Evaluability(state=state, frame=frame, blocked_by=blocked))
