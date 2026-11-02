"""How a declared unknown travels: from intake, into every check that reads it.

CLAUDE.md states the rule twice and the codebase implements it nowhere:

> ``unknown`` is first-class and propagates.
> A check reading an unknown field returns **cannot be settled** and names what is
> missing.

``types/conditions.py`` holds up its end. ``Condition`` makes "a value exists iff it
is known" structural, ``Conditions.declared_unknown`` names the unknown fields in a
stable order, and its docstring says exactly what is supposed to happen next: *"A
check that reads one of these must return cannot be settled and NAME what is
missing."* Then nothing reads it. ``grep`` finds ``declared_unknown`` in three test
files and in one docstring, and in no check anywhere -- so the propagation is a rule
the type is shaped for and no caller performs.

``workflows/de.py`` holds up the other end, by hand, four times over. Its
``CheckState`` is three-valued, its ``CheckResult.missing`` is actionable prose, and
its four checks each hand-roll the same two steps: notice an input is absent, refuse,
name what would settle it. Not one of them refuses on a ``Condition`` -- they refuse
on a frame, a replicate count, an abundance table, a second method's result. The
condition fields the study genuinely does not know (segmentation and fixation on the
demo study; modality and method too, on the kidney one) propagate into nothing.

This module is that missing edge, and it is one function.

**Why the return type is not ``CheckResult``.** It would be the obvious shape and it
is unavailable: ``CheckState`` and ``CheckResult`` live in ``workflows/de.py``, and
``core/`` cannot import ``workflows/`` -- the dependency runs the other way, and
inverting it would make a broken workflow able to break the core. :class:`Unsettled`
is therefore the primitive, carrying exactly the two tuples a ``CheckResult`` needs
spliced into it, and nothing else. **That the three-valued check state lives in a
workflow rather than in ``types/`` is a real defect** -- it is the vocabulary every
workflow's checks need, and the second workflow that wants it will either import from
``de.py`` or write a third copy -- but ``types/`` is frozen, so it is reported rather
than moved.

**``missing`` is sentences, not field names.** It is built to drop straight into
``CheckResult.missing``, whose contract is that each entry names a thing that would
settle the check. "segmentation" is a field name a biologist cannot act on; "the
segmentation version this result was produced under, which was declared unknown at
intake" is something they can go and get. The bare names are not duplicated here:
``Conditions.declared_unknown`` already returns them, and a second copy with a
different order is how two callers come to disagree about what was missing.

**A blocking field can never appear in ``missing``, and that is not a special case.**
``Conditions.__post_init__`` refuses to construct an object whose ``assay`` or
``tissue`` is unknown, so a check that declares it reads ``tissue`` is settled by
construction. That is the §6.3 line doing its job one layer down: the blocking fields
block the study, the rest propagate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, Conditions,
)
from panoscope.types.trace import Trace, Traced

#: Every ``Condition`` field, read off the §6.3 table rather than re-listed. A
#: seventh field has to land in one of those two tuples to exist at all, so it
#: appears here the moment it is added -- there is no second list to forget.
CONDITION_FIELDS: Final[tuple[str, ...]] = BLOCKS_THE_STUDY + DECLARED_UNKNOWN_OK


class UnknownConditionField(ValueError):
    """Raised when a check declares it reads a field ``Conditions`` does not have.

    A programming error, not a study problem, and it must not be silent: a typo in a
    declared dependency would otherwise make the check read *fewer* conditions than
    it claims and settle on an unknown it never looked at -- the exact failure this
    module exists to prevent, arriving through the mechanism meant to prevent it.
    """


@dataclass(frozen=True, slots=True)
class Unsettled:
    """What a check declared it reads, and which of those the study does not know.

    Two tuples rather than a boolean, for the reason ``CheckState`` has three states
    and ``MeasuredState`` has four: "we read every condition this needs" and "we
    could not read one" are opposite readings that a flag renders identically, and
    only the second is actionable. ``reading`` is kept beside ``missing`` so a
    settled result still says what it checked -- a check that names its inputs only
    when they are absent cannot be audited when it passes.
    """

    reading: tuple[str, ...]   # the condition fields the check declared it reads
    missing: tuple[str, ...]   # actionable sentences, one per unknown field

    @property
    def settled(self) -> bool:
        """True when every declared condition is known. Never a judgement about the
        study -- only about whether the question could be asked of it."""
        return not self.missing

    def __str__(self) -> str:
        if self.settled:
            return f"every condition read is known: {', '.join(self.reading) or '-'}"
        return (f"cannot be settled: {len(self.missing)} of "
                f"{len(self.reading)} condition(s) declared unknown")


def requires(conditions: Conditions, *fields: str) -> Traced[Unsettled]:
    """The conditions a check depends on, and the unknowns that stop it settling.

    ``fields`` is the check's own declaration of what it reads. Declaring it is the
    point: a check that reaches into ``Conditions`` directly can read an unknown and
    carry on, and nothing downstream can tell that it did -- the answer it returns is
    shaped exactly like one it reached honestly.

    Order is ``Conditions``' declaration order, never the caller's. ``missing`` is
    hashed into a record through whatever ``CheckResult`` it is spliced into, and
    two checks reading the same two fields in different orders must not produce two
    different records of the same gap.

    ``UNKNOWN`` is not reported as a caveat on the answer. It *is* the answer: the
    check could not be asked. That distinction is why this returns a tuple of things
    to go and get rather than a confidence adjustment -- CLAUDE.md's rule is that the
    band is a function of one method's output, and a missing condition is not that
    method's output being weaker.
    """
    t = Trace()
    bad = tuple(f for f in fields if f not in CONDITION_FIELDS)
    if bad:
        raise UnknownConditionField(
            f"{list(bad)} is not a Conditions field; known: "
            f"{list(CONDITION_FIELDS)}. A declared dependency that does not exist "
            "makes the check read fewer conditions than it claims"
        )

    declared = tuple(f for f in CONDITION_FIELDS if f in fields)
    unknown = tuple(f for f in conditions.declared_unknown if f in fields)
    missing = tuple(
        f"{f}: declared unknown at intake, so this check reads nothing for it -- "
        "state it, or record why it cannot be known"
        for f in unknown
    )
    out = Unsettled(reading=declared, missing=missing)

    if unknown:
        t.note("propagate:cannot_be_settled", ", ".join(unknown),
               (f"{', '.join(unknown)} unknown of "
                f"{', '.join(declared) or 'no fields'} read",) + missing)
    else:
        t.note("propagate:conditions_known", ", ".join(declared) or "none",
               (f"every condition read is known: "
                f"{', '.join(declared) or 'the check declared none'}",))
    return t.done(out)
