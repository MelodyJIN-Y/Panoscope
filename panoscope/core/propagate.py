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

``workflows/de.py`` holds up the other end, by hand, four times over. Its checks are
three-valued, their ``missing`` is actionable prose, and each one hand-rolls the same
two steps: notice an input is absent, refuse, name what would settle it. Not one of
them refuses on a ``Condition`` -- they refuse on a frame, a replicate count, an
abundance table, a second method's result. The condition fields the study genuinely
does not know (segmentation and fixation on the demo study; modality and method too,
on the kidney one) propagate into nothing.

This module is that missing edge, and it is one function.

**The return type is a ``CheckResult``**, the same object a workflow's own checks
return, so a check that cannot be asked composes into the checklist rather than
needing a second shape spliced into it::

    r = propagate.requires(conditions, COMPOSITION, "segmentation")
    if not r.value.settled:
        return r

``CheckState`` and ``CheckResult`` live in ``types/contracts.py``, beside the
``Check`` they answer. They were in ``workflows/de.py`` and this module carried its
own two-tuple primitive instead, because ``core/`` cannot import ``workflows/`` --
the dependency runs the other way, and inverting it would let a broken workflow break
the core. Moving the vocabulary to ``types/`` was the reviewed contract change that
removed the primitive; a type nobody emits is dead vocabulary.

**``OK`` here is not a verdict on the result.** It says only that every condition the
check declared is known -- that the question could be asked. The caller keeps going
and overwrites it with its own answer. Only ``CANNOT_BE_SETTLED`` is returned as-is,
because that *is* the answer: there is nothing left to ask.

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

**Why it still has no caller, and which caller it is waiting for.** The one layer
that emits a ``CheckResult`` is ``workflows/de.py``, whose four checks hand-roll the
two steps above; ``de-interpretation`` has no registered method
(``methods.for_workflow`` returns nothing for it), so those checks never run outside
their tests and there is nowhere for this edge to attach. The marker workflow is not
the alternative: its four declared checks read the measurement, the panel frame and a
curated marker list, and not one ``Condition`` field. Giving one of them a condition
dependency so that this module would have a caller would be inventing a dependency
the check does not have -- and on the demo, where ``segmentation`` and ``fixation``
are declared unknown, it would turn a settled check into CANNOT BE SETTLED on a
fabricated ground. ``tests/test_core_unwired.py`` pins the registry emptiness that
makes this the right answer, so it fails the day a DE method registers.
"""

from __future__ import annotations

from typing import Final

from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, Conditions,
)
from panoscope.types.contracts import CheckResult, CheckState
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


def requires(conditions: Conditions, check_id: str,
             *fields: str) -> Traced[CheckResult]:
    """The conditions a check depends on, and the unknowns that stop it settling.

    ``fields`` is the check's own declaration of what it reads. Declaring it is the
    point: a check that reaches into ``Conditions`` directly can read an unknown and
    carry on, and nothing downstream can tell that it did -- the answer it returns is
    shaped exactly like one it reached honestly.

    ``check_id`` is the registered id of the check this answers for, not prose. It is
    required rather than defaulted because a ``CheckResult`` with no ``check_id``
    cannot be put on the review surface next to the checks that did fire, which is
    the one thing an unsettled check exists to be.

    Order is ``Conditions``' declaration order, never the caller's. ``missing`` is
    hashed into a record through the ``CheckResult`` it travels in, and two checks
    reading the same two fields in different orders must not produce two different
    records of the same gap.

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
    read = ", ".join(declared) or "the check declared none"

    if unknown:
        out = CheckResult(
            check_id, CheckState.CANNOT_BE_SETTLED,
            headline=(f"CANNOT BE SETTLED -- {', '.join(unknown)} unknown of "
                      f"{read} read"),
            because=(f"conditions read: {read}",),
            missing=missing,
        )
        t.note("propagate:cannot_be_settled", ", ".join(unknown),
               (out.headline,) + missing)
    else:
        out = CheckResult(
            check_id, CheckState.OK,
            headline=f"every condition read is known: {read}",
            because=(f"conditions read: {read}",),
        )
        t.note("propagate:conditions_known", ", ".join(declared) or "none",
               (out.headline,))
    return t.done(out)
