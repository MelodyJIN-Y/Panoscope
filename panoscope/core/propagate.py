"""How a declared unknown travels from intake into every check that reads it.

CLAUDE.md: ``unknown`` is first-class and propagates; a check reading an unknown field
returns **cannot be settled** and names what is missing. :func:`requires` is that edge::

    r = propagate.requires(conditions, COMPOSITION, "segmentation")
    if not r.value.settled:
        return r

- It returns a ``CheckResult`` (``types/contracts.py``), so an unaskable check composes
  into the checklist like any other.
- ``OK`` means only that every declared condition is known; the caller overwrites it with
  its own answer. ``CANNOT_BE_SETTLED`` is returned as-is.
- ``missing`` is actionable sentences, not field names; the bare names come from
  ``Conditions.declared_unknown``.
- A blocking field (§6.3) can never appear in ``missing``: ``Conditions`` refuses to
  construct with one unknown.

The only caller is ``workflows/de.composition_vs_expression`` (segmentation: abundance is
a share of cells, and cells are what segmentation produces). ``tests/test_core_unwired.py``
pins the caller set, so a new dependency has to be argued for: a check declaring a
condition it does not read silences real results.
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

    ``fields`` is the check's declaration of what it reads; ``check_id`` is its registered
    id (required, so the result can sit on the review surface). ``missing`` follows
    ``Conditions``' declaration order, never the caller's, since it is hashed into records.
    An unknown is the answer (cannot be settled), never a confidence adjustment.
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
