"""The one entry point into interpretation -- and the one guard the workflows cannot
write for themselves.

``IMPLEMENTATION.md`` §6 gives this module one signature,
``interpret(ev, workflow, reference) -> Traced[Assignment | Finding]``, and describes
it as "dispatch on ``workflow.task_kind``". **All three workflows already implement
``interpret``**, each composing ``core/`` the way its question needs -- marker calls
``core.verdict.assess``, enrichment tiers per item, DE runs its four checks and
delegates the join to ``core.reconcile``. So this module is deliberately thin: a
fourth copy of that composition would be a second place the band is computed, which
is the one thing the whole package is arranged to prevent.

What is left over, once the composition is where it belongs, is a single invariant
that no workflow is in a position to enforce about itself:

**The output SHAPE must match the declared task kind.**

Three gates already exist and all three leave the same hole. ``Finding.__post_init__``
raises on ``task_kind=ASSIGNMENT``; ``UISpec.__post_init__`` refuses a
``LabelBandHeader`` on a non-assignment surface; ``Assignment`` simply has a ``band``
field and ``Finding`` has none. Every one of them is a check a workflow makes against
*the object it just built*. None of them compares that object against **what the
workflow said it was**. A CHARACTERISATION workflow that returned an ``Assignment``
would pass all three -- the ``Assignment`` is well-formed, its band is well-formed,
and nothing it touches knows there was no label space -- and it would put a
unit-level confidence on a question for which one is undefined, which CLAUDE.md names
as a thing never to do. The mirror case is quieter and no better: an ASSIGNMENT
workflow returning a ``Finding(task_kind=CHARACTERISATION)`` is constructible today,
and it reads as interpreted while the one number the biologist acts on has been
dropped on the floor rather than computed.

Both are caught here, because here is the only place that holds the declaration and
the object at the same time. That is the whole module, and it is why it raises rather
than recording: a shape that contradicts its own declaration is not a finding about
the study, it is a registry that is wrong, and a decision record describing it would
be an audit trail for a bug.

**No decision of its own.** ``interpret`` absorbs the workflow's trace and returns it
unchanged. Noting "dispatched to marker-interpretation as ASSIGNMENT" would add a
line to the record that restates ``InterpretationRecord.task_kind``, which the record
already carries -- the duplicate nobody updates. ``core/record.py`` returns ``Traced``
with empty decisions for the same reason, and for the same reason it still returns
``Traced``: a caller unwrapping one value in this package should not have to remember
which functions are the exception.

**``reference`` stays ``Any`` and is passed straight through.** Each workflow declares
its own reference carrier (``MarkerReference``, ``EnrichmentReference``, ``DEAudit``)
and nothing here may look inside one: reference data selects and orders evidence, and
a dispatcher that read a field off it would be a place a reference value could enter
an output without passing the workflow that knows what it is allowed to do.
"""

from __future__ import annotations

from typing import Any

from panoscope.types.contracts import Workflow
from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.outputs import Assignment, Finding, TaskKind
from panoscope.types.trace import Trace, Traced


class ShapeContradictsTaskKind(TypeError):
    """Raised when a workflow's output shape disagrees with its declared task kind.

    A ``TypeError`` rather than a ``ValueError``: nothing about the study is wrong
    here, and no input the biologist could supply causes it. A registry entry is
    claiming to answer one kind of question and answering another, which is a
    programming error that must stop before the object reaches a surface built for
    the other shape.
    """


def interpret(ev: NormalisedEvidence, workflow: Workflow,
              reference: Any) -> Traced[Assignment | Finding]:
    """One unit's evidence, through the workflow that owns the question.

    ``ASSIGNMENT`` yields an :class:`~panoscope.types.outputs.Assignment`, which has a
    label space and therefore a band. ``CHARACTERISATION`` and ``COMPARISON`` yield a
    :class:`~panoscope.types.outputs.Finding`, which has no field a band could go in.
    The union in the return type is the type-level form of that rule and this function
    is where it is actually checked, because a workflow can only ever check the object
    against itself.

    The ``Finding`` branch checks the task kind twice over -- the shape, then the
    value of ``Finding.task_kind`` -- and the second is not redundant. ``Finding``
    refuses only ``ASSIGNMENT``, so a COMPARISON workflow emitting
    ``Finding(task_kind=CHARACTERISATION)`` is well-formed and silently routes a DE
    audit to the enrichment surface, where the per-item counts render and every
    confound check it computed is simply not shown.
    """
    t = Trace()
    out = t.absorb(workflow.interpret(ev, reference))

    if workflow.task_kind is TaskKind.ASSIGNMENT:
        if not isinstance(out, Assignment):
            raise ShapeContradictsTaskKind(
                f"{workflow.name} declares {TaskKind.ASSIGNMENT.value} but returned "
                f"{type(out).__name__} for {ev.unit}: an assignment has a label "
                "space and therefore a band, and a Finding has no field to put one "
                "in -- the band would be dropped on the floor rather than computed"
            )
        return t.done(out)

    if not isinstance(out, Finding):
        raise ShapeContradictsTaskKind(
            f"{workflow.name} declares {workflow.task_kind.value} but returned "
            f"{type(out).__name__} for {ev.unit}: there is no label space, so a "
            "unit-level band is undefined, not merely unhelpful"
        )
    if out.task_kind is not workflow.task_kind:
        raise ShapeContradictsTaskKind(
            f"{workflow.name} declares {workflow.task_kind.value} but its Finding "
            f"for {ev.unit} is stamped {out.task_kind.value}; the stamp is what the "
            "record and the review surface route on, so the unit would be read "
            "under a question nobody asked of it"
        )
    return t.done(out)
