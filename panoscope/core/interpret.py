"""The one entry point into interpretation, and the one guard workflows cannot write.

IMPLEMENTATION.md §6: ``interpret(ev, workflow, reference) -> Traced[Assignment |
Finding]``. Each workflow composes ``core/`` in its own ``interpret``, so this module
only dispatches and enforces one invariant:

**The output SHAPE must match the declared task kind.** An ASSIGNMENT workflow must
return an ``Assignment``; any other task kind a ``Finding`` of that kind. Only here are
the declaration and the object held together, so a mismatch raises: it is a wrong
registry, not a finding about the study.

``interpret`` adds no decision of its own; the workflow's trace passes through unchanged.
``reference`` stays ``Any`` and is passed straight through, unread.
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

    ``ASSIGNMENT`` yields an :class:`~panoscope.types.outputs.Assignment` (a band);
    ``CHARACTERISATION`` and ``COMPARISON`` yield a
    :class:`~panoscope.types.outputs.Finding` (no band field). For a ``Finding`` the
    ``task_kind`` stamp is checked too, since ``Finding`` itself refuses only
    ``ASSIGNMENT`` and the stamp routes the review surface.
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
