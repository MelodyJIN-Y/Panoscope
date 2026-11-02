"""The workflow registry: one entry per interpretive question.

Data, not types -- the same pattern as ``SHAPES`` in ``types/subject.py``, ``RULES``
in ``types/trace.py`` and ``FLAGS`` in ``types/evidence.py``. Adding a question is a
dict entry and an import; no type changes, no branch anywhere in ``core/``, and the
contract tests parameterise over whatever is in :data:`WORKFLOWS`, so adding one
proves conformance rather than merely asserting it.

**Why a ``register`` call and not a dict literal.** ``SHAPES`` can be a literal
because a ``Shape`` is inert. A ``Workflow`` is a live object that must satisfy a
Protocol, must own a renderable ``UISpec``, and must not collide with another
workflow's id -- and a literal has nowhere to put those checks. So the literal
becomes one call per entry at the bottom of this module, which reads the same and
fails at import rather than at render time.

**Duplicate ids raise.** ``DuplicateRegistration`` is reused from ``types/uispec.py``
rather than redefined: one exception for one failure, so a caller can catch
registration collisions across both registries with one name. Last-write-wins between
two import-time registrations is decided by import order, which is decided by
nothing, and the loser fails far from the cause.

**This module must not import ``panoscope.methods``.** The dependency runs one way --
methods validate against workflows -- so that a broken method cannot make the
workflow registry unimportable, and so the import graph has no cycle for
``test_every_first_party_module_imports_cleanly`` to trip over. The inverse lookup
("which methods answer this question?") therefore lives in
``panoscope.methods.for_workflow``.
"""

from __future__ import annotations

from panoscope.types.contracts import Workflow
from panoscope.types.uispec import DuplicateRegistration

__all__ = ["WORKFLOWS", "UnregisteredWorkflow", "DuplicateRegistration",
           "register", "get", "names"]


class UnregisteredWorkflow(KeyError):
    """Raised for a workflow id nobody registered. Never defaulted to another one.

    Falling back to "the first registered workflow" would answer a question the
    biologist did not ask, with a band computed under a rubric that does not apply --
    and it would look like a result rather than like a missing registration.
    """


#: The registry. One entry per interpretive question, keyed by ``Workflow.name``.
WORKFLOWS: dict[str, Workflow] = {}


def register(workflow: Workflow) -> None:
    """Add one question to the registry. Raises on a collision; never overwrites."""
    if workflow.name in WORKFLOWS:
        raise DuplicateRegistration(
            f"workflow {workflow.name!r} is already registered "
            f"({type(WORKFLOWS[workflow.name]).__name__}); two registrations for one "
            "id are resolved by import order, which is decided by nothing"
        )
    WORKFLOWS[workflow.name] = workflow


def get(name: str) -> Workflow:
    """The workflow for ``name``. Raises rather than guessing at a near match."""
    try:
        return WORKFLOWS[name]
    except KeyError:
        raise UnregisteredWorkflow(
            f"no workflow registered as {name!r}; known: {sorted(WORKFLOWS)}"
        ) from None


def names() -> tuple[str, ...]:
    """Registered ids, sorted. Sorted because this feeds listings and records, and
    dict insertion order is import order, which is not a fact about the study."""
    return tuple(sorted(WORKFLOWS))


# --------------------------------------------------------------------------- #
# The entries. One line per question; this is the "dict literal" in call form.
# --------------------------------------------------------------------------- #
from panoscope.workflows import marker as _marker  # noqa: E402  (registration order)
from panoscope.workflows import enrichment as _enrichment  # noqa: E402

register(_marker.WORKFLOW)
register(_enrichment.WORKFLOW)
