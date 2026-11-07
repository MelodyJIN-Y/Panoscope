"""The workflow registry: one entry per interpretive question.

Data, not types (like ``SHAPES`` / ``RULES`` / ``FLAGS``): adding a question is one
``register`` call and an import, and the contract tests parameterise over
:data:`WORKFLOWS`. ``register`` rather than a dict literal so the Protocol, the
``UISpec`` and id uniqueness are checked at import.

**Duplicate ids raise** ``DuplicateRegistration`` (shared with ``types/uispec.py``);
import order is no tie-break.

**This module must not import ``panoscope.methods``**: methods validate against
workflows, one way, so the inverse lookup lives in ``panoscope.methods.for_workflow``.
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
from panoscope.workflows import de as _de  # noqa: E402

register(_marker.WORKFLOW)
register(_enrichment.WORKFLOW)
# The third question. Registered now that there is a DE artifact to answer it
# against (``data/studies/synthetic_multisample_fixture/inputs/de_edger.csv``, a
# DECLARED synthetic fixture) and a composition root to run it
# (``pipeline/stages/de.py``). Until both were true this line was deliberately
# absent, because a workflow registered against nothing shows up in every listing
# as a question the study can be asked.
register(_de.WORKFLOW)
