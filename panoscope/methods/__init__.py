"""The method registry: one entry per way of producing evidence.

Data, not types, like ``SHAPES`` / ``RULES`` / ``FLAGS`` -- adding a method is an
import and one ``register`` call at the bottom of this module. Nothing in ``core/``
branches on which method produced a number, because everything downstream reads
``NormalisedEvidence`` and never a raw method table.

**The one thing this registry is for.** ``contracts.py`` makes ``does_not_measure``
and ``assumptions`` *present* -- a Protocol can do no more than that, since a missing
attribute is the only thing structural typing can see. :func:`register` makes them
**non-empty**, and that is the check the whole design rests on. A method that
declares no limits is one whose output reads as having none: a ``glm_coef`` with no
``does_not_measure`` beside it is indistinguishable from a fold change, and the
confidence assessment silently stops being an audit of someone else's output. The
check lives at the registration chokepoint rather than only in a test so that it also
bites an out-of-tree plugin, which no test in this repo will ever see.

**A method registers INTO a workflow.** :func:`register` resolves
``module.workflow`` against ``panoscope.workflows`` and refuses an id nobody
registered. An orphan method is worse than a missing one: it loads, produces
evidence, and has no question to answer with it, so the evidence surfaces under
whatever surface a caller happened to pick.

**Duplicate names raise.** ``DuplicateRegistration`` is reused from
``types/uispec.py``; see the workflow registry's docstring for why import order is
not an acceptable tie-break.

The registry holds *instances*, not classes. A declaration a later import could
rebind is a declaration nothing downstream can trust, which is why
``methods/jazzpanda.py`` builds its ``MODULE`` from a frozen dataclass.
"""

from __future__ import annotations

from panoscope import workflows
from panoscope.types.contracts import MethodModule
from panoscope.types.uispec import DuplicateRegistration

__all__ = ["METHODS", "UndeclaredLimits", "UnregisteredMethod",
           "DuplicateRegistration", "register", "get", "for_workflow", "names"]

#: The two fields a method may not leave empty. Named here rather than inline so the
#: test and the check read the same list -- CLAUDE.md's rule has one home.
MANDATORY_DECLARATIONS: tuple[str, ...] = ("does_not_measure", "assumptions")


class UndeclaredLimits(ValueError):
    """Raised for a method that declares no limits.

    Not a warning and not a lint. The limits are the reason a method module exists;
    without them the module is a number with an opinion attached, and every layer
    above it would be free to read that number as whatever it needed.
    """


class UnregisteredMethod(KeyError):
    """Raised for a method name nobody registered. Never defaulted to another one."""


#: The registry, keyed by ``MethodModule.name``.
METHODS: dict[str, MethodModule] = {}


def register(module: MethodModule) -> None:
    """Add one method. Three refusals, each naming the failure it prevents."""
    for attr in MANDATORY_DECLARATIONS:
        if not getattr(module, attr, None):
            raise UndeclaredLimits(
                f"{module.name!r}: {attr} is mandatory and must be non-empty. A "
                "method that declares no limits is one whose output reads as having "
                "none -- which is exactly the input that gets a spatial coefficient "
                "read as a fold change."
            )
    if module.workflow not in workflows.WORKFLOWS:
        raise workflows.UnregisteredWorkflow(
            f"{module.name!r} registers into workflow {module.workflow!r}, which is "
            f"not registered; known: {list(workflows.names())}. A method with no "
            "question to answer still produces evidence, and that evidence would "
            "surface under whichever surface a caller happened to pick."
        )
    if module.name in METHODS:
        raise DuplicateRegistration(
            f"method {module.name!r} is already registered "
            f"({type(METHODS[module.name]).__name__}); two registrations for one id "
            "are resolved by import order, which is decided by nothing"
        )
    METHODS[module.name] = module


def get(name: str) -> MethodModule:
    """The method for ``name``. Raises rather than guessing at a near match."""
    try:
        return METHODS[name]
    except KeyError:
        raise UnregisteredMethod(
            f"no method registered as {name!r}; known: {sorted(METHODS)}"
        ) from None


def for_workflow(workflow_id: str) -> tuple[MethodModule, ...]:
    """Every method answering one question, by name.

    The inverse lookup lives here and not on the workflow so the import graph stays
    one-way: a method knows its workflow, a workflow knows nothing about methods.
    Sorted, because this feeds listings and the interpretation record, and dict
    insertion order is import order.
    """
    return tuple(m for _n, m in sorted(METHODS.items())
                 if m.workflow == workflow_id)


def names() -> tuple[str, ...]:
    return tuple(sorted(METHODS))


# --------------------------------------------------------------------------- #
# The entries. One line per method; this is the "dict literal" in call form.
# A second method answering the same question is one more line here and a new
# file beside jazzpanda.py -- no type changes anywhere.
# --------------------------------------------------------------------------- #
from panoscope.methods import jazzpanda as _jazzpanda  # noqa: E402  (registration order)
from panoscope.methods import jazzpanda_cgst as _jazzpanda_cgst  # noqa: E402

register(_jazzpanda.MODULE)
register(_jazzpanda_cgst.MODULE)
