"""The method registry: one entry per way of producing evidence.

Adding a method is an import and one ``register`` call at the bottom of this module.
Nothing in ``core/`` branches on which method produced a number; everything
downstream reads ``NormalisedEvidence``.

* :func:`register` makes ``does_not_measure`` and ``assumptions`` **non-empty** (the
  Protocol can only make them present), at the chokepoint so it also bites an
  out-of-tree plugin. A method that declares no limits reads as having none.
* **A method registers INTO a workflow**: ``module.workflow`` must name a registered
  workflow; an orphan method is refused.
* **Duplicate names raise** (``DuplicateRegistration``, from ``types/uispec.py``).

The registry holds frozen *instances*, not classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from panoscope import workflows
from panoscope.types.contracts import MethodModule, UnrecognisedResult
from panoscope.types.uispec import DuplicateRegistration

__all__ = ["METHODS", "UndeclaredLimits", "UnregisteredMethod",
           "DuplicateRegistration", "register", "get", "for_workflow",
           "recognise", "recognise_inputs"]

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


def recognise(table: Any) -> MethodModule | None:
    """The one registered method whose declared schema this table matches; ``None``
    when none does. Never guessed: two matches raise, since a tie broken by import
    order would make the record's method stamp arbitrary."""
    hits = []
    for m in METHODS.values():
        try:
            m.detect_unit(table)
        except UnrecognisedResult:
            continue
        hits.append(m)
    if len(hits) > 1:
        raise ValueError(
            f"{len(hits)} methods accept this table "
            f"({', '.join(sorted(m.name for m in hits))}); narrow one method's schema")
    return hits[0] if hits else None


#: Result-like files a collaborator may hand over. Only CSVs are read by a method; the
#: rest are listed as ``None`` so the biologist sees them, never silently skipped.
_RESULT_SUFFIXES = (".csv", ".tsv", ".txt", ".xlsx")


def recognise_inputs(folder: Path) -> dict[Path, MethodModule | None]:
    """Every top-level result file in ``folder`` -> the method that reads it, or ``None``."""
    import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

    return {p: recognise(pd.read_csv(p)) if p.suffix == ".csv" else None
            for p in sorted(Path(folder).iterdir())
            # A dotfile (macOS ``._x.csv`` from a USB drive) is metadata, not a result.
            if p.is_file() and not p.name.startswith(".")
            and p.suffix.lower() in _RESULT_SUFFIXES}


# --------------------------------------------------------------------------- #
# The entries. One line per method; this is the "dict literal" in call form.
# A second method answering the same question is one more line here and a new
# file beside jazzpanda.py -- no type changes anywhere.
# --------------------------------------------------------------------------- #
from panoscope.methods import jazzpanda as _jazzpanda  # noqa: E402  (registration order)
from panoscope.methods import jazzpanda_cgst as _jazzpanda_cgst  # noqa: E402
from panoscope.methods import edger as _edger  # noqa: E402
from panoscope.methods import seurat_de as _seurat_de  # noqa: E402

register(_jazzpanda.MODULE)
register(_jazzpanda_cgst.MODULE)
# TWO methods answering one question, which is what makes §4's extensibility claim
# checkable rather than asserted: nothing in core/, in workflows/de.py or in the
# composition root knows either of these exists, and the root picks between them by
# asking each whether it can read the table
# (:func:`recognise`). They differ where the METHODS differ --
# edgeR's unit of replication is the sample, Seurat's is the cell -- and that
# difference lives in their declared assumptions, not in a branch anywhere.
register(_edger.MODULE)
register(_seurat_de.MODULE)
