"""The interpretation record: assembled from the pieces, serialised, never derived.

``types/record.py`` declares the shape; this module is the only place that *fills*
it. Everything it puts in a record came up through ``core/`` from a method's table --
it reads no file, computes no statistic and writes no English. A record this module
produced can therefore be rebuilt from the same hashed inputs and come out
byte-identical, which is the whole reproducibility claim (``IMPLEMENTATION.md`` §10).

**Nothing here invents a decision.** ``build`` concatenates the ``Traced`` values its
caller already got back from ``core.measure`` and ``core.verdict``, in call order.
There is no ``Trace`` parameter and no ``note`` call in this module: a record
assembler that could record would be narrating rather than transcribing, and the
record would stop being an audit of the analysis and start being a second opinion
about it.

**Decisions are NOT filtered on the way in.** A ``ref`` can anchor on something that
is not one of the unit's subjects -- ``claim:…``, a reference edge -- and a build
step that quietly kept only the subject-anchored ones would delete exactly the
reasoning the biologist is least able to reconstruct. :func:`unit_decisions` does
that narrowing when a caller actually needs it (a method loads a whole file in one
trace; a record is one unit), and it is opt-in for that reason.

**``prose`` is separable, by name read off the type.** :func:`to_dict` takes
``generated=False`` and drops every field
:data:`~panoscope.types.record.GENERATED_FIELDS` names -- not the string ``"prose"``,
which is the version that silently stops covering the second generated field the day
one is added. That projection is the hashable half of the record: a model writing
prose into a record must not be able to move its digest.

**``write`` is the one IO function in ``core/``, and it exists to own one rule.** §10:
"Never mutate the committed per-unit JSONs." Records go to
``interp/records/<task_kind>/<unit>.json`` and nowhere else; keeping the path
construction and the only write behind one function is what makes that checkable
instead of conventional. The v1 tree under ``interp/clusters/`` is not a path this
module can produce.

Returning ``Traced`` from the serialisation helpers is ``core/``'s uniform contract
(``tests/test_contracts.py``), kept here for the reason ``core/compat.py`` keeps it:
a caller that unwraps one value should not have to remember which functions in the
package are the exception. Their decision tuples are empty, which is the honest
report -- transcribing a record is not a decision about it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Final

from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.outputs import TaskKind
from panoscope.types.record import GENERATED_FIELDS, InterpretationRecord, ReadingRecord
from panoscope.types.trace import Decision, Traced

#: Declared, never derived. A version read off a package version or a file mtime
#: would move the golden record on an install that changed nothing.
SPEC_VERSION: Final[str] = "0.1"

#: ``interp/<RECORDS_DIR>/<task_kind>/<unit>.json`` (§10). The v1 per-unit files live
#: under ``interp/clusters/`` and are a *generated view* (``core/compat.py``); this
#: directory is the record itself and the two never collide.
RECORDS_DIR: Final[str] = "records"


def unit_decisions(evidence: NormalisedEvidence,
                   *traced: Traced[Any]) -> Traced[tuple[Decision, ...]]:
    """The decisions that read THIS unit's subjects, in call order.

    A ``MethodModule.load`` traces one read of a whole file: nine clusters' measure
    decisions arrive in one tuple, and a record for one of them must not carry the
    other eight's. The narrowing is by ``ref`` anchor against the unit's own
    ``Subject.key``\\s -- the same join key the interface clicks back on -- rather
    than by position, because position is a property of how a loader happened to
    loop.

    A decision carrying NO refs is kept. That is deliberate and it is the limit of
    this function: ``band:no_canonical_driver`` has nothing to anchor on (its content
    is that nothing qualified), and dropping it would delete the one decision that
    explains a Low band. Pass a single unit's traces and the two rules agree; pass a
    multi-unit trace and a ref-less decision from another unit would come along, so
    narrow the method trace and leave the per-unit ones alone.

    The value and the decisions are the SAME tuple. This function's product is a
    decision list, and returning it in the slot :func:`build` splices from is what
    lets a caller write ``build(ev, reading, unit_decisions(ev, loaded), verdict)``
    and get the narrowed decisions rather than silently none of them.
    """
    anchors = {evidence.unit_subject.key} | {i.subject.key for i in evidence.items}
    kept = tuple(
        d for t in traced for d in t.decisions
        if not d.refs or any(r.split("#", 1)[0] in anchors for r in d.refs)
    )
    return Traced(value=kept, decisions=kept)


def build(evidence: NormalisedEvidence, reading: ReadingRecord,
          *traced: Traced[Any], task_kind: TaskKind,
          claims_used: tuple[tuple[str, str], ...] = (),
          undecidable: tuple[str, ...] = (),
          prose: str | None = None) -> Traced[InterpretationRecord]:
    """One unit's record, from the pieces that already exist.

    ``*traced`` is the measure and verdict results the caller already holds, spliced
    in call order: taking the ``Traced`` values rather than bare decision tuples is
    what stops a caller assembling a record whose reasoning came from somewhere other
    than the analysis that produced it.

    ``spec_version`` is filled from :data:`SPEC_VERSION` and is not a parameter --
    one declared constant, so two writers cannot stamp two versions on the same
    shape.

    ``claims_used`` carries ``(claim_id, verdict)`` pairs including the refusals:
    a claim considered and withheld is part of the reasoning, and a record that
    lists only the applied ones reads as though the evidence all pointed one way.
    """
    return Traced(InterpretationRecord(
        unit=evidence.unit,
        task_kind=task_kind,
        reading=reading,
        decisions=tuple(d for t in traced for d in t.decisions),
        claims_used=tuple(claims_used),
        undecidable=tuple(undecidable),
        spec_version=SPEC_VERSION,
        prose=prose,
    ))


def to_dict(record: InterpretationRecord,
            generated: bool = True) -> Traced[dict[str, Any]]:
    """The record as plain JSON-able data, in field-declaration order.

    ``generated=False`` drops every field ``GENERATED_FIELDS`` names, which is the
    gate ``grounding_check`` reads off the type. It is also the hashable projection:
    the deterministic half of a record must have the same bytes whether or not a
    model was reachable when it was built.
    """
    out = dataclasses.asdict(record)
    if not generated:
        for field in GENERATED_FIELDS:
            out.pop(field, None)
    return Traced(out)


def path_for(interp: str | Path, record: InterpretationRecord) -> Traced[Path]:
    """``<interp>/records/<task_kind>/<unit>.json``. Construction only, no IO."""
    return Traced(Path(interp) / RECORDS_DIR / record.task_kind.value
                  / f"{record.unit}.json")


def write(record: InterpretationRecord, interp: str | Path) -> Traced[Path]:
    """Serialise one record under the records path. The v1 tree is never touched.

    Deterministic bytes: field order is the dataclass's, ``ensure_ascii`` stays on so
    a non-ASCII identifier cannot change the encoding, and nothing here reads a
    clock. ``prose`` is written when present -- the gate is on *reading* it
    (:func:`to_dict`), not on persisting it, because a record whose generated layer
    was dropped on the way to disk could not be re-gated afterwards.
    """
    p = path_for(interp, record).value
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(record).value, indent=2) + "\n",
                 encoding="utf-8")
    return Traced(p)
