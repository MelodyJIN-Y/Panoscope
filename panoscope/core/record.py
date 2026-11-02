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


class UnattributableTrace(ValueError):
    """Raised when a trace cannot be cut into the segment belonging to this unit.

    A programming error at the call site, not a fact about the study. Records are
    built one unit at a time, so a trace that holds no decision of this unit's -- or
    that opens with decisions belonging to no unit at all -- has been joined to the
    wrong record. Both alternatives are silent: keeping nothing writes an empty
    source trace, which reads as "nothing was decided here", and keeping everything
    is the leak this segmentation exists to close.
    """


def unit_decisions(evidence: NormalisedEvidence,
                   *traced: Traced[Any]) -> Traced[tuple[Decision, ...]]:
    """The decisions that read THIS unit's subjects, in call order.

    A ``MethodModule.load`` traces one read of a whole file: nine clusters' measure
    decisions arrive in one tuple, and a record for one of them must not carry the
    other eight's.

    **The anchor join alone cannot do that, and used alone it leaked.** An item
    subject is keyed on the entity, not on the unit it was read under -- ``ERBB2``
    is ``gene|g=gene:ERBB2`` under every cluster -- so narrowing a nine-unit trace by
    ``Subject.key`` membership kept all nine clusters' ``measure:*`` decisions in all
    nine records. Measured on the demo: 2,522 decisions on each assignment record,
    off a 2,529-decision trace in which this unit's own share is 281.

    So each trace is first cut into **unit segments**. A decision anchored on a
    subject of the unit's own shape (``unit|…``) opens that unit's segment and every
    decision after it belongs to that unit until the next such anchor.
    ``core.measure.measure_unit`` is what makes the opener reliable: it resolves the
    unit's own capability verdict *before* any item's, and says why -- "this unit has
    four cells" explains every untestable item under it and must precede them.

    A trace carrying no unit anchor at all is wholly this unit's. That is the
    per-unit case -- a ``verdict``, a single ``measure`` -- and it is left alone.

    Inside this unit's segment the anchor join still runs, because a segment can
    carry decisions that read something other than the unit's own subjects.

    A decision carrying NO refs is kept **within the segment**.
    ``band:no_canonical_driver`` has nothing to anchor on -- its content is that
    nothing qualified -- and dropping it would delete the one decision that explains
    a Low band. Segmenting first is what makes keeping it safe; before, a ref-less
    decision from any unit came along to every record.

    Two shapes raise rather than guess, because the silent readings of both are the
    failure this function exists to prevent: see :class:`UnattributableTrace`.

    The value and the decisions are the SAME tuple. This function's product is a
    decision list, and returning it in the slot :func:`build` splices from is what
    lets a caller write ``build(ev, reading, unit_decisions(ev, loaded), verdict)``
    and get the narrowed decisions rather than silently none of them.
    """
    mine = evidence.unit_subject.key
    unit_prefix = evidence.unit_subject.shape + "|"
    anchors = {mine} | {i.subject.key for i in evidence.items}

    kept: list[Decision] = []
    for t in traced:
        heads = [tuple(r.split("#", 1)[0] for r in d.refs) for d in t.decisions]
        opens = [next((h for h in hs if h.startswith(unit_prefix)), None)
                 for hs in heads]
        seen = [o for o in opens if o is not None]
        if seen and mine not in seen:
            raise UnattributableTrace(
                f"a trace covering {sorted(set(seen))} was passed to the record "
                f"assembler for {mine}: it carries no decision of this unit's, so "
                "every decision in it belongs to some other unit's record"
            )
        if seen and opens.index(seen[0]) != 0:
            raise UnattributableTrace(
                f"{opens.index(seen[0])} decision(s) precede the first unit anchor "
                f"in a trace covering {sorted(set(seen))}: they belong to no unit, "
                "and core.measure.measure_unit resolves the unit before any item "
                "precisely so that this cannot happen"
            )
        # No unit anchor anywhere: a per-unit trace, wholly this unit's.
        inside = not seen
        for d, hs, opener in zip(t.decisions, heads, opens):
            if opener is not None:
                inside = opener == mine
            if inside and (not d.refs or any(h in anchors for h in hs)):
                kept.append(d)
    out = tuple(kept)
    return Traced(value=out, decisions=out)


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
