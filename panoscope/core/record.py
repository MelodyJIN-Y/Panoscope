"""The interpretation record: assembled from the pieces, serialised, never derived.

``types/record.py`` declares the shape; this module is the only place that fills it. It
reads no file, computes no statistic and writes no English, so a record rebuilt from the
same hashed inputs is byte-identical (IMPLEMENTATION.md §10).

- ``build`` concatenates the caller's ``Traced`` decisions in call order and records none
  of its own. Decisions are not filtered on the way in; :func:`unit_decisions` narrows
  them only when a caller opts in.
- :func:`to_dict` with ``generated=False`` drops every field in
  :data:`~panoscope.types.record.GENERATED_FIELDS`: the hashable half, which prose cannot
  move.
- ``write`` is the one IO function in ``core/`` and owns §10's rule "Never mutate the
  committed per-unit JSONs": records go to ``interp/records/<task_kind>/<unit>.json`` only.

Serialisation helpers return ``Traced`` with empty decisions, per ``core/``'s uniform
contract (``tests/test_contracts.py``).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Final

from panoscope.core.failure_modes import detect
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

    A ``MethodModule.load`` traces a whole file; a record for one unit must not carry the
    others' decisions. Item subjects are keyed on the entity, not the unit, so the anchor
    join alone cannot narrow it.

    So each trace is first cut into **unit segments**: a decision anchored on a ``unit|…``
    subject opens that unit's segment (``core.measure.measure_unit`` resolves the unit
    before its items). A trace with no unit anchor is wholly this unit's. Inside the
    segment the anchor join runs; ref-less decisions in the segment are kept
    (``band:no_canonical_driver`` has nothing to anchor on).

    Ambiguous shapes raise :class:`UnattributableTrace` rather than guess.

    The value and the decisions are the SAME tuple, so
    ``build(ev, reading, unit_decisions(ev, loaded), verdict)`` splices them in.
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
        # §9.8's named modes, indexed off the SAME traces the decisions come from
        # and off the evidence's own flags. Here rather than in each composition
        # root because all three roots already hand ``build`` exactly what
        # ``detect`` needs, and three call sites would be three chances for one to
        # forget -- a record silently missing a mode reads as a unit that did not
        # raise it. ``detect`` recomputes nothing and returns no decisions of its
        # own, so this can neither add a line to the audit trail nor move a number.
        failure_modes=tuple(m.id for m in detect(evidence, *traced).value),
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
