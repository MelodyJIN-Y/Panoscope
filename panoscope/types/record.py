"""The record: what was read, what was decided, and which half of that a model wrote.

Rebuilt from the same hashed inputs it must come out byte-identical (§10's migration
gate). The types make that a property, not a discipline:

- ``drop_reasons`` is a sorted tuple of pairs, never a dict (hashable; byte-stable).
- Dropped rows are counted and named: ``rows_in``, ``rows_dropped``, ``drop_reasons``.
- ``entities_outside_frame`` (panel-absence: absence argues nothing) and
  ``entities_not_in_reference`` (a gap in our knowledge of something measured) stay
  apart.
- ``prose`` is the only generated field; ``grounding_check`` gates it and ``prose=None``
  is a complete record (the deterministic fallback). :data:`GENERATED_FIELDS` names it
  for code to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panoscope.types.measured import FrameRef
from panoscope.types.outputs import TaskKind
from panoscope.types.trace import Decision


@dataclass(frozen=True, slots=True)
class ReadingRecord:
    """What one method's output file actually said, and what reading it cost.

    Hashed, not described: ``sha256`` is over the file bytes, so "the same inputs"
    is a checkable statement and not a claim about a path that may have been
    overwritten underneath us.
    """

    path: str
    sha256: str
    statistic_column: str
    unit_column: str
    unit_kind: str                                   # open vocabulary
    unit_detection_because: tuple[str, ...]
    rows_in: int
    rows_dropped: int
    drop_reasons: tuple[tuple[str, int], ...]        # SORTED pairs, never a dict
    entities_outside_frame: tuple[str, ...]          # never measured -> not evidence
    entities_not_in_reference: tuple[str, ...]       # measured, but unannotated
    frames: tuple[FrameRef, ...]
    clustering_mode: str                             # from PlatformAdapter, never guessed

    def __post_init__(self) -> None:
        # Byte-determinism: the serialised record must not depend on the order a
        # reader happened to encounter the rows it dropped.
        object.__setattr__(self, "drop_reasons", tuple(sorted(self.drop_reasons)))


GENERATED_FIELDS: Final[tuple[str, ...]] = ("prose",)
"""Fields on :class:`InterpretationRecord` a model may have written.

Read by ``grounding_check`` so the gate is defined by the type rather than by a
string literal in the gate. Everything NOT named here is deterministic output from
``core/`` and must render identically with the model offline.
"""


@dataclass(frozen=True, slots=True)
class InterpretationRecord:
    """One unit's interpretation, with everything needed to re-derive it and click back.

    ``decisions`` is the record's spine: a band with no decisions behind it is a
    number nobody can argue with, and arguing with it is the point of the tool.
    ``claims_used`` carries ``(claim_id, verdict)`` rather than claim ids alone
    because a claim that was *considered and refused* is as much a part of the
    reasoning as one that applied -- dropping the refusals makes the record read as
    though the evidence all pointed one way.

    ``undecidable`` is first-class, in the same spirit as ``Provenance.UNKNOWN``:
    what the data could not settle is reported, not omitted. An empty tuple means
    nothing was undecidable, which is a different statement from not having looked.
    """

    unit: str
    task_kind: TaskKind
    reading: ReadingRecord
    decisions: tuple[Decision, ...]
    claims_used: tuple[tuple[str, str], ...]         # (claim_id, verdict), call order
    undecidable: tuple[str, ...]
    spec_version: str
    #: ARCHITECTURE §9.8's named failure modes this unit's reading raised, as
    #: REGISTERED ids in ``core/failure_modes.MODES`` declaration order; each names
    #: a decision already in ``decisions`` (``failure_modes.explain`` resolves one).
    #: After ``spec_version`` only because it has a default.
    failure_modes: tuple[str, ...] = ()
    prose: str | None = None                         # GENERATED layer; gated separately
