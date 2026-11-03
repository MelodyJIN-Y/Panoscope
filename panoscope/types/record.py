"""The record: what was read, what was decided, and which half of that a model wrote.

The record is the reproducibility claim. Rebuild it from the same hashed inputs and
it must come out byte-identical, which is what lets §10's migration gate compare a v3
record against the committed v1 JSON field by field instead of eyeballing a report.
Everything in this module exists to make that identity a property of the types rather
than a discipline the writers are asked to keep.

**``drop_reasons`` is a sorted tuple of pairs, never a dict.** A dict field on a
frozen dataclass is unhashable, and its insertion order flows through
``dataclasses.asdict`` into the serialised bytes -- so two runs that dropped the same
rows for the same reasons would hash differently because a reader happened to
encounter a malformed row before an out-of-frame one. ``EvidenceItem.corroborating``
has the same shape for the same reason and sorts in ``__post_init__``; this does too,
so neither type depends on its callers remembering. ``IMPLEMENTATION.md``'s v2-to-v3
table lists both dicts as the pair of bugs that made ``test_record_deterministic``
unsatisfiable.

**Dropped rows are counted and named, not silently discarded.** ``rows_in`` minus
``rows_dropped`` is the only place the biologist can see that a quarter of the table
never reached the interpretation, and ``drop_reasons`` is what makes that actionable
rather than alarming.

**``entities_outside_frame`` and ``entities_not_in_reference`` are two different
facts and stay apart.** The first is the panel-absence rule -- the apparatus could
not have seen it, so its absence argues nothing. The second is a gap in *our*
knowledge of something that was measured perfectly well. Collapsing them into one
"unknown" list is exactly the four-states-into-a-boolean mistake ``measured.py``
exists to prevent, one layer up: a gene we measured but cannot annotate is a
literature gap to report, not a measurement that did not happen.

**``prose`` is the generated layer and is marked as such.** Every other field on
:class:`InterpretationRecord` is deterministic output from ``core/``; ``prose`` is
the only one a model may have written, so ``grounding_check`` gates it separately and
``prose=None`` is a complete, valid record -- the one the demo falls back to when a
request is slow or fails. :data:`GENERATED_FIELDS` names it in a form code can read,
because a gate that hardcodes the string ``"prose"`` silently stops covering the
second generated field the day one is added.
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
    #: REGISTERED ids in ``core/failure_modes.MODES`` declaration order.
    #:
    #: Ids and not prose, for the reason ``Verdict.demotions`` carries rule ids: a
    #: mode's ``what`` and ``ask`` are human-readable and never matched on, so
    #: copying them into every record would be a second copy of the registry that
    #: drifts the first time a sentence is reworded. ``failure_modes.explain``
    #: resolves one and raises if its detector has left ``RULES`` or ``FLAGS``.
    #:
    #: Nothing is DETECTED to fill this. Every id restates a decision already in
    #: ``decisions``; what it adds is the naming -- a biologist reading
    #: ``adjacency:bleed_candidate`` beside ``band:fragile_cap`` has no way to know
    #: the first is transcript bleed and the second a thin profile. §9.8 asks for
    #: the modes to be "named, computed, more actionable than a low band", and
    #: computed they already were.
    #:
    #: Declared after ``spec_version`` only because a dataclass field with a default
    #: cannot precede one without. It is record content, not metadata.
    failure_modes: tuple[str, ...] = ()
    prose: str | None = None                         # GENERATED layer; gated separately
