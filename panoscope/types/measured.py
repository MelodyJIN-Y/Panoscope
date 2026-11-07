"""Measurement status: the panel-absence rule, generalised.

> A negative observation is evidence only if the measurement was **capable** of
> producing a positive one. Capability is a property of the apparatus, declared
> *before* the result is read. Where capability is absent, the negative is not weak
> evidence -- it is **no** evidence, and it must be reported as such rather than
> dropped.

**Four states, not two.** ``TESTED_ABSENT`` is the productive half: a rival is argued
against because its markers were measured and peaked elsewhere.

Composite subjects take the *meet* over required roles, naming what pulled it down, then
are demoted by coverage over member roles.

**One frame per entity kind.** The panel is the gene frame; the annotation label set is
the cell-type frame; the design's replicate counts are the condition frame;
detectability is a second gene frame.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from functools import lru_cache
from enum import StrEnum

from panoscope.types.subject import Entity, EntityKind, Gate, Subject


class MeasuredState(StrEnum):
    NEVER_MEASURED = "never_measured"  # not in the frame        -> absence NOT evidence
    UNDERPOWERED = "underpowered"      # in frame, below a floor -> absence NOT evidence
    TESTED_ABSENT = "tested_absent"    # tested, not reported    -> absence IS evidence
    OBSERVED = "observed"              # reported with a statistic


_RANK: dict[MeasuredState, int] = {
    MeasuredState.NEVER_MEASURED: 0,
    MeasuredState.UNDERPOWERED: 1,
    MeasuredState.TESTED_ABSENT: 2,
    MeasuredState.OBSERVED: 3,
}


@dataclass(frozen=True, slots=True)
class Coverage:
    """K of N against a named floor.

    ONE home for every K-of-N in the system: panel hits in a gene set, leading-edge
    size, replicates in a condition, cells in a niche, permutations in a test.
    """

    what: str
    k: int
    n: int
    floor: int

    @property
    def ok(self) -> bool:
        return self.k >= self.floor

    @property
    def fraction(self) -> float:
        return (self.k / self.n) if self.n else 0.0

    def __str__(self) -> str:
        return f"{self.what} {self.k}/{self.n} (floor {self.floor})"


@lru_cache(maxsize=256)
def _normalised(members: frozenset[str]) -> frozenset[str]:
    """``members`` stripped and upper-cased, memoised on the frozenset itself.

    A frame's members are read once per study and asked about thousands of times,
    so the normalisation is cached rather than rebuilt per call. Keyed on the
    frozenset, which is both hashable and the whole input, so two frames with the
    same members share one answer and a frame built from a different file can never
    read another's.
    """
    return frozenset(str(m).strip().upper() for m in members)


@dataclass(frozen=True, slots=True)
class Frame:
    """What the apparatus was capable of observing, for ONE entity kind.

    The generalisation of ``data.panel_contains`` -- which the original docstring
    already called "THE absence primitive". The generalisation is simply: there is
    one of those per entity kind.
    """

    kind: EntityKind
    name: str            # "panel:xenium_hbc_v1" | "annotation:leiden_r0.8" | "detect:nosig"
    members: frozenset[str]

    @property
    def digest(self) -> str:
        """sha256 over the sorted members. DERIVED, never stored separately."""
        h = hashlib.sha256()
        for m in sorted(self.members):
            h.update(m.encode())
            h.update(b"\x00")
        return h.hexdigest()

    def contains(self, entity: Entity) -> bool:
        """Membership, normalised on BOTH sides: stripped and case-folded.

        Matches ``agent/data.panel_contains``, so both layers admit the same genes.
        ``members`` keeps the file's own spellings, so ``digest`` and ``size`` still describe
        what the store declared.
        """
        return _normalised(self.members).__contains__(str(entity.id).strip().upper())

    def ref(self) -> "FrameRef":
        return FrameRef(self.kind, self.name, self.digest, len(self.members))


@dataclass(frozen=True, slots=True)
class FrameRef:
    """What a Claim carries. Membership lives once, in the manifest.

    Claims are git-tracked one-file-each and re-read on every retrieval; inlining a
    280-symbol panel into each is ~8 KB per claim and makes equality accidentally
    exact -- one extra negative-control probe would read as a different world.
    """

    kind: EntityKind
    name: str
    digest: str
    size: int


@dataclass(frozen=True, slots=True)
class MeasuredWorld:
    """Every capability frame for one study. Several frames may share an entity kind."""

    frames: tuple[Frame, ...] = ()

    def for_kind(self, kind: EntityKind) -> tuple[Frame, ...]:
        return tuple(f for f in self.frames if f.kind is kind)

    def refs(self) -> tuple[FrameRef, ...]:
        return tuple(f.ref() for f in self.frames)


@dataclass(frozen=True, slots=True)
class Measured:
    """Whether a subject was CAPABLE of being observed here, and why."""

    state: MeasuredState
    frame: str
    because: tuple[str, ...] = ()
    blocked_by: tuple[Entity, ...] = ()   # constituents that pulled the meet down
    coverage: tuple[Coverage, ...] = ()

    @property
    def testable(self) -> bool:
        return self.state in (MeasuredState.OBSERVED, MeasuredState.TESTED_ABSENT)

    @property
    def absence_is_evidence(self) -> bool:
        """The whole point. Only a tested absence argues anything."""
        return self.state is MeasuredState.TESTED_ABSENT

    @property
    def carries_number(self) -> bool:
        """A statistic may exist only where something was actually observed.

        Enforced structurally at the EvidenceItem boundary, so quoting a number for
        an unmeasured thing is unrepresentable rather than merely tested for.
        """
        return self.state is MeasuredState.OBSERVED


def _state_for(entity: Entity, world: MeasuredWorld,
               reported: frozenset[str]) -> tuple[MeasuredState, str]:
    """Status of ONE entity. Every frame for its kind must admit it."""
    frames = world.for_kind(entity.kind)
    if not frames:
        return MeasuredState.NEVER_MEASURED, f"no frame declared for {entity.kind.value}"
    for f in frames:
        if not f.contains(entity):
            return MeasuredState.NEVER_MEASURED, f.name
    # Admitted states name every frame that admitted the entity (it passed all).
    admitted = "+".join(f.name for f in frames)
    # Normalised on BOTH sides, as ``Frame.contains`` is: a spelling difference
    # between a curated list and a method's table must not manufacture a
    # TESTED_ABSENT (measured absence is evidence).
    if str(entity.id).strip().upper() in _normalised(reported):
        return MeasuredState.OBSERVED, admitted
    return MeasuredState.TESTED_ABSENT, admitted


def resolve(subj: Subject, world: MeasuredWorld, reported: frozenset[str],
            coverage: tuple[Coverage, ...] = ()) -> Measured:
    """MEET over required roles, DEMOTED by coverage over member roles.

    Returns plain data; the caller records the decision. Pure: no IO, no clock.
    """
    required = subj.roles_with_gate(Gate.REQUIRED)
    if not required:  # degenerate shape; treat the whole subject as context
        return Measured(MeasuredState.OBSERVED, "none",
                        because=("no required roles",), coverage=coverage)

    per: list[tuple[Entity, MeasuredState, str]] = []
    for _role, ent in required:
        st, fr = _state_for(ent, world, reported)
        per.append((ent, st, fr))

    worst = min(_RANK[s] for _e, s, _f in per)
    state = next(s for _e, s, _f in per if _RANK[s] == worst)
    blocked = tuple(e for e, s, _f in per if _RANK[s] == worst
                    and s is not MeasuredState.OBSERVED)
    frame = next((f for _e, s, f in per if _RANK[s] == worst), "none")

    because: list[str] = []
    if blocked:
        because.append(
            f"{', '.join(str(e) for e in blocked)} {state.value} in {frame}"
        )

    # Coverage demotes an otherwise-testable subject, never promotes one.
    failing = tuple(c for c in coverage if not c.ok)
    if failing and _RANK[state] > _RANK[MeasuredState.UNDERPOWERED]:
        state = MeasuredState.UNDERPOWERED
        because.append("; ".join(str(c) for c in failing))

    return Measured(state=state, frame=frame, because=tuple(because),
                    blocked_by=blocked, coverage=coverage)
