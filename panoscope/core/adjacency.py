"""The spatial adjacency check: segmentation bleed, or a real specificity problem.

ARCHITECTURE §9.9, in full:

> A gene supports unit C but also localizes in unit D. If C and D are spatially
> interdigitated that is a **segmentation bleed candidate**; if D sits elsewhere it
> is a **genuine specificity problem**. Same statistic, different readings, and only
> spatial data separates them. Display flag, not a band input, until its firing rate
> is measured.

Two things follow, and they are the whole module.

**It is the same number read two ways.** ``EvidenceItem.shared_with`` already carries
the other units this subject localises in; the method computed it once. Nothing here
recomputes or re-derives it -- a second implementation of "which units share this
marker" is the duplicate nobody updates. What this adds is the one fact the method
cannot supply: whether those units are spatially interdigitated with this one.

**Display flag, not a band input.** There is no band field, no score, no weight, and
nothing in ``core/verdict.py`` reads this module -- the ladder there takes
``EvidenceItem`` flags and corroborators, never an :class:`Adjacency`. Until the
firing rate is measured, a reading that moved the band would be an unvalidated
heuristic moving the one number the biologist acts on.

**What is deliberately NOT decided here.** Whether two units are interdigitated is a
spatial fact about coordinates, and ``core/`` reads no files. It arrives as
``interdigitated`` -- the set of units the caller has established neighbour this one
-- so the judgement is the caller's and is auditable as an input rather than being
computed from a threshold invented in this module. A caller that passes an empty set
gets every shared unit read as a specificity problem, which is the conservative
reading: "the marker also marks something else" is true either way, and only the
*excuse* for it (bleed) requires evidence this module does not have.

**And the check only runs on an observed item.** A subject the frame refused has no
localisation to share, so ``shared_with`` on such an item is a contradiction in the
reader rather than a finding; it produces no reading at all instead of a flag that
would read as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.types.evidence import EvidenceItem
from panoscope.types.subject import Subject
from panoscope.types.trace import Trace, Traced, ref


class AdjacencyRead(StrEnum):
    """The two readings of one shared localisation. Never collapsed into "shared".

    They are different facts to a biologist and they have different fixes: bleed is
    a segmentation problem, a specificity problem is a marker problem. A single
    "this marker is not unique" flag tells them to do nothing in particular.
    """

    BLEED_CANDIDATE = "bleed_candidate"  # the other unit is interdigitated with this one
    SPECIFICITY = "specificity"          # the other unit sits elsewhere


@dataclass(frozen=True, slots=True)
class SharedUnit:
    unit: str
    reads_as: AdjacencyRead


@dataclass(frozen=True, slots=True)
class Adjacency:
    """One subject's shared localisations, each read against spatial neighbourhood.

    No band, no score, no weight: §9.9 says display flag, and a field here is the
    only thing that would let it become a band input by accident.
    """

    unit: str
    subject: Subject
    shared: tuple[SharedUnit, ...] = ()

    def reading(self, reads_as: AdjacencyRead) -> tuple[str, ...]:
        return tuple(s.unit for s in self.shared if s.reads_as is reads_as)

    @property
    def bleed_candidates(self) -> tuple[str, ...]:
        return self.reading(AdjacencyRead.BLEED_CANDIDATE)

    @property
    def specificity_problems(self) -> tuple[str, ...]:
        return self.reading(AdjacencyRead.SPECIFICITY)


def adjacency_check(item: EvidenceItem, unit: str,
                    interdigitated: frozenset[str]) -> Traced[Adjacency]:
    """Read each of this item's shared units as bleed or as a specificity problem.

    Order is ``item.shared_with``'s, which is the method's own; it is not sorted,
    because this order is hashed into the record and the method's order is the only
    one the data supports.
    """
    t = Trace()
    if not item.measured.carries_number:
        # Nothing was localised here, so nothing can be shared. The item's own
        # ``measure:*`` decision already states why; restating it would be the
        # second copy that drifts.
        return t.done(Adjacency(unit=unit, subject=item.subject))

    anchor = (ref(item.subject),)
    shared: list[SharedUnit] = []
    for other in item.shared_with:
        if other == unit:
            continue
        if other in interdigitated:
            t.note("adjacency:bleed_candidate", other,
                   (f"{item.subject.display()} also localises in {other}, which is "
                    f"interdigitated with {unit}",), anchor)
            shared.append(SharedUnit(other, AdjacencyRead.BLEED_CANDIDATE))
        else:
            t.note("adjacency:specificity", other,
                   (f"{item.subject.display()} also localises in {other}, which does "
                    f"not neighbour {unit}",), anchor)
            shared.append(SharedUnit(other, AdjacencyRead.SPECIFICITY))

    return t.done(Adjacency(unit=unit, subject=item.subject, shared=tuple(shared)))
