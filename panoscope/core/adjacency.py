"""The spatial adjacency check: segmentation bleed, or a real specificity problem.

ARCHITECTURE §9.9:

> A gene supports unit C but also localizes in unit D. If C and D are spatially
> interdigitated that is a **segmentation bleed candidate**; if D sits elsewhere it
> is a **genuine specificity problem**. Same statistic, different readings, and only
> spatial data separates them. Display flag, not a band input, until its firing rate
> is measured.

- Reads ``EvidenceItem.shared_with`` as the method reported it; recomputes nothing.
- Display flag only: no band field, and ``core/verdict.py`` does not read this module.
- ``interdigitated`` is supplied by the caller (``core/`` reads no files). An empty set
  reads every shared unit as a specificity problem, the conservative reading.
- Only an observed item is read; a refused subject has no localisation to share.

No caller yet: nothing produces ``interdigitated`` and no registered method populates
``shared_with``. ``tests/test_core_unwired.py`` pins that, so the day a method reports
``shared_with`` the test fails and this gets wired.
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
