"""Replicate concordance: a unit's profile across samples, under JOINT clustering.

ARCHITECTURE §9.10:

> Under joint clustering, compare a unit's profile across replicates. Divergence is a
> flag: batch effect, section quality, or real biology, and the check does not claim
> which. **Agreement never raises confidence** -- replicates are not independent
> witnesses. Raises rather than returning a number under independent clustering.

- Under independent clustering :class:`IndependentClustering` is raised: equal labels
  across samples are not the same unit.
- :class:`Concordance` carries no band, score or weight; ``divergent`` is a flag only.
- The lattice applies: comparison shares ``reconcile._compare``, so a subject never
  measured in one sample is not a divergence.
- A subject a sample emitted no item for is reported not-comparable, naming the sample,
  without guessing a measurement status.
- Replicates do not share one capability frame: each :class:`Replicate`'s items carry
  that sample's own frames (``pipeline/records.py`` via ``adapter.for_sample``), and
  ``_compare`` reads state off the item.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoscope.core.reconcile import Comparison, _compare
from panoscope.core.verdict import (
    REPLICATE_COVERAGE_FLOOR, REPLICATE_COVERAGE_WHAT, _coverage_floor,
)
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Coverage
from panoscope.types.subject import Subject
from panoscope.types.trace import Trace, Traced, ref


class IndependentClustering(ValueError):
    """Raised when concordance is asked for under independent clustering (§9.10).

    Not a programming error: it is the expected outcome of asking a question the
    study's design does not license. The same unit id in two independently clustered
    samples is two different units wearing one label, and any answer would be about
    a join that never happened.
    """


class TooFewReplicates(ValueError):
    """Raised below two samples. One replicate has nothing to be concordant with."""


@dataclass(frozen=True, slots=True)
class Replicate:
    """One sample's items for ONE unit, as the method reported them."""

    sample: str
    items: tuple[EvidenceItem, ...]


@dataclass(frozen=True, slots=True)
class SubjectConcordance:
    """What the replicates said about one subject.

    ``outcome`` is :class:`Comparison`, not an agreement word. Replicates are the
    same question by construction, so there is no question-sameness mapping to do --
    and "agree" is a word that must not attach to replicates, because the next thing
    anyone asks of agreement is whether it raises the band.
    """

    subject: Subject
    outcome: Comparison
    because: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()   # sample names that could not speak to this subject


@dataclass(frozen=True, slots=True)
class Concordance:
    """One unit across replicates. No band, no score -- agreement never raises it.

    ``coverage`` is K of N comparable subjects against a floor, in the one home the
    system has for K-of-N. Without it, "one of forty subjects was comparable" and
    "all forty were" would both read as *not divergent*, which is the same
    denominator-thrown-away failure ``Rivalry.coverage`` replaced.
    """

    unit: str
    samples: tuple[str, ...]
    reference: str                  # the sample every other is compared against
    per_subject: tuple[SubjectConcordance, ...]
    coverage: Coverage

    @property
    def divergent(self) -> bool:
        """A flag, never a number. It does not say batch, section or biology."""
        return any(s.outcome is Comparison.CONFLICT for s in self.per_subject)

    @property
    def diverging_subjects(self) -> tuple[Subject, ...]:
        return tuple(s.subject for s in self.per_subject
                     if s.outcome is Comparison.CONFLICT)


def concordance(unit: str, replicates: tuple[Replicate, ...],
                joint: bool) -> Traced[Concordance]:
    """Compare a unit's profile across replicates. Raises under independent clustering.

    The first replicate is the reference and every other is compared against it. Subject
    order is the reference sample's item order, then subjects only later samples reported,
    in their order (hashed into the record; not sorted).

    The coverage floor is :data:`~panoscope.core.verdict.REPLICATE_COVERAGE_FLOOR`, not a
    parameter, since it feeds ``verify``.

    A subject is CONFLICT if it conflicts in any replicate, NOT_COMPARABLE if it could not
    be compared in any and never conflicted, and CONSISTENT only if compared at least once
    without conflict.
    """
    if not joint:
        raise IndependentClustering(
            f"{unit}: replicate concordance requires joint clustering (§9.10). Under "
            "independent clustering the same unit id in two samples is two different "
            "units sharing a label, and any comparison is a join on a meaningless key"
        )
    if len(replicates) < 2:
        raise TooFewReplicates(
            f"{unit}: concordance needs at least two replicates, got "
            f"{len(replicates)}"
        )

    t = Trace()
    ref_rep, others = replicates[0], replicates[1:]
    by_sample = {r.sample: {i.subject.key: i for i in r.items} for r in replicates}

    seen: set[str] = set()
    order: list[Subject] = []
    for rep in replicates:
        for item in rep.items:
            if item.subject.key not in seen:
                seen.add(item.subject.key)
                order.append(item.subject)

    per: list[SubjectConcordance] = []
    comparable = 0
    for subj in order:
        anchor = (ref(subj),)
        base = by_sample[ref_rep.sample].get(subj.key)
        because: list[str] = []
        blocked: list[str] = []
        conflict = False
        compared = False

        if base is None:
            blocked.append(ref_rep.sample)
            because.append(f"{ref_rep.sample} reported no item for {subj.key}")

        for rep in others:
            other = by_sample[rep.sample].get(subj.key)
            if other is None:
                blocked.append(rep.sample)
                because.append(f"{rep.sample} reported no item for {subj.key}")
                continue
            if base is None:
                continue
            cmp_, why, blocked_in = _compare(base, ref_rep.sample, other, rep.sample)
            because.extend(why)
            blocked.extend(b for b in blocked_in if b not in blocked)
            if cmp_ is Comparison.CONFLICT:
                conflict = True
                compared = True
            elif cmp_ is Comparison.CONSISTENT:
                compared = True

        if conflict:
            outcome = Comparison.CONFLICT
            t.note("concordance:divergent", subj.key, tuple(because), anchor)
        elif compared:
            outcome = Comparison.CONSISTENT
            t.note("concordance:consistent", subj.key, tuple(because), anchor)
        else:
            outcome = Comparison.NOT_COMPARABLE
            t.note("concordance:not_comparable", subj.key, tuple(because), anchor)

        if outcome is not Comparison.NOT_COMPARABLE:
            comparable += 1
        per.append(SubjectConcordance(subject=subj, outcome=outcome,
                                      because=tuple(because), blocked=tuple(blocked)))

    return t.done(Concordance(
        unit=unit,
        samples=tuple(r.sample for r in replicates),
        reference=ref_rep.sample,
        per_subject=tuple(per),
        coverage=Coverage(REPLICATE_COVERAGE_WHAT, comparable, len(order),
                          _coverage_floor(len(order), REPLICATE_COVERAGE_FLOOR)),
    ))
