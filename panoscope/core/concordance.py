"""Replicate concordance: a unit's profile across samples, under JOINT clustering.

ARCHITECTURE §9.10, in full:

> Under joint clustering, compare a unit's profile across replicates. Divergence is a
> flag: batch effect, section quality, or real biology, and the check does not claim
> which. **Agreement never raises confidence** -- replicates are not independent
> witnesses. Raises rather than returning a number under independent clustering.

Three constraints, all of them enforced rather than documented.

**It raises under independent clustering.** Cluster 3 in sample A and cluster 3 in
sample B are the same label by coincidence of numbering, not the same unit, and
comparing their profiles is a join on a key that does not mean anything. Returning a
"divergent" flag there would be confidently wrong about a comparison that was never
licensed, so :class:`IndependentClustering` is raised instead -- the one place this
package prefers a traceback to an output, because there is no honest output.

**Agreement never raises confidence**, and there is no field it could raise one in.
:class:`Concordance` carries no band, no score and no weight. ``divergent`` is a flag
and a flag only. The asymmetry is the point: replicates of one study share a panel,
a protocol, an operator and usually a day, so their agreeing is close to tautological
while their disagreeing is informative.

**The check does not claim which.** Batch effect, section quality and real biology
produce the same divergence, and this module cannot separate them; it reports the
divergence and names the subjects and samples it was read off, and stops there.

**The lattice applies here too, and it is why this shares ``reconcile._compare``.**
A subject never measured in sample B is not a divergence from sample A -- the join
across samples has exactly the same three outcomes as the join across methods, and
re-implementing the gate here would be a second place the four states could be
collapsed back into two.

**What is deliberately left undefined.** A subject a sample emitted *no item at all*
for is reported as not-comparable naming that sample, and no attempt is made to
adjudicate it against that sample's frames the way ``reconcile`` does. The honest
reading of a missing row is that this module cannot tell whether the reader dropped
it or the method never emitted it, and guessing either way would invent a measurement
status, which is the one thing the lattice exists to prevent.

**The replicates do NOT share one capability frame, and the items are what say so.**
Joint clustering joins the *units*; it says nothing about what each section was
capable of observing. One panel was run across the sections, so the panel frame is
shared -- but detectability is read off each section's own markers table, and a gene
that produced no usable signal there was not tested there. Each :class:`Replicate`'s
items therefore arrive already carrying that sample's own frames
(``pipeline/records.py`` builds them through ``adapter.for_sample``), and ``_compare``
reads each state off the item rather than off any frame this module holds -- which is
why it holds none. When one pooled frame was handed to every replicate instead, a
gene reported in one section and ``NoSig`` in the other came out of ``_compare`` as a
measured absence and therefore a CONFLICT: a divergence manufactured out of a
measurement that was never made.
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

    The first replicate is the reference and every other is compared against it,
    rather than all pairs: with the reference fixed the result is O(n) decisions in a
    stated order, and "sample B diverges from sample A" is the statement a biologist
    acts on. All-pairs would add a quadratic record that says the same thing twice in
    an order nothing pins.

    Subject order is the reference sample's own item order, then any subject only a
    later sample reported, in that sample's order. Not sorted: this order is hashed
    into the record, and most of these rows have no number to sort by.

    The ``floor`` the coverage is counted against is
    :data:`~panoscope.core.verdict.REPLICATE_COVERAGE_FLOOR`, not a parameter. It was
    one -- defaulting to ``1``, which ``Coverage.ok`` clears at every point of the
    gradient -- and a threshold a caller supplies is a threshold a caller can choose,
    which is the seam ``verify`` is not allowed to have: this K of N now demotes a
    band and sets the rubric's flag. The constant lives beside the other tuned band
    constants, with the honest note that it is structural rather than calibrated and
    why.

    One subject is CONFLICT if it conflicts in **any** replicate, NOT_COMPARABLE if
    it could not be compared in any and never conflicted, and CONSISTENT only if at
    least one comparison was made and none conflicted. A subject blocked in one
    sample and divergent in another is reported as divergent -- the divergence is a
    real measurement and the block is a caveat on top of it, not a reason to drop it.
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
