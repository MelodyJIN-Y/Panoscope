"""The rival test: what else could produce this reading, and what would settle it.

Not "name a competing cell type" (ARCHITECTURE §9.6): name a specific alternative, name
the measured evidence that separates it, and say what the measurement cannot settle.
Per IMPLEMENTATION.md §10, the buckets are filters on ``Measured.state``:

- ``supports_call``  -- ``OBSERVED``, canonical for the current reading;
- ``supports_rival`` -- ``OBSERVED``, canonical for the rival: ambiguity, not refutation;
- ``against_rival``  -- ``TESTED_ABSENT``: in frame, measured, not reported here. The
  productive half;
- ``undecidable``    -- ``NEVER_MEASURED`` or ``UNDERPOWERED``. Argues nothing, and is
  reported rather than dropped.

Panel membership alone is not capability: a NoSig gene fails the detectability frame and
is ``NEVER_MEASURED``, so it cannot argue against a rival. ``Measured.frame`` names which
frame refused each marker (off-panel and undetectable are different facts).

Reference data selects; it never supplies a value. ``canonical_call`` and
``canonical_rival`` choose which subjects are asked about; every number comes from
``ev.items``. A canonical marker this unit did not report becomes an ``EvidenceItem``
with no statistic.

Not here: deriving the rival (the workflow's question) and prose (rendering).
"""

from __future__ import annotations

from dataclasses import dataclass

from panoscope.core.measure import resolve
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Coverage, MeasuredWorld
from panoscope.types.subject import gene, subject
from panoscope.types.trace import Trace, Traced, ref


@dataclass(frozen=True, slots=True)
class Rivalry:
    """One alternative account, and the measured evidence that bears on it.

    ``coverage`` is K of N over the rival's canonical set: how many of the markers
    that would separate the two accounts the apparatus was capable of speaking about
    at all. It replaces v1's ``settleable_on_panel``, which was
    ``bool(b_here or b_elsewhere)`` -- the same question with the floor hardcoded at
    one and the denominator thrown away, so "one of six canonical markers was
    testable" and "all six were" read identically.
    """

    call: str
    alternative: str
    coverage: Coverage
    supports_call: tuple[EvidenceItem, ...] = ()
    supports_rival: tuple[EvidenceItem, ...] = ()
    against_rival: tuple[EvidenceItem, ...] = ()
    undecidable: tuple[EvidenceItem, ...] = ()

    @property
    def settleable(self) -> bool:
        """Whether enough of the rival's canonical set was in frame to decide anything."""
        return self.coverage.ok


def _norm(text: str) -> str:
    """Identity, case- and whitespace-insensitive -- the one spelling rule.

    The same rule ``Frame.contains`` and ``types.measured._state_for`` apply, kept
    here rather than imported so this module states the rule it depends on. A
    vendor who writes erbb2 means ERBB2, and a curated list that writes Cd3e means
    the gene the table calls CD3E.
    """
    return str(text).strip().upper()


def _names(items: tuple[EvidenceItem, ...] | list[EvidenceItem]) -> str:
    return ", ".join(i.subject.display() for i in items)


def _unreported(symbol: str, statistic_name: str, world: MeasuredWorld,
                reported: frozenset[str]) -> Traced[EvidenceItem]:
    """A canonical marker this unit did not report, adjudicated against the frames.

    It carries no statistic by construction, which is the whole point: v1 attached
    ``glm_coef=None`` to such a gene by convention and nothing enforced it. Here
    ``EvidenceItem`` raises if the state and the statistic disagree, so if ``reported``
    ever claimed this symbol was OBSERVED while the unit produced no item for it, the
    contradiction surfaces loudly instead of becoming a quietly numberless row.
    """
    t = Trace()
    subj = subject("gene", g=gene(symbol))
    m = t.absorb(resolve(subj, world, reported))
    return t.done(EvidenceItem(subject=subj, statistic=None,
                               statistic_name=statistic_name, measured=m))


def rival_test(ev: NormalisedEvidence, call: str, alternative: str,
               canonical_call: tuple[str, ...], canonical_rival: tuple[str, ...],
               world: MeasuredWorld, floor: int = 1) -> Traced[Rivalry]:
    """Bucket the evidence that separates ``call`` from ``alternative`` for one unit.

    ``reported`` is derived from ``ev.items``, so an OBSERVED rival marker always has an
    item carrying its number.

    ``supports_call`` follows ``ev.items`` order (the method's ranking); the rival's markers
    follow ``canonical_rival`` order, since a reference list has no ranking to sort by.

    ``statistic_name`` is read off the unit's first item (one method per unit).
    """
    t = Trace()
    stat_name = ev.items[0].statistic_name if ev.items else ""
    # ponytail: flat ids, as the lattice itself keys on bare ids -- the same trade
    # core/measure.py's measure_unit makes, and for the same reason.
    reported = frozenset(
        e.id for i in ev.items if i.measured.carries_number for e in i.subject.entities()
    )
    # Keyed case- and whitespace-insensitively, because the two sides come from
    # different files: the items are spelled as the METHOD'S TABLE spells them, the
    # canonical lists as the REFERENCE spells them. An exact lookup missed on a case
    # variant and sent an observed gene down the ``_unreported`` path, where
    # ``resolve`` -- which does normalise -- returns OBSERVED for a statistic of
    # None, and ``EvidenceItem`` refuses the contradiction. Before ``resolve`` was
    # normalised it did not refuse: it returned TESTED_ABSENT, and the gene became a
    # measured absence SEPARATING the call from that rival. Evidence against a
    # rival, produced by a capitalisation.
    by_key = {_norm(i.subject.key): i for i in ev.items}

    wanted = frozenset(_norm(c) for c in canonical_call)
    supports_call = tuple(
        i for i in ev.items
        if i.measured.carries_number
        and any(_norm(e.id) in wanted for e in i.subject.entities())
    )

    supports_rival: list[EvidenceItem] = []
    against_rival: list[EvidenceItem] = []
    undecidable: list[EvidenceItem] = []
    for symbol in canonical_rival:
        item = by_key.get(_norm(subject("gene", g=gene(symbol)).key))
        if item is None:
            item = t.absorb(_unreported(symbol, stat_name, world, reported))
        if item.measured.carries_number:
            supports_rival.append(item)
        elif item.measured.absence_is_evidence:
            against_rival.append(item)
        else:
            undecidable.append(item)

    cov = Coverage("canonical_in_frame", len(supports_rival) + len(against_rival),
                   len(canonical_rival), floor)
    refs = (ref(ev.unit_subject),) + tuple(
        ref(i.subject) for i in (*supports_rival, *against_rival, *undecidable)
    )
    because = (f"{alternative}: {cov}",)
    # The out-of-frame markers are named on EVERY verdict, not only when they decide
    # it. ``agent/skeptic.py``'s elif chain reached its off-panel branch only when
    # nothing localised elsewhere, so a rival that was ruled out on four panel markers
    # never mentioned the fifth that was never measured. WHICH frame refused each one
    # is already in the ``measure:*`` decision that adjudicated it, and is not
    # restated here -- a second copy is the one that drifts.
    unmeasured = ((f"{_names(undecidable)} not in frame; they argue nothing",)
                  if undecidable else ())

    if not cov.ok:
        t.note("rival:undecidable", alternative, because + unmeasured, refs)
    elif supports_rival:
        t.note("rival:ambiguous", alternative, because
               + (f"{_names(supports_rival)} observed in {ev.unit}",) + unmeasured, refs)
    else:
        t.note("rival:argued_against", alternative, because
               + (f"{_names(against_rival)} in frame, tested, not reported in {ev.unit}",)
               + unmeasured, refs)

    return t.done(Rivalry(
        call=call,
        alternative=alternative,
        coverage=cov,
        supports_call=supports_call,
        supports_rival=tuple(supports_rival),
        against_rival=tuple(against_rival),
        undecidable=tuple(undecidable),
    ))
