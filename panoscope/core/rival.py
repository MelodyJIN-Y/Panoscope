"""The rival test: what else could produce this reading, and what would settle it.

The rival test is not "name a competing cell type" (ARCHITECTURE §9.6). It is the
biologist asking **what else could produce this pattern**, and it has one shape
across every task kind: name a specific alternative account, name the measured
evidence that separates it from the current reading, then say honestly what the
measurement cannot settle. Only the space of alternatives changes.

This is ``agent/discriminate.py`` reimplemented against the four-state lattice, per
IMPLEMENTATION.md §10: *the four buckets become filters on ``Measured.state``*.

**The private vocabulary this replaces.** v1 carried its own status flag -- an
``off_panel: bool`` on every discriminator marker (``agent/discriminate.py:329``,
``agent/skeptic.py:177``) -- and asked exactly one capability question,
``data.panel_contains(gene)``. Everything that answered True and was not a top marker
of this unit was filed under "localizes elsewhere", which v1 then rendered as *that
argues against B*. So the only thing a rival's marker had to do in order to count as
evidence against the rival was **appear in the panel file**.

**The failure that causes.** 49 of the demo study's 280 panel genes yield no usable
spatial signal; jazzPanda labels them ``NoSig``. They are in the panel and they are
not a top marker of any unit, so v1 swept all of them into the against-the-rival
bucket and printed "not significant in any cluster -- that argues against B". Ask it
whether a macrophage cluster could be NK cells and it rules NK out on GNLY, PRF1,
KLRF1, NCAM1 and KLRC1: five real panel genes, none of which the measurement was
capable of localizing anywhere. A negative observation is evidence only if the
measurement could have produced a positive one, so that verdict is unfounded in
exactly the way the whole lattice exists to prevent -- and it is unfounded
*confidently*, which is worse than refusing. Under the detectability frame those five
are ``NEVER_MEASURED``, the coverage of the rival's canonical set is 0 of 5, and the
honest output is that the panel cannot settle it.

**And a boolean cannot say which frame refused.** "off the panel" and "on the panel
but undetectable" are different facts to a biologist -- the first is fixed by ordering
a different panel, the second is not -- and ``off_panel=True`` collapses them.
``Measured.frame`` names the frame that refused each marker -- carried on the item and
spelled out in the ``measure:*`` decision that adjudicated it -- so the record can say
*CD3D is not in panel:xenium_hbc_v1* and *GNLY failed detect:nosig* in the same
breath.

**What each bucket now is.** A filter on ``Measured.state``, nothing else:

- ``supports_call``  -- ``OBSERVED``, canonical for the current reading;
- ``supports_rival`` -- ``OBSERVED``, canonical for the rival: genuine rival signal
  here, which is ambiguity and not a refutation;
- ``against_rival``  -- ``TESTED_ABSENT``: in frame, measured, and not reported here.
  **The productive half.** A rival is argued against precisely when its markers were
  measured and peaked elsewhere;
- ``undecidable``    -- ``NEVER_MEASURED`` or ``UNDERPOWERED``. Argues nothing, and is
  reported rather than dropped. ``agent/skeptic.py``'s ``elif`` chain reached the
  off-panel branch only when nothing localized elsewhere, so for c3-vs-T_Cells the
  biologist was told the panel markers localize to c5 and never told that CD3D was
  never measured at all.

**Reference data selects; it never supplies a value.** ``canonical_call`` and
``canonical_rival`` are reference lists. They choose *which* subjects are asked about
and in what order. Every number in the result comes from ``ev.items`` -- that is, from
the method's own output, adjudicated by ``core.measure`` -- and a canonical marker
this unit did not report is built here as an ``EvidenceItem`` with no statistic at
all, which is the shape IMPLEMENTATION.md §10 gives the deleted ``OffPanelNote``.

**Deliberately not here.** Deriving the rival from the unit's own off-type markers
(v1's ``_derive_alt``) needs the whole canonical map rather than one rival's list, and
it is the workflow's question which alternative is worth testing; and prose
(``settle_summary``/``settle_line``) is rendering, which core does not do. Both are
the caller's, and both are strictly easier to write against the buckets below than
against v1's.
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

    ``reported`` is derived from ``ev.items`` rather than taken from ``ev.reported``.
    The two agree for well-formed evidence, but deriving it makes "this unit reported
    a number for X" and "this unit has an OBSERVED item for X" the same statement by
    construction, so a rival marker can never be resolved as OBSERVED while having no
    item to carry the number.

    The two iteration orders differ on purpose. ``supports_call`` walks ``ev.items``,
    which is the method's own output order (glm_coef descending for jazzPanda), so the
    strongest support leads. The rival's markers walk ``canonical_rival``, because a
    reference list has no ranking and most of them have no statistic to rank by --
    sorting them would invent an order the data does not support, and decision order
    is hashed into the record.

    ``statistic_name`` is read off the unit's first item: every item under one unit
    comes from one method and so shares it, and a unit with no items has no number to
    name.
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
