"""The confidence band: the one number the biologist acts on, as recorded decisions.

A pure ladder wrapped by a traced step that records which rung was taken and why. The
band is pinned by the golden record and by the committed verdicts CSV downstream R reads.

Each adjustment is a REGISTERED rule id: ``demotions`` carries ids the next layer
branches on; English lives in ``RULES[...].what`` and the UI copy table. Ids are matched
on (``core/failure_modes.py``, the UI, the grounding tests), so they are never renamed
for prose -- ``band:driver_nosig`` keeps its id though its basis now reads "canonical
driver tested-absent".

- The three specificity demotes key on the ``specificity:*`` flags ``evidence.py`` owns,
  never on re-derived corroborators; the promote reads the pearson value itself.
- ``verify`` is rubric output computed here, never a parameter
  (``test_verify_is_not_an_input_anywhere``).
- ``band:demote_pearson_low`` and ``band:driver_nosig`` are unreachable from today's
  driver rule and kept so that loosening the selection rule cannot silently drop them.
- ``band:demote_canonical_coverage`` reads the capability frames (K of N of the label's
  canonical list); the curated list selects which measurement is asked, never a value.
- ``band:demote_replicate_coverage`` reads §9.10's K of N. It only subtracts: replicates
  are not independent witnesses, so there is no matching promote.
- NoSig is ``TESTED_ABSENT`` for this unit (tested, assigned elsewhere); a gene NoSig
  everywhere fails the detectability frame and is ``NEVER_MEASURED`` before it gets here.

The thresholds below are the one copy: ``core/`` does not import ``agent/``, and
``methods/jazzpanda.BAND_EDGES`` is derived from :data:`CONFIDENCE_BANDS`.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Coverage, MeasuredState
from panoscope.types.trace import Trace, Traced, fmt_num, ref

# --------------------------------------------------------------------------- #
# The ladder and its thresholds. The one home of the glm_coef band edges;
# methods/jazzpanda.BAND_EDGES is derived from CONFIDENCE_BANDS.
# --------------------------------------------------------------------------- #
BANDS: tuple[str, ...] = ("Very High", "High", "Medium-High", "Medium", "Low")

CONFIDENCE_BANDS: dict[str, float] = {
    "Very High": 10.0,
    "High": 5.0,
    "Medium-High": 2.5,
    "Medium": 1.0,
    "Low": 0.0,
}

PROMOTE_PEARSON: float = 0.85   # strong spatial specificity on the driver -> +1 band
FRAGILE_MARKER_COUNT: int = 2   # "one or two assigned markers" -> capped, always verify

#: How much of the LABEL'S canonical support the panel had to be able to ask for,
#: as an exact integer ratio so no float epsilon decides a band. One in three:
#: chosen between the worst demo coverage (c7, 4/7) and the synthetic fixture
#: (1/6), so it demotes the fixture and no demo unit.
CANONICAL_COVERAGE_FLOOR: tuple[int, int] = (1, 3)

#: ``Coverage.what`` for the canonical-support count. One string, because the
#: interface clicks back to it.
CANONICAL_COVERAGE_WHAT: str = "canonical markers in frame"

#: How much of a unit's profile the *second reading* had to be able to speak to
#: before "the sections did not diverge" is a statement about the unit. Same
#: exact-integer ratio shape as :data:`CANONICAL_COVERAGE_FLOOR`, and read by
#: ``core/concordance.py``, which is where that K of N is counted.
#:
#: **NOT calibrated**: structural, a majority. The only multi-sample study is the
#: synthetic fixture (9/12), which sits above it. Replace with a measured value when
#: a second multi-sample study lands; the K of N reaches ``undecidable`` either way.
REPLICATE_COVERAGE_FLOOR: tuple[int, int] = (1, 2)

#: ``Coverage.what`` for the replicate-comparison count, for the same reason the
#: canonical one is a constant: the interface clicks back to it.
REPLICATE_COVERAGE_WHAT: str = "subjects_comparable"

LOW_BAND: str = "Low"
#: The band of a unit with no label: nothing to assess, so no band and no verify.
NO_BAND: str = ""
FRAGILE_CAP_BAND: str = "Medium-High"
FRAGILE_FLOOR_BAND: str = "Medium"

# The three specificity flags, by the demote they drive. Ids, never prose.
FLAG_LOW_PEARSON: str = "specificity:low"
FLAG_LOCALISES_ELSEWHERE: str = "specificity:localises_elsewhere"
FLAG_NOT_UNIQUE: str = "specificity:not_unique"

PEARSON: str = "pearson"        # the corroborating statistic the promote reads


@dataclass(frozen=True, slots=True)
class Verdict:
    """The band, and everything the layer above branches on to explain it.

    ``demotions`` holds REGISTERED rule ids in the order applied. It mirrors the
    ``Decision`` tuple's rule ids so a consumer can ask *whether* the band moved without
    parsing the audit record.

    ``driver`` is the item the band was read off, so the interface clicks back to it rather
    than re-running the selection.
    """

    band: str
    basis: str
    verify: bool
    demotions: tuple[str, ...]
    driver: EvidenceItem | None
    fragile: bool
    coverage: Coverage | None = None
    #: The §9.10 replicate comparison's K of N, or ``None`` for *not assessed* --
    #: a single-sample study, independent clustering, or a unit only one section
    #: reported. ``None`` and ``Coverage(k=0, n=0)`` are different answers and are
    #: kept apart: the first says the question was never askable here, the second
    #: that it was asked and nothing could be compared.
    replicate_coverage: Coverage | None = None


# --------------------------------------------------------------------------- #
# The pure ladder. No tracing -- hammerable.
# --------------------------------------------------------------------------- #
def _band_for_coef(coef: float | None) -> str:
    """Magnitude of the driver's statistic -> band label. Bigger is more confident.

    ``abs()``: a strongly negative coefficient is a strong measurement, and the
    direction is carried by ``EvidenceItem.direction`` rather than being smuggled
    into the band as a weakness.
    """
    if coef is None:
        return LOW_BAND
    c = abs(float(coef))
    for label in ("Very High", "High", "Medium-High", "Medium"):
        if c >= CONFIDENCE_BANDS[label]:
            return label
    return LOW_BAND


def _demote(band: str) -> str:
    return BANDS[min(BANDS.index(band) + 1, len(BANDS) - 1)]


def _promote(band: str) -> str:
    return BANDS[max(BANDS.index(band) - 1, 0)]


def _coverage_floor(n: int, ratio: tuple[int, int] = CANONICAL_COVERAGE_FLOOR) -> int:
    """Smallest K that clears ``ratio`` out of ``n``. Defaults to the canonical one.

    ``ceil(n * num / den)`` in integer arithmetic. Not ``math.ceil(n * 1 / 3)``:
    that is a float comparison deciding a band, and ``0.1 * 3 > 0.3`` is how a
    tuned threshold silently moves a study by one rung.

    ``ratio`` is a parameter rather than a second copy of these three lines in
    ``core/concordance.py``, which needs the same arithmetic for
    :data:`REPLICATE_COVERAGE_FLOOR`. The default is the canonical floor because
    every call that existed before this parameter did meant that one.
    """
    num, den = ratio
    return -(-n * num // den)


def _canonical_coverage(items: tuple[EvidenceItem, ...],
                        canonical: frozenset[str]) -> Coverage:
    """K of N canonical markers the measurement was CAPABLE of reporting.

    N is the label's whole curated list; K is how many a capability frame admitted
    (``measured.testable``: OBSERVED or TESTED_ABSENT; not UNDERPOWERED). A canonical marker
    with no item is off-panel. Derived from the items rather than passed in, so no caller
    can forget it. A caller passing only assigned rows under-counts K, which errs toward
    "look again".
    """
    in_frame = {
        e.id.upper()
        for i in items if i.measured.testable
        for e in i.subject.entities()
    }
    n = len(canonical)
    return Coverage(CANONICAL_COVERAGE_WHAT, len(canonical & in_frame), n,
                    _coverage_floor(n))


def _v1_coverage_sentence(cov: Coverage) -> str:
    """The coverage demote in v1's on-disk English.

    Underscored because ``tests/test_contracts.py`` requires every PUBLIC function
    in ``core/`` to return ``Traced`` -- this one is a renderer, not a step, and has
    no decision to record. ``core/compat.py`` imports it anyway, exactly as it
    already imports ``_drivers``: it renders the v1 record from registered rule ids
    and must not hold a second copy of this string.
    """
    return (f"canonical coverage {cov.k}/{cov.n} below floor {cov.floor} "
            f"-> -1 band")


def _corroborator(item: EvidenceItem, name: str) -> float | None:
    return next((v for k, v in item.corroborating if k == name), None)


def _is_canonical(item: EvidenceItem, canonical: frozenset[str]) -> bool:
    """Case-insensitive, over every entity in the subject.

    Over entities rather than over a named gene role so that a composite subject --
    a ligand-receptor pair, a module's hub -- counts as canonical when a constituent
    is, instead of silently never counting at all under a shape nobody updated here.
    """
    return any(e.id.upper() in canonical for e in item.subject.entities())


def _drivers(items: tuple[EvidenceItem, ...],
             canonical: frozenset[str]) -> tuple[EvidenceItem, ...]:
    """Canonical markers eligible to carry the call, in the order given.

    The order is the caller's (glm_coef descending, as the method emitted it) and is
    never re-sorted here: sorting on the statistic would have to put items whose
    statistic is None somewhere arbitrary, and the first element is the whole
    decision.

    A canonical marker flagged ``specificity:low`` is NOT eligible, which
    is why ``band:demote_pearson_low`` cannot fire.
    """
    return tuple(
        i for i in items
        if _is_canonical(i, canonical)
        and i.measured.carries_number
        and FLAG_LOW_PEARSON not in i.flags
    )


def _verify(band: str, driver: EvidenceItem | None, fragile: bool,
            coverage: Coverage, replicate: Coverage | None = None) -> bool:
    """The verify rubric: six conditions, never caller-set.

    Includes thin canonical coverage and thin replicate coverage, each a threshold on a
    measured K of N. ``replicate=None`` means the comparison was never askable.
    """
    nosig = driver is not None and not driver.measured.carries_number
    return (band == LOW_BAND or nosig or fragile or driver is None
            or (bool(coverage.n) and not coverage.ok)
            or (replicate is not None and bool(replicate.n) and not replicate.ok))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assess(items: tuple[EvidenceItem, ...], canonical: frozenset[str],
           replicate: Coverage | None = None, called: bool = True) -> Traced[Verdict]:
    """Band and ``verify`` for one unit, with the decision behind every adjustment.

    ``canonical`` is the label's curated marker set, passed in (``core/`` does no IO). It
    selects which measurement is asked; the measurement alone supplies the number.

    Order of operations (load-bearing): base band from the driver's statistic, at most one
    promote, one demote per failing specificity condition, then the fragile cap, then the
    fragile floor.

    ``replicate`` is ``core.concordance.Concordance.coverage`` (§9.10) or ``None`` where it
    could not be asked. It only subtracts, supplies no value, and records nothing when it
    clears its floor. ``None`` means "never asked", never "asked and fine";
    ``pipeline/records.py`` writes the reason into ``undecidable``.

    ``called=False`` (the unit has no label yet): :data:`NO_BAND` and no verify --
    there is no claim to assess or to re-check.
    """
    t = Trace()
    if not called:
        t.note("band:not_called", "no band",
               ("the unit has no label yet, so there is no call to assess",), ())
        return t.done(Verdict(band=NO_BAND, basis="not yet called", verify=False,
                              demotions=(), driver=None,
                              fragile=False, replicate_coverage=replicate))
    canon = frozenset(g.upper() for g in canonical)

    # "Assigned markers" in the old sense: the ones that produced a number. An item
    # the frame refused is a caveat about the call, never a reason to call it robust.
    assigned = tuple(i for i in items if i.measured.carries_number)
    fragile = len(assigned) <= FRAGILE_MARKER_COUNT

    # Canonical COVERAGE: how much of the label's curated support the measurement
    # could have spoken to at all. Reference data selecting what gets asked, never
    # supplying a value -- the number below is still read off the driver alone.
    cov = _canonical_coverage(items, canon)

    drivers = _drivers(items, canon)
    driver = drivers[0] if drivers else None

    if driver is None:
        # No canonical support, or the only canonical evidence is an absence. Both
        # land on Low; they are told apart because "we measured it and it is not
        # here" and "nothing canonical was reported" are different answers, and only
        # the second might just mean the curated marker list is thin.
        nosig = next(
            (i for i in items
             if _is_canonical(i, canon)
             and i.measured.state is MeasuredState.TESTED_ABSENT),
            None,
        )
        if nosig is not None:
            rule, basis = "band:driver_nosig", "canonical driver tested-absent"
            because = (f"the leading canonical marker is tested-absent in "
                       f"{nosig.measured.frame}",)
            refs = (ref(nosig.subject),)
        else:
            rule, basis = "band:no_canonical_driver", "no canonical driver"
            because = (f"{len(assigned)} marker(s) carry a number, none canonical",)
            refs = ()
        t.note(rule, LOW_BAND, because, refs)
        # No second demote here: the band is already at the floor, and recording a
        # coverage demote that moved nothing would read as a rung the call lost.
        # The Coverage still travels, because "nothing canonical was reported" and
        # "almost nothing canonical was measurABLE" are different answers and the
        # interface has to be able to say which.
        return t.done(Verdict(band=LOW_BAND, basis=basis,
                              verify=_verify(LOW_BAND, None, fragile, cov, replicate),
                              demotions=(rule,), driver=None, fragile=fragile,
                              coverage=cov, replicate_coverage=replicate))

    band = _band_for_coef(driver.statistic)
    basis = "glm_coef direct"
    demotions: list[str] = []
    t.note("band:from_coef", band,
           (f"{driver.statistic_name}={fmt_num(driver.statistic)} on "
            f"{driver.subject.display()}",),
           (ref(driver.subject, driver.statistic_name, driver.statistic),))

    pearson = _corroborator(driver, PEARSON)
    if pearson is not None and pearson >= PROMOTE_PEARSON:
        promoted = _promote(band)
        if promoted != band:
            # Guarded on a real change, so the record never claims a promote the top
            # of the ladder absorbed; "Very High -> Very High" would read as evidence.
            band = promoted
            demotions.append("band:promote_pearson_high")
            t.note("band:promote_pearson_high", band,
                   (f"{PEARSON}={fmt_num(pearson)} at or above "
                    f"{fmt_num(PROMOTE_PEARSON)}",),
                   (ref(driver.subject, PEARSON, pearson),))

    # One demote per failing condition. Code order IS record order; each is a
    # single rung, so the band is order-independent but the record is not.
    for flag, rule in (
        (FLAG_LOW_PEARSON, "band:demote_pearson_low"),
        (FLAG_LOCALISES_ELSEWHERE, "band:demote_localises_elsewhere"),
        (FLAG_NOT_UNIQUE, "band:demote_not_unique"),
    ):
        if flag in driver.flags:
            band = _demote(band)
            demotions.append(rule)
            t.note(rule, band, (f"driver carries {flag}",), (ref(driver.subject),))

    # Then the condition that is about the APPARATUS rather than about the driver:
    # too little of the label's canonical list was ever in frame. One rung, like
    # every other demote. It is placed LAST among the rungs and before the fragile
    # cap/floor, which is not cosmetic: the band is order-independent (each rung is
    # one step) but the record is not, and putting it ahead of the promote would let
    # a high-pearson driver cancel it out -- the apparatus demote undone by a
    # property of the one marker that survived.
    if cov.n and not cov.ok:
        band = _demote(band)
        demotions.append("band:demote_canonical_coverage")
        t.note("band:demote_canonical_coverage", band, (str(cov),),
               (ref(driver.subject),))

    # And the second condition about the apparatus rather than the driver: how much
    # of this unit's profile the second section could be compared on. Placed after
    # the canonical demote because they can both fire and the record should read in
    # that order -- the panel could not ask much, AND the replicate could not check
    # much -- and before the fragile cap/floor for the reason the canonical one is:
    # ahead of the promote, a high-pearson driver would cancel out a statement about
    # the apparatus. One rung, like every other demote, and never more than one
    # however thin the coverage gets: the rung says "this is not settled", and
    # walking the band down by degrees would turn a coverage gap into a weak
    # measurement, which it is not.
    if replicate is not None and replicate.n and not replicate.ok:
        band = _demote(band)
        demotions.append("band:demote_replicate_coverage")
        t.note("band:demote_replicate_coverage", band, (str(replicate),),
               (ref(driver.subject),))

    if fragile:
        basis = "glm_coef direct (fragile cap)"
        if BANDS.index(band) < BANDS.index(FRAGILE_CAP_BAND):
            band = FRAGILE_CAP_BAND
            demotions.append("band:fragile_cap")
            t.note("band:fragile_cap", band,
                   (f"{len(assigned)} assigned marker(s), at or below "
                    f"{FRAGILE_MARKER_COUNT}",),
                   (ref(driver.subject),))
        if BANDS.index(band) > BANDS.index(FRAGILE_FLOOR_BAND):
            # The floor is not generosity. A fragile cluster WITH a real canonical
            # driver is expressed as "thin, look again" (verify, below), not as a Low
            # that reads identically to a cluster nothing supports at all.
            band = FRAGILE_FLOOR_BAND
            demotions.append("band:fragile_floor")
            t.note("band:fragile_floor", band,
                   (f"{len(assigned)} assigned marker(s) but a canonical driver",),
                   (ref(driver.subject),))

    return t.done(Verdict(
        band=band, basis=basis,
        verify=_verify(band, driver, fragile, cov, replicate),
        demotions=tuple(demotions), driver=driver, fragile=fragile,
        coverage=cov, replicate_coverage=replicate,
    ))
