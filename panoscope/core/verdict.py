"""The confidence band: the one number the biologist acts on, as recorded decisions.

This is ``agent/verdict.py::_compute_band`` moved into ``core/``, under the layering
``measure.py`` sets out: a **pure ladder** that a property test can hammer, wrapped by
a traced step that records which rung was taken and why. The band itself is unchanged
-- ``tests/test_core_verdict.py`` runs both implementations over all nine demo
clusters and asserts band, basis and the ordered adjustments are equal, because the
band is pinned by the golden record and by a committed CSV that downstream R reads.

**The failure this module exists to prevent.** The old band was a string returned
beside a tuple of English sentences ("driver localizes better with another cluster ->
-1 band"). The interface decided whether to draw a badge by matching that exact
sentence (``ui/evidence_table.py:53``), so a copy edit in the analysis layer silently
stopped a badge rendering and the test, asserting the same literal, stayed green. Here
each adjustment is a REGISTERED rule id: ``demotions`` carries ids the next layer
branches on, and the English lives in ``RULES[...].what`` and the UI's copy table,
where editing it is harmless. The ``Decision`` record carries the same ids with the
numbers that triggered them, so the value says *what* and the record says *why*.

**Why the three demotes key on flags and the promote keys on a number.** The three
specificity conditions are already computed once, by the method layer, and registered
as ``specificity:*`` flags on the ``EvidenceItem`` (``evidence.py`` FLAGS, transcribed
from ``_caveats_for``). Re-deriving them here from the raw corroborators would be a
second implementation of a rule ``evidence.py`` already owns -- the duplicate nobody
updates, which is exactly how the §2.2 spec bug got written down. The promote has no
flag because no caveat corresponds to it: "the driver is unusually specific" is not a
caveat, so it reads the pearson value itself.

**``verify`` is rubric output and is computed here, never offered to anyone.** It is
a field on ``Verdict`` rather than a parameter for the same reason ``Finding.verify``
is a plain field downstream: by the time it is stored the decision has been made, by
code, from four conditions, with no seam for a model or a caller to express a
preference. ``test_verify_is_not_an_input_anywhere`` pins that the seam does not exist.

**Two rules are unreachable from today's driver rule, deliberately kept.**
``band:demote_pearson_low`` cannot fire, because a canonical marker flagged
``specificity:low`` is not eligible to be the driver in the first place -- the low
pearson is spent on *disqualification* rather than on a demote. ``band:driver_nosig``
cannot fire from anything ``agent/verdict.py`` can build, since its evidence is
assigned rows only. Both are kept because the selection rule above them is the part
most likely to be loosened, and a transcription that silently drops the branches would
loosen it into a band change nobody reviewed. The equivalence test asserts neither
fires on the demo study, so "unreachable" is a measured claim and not a comment.

**Canonical coverage is the one modifier that is about the apparatus.** Every other
rung reads the driver: its coefficient, its pearson, how many markers carry a number.
``band:demote_canonical_coverage`` reads the capability frames instead -- K of N of
the label's canonical list that a frame admitted. It exists because the rest of the
ladder had a cliff at one: a unit with a single canonical marker on panel and the
rest never measured took the full band with no demotion and ``verify=False``, while
zero canonical markers was handled honestly. The curated list still supplies no
value; it selects which measurement is asked, and the one it selects is still the
only thing that answers.

**A note on NoSig.** ``MarkerEvidence.top_cluster == "NoSig"`` has no field here, and
should not get one: NoSig means *in the panel, tested, not assigned to this unit*,
which is precisely ``TESTED_ABSENT``. So the NoSig branch keys on the measurement
status, which is the general form of the thing the old string encoded.

Thresholds are duplicated from ``agent/config`` rather than imported -- ``core/`` does
not import ``agent/`` -- and ``test_thresholds_match_the_live_config`` fails the moment
the two diverge. A duplicated constant with a test is a smaller problem than an
inverted layering.
"""

from __future__ import annotations

from dataclasses import dataclass

from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Coverage, MeasuredState
from panoscope.types.trace import Trace, Traced, fmt_num, ref

# --------------------------------------------------------------------------- #
# The ladder and its thresholds. Transcribed from agent/config + agent/verdict;
# pinned against the live values by tests/test_core_verdict.py.
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
LOW_PEARSON: float = 0.30       # below this a canonical marker cannot drive the call
FRAGILE_MARKER_COUNT: int = 2   # "one or two assigned markers" -> capped, always verify

#: How much of the LABEL'S canonical support the panel had to be able to ask for,
#: as an exact integer ratio so no float epsilon decides a band. One in three: a
#: confident call must rest on a measurement that could have tested at least a
#: third of what the reference list says defines the label. Two thirds of a
#: canonical list may be off-panel or undetectable and the call still stands on
#: its own number -- below that the panel was not asking this question.
#:
#: Chosen inside a band verified against every study on disk: the worst demo
#: coverage is c7 at 4/7 = 0.571 and the synthetic fixture sits at 1/6 = 0.167,
#: so any floor strictly between those two demotes the fixture and no demo unit.
#: 1/3 sits in the middle of that band rather than on either edge, where one new
#: study would tip it.
CANONICAL_COVERAGE_FLOOR: tuple[int, int] = (1, 3)

#: ``Coverage.what`` for the canonical-support count. One string, because the
#: interface clicks back to it.
CANONICAL_COVERAGE_WHAT: str = "canonical markers in frame"

LOW_BAND: str = "Low"
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

    ``demotions`` holds REGISTERED rule ids in the order they were applied, which is
    the same order and the same length as the English tuple ``agent/verdict.py``
    returns. It duplicates the ``Decision`` tuple's rule ids on purpose: a consumer
    that needs to know *whether* the band was capped should not have to parse the
    audit record to find out, and a consumer that needs to know *why* should not be
    reading a field. Value and record, not one thing twice.

    ``driver`` is the item the band was read off, carried so the interface can click
    back to the measurement rather than re-running the selection and possibly
    disagreeing with it.
    """

    band: str
    basis: str
    verify: bool
    demotions: tuple[str, ...]
    driver: EvidenceItem | None
    fragile: bool
    coverage: Coverage | None = None


# --------------------------------------------------------------------------- #
# The pure ladder. No tracing -- hammerable.
# --------------------------------------------------------------------------- #
def _band_for_coef(coef: float | None) -> str:
    """Magnitude of the driver's statistic -> band label. Bigger is more confident.

    ``abs()`` matches ``config.band_for_coef``: a strongly negative coefficient is a
    strong measurement, and the direction is carried by ``EvidenceItem.direction``
    rather than being smuggled into the band as a weakness.
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


def _coverage_floor(n: int) -> int:
    """Smallest K that clears :data:`CANONICAL_COVERAGE_FLOOR` out of ``n``.

    ``ceil(n * num / den)`` in integer arithmetic. Not ``math.ceil(n * 1 / 3)``:
    that is a float comparison deciding a band, and ``0.1 * 3 > 0.3`` is how a
    tuned threshold silently moves a study by one rung.
    """
    num, den = CANONICAL_COVERAGE_FLOOR
    return -(-n * num // den)


def _canonical_coverage(items: tuple[EvidenceItem, ...],
                        canonical: frozenset[str]) -> Coverage:
    """K of N canonical markers the measurement was CAPABLE of reporting.

    N is the label's whole curated list; K is how many of them a capability frame
    admitted. ``Coverage`` is the one home for every K-of-N in the system, so this
    is counted, never re-invented.

    **Derived from the items rather than taken as an argument, on purpose.** An
    optional ``coverage=`` parameter is a soft default: every caller that forgot it
    would silently get the old, uncapped band back -- and ``workflows/marker.py``
    and ``pipeline/records.py`` are different callers. Deriving it means there is no
    seam to forget.

    **What "in frame" means here.** ``measured.testable`` -- OBSERVED or
    TESTED_ABSENT. ``UNDERPOWERED`` is deliberately NOT counted: the whole rule is
    that a negative is evidence only where a positive was possible, and
    underpowered says it was not. A canonical marker with no item at all is not in
    frame either; that is the off-panel case, which is the common one -- an
    off-panel gene never appears in a method's table, so its absence from ``items``
    IS the panel refusing it.

    The direction of the residual risk is stated so it stays a decision: a caller
    that hands over only the rows a unit was assigned under-counts K and demotes
    further than it should. That errs toward "look again", never toward a
    confident call the panel could not support.
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
    """The coverage demote in the English ``agent/verdict.py`` writes to disk.

    Underscored because ``tests/test_contracts.py`` requires every PUBLIC function
    in ``core/`` to return ``Traced`` -- this one is a renderer, not a step, and has
    no decision to record. ``core/compat.py`` imports it anyway, exactly as it
    already imports ``_drivers``: it renders the v1 record from registered rule ids
    and must not hold a second copy of this string. ``agent/verdict.py`` DOES hold
    a second copy, only because ``core/`` may not be imported from under it, and
    ``tests/test_core_verdict.py`` fails the moment the two diverge.
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

    A canonical marker flagged ``specificity:low`` is NOT eligible. That is the
    ``role == "supports"`` filter from ``agent/verdict.py::_build_evidence``, and it
    is why ``band:demote_pearson_low`` cannot fire.
    """
    return tuple(
        i for i in items
        if _is_canonical(i, canonical)
        and i.measured.carries_number
        and FLAG_LOW_PEARSON not in i.flags
    )


def _verify(band: str, driver: EvidenceItem | None, fragile: bool,
            coverage: Coverage) -> bool:
    """The rubric. Five conditions, no argument through which a caller states a wish.

    Private and computed, never passed in: ``verify`` is the flag that says a human
    must look again, so a layer that could set it could also clear it. The band being
    Low is one condition of five -- a confident-looking band standing on two markers
    still asks for a second pair of eyes.

    Thin canonical coverage is the fifth, and it is the reason the demote alone was
    not enough. One rung off a Very High leaves High, which still reads as a settled
    call; "re-check this" is what the rubric owes a call the panel could barely ask
    for. ``coverage`` is a measured K-of-N, not a preference: no caller supplies it.
    """
    nosig = driver is not None and not driver.measured.carries_number
    return (band == LOW_BAND or nosig or fragile or driver is None
            or (bool(coverage.n) and not coverage.ok))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assess(items: tuple[EvidenceItem, ...], canonical: frozenset[str]) -> Traced[Verdict]:
    """Band and ``verify`` for one unit, with the decision behind every adjustment.

    ``canonical`` is the study's own canonical-marker set for the label under test,
    passed in rather than looked up: ``core/`` does no IO, and a curated marker list
    is reference data, which may select and order evidence but may never supply a
    value. It does exactly that here -- it decides *which* measurement is asked for
    the band, and the measurement alone supplies the number.

    Order of operations, which is load-bearing and is the transcription: base band
    from the driver's statistic, then at most one promote, then one demote per
    failing specificity condition, then the fragile cap, then the fragile floor.
    Promote-before-demote is not arbitrary -- a driver that is both unusually
    specific and shares its pattern should land back where it started, not a rung
    lower.
    """
    t = Trace()
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
            rule, basis = "band:driver_nosig", "driver NoSig"
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
                              verify=_verify(LOW_BAND, None, fragile, cov),
                              demotions=(rule,), driver=None, fragile=fragile,
                              coverage=cov))

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

    # One demote per failing condition. Code order IS record order, and it is the
    # order agent/verdict.py applies them in; each is a single rung, so the band is
    # order-independent but the record is not.
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
        verify=_verify(band, driver, fragile, cov),
        demotions=tuple(demotions), driver=driver, fragile=fragile,
        coverage=cov,
    ))
