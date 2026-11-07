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

**And ``band:demote_replicate_coverage`` is the second.** It reads §9.10's K of N
-- how many of this unit's subjects a second section could be compared on at all.
Before it, nothing in the confidence apparatus read concordance: ``assess`` was
called on the POOLED items and the concordance decisions were spliced into the
record afterwards, so a unit comparable on 1 of 12 subjects and one comparable on 9
of 12 both came out Very High with ``verify`` false, byte-identical. The only trace
was one sentence in ``undecidable`` carrying no flag, because the floor was a
placeholder ``1`` that nothing could fall below. The rung only ever subtracts:
replicates are not independent witnesses, so there is no promote to match it, and a
replicate coverage that clears its floor records no decision at all rather than one
the call is seen to have survived.

**A note on NoSig, and why nothing printed here says it.**
``MarkerEvidence.top_cluster == "NoSig"`` has no field here, and should not get one:
NoSig means *in the panel, tested, not assigned to this unit*, which is precisely
``TESTED_ABSENT``. So the NoSig branch keys on the measurement status, which is the
general form of the thing the old string encoded.

The sentinel answers two different questions and the two answers are not the same
fact. ``platforms/xenium.py`` asks *did this gene produce a usable signal anywhere in
this study?* and builds the detectability frame from the complement of the sentinel,
so a gene whose only row is NoSig fails that frame and is ``NEVER_MEASURED`` -- its
absence argues nothing. This module asks *was this gene tested and not assigned to
THIS unit?*, and an item that reached ``TESTED_ABSENT`` has already passed every
frame for its kind (``_state_for``: every frame must admit), detectability included.
The two are consistent, not in tension: a gene that is NoSig everywhere never gets
here, and a gene that gets here was measured well enough to be assigned somewhere
else. Which is why ``basis`` says *canonical driver tested-absent* and not "NoSig".
The v1 string it replaces -- "driving marker is NoSig" -- stated the opposite of the
measurement on a joint-clustered study, where a canonical marker peaking in a
different cluster is the ordinary case and not a failed assay.

The rule id stays ``band:driver_nosig``. Ids are matched on -- by
``core/failure_modes.py``, by the UI copy table, by the grounding tests -- and are
never read by a biologist; renaming one to improve prose is how a badge stops
rendering against a green test. The English is the part that was wrong, and the
English is the part that moved.

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

#: How much of a unit's profile the *second reading* had to be able to speak to
#: before "the sections did not diverge" is a statement about the unit. Same
#: exact-integer ratio shape as :data:`CANONICAL_COVERAGE_FLOOR`, and read by
#: ``core/concordance.py``, which is where that K of N is counted.
#:
#: **This one is NOT calibrated, and saying so is the honest part.** The value it
#: replaces was ``1``, a placeholder that kept ``Coverage.ok`` true at every point
#: of the gradient, so 1-of-12 and 9-of-12 reached the band engine identical.
#: Calibrating a replacement needs multi-sample studies, and the repository holds
#: exactly one: ``data/studies/synthetic_multisample_fixture``, whose three units
#: all sit at 9/12 = 0.750, with 7/12 = 0.583 reachable by the dropout edit
#: ``tests/test_concordance_wired.py`` already makes. Four points from one
#: synthetic study, all on one side of any line anyone would draw, is not a
#: distribution; a ratio fitted to it would be fitted to the fixture. The demo
#: contributes nothing at all, being single-sample and therefore not-assessed.
#:
#: So the number is structural rather than tuned: a **majority**. Below half the
#: comparison read a minority of the unit's subjects, and a negative result over a
#: minority is not a negative result about the unit -- which is the one claim the
#: demote exists to stop. It is deliberately placed where the only real data lies
#: ABOVE it, so the rule cannot be firing merely because a fixture was built to
#: trip it. Replace it with a measured value when a second multi-sample study
#: lands; the K of N travels into ``undecidable`` unconditionally either way, so
#: the floor decides only whether the band moves with it.
REPLICATE_COVERAGE_FLOOR: tuple[int, int] = (1, 2)

#: ``Coverage.what`` for the replicate-comparison count, for the same reason the
#: canonical one is a constant: the interface clicks back to it.
REPLICATE_COVERAGE_WHAT: str = "subjects_comparable"

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
            coverage: Coverage, replicate: Coverage | None = None) -> bool:
    """The rubric. Six conditions, no argument through which a caller states a wish.

    Private and computed, never passed in: ``verify`` is the flag that says a human
    must look again, so a layer that could set it could also clear it. The band being
    Low is one condition of six -- a confident-looking band standing on two markers
    still asks for a second pair of eyes.

    Thin canonical coverage is the fifth, and it is the reason the demote alone was
    not enough. One rung off a Very High leaves High, which still reads as a settled
    call; "re-check this" is what the rubric owes a call the panel could barely ask
    for. ``coverage`` is a measured K-of-N, not a preference: no caller supplies it.

    Thin REPLICATE coverage is the sixth, and it is the same argument about the
    second reading rather than about the panel. Its input is the one that does
    arrive from outside, because a second section cannot be derived from one unit's
    items; ``None`` means the comparison was never askable (one sample, independent
    clustering, one section silent about this unit) and is the only value that reads
    as silence. A ``Coverage`` below its floor says the sections were compared on a
    minority of this unit's subjects -- so the number on screen is the POOLED one
    and most of the profile behind it has no second reading at all. It is still a
    threshold on a measured K-of-N, not a judgement, and still nothing a caller can
    express a preference through: the only thing passed in is the count.
    """
    nosig = driver is not None and not driver.measured.carries_number
    return (band == LOW_BAND or nosig or fragile or driver is None
            or (bool(coverage.n) and not coverage.ok)
            or (replicate is not None and bool(replicate.n) and not replicate.ok))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def assess(items: tuple[EvidenceItem, ...], canonical: frozenset[str],
           replicate: Coverage | None = None) -> Traced[Verdict]:
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

    ``replicate`` is ``core.concordance.Concordance.coverage`` -- K of N of this
    unit's subjects that a second section could be compared on at all -- or ``None``
    where §9.10 could not be asked. It is the one argument that is neither the
    method's items nor reference data, and it is here because the defect it closes
    was exactly that it was not: the band was computed from the POOLED items and the
    concordance decisions were spliced into the record afterwards, so a unit
    comparable on 1 of 12 subjects and one comparable on 9 of 12 came out
    byte-identical at Very High with ``verify`` false.

    It can only SUBTRACT. Concordance is a second reading of the same method on the
    same panel, so §9.10 already forbids agreement from raising anything -- there is
    no promote branch below, and a replicate Coverage that clears its floor records
    no decision at all, because a rung nobody lost must not appear in the audit as
    one the call survived. It also supplies no value: the band is still read off the
    driver's own statistic, and this only moves it down a step.

    It is optional because *not assessed* is the common case and a real one -- the
    demo is single-sample. That is a softer seam than ``_canonical_coverage``, which
    is derived precisely so no caller can forget it, and the difference is forced: a
    second section is not recoverable from one unit's items. ``None`` therefore means
    "never asked" and never "asked and fine"; the composition root in
    ``pipeline/records.py`` writes the reason into ``undecidable`` in the same breath,
    so a reader can always tell which.
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
