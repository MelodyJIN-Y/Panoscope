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
from panoscope.types.measured import MeasuredState
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


def _verify(band: str, driver: EvidenceItem | None, fragile: bool) -> bool:
    """The rubric. Four conditions, no argument through which a caller states a wish.

    Private and computed, never passed in: ``verify`` is the flag that says a human
    must look again, so a layer that could set it could also clear it. The band being
    Low is one condition of four -- a confident-looking band standing on two markers
    still asks for a second pair of eyes.
    """
    nosig = driver is not None and not driver.measured.carries_number
    return band == LOW_BAND or nosig or fragile or driver is None


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
        return t.done(Verdict(band=LOW_BAND, basis=basis,
                              verify=_verify(LOW_BAND, None, fragile),
                              demotions=(rule,), driver=None, fragile=fragile))

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
        verify=_verify(band, driver, fragile),
        demotions=tuple(demotions), driver=driver, fragile=fragile,
    ))
