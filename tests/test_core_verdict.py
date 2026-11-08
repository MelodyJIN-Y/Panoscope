"""Invariants for the confidence band in ``core/``.

The band is the single most load-bearing number this project produces: it is pinned
by ``tests/test_golden.py``, written into a committed 11-column CSV that downstream R
reads, and rendered as the pill the biologist acts on. ``core/verdict.py`` is the one
engine that computes it; the demo's end-to-end output is pinned byte-for-byte by the
golden hash, and this module pins the rules themselves.

Synthetic items cover what the demo study does not contain: a tested-absent canonical
driver, a cap, a floor, and a promote at the top of the ladder. Those are built by
hand rather than by perturbing the real study, because a fixture that mutates the demo
tree is the fixture that breaks the golden hash for everyone else on the branch.
"""

from __future__ import annotations

import inspect

import pytest

from panoscope.agent import config as cfg
from panoscope.core import compat as COMPAT
from panoscope.core import verdict as C
from panoscope.pipeline import records
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Coverage, Measured, MeasuredState
from panoscope.types.subject import gene, subject

_FRAME = "panel:xenium_hbc_v1"

#: A canonical list whose markers are mostly NOT in frame: the coverage demote is
#: the one rung that reads the capability frames rather than the driver.
_THIN: frozenset[str] = frozenset({"ERBB2", "C1", "C2", "C3", "C4", "C5"})


def _item(name: str, coef: float | None, pearson: float | None = None,
          flags: tuple[str, ...] = (),
          state: MeasuredState = MeasuredState.OBSERVED) -> EvidenceItem:
    """One EvidenceItem in the shape the method layer hands to the band."""
    observed = state is MeasuredState.OBSERVED
    return EvidenceItem(
        subject=subject("gene", g=gene(name)),
        statistic=coef if observed else None,
        statistic_name="glm_coef",
        measured=Measured(state, _FRAME),
        corroborating=(("pearson", pearson),) if pearson is not None and observed else (),
        flags=flags,
    )


def _inputs(cluster: str):
    """(items, canonical) for one demo cluster -- core's two arguments, as the
    composition root builds them."""
    b = records.build_all(cfg.current_study())[cluster]
    return b.evidence.items, b.canonical


# --------------------------------------------------------------------------- #
# The demo exercises the ladder
# --------------------------------------------------------------------------- #
@pytest.mark.req("SKM-3d-04")
def test_demotes_are_recorded_in_rule_order():
    """Each demote is one rung, so the band survives a reordering while the record --
    and the badges it draws -- do not. Swapping two demote branches once left a
    nine-cluster sweep green because no demo cluster carries both flags."""
    driver = _item("ERBB2", 30.0, 0.9,
                   (C.FLAG_NOT_UNIQUE, C.FLAG_LOCALISES_ELSEWHERE))
    filler = tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    v = C.assess((driver, *filler), _THIN).value
    assert v.demotions == ("band:demote_localises_elsewhere", "band:demote_not_unique",
                           "band:demote_canonical_coverage")
    assert v.band == "Medium" and v.verify


def test_the_equivalence_sweep_is_not_vacuous():
    """Nine clusters all reading Low with no adjustments would pass the sweep while
    testing nothing. Pins that the demo really exercises the ladder."""
    bands, adjusted = set(), 0
    for cluster in cfg.CLUSTER_ORDER:
        v = C.assess(*_inputs(cluster)).value
        bands.add(v.band)
        adjusted += bool(v.demotions)
    assert len(bands) > 1, f"every demo cluster read the same band: {bands}"
    assert adjusted, "no demo cluster took a single adjustment"


def test_the_two_unreachable_rules_stay_unreachable_on_the_demo():
    """Both are kept as live branches (see the module docstring). This is the
    measurement behind that claim -- if one starts firing, the driver rule has moved
    and the band has moved with it."""
    fired = {d.rule for c in cfg.CLUSTER_ORDER
             for d in C.assess(*_inputs(c)).decisions}
    assert "band:demote_pearson_low" not in fired
    assert "band:driver_nosig" not in fired


# --------------------------------------------------------------------------- #
# Reproducibility -- the band is a function of its inputs and nothing else
# --------------------------------------------------------------------------- #
def test_the_same_inputs_give_the_same_record_every_time():
    """No clock, no randomness, no set iteration order. A band that moves between two
    calls in one process destroys the reproducibility claim the record exists to
    make, and would surface as a moving golden hash long after the cause."""
    items, canonical = _inputs("c1")
    first = C.assess(items, canonical)
    for _ in range(5):
        again = C.assess(items, canonical)
        assert again.value == first.value
        assert again.decisions == first.decisions


def test_a_reordered_canonical_set_does_not_move_the_band():
    """``canonical`` arrives as a set. If anything iterated it, the driver -- and so
    the band -- would depend on hash order rather than on glm_coef."""
    items, canonical = _inputs("c1")
    shuffled = frozenset(reversed(sorted(canonical)))
    assert C.assess(items, shuffled).value == C.assess(items, canonical).value


def test_the_record_and_the_value_agree_on_the_adjustments():
    """``demotions`` duplicates the decision rule ids by design; this is the test that
    keeps the duplication honest rather than merely cheap."""
    for cluster in cfg.CLUSTER_ORDER:
        t = C.assess(*_inputs(cluster))
        from_record = tuple(d.rule for d in t.decisions if d.rule != "band:from_coef")
        assert t.value.demotions == from_record, cluster


# --------------------------------------------------------------------------- #
# Canonical coverage -- the rung that reads the apparatus, not the driver
# --------------------------------------------------------------------------- #
def test_the_floor_is_an_exact_ratio_and_not_a_float_comparison():
    """``ceil(n/3)`` by integer arithmetic. A float version is off by one wherever
    ``n * (1/3)`` lands a hair above an integer, which moves a band by a rung on
    some panel sizes and not others -- the hardest kind of threshold bug to see."""
    import math
    from fractions import Fraction

    num, den = C.CANONICAL_COVERAGE_FLOOR
    # Ratio-agnostic on purpose: this must keep testing ROUNDING after a tuning
    # pass changes the ratio, and the ratio itself is pinned by the census in
    # tests/test_second_study.py rather than by a literal here.
    for n in range(0, 200):
        assert C._coverage_floor(n) == math.ceil(Fraction(n * num, den)), n
    # A K at the floor clears it and one below does not -- the two cases a
    # rounding slip swaps, at every panel size a curated list plausibly has.
    for n in range(1, 40):
        f = C._coverage_floor(n)
        assert Coverage("x", f, n, f).ok
        assert not Coverage("x", f - 1, n, f).ok


def test_the_v1_coverage_sentence_is_pinned():
    """``core/compat.py`` renders the v1 record from rule ids; the demo never fires
    this rule, so nothing else in the suite pins the sentence."""
    assert C._v1_coverage_sentence(
        Coverage(C.CANONICAL_COVERAGE_WHAT, 1, 6, 2)
    ) == "canonical coverage 1/6 below floor 2 -> -1 band"


@pytest.mark.req("CORP-B2")
def test_coverage_counts_the_frame_and_not_this_unit_s_result():
    """K is a fact about the APPARATUS. A canonical marker that was in frame, tested
    and not reported here still counts: the panel could have reported it. Counting
    only what this unit observed would make coverage a second reading of the band
    and would demote every unit that is not the marker's own."""
    canon = frozenset({"A", "B", "C"})
    items = (_item("A", 10.0, 0.5),
             _item("B", None, state=MeasuredState.TESTED_ABSENT),
             _item("C", None, state=MeasuredState.NEVER_MEASURED))
    cov = C._canonical_coverage(items, canon)
    assert (cov.k, cov.n) == (2, 3), str(cov)


def test_an_underpowered_canonical_marker_does_not_count_as_coverage():
    """The whole rule is that a negative is evidence only where a positive was
    possible. UNDERPOWERED says it was not, so it cannot buy canonical support --
    counting it would let a measurement that could not have answered the question
    look like one that did."""
    canon = frozenset({"A", "B"})
    items = (_item("A", 10.0, 0.5),
             _item("B", None, state=MeasuredState.UNDERPOWERED))
    assert C._canonical_coverage(items, canon).k == 1


def test_a_canonical_marker_with_no_item_at_all_is_not_in_frame():
    """The off-panel case, and the common one: a gene the panel never carried never
    appears in a method's table, so its absence from ``items`` IS the refusal. If
    this counted as in-frame the demote could never fire on the study it exists for,
    because an IO panel's missing kidney markers are missing exactly this way."""
    assert C._canonical_coverage((_item("A", 10.0, 0.5),),
                                 frozenset({"A", "B", "C", "D", "E", "F"})).k == 1


def test_thin_coverage_demotes_one_rung_and_sets_verify():
    """The cliff, as a before/after on one set of numbers.

    Same driver, same coefficient, same everything -- only the size of the curated
    list the panel was asked to cover differs. That is the whole claim: the band is
    still the driver's number, and coverage subtracts from it.
    """
    driver = _item("ERBB2", 30.0, 0.5)
    filler = tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    items = (driver, *filler)

    covered = C.assess(items, frozenset({"ERBB2"})).value
    assert covered.band == "Very High" and covered.verify is False
    assert covered.demotions == ()

    thin = C.assess(items, _THIN).value
    assert thin.band == "High", "one rung, not a collapse to Low"
    assert thin.verify is True, "a rung alone still reads as a settled call"
    assert thin.demotions == ("band:demote_canonical_coverage",)
    assert str(thin.coverage) == "canonical markers in frame 1/6 (floor 2)"


def test_the_coverage_decision_carries_its_k_of_n():
    """A demote whose record does not say what it was measured against is a rung the
    biologist cannot argue with."""
    t = C.assess((_item("ERBB2", 30.0, 0.5),
                  *(_item(f"G{i}", 1.0, 0.5) for i in range(3))), _THIN)
    d = next(d for d in t.decisions if d.rule == "band:demote_canonical_coverage")
    assert "1/6" in " ".join(d.because) and "floor 2" in " ".join(d.because)
    assert d.outcome == "High", "the record carries the rung it landed on"
    assert d.refs, "and what it read, so the sentence clicks back"


def test_no_canonical_driver_does_not_also_take_a_coverage_demote():
    """Already at the floor. A second rung would record a demotion that moved
    nothing, and the record would read as a call that lost two.

    The Coverage still travels, because "nothing canonical was reported" and "almost
    nothing canonical was measurABLE" are different answers.
    """
    v = C.assess((_item("ACTA2", 30.0, 0.9), _item("PTPRC", 20.0, 0.9),
                  _item("CD3E", 15.0, 0.9), _item("MS4A1", 9.0, 0.9)), _THIN).value
    assert v.band == "Low"
    assert v.demotions == ("band:no_canonical_driver",)
    assert v.coverage is not None and (v.coverage.k, v.coverage.n) == (0, 6)


@pytest.mark.req("SKM-3b-01")
def test_an_empty_canonical_list_never_demotes():
    """A study whose annotation lists no canonical markers for a label says nothing
    about coverage, and silence is not a floor failure. ``0/0`` must not read as the
    worst possible coverage -- there is no driver either, so the band is already
    Low, and an extra rung here would be the apparatus arguing from an empty list."""
    v = C.assess((_item("ACTA2", 30.0, 0.9),) * 4, frozenset()).value
    assert v.coverage.n == 0
    assert v.demotions == ("band:no_canonical_driver",)


# --------------------------------------------------------------------------- #
# verify comes from the rubric, never from a caller
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-HR-05")
def test_verify_is_not_an_input_anywhere_in_the_module():
    """"``verify`` comes from the deterministic rubric -- never offer the model the
    choice." A parameter is how the choice gets offered: nobody adds a required one,
    they add ``verify: bool | None = None`` and every call site keeps working."""
    offenders = [
        f"{name}({p})"
        for name, fn in vars(C).items()
        if inspect.isfunction(fn) and fn.__module__ == C.__name__
        for p in inspect.signature(fn).parameters
        if "verify" in p
    ]
    assert not offenders, "verify is settable from outside: " + "; ".join(offenders)


@pytest.mark.req("CLAUDE-HR-05", "SKM-3d-03")
def test_verify_fires_on_each_of_its_five_conditions():
    canonical = frozenset({"ERBB2", "KRT7", "EPCAM", "FOXA1"})
    strong = tuple(_item(g, 20.0, 0.5) for g in ("ERBB2", "KRT7", "EPCAM", "FOXA1"))

    assert C.assess(strong, canonical).value.verify is False            # the control
    assert C.assess(strong[:2], canonical).value.verify is True         # fragile
    assert C.assess((_item("ACTA2", 20.0, 0.5),) * 4,                   # no driver
                    canonical).value.verify is True
    low = C.assess(tuple(_item(g, 0.4, 0.5) for g in canonical), canonical)
    assert low.value.band == "Low" and low.value.verify is True         # Low band

    # thin canonical coverage: a confident band on a question the panel could
    # barely pose. The control above differs only in the size of the curated list.
    thin = C.assess(strong, canonical | {"X1", "X2", "X3", "X4", "X5", "X6", "X7",
                                         "X8", "X9", "X10"})
    assert thin.value.band != C.LOW_BAND, "verify, not a collapse"
    assert thin.value.verify is True


# --------------------------------------------------------------------------- #
# The branches the demo study does not reach
# --------------------------------------------------------------------------- #
def test_no_canonical_driver_reads_low_with_its_rule():
    v = C.assess((_item("ACTA2", 30.0, 0.9), _item("PTPRC", 20.0, 0.9),
                  _item("CD3E", 15.0, 0.9), _item("MS4A1", 9.0, 0.9)),
                 frozenset({"ERBB2"})).value
    assert v.band == "Low"
    assert v.basis == "no canonical driver"
    assert v.demotions == ("band:no_canonical_driver",)
    assert v.driver is None


@pytest.mark.req("CLAUDE-HR-02")
def test_a_tested_absent_canonical_marker_is_the_nosig_branch():
    """NoSig means in-panel, tested, not assigned here -- which is TESTED_ABSENT. It
    reads Low like 'no canonical driver', but it is a different answer: the marker WAS
    measured, so its absence argues, and the record has to say which happened.

    ``basis`` is the ``band_basis`` column of ``interp/verdicts.csv`` and of the
    Summary table, so it is read by a biologist and not only by code. It must not say
    "NoSig": an item only reaches TESTED_ABSENT once every frame for its kind admitted
    it, detectability included, so "no usable signal" is the one thing this branch has
    ruled out. The band, the rule id and ``verify`` are pinned here unchanged -- the
    words moved, the ladder did not.
    """
    v = C.assess(
        (_item("ERBB2", None, state=MeasuredState.TESTED_ABSENT),
         _item("ACTA2", 30.0, 0.9), _item("PTPRC", 20.0, 0.9), _item("CD3E", 9.0, 0.9)),
        frozenset({"ERBB2"}),
    ).value
    assert v.band == "Low"
    assert v.basis == "canonical driver tested-absent"
    assert v.demotions == ("band:driver_nosig",)
    assert v.verify is True
    assert "nosig" not in v.basis.lower(), (
        "the band_basis the biologist reads says the measurement produced no signal, "
        "when TESTED_ABSENT means it produced one and it was not assigned here"
    )


def test_a_low_pearson_canonical_marker_is_disqualified_not_demoted():
    """The reason ``band:demote_pearson_low`` cannot fire. If this ever flips to a
    demote, the band for every weakly-specific cluster moves a rung."""
    v = C.assess(
        (_item("ERBB2", 30.0, 0.1, flags=(C.FLAG_LOW_PEARSON,)),
         _item("ACTA2", 20.0, 0.9), _item("CD3E", 9.0, 0.9), _item("MS4A1", 8.0, 0.9)),
        frozenset({"ERBB2"}),
    ).value
    assert v.driver is None
    assert v.demotions == ("band:no_canonical_driver",)


@pytest.mark.req("SKM-3d-02")
def test_the_fragile_cap_and_the_floor_are_mutually_exclusive():
    """Both key on 'fragile', but the cap pulls a high band down to Medium-High and
    the floor lifts a Low up to Medium, so one call can only ever take one of them."""
    canonical = frozenset({"TPSAB1"})
    capped = C.assess((_item("TPSAB1", 30.0, 0.5),), canonical).value
    assert capped.band == "Medium-High"
    assert capped.demotions == ("band:fragile_cap",)
    assert capped.basis == "glm_coef direct (fragile cap)"

    floored = C.assess(
        (_item("TPSAB1", 0.5, 0.5, flags=(C.FLAG_NOT_UNIQUE,)),), canonical
    ).value
    assert floored.band == "Medium"
    assert floored.demotions == ("band:demote_not_unique", "band:fragile_floor")


def test_a_promote_the_ladder_absorbs_is_not_recorded():
    """A 'Very High -> Very High' promote in the record reads as evidence the band was
    raised. It was not; the driver was already at the top of the ladder."""
    canonical = frozenset({"ERBB2"})
    filler = tuple(_item(f"G{i}", 1.0, 0.5) for i in range(4))
    top = C.assess((_item("ERBB2", 30.0, 0.99), *filler), canonical).value
    assert top.band == "Very High"
    assert top.demotions == ()

    lower = C.assess((_item("ERBB2", 6.0, 0.99), *filler), canonical).value
    assert lower.band == "Very High"
    assert lower.demotions == ("band:promote_pearson_high",)


def test_the_promote_needs_the_number_and_the_demotes_need_only_the_flag():
    """Documents the asymmetry so it stays a decision and not an accident: a driver
    with no corroborating pearson simply cannot be promoted, and never raises."""
    canonical = frozenset({"ERBB2"})
    filler = tuple(_item(f"G{i}", 1.0) for i in range(4))
    assert C.assess((_item("ERBB2", 6.0), *filler), canonical).value.demotions == ()


def test_the_canonical_census_and_the_coverage_it_describes_count_one_set_of_numbers():
    """``_canonical_coverage``'s docstring says ``core.compat`` counts it the same
    way. Verified here instead of repeated: if the two disagree, the band and the
    sentence printed beside the band are computed from different numbers, which is a
    fabricated measurement status wearing a denominator.

    The invariant is exact, not approximate. Every canonical NAME falls in exactly
    one clause, so the clauses sum to ``cov.n``; the two TESTABLE clauses (OBSERVED
    and TESTED_ABSENT -- the lattice's definition of "a positive was possible", and
    what ``_canonical_coverage`` builds ``in_frame`` from) sum to ``cov.k``.

    The grid is the shapes that used to break it: a gene read twice, a composite
    subject carrying two canonical names, UNDERPOWERED (in frame, not testable) and
    a canonical name with no item at all.
    """
    canon = frozenset({"C1", "C2", "C3", "C4", "C5"})
    items = (
        _item("C1", 2.0, 0.5),                                   # OBSERVED
        _item("C1", None, state=MeasuredState.NEVER_MEASURED),   # same gene, again
        _item("C2", None, state=MeasuredState.TESTED_ABSENT),
        _item("C3", None, state=MeasuredState.UNDERPOWERED),
        EvidenceItem(                                            # composite subject
            subject=subject("gene_pair", a=gene("C4"), b=gene("NOTCANON")),
            statistic=None, statistic_name="glm_coef",
            measured=Measured(MeasuredState.NEVER_MEASURED, _FRAME)),
        # C5 has no item at all -- the panel refusing it before the method ran.
    )
    cov = C._canonical_coverage(items, canon)
    counts = _clause_counts(COMPAT._no_canonical_reason(items, canon))

    assert sum(counts.values()) == cov.n == 5
    assert counts["tested and not assigned to this unit (tested-absent)"] \
        + counts["measured, but too unspecific to carry the call"] == cov.k == 2


def _clause_counts(sentence: str) -> dict[str, int]:
    """``"of the N ...: 2 off-panel ...; 1 ..."`` -> ``{clause: count}``."""
    body = sentence.split(": ", 1)[1].replace(COMPAT._NOTHING_COULD_SUPPORT, "")
    return {c.split(" ", 1)[1]: int(c.split(" ", 1)[0]) for c in body.split("; ")}


# --------------------------------------------------------------------------- #
# Replicate coverage: the second rung that reads the apparatus, not the driver
# --------------------------------------------------------------------------- #
#: A clean call: one strong canonical driver and enough company not to be fragile.
#: Deliberately NOT one of the demo clusters -- the point is a band that nothing
#: else touches, so anything that moves below is this rung and only this rung.
def _clean() -> tuple[EvidenceItem, ...]:
    return (_item("ERBB2", 21.0, 0.5), _item("G2", 9.0, 0.5),
            _item("G3", 8.0, 0.5), _item("G4", 7.0, 0.5))


def _replicate(k: int, n: int) -> Coverage:
    return Coverage(C.REPLICATE_COVERAGE_WHAT, k, n,
                    C._coverage_floor(n, C.REPLICATE_COVERAGE_FLOOR))


def test_replicate_coverage_only_ever_subtracts():
    """§9.10: agreement never raises confidence, so there is no promote to match.

    The grid walks a 12-subject comparison from nothing comparable to everything
    comparable against a fixed items/canonical pair. The band may fall and must
    never rise above the not-assessed baseline, and ``demotions`` must stay empty
    of this rule wherever the floor is cleared -- a decision recorded for a rung
    nobody lost reads in the audit as a check the call survived, which is the one
    reassurance replicates are not allowed to supply.
    """
    items, canon = _clean(), frozenset({"ERBB2"})
    base = C.assess(items, canon).value
    assert base.band == "Very High" and base.verify is False
    assert base.replicate_coverage is None

    floor = C._coverage_floor(12, C.REPLICATE_COVERAGE_FLOOR)
    for k in range(0, 13):
        v = C.assess(items, canon, _replicate(k, 12)).value
        assert C.BANDS.index(v.band) >= C.BANDS.index(base.band), k
        fired = "band:demote_replicate_coverage" in v.demotions
        assert fired is (k < floor), (k, v.demotions)
        assert v.band == ("High" if fired else "Very High"), k
        assert v.verify is fired, k


def test_the_replicate_rung_is_one_step_however_thin_the_coverage():
    """0 of 12 and 5 of 12 both cost exactly one rung.

    Walking the band down by degrees would turn a coverage gap into a weak
    measurement, and it is not one: the driver's statistic is unchanged and the
    rung says only that the comparison was not about this unit.
    """
    items, canon = _clean(), frozenset({"ERBB2"})
    bands = {C.assess(items, canon, _replicate(k, 12)).value.band for k in (0, 1, 5)}
    assert bands == {"High"}


def test_not_assessed_is_not_assessed_and_not_zero_coverage():
    """``None`` and ``Coverage(k=0, n=0)`` are different answers and stay apart.

    The demo is single-sample, so §9.10 resolves to not-assessed on every unit --
    and a ``None`` that behaved like an empty comparison would demote all nine and
    move the golden. An empty comparison (n == 0) is also not a demote: nothing was
    compared because there was nothing to compare, which the record says in words.
    """
    items, canon = _clean(), frozenset({"ERBB2"})
    for replicate in (None, Coverage(C.REPLICATE_COVERAGE_WHAT, 0, 0, 0)):
        v = C.assess(items, canon, replicate).value
        assert v.band == "Very High" and v.verify is False, replicate
        assert "band:demote_replicate_coverage" not in v.demotions, replicate


def test_the_replicate_floor_is_a_majority_and_an_exact_ratio():
    """A majority, by integer arithmetic, with the declared ratio read not assumed.

    The constant is structural rather than calibrated (there is one multi-sample
    study on disk, at 9/12 on every unit), and this pins the two properties that
    must survive the tuning pass that replaces it: K at the floor clears and one
    below does not, at every subject count a unit plausibly has.
    """
    import math
    from fractions import Fraction

    num, den = C.REPLICATE_COVERAGE_FLOOR
    for n in range(0, 200):
        assert C._coverage_floor(n, C.REPLICATE_COVERAGE_FLOOR) == math.ceil(
            Fraction(n * num, den)), n
    for n in range(1, 40):
        f = C._coverage_floor(n, C.REPLICATE_COVERAGE_FLOOR)
        assert Coverage("x", f, n, f).ok
        assert not Coverage("x", f - 1, n, f).ok


# --------------------------------------------------------------------------- #
# A gene frame admits a gene however it is spelled
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "panel_spelling,marker_spelling,case",
    [
        ("CD68", "Cd68", "markers carry the vendor's mixed case"),
        ("CD68 ", "CD68", "the panel file carries a trailing space"),
        ("  cd68", "CD68 ", "both sides are untidy, in opposite ways"),
    ],
)
def test_the_two_engines_admit_the_same_gene_however_it_is_spelled(
    panel_spelling: str, marker_spelling: str, case: str,
) -> None:
    """``Frame.contains`` and ``data.panel_contains`` are one primitive, twice.

    ``agent/`` may not import ``core/``, so v1's absence primitive is a
    transcription of v2's rather than a call to it -- and the two stopped being
    transcriptions when one case-folded without stripping and the other did neither.
    Measured on a constructed store before this was closed: Very High with ``verify``
    False from the agent engine against Medium-High with ``verify`` True from core,
    on one contested row, with the engine the golden reads being the *less*
    conservative of the two.

    This is the input the whole suite was missing: no study anywhere in the repo has
    a ``panel.parquet`` and a ``markers_top.csv`` that disagree about how to spell a
    gene, which is the single input that separates the two implementations.
    """
    from panoscope.types.measured import Frame
    from panoscope.types.subject import Entity, EntityKind

    assert (str(marker_spelling).strip().upper()
            == str(panel_spelling).strip().upper()), "the fixture must be one gene"

    frame = Frame(kind=EntityKind.GENE, name="panel:t",
                  members=frozenset({panel_spelling}))
    assert frame.contains(Entity(EntityKind.GENE, marker_spelling)) is True, case


def test_a_frame_still_refuses_a_gene_that_is_genuinely_absent() -> None:
    """The other direction: normalising identity must not admit a different gene."""
    from panoscope.types.measured import Frame
    from panoscope.types.subject import Entity, EntityKind

    frame = Frame(kind=EntityKind.GENE, name="panel:t", members=frozenset({"CD68"}))
    for other in ("CD69", "CD8A", "CD6", "CD688", "C D68"):
        assert frame.contains(Entity(EntityKind.GENE, other)) is False, other
