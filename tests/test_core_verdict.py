"""Invariants for the confidence band in ``core/``.

The band is the single most load-bearing number this project produces: it is pinned
by ``tests/test_golden.py``, written into a committed 11-column CSV that downstream R
reads, and rendered as the pill the biologist acts on. So the central test here is not
an inspection of the new code -- it feeds the SAME demo evidence through
``agent.verdict`` and through ``core.verdict`` and asserts band, basis, ``verify`` and
the ordered adjustments are equal, cluster by cluster. "Looks right" is what lets a
reordered demote or an absorbed promote through.

The two implementations speak different vocabularies on purpose: ``agent`` returns a
tuple of English sentences, ``core`` returns registered rule ids. ``_RULE_FOR``
translates, and ``test_the_translation_table_is_exhaustive`` fails if ``agent`` ever
emits a sentence the table does not know -- otherwise an unmapped adjustment would
silently compare as "no adjustment on either side".

Synthetic items cover what the demo study does not contain: a tested-absent canonical
driver, a cap, a floor, and a promote at the top of the ladder. Those are built by
hand rather than by perturbing the real study, because a fixture that mutates the demo
tree is the fixture that breaks the golden hash for everyone else on the branch.
"""

from __future__ import annotations

import inspect

import pytest

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import verdict as A
from panoscope.core import verdict as C
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Measured, MeasuredState
from panoscope.types.subject import gene, subject
from panoscope.types.trace import RULES

_FRAME = "panel:xenium_hbc_v1"

# agent's English -> core's registered rule id. The prose halves are verbatim from
# ``agent/verdict.py``; the numbers it interpolates are stripped by prefix matching,
# which is why the two pearson entries are prefixes and not whole strings.
_RULE_FOR: tuple[tuple[str, str], ...] = (
    ("no canonical marker supports the call", "band:no_canonical_driver"),
    ("driving marker is NoSig", "band:driver_nosig"),
    ("high pearson on driver", "band:promote_pearson_high"),
    ("low pearson on driver", "band:demote_pearson_low"),
    ("driver localizes better with another cluster", "band:demote_localises_elsewhere"),
    ("driver spatial pattern not unique", "band:demote_not_unique"),
    ("fragile cluster (<=2 markers): capped", "band:fragile_cap"),
    ("fragile cluster with canonical driver: floored", "band:fragile_floor"),
)

# agent's caveat prose (``_caveats_for``) -> the registered flag the method layer
# carries on an EvidenceItem. This is the transcription ``evidence.py`` FLAGS records.
_FLAG_FOR: dict[str, str] = {
    "localizes better with another cluster": C.FLAG_LOCALISES_ELSEWHERE,
    "spatial pattern not unique": C.FLAG_NOT_UNIQUE,
    "low spatial specificity": C.FLAG_LOW_PEARSON,
}


def _rule_for(sentence: str) -> str:
    for prefix, rule in _RULE_FOR:
        if sentence.startswith(prefix):
            return rule
    raise AssertionError(
        f"agent emitted an adjustment the table does not map: {sentence!r}"
    )


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


def _items_for(cluster: str, cell_type: str) -> tuple[EvidenceItem, ...]:
    """The demo cluster's rows, as ``EvidenceItem``s.

    Built from ``agent.verdict._build_evidence`` so that both implementations are
    demonstrably reading the same numbers: if this test re-read the CSV itself, an
    equivalence failure could not be told apart from a fixture that drifted.
    """
    return tuple(
        EvidenceItem(
            subject=subject("gene", g=gene(e.gene)),
            statistic=e.glm_coef,
            statistic_name="glm_coef",
            measured=Measured(MeasuredState.OBSERVED, _FRAME),
            corroborating=(("pearson", e.pearson),
                           ("max_gc_corr", e.max_gc_corr),
                           ("max_gg_corr", e.max_gg_corr)),
            flags=tuple(_FLAG_FOR[c] for c in e.caveats),
        )
        for e in A._build_evidence(cluster, cell_type)
    )


def _inputs(cluster: str):
    """(items, canonical) for one demo cluster -- core's two arguments."""
    cell_type = annotation.meta_for(cluster)["cell_type"]
    return _items_for(cluster, cell_type), frozenset(
        annotation.canonical_markers(cell_type)
    )


# --------------------------------------------------------------------------- #
# Behavioural equivalence -- the whole contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cluster", list(cfg.CLUSTER_ORDER))
def test_core_band_equals_agent_band_on_every_demo_cluster(cluster):
    """Band, basis, verify and the ORDERED adjustments, field by field.

    Order matters independently of the band: each demote is one rung, so the band
    survives a reordering while the record -- and the badges the record draws -- do
    not.
    """
    av = A.verdict_for_cluster(cluster)
    cv = C.assess(*_inputs(cluster)).value
    assert cv.band == av.confidence, f"{cluster}: band {cv.band!r} != {av.confidence!r}"
    assert cv.basis == av.band_basis, f"{cluster}: basis {cv.basis!r} != {av.band_basis!r}"
    assert cv.verify == av.verify, f"{cluster}: verify {cv.verify} != {av.verify}"
    assert cv.demotions == tuple(_rule_for(d) for d in av.demotions), (
        f"{cluster}: adjustments {cv.demotions} != {av.demotions}"
    )
    assert cv.fragile == av.small_n, f"{cluster}: fragile {cv.fragile} != {av.small_n}"


def _grid():
    """The reachable (coef, pearson, max_gc, max_gg, fragile) corners.

    ``pearson`` never goes below ``LOW_PEARSON``: below it a canonical marker is
    disqualified as driver rather than demoted, so ``_compute_band`` is never called
    with one by the real pipeline. That one structural difference has its own test
    (``test_a_low_pearson_canonical_marker_is_disqualified_not_demoted``); including
    it here would compare the two implementations outside the domain either of them
    actually runs on.
    """
    for coef in (0.5, 1.5, 3.0, 6.0, 12.0, 30.0):
        for pearson in (0.30, 0.50, 0.86, 0.99):
            for gc in (0.0, pearson + 0.10):           # localises elsewhere, or not
                for gg in (0.50, 0.99):                # pattern shared, or not
                    for fragile in (False, True):
                        yield coef, pearson, gc, gg, fragile


@pytest.mark.parametrize("coef,pearson,gc,gg,fragile", list(_grid()))
def test_core_matches_the_agent_band_function_across_the_reachable_grid(
    coef, pearson, gc, gg, fragile
):
    """The demo fires at most ONE adjustment per cluster, so the nine-cluster sweep
    cannot see the order of two demotes, nor a cap, nor promote-then-demote. This
    grid does: it drives ``agent.verdict._compute_band`` and ``core.assess`` from one
    set of numbers and compares band, basis and ordered adjustments at every corner.

    It is the test that caught the real gap -- swapping the two demote branches in
    ``core`` left the nine-cluster sweep entirely green, because no demo cluster ever
    carries both flags.
    """
    caveats = A._caveats_for(pearson, gc, gg)
    me = A.MarkerEvidence(
        gene="ERBB2", top_cluster="c1", glm_coef=coef, pearson=pearson,
        max_gg_corr=gg, max_gc_corr=gc, p_value=None, within_cluster_pctile=1.0,
        is_canonical=True, is_on_panel=True, role="supports", caveats=caveats,
    )
    a_band, a_basis, a_changes = A._compute_band(me, fragile=fragile)

    driver = _item("ERBB2", coef, pearson, tuple(_FLAG_FOR[c] for c in caveats))
    # fragile is len(assigned) <= 2, so it is set by how many items carry a number.
    filler = () if fragile else tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    cv = C.assess((driver, *filler), frozenset({"ERBB2"})).value

    assert cv.fragile is fragile
    assert cv.band == a_band
    assert cv.basis == a_basis
    assert cv.demotions == tuple(_rule_for(c) for c in a_changes)


def test_the_translation_table_is_exhaustive():
    """An unmapped sentence would compare as 'no adjustment' on both sides, and the
    equivalence test above would pass while the two implementations disagreed."""
    for cluster in cfg.CLUSTER_ORDER:
        for sentence in A.verdict_for_cluster(cluster).demotions:
            assert _rule_for(sentence) in RULES


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


def test_thresholds_match_the_live_config():
    """``core/`` may not import ``agent/``, so the ladder is duplicated. This is the
    drift alarm; without it the two bands diverge silently at the next tuning pass."""
    assert C.CONFIDENCE_BANDS == cfg.CONFIDENCE_BANDS
    assert C.BANDS == A.BANDS
    assert C.PROMOTE_PEARSON == A.PROMOTE_PEARSON
    assert C.LOW_PEARSON == A.LOW_PEARSON
    assert C.FRAGILE_MARKER_COUNT == A.FRAGILE_MARKER_COUNT
    assert C.FRAGILE_FLOOR_BAND == A.FRAGILE_FLOOR_BAND
    for coef in (-12.0, 0.0, 0.99, 1.0, 2.5, 5.0, 9.999, 10.0, 21.4):
        assert C._band_for_coef(coef) == cfg.band_for_coef(coef), coef
    assert C._band_for_coef(None) == cfg.band_for_coef(None)


# --------------------------------------------------------------------------- #
# verify comes from the rubric, never from a caller
# --------------------------------------------------------------------------- #
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


def test_verify_fires_on_each_of_its_four_conditions():
    canonical = frozenset({"ERBB2", "KRT7", "EPCAM", "FOXA1"})
    strong = tuple(_item(g, 20.0, 0.5) for g in ("ERBB2", "KRT7", "EPCAM", "FOXA1"))

    assert C.assess(strong, canonical).value.verify is False            # the control
    assert C.assess(strong[:2], canonical).value.verify is True         # fragile
    assert C.assess((_item("ACTA2", 20.0, 0.5),) * 4,                   # no driver
                    canonical).value.verify is True
    low = C.assess(tuple(_item(g, 0.4, 0.5) for g in canonical), canonical)
    assert low.value.band == "Low" and low.value.verify is True         # Low band


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


def test_a_tested_absent_canonical_marker_is_the_nosig_branch():
    """NoSig means in-panel, tested, not assigned here -- which is TESTED_ABSENT. It
    reads Low like 'no canonical driver', but it is a different answer: the marker WAS
    measured, so its absence argues, and the record has to say which happened."""
    v = C.assess(
        (_item("ERBB2", None, state=MeasuredState.TESTED_ABSENT),
         _item("ACTA2", 30.0, 0.9), _item("PTPRC", 20.0, 0.9), _item("CD3E", 9.0, 0.9)),
        frozenset({"ERBB2"}),
    ).value
    assert v.band == "Low"
    assert v.basis == "driver NoSig"
    assert v.demotions == ("band:driver_nosig",)
    assert v.verify is True


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
