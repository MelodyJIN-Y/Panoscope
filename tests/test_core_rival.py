"""Invariants for the rival test, and the one place it deliberately disagrees with v1.

Two jobs, and the second is the point of the module.

**Equivalence.** ``agent/discriminate.py`` is the shipped discriminator, read by
``agent/skeptic.py`` and rendered in the review surface. Replacing its private status
vocabulary with the four-state lattice must not quietly change what the demo study
says, so the first test drives it and ``rival_test`` over the same clusters and the
same reference lists and asserts the buckets agree **element for element, in order**
-- not as sets, because decision and render order are hashed into the record.

Since the cutover ``discriminate.py`` keeps its published surface and calls
``rival_test`` as its engine, so that test now guards the ADAPTER rather than two
engines. It is the honest reading of it and it is said in its own docstring: an
equivalence test that compares an implementation to itself proves nothing it does not
admit to.

**Divergence.** The one place it disagrees with the retired bucketing, and the reason
for the task. That engine asked one capability question, ``panel_contains``, so every
on-panel gene that was not a top marker of this unit counted as evidence *against* the
rival -- including the 49 of 280 demo panel genes that produce no usable spatial
signal. ``test_an_undetectable_rival_no_longer_argues`` drives a rival whose canonical
markers are all such genes: it used to rule the rival out on five genes the
measurement never usably saw, which is the panel-absence rule inverted into a
confident false negative, and it now reports all five as undecidable and refuses.

The reference lists (canonical markers) are inputs here, as they are to both
implementations. The NK set is five real HGNC symbols that are genuinely on the demo
panel and genuinely ``NoSig`` in the committed ``markers_top.csv`` -- the test asserts
that before using them, so the case cannot rot into a fiction if the fixture changes.
Nothing about any measurement is invented; reference data selects which subjects are
asked about and supplies no value, here as in the module.
"""

from __future__ import annotations

import pytest

from panoscope.agent import annotation, data
from panoscope.agent import discriminate as v1
from panoscope.core.measure import Reading, measure_unit
from panoscope.core.rival import rival_test
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.measured import Frame, MeasuredState, MeasuredWorld
from panoscope.types.subject import EntityKind, cell_type, gene, subject
from panoscope.types.trace import RULES

PANEL_FRAME = "panel:demo"
DETECT_FRAME = "detect:nosig"

# Real NK canonical markers. Every one is on the demo panel and every one is NoSig,
# which ``test_the_divergence_case_is_real`` re-checks against the committed file.
NK = ("GNLY", "PRF1", "KLRF1", "NCAM1", "KLRC1")
# Deliberately mixed: one undetectable, one tested-and-localised-elsewhere, one
# off-panel. Three states, one rival, all three statuses read off the real study.
MIXED = ("GNLY", "KLRD1", "GZMB")


@pytest.fixture
def world() -> MeasuredWorld:
    """The demo study's capability frames.

    Two gene frames, not one: the panel says what could be looked at and
    ``detect:nosig`` says what looking at it produced. ARCHITECTURE §9.5 puts
    detectability here rather than in a special case, and splitting them is what lets
    ``Measured.frame`` name *which* one refused a marker.

    Panel membership is derived from the file. A hardcoded 280 would turn the second
    study's immuno-oncology panel into 280 genes that are not there.
    """
    markers = data.load_markers()
    return MeasuredWorld((
        Frame(EntityKind.GENE, PANEL_FRAME,
              frozenset(data.load_panel()["gene"].astype(str))),
        Frame(EntityKind.GENE, DETECT_FRAME,
              frozenset(markers[markers["top_cluster"] != "NoSig"]["gene"].astype(str))),
        Frame(EntityKind.CELL_TYPE, "annotation:cluster_key",
              frozenset(annotation.all_canonical())),
    ))


def _conditions(world: MeasuredWorld) -> Conditions:
    return Conditions(
        assay=Condition("xenium", Provenance.READ),
        modality=Condition("spatial_transcriptomics", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=Condition("jazzpanda", Provenance.STATED),
        segmentation=UNKNOWN, fixation=UNKNOWN,
        frames=tuple(f.ref() for f in world.frames),
    )


def _evidence(cluster: str, world: MeasuredWorld):
    """One cluster's jazzPanda output, through the real measure step.

    jazzPanda assigns each gene to a single winning cluster, so a unit's readings are
    exactly its own top markers and every other panel gene is tested-and-not-reported
    *here*. That is the data fact v1's docstring describes and the lattice encodes.
    """
    call = data.cell_type_for(cluster)
    rows = data.get_cluster_markers(cluster)
    readings = tuple(
        Reading(subject=subject("gene", g=gene(str(r.gene))),
                statistic=float(r.glm_coef), statistic_name="glm_coef",
                corroborating=(("pearson", float(r.pearson)),))
        for r in rows.itertuples()
    )
    reported = (
        Frame(EntityKind.GENE, f"reported:{cluster}",
              frozenset(str(r.gene) for r in rows.itertuples())),
        Frame(EntityKind.CELL_TYPE, f"reported:{cluster}", frozenset({call})),
    )
    return measure_unit(
        unit=cluster, unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type(call)), module="jazzpanda",
        readings=readings, world=world, reported=reported,
        conditions=_conditions(world),
    ).value


def _symbols(items) -> list[str]:
    return [i.subject.one("g").id for i in items]


# --------------------------------------------------------------------------- #
# Equivalence with the implementation this replaces
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cluster", ["c3", "c4", "c7"])
def test_the_buckets_match_the_v1_discriminator(cluster: str, world) -> None:
    """The three demo clusters where a rival is defined at all, bucket for bucket.

    This WAS a two-engine comparison. Since the cutover ``agent/discriminate.py`` is a
    thin surface over ``rival_test``, so what it now proves is narrower and still
    worth having: the adapter in ``discriminate.py`` maps every bucket across --
    ``supports_call``/``supports_rival``/``against_rival``/``undecidable`` ->
    ``supporting_A``/``b_here``/``b_elsewhere``/``offpanel_absent``, and ``settleable``
    -> ``settleable_on_panel`` -- without dropping or reordering an item. Content *and*
    order: the record hashes decision order and the review surface reads markers
    top-down.

    v1 derives the rival from the unit's own off-type markers and ``core.rival`` takes
    it as an argument, which is why the derived rival is handed across here.
    """
    old = v1.discriminate(cluster)
    assert old.alt_B is not None, f"{cluster} has no derived rival to compare on"

    out = rival_test(
        _evidence(cluster, world), old.call_A, old.alt_B,
        annotation.canonical_markers(old.call_A),
        annotation.canonical_markers(old.alt_B), world,
    )
    new = out.value

    assert _symbols(new.supports_call) == [m.gene for m in old.supporting_A]
    assert _symbols(new.supports_rival) == [m.gene for m in old.b_here]
    assert _symbols(new.against_rival) == [m.gene for m in old.b_elsewhere]
    assert _symbols(new.undecidable) == [m.gene for m in old.offpanel_absent]
    assert new.settleable is old.settleable_on_panel


@pytest.mark.parametrize("cluster", ["c3", "c4", "c7"])
def test_the_numbers_still_come_only_from_this_units_own_markers(cluster: str, world) -> None:
    """v1's hard rule, now structural rather than conventional: a jazzPanda number is
    attached only to a gene that is a top marker of the cluster in question.

    v1 enforced it by passing ``glm_coef=None`` at three call sites. Here the state
    carries it -- ``EvidenceItem`` raises if a statistic and a status disagree -- so
    the rule holds for buckets nobody has written yet.
    """
    old = v1.discriminate(cluster)
    new = rival_test(
        _evidence(cluster, world), old.call_A, old.alt_B,
        annotation.canonical_markers(old.call_A),
        annotation.canonical_markers(old.alt_B), world,
    ).value
    for item in (*new.supports_call, *new.supports_rival):
        assert item.statistic is not None
        assert item.measured.state is MeasuredState.OBSERVED
    for item in (*new.against_rival, *new.undecidable):
        assert item.statistic is None
        assert item.corroborating == ()


def test_a_rival_with_no_signal_here_is_argued_against(world) -> None:
    """The productive half. c1 is Tumor; every canonical T-cell marker on the panel
    was measured and peaked in another cluster, so the absence here *is* evidence and
    the rival is ruled out -- while CD3D, never measured, is reported alongside rather
    than folded into the ruling.

    v1 reached the same verdict; ``agent/skeptic.py``'s ``elif`` chain then dropped
    CD3D on the floor, because its off-panel branch runs only when nothing localised
    elsewhere. Here both survive into one result.
    """
    out = rival_test(_evidence("c1", world), "Tumor", "T_Cells",
                     annotation.canonical_markers("Tumor"),
                     annotation.canonical_markers("T_Cells"), world)
    assert out.value.supports_rival == ()
    assert _symbols(out.value.against_rival) == ["IL7R", "PTPRC", "TRAC", "CD3E",
                                                 "CD8A", "CD4"]
    assert _symbols(out.value.undecidable) == ["CD3D"]
    assert out.decisions[-1].rule == "rival:argued_against"
    assert any("CD3D" in b for b in out.decisions[-1].because), (
        "the never-measured marker vanished from the verdict's reasoning"
    )


def test_rival_signal_here_beats_a_ruling_out(world) -> None:
    """One canonical rival marker observed here makes the unit ambiguous, whatever
    the others did. v1's prose ladder put ``b_here`` first for the same reason:
    "mixed signal, re-check this" outranks "leans A"."""
    out = rival_test(_evidence("c3", world), "Macrophages", "T_Cells",
                     annotation.canonical_markers("Macrophages"),
                     annotation.canonical_markers("T_Cells"), world)
    assert _symbols(out.value.supports_rival) == ["CD4"]
    assert out.value.against_rival, "the tested-absent markers were lost"
    assert out.decisions[-1].rule == "rival:ambiguous"


# --------------------------------------------------------------------------- #
# The divergence: an undetectable marker is not a measured absence
# --------------------------------------------------------------------------- #
def test_the_divergence_case_is_real() -> None:
    """Guard the fixture the divergence rests on. If these five genes ever stop being
    on-panel-and-NoSig, the test below would still pass while demonstrating nothing."""
    for symbol in NK:
        assert data.panel_contains(symbol), f"{symbol} is no longer on the demo panel"
        assert str(data.get_marker(symbol)["top_cluster"]) == "NoSig", (
            f"{symbol} now carries spatial signal; pick another divergence case"
        )


@pytest.fixture
def nk_is_a_known_type(monkeypatch):
    """Put NK cells in the reference vocabulary so BOTH implementations can be asked.

    v1 refuses a cell type that has no canonical marker set, so without this it would
    decline the question rather than answer it wrongly -- and the point is that it
    answers it wrongly. The markers are real; only their presence in the demo study's
    annotation is supplied, which is reference data doing the one job reference data
    is allowed to do.
    """
    merged = {**annotation.all_canonical(), "NK_Cells": NK}
    monkeypatch.setattr(annotation, "all_canonical", lambda dataset_id=None: merged)
    monkeypatch.setattr(annotation, "canonical_markers",
                        lambda ct, dataset_id=None: merged.get(ct, ()))
    return merged


def test_an_undetectable_rival_no_longer_argues(world, nk_is_a_known_type) -> None:
    """**The reason this module exists.**

    All five canonical NK markers are on the demo panel and none of them produced
    usable spatial signal. v1 sees ``panel_contains`` return True five times, files
    them under "localizes elsewhere" with ``top_cluster='NoSig'``, and rules NK out:

        "NK Cells markers are on the panel but GNLY, PRF1, KLRF1, NCAM1, KLRC1
         (not significant in any cluster) - that argues against NK Cells.
         On the measured markers this leans Macrophages."

    A gene the measurement could not localize anywhere cannot argue that the lineage
    is absent *here*. Under the detectability frame all five are NEVER_MEASURED, the
    rival's canonical coverage is 0 of 5, and the honest answer is that the panel
    cannot settle it. This is a correction, not a regression: the old answer was
    confidently wrong, which is worse than a refusal.
    """
    # The shipped discriminator now routes through ``rival_test``:
    # ``agent/discriminate.py`` keeps its surface and ``core.rival`` is the engine, so
    # the correction reaches the review surface rather than sitting beside it. The old
    # answer is quoted above and recorded in the commit that performed the cutover; no
    # shipped path produces it any more, which is the point.
    old = v1.discriminate("c3", "NK_Cells")
    assert old.settleable_on_panel is False                    # the panel cannot decide
    assert old.b_elsewhere == (), "an undetectable marker argued against the rival"
    assert [m.gene for m in old.offpanel_absent] == list(NK)
    # on the panel, so this is NOT off-panel news -- the half ``off_panel: bool`` lost
    assert all(m.on_panel is True for m in old.offpanel_absent)
    summary = v1.settle_summary(old)
    assert "argues against NK Cells" not in summary
    assert "produced no usable spatial signal" in summary
    assert "cannot settle" in summary
    # and the trace says which frame refused them. "off_panel=True" here would be a
    # fabricated fact about the panel, which carried all five.
    assert not [s for s in old.source_trace if "off_panel=True" in s]
    assert {s for s in old.source_trace if s.startswith("detect:")} == {
        f"detect:{g}:no_usable_signal=True" for g in NK
    }

    out = rival_test(_evidence("c3", world), "Macrophages", "NK_Cells",
                     annotation.canonical_markers("Macrophages"), NK, world)
    new = out.value
    assert new.against_rival == (), "an undetectable marker argued against the rival"
    assert _symbols(new.undecidable) == list(NK)
    assert all(i.measured.state is MeasuredState.NEVER_MEASURED for i in new.undecidable)
    assert all(i.measured.frame == DETECT_FRAME for i in new.undecidable)
    assert new.settleable is False
    assert (new.coverage.k, new.coverage.n) == (0, 5)
    assert out.decisions[-1].rule == "rival:undecidable"


def test_the_record_names_which_frame_refused_each_marker(world) -> None:
    """The half ``off_panel: bool`` could not express. Off-panel and on-panel-but-
    undetectable are both NEVER_MEASURED and are not the same news: the first is fixed
    by ordering a different panel, the second is not. One rival, three states."""
    out = rival_test(_evidence("c3", world), "Macrophages", "NK_Cells",
                     annotation.canonical_markers("Macrophages"), MIXED, world)
    frames = {i.subject.one("g").id: (i.measured.state, i.measured.frame)
              for i in (*out.value.against_rival, *out.value.undecidable)}
    assert frames["GNLY"] == (MeasuredState.NEVER_MEASURED, DETECT_FRAME)
    assert frames["GZMB"] == (MeasuredState.NEVER_MEASURED, PANEL_FRAME)
    assert frames["KLRD1"][0] is MeasuredState.TESTED_ABSENT
    assert _symbols(out.value.against_rival) == ["KLRD1"]
    # and the record SAYS which frame, once, in the decision that adjudicated it
    assert {b for d in out.decisions for b in d.because} >= {
        "gene:GNLY never_measured in detect:nosig",
        "gene:GZMB never_measured in panel:demo",
    }


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def test_every_marker_gets_a_measure_decision_before_the_verdict(world) -> None:
    """Decision order is code order and is hashed. Each canonical rival marker the
    unit did not report is adjudicated first, in reference-list order, and the
    rivalry's own verdict comes last -- so the record reads as the reasoning rather
    than as the conclusion with notes attached."""
    out = rival_test(_evidence("c1", world), "Tumor", "T_Cells",
                     annotation.canonical_markers("Tumor"),
                     annotation.canonical_markers("T_Cells"), world)
    assert [d.rule for d in out.decisions] == [
        "measure:tested_absent", "measure:tested_absent", "measure:tested_absent",
        "measure:tested_absent",                            # IL7R PTPRC TRAC CD3E
        "measure:never_measured",                           # CD3D, off panel
        "measure:tested_absent", "measure:tested_absent",   # CD8A, CD4 (peaks in c3)
        "rival:argued_against",
    ]
    for d in out.decisions:
        assert d.rule in RULES
    assert out.decisions[-1].step == "rival"


def test_no_decision_ref_carries_a_number(world) -> None:
    """A ref reading ``#glm_coef=...`` asserts a measurement. The rivalry's verdict
    reads subjects, not values, and most of them have no value to read."""
    out = rival_test(_evidence("c3", world), "Macrophages", "T_Cells",
                     annotation.canonical_markers("Macrophages"),
                     annotation.canonical_markers("T_Cells"), world)
    refs = [r for d in out.decisions for r in d.refs]
    assert refs and all("#" not in r for r in refs), refs


def test_the_same_input_produces_byte_identical_decisions(world) -> None:
    """``because`` strings are hashed into the golden record, and this module builds
    them from frozensets and dict lookups -- the two places iteration order leaks."""
    args = (_evidence("c4", world), "Myoepithelial", "Mast_Cells",
            annotation.canonical_markers("Myoepithelial"),
            annotation.canonical_markers("Mast_Cells"), world)
    assert rival_test(*args).decisions == rival_test(*args).decisions
