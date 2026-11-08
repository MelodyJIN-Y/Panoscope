"""Invariants for the measure step -- the first ``core/`` module.

These pin what the step *guarantees*, not how it computes: the lattice itself is
tested in ``test_types_measured.py``, and re-asserting it here would be a second
copy of the same expectations drifting away from the first.

What is pinned here is the thing only ``core/`` can get wrong: a number surviving
into an item the frame refused, a number vanishing without a decision, and the
order the record puts those decisions in. The last one matters because decision
order is hashed into the golden record, so a "harmless" reshuffle is a
reproducibility break that no behavioural test would notice.
"""

from __future__ import annotations

import pytest

from panoscope.core.measure import Reading, measure, measure_unit, resolve
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.measured import (
    Coverage, Frame, MeasuredState, MeasuredWorld,
)
from panoscope.types.subject import EntityKind, cell_type, gene, gene_set, subject
from panoscope.types.trace import RULES, Traced

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM", "ACTA2"}))
ANNOT = Frame(EntityKind.CELL_TYPE, "annotation:test", frozenset({"Tumor", "Stromal"}))
SETS = Frame(EntityKind.GENE_SET, "collection:test", frozenset({"HALLMARK_EMT"}))
WORLD = MeasuredWorld((PANEL, ANNOT, SETS))

REPORTED_GENES = Frame(EntityKind.GENE, "reported:jazzpanda", frozenset({"ERBB2", "LUM"}))
REPORTED_TYPES = Frame(EntityKind.CELL_TYPE, "reported:jazzpanda", frozenset({"Tumor"}))
REPORTED = frozenset(REPORTED_GENES.members | REPORTED_TYPES.members)

ERBB2 = subject("gene", g=gene("ERBB2"))    # in frame, reported     -> OBSERVED
ACTA2 = subject("gene", g=gene("ACTA2"))    # in frame, not reported -> TESTED_ABSENT
COL1A1 = subject("gene", g=gene("COL1A1"))  # off panel              -> NEVER_MEASURED
TUMOR = subject("cell_type", ct=cell_type("Tumor"))

CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("jazzpanda", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(), ANNOT.ref(), SETS.ref()),
)


def _read(subj, stat, **kw) -> Reading:
    return Reading(subject=subj, statistic=stat, statistic_name="glm_coef", **kw)


# --------------------------------------------------------------------------- #
# The four states, and the number that may exist in exactly one of them
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "subj, state, rule",
    [
        (ERBB2, MeasuredState.OBSERVED, "measure:observed"),
        (ACTA2, MeasuredState.TESTED_ABSENT, "measure:tested_absent"),
        (COL1A1, MeasuredState.NEVER_MEASURED, "measure:never_measured"),
    ],
)
def test_each_state_is_recorded_under_its_own_rule(subj, state, rule) -> None:
    """Four states, four rules. Collapsing two of them onto one id would make the
    UI unable to tell "never looked" from "looked and found nothing" -- which is
    the distinction the whole lattice exists to carry."""
    out = resolve(subj, WORLD, REPORTED)
    assert out.value.state is state
    assert [d.rule for d in out.decisions] == [rule]
    assert out.decisions[0].outcome == state.value


def test_a_number_survives_only_where_the_subject_was_observed() -> None:
    observed = measure(_read(ERBB2, 21.43781), WORLD, REPORTED).value
    assert observed.statistic == 21.43781

    for subj in (ACTA2, COL1A1):
        item = measure(_read(subj, 21.43781), WORLD, REPORTED).value
        assert item.statistic is None, f"{subj.key} kept a number it was refused"


def test_a_withheld_number_is_recorded_rather_than_dropped() -> None:
    """The quiet failure. An item whose statistic vanished with no decision reads
    exactly like a method that reported nothing, so the biologist cannot tell a
    real refusal from a lost measurement."""
    out = measure(_read(COL1A1, 21.43781), WORLD, REPORTED)
    withheld = [d for d in out.decisions if d.outcome == "statistic withheld"]
    assert len(withheld) == 1
    assert withheld[0].rule == "measure:never_measured"
    assert "glm_coef=21.4378" in withheld[0].because[0]
    assert "never_measured" in withheld[0].because[0]


def test_nothing_is_recorded_when_the_method_reported_no_number() -> None:
    """A method that never claimed a number has not been refused one. Emitting a
    withholding there would manufacture a contradiction out of an ordinary absence,
    and every untested gene on the panel would carry a spurious decision."""
    out = measure(_read(ACTA2, None), WORLD, REPORTED)
    assert [d.outcome for d in out.decisions] == [MeasuredState.TESTED_ABSENT.value]


def test_corroborators_are_withheld_with_the_statistic() -> None:
    """``EvidenceItem`` guards ``statistic`` only. ``corroborating`` is a second
    channel for numbers about the same subject, so a pearson left behind would put
    a measurement back into an item the frame says was never measured."""
    out = measure(_read(COL1A1, 21.43781, corroborating=(("pearson", 0.41),)),
                  WORLD, REPORTED)
    assert out.value.corroborating == ()
    assert any("pearson=0.4100" in b for d in out.decisions for b in d.because)


def test_no_decision_ref_carries_a_number_for_an_unobserved_subject() -> None:
    """A ref is a click-back anchor; one reading ``#glm_coef=21.4378`` asserts a
    measurement. For a refused subject that is the project's worst failure wearing
    a provenance string."""
    for subj in (ACTA2, COL1A1):
        out = measure(_read(subj, 21.43781), WORLD, REPORTED)
        refs = [r for d in out.decisions for r in d.refs]
        assert refs and all("#" not in r for r in refs), refs


# --------------------------------------------------------------------------- #
# Coverage, composites, and the frame boundary
# --------------------------------------------------------------------------- #
def test_coverage_demotes_an_otherwise_observed_subject_and_takes_its_number() -> None:
    """K of N below the floor is the enrichment ``untestable`` tier, generalised:
    the set was looked at, the look was too thin to support the number."""
    emt = subject("gene_set", set=gene_set("HALLMARK_EMT"),
                  member=[gene("ERBB2"), gene("COL1A1")])
    out = measure(
        _read(emt, 4.2, coverage=(Coverage("panel_hits", 2, 200, 3),)),
        WORLD, REPORTED | {"HALLMARK_EMT"},
    )
    assert out.value.measured.state is MeasuredState.UNDERPOWERED
    assert out.value.statistic is None
    assert [d.rule for d in out.decisions] == [
        "measure:coverage_floor", "measure:coverage_floor",
    ]
    assert any("panel_hits 2/200 (floor 3)" in b for b in out.decisions[0].because)


def test_a_composite_names_the_constituent_that_blocked_it() -> None:
    """"Unmeasured" on its own is not actionable. The record must say WHICH partner
    was off-panel, or the biologist cannot tell whether a different panel would fix
    it."""
    lr = subject("interaction", ligand=gene("COL1A1"), receptor=gene("ERBB2"),
                 source=cell_type("Tumor"), target=cell_type("Tumor"))
    out = measure(_read(lr, 1.0), WORLD, REPORTED)
    assert out.value.measured.state is MeasuredState.NEVER_MEASURED
    assert [str(e) for e in out.value.measured.blocked_by] == ["gene:COL1A1"]
    assert any("gene:COL1A1" in b for d in out.decisions for b in d.because)


# --------------------------------------------------------------------------- #
# The record: order, registration, determinism
# --------------------------------------------------------------------------- #
def test_the_unit_verdict_precedes_every_item_and_items_keep_row_order() -> None:
    """Decision order is code order and is hashed into the golden record. The unit's
    own verdict explains every untestable item beneath it, so a record that states
    it afterwards reads as an afterthought rather than as the reason."""
    out = measure_unit(
        unit="c1", unit_kind="cluster", unit_subject=TUMOR, module="jazzpanda",
        readings=(_read(ERBB2, 21.43781), _read(COL1A1, 3.0), _read(ACTA2, None)),
        world=WORLD, reported=(REPORTED_GENES, REPORTED_TYPES), conditions=CONDITIONS,
    )
    assert [d.rule for d in out.decisions] == [
        "measure:observed",                                    # the unit, first
        "measure:observed",                                    # ERBB2
        "measure:never_measured", "measure:never_measured",    # COL1A1: state, withheld
        "measure:tested_absent",                               # ACTA2
    ]
    assert [i.subject.key for i in out.value.items] == [
        ERBB2.key, COL1A1.key, ACTA2.key,
    ]
    assert out.value.reported == (REPORTED_GENES, REPORTED_TYPES)
    assert out.value.unit_measured.state is MeasuredState.OBSERVED


def test_the_unit_can_be_underpowered_while_its_items_are_fine() -> None:
    """"Too few cells in this unit" is a property of the design, not of any marker
    in it. Pushed down onto the items it would be duplicated once per item, and two
    items could then disagree about the same unit."""
    out = measure_unit(
        unit="c9", unit_kind="cluster", unit_subject=TUMOR, module="jazzpanda",
        readings=(_read(ERBB2, 21.43781),), world=WORLD,
        reported=(REPORTED_GENES, REPORTED_TYPES), conditions=CONDITIONS,
        unit_coverage=(Coverage("cells_in_unit", 2, 200, 30),),
    )
    assert out.value.unit_measured.state is MeasuredState.UNDERPOWERED
    assert out.value.items[0].statistic == 21.43781
    assert out.decisions[0].rule == "measure:coverage_floor"


def test_every_emitted_rule_is_registered_and_belongs_to_the_measure_step() -> None:
    """The static scan in ``test_contracts`` only sees literal ids. This is the
    runtime half: what the module actually emits over a mixed unit."""
    out = measure_unit(
        unit="c1", unit_kind="cluster", unit_subject=TUMOR, module="jazzpanda",
        readings=(_read(ERBB2, 21.43781), _read(COL1A1, 3.0), _read(ACTA2, None)),
        world=WORLD, reported=(REPORTED_GENES, REPORTED_TYPES), conditions=CONDITIONS,
    )
    for d in out.decisions:
        assert d.rule in RULES
        assert d.step == "measure"


def test_the_same_input_produces_byte_identical_decisions() -> None:
    """``because`` strings are hashed into the golden record. Anything that varies
    run to run -- a set iteration, a clock, a dict order -- breaks the
    reproducibility claim rather than merely a test."""
    args = (_read(COL1A1, 21.43781, corroborating=(("pearson", 0.41), ("gc", 0.2))),
            WORLD, REPORTED)
    first, second = measure(*args), measure(*args)
    assert first.decisions == second.decisions
    assert isinstance(first, Traced)


def test_the_unit_is_observed_even_when_no_reported_frame_names_it() -> None:
    """A unit the method emitted rows for is reported by construction; reading it
    as TESTED_ABSENT would call the cluster's own measurement absence evidence."""
    stromal = subject("cell_type", ct=cell_type("Stromal"))  # in frame, never reported
    out = measure_unit(
        unit="c2", unit_kind="cluster", unit_subject=stromal, module="jazzpanda",
        readings=(_read(ERBB2, 1.0),), world=WORLD, reported=(REPORTED_GENES,),
        conditions=CONDITIONS,
    )
    assert out.value.unit_measured.state is MeasuredState.OBSERVED
