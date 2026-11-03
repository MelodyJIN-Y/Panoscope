"""The DE workflow, against a SYNTHETIC fixture -- and the fixture says so.

**There is no DE artifact in this repository and no second study.** Every number
below was written by hand for this file: the panel is four genes, the fold changes
are round, and nothing here came out of an analysis. That is stated at the top
because the alternative -- a plausible-looking CSV in ``data/`` -- would be a
fabricated result, which is the worst failure this project can produce. A workflow
registered and demonstrated against a *declared* synthetic fixture is an honest
deliverable; a fake dataset is not.

What the fixture is built to demonstrate, in ARCHITECTURE §9.7's order:

1. **panel scoping** -- the result is panel-scoped, K of N stated, and the canonical
   genes for the affected process that were never on the panel are NAMED.
2. **the replicate refusal** -- one biological replicate, so the condition-level
   claim *cannot be settled*, and the refusal names the replicate that is missing.
   §9.7 calls this the clearest demonstration of the refusal behaviour, and it is the
   test to read first.
3. **composition versus expression** -- the same genes with this unit's abundance
   moving across arms, and with it held still.
4. **cross-method coherence** -- run through ``core/reconcile.py``, with a genuine
   TENSION and a could-not-be-compared side by side in one result, because those are
   the two outcomes a boolean would render identically.

Plus the two invariants that are about the shape rather than the content: §9.4's
rubric divergence (``glm_coef`` 3.0 and ``avg_log2FC`` 3.0 must NOT read the same),
and the no-band guard for a COMPARISON, checked as a field name rather than as a
behaviour -- a behavioural test only catches a band that is filled in, while the
field guard fails the moment the field is declared, which is when it costs nothing
to undo.

**The pair IS registered globally now**, which it was not when this file was
written: ``panoscope/workflows/__init__.py`` registers ``de.WORKFLOW`` and
``panoscope/methods/__init__.py`` registers both ``edger`` and ``seurat_de``. That
landed with the first DE artifact -- a declared-synthetic table in
``synthetic_multisample_fixture`` -- and the composition root that runs it,
``pipeline/stages/de.py``. Tests that need a registry WITHOUT the pair pop the
entries inside :func:`isolated_registries`, which snapshots and restores both.
``tests/test_registries.py`` no longer uses ``de-interpretation`` as its
unregistered example, for the obvious reason.

Run the printed demonstrations with ``pytest -s -k demonstration``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from panoscope import methods, workflows
from panoscope.core.interpret import interpret
from panoscope.core.reconcile import Agreement, Relationship, Side, reconcile
from panoscope.methods import edger, jazzpanda, seurat_de
from panoscope.types.claim import Authority
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import MethodModule, UnrecognisedResult, Workflow
from panoscope.types.evidence import Direction, EvidenceItem, NormalisedEvidence
from panoscope.types.measured import (
    Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.outputs import AssignmentIsNotAFinding, Finding, TaskKind, Tier
from panoscope.types.subject import EntityKind, gene, subject
from panoscope.types.subject import unit as unit_entity
from panoscope.types.trace import RULES
from panoscope.types.uispec import (
    DegradedUpward, HeaderKind, LabelBandHeader, UISpec,
)
from panoscope.workflows import de

# --------------------------------------------------------------------------- #
# THE SYNTHETIC FIXTURE. Four genes, two clusters, one sample. Invented.
# --------------------------------------------------------------------------- #
#: The DE study's panel. Four genes, so "K of N" has a small, checkable N.
PANEL = Frame(EntityKind.GENE, "panel:synthetic_de_v1",
              frozenset({"ERBB2", "MKI67", "ACTA2", "LUM"}))
#: The annotation frame: the unit capability frame for this study.
CLUSTERS = Frame(EntityKind.UNIT, "annotation:synthetic", frozenset({"c1", "c2"}))
WORLD = MeasuredWorld((PANEL, CLUSTERS))

#: The OTHER method's panel. It carries CD3E, which the DE panel does not -- which
#: is what makes "could not be compared" nameable on one side only.
SPATIAL_PANEL = Frame(EntityKind.GENE, "panel:synthetic_spatial_v1",
                      frozenset({"ERBB2", "MKI67", "ACTA2", "LUM", "CD3E"}))
SPATIAL_WORLD = MeasuredWorld((SPATIAL_PANEL, CLUSTERS))

CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("seurat_de", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(), CLUSTERS.ref()),
)

#: A Seurat ``FindAllMarkers`` table, by hand. CD3E is off this panel on purpose:
#: the panel-absence rule has to reach from a CSV row all the way to a refusal.
FINDALLMARKERS_CSV = (
    "gene,cluster,avg_log2FC,p_val,p_val_adj,pct.1,pct.2\n"
    "ERBB2,c1,3.0,1.0e-30,1.0e-26,0.95,0.20\n"
    "CD3E,c1,2.2,1.0e-12,1.0e-09,0.60,0.15\n"
    "LUM,c2,-1.4,1.0e-08,1.0e-05,0.70,0.30\n"
)


class _Adapter:
    """The smallest thing satisfying what ``load`` uses of a PlatformAdapter.

    ``samples`` is the member under test: the sample count lives first-hand on the
    adapter and ``Conditions`` deliberately cannot express it, so this is the only
    place the replicate refusal can get its number from.
    """

    name = "synthetic"

    def __init__(self, samples: tuple[str, ...] = ("sample1",)) -> None:
        self._samples = list(samples)

    def frames(self, store: Any) -> tuple[Frame, ...]:
        return (PANEL, CLUSTERS)

    def samples(self, store: Any) -> list[str]:
        return list(self._samples)

    def standing_conditions(self, store: Any) -> Conditions:
        return CONDITIONS


@pytest.fixture
def isolated_registries():
    """Snapshot both registries and restore them, whatever the test did."""
    m, w = dict(methods.METHODS), dict(workflows.WORKFLOWS)
    try:
        yield
    finally:
        methods.METHODS.clear()
        methods.METHODS.update(m)
        workflows.WORKFLOWS.clear()
        workflows.WORKFLOWS.update(w)


@pytest.fixture
def loaded(tmp_path):
    """The synthetic table, through ``load``, with ONE sample. Traced."""
    csv = tmp_path / "findallmarkers.csv"
    csv.write_text(FINDALLMARKERS_CSV)
    return seurat_de.MODULE.load(str(csv), _Adapter(), object())


def _item(symbol: str, stat: float | None, state: MeasuredState,
          statistic_name: str = "avg_log2FC",
          frame: str = PANEL.name,
          direction: Direction | None = None) -> EvidenceItem:
    return EvidenceItem(
        subject=subject("gene", g=gene(symbol)),
        statistic=stat,
        statistic_name=statistic_name,
        measured=Measured(state, frame=frame),
        direction=direction,
    )


def _evidence(*items: EvidenceItem, module: str = "seurat_de", unit: str = "c1",
              unit_measured: Measured | None = None) -> NormalisedEvidence:
    return NormalisedEvidence(
        unit=unit, unit_kind="cluster",
        unit_subject=subject("unit", u=unit_entity(unit)),
        unit_measured=unit_measured or Measured(MeasuredState.OBSERVED,
                                                frame=CLUSTERS.name),
        module=module, items=items, reported=(PANEL,), conditions=CONDITIONS,
    )


def _spatial_side() -> Side:
    """The marker workflow's side of the join, for the SAME unit.

    ERBB2 and ACTA2 carry spatial coefficients; CD3E is not an item here, so
    ``reconcile`` adjudicates it against this side's own frames -- which is the
    difference between "never in that panel" and "in it, tested, not reported".
    """
    return Side(
        module="jazzpanda_spatial_glm",
        workflow="marker-interpretation",
        task_kind=TaskKind.ASSIGNMENT,
        unit="c1",
        authority=Authority.MEASUREMENT,
        items=(
            _item("ERBB2", 12.0, MeasuredState.OBSERVED, "glm_coef",
                  frame=SPATIAL_PANEL.name),
            _item("ACTA2", 4.0, MeasuredState.OBSERVED, "glm_coef",
                  frame=SPATIAL_PANEL.name),
        ),
        world=SPATIAL_WORLD,
    )


# --------------------------------------------------------------------------- #
# Registration: the third question, one entry, no type change
# --------------------------------------------------------------------------- #
def test_the_de_workflow_and_two_methods_are_registered() -> None:
    """Registered for real now, not inside a fixture.

    The pair landed once there was an artifact to answer the question against
    (``synthetic_multisample_fixture/inputs/de_edger.csv``, declared synthetic) and
    a composition root to run it (``pipeline/stages/de.py``). TWO methods, which is
    what makes §4's extensibility claim checkable rather than asserted: the root
    picks between them by schema, and nothing in ``core/`` knows either exists.
    """
    assert workflows.get(de.WORKFLOW_ID).task_kind is TaskKind.COMPARISON
    registered = methods.for_workflow(de.WORKFLOW_ID)
    assert seurat_de.MODULE in registered
    assert edger.MODULE in registered
    assert isinstance(de.WORKFLOW, Workflow)
    assert isinstance(seurat_de.MODULE, MethodModule)
    assert isinstance(edger.MODULE, MethodModule)


def test_the_method_cannot_register_before_its_question_exists(isolated_registries):
    """Order is the point: a method with no question still produces evidence, and
    that evidence would surface under whichever surface a caller happened to pick.

    Both entries are popped first, because the pair is registered at import now.
    The fixture puts them back.
    """
    workflows.WORKFLOWS.pop(de.WORKFLOW_ID, None)
    methods.METHODS.pop(seurat_de.MODULE.name, None)
    with pytest.raises(workflows.UnregisteredWorkflow, match=de.WORKFLOW_ID):
        methods.register(seurat_de.MODULE)


@pytest.mark.parametrize("field", ["does_not_measure", "assumptions"])
def test_the_de_method_declares_its_limits(isolated_registries, field):
    empty = "" if field == "does_not_measure" else ()
    with pytest.raises(methods.UndeclaredLimits, match=field):
        methods.register(dataclasses.replace(seurat_de.MODULE, name="broken",
                                             **{field: empty}))
    assert "broken" not in methods.METHODS


def test_does_not_measure_says_nothing_spatial_at_all():
    """The mirror image of jazzPanda's declaration, which is the whole reason two
    methods are worth having: each names the other's strength as its own blind spot,
    so a reader handed a bare number can tell which question produced it."""
    low = seurat_de.MODULE.does_not_measure.lower()
    assert "nothing spatial at all" in low
    assert "dispersed cluster scores identically" in low
    assert "composition" in low
    assert len(seurat_de.MODULE.assumptions) >= 3
    # and the other direction, so the pair cannot drift apart silently
    assert "fold change" in jazzpanda.MODULE.does_not_measure.lower()


# --------------------------------------------------------------------------- #
# Reading the result: schema detection and the frame
# --------------------------------------------------------------------------- #
def test_detect_unit_reads_the_schema_and_refuses_what_it_cannot():
    rows = [{"gene": "ERBB2", "cluster": "c1", "avg_log2FC": 3.0,
             "p_val_adj": 1e-26},
            {"gene": "LUM", "cluster": "c2", "avg_log2FC": -1.4,
             "p_val_adj": 1e-05}]
    spec = seurat_de.MODULE.detect_unit(rows)
    assert spec.unit_kind == "cluster" and spec.unit_column == "cluster"
    assert spec.units == ("c1", "c2")
    assert spec.because

    # A jazzPanda table is not a FindAllMarkers table, and guessing would route the
    # rows to an interpretive question they do not answer.
    with pytest.raises(UnrecognisedResult):
        seurat_de.MODULE.detect_unit(
            [{"gene": "ERBB2", "top_cluster": "c1", "glm_coef": 12.0}])


def test_load_applies_the_panel_frame_and_the_replicate_floor(loaded):
    out = loaded.value
    assert set(out) == {"c1", "c2"}
    assert loaded.decisions, "load must carry the decisions it made applying the frame"

    by_name = {i.subject.display(): i for i in out["c1"].items}
    assert by_name["ERBB2"].statistic == 3.0
    assert by_name["ERBB2"].direction is Direction.UP
    # CD3E is in the file and off the panel: no number, and the pct.* corroborators
    # are withheld with it -- a detection rate surviving here would put a
    # measurement back into an item the frame says was never measured.
    assert by_name["CD3E"].statistic is None
    assert by_name["CD3E"].measured.state is MeasuredState.NEVER_MEASURED
    assert by_name["CD3E"].corroborating == ()
    assert any(d.rule == "measure:never_measured" for d in loaded.decisions)

    # The UNIT is underpowered, not the items: one biological replicate is a fact
    # about the design (ARCHITECTURE 9.5), and ERBB2 still carries its number.
    assert out["c1"].unit_measured.state is MeasuredState.UNDERPOWERED
    cov = out["c1"].unit_measured.coverage[0]
    assert (cov.what, cov.k, cov.floor) == (de.REPLICATE_COVERAGE, 1, 2)

    # Direction is a field, not a weakness: LUM is down and still a result.
    assert out["c2"].items[0].direction is Direction.DOWN


def test_absence_reads_as_keeps_the_four_states_apart_in_de_terms():
    g = subject("gene", g=gene("MKI67"))
    never = seurat_de.MODULE.absence_reads_as(
        g, Measured(MeasuredState.NEVER_MEASURED, frame=PANEL.name))
    tested = seurat_de.MODULE.absence_reads_as(
        g, Measured(MeasuredState.TESTED_ABSENT, frame=PANEL.name))
    assert "argues nothing" in never
    assert "argues against" in tested
    assert "says nothing about where it sits" in tested   # DE's terms, not spatial
    assert never != tested


# --------------------------------------------------------------------------- #
# ARCHITECTURE 9.4: the rubrics must disagree on the same number
# --------------------------------------------------------------------------- #
def test_the_two_rubrics_disagree_on_the_same_number():
    """The invariant ARCHITECTURE §9.4 calls the single most important test in the DE
    work. High effect size, low spatial localisation: the same float 3.0 read through
    two ladders must come out differently, because each maps *its own* statistic
    through *its own* edges. If this ever passes by accident -- both reading one
    shared ladder in ``core/`` -- the framework has gone method-blind and every band
    stops being a statement about a measurement.
    """
    spatial = _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED, "glm_coef"),
                        module="jazzpanda_spatial_glm")
    expression = _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED, "avg_log2FC"))

    jazz = jazzpanda.MODULE.rubric(spatial).value
    mean = seurat_de.MODULE.rubric(expression).value
    assert jazz == "Medium-High", jazz   # glm_coef 3.0: a middling spatial signal
    assert mean == "Very High", mean     # avg_log2FC 3.0: an eight-fold change
    assert jazz != mean
    # and the ladders themselves are separate objects, not one shared constant
    assert seurat_de.BAND_EDGES != jazzpanda.BAND_EDGES


def test_the_de_ladder_is_an_effect_size_and_never_reaches_a_unit():
    """The method declares a ladder because the Protocol requires one and §9.4 tests
    it -- but a COMPARISON has no label space, so nothing may put it on a unit."""
    ev = _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED))
    assert seurat_de.MODULE.rubric(ev).value == "Very High"
    out = interpret(ev, de.WORKFLOW, de.DEAudit(world=WORLD))
    assert "Very High" not in out.value.summary
    assert not any(d.rule.startswith("band:") for d in out.decisions)


# --------------------------------------------------------------------------- #
# The no-band guard, for COMPARISON
# --------------------------------------------------------------------------- #
def test_no_band_on_a_comparison():
    """Four ways round, because three of them are the ways someone would try.

    The field-name guard is the important one: a behavioural test catches a band
    that is filled in, while this fails the moment the field is *declared*, which is
    when the mistake costs nothing to undo.
    """
    fields = {f.name for f in dataclasses.fields(Finding)}
    assert "band" not in fields and "confidence" not in fields

    out = interpret(
        _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED)),
        de.WORKFLOW, de.DEAudit(world=WORLD),
    ).value
    assert isinstance(out, Finding)
    assert out.task_kind is TaskKind.COMPARISON
    assert not hasattr(out, "band")

    # An assignment routed through Finding is an assignment whose band was dropped
    # on the floor rather than computed.
    with pytest.raises(AssignmentIsNotAFinding):
        Finding(unit="c1", unit_kind="cluster", task_kind=TaskKind.ASSIGNMENT,
                items=(), summary="")

    # And the surface refuses a band header on this task kind, before a pixel.
    spec = de.WORKFLOW.uispec()
    assert spec.header.kind is HeaderKind.CONFOUND_CHECKLIST
    with pytest.raises(DegradedUpward):
        UISpec(workflow_id=spec.workflow_id, title=spec.title,
               task_kind=spec.task_kind, unit_kind=spec.unit_kind,
               item_kind=spec.item_kind,
               interpretive_question=spec.interpretive_question,
               header=LabelBandHeader(), columns=spec.columns, blocks=spec.blocks)


def test_the_surface_can_say_never_measured_and_declares_the_four_checks():
    spec = de.WORKFLOW.uispec()
    covered = {s for b in spec.blocks for s in b.statuses}
    assert set(MeasuredState) == covered, "every state needs somewhere to render"
    assert {c.id for c in de.WORKFLOW.checks()} == {
        de.PANEL_SCOPING, de.REPLICATE_CONFOUND, de.COMPOSITION, de.COHERENCE}
    assert {c.key for c in spec.header.checks} == {c.id for c in de.WORKFLOW.checks()}


# --------------------------------------------------------------------------- #
# Check 1: panel scoping
# --------------------------------------------------------------------------- #
def test_panel_scoping_states_k_of_n_and_names_what_was_never_measured(loaded):
    ev = loaded.value["c1"]
    r = de.panel_scoping(ev, de.DEAudit(
        world=WORLD, process="T-cell infiltration",
        canonical=("CD3E", "PTPRC", "ERBB2"),
    )).value
    assert r.state is de.CheckState.FLAGGED
    assert "panel-scoped, not transcriptome-wide" in r.headline
    assert "1/4" in r.headline                     # K of N, read from the frame
    assert "CD3E" in r.headline and "PTPRC" in r.headline
    assert all("never on" in m for m in r.missing)


def test_panel_scoping_refuses_without_a_canonical_list_or_a_frame(loaded):
    ev = loaded.value["c1"]
    no_list = de.panel_scoping(ev, de.DEAudit(world=WORLD)).value
    assert no_list.state is de.CheckState.CANNOT_BE_SETTLED
    assert "canonical gene list" in no_list.missing[0]

    no_frame = de.panel_scoping(ev, de.DEAudit()).value
    assert no_frame.state is de.CheckState.CANNOT_BE_SETTLED
    assert "never asserted" in no_frame.missing[0]   # panel size is never a constant


def test_panel_scoping_passes_when_every_canonical_gene_is_on_the_panel(loaded):
    r = de.panel_scoping(loaded.value["c1"], de.DEAudit(
        world=WORLD, process="proliferation", canonical=("MKI67", "ERBB2"))).value
    assert r.state is de.CheckState.OK
    assert "Every canonical gene" in r.headline


# --------------------------------------------------------------------------- #
# Check 2: the replicate refusal -- ARCHITECTURE 9.7's headline behaviour
# --------------------------------------------------------------------------- #
def test_a_single_replicate_cannot_settle_a_condition_level_claim(loaded):
    """§9.7's headline: the honest output is *cannot be settled*, naming what is
    missing. Not a low band -- there is no band -- and not a quiet pass."""
    r = de.replicate_confound(loaded.value["c1"], de.DEAudit(world=WORLD)).value
    assert r.state is de.CheckState.CANNOT_BE_SETTLED
    assert "CANNOT BE SETTLED" in r.headline
    assert "biological replicates 1/1 (floor 2)" in r.headline
    assert "confounded by construction" in r.headline
    # The refusal NAMES what is missing. "insufficient information" is not actionable.
    assert "1 further biological replicate(s) (have 1, floor 2)" in r.missing[0]


def test_the_replicate_check_cannot_pass_from_a_result_table_alone(tmp_path):
    """Two samples clear the floor and it STILL cannot be settled, for a different
    reason and under a different rule id: a FindAllMarkers table has one number per
    gene for the whole unit, so which replicate drives it is not in the file. An OK
    here would be a check that passed because it never ran."""
    csv = tmp_path / "two_samples.csv"
    csv.write_text(FINDALLMARKERS_CSV)
    out = seurat_de.MODULE.load(str(csv), _Adapter(("s1", "s2")), object())
    ev = out.value["c1"]
    # The floor is cleared, so the coverage no longer demotes the unit. (The unit
    # itself reads TESTED_ABSENT because a cluster id is not in the gene frame's
    # reported set -- which is the lattice doing its job, not a replicate verdict.)
    cov = ev.unit_measured.coverage[0]
    assert (cov.k, cov.floor, cov.ok) == (2, 2, True)
    assert ev.unit_measured.state is not MeasuredState.UNDERPOWERED

    traced = de.replicate_confound(ev, de.DEAudit(world=WORLD))
    assert traced.value.state is de.CheckState.CANNOT_BE_SETTLED
    assert "per-sample effect sizes" in traced.value.missing[0]
    assert [d.rule for d in traced.decisions] == ["de:replicate_not_in_result"]

    # and the undeclared case is a third, separate refusal
    undeclared = de.replicate_confound(_evidence(), de.DEAudit()).value
    assert undeclared.state is de.CheckState.CANNOT_BE_SETTLED
    assert "never declared" in undeclared.headline


# --------------------------------------------------------------------------- #
# Check 3: composition versus expression
# --------------------------------------------------------------------------- #
def test_composition_is_confounded_when_the_units_share_of_cells_moves(loaded):
    ev = loaded.value["c1"]
    moved = de.composition_vs_expression(ev, de.DEAudit(
        world=WORLD, abundance=(("treated", 0.31), ("untreated", 0.12)))).value
    assert moved.state is de.CheckState.FLAGGED
    assert "CONFOUNDED" in moved.headline
    assert "became more numerous" in moved.headline

    held = de.composition_vs_expression(ev, de.DEAudit(
        world=WORLD, abundance=(("treated", 0.12), ("untreated", 0.12)))).value
    assert held.state is de.CheckState.OK

    absent = de.composition_vs_expression(ev, de.DEAudit(world=WORLD)).value
    assert absent.state is de.CheckState.CANNOT_BE_SETTLED
    assert "share of cells under each arm" in absent.missing[0]


# --------------------------------------------------------------------------- #
# Check 4: cross-method coherence, through core/reconcile.py
# --------------------------------------------------------------------------- #
def test_coherence_runs_through_reconcile_and_keeps_conflict_apart_from_silence(
        loaded):
    """The two outcomes a boolean would render identically, in ONE result.

    - ACTA2: the spatial method observed it; the DE panel HAD it, tested it and did
      not report it -- a measured absence, so they genuinely conflict (TENSION).
    - CD3E: the DE panel never had it, so its silence is not a negative observation
      and the pair is NOT_COMPARABLE, naming the side and the frame.

    Different questions, so nothing here can ever come back AGREE (§9.11).
    """
    ev = loaded.value["c1"]
    traced = de.cross_method_coherence(
        ev, de.DEAudit(world=WORLD, spatial=_spatial_side()))
    r = traced.value

    assert r.state is de.CheckState.FLAGGED
    assert "TENSION" in r.headline and "ACTA2" in r.headline
    assert any("CD3E" in m and "never_measured" in m for m in r.missing)
    # The join itself came from core/reconcile.py, not from here.
    rules = {d.rule for d in traced.decisions}
    assert {"reconcile:tension", "reconcile:not_comparable",
            "reconcile:coherent"} <= rules
    assert "reconcile:agree" not in rules


def test_coherence_refuses_when_there_is_no_second_method(loaded):
    r = de.cross_method_coherence(loaded.value["c1"], de.DEAudit(world=WORLD)).value
    assert r.state is de.CheckState.CANNOT_BE_SETTLED
    assert "not the same as checked and coherent" in r.missing[0]


def test_reference_authority_can_never_make_a_de_result_coherent(loaded):
    """The rule about reference data, reached through the workflow: a curated side
    lands on NOT_COMPARABLE before any number is read."""
    curated = dataclasses.replace(_spatial_side(), module="curated_atlas",
                                  authority=Authority.CURATED)
    r = de.cross_method_coherence(
        loaded.value["c1"], de.DEAudit(world=WORLD, spatial=curated)).value
    assert r.state is de.CheckState.CANNOT_BE_SETTLED
    assert any("reference authority" in b for b in r.because)


# --------------------------------------------------------------------------- #
# interpret: the Finding, and the deterministic verify
# --------------------------------------------------------------------------- #
def test_interpret_tiers_by_the_measurement_and_flags_verify_deterministically(
        loaded):
    ev = loaded.value["c1"]
    out = interpret(ev, de.WORKFLOW, de.DEAudit(
        world=WORLD, process="T-cell infiltration", canonical=("CD3E",),
        spatial=_spatial_side(),
    ))
    found = out.value
    by_name = {i.subject.display(): i for i in found.items}

    # ERBB2 was observed, but every check on this unit is flagged or unsettled, so
    # the number is suggestive rather than supported -- a large fold change under a
    # one-replicate design is not better supported than a small one.
    assert by_name["ERBB2"].tier is Tier.SUGGESTIVE
    assert by_name["ERBB2"].verify is True
    # CD3E was never measured: untestable, and NOT a zero and NOT a blank.
    assert by_name["CD3E"].tier is Tier.UNTESTABLE
    assert by_name["CD3E"].evidence.statistic is None
    assert found.untestable_count == 1

    assert "No band" in found.summary
    assert "Cannot be settled" in found.summary
    assert "replicate_confound" in found.summary


def test_a_tested_absence_is_answered_and_not_flagged_for_recheck():
    """The productive half of the lattice must not read as a refusal: MKI67 was on
    the panel, was tested and was not reported, which is an answer the biologist
    already has."""
    ev = _evidence(_item("MKI67", None, MeasuredState.TESTED_ABSENT))
    item = interpret(ev, de.WORKFLOW, de.DEAudit(world=WORLD)).value.items[0]
    assert item.tier is Tier.NOT_SUPPORTED
    assert item.verify is False


# --------------------------------------------------------------------------- #
# The rival test: for a COMPARISON the rival is a CONFOUND, from the data
# --------------------------------------------------------------------------- #
def test_the_rival_space_is_the_data_and_an_unknown_confound_is_reported_unasked(
        loaded):
    assert seurat_de.MODULE.rival_space == "data"
    assert seurat_de.MODULE.traversals() == ()

    ev = loaded.value["c1"]
    named = de.WORKFLOW.rival_test(ev, de.REPLICATE_CONFOUND,
                                   de.DEAudit(world=WORLD)).value
    assert "CANNOT BE SETTLED" in named.undecidable[0]
    assert named.separated_by == ()

    unknown = de.WORKFLOW.rival_test(ev, "Myoepithelial",
                                     de.DEAudit(world=WORLD)).value
    assert "no confound check is registered" in unknown.undecidable[0]
    assert "come from the data" in unknown.undecidable[0]

    settled = de.WORKFLOW.rival_test(ev, de.COMPOSITION, de.DEAudit(
        world=WORLD, abundance=(("a", 0.1), ("b", 0.1)))).value
    assert settled.undecidable == ()


# --------------------------------------------------------------------------- #
# Every decision this workflow emits is a registered rule
# --------------------------------------------------------------------------- #
def test_every_de_decision_is_registered(loaded):
    traced = interpret(loaded.value["c1"], de.WORKFLOW, de.DEAudit(
        world=WORLD, process="T-cell infiltration", canonical=("CD3E",),
        spatial=_spatial_side(), relationship=Relationship.PARTLY_INDEPENDENT,
    ))
    emitted = {d.rule for d in traced.decisions if d.rule.startswith("de:")}
    assert emitted, "the four checks must record what they decided"
    assert emitted <= set(RULES)
    assert {RULES[r].step for r in emitted} == {"de"}


# --------------------------------------------------------------------------- #
# The demonstrations, printed. Run with ``pytest -s -k demonstration``.
# --------------------------------------------------------------------------- #
def test_refusal_demonstration(loaded, capsys):
    """§9.7's headline behaviour, printed: a single-replicate DE result producing
    *cannot be settled*, naming what is missing."""
    ev = loaded.value["c1"]
    audit = de.DEAudit(world=WORLD, process="T-cell infiltration",
                       canonical=("CD3E", "PTPRC", "ERBB2"))
    lines = [
        "SYNTHETIC FIXTURE - every number below was written by hand for this test",
        f"workflow  {de.WORKFLOW.name}  [{de.WORKFLOW.task_kind.value}]   band: none",
        f"question  {de.WORKFLOW.question}",
        f"method    {seurat_de.MODULE.name}  (statistic: "
        f"{seurat_de.MODULE.statistic_name})",
        f"design    {len(_Adapter().samples(None))} sample(s); unit is "
        f"{ev.unit_measured.state.value} in {ev.unit_measured.frame}",
        "",
    ]
    for fn in (de.panel_scoping, de.replicate_confound,
               de.composition_vs_expression, de.cross_method_coherence):
        r = fn(ev, audit).value
        lines += [f"  [{r.state.value.upper()}] {r.check_id}", f"      {r.headline}"]
        lines += [f"      missing: {m}" for m in r.missing]
        lines.append("")
    lines += ["SUMMARY", "  " + interpret(ev, de.WORKFLOW, audit).value.summary]
    print("\n".join(lines))

    out = capsys.readouterr().out
    assert "CANNOT_BE_SETTLED" in out and "replicate" in out


def test_coherence_demonstration(loaded, capsys):
    """The cross-method join, printed: genuine conflict and could-not-be-compared
    side by side, straight out of ``core/reconcile.py``."""
    ev = loaded.value["c1"]
    spatial = _spatial_side()
    mine = Side(module=ev.module, workflow=de.WORKFLOW_ID,
                task_kind=TaskKind.COMPARISON, unit=ev.unit,
                authority=Authority.MEASUREMENT, items=ev.items, world=WORLD)
    joined = reconcile(spatial, mine, Relationship.PARTLY_INDEPENDENT).value

    lines = [
        f"reconcile  {joined.left}  x  {joined.right}  on {joined.unit}",
        f"           same_question={joined.same_question}  "
        f"relationship={joined.relationship.value}",
        "",
    ]
    for p in joined.pairs:
        lines.append(f"  {p.subject.display():<8} {p.outcome.value.upper():<16}"
                     f" corroborates={p.corroborates}")
        lines += [f"      {b}" for b in p.because]
        if p.blocked:
            lines.append(f"      blocked: {', '.join(p.blocked)}")
    print("\n".join(lines))

    out = capsys.readouterr().out
    assert "TENSION" in out and "NOT_COMPARABLE" in out
    # Different questions never AGREE (9.11) -- COHERENT is as close as it gets,
    # and a coherent pair is at most a second, partly independent witness. It is
    # never a band: there is no band on either side of this join.
    assert "AGREE" not in out
    assert Agreement.AGREE not in {p.outcome for p in joined.pairs}
    assert [p.subject.display() for p in joined.corroborated] == ["ERBB2"]
