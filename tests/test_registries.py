"""The extensibility claim, tested rather than asserted.

ARCHITECTURE §4 says the core does not change when a registry gains a member. That
is a claim about code nobody has written yet, so the only way to hold it honest is to
add a member here -- :class:`MeanExpressionDE`, a deliberately trivial second method
-- and show it registers into the *same* workflow, satisfies the *same* Protocol and
produces a *different* band, with no type anywhere changing to admit it. If adding
the stub below had required one edit to ``types/`` or ``core/``, the claim would be
false and this file is where that shows up.

**Three invariants, each with a named failure.**

1. ``test_a_method_must_declare_its_limits`` -- the one the whole design rests on.
   ``contracts.py``'s Protocol can only make ``does_not_measure`` and ``assumptions``
   *present*; emptiness is invisible to structural typing. A method that declares no
   limits is one whose output reads as having none, and a ``glm_coef`` with nothing
   beside it is indistinguishable from a fold change.
2. ``test_duplicate_*_registration_raises`` -- two registrations for one id are
   resolved by import order, which is decided by nothing, and the loser fails at
   render time far from the cause.
3. ``test_two_rubrics_disagree_on_the_same_number`` -- §9.4's invariant. A
   spatial-localisation rubric and a mean-expression rubric must produce different
   bands on the same number. If they agree the framework is method-blind and the
   value proposition inverts: the whole point is that the band is a function of *one
   method's* output.

The registries are module-level dicts, so every test that registers anything does it
inside :func:`isolated_registries`, which snapshots and restores both. A test that
leaked a stub would make a later ``DuplicateRegistration`` test pass for the wrong
reason, which is the quietest way for a registration test to stop testing anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pytest

from panoscope import methods, workflows
from panoscope.core.interpret import interpret
from panoscope.methods import jazzpanda
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import (
    MethodModule, UnitSpec, UnrecognisedResult, Workflow,
)
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Frame, Measured, MeasuredState, MeasuredWorld
from panoscope.types.outputs import Assignment, TaskKind
from panoscope.types.subject import EntityKind, Subject, gene, subject
from panoscope.types.subject import unit as unit_entity
from panoscope.types.trace import Trace, Traced
from panoscope.types.uispec import DuplicateRegistration
from panoscope.workflows import marker

WORKFLOW_ID = "marker-interpretation"

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM", "ACTA2"}))
CLUSTERS = Frame(EntityKind.UNIT, "annotation:test", frozenset({"c1", "c2"}))
WORLD = MeasuredWorld((PANEL, CLUSTERS))
CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("jazzpanda", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(), CLUSTERS.ref()),
)


# --------------------------------------------------------------------------- #
# The second method. THIS is the extensibility claim: a new file, a new entry,
# no type change. It is trivial on purpose -- the point is what it takes to add
# one, not what it does once added.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MeanExpressionDE:
    """Cluster-vs-rest mean expression, declared. A stand-in for ``seurat_de.py``.

    Its ``does_not_measure`` is the mirror image of jazzPanda's, which is the whole
    reason two methods in one workflow is worth building: each names the other's
    strength as its own blind spot, so a reader handed a number can tell which
    question produced it.
    """

    name: str = "stub_mean_expression_de"
    workflow: str = WORKFLOW_ID
    statistic_name: str = "log2FC"
    measures: str = (
        "the log2 ratio of a gene's mean expression in this cluster's cells against "
        "all other cells"
    )
    does_not_measure: str = (
        "spatial arrangement of any kind: a gene uniformly raised across every cell "
        "of a dispersed cluster scores identically to one whose transcripts "
        "concentrate in a single region"
    )
    assumptions: tuple[str, ...] = (
        "cluster labels are fixed and produced upstream",
        "segmentation assigned transcripts to the right cells",
    )
    rival_space: Literal["reference", "data", "none"] = "none"
    citation: str = "(test fixture; stands in for seurat_de.py, no paper)"

    #: Its OWN edges, on its OWN statistic. A log2FC of 3 is a large effect; a
    #: glm_coef of 3 is a middling one. Sharing one ladder would force the two to
    #: agree, which is the method-blindness §9.4 forbids.
    BAND_EDGES = (("Very High", 2.0), ("High", 1.0), ("Medium", 0.5), ("Low", 0.0))

    def detect_unit(self, table: Any) -> UnitSpec:
        return UnitSpec(unit_kind="cluster", unit_column="cluster", units=("c1",))

    def load(self, path: str, adapter: Any, store: Any) -> dict[str, NormalisedEvidence]:
        return {}

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        return f"{subject.display()}: no mean-expression difference between conditions"

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        t = Trace()
        lead = next((i for i in ev.items if i.measured.carries_number), None)
        if lead is None:
            return t.done("Low")
        c = abs(float(lead.statistic or 0.0))
        band = next((lab for lab, edge in self.BAND_EDGES if c >= edge), "Low")
        t.note("band:from_coef", band, (f"log2FC={c}",))
        return t.done(band)

    def traversals(self) -> tuple[tuple[str, str], ...]:
        return ()


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def isolated_registries():
    """Snapshot both registries and restore them, whatever the test did.

    Module-level dicts survive between tests in one process, so a leaked stub would
    make a later duplicate-registration test raise for the wrong reason and pass
    without testing anything.
    """
    m, w = dict(methods.METHODS), dict(workflows.WORKFLOWS)
    try:
        yield
    finally:
        methods.METHODS.clear()
        methods.METHODS.update(m)
        workflows.WORKFLOWS.clear()
        workflows.WORKFLOWS.update(w)


def _item(symbol: str, stat: float | None, state: MeasuredState,
          statistic_name: str = "glm_coef",
          corroborating: tuple[tuple[str, float], ...] = (),
          flags: tuple[str, ...] = ()) -> EvidenceItem:
    return EvidenceItem(
        subject=subject("gene", g=gene(symbol)),
        statistic=stat,
        statistic_name=statistic_name,
        measured=Measured(state, frame=PANEL.name),
        corroborating=corroborating,
        flags=flags,
    )


def _evidence(*items: EvidenceItem, module: str = "jazzpanda_spatial_glm",
              unit: str = "c1") -> NormalisedEvidence:
    return NormalisedEvidence(
        unit=unit, unit_kind="cluster",
        unit_subject=subject("niche", niche=unit_entity(unit)),
        unit_measured=Measured(MeasuredState.OBSERVED, frame=CLUSTERS.name),
        module=module, items=items, reported=(PANEL,), conditions=CONDITIONS,
    )


# --------------------------------------------------------------------------- #
# The registry, working
# --------------------------------------------------------------------------- #
def test_jazzpanda_is_registered_into_the_marker_workflow():
    m = methods.get("jazzpanda_spatial_glm")
    assert m.workflow == WORKFLOW_ID
    assert workflows.get(WORKFLOW_ID).task_kind is TaskKind.ASSIGNMENT
    assert m in methods.for_workflow(WORKFLOW_ID)


def test_registered_members_satisfy_their_protocols():
    """Parameterised over the registries, not over a hand-written list: a member
    added later is covered without anyone remembering to add it here."""
    for m in methods.METHODS.values():
        assert isinstance(m, MethodModule), m
    for w in workflows.WORKFLOWS.values():
        assert isinstance(w, Workflow), w


def test_a_second_method_registers_into_the_same_workflow(isolated_registries):
    """The extensibility claim: one entry, no type change, same question."""
    methods.register(MeanExpressionDE())
    answering = {m.name for m in methods.for_workflow(WORKFLOW_ID)}
    assert answering == {"jazzpanda_spatial_glm", "stub_mean_expression_de"}
    # Same workflow object, same task kind: the question did not fork.
    assert workflows.get(WORKFLOW_ID).task_kind is TaskKind.ASSIGNMENT


def test_task_kind_belongs_to_the_method_not_the_use():
    """Both methods answer an ASSIGNMENT because they register into an ASSIGNMENT
    workflow. There is no per-call override, and that absence is the point: a reader
    cannot turn a characterisation into an assignment by asking differently."""
    for m in methods.for_workflow(WORKFLOW_ID):
        assert workflows.get(m.workflow).task_kind is TaskKind.ASSIGNMENT


# --------------------------------------------------------------------------- #
# The checks that must bite
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["does_not_measure", "assumptions"])
def test_a_method_must_declare_its_limits(isolated_registries, field):
    """The whole point of the design. The Protocol makes these present; emptiness is
    what a Protocol cannot see, so the registry refuses it."""
    empty = "" if field == "does_not_measure" else ()
    stub = MeanExpressionDE(name="broken_method", **{field: empty})
    with pytest.raises(methods.UndeclaredLimits, match=field):
        methods.register(stub)
    assert "broken_method" not in methods.METHODS


def test_duplicate_method_registration_raises(isolated_registries):
    with pytest.raises(DuplicateRegistration, match="jazzpanda_spatial_glm"):
        methods.register(jazzpanda.MODULE)


def test_duplicate_workflow_registration_raises(isolated_registries):
    with pytest.raises(DuplicateRegistration, match=WORKFLOW_ID):
        workflows.register(marker.WORKFLOW)


def test_a_method_cannot_register_into_a_workflow_nobody_declared(isolated_registries):
    """An orphan method is worse than a missing one: it produces evidence with no
    question to answer, which then surfaces under whichever surface a caller picks."""
    orphan = MeanExpressionDE(name="orphan", workflow="de-interpretation")
    with pytest.raises(workflows.UnregisteredWorkflow, match="de-interpretation"):
        methods.register(orphan)


def test_unknown_names_raise_rather_than_defaulting():
    with pytest.raises(methods.UnregisteredMethod):
        methods.get("nope")
    with pytest.raises(workflows.UnregisteredWorkflow):
        workflows.get("nope")


# --------------------------------------------------------------------------- #
# §9.4: the rubrics must disagree
# --------------------------------------------------------------------------- #
def test_two_rubrics_disagree_on_the_same_number():
    """High effect size, low spatial localisation -- the constructed case §9.4 names.

    The two methods read the same float and return different bands, because each maps
    *its own* statistic through *its own* edges. If this ever passes by accident --
    both rubrics reading one shared ladder in ``core/`` -- the framework has gone
    method-blind and every band stops being a statement about a measurement.
    """
    spatial = _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED,
                              corroborating=(("pearson", 0.10),)))
    de = _evidence(_item("ERBB2", 3.0, MeasuredState.OBSERVED,
                         statistic_name="log2FC"),
                   module="stub_mean_expression_de")

    jazz = jazzpanda.MODULE.rubric(spatial).value
    mean = MeanExpressionDE().rubric(de).value
    assert jazz == "Medium-High", jazz   # glm_coef 3.0 is a middling spatial signal
    assert mean == "Very High", mean     # log2FC 3.0 is a large fold change
    assert jazz != mean


def test_the_method_declares_the_rubric_and_core_applies_the_policy():
    """The PM's split, pinned. The method's rubric is the BASE band only: no promote,
    no demote, no cap, no floor. ``core/verdict.py`` owns those, so a specificity flag
    that core demotes on must leave the method's own answer untouched."""
    from panoscope.core import verdict as core_verdict

    flagged = _evidence(_item("ERBB2", 6.0, MeasuredState.OBSERVED,
                              corroborating=(("pearson", 0.95),),
                              flags=("specificity:not_unique",)))
    assert jazzpanda.MODULE.rubric(flagged).value == "High"   # 6.0 -> High, full stop
    # Same evidence through the policy layer: promoted on pearson, demoted on the
    # flag, and the fragile cap applies because one marker carries the call.
    out = core_verdict.assess(flagged.items, frozenset({"ERBB2"}))
    assert "band:promote_pearson_high" in out.value.demotions
    assert "band:demote_not_unique" in out.value.demotions
    assert out.value.verify is True


def test_the_declared_edges_match_the_live_policy_ladder():
    """A duplicated constant with a test is a smaller problem than an inverted
    layering -- the same trade ``core/verdict.py`` makes against ``agent/config``.
    The day ``core`` reads the ladder off the registry, this test deletes itself."""
    from panoscope.core import verdict as core_verdict

    assert dict(jazzpanda.BAND_EDGES) == core_verdict.CONFIDENCE_BANDS
    assert tuple(lab for lab, _ in jazzpanda.BAND_EDGES) == core_verdict.BANDS


# --------------------------------------------------------------------------- #
# The method's own declarations
# --------------------------------------------------------------------------- #
def test_jazzpanda_declares_what_it_does_not_measure():
    m = jazzpanda.MODULE
    low = m.does_not_measure.lower()
    assert "fold change" in low and "mean expression" in low
    assert len(m.assumptions) >= 3
    assert m.statistic_name == "glm_coef"


def test_detect_unit_reads_the_schema_and_refuses_what_it_cannot():
    rows = [{"gene": "ERBB2", "top_cluster": "c1", "glm_coef": 12.0},
            {"gene": "LUM", "top_cluster": "c2", "glm_coef": 3.0},
            {"gene": "ACTA2", "top_cluster": "NoSig", "glm_coef": 0.01}]
    spec = jazzpanda.MODULE.detect_unit(rows)
    assert spec.unit_kind == "cluster" and spec.unit_column == "top_cluster"
    # NoSig is tested-but-unassigned, not a unit: admitting it would create a unit
    # whose every item is an absence and then band it as though something was found.
    assert spec.units == ("c1", "c2")
    assert spec.because

    with pytest.raises(UnrecognisedResult):
        jazzpanda.MODULE.detect_unit([{"gene": "ERBB2", "avg_log2FC": 2.0}])


def test_absence_reads_as_keeps_the_four_states_apart():
    g = subject("gene", g=gene("TP63"))
    never = jazzpanda.MODULE.absence_reads_as(
        g, Measured(MeasuredState.NEVER_MEASURED, frame="panel:test"))
    tested = jazzpanda.MODULE.absence_reads_as(
        g, Measured(MeasuredState.TESTED_ABSENT, frame="panel:test"))
    assert "argues nothing" in never
    assert "argues against" in tested
    assert never != tested


# --------------------------------------------------------------------------- #
# load: reads precomputed output, never recomputes
# --------------------------------------------------------------------------- #
class _StubAdapter:
    """The smallest thing satisfying what ``load`` uses of a PlatformAdapter."""

    name = "stub"

    def frames(self, store: Any) -> tuple[Frame, ...]:
        return (PANEL, CLUSTERS)

    def standing_conditions(self, store: Any) -> Conditions:
        return CONDITIONS


def test_load_reads_the_table_and_applies_the_frame(tmp_path):
    """One off-panel gene (CD3E) is in the file and must come back NEVER_MEASURED
    with no number -- the panel-absence rule reaching all the way from a CSV row."""
    csv = tmp_path / "markers_top.csv"
    csv.write_text(
        "gene,top_cluster,glm_coef,pearson,max_gg_corr,max_gc_corr\n"
        "ERBB2,c1,12.0,0.90,0.50,0.90\n"
        "CD3E,c1,4.0,0.80,0.40,0.80\n"
        "LUM,c2,3.0,0.20,0.99,0.95\n"
        "ACTA2,NoSig,0.01,0.10,0.10,0.10\n"
    )
    loaded = jazzpanda.MODULE.load(str(csv), _StubAdapter(), object())
    # load returns Traced: the decisions taken while applying the frame -- which
    # statistics were withheld and why -- are the audit trail, not a side effect.
    out = loaded.value
    assert set(out) == {"c1", "c2"}
    assert loaded.decisions, "load must carry the decisions it made applying the frame"
    assert any(d.rule == "measure:never_measured" for d in loaded.decisions)

    by_key = {i.subject.display(): i for i in out["c1"].items}
    assert by_key["ERBB2"].statistic == 12.0
    assert by_key["CD3E"].statistic is None
    assert by_key["CD3E"].measured.state is MeasuredState.NEVER_MEASURED
    # The corroborators were withheld with the statistic; a pearson surviving here
    # would put a measurement back into an item the frame says was never measured.
    assert by_key["CD3E"].corroborating == ()

    # The three specificity flags are raised ONCE, here, as registered ids.
    lum = out["c2"].items[0]
    assert "specificity:not_unique" in lum.flags
    assert "specificity:localises_elsewhere" in lum.flags
    assert "specificity:low" in lum.flags


# --------------------------------------------------------------------------- #
# The workflow
# --------------------------------------------------------------------------- #
def test_the_marker_workflow_declares_an_assignment_surface():
    spec = marker.WORKFLOW.uispec()
    assert spec.task_kind is TaskKind.ASSIGNMENT
    assert spec.header.kind.value == "label_band"
    covered = {s for b in spec.blocks for s in b.statuses}
    assert set(MeasuredState) == covered, "every state needs somewhere to render"
    assert {c.id for c in marker.WORKFLOW.checks()} >= {"panel_coverage"}


def test_interpret_partitions_by_whether_a_number_exists():
    ev = _evidence(
        _item("ERBB2", 12.0, MeasuredState.OBSERVED,
              corroborating=(("pearson", 0.90),)),
        _item("LUM", 3.0, MeasuredState.OBSERVED),
        _item("ACTA2", 2.0, MeasuredState.OBSERVED),
        _item("CD3E", None, MeasuredState.NEVER_MEASURED),
    )
    ref = marker.MarkerReference(label="Tumor", canonical=("ERBB2",), world=WORLD)
    out = interpret(ev, marker.WORKFLOW, ref)

    assert isinstance(out.value, Assignment)
    # Three markers carry a number, so the fragile cap does not apply and the band
    # is the driver's own: glm_coef 12.0, promoted once on pearson 0.90.
    assert out.value.band == "Very High"
    assert [d.subject.display() for d in out.value.drivers] == ["ERBB2", "LUM", "ACTA2"]
    assert [u.subject.display() for u in out.value.unmeasured] == ["CD3E"]
    assert "never measured here: cd3e" in out.value.rationale.lower()
    # The decisions came up from core/, not from here: the workflow composes.
    assert any(d.rule.startswith("band:") for d in out.decisions)


def test_rival_test_reports_an_uncurated_rival_rather_than_settling_it():
    ev = _evidence(_item("ERBB2", 12.0, MeasuredState.OBSERVED))
    ref = marker.MarkerReference(label="Tumor", canonical=("ERBB2",), world=WORLD)
    out = marker.WORKFLOW.rival_test(ev, "Myoepithelial", ref)
    assert out.value.separated_by == ()
    assert "no canonical marker list" in out.value.undecidable[0]


def test_rival_test_separates_on_a_measured_absence():
    """ACTA2 is on the panel and was not reported for c1 -- a tested absence, which
    is the half of the lattice that argues. An off-panel rival marker must land in
    ``undecidable`` instead, never as evidence against."""
    ev = _evidence(_item("ERBB2", 12.0, MeasuredState.OBSERVED))
    ref = marker.MarkerReference(
        label="Tumor", canonical=("ERBB2",), world=WORLD,
        rivals=(("Myoepithelial", ("ACTA2", "TP63")),),
    )
    out = marker.WORKFLOW.rival_test(ev, "Myoepithelial", ref)
    assert [i.subject.display() for i in out.value.separated_by] == ["ACTA2"]
    assert any("TP63" in u and "never_measured" in u for u in out.value.undecidable)


# --------------------------------------------------------------------------- #
# The demonstration, printed. Run with ``pytest -s -k demonstration``.
# --------------------------------------------------------------------------- #
def test_registry_demonstration(isolated_registries, capsys):
    methods.register(MeanExpressionDE())
    w = workflows.get(WORKFLOW_ID)
    lines = [
        f"workflow  {w.name}  [{w.task_kind.value}]",
        f"question  {w.question}",
        f"surface   {w.uispec().workflow_id} / {w.uispec().header.kind.value}",
        "",
    ]
    for m in methods.for_workflow(WORKFLOW_ID):
        lines += [
            f"  method          {m.name}  (statistic: {m.statistic_name})",
            f"    measures          {m.measures}",
            f"    does_not_measure  {m.does_not_measure}",
            "    assumptions",
            *[f"      - {a}" for a in m.assumptions],
            f"    citation          {m.citation}",
            "",
        ]
    print("\n".join(lines))
    out = capsys.readouterr().out
    assert "jazzpanda_spatial_glm" in out and "stub_mean_expression_de" in out
