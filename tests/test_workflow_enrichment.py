"""The second workflow, tested as a PAIR with the first.

The extensibility claim is not "a registry exists"; it is that a *different kind of
question* registers into it without the types moving. So almost every test here is
written against both workflows at once: markers are an ASSIGNMENT with a band,
enrichment is a CHARACTERISATION with per-item counts and no band anywhere, and both
registries accepted their methods with no change to ``types/`` or ``core/``.

**The invariant this file exists to pin.** CLAUDE.md: *no label space means a
unit-level band is undefined, not merely unhelpful*. Four independent gates have to
bite for that to be structural rather than reviewed, and each is tested here by
trying to break it:

1. ``Finding`` has no ``band`` field to fill -- a field-name guard, which fails the
   moment the field is *declared* rather than only when it is populated;
2. ``UISpec`` refuses a ``LabelBandHeader`` on a non-assignment surface, which is the
   bug ``ui/enrichment_table.py`` actually shipped: it imported the assignment header
   and inherited a confidence band it had no right to;
3. the method's ``rubric`` raises rather than returning a band;
4. ``Assignment`` is not reachable from this workflow at all -- ``interpret`` returns
   a ``Finding``, and the union in ``Workflow.interpret`` is the type-level form of
   the same rule.

The other half is the panel-coverage rule, which is the enrichment edition of the
panel-absence rule: a set whose K of N falls below the floor is UNDERPOWERED, its
statistic is withheld, and its absence therefore argues nothing -- as distinct from a
set that WAS scored and did not clear the reporting bar, which argues. Collapsing
those two into "unknown" is the failure the four states exist to stop, and
``test_a_below_floor_set_argues_nothing`` is where it would show up.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from panoscope import methods, workflows
from panoscope.methods import jazzpanda, jazzpanda_cgst
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import MethodModule, UnrecognisedResult, Workflow
from panoscope.types.measured import Frame, Measured, MeasuredState, MeasuredWorld
from panoscope.types.outputs import Assignment, Finding, FindingItem, TaskKind, Tier
from panoscope.types.subject import EntityKind, gene_set, subject
from panoscope.types.trace import RULES
from panoscope.types.uispec import (
    DegradedUpward, HeaderKind, LabelBandHeader, UISpec,
)
from panoscope.workflows import enrichment, marker

WORKFLOW_ID = "enrichment-interpretation"

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({
    "MKI67", "TOP2A", "CCND1", "CENPF", "LUM", "ACTA2", "FN1", "SCD", "SMS",
    "CXCL9", "STAT1", "CD74",
}))
CLUSTERS = Frame(EntityKind.UNIT, "clusters:test", frozenset({"c1", "c2"}))
CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("jazzpanda_cgst", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(), CLUSTERS.ref()),
)

# One row per state the lattice can produce, so every assertion below is about a
# real load rather than a hand-built item:
#   G2M      -- 4 panel genes, q below the bar            -> OBSERVED, enriched
#   EMT      -- 3 panel genes, q between the two bars     -> OBSERVED, suggestive
#   FATTY    -- TPD52 is OFF PANEL, so K is 2 of a floor of 3 -> UNDERPOWERED
#   APOPTOSIS-- scored, q above the method's reporting bar    -> TESTED_ABSENT
#   IFN_G    -- scored for c2 only; for c1 it is a rival the frames must adjudicate
FIXTURE_CSV = (
    "gene_set,cluster,test_statistic,p_adj_bh,geneset_size,genes_tested,"
    "genes_selected,gc_corr\n"
    'HALLMARK_G2M_CHECKPOINT,c1,6.4,1e-9,200,"MKI67,TOP2A,CCND1,CENPF",'
    '"MKI67,TOP2A,CCND1,CENPF",0.83\n'
    'HALLMARK_EMT,c1,2.1,0.12,200,"LUM,ACTA2,FN1","LUM,ACTA2,FN1",0.41\n'
    'HALLMARK_FATTY_ACID_METABOLISM,c1,5.0,1e-6,158,"SCD,SMS,TPD52",'
    '"SCD,SMS,TPD52",0.52\n'
    'HALLMARK_APOPTOSIS,c1,0.5,0.80,161,"MKI67,TOP2A,CCND1","MKI67,TOP2A,CCND1",0.10\n'
    'HALLMARK_INTERFERON_GAMMA_RESPONSE,c2,11.0,0.0,200,"CXCL9,STAT1,CD74",'
    '"CXCL9,STAT1,CD74",0.90\n'
)


class _StubAdapter:
    """The smallest thing satisfying what ``load`` uses of a PlatformAdapter."""

    name = "stub"

    def frames(self, store: Any) -> tuple[Frame, ...]:
        return (PANEL, CLUSTERS)

    def standing_conditions(self, store: Any) -> Conditions:
        return CONDITIONS


@pytest.fixture
def loaded(tmp_path):
    path = tmp_path / "enrichment.csv"
    path.write_text(FIXTURE_CSV, encoding="utf-8")
    return jazzpanda_cgst.MODULE.load(str(path), _StubAdapter(), object())


@pytest.fixture
def c1(loaded):
    return loaded.value["c1"]


@pytest.fixture
def world(tmp_path):
    """The study's frames PLUS the collection frame, which only the method can
    declare -- the adapter knows panels, not which sets a result was run against."""
    import csv
    import io
    rows = list(csv.DictReader(io.StringIO(FIXTURE_CSV)))
    return MeasuredWorld(
        (PANEL, CLUSTERS, jazzpanda_cgst.MODULE.scored_frame(rows)))


def _by_set(ev) -> dict[str, Any]:
    return {i.subject.display(): i for i in ev.items}


# --------------------------------------------------------------------------- #
# The pair: two questions, two task kinds, one unchanged core
# --------------------------------------------------------------------------- #
def test_both_registries_accepted_a_second_member():
    """One file, one registration line each. No type changed to admit either."""
    w = workflows.get(WORKFLOW_ID)
    m = methods.get("jazzpanda_cgst")
    assert isinstance(w, Workflow) and isinstance(m, MethodModule)
    assert m.workflow == WORKFLOW_ID
    assert m in methods.for_workflow(WORKFLOW_ID)
    assert {m.name for m in methods.for_workflow("marker-interpretation")} == {
        "jazzpanda_spatial_glm"}


def test_the_two_workflows_ask_different_task_kinds():
    assert marker.WORKFLOW.task_kind is TaskKind.ASSIGNMENT
    assert enrichment.WORKFLOW.task_kind is TaskKind.CHARACTERISATION
    assert marker.WORKFLOW.uispec().header.kind is HeaderKind.LABEL_BAND
    assert enrichment.WORKFLOW.uispec().header.kind is HeaderKind.CENSUS


def test_task_kind_belongs_to_the_method_not_the_use():
    """A reader who points the pathways surface at a marker table does not thereby
    turn an enrichment into an assignment; the task kind came with the workflow the
    method registered into, and there is no per-call override."""
    for m in methods.for_workflow(WORKFLOW_ID):
        assert workflows.get(m.workflow).task_kind is TaskKind.CHARACTERISATION


# --------------------------------------------------------------------------- #
# The no-band rule, four gates, each tried
# --------------------------------------------------------------------------- #
def test_the_characterisation_output_has_no_band_field():
    """A field-name guard, not a behavioural one: it fails when the field is
    DECLARED, which is when the mistake still costs nothing to undo."""
    for t in (Finding, FindingItem):
        names = {f.name for f in dataclasses.fields(t)}
        assert not (names & {"band", "confidence", "confidence_score"}), names
    # The assignment side is the control: it HAS one, and that asymmetry is the
    # whole boundary between the two output types.
    assert "band" in {f.name for f in dataclasses.fields(Assignment)}


def test_a_band_header_on_this_surface_is_unconstructable():
    spec = enrichment.WORKFLOW.uispec()
    assert not hasattr(spec.header, "band_field")
    with pytest.raises(DegradedUpward, match="no label space"):
        dataclasses.replace(spec, header=LabelBandHeader(label_space="nope"))


def test_asking_a_characterisation_method_for_a_band_raises(c1):
    with pytest.raises(jazzpanda_cgst.NoBandForThisTaskKind, match="undefined"):
        jazzpanda_cgst.MODULE.rubric(c1)
    # The assignment method answers the same call with a band, from its own edges.
    assert jazzpanda.MODULE.rubric(c1).value in dict(jazzpanda.BAND_EDGES)


def test_interpret_returns_a_finding_whose_header_is_per_item_counts(c1):
    out = enrichment.WORKFLOW.interpret(c1, enrichment.EnrichmentReference(
        cell_type="Tumor", world=MeasuredWorld((PANEL, CLUSTERS))))
    f = out.value
    assert isinstance(f, Finding) and not isinstance(f, Assignment)
    assert f.task_kind is TaskKind.CHARACTERISATION
    assert "1 enriched, 1 suggestive, 1 not enriched, 1 untestable of 4 set(s)" in \
        f.summary
    assert "top HALLMARK_G2M_CHECKPOINT cgst_z 6.4000 on 4/200 panel genes" in f.summary
    # The mandatory, deterministic panel-scope caveat. A reader forms a genome-wide
    # claim from "enriched for G2M" unless the denominator is said out loud.
    assert "not genome-wide" in f.summary
    # No band word anywhere in the rendered sentence, and no band in the decisions.
    assert not any(b.lower() in f.summary.lower()
                   for b in ("very high", "medium-high", "confidence band"))
    assert not any(d.rule.startswith("band:") for d in out.decisions)
    assert all(d.rule.startswith("enrichment:") for d in out.decisions)


# --------------------------------------------------------------------------- #
# The panel-coverage rule: K of N is a Coverage, and below the floor it refuses
# --------------------------------------------------------------------------- #
def test_panel_coverage_is_counted_against_the_frame_not_read_off_the_table(c1):
    """TPD52 is in the result's ``genes_tested`` and NOT in the panel frame. Taking
    the method's word (3) instead of the apparatus's (2) would put the set above the
    floor on a gene the apparatus could never have measured."""
    fatty = _by_set(c1)["HALLMARK_FATTY_ACID_METABOLISM"]
    panel = next(c for c in fatty.measured.coverage if c.what == "panel_hits")
    assert (panel.k, panel.n, panel.floor) == (2, 158, 3)
    assert not panel.ok


def test_a_below_floor_set_is_underpowered_and_therefore_non_arguing(c1, loaded):
    """The headline rule. Below the floor the set is UNDERPOWERED: no statistic
    survives, no corroborator survives with it, and its absence argues nothing --
    which is a different answer from the set that WAS scored and came back null."""
    fatty = _by_set(c1)["HALLMARK_FATTY_ACID_METABOLISM"]
    assert fatty.measured.state is MeasuredState.UNDERPOWERED
    assert fatty.statistic is None, "a statistic the frame refused must not survive"
    assert fatty.corroborating == (), "a q beside a withheld z is a measurement back in"
    assert not fatty.measured.absence_is_evidence
    # The withholding is RECORDED, naming the number refused: a statistic silently
    # dropped is indistinguishable from a method that reported nothing.
    withheld = [d for d in loaded.decisions if "withheld" in d.outcome]
    assert any("5.0000" in b for d in withheld for b in d.because)

    out = enrichment.WORKFLOW.interpret(c1, enrichment.EnrichmentReference())
    item = next(i for i in out.value.items
                if i.subject.display() == "HALLMARK_FATTY_ACID_METABOLISM")
    assert item.tier is Tier.UNTESTABLE
    assert any("panel_hits 2/158" in b for b in item.because)
    assert out.value.untestable_count == 1


def test_measured_absence_and_untestable_stay_apart(c1):
    """The productive half. APOPTOSIS was scored against c1 and did not clear the
    method's reporting bar -- it argues. FATTY could not be asked -- it does not.
    One tier for both is the collapse the four states exist to stop."""
    items = _by_set(c1)
    apop = items["HALLMARK_APOPTOSIS"]
    assert apop.measured.state is MeasuredState.TESTED_ABSENT
    assert apop.statistic is None and apop.measured.absence_is_evidence

    out = enrichment.WORKFLOW.interpret(c1, enrichment.EnrichmentReference())
    tiers = {i.subject.display(): i.tier for i in out.value.items}
    assert tiers["HALLMARK_APOPTOSIS"] is Tier.NOT_SUPPORTED
    assert tiers["HALLMARK_FATTY_ACID_METABOLISM"] is Tier.UNTESTABLE
    assert tiers["HALLMARK_G2M_CHECKPOINT"] is Tier.SUPPORTED
    assert tiers["HALLMARK_EMT"] is Tier.SUGGESTIVE


def test_verify_comes_from_the_rubric_and_flags_the_weak_call(c1):
    """``verify`` is deterministic and is never offered to a caller as an argument.
    A suggestive program carries it; an untestable one does not -- it is not a call
    a second look could settle."""
    out = enrichment.WORKFLOW.interpret(c1, enrichment.EnrichmentReference())
    verify = {i.subject.display(): i.verify for i in out.value.items}
    assert verify["HALLMARK_EMT"] is True
    assert verify["HALLMARK_FATTY_ACID_METABOLISM"] is False
    # Never offered as an argument: a layer that could set it could also clear it.
    import inspect
    for fn in (enrichment.EnrichmentInterpretation.interpret,
               enrichment.EnrichmentInterpretation.rival_test):
        assert "verify" not in inspect.signature(fn).parameters


# --------------------------------------------------------------------------- #
# The method's declaration
# --------------------------------------------------------------------------- #
def test_the_method_declares_what_it_does_not_measure():
    m = jazzpanda_cgst.MODULE
    low = m.does_not_measure.lower()
    assert "genome-wide" in low and "pathway activity" in low
    assert len(m.assumptions) >= 3
    assert m.statistic_name == "cgst_z"
    # The registry refuses an empty declaration; the Protocol can only see presence.
    assert all(getattr(m, a) for a in methods.MANDATORY_DECLARATIONS)


def test_detect_unit_reads_the_schema_and_refuses_what_it_cannot(loaded):
    assert sorted(loaded.value) == ["c1", "c2"]
    spec = jazzpanda_cgst.MODULE.detect_unit(
        [dict(gene_set="HALLMARK_EMT", cluster="c1", test_statistic=1.0,
              p_adj_bh=0.01, geneset_size=200, genes_tested="LUM",
              genes_selected="LUM")])
    assert (spec.unit_kind, spec.unit_column, spec.units) == ("cluster", "cluster", ("c1",))
    assert spec.because
    with pytest.raises(UnrecognisedResult, match="missing column"):
        jazzpanda_cgst.MODULE.detect_unit([{"gene": "ERBB2", "glm_coef": 2.0}])


def test_absence_reads_as_keeps_the_four_states_apart():
    s = subject("gene_set", set=gene_set("HALLMARK_EMT"))
    said = {
        st: jazzpanda_cgst.MODULE.absence_reads_as(s, Measured(st, frame="f"))
        for st in MeasuredState
    }
    assert len(set(said.values())) == 4
    assert "argues nothing" in said[MeasuredState.NEVER_MEASURED]
    assert "argues against" in said[MeasuredState.TESTED_ABSENT]
    assert "too little" in said[MeasuredState.UNDERPOWERED]


def test_the_method_licenses_no_reference_traversal():
    """An over-representation z says nothing about what a gene marks, and the only
    registered relation today is ``marks``. Following it would let the enrichment
    result argue about identity, which its own declaration forbids."""
    assert jazzpanda_cgst.MODULE.traversals() == ()


# --------------------------------------------------------------------------- #
# The rival test: another PROGRAM, adjudicated by the frames
# --------------------------------------------------------------------------- #
def test_a_rival_program_is_separated_only_by_a_measured_absence(c1):
    ref = enrichment.EnrichmentReference(world=MeasuredWorld((PANEL, CLUSTERS)))
    out = enrichment.WORKFLOW.rival_test(c1, "HALLMARK_APOPTOSIS", ref)
    assert [i.subject.display() for i in out.value.separated_by] == [
        "HALLMARK_APOPTOSIS"]
    assert any(d.rule == "rival:argued_against" for d in out.decisions)


def test_a_program_outside_the_collection_is_undecidable_not_evidence(c1, world):
    """The panel-absence rule at the collection level: a set nobody scored cannot
    argue, and saying so is the productive output, not a failure field."""
    ref = enrichment.EnrichmentReference(world=world)
    out = enrichment.WORKFLOW.rival_test(c1, "HALLMARK_MYOGENESIS", ref)
    assert out.value.separated_by == ()
    assert any("never_measured in collection:" in u for u in out.value.undecidable)
    assert any(d.rule == "rival:undecidable" for d in out.decisions)


def test_a_set_scored_for_another_unit_is_a_measured_absence_here(c1, world):
    """The distinction the collection frame exists for, and the one the rival test
    would otherwise collapse. IFN gamma was scored for c2 and is not among c1's
    rows: in frame, asked, not reported -> it ARGUES. MYOGENESIS was never scored
    -> it does not. Without ``scored_frame`` both read as never measured, and the
    productive half of the lattice silently disappears from this workflow."""
    ref = enrichment.EnrichmentReference(world=world)
    out = enrichment.WORKFLOW.rival_test(
        c1, "HALLMARK_INTERFERON_GAMMA_RESPONSE", ref)
    assert [i.subject.display() for i in out.value.separated_by] == [
        "HALLMARK_INTERFERON_GAMMA_RESPONSE"]
    assert out.value.separated_by[0].measured.state is MeasuredState.TESTED_ABSENT
    assert out.value.separated_by[0].statistic is None
    assert any(d.rule == "rival:argued_against" for d in out.decisions)


def test_a_rival_that_is_also_enriched_separates_nothing(c1):
    ref = enrichment.EnrichmentReference(world=MeasuredWorld((PANEL, CLUSTERS)))
    out = enrichment.WORKFLOW.rival_test(c1, "HALLMARK_G2M_CHECKPOINT", ref)
    assert out.value.separated_by == ()
    assert any("both programs are running" in u for u in out.value.undecidable)
    assert any(d.rule == "rival:ambiguous" for d in out.decisions)


# --------------------------------------------------------------------------- #
# The surface
# --------------------------------------------------------------------------- #
def test_the_surface_can_say_never_measured_and_never_blanks_a_number():
    spec = enrichment.WORKFLOW.uispec()
    assert isinstance(spec, UISpec) and spec.workflow_id == "pathways"
    covered = {s for b in spec.blocks for s in b.statuses}
    assert set(MeasuredState) == covered, "every state needs somewhere to render"
    assert {c.id for c in enrichment.WORKFLOW.checks()} >= {"panel_coverage"}
    # The K of N renders as text on EVERY row, including the ones a numeric column
    # would hatch -- those are the rows whose coverage is the reason for the hatch.
    coverage_col = next(c for c in spec.columns if c.key == "coverage")
    assert coverage_col.kind.value == "text"


def test_every_decision_rule_this_workflow_emits_is_registered():
    assert {r for r in RULES if r.startswith("enrichment:")} == {
        "enrichment:enriched", "enrichment:suggestive",
        "enrichment:not_enriched", "enrichment:untestable"}


# --------------------------------------------------------------------------- #
# The demonstration, printed. Run with ``pytest -s -k demonstration``.
# --------------------------------------------------------------------------- #
def test_the_pair_demonstration(c1, capsys):
    lines: list[str] = []
    for w in (marker.WORKFLOW, enrichment.WORKFLOW):
        spec = w.uispec()
        band = getattr(spec.header, "band_edges_ref", None)
        lines += [
            f"workflow  {w.name}  [{w.task_kind.value}]",
            f"question  {w.question}",
            f"surface   {spec.workflow_id} / {spec.header.kind.value}",
            f"band      {band or 'NONE - no label space, so a unit-level band is undefined'}",
            f"methods   {[m.name for m in methods.for_workflow(w.name)]}",
            "",
        ]
    out = enrichment.WORKFLOW.interpret(c1, enrichment.EnrichmentReference(
        cell_type="Tumor", world=MeasuredWorld((PANEL, CLUSTERS))))
    lines += ["header (per-item counts, no band):", "  " + out.value.summary, ""]
    for i in out.value.items:
        cov = ", ".join(str(c) for c in i.evidence.measured.coverage)
        lines.append(
            f"  {i.subject.display():38} {i.tier.value:14} "
            f"{i.evidence.measured.state.value:15} "
            f"stat={i.evidence.statistic!s:7} verify={str(i.verify):5} {cov}"
        )
    print("\n".join(lines))
    printed = capsys.readouterr().out
    assert "assignment" in printed and "characterisation" in printed
    assert "NONE - no label space" in printed
