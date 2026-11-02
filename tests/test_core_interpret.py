"""The three modules that finish ``core/``, and the one invariant each of them owns.

All three are thin, and two of them are thin because the work was already done
somewhere better. That is the finding these tests are written to keep true, so each
module is tested for the thing it adds and explicitly *not* re-tested for the thing
it delegates: there is no band assertion here beyond "the band the workflow computed
came back unchanged", because ``tests/test_core_verdict.py`` already drives the
ladder against all nine demo clusters, and a second set of band expectations in this
file would be the duplicate that drifts.

**Real demo evidence, not a fixture, for the two headline cases.** ``c4``'s markers
come out of the committed ``inputs/markers_top.csv`` through ``core.measure``, and
its gene sets out of the committed ``interp/enrichment.csv``; the specificity flags
are raised by ``methods/jazzpanda._flags_for`` on the real ``pearson`` and
``max_gc_corr`` columns rather than written by hand. Nothing in this file invents a
number. The two synthetic workflows further down are deliberately *mis*-declared --
they exist only to make each guard fire, and each one is a shape that is
constructible today and that nothing else in the repo refuses.
"""

from __future__ import annotations

import csv
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from panoscope.agent import annotation, data
from panoscope.core import failure_modes, propagate
from panoscope.core.interpret import ShapeContradictsTaskKind, interpret
from panoscope.core.measure import Reading, measure_unit
from panoscope.core.verdict import assess
from panoscope.methods import jazzpanda
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.evidence import FLAGS, NormalisedEvidence
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.outputs import Assignment, Finding, FindingItem, TaskKind, Tier
from panoscope.types.subject import EntityKind, cell_type, gene, gene_set, subject
from panoscope.types.trace import RULES, Trace, Traced
from panoscope.workflows import enrichment, marker

PKG = Path(__file__).resolve().parents[1] / "panoscope"
STUDY = (Path(__file__).resolve().parents[1] / "data" / "studies"
         / "xenium_hbreast_sample1")

#: The unit both headline cases are read for. c4 is the only demo cluster with a
#: double-figure number of surfaced gene sets, so the enrichment Finding has both
#: halves of the lattice in it rather than only the reported one.
UNIT = "c4"

#: The unit the failure-mode index is read for. c9 is the demo study's two-marker
#: cluster, so its real verdict fires ``band:fragile_floor`` *and* its real markers
#: carry a ``specificity:*`` flag -- one mode from each of the two sources
#: ``failure_modes.detect`` reads, which is what makes the ordering claim testable
#: at all. Every other demo cluster fires exactly one mode, and a one-element tuple
#: is in every order at once.
THIN_UNIT = "c9"

PANEL_FRAME = "panel:demo"
DETECT_FRAME = "detect:nosig"
COLLECTION_FRAME = "collection:demo"


# --------------------------------------------------------------------------- #
# Real demo evidence
# --------------------------------------------------------------------------- #
def _enrichment_rows() -> list[dict[str, str]]:
    with (STUDY / "interp" / "enrichment.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def world() -> MeasuredWorld:
    """The demo study's capability frames, every one derived from a file.

    Four frames and not three: the panel says what could be looked at,
    ``detect:nosig`` says what looking at it produced, the annotation label set is
    the cell-type frame, and the collection is the gene-set frame. Panel size is
    read, never asserted -- a hardcoded 280 is a constant that is wrong for the
    second study.
    """
    markers = data.load_markers()
    return MeasuredWorld((
        Frame(EntityKind.GENE, PANEL_FRAME,
              frozenset(data.load_panel()["gene"].astype(str))),
        Frame(EntityKind.GENE, DETECT_FRAME,
              frozenset(markers[markers["top_cluster"] != "NoSig"]["gene"].astype(str))),
        Frame(EntityKind.CELL_TYPE, "annotation:cluster_key",
              frozenset(annotation.all_canonical())),
        Frame(EntityKind.GENE_SET, COLLECTION_FRAME,
              frozenset(r["gene_set"] for r in _enrichment_rows())),
    ))


def _conditions(world: MeasuredWorld, method: str) -> Conditions:
    """The demo study's conditions. ``segmentation`` and ``fixation`` are genuinely
    unknown here -- §6.3's declared-unknown-OK half -- which is what gives
    ``core.propagate`` something real to propagate below."""
    return Conditions(
        assay=Condition("xenium", Provenance.READ),
        modality=Condition("spatial_transcriptomics", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=Condition(method, Provenance.STATED),
        segmentation=UNKNOWN, fixation=UNKNOWN,
        frames=tuple(f.ref() for f in world.frames),
    )


def _marker_evidence(unit: str, world: MeasuredWorld) -> NormalisedEvidence:
    """One cluster's jazzPanda output, flags and all, through the real measure step.

    The ``specificity:*`` flags come from ``methods/jazzpanda._flags_for`` on the
    row's own ``pearson``/``max_gc_corr``/``max_gg_corr``. They are raised by the
    method layer at load in production, which is the half of ``failure_modes.detect``
    that a walk over decisions alone would never see.
    """
    rows = data.get_cluster_markers(unit)
    readings = tuple(
        Reading(subject=subject("gene", g=gene(str(r.gene))),
                statistic=float(r.glm_coef), statistic_name="glm_coef",
                corroborating=(("pearson", float(r.pearson)),),
                flags=jazzpanda._flags_for(float(r.pearson), float(r.max_gc_corr),
                                           float(r.max_gg_corr)))
        for r in rows.itertuples()
    )
    call = data.cell_type_for(unit)
    reported = (
        Frame(EntityKind.GENE, f"reported:{unit}",
              frozenset(str(r.gene) for r in rows.itertuples())),
        Frame(EntityKind.CELL_TYPE, f"reported:{unit}", frozenset({call})),
    )
    return measure_unit(
        unit=unit, unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type(call)), module="jazzpanda",
        readings=readings, world=world, reported=reported,
        conditions=_conditions(world, "jazzpanda"),
    ).value


@pytest.fixture
def marker_evidence(world: MeasuredWorld) -> NormalisedEvidence:
    return _marker_evidence(UNIT, world)


@pytest.fixture
def thin_evidence(world: MeasuredWorld) -> NormalisedEvidence:
    """``c9``: two assigned markers, so the band is capped and floored for real."""
    return _marker_evidence(THIN_UNIT, world)


@pytest.fixture
def enrichment_evidence(world: MeasuredWorld) -> NormalisedEvidence:
    """``c4``'s gene sets, plus every set in the collection it did NOT surface.

    The unsurfaced ones are built as numberless readings on purpose: they are in the
    collection frame and were not reported for this unit, which is ``TESTED_ABSENT``
    -- the productive half. Dropping them would make the Finding read as "ten
    programs, all enriched" when the honest headline is ten of twenty-four.

    ``q_value`` is carried across as ``p_adj_bh`` because that is the corroborator
    name ``methods/jazzpanda_cgst`` declares; renaming a column at the loader
    boundary is what a method module is for. No number is changed.
    """
    rows = _enrichment_rows()
    mine = [r for r in rows if r["cluster"] == UNIT]
    surfaced = {r["gene_set"] for r in mine}
    readings = tuple(
        Reading(subject=subject("gene_set", set=gene_set(r["gene_set"])),
                statistic=float(r["score"]), statistic_name=r["score_kind"],
                corroborating=(("p_adj_bh", float(r["q_value"])),),
                coverage=(Coverage("panel_hits", int(r["panel_hits"]),
                                   int(r["set_size_full"]), 3),))
        for r in mine
    ) + tuple(
        Reading(subject=subject("gene_set", set=gene_set(name)), statistic=None,
                statistic_name=mine[0]["score_kind"])
        for name in sorted({r["gene_set"] for r in rows} - surfaced)
    )
    return measure_unit(
        unit=UNIT, unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type(data.cell_type_for(UNIT))),
        module="jazzpanda_cgst", readings=readings, world=world,
        reported=(Frame(EntityKind.GENE_SET, f"reported:{UNIT}",
                        frozenset(surfaced)),),
        conditions=_conditions(world, "jazzpanda_cgst"),
    ).value


# --------------------------------------------------------------------------- #
# core/interpret.py -- both shapes, from one call, over real demo evidence
# --------------------------------------------------------------------------- #
def test_an_assignment_comes_back_for_an_assignment_workflow(marker_evidence, world):
    """The ASSIGNMENT branch, end to end on the real study.

    Every assertion is about the shape and the provenance of the value, not about
    the band: ``test_core_verdict.py`` owns the ladder, and re-asserting a band here
    would make a tuning change fail in two files for one reason.
    """
    call = data.cell_type_for(UNIT)
    out = interpret(marker_evidence, marker.WORKFLOW,
                    marker.MarkerReference(label=call,
                                           canonical=annotation.canonical_markers(call),
                                           world=world))
    a = out.value
    assert isinstance(a, Assignment)
    assert (a.unit, a.unit_kind, a.label) == (UNIT, "cluster", call)
    assert a.band in {lab for lab, _edge in jazzpanda.BAND_EDGES}
    assert a.drivers and all(d.statistic is not None for d in a.drivers)
    assert all(u.statistic is None for u in a.unmeasured)
    assert a.rationale.startswith(f"{UNIT} reads as {call} at {a.band}")
    assert any(d.rule.startswith("band:") for d in out.decisions)


def test_a_finding_comes_back_for_a_characterisation_workflow(enrichment_evidence):
    """The no-band branch, end to end on the real study.

    The field-name check is the point and it is checked on the instance rather than
    trusted from the type: ``Finding`` is the shape a characterisation is allowed to
    come back in precisely because there is nowhere in it to put a unit-level
    confidence.
    """
    out = interpret(enrichment_evidence, enrichment.WORKFLOW,
                    enrichment.EnrichmentReference(cell_type=data.cell_type_for(UNIT)))
    f = out.value
    assert isinstance(f, Finding)
    assert f.task_kind is TaskKind.CHARACTERISATION
    for forbidden in ("band", "confidence", "confidence_score"):
        assert not hasattr(f, forbidden), f"a Finding grew a {forbidden!r} field"

    tiers = {i.tier for i in f.items}
    assert Tier.SUPPORTED in tiers, "no set cleared the enriched bar on the real data"
    assert Tier.NOT_SUPPORTED in tiers, (
        "every set in the collection surfaced for this unit -- the measured-absent "
        "half of the lattice is not being exercised"
    )
    assert "Panel-scoped" in f.summary


def test_one_entry_point_produces_both_shapes_for_one_unit(marker_evidence,
                                                           enrichment_evidence, world):
    """§6's signature, doing the only thing it is for: the caller does not branch.

    Same function, same unit, two workflows, two shapes -- and the shape is decided
    by the workflow's declaration rather than by anything the caller passed.
    """
    call = data.cell_type_for(UNIT)
    pairs = (
        (marker_evidence, marker.WORKFLOW,
         marker.MarkerReference(label=call,
                                canonical=annotation.canonical_markers(call),
                                world=world)),
        (enrichment_evidence, enrichment.WORKFLOW,
         enrichment.EnrichmentReference(cell_type=call)),
    )
    shapes = [type(interpret(ev, wf, ref).value).__name__ for ev, wf, ref in pairs]
    assert shapes == ["Assignment", "Finding"]


def test_interpret_records_no_decision_of_its_own(marker_evidence, world):
    """The dispatcher transcribes; it does not narrate.

    A decision added here would restate ``InterpretationRecord.task_kind``, which the
    record already carries. Equality against the workflow's own trace is the
    strongest form of that: not "roughly the same", the same tuple.
    """
    call = data.cell_type_for(UNIT)
    ref_ = marker.MarkerReference(label=call,
                                  canonical=annotation.canonical_markers(call),
                                  world=world)
    direct = marker.WORKFLOW.interpret(marker_evidence, ref_)
    through = interpret(marker_evidence, marker.WORKFLOW, ref_)
    assert through.decisions == direct.decisions
    assert through.value == direct.value


# --------------------------------------------------------------------------- #
# core/interpret.py -- the guard, which is the whole reason the module exists
# --------------------------------------------------------------------------- #
def _finding(ev: NormalisedEvidence, task_kind: TaskKind) -> Finding:
    return Finding(unit=ev.unit, unit_kind=ev.unit_kind, task_kind=task_kind,
                   items=(), summary="")


def _assignment(ev: NormalisedEvidence) -> Assignment:
    return Assignment(unit=ev.unit, unit_kind=ev.unit_kind, label="Tumor",
                      label_id=None, band="High", verify=False, drivers=(),
                      unmeasured=(), rationale="")


@dataclasses.dataclass(frozen=True, slots=True)
class _Miswired:
    """A workflow whose declaration and output shape disagree.

    Every combination below is constructible today and is refused by nothing else:
    ``Finding`` guards only ``task_kind=ASSIGNMENT``, ``Assignment`` guards only the
    drivers partition, and ``UISpec`` is not consulted on this path at all.
    """

    name: str
    task_kind: TaskKind
    returns: Any
    question: str = "a question nobody should be able to ask this way"
    skill_path: str = ""

    def interpret(self, ev: NormalisedEvidence, reference: Any) -> Traced[Any]:
        t = Trace()
        t.note("measure:observed", "a decision made before the shape was checked")
        return t.done(self.returns(ev))


def test_a_characterisation_may_not_return_an_assignment(marker_evidence):
    """The band that would otherwise land on a question with no label space.

    This is the hole the three existing gates leave: the ``Assignment`` is
    well-formed, its band is well-formed, and nothing it touches knows there was no
    label space for the remaining probability mass to sit on.
    """
    wf = _Miswired("miswired-characterisation", TaskKind.CHARACTERISATION, _assignment)
    with pytest.raises(ShapeContradictsTaskKind, match="characterisation"):
        interpret(marker_evidence, wf, None)


def test_an_assignment_may_not_return_a_finding(marker_evidence):
    """The mirror, which is quieter and no better: the unit reads as interpreted
    while the one number the biologist acts on was never computed."""
    wf = _Miswired("miswired-assignment", TaskKind.ASSIGNMENT,
                   lambda ev: _finding(ev, TaskKind.CHARACTERISATION))
    with pytest.raises(ShapeContradictsTaskKind, match="dropped on the floor"):
        interpret(marker_evidence, wf, None)


def test_a_finding_must_carry_the_task_kind_its_workflow_declared(marker_evidence):
    """The second check on the ``Finding`` branch, which is not redundant.

    ``Finding`` refuses only ASSIGNMENT, so a COMPARISON workflow stamping
    CHARACTERISATION is well-formed -- and the stamp is what the record and the
    review surface route on, so the DE audit would render on the enrichment surface
    with every confound check silently not shown.
    """
    wf = _Miswired("miswired-comparison", TaskKind.COMPARISON,
                   lambda ev: _finding(ev, TaskKind.CHARACTERISATION))
    with pytest.raises(ShapeContradictsTaskKind, match="stamped characterisation"):
        interpret(marker_evidence, wf, None)


@pytest.mark.parametrize("task_kind",
                         [TaskKind.CHARACTERISATION, TaskKind.COMPARISON])
def test_a_correctly_declared_workflow_passes(marker_evidence, task_kind):
    """Guard against a guard that refuses everything: both no-band kinds go through,
    and the callee's decisions arrive spliced rather than dropped."""
    wf = _Miswired(f"well-formed-{task_kind.value}", task_kind,
                   lambda ev: _finding(ev, task_kind))
    out = interpret(marker_evidence, wf, None)
    assert out.value.task_kind is task_kind
    assert [d.rule for d in out.decisions] == ["measure:observed"]


# --------------------------------------------------------------------------- #
# core/failure_modes.py -- an index, so the test is that it indexes live things
# --------------------------------------------------------------------------- #
def test_every_detector_id_is_registered():
    """A mode pointing at an id nothing can emit is a badge nobody can test.

    Exactly one registry per id, too: a detector in both ``RULES`` and ``FLAGS``
    would mean two layers claiming the same name, and ``detect`` could not say which
    of them fired.
    """
    for mode in failure_modes.MODES.values():
        for d in mode.detected_by:
            assert (d in RULES) ^ (d in FLAGS), (
                f"{mode.id} -> {d!r} is in neither RULES nor FLAGS, or in both"
            )


def test_no_mode_is_dead_vocabulary():
    """The repo's own rule, applied to this module: a registry nobody emits is a
    traversal nobody can test.

    The scan is for the id as a source literal anywhere in ``panoscope/`` outside the
    two registries and outside this index -- not for a literal passed to ``note``,
    because ``core/verdict.py`` assigns ``band:driver_nosig`` to a local and notes
    the local, which is an emitter all the same.
    """
    skip = {"types/trace.py", "types/evidence.py", "core/failure_modes.py"}
    sources = [
        p.read_text(encoding="utf-8") for p in PKG.rglob("*.py")
        if str(p.relative_to(PKG)).replace("\\", "/") not in skip
    ]
    orphans = [
        f"{mode.id} -> {d}"
        for mode in failure_modes.MODES.values()
        for d in mode.detected_by
        if not any(f'"{d}"' in src or f"'{d}'" in src for src in sources)
    ]
    assert not orphans, ("failure modes naming detectors nobody emits: "
                         + "; ".join(orphans))


def _thin_verdict(thin_evidence: NormalisedEvidence):
    call = data.cell_type_for(THIN_UNIT)
    return assess(thin_evidence.items, frozenset(annotation.canonical_markers(call)))


def test_detect_reads_both_the_flags_and_the_decisions(thin_evidence):
    """The real c9 evidence and its real verdict, together.

    Two sources, two modes: ``marker:nonspecific`` is reachable only from the
    ``EvidenceItem`` flags the method layer raised at load, and ``unit:thin_profile``
    only from the ``band:fragile_floor`` decision ``core.verdict`` makes about the
    unit. A traversal over either source alone reports half of what the record
    already says about this cluster.
    """
    verdict = _thin_verdict(thin_evidence)
    from_both = {m.id for m in failure_modes.detect(thin_evidence, verdict).value}
    from_flags = {m.id for m in failure_modes.detect(thin_evidence).value}

    assert from_flags == {"marker:nonspecific"}, (
        "no specificity flag survived from the method layer into the index, or a "
        "mode fired from evidence that carries no decisions"
    )
    assert from_both == {"marker:nonspecific", "unit:thin_profile"}, (
        "the decisions half of the traversal is not reading the verdict trace"
    )
    assert all(d in FLAGS or d in RULES
               for mid in from_both for d in failure_modes.MODES[mid].detected_by)


def test_detect_is_in_registry_order_not_firing_order(thin_evidence):
    """Firing order is a property of which trace the caller passed first, which is a
    property of the call site. Registry order is a property of this module.

    c9 is the unit this can be asked of: it fires two modes, and they are declared
    in ``MODES`` in the opposite order to the one they fire in -- the flag is raised
    at load, long before the ``band:fragile_floor`` decision, and ``unit:thin_profile``
    is declared first. A single-mode unit is in every order at once and proves
    nothing.
    """
    verdict = _thin_verdict(thin_evidence)
    ids = [m.id for m in failure_modes.detect(thin_evidence, verdict).value]
    assert len(ids) > 1, "the ordering claim is vacuous on a single-mode unit"
    assert ids == [m for m in failure_modes.MODES if m in set(ids)]


def test_detect_records_no_decision(thin_evidence):
    """Every mode restates a decision the record already carries; a second line
    saying the same thing is the copy that drifts."""
    verdict = _thin_verdict(thin_evidence)
    assert verdict.decisions, "the verdict this reads carries no decisions at all"
    assert failure_modes.detect(thin_evidence, verdict).decisions == ()


def test_explain_refuses_an_unregistered_mode():
    with pytest.raises(failure_modes.UnregisteredDetector, match="no failure mode"):
        failure_modes.explain("unit:invented_just_now")


def test_explain_refuses_a_mode_whose_detector_left_the_registry(monkeypatch):
    """The failure this guard is for is a RENAME in ``RULES``, which would otherwise
    leave a mode silently un-raisable while every test stayed green."""
    broken = dataclasses.replace(failure_modes.MODES["unit:thin_profile"],
                                 detected_by=("band:renamed_yesterday",))
    monkeypatch.setitem(failure_modes.MODES, "unit:thin_profile", broken)
    with pytest.raises(failure_modes.UnregisteredDetector, match="neither RULES"):
        failure_modes.explain("unit:thin_profile")


def test_the_undetected_modes_are_named_rather_than_absent():
    """§9.8 lists four; two have no detector in this repo. Naming them is what lets
    a reviewer tell "asked and nothing found" from "never asked"."""
    assert len(failure_modes.UNDETECTED) == 2
    assert all("no " in u for u in failure_modes.UNDETECTED), (
        "an undetected mode does not say WHY it cannot be detected"
    )
    assert not any(u in failure_modes.MODES for u in failure_modes.UNDETECTED)


# --------------------------------------------------------------------------- #
# core/propagate.py -- the edge from Conditions into a check
# --------------------------------------------------------------------------- #
@pytest.fixture
def conditions(world: MeasuredWorld) -> Conditions:
    return _conditions(world, "jazzpanda")


def test_a_declared_unknown_cannot_be_settled_and_names_itself(conditions):
    """CLAUDE.md's rule, as one call. The demo study genuinely does not know its
    segmentation version, so this is the real case and not a constructed one."""
    out = propagate.requires(conditions, "segmentation")
    assert not out.value.settled
    assert out.value.reading == ("segmentation",)
    assert len(out.value.missing) == 1
    assert out.value.missing[0].startswith("segmentation:")
    assert [d.rule for d in out.decisions] == ["propagate:cannot_be_settled"]


def test_a_known_condition_settles_and_still_says_what_it_read(conditions):
    """A check that names its inputs only when they are absent cannot be audited
    when it passes."""
    out = propagate.requires(conditions, "assay", "modality")
    assert out.value.settled
    assert out.value.reading == ("assay", "modality")
    assert out.value.missing == ()
    assert [d.rule for d in out.decisions] == ["propagate:conditions_known"]


def test_one_known_condition_never_settles_an_unknown_one(conditions):
    """The meet, not a majority. An unknown never silently applies because another
    field beside it happened to be known."""
    out = propagate.requires(conditions, "tissue", "fixation")
    assert not out.value.settled
    assert out.value.reading == ("tissue", "fixation")
    assert [m.split(":")[0] for m in out.value.missing] == ["fixation"]


def test_a_blocking_field_is_settled_by_construction(conditions):
    """§6.3's line, one layer down: ``Conditions`` cannot be built with an unknown
    ``assay`` or ``tissue``, so a check declaring them can never be blocked on them."""
    assert propagate.requires(conditions, "assay", "tissue").value.settled


def test_order_is_conditions_order_and_not_the_callers(conditions):
    """``missing`` is hashed into whatever record it is spliced into. Two checks
    reading the same two fields must not write two different records of one gap."""
    a = propagate.requires(conditions, "fixation", "segmentation")
    b = propagate.requires(conditions, "segmentation", "fixation")
    assert a.value == b.value
    assert a.decisions == b.decisions
    assert [m.split(":")[0] for m in a.value.missing] == ["segmentation", "fixation"]


def test_a_field_that_is_not_a_condition_raises(conditions):
    """A typo in a declared dependency would make the check read fewer conditions
    than it claims, and settle on an unknown it never looked at."""
    with pytest.raises(propagate.UnknownConditionField,
                       match="not a Conditions field"):
        propagate.requires(conditions, "segmentaton")


def test_the_field_list_is_derived_from_the_section_63_table():
    """Not a second list. A seventh ``Condition`` must land in one of the two §6.3
    tuples to exist at all, so it appears here without anyone remembering to add it."""
    declared = {f.name for f in dataclasses.fields(Conditions)} - {"frames"}
    assert set(propagate.CONDITION_FIELDS) == declared


def test_declaring_nothing_is_settled_and_says_so(conditions):
    """A degenerate but reachable call: a check with no condition dependencies is
    settled, and the record says it declared none rather than going silent."""
    out = propagate.requires(conditions)
    assert out.value.settled and out.value.reading == ()
    assert "declared none" in out.decisions[0].because[0]


# --------------------------------------------------------------------------- #
# The shared contract, re-checked on the three new modules specifically
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fn", [interpret, failure_modes.detect,
                                failure_modes.explain, propagate.requires])
def test_the_new_core_functions_return_traced(fn):
    """``tests/test_contracts.py`` scans the whole package; this is the same rule
    named on the four functions this file adds, so a regression here says which."""
    import typing
    assert typing.get_origin(typing.get_type_hints(fn)["return"]) is Traced


def test_a_mistiered_item_is_still_unrepresentable(marker_evidence):
    """Not this module's guard, and that is the point: ``interpret`` composes, so a
    tier that contradicts its measurement must still be refused by ``FindingItem``
    rather than by anything added here."""
    absent = Measured(MeasuredState.NEVER_MEASURED, PANEL_FRAME)
    item = dataclasses.replace(marker_evidence.items[0], statistic=None,
                               measured=absent, corroborating=())
    with pytest.raises(Exception):
        FindingItem(evidence=item, tier=Tier.SUPPORTED, verify=False)
