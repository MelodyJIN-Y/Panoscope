"""Invariants for the review-surface contract.

The tests are the deliverable here more than anywhere else in ``types/``, because
the properties are about what a surface is FORBIDDEN to declare, and a forbidden
construction leaves no trace once someone deletes the guard.

Four are the point of the module:

* a figure that cannot state its conditions cannot be constructed (R4), at both the
  declaration and the record;
* a surface cannot degrade UPWARD into a claim its task kind does not have (R2) --
  the ``ui/enrichment_table.py`` bug, made unrepresentable;
* a declared surface always has somewhere to render ``never_measured`` (R8);
* a numeric column always hatches where no number exists, so an unmeasured item is
  never a blank cell or a zero.
"""

from __future__ import annotations

import dataclasses

import pytest

import panoscope.types.uispec as U
from panoscope.types.measured import MeasuredState
from panoscope.types.outputs import TaskKind
from panoscope.types.uispec import (
    _REQUIRED_COND, BlankWhereNoNumber, BlockSpec, CensusHeader, ColumnKind,
    ColumnSpec, Conditionality, ConfoundChecklistHeader, DegradedUpward,
    HeaderKind, LabelBandHeader,
    NowhereToSayNeverMeasured, PlotKind, PlotSpec, ProvenanceOnlyHeader, UISpec,
    UndrawableFigure, degraded_spec,
)

COND = dict(
    drawn="transcript detections, pre-segmentation",
    denominator="280 of ~20,000 transcriptome genes on panel",
    transform="50 um hex bins, area-normalised (tx/um2), sqrt colour",
    not_this="not per-cell expression; not a statistic",
)

NEVER_BLOCK = BlockSpec("never", "NEVER MEASURED - {count} off panel",
                        (MeasuredState.NEVER_MEASURED,))
SUPPORTING_BLOCK = BlockSpec("supporting", "SUPPORTING - {count}",
                             (MeasuredState.OBSERVED,))
PLOT = PlotSpec("cell_map", PlotKind.CELL_MAP, "Cell map",
                conditionality_builder="panoscope.ui.cell_map.conditionality",
                accepts_subject_kinds=("gene",))


def _spec(**kw) -> UISpec:
    base = dict(
        workflow_id="markers", title="Marker genes", task_kind=TaskKind.ASSIGNMENT,
        unit_kind="cluster", item_kind="gene",
        interpretive_question="Which cell type is this cluster?",
        header=LabelBandHeader(label_space="cluster_key.json (9 cell types)"),
        columns=(ColumnSpec("gene", "GENE", ColumnKind.TEXT, hatch_when=()),),
        blocks=(SUPPORTING_BLOCK, NEVER_BLOCK),
    )
    return UISpec(**{**base, **kw})


# --------------------------------------------------------------------------- #
# R4: no conditionality record, no plot
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", _REQUIRED_COND)
@pytest.mark.req("ARCH-13.2-01")
def test_conditionality_refuses_to_omit_a_required_field(missing):
    """A volcano on a 280-gene panel is pixel-identical to a genome-wide one, and
    the reader assumes genome-wide. Each of the four answers a question the pixels
    cannot: what was drawn, out of what, through what transform, and what it is not.
    """
    with pytest.raises(UndrawableFigure) as e:
        Conditionality(**{**COND, missing: ""})
    assert missing in str(e.value)


@pytest.mark.parametrize("missing", _REQUIRED_COND)
def test_whitespace_is_not_a_conditionality(missing):
    """The cheapest way past a required field is a space. It is still no record."""
    with pytest.raises(UndrawableFigure):
        Conditionality(**{**COND, missing: "   "})


@pytest.mark.req("UI-2.4-01")
def test_caption_carries_the_denominator():
    """The denominator is the field that matters most and the one that gets
    truncated when the caption shares a line with the transform."""
    line1, line2 = Conditionality(**COND).caption()
    assert COND["drawn"] in line1 and COND["transform"] in line1
    assert COND["denominator"] in line2 and COND["not_this"] in line2


def test_plot_cannot_be_declared_without_a_conditionality_builder():
    """R4 at declaration time, which is the cheap place to enforce it.

    ``Conditionality`` guards the record; this guards the PROMISE. Without it a
    workflow can declare a figure nobody ever wrote a conditionality for, and the
    failure surfaces at render, in front of the biologist.
    """
    with pytest.raises(UndrawableFigure):
        PlotSpec("cell_map", PlotKind.CELL_MAP, "Cell map",
                 conditionality_builder="")
    with pytest.raises(UndrawableFigure):
        PlotSpec("cell_map", PlotKind.CELL_MAP, "Cell map",
                 conditionality_builder="  ")


def test_plot_subject_kinds_are_registered_shapes():
    """``celltype_pair`` is a typo for the registered ``cell_type_pair``; caught at
    declaration it is a traceback, and at render it is an empty view."""
    with pytest.raises(ValueError):
        PlotSpec("x", PlotKind.DOTPLOT, "x", conditionality_builder="a.b",
                 accepts_subject_kinds=("celltype_pair",))


def test_no_spec_type_can_carry_a_computation():
    """"Plots render, they do not derive", structurally: there is nowhere to put a
    formula, so no renderer ever decides what counts as measured.
    """
    offenders = []
    for name in dir(U):
        obj = getattr(U, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            if "Callable" in str(f.type):
                offenders.append(f"{obj.__name__}.{f.name}")
    assert not offenders, "code in a renderer-facing spec: " + ", ".join(offenders)


# --------------------------------------------------------------------------- #
# R2: degradation goes DOWN
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [TaskKind.CHARACTERISATION, TaskKind.COMPARISON])
@pytest.mark.req("CLAUDE-TK-04")
def test_a_no_label_space_surface_cannot_take_an_assignment_header(kind):
    """THE bug: ``ui/enrichment_table.py`` imported the assignment header into a
    characterisation surface and inherited a confidence band it had no right to.

    Both non-assignment kinds are covered, not just characterisation: "does this
    result make sense?" has no label space either, so the band is undefined there
    for exactly the same reason.
    """
    with pytest.raises(DegradedUpward):
        _spec(workflow_id="pathways", task_kind=kind,
              header=LabelBandHeader(label_space="borrowed"))


def test_assignment_must_declare_its_band_rubric():
    """The converse. An assignment routed through a census header is an assignment
    whose band was dropped on the floor rather than computed."""
    with pytest.raises(DegradedUpward):
        _spec(header=CensusHeader())


def test_characterisation_with_a_census_header_is_fine():
    s = _spec(workflow_id="pathways", task_kind=TaskKind.CHARACTERISATION,
              item_kind="gene_set", header=CensusHeader())
    assert s.header.kind is HeaderKind.CENSUS


def test_comparison_with_a_checklist_header_is_fine():
    s = _spec(workflow_id="contrasts", task_kind=TaskKind.COMPARISON,
              unit_kind="contrast", item_kind="gene",
              header=ConfoundChecklistHeader())
    assert s.header.kind is HeaderKind.CONFOUND_CHECKLIST


def test_undeclared_task_kind_may_claim_nothing_else():
    """An unregistered analysis type degrades toward making NO claim. A provenance
    banner on a surface that still declares a task kind is the reverse: a claim
    wearing a disclaimer."""
    with pytest.raises(DegradedUpward):
        _spec(task_kind=None, item_kind=None, header=LabelBandHeader())
    with pytest.raises(DegradedUpward):
        _spec(header=ProvenanceOnlyHeader())


def test_undeclared_task_kind_declares_no_plots_and_no_pins():
    """Rule 3 of the degraded surface: a plot needs a conditionality, nobody
    declared one, R4 applies. And a pin selects a subject this surface cannot name.
    """
    with pytest.raises(DegradedUpward):
        _spec(task_kind=None, item_kind=None, header=ProvenanceOnlyHeader(),
              primary_plot=PLOT)
    with pytest.raises(DegradedUpward):
        _spec(task_kind=None, item_kind=None, header=ProvenanceOnlyHeader(),
              pin_kinds=("gene",))


def test_task_kind_stays_frozen_at_three():
    """``docs/interface-design.md`` §3 sketches a fourth, ``"unknown"``. CLAUDE.md
    freezes the enum at three and ``None`` spells the degraded case, so no
    exhaustive match over TaskKind has to grow a branch meaning "we know nothing".
    """
    assert len(TaskKind) == 3
    assert degraded_spec("x.csv", ()).task_kind is None


# --------------------------------------------------------------------------- #
# R8: somewhere to say "never measured"
# --------------------------------------------------------------------------- #
def test_a_declared_surface_must_render_never_measured():
    """Without a block for it, an off-panel marker is simply missing from the table
    -- indistinguishable from one that was measured and found nothing, which is the
    collapse the four states exist to prevent."""
    with pytest.raises(NowhereToSayNeverMeasured):
        _spec(blocks=(SUPPORTING_BLOCK,))


def test_a_declared_block_must_declare_a_status():
    """An empty ``statuses`` is the degraded spelling; on a registered workflow it
    would render rows with no status at all."""
    with pytest.raises(NowhereToSayNeverMeasured):
        _spec(blocks=(BlockSpec("all", "ROWS - {count}"), NEVER_BLOCK))


def test_coverage_ribbon_orders_every_state():
    """The table's own conditionality. It is what stops a screen of twelve
    supporting markers from reading as twelve out of twelve."""
    assert set(_spec().coverage_ribbon.order) == set(MeasuredState)


# --------------------------------------------------------------------------- #
# Measurement status is visible, not implied
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [ColumnKind.NUMBER, ColumnKind.BAND_MICROBAR,
                                  ColumnKind.COVERAGE_BAR, ColumnKind.SPARKSTRIP])
def test_a_numeric_column_must_hatch_where_no_number_exists(kind):
    """"A never-measured item is not a blank row and not a zero."

    A blank reads as *we looked and found nothing*; a hatch reads as *we could not
    look*. That is the difference between evidence and no evidence, and it is the
    one claim the interface exists to demonstrate.
    """
    with pytest.raises(BlankWhereNoNumber):
        ColumnSpec("glm_coef", "glm_coef", kind, hatch_when=())
    with pytest.raises(BlankWhereNoNumber):   # a partial list is still a blank cell
        ColumnSpec("glm_coef", "glm_coef", kind,
                   hatch_when=(MeasuredState.NEVER_MEASURED,))


def test_the_hatch_default_is_derived_from_the_measurement_rule():
    """Not a second hand-maintained list of states. ``evidence.py`` already decides
    where a statistic may exist; widening that property must move the hatching with
    it, and the copy nobody updates is always the one that governs.
    """
    default = ColumnSpec("glm_coef", "glm_coef", ColumnKind.NUMBER).hatch_when
    assert set(default) == set(MeasuredState) - {MeasuredState.OBSERVED}


def test_a_text_column_need_not_hatch():
    """A gene symbol is known whether or not the gene was measured; hatching it
    would hide the one thing an off-panel row exists to show -- its name."""
    assert ColumnSpec("gene", "GENE", ColumnKind.TEXT, hatch_when=()).hatch_when == ()


# --------------------------------------------------------------------------- #
# The degraded surface
# --------------------------------------------------------------------------- #
def test_the_degraded_surface_claims_nothing():
    """Rules 1-5 of ``degraded_spec``, as one assertion each. Nothing is called
    detected and -- critically -- nothing is called absent, because an unregistered
    method has told us nothing about what it measured."""
    s = degraded_spec("weird_output.csv", ("gene", "score"))
    assert s.header.kind is HeaderKind.PROVENANCE_ONLY      # 1. no header claim
    assert all(b.statuses == () for b in s.blocks)          # 2. no status assigned
    assert s.primary_plot is None and s.detail_plots == ()  # 3. no plots
    assert s.pin_kinds == () and s.csv_columns == ()        # 4. no pin, no export
    assert s.decision_states == ("open",)                   # 4. no sign-off
    assert [c.key for c in s.columns] == ["gene", "score"]  # 5. verbatim, as text
    assert all(c.kind is ColumnKind.TEXT for c in s.columns)
    assert s.coverage_ribbon.enabled is False


def test_the_degraded_surface_still_says_rows_exist():
    """Degrading to no claim is not degrading to a blank screen: the artifact, its
    columns and its row count still render, which is what makes the fallback useful
    enough that nobody is tempted to borrow a real header instead."""
    s = degraded_spec("weird_output.csv", ("gene",))
    assert "{count}" in s.blocks[0].label_template and s.blocks[0].searchable


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #
@pytest.mark.req("IMPL-0-01")
def test_no_mutable_fields_in_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. The record
    determinism invariant cannot hold if any contract carries one."""
    bad = []
    for name in dir(U):
        obj = getattr(U, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            t = str(f.type)
            if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)


def test_specs_are_frozen_and_hashable():
    s = _spec()
    assert {s, _spec()} == {s}
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.task_kind = TaskKind.CHARACTERISATION  # type: ignore[misc]


@pytest.mark.req("IMPL-4-01", "ARCH-13.3-01", gap=True)
def test_the_marker_surface_declares_the_plots_it_draws():
    """interface-design §3.2: the marker surface declares its cell map (primary) and
    density / feature UMAP / dot plot (detail) as ``PlotSpec``s, each with a
    conditionality builder that resolves. Today the app draws all four from
    ``ui/spatial_stage.py`` while the workflow declares none, so no plot passes
    through the R4 refusal the spec puts on it."""
    import importlib

    from panoscope import workflows

    spec = workflows.get("marker-interpretation").uispec()
    plots = ((spec.primary_plot,) if spec.primary_plot else ()) + spec.detail_plots
    assert {p.key for p in plots} >= {"cell_map", "density", "feature_umap", "dotplot"}
    for p in plots:
        mod, _, fn = p.conditionality_builder.rpartition(".")
        assert callable(getattr(importlib.import_module(mod), fn, None)), p.key
