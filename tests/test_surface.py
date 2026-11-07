"""The declared surface, rendered -- and the four guards that bite before it is.

``types/uispec.py`` made the review surface declarable and ``workflows/`` declared
three. Until ``ui/surface.py`` landed, nothing read them: the claim that *"the
review surface is part of the interpretation contract, declared by the workflow"*
had no pixel behind it. These tests are what makes that claim checkable rather than
asserted.

In descending order of what they buy:

1. **Degradation goes DOWN.** A characterisation surface handed the assignment
   header raises. This is the bug ``docs/interface-design.md`` §3.1 names in this
   repo's own code, caught at the renderer -- the end the type system cannot reach,
   because the illegal *declaration* was already unconstructable while the illegal
   *import* was what actually shipped.
2. **A never-measured cell hatches.** Not a blank, which reads as *we looked and
   found nothing*, and not a zero, which reads as a measurement. And a number
   arriving on a row whose state forbids one is refused rather than drawn.
3. **Click-back.** A row or a header sentence with no ``Subject.key`` raises
   instead of rendering as ungrounded prose that looks identical to grounded prose.
4. **Every declared surface renders**, including the degraded one -- parameterised
   over the declarations themselves, so a fourth workflow proves conformance rather
   than merely asserting it.
"""

from __future__ import annotations

import dataclasses

import pytest

from panoscope.types.measured import MeasuredState
from panoscope.types.uispec import (
    ColumnKind,
    ColumnSpec,
    DegradedUpward,
    HeaderKind,
    degraded_spec,
)
from panoscope.ui import enrichment_table as et
from panoscope.ui import surface
from panoscope.workflows import de, enrichment, marker

MARKER = marker.WORKFLOW.uispec()
PATHWAYS = enrichment.WORKFLOW.uispec()
CONTRASTS = de.WORKFLOW.uispec()
DEGRADED = degraded_spec("some_tool_output.csv", ("gene", "score", "note"))

ALL_SPECS = [MARKER, PATHWAYS, CONTRASTS, DEGRADED]


def _row(state: MeasuredState, **values) -> surface.SurfaceRow:
    """A row whose state is given and whose numbers obey it."""
    return surface.SurfaceRow(key=f"gene|g={values.get('gene', 'X')}",
                              state=state, values=values, reason="because")


def _pathway(gene_set="HALLMARK_E2F_TARGETS", *, tier="enriched", score=5.1,
             q=0.001, panel_hits=42, set_size=200, leading=8):
    """A synthetic ``PathwayEvidence``, so the adapter tests touch no data file."""
    from panoscope.agent.types import PathwayEvidence

    return PathwayEvidence(
        gene_set=gene_set, gene_set_collection="MSigDB_Hallmark", score=score,
        score_kind="jazzpanda_enrichment", p_value=q, q_value=q,
        set_size_full=set_size, panel_hits=panel_hits,
        panel_coverage=panel_hits / set_size,
        leading_edge=tuple(f"G{i}" for i in range(leading)), n_leading_edge=leading,
        gc_corr=0.4, tier=tier,
        panel_scope_caveat=f"Panel-scoped: {panel_hits} of {set_size}.",
    )


def _ce():
    """A synthetic ``ClusterEnrichment`` -- one set per state the method produces."""
    from panoscope.agent.types import ClusterEnrichment

    enriched = (_pathway("HALLMARK_E2F_TARGETS"),)
    thin = _pathway("HALLMARK_APICAL_SURFACE", tier="untestable", q=0.4,
                    panel_hits=2, leading=1)
    absent = _pathway("HALLMARK_MYOGENESIS", tier="untestable", q=0.9,
                      panel_hits=40, leading=12)
    # No confidence / confidence_score / demotions: the band on a CHARACTERISATION is
    # deleted, so there is no longer a value this fixture could hand the header to be
    # accidentally rendered. The absence is now structural rather than asserted.
    return ClusterEnrichment(
        cluster="c1", cell_type="Tumor", method="jazzpanda_enrichment",
        enriched=enriched, suggestive=(), all_tested=(*enriched, thin, absent),
        top_theme="HALLMARK_E2F_TARGETS", verify=False, source_trace=(),
    )


def _pathway_header():
    ce = _ce()
    return et._header_values(ce, [et._row("c1", p) for p in ce.all_tested])


# --------------------------------------------------------------------------- #
# 1. Degradation goes DOWN -- the bug the design doc names, caught
# --------------------------------------------------------------------------- #
def test_a_characterisation_surface_cannot_render_an_assignment_header():
    """THE defect. ``ui/enrichment_table.py`` imported the assignment header and
    inherited a confidence band on a task with no label space. Reaching for it from
    the pathways surface now raises instead of drawing a band."""
    with pytest.raises(DegradedUpward, match="no label space"):
        surface.label_band_html(PATHWAYS, {"confidence": "High", "label": "Tumor"})


def test_a_comparison_surface_cannot_render_an_assignment_header_either():
    """The rule is about the task kind, not about the one page that got it wrong."""
    with pytest.raises(DegradedUpward, match="no label space"):
        surface.label_band_html(CONTRASTS, {"confidence": "High"})


def test_the_degraded_surface_cannot_render_an_assignment_header():
    """An unregistered analysis type degrades toward making NO claim -- it may not
    borrow the one task kind that has a band."""
    with pytest.raises(DegradedUpward):
        surface.label_band_html(DEGRADED, {"confidence": "High"})


def test_the_assignment_surface_does_render_its_band():
    """The guard refuses the wrong task kind, not every caller -- otherwise it would
    pass by doing nothing."""
    html = surface.label_band_html(
        MARKER, {"confidence": "High", "label": "Tumor", "key_markers": ("ERBB2",)})
    assert 'class="cf ' in html and "Tumor" in html and "ERBB2" in html


def test_the_pathways_header_carries_no_band_anywhere():
    """Rendered end to end from the real declaration: counts, not a band."""
    html = surface.header_html(PATHWAYS, _pathway_header())
    assert 'class="cf ' not in html
    assert "confidence" not in html.lower()
    assert "enriched" in html and "untestable" in html


def test_a_registered_workflow_cannot_degrade_to_the_provenance_header():
    """Degrading DOWN is also wrong when there is a real interpretation to show:
    hiding a working surface behind a disclaimer is the mirror-image dishonesty."""
    with pytest.raises(DegradedUpward, match="undeclared"):
        surface.provenance_html(PATHWAYS, {})


def test_the_degraded_surface_claims_nothing():
    html = surface.header_html(DEGRADED, {"provenance": ("sha256 9f3a",)})
    assert "Unregistered analysis type" in html
    assert 'class="cf ' not in html
    assert DEGRADED.primary_plot is None and not DEGRADED.pin_kinds
    assert DEGRADED.csv_columns == () and DEGRADED.decision_states == ("open",)


# --------------------------------------------------------------------------- #
# 2. A never-measured item is not a blank row and not a zero
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", [MeasuredState.NEVER_MEASURED,
                                   MeasuredState.TESTED_ABSENT])
def test_a_cell_with_no_number_hatches(state):
    """The headline rule, at the one place a pixel can carry it."""
    col = ColumnSpec("statistic", "glm_coef", ColumnKind.NUMBER)
    cell = surface.cell_html(col, _row(state, statistic=None))
    assert "pano-sf-hatch" in cell
    assert "0.00" not in cell                            # not a zero
    assert cell.strip() and "&mdash;" not in cell        # not a blank


def test_an_underpowered_cell_hatches_too():
    """In frame but below the floor is still no number, and still not a zero."""
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    assert "pano-sf-hatch" in surface.cell_html(
        col, _row(MeasuredState.UNDERPOWERED, statistic=None))


def test_the_hatch_says_why_rather_than_implying_it():
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    row = surface.SurfaceRow(key="gene_set|set=S", state=MeasuredState.UNDERPOWERED,
                             values={"statistic": None},
                             reason="3 of 200 set genes on panel (floor 5)")
    assert "3 of 200 set genes on panel" in surface.cell_html(col, row)


def test_a_number_on_a_row_with_no_measurement_is_refused():
    """``statistic is None`` iff the state is not OBSERVED. The renderer does not
    quietly drop the number -- it refuses, because a caller that produced one has
    a bug the hatch would hide."""
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    with pytest.raises(DegradedUpward, match="only where something was observed"):
        surface.cell_html(col, _row(MeasuredState.NEVER_MEASURED, statistic=4.2))


def test_an_observed_zero_is_a_zero_and_not_a_hatch():
    """The other half. A measured zero is a result and must read as one."""
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    cell = surface.cell_html(col, _row(MeasuredState.OBSERVED, statistic=0.0))
    assert "0.00" in cell and "pano-sf-hatch" not in cell


def test_the_never_measured_block_renders_its_header_at_zero():
    """A block that disappears at zero is the panel-absence rule disappearing at
    the moment it had something to say."""
    block = next(b for b in PATHWAYS.blocks
                 if MeasuredState.NEVER_MEASURED in b.statuses)
    html = surface.block_html(PATHWAYS, block, [_row(MeasuredState.OBSERVED)])
    assert "NEVER SCORED - 0" in html
    assert block.empty_text and block.empty_text in html


@pytest.mark.parametrize("spec", [MARKER, PATHWAYS, CONTRASTS],
                         ids=lambda s: s.workflow_id)
def test_every_declared_surface_can_show_a_never_measured_row(spec):
    """R8 with pixels: the row exists, says so, and carries no number."""
    rows = [surface.SurfaceRow(key="gene|g=SLC1A3",
                               state=MeasuredState.NEVER_MEASURED,
                               values={}, reason="not on this panel")]
    html = surface.table_html(spec, rows)
    assert "pano-sf-hatch" in html and "not on this panel" in html


def test_the_coverage_ribbon_labels_every_count():
    """What stops twelve supporting rows from reading as twelve of twelve."""
    rows = [_row(MeasuredState.OBSERVED, gene="A"),
            _row(MeasuredState.OBSERVED, gene="B"),
            _row(MeasuredState.NEVER_MEASURED, gene="C")]
    html = surface.ribbon_html(MARKER, rows)
    assert "2 observed" in html and "1 never measured" in html and "of 3" in html


def test_sorting_happens_within_a_block_never_across_one():
    """Sorting across blocks is how a top-N hides every absence at the bottom."""
    block = next(b for b in MARKER.blocks if b.sort_by == "statistic")
    rows = [_row(MeasuredState.OBSERVED, gene="LOW", statistic=1.0),
            _row(MeasuredState.NEVER_MEASURED, gene="OFF"),
            _row(MeasuredState.OBSERVED, gene="HIGH", statistic=9.0)]
    got = surface.rows_for(block, rows)
    assert [r.values["gene"] for r in got] == ["HIGH", "LOW"]


# --------------------------------------------------------------------------- #
# 3. Click-back
# --------------------------------------------------------------------------- #
def test_a_row_with_nothing_to_click_is_not_rendered():
    row = surface.SurfaceRow(key="", state=MeasuredState.OBSERVED,
                             values={"gene": "ERBB2", "statistic": 4.2})
    with pytest.raises(surface.NothingToClick):
        surface.row_html(MARKER, row)


def test_a_header_sentence_with_nothing_to_click_is_not_rendered():
    """The census header's top-item line is a claim about a specific item, so it
    resolves to that item or it does not render."""
    values = dict(_pathway_header())
    values.pop("top_key")
    with pytest.raises(surface.NothingToClick):
        surface.census_html(PATHWAYS, values)


def test_every_row_resolves_to_its_subject_key():
    html = surface.row_html(MARKER, _row(MeasuredState.OBSERVED, gene="ERBB2",
                                         statistic=4.2))
    assert 'data-claim="gene|g=ERBB2"' in html


# --------------------------------------------------------------------------- #
# 4. Plots render, they do not derive
# --------------------------------------------------------------------------- #
def test_a_declaration_carrying_a_callable_is_refused():
    """A callable leaking into a column means the renderer is about to decide what
    counts as measured."""
    bad = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    object.__setattr__(bad, "unit", lambda row: row.values["statistic"] * 2)
    spec = dataclasses.replace(MARKER, columns=(*MARKER.columns, bad))
    with pytest.raises(surface.SpecCarriesCode):
        surface.table_html(spec, [_row(MeasuredState.OBSERVED)])


# --------------------------------------------------------------------------- #
# 5. Every declared surface renders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.workflow_id)
def test_a_declared_surface_renders_its_whole_table(spec):
    """Parameterised over the declarations: adding a workflow proves conformance."""
    rows = [surface.SurfaceRow(key=f"k{i}", state=state, values={}, reason="r")
            for i, state in enumerate(MeasuredState)]
    html = surface.table_html(spec, rows)
    for block in spec.blocks:
        assert block.label_template.format(
            count=len(surface.rows_for(block, rows))) in html
    for col in spec.columns:
        assert col.label in html


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.workflow_id)
def test_the_declared_header_variant_is_the_one_rendered(spec):
    """No surface may borrow another task kind's header, in either direction."""
    expected = {
        HeaderKind.LABEL_BAND: surface.label_band_html,
        HeaderKind.CENSUS: surface.census_html,
        HeaderKind.CONFOUND_CHECKLIST: surface.checklist_html,
        HeaderKind.PROVENANCE_ONLY: surface.provenance_html,
    }[spec.header.kind]
    values = {"label": "X", "confidence": "High", "cell_type": "Tumor",
              "top_key": "gene_set|set=S"}
    assert surface.header_html(spec, values) == expected(spec, values)


def test_the_renderer_never_invents_a_column_the_spec_did_not_declare():
    """The declaration is the whole contract: a value with no column is not drawn."""
    rows = [_row(MeasuredState.OBSERVED, gene="ERBB2", statistic=4.2,
                 secret_field="NOT_DECLARED")]
    assert "NOT_DECLARED" not in surface.table_html(MARKER, rows)


# --------------------------------------------------------------------------- #
# 6. The wired page: the Pathways adapter
# --------------------------------------------------------------------------- #
def test_the_adapter_keeps_the_two_ways_of_failing_apart():
    """Below the capability floor and tested-but-not-reported are different facts:
    one argues nothing, the other argues against the program running here."""
    states = {p.gene_set: p.measured_state for p in _ce().all_tested}
    assert states["HALLMARK_E2F_TARGETS"] is MeasuredState.OBSERVED
    assert states["HALLMARK_APICAL_SURFACE"] is MeasuredState.UNDERPOWERED
    assert states["HALLMARK_MYOGENESIS"] is MeasuredState.TESTED_ABSENT


def test_a_set_the_test_never_scored_is_not_tested_absent():
    """THREE ways to fail, not two, and only one of them argues.

    ``tier == "untestable"`` catches a set below a capability floor, a set scored
    above the q bar, AND a set for which the test returned no q at all. The
    adapter read the floors and then fell through, so that last row -- never
    scored -- resolved to TESTED_ABSENT and ``_reason`` told the biologist
    "absence IS evidence here" about a test that produced no result. That is the
    panel-absence rule inverted on the second workflow.

    Neither branch occurs in the bundled demo (0 of 20 untestable rows), which is
    why nothing caught it and why this fixture is synthetic.
    """
    from panoscope.agent import config as cfg

    unscored = _pathway("HALLMARK_HYPOXIA", tier="untestable", q=None,
                        panel_hits=40, leading=12)
    assert unscored.panel_hits >= cfg.MIN_PANEL_HITS
    assert unscored.n_leading_edge >= cfg.MIN_LEADING_EDGE, (
        "the row must clear both floors, or it would be UNDERPOWERED for the "
        "other reason and the test would prove nothing"
    )

    state = unscored.measured_state
    assert state is MeasuredState.NEVER_MEASURED, state
    assert "absence IS evidence" not in et._reason(unscored, state)

    # ...and the row that WAS scored still argues, or the fix cost the productive half.
    scored = _pathway("HALLMARK_MYOGENESIS", tier="untestable", q=0.9,
                      panel_hits=40, leading=12)
    assert scored.measured_state is MeasuredState.TESTED_ABSENT
    assert "absence IS evidence" in et._reason(scored, scored.measured_state)


def test_the_adapter_withholds_a_number_where_the_state_forbids_one():
    rows = {p.gene_set: et._row("c1", p) for p in _ce().all_tested}
    assert rows["HALLMARK_E2F_TARGETS"].values["statistic"] == 5.1
    for gs in ("HALLMARK_APICAL_SURFACE", "HALLMARK_MYOGENESIS"):
        assert rows[gs].values["statistic"] is None
        assert rows[gs].values["p_adj_bh"] is None
        # ...but the K of N that EXPLAINS the missing number is still shown.
        assert rows[gs].values["coverage"]


def test_the_pathways_page_renders_untestable_programs_as_rows():
    """They used to not render at all, which is indistinguishable from a set that
    was tested and found nothing."""
    ce = _ce()
    rows = [et._row("c1", p) for p in ce.all_tested]
    html = surface.table_html(PATHWAYS, rows)
    assert "Apical Surface" in html and "Myogenesis" in html
    assert "TOO LITTLE OF THE SET ON PANEL - 1" in html
    assert "SCORED, NOT ENRICHED HERE - 1" in html
    assert "pano-sf-hatch" in html


def test_the_pathways_page_no_longer_imports_the_assignment_header():
    """The defect was an import, so this is asserted against the source: a renderer
    that cannot reach the assignment header is the fix, not a comment."""
    import inspect

    src = inspect.getsource(et)
    assert "verdict_header" not in src
    assert "confidence_chip" not in src


def test_the_pathways_header_attributes_the_imported_cell_type():
    """The marker workflow's call, shown as context and never re-derived here."""
    html = surface.header_html(PATHWAYS, _pathway_header())
    assert "Tumor" in html
    assert "not re-derived here" in html


def test_the_pathways_header_names_the_gates_it_applied():
    html = surface.header_html(PATHWAYS, _pathway_header())
    assert "enriched: q &lt; 0.05" in html
    assert "testable: &gt;= 3 set genes on panel" in html
