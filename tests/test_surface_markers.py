"""The markers page, rendered from its declaration — and the asymmetry with Pathways.

``tests/test_surface.py`` pins the generic renderer and the CHARACTERISATION
conversion. This file pins the ASSIGNMENT one, which differs in exactly two ways
and must keep differing:

* **It keeps its band**, because it is the only task kind with a label space.
  ``surface.label_band_html`` refuses every other, so the asymmetry is structural.
* **Its absences are measurement states, not a sibling field.** The page used to
  render ``ClusterVerdict.evidence`` — the v1 compatibility view, which carries only
  the rows that were measured. A canonical marker the panel never carried lived in
  ``offpanel_notes``, which this table never read, so it rendered as nothing at all.
  The rows now come from the record's own evidence and a never-measured marker is a
  hatched row that names the frame that refused it.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from panoscope.types.measured import MeasuredState
from panoscope.types.uispec import DegradedUpward
from panoscope.ui import evidence_table as et
from panoscope.ui import surface
from panoscope.workflows import enrichment as enrichment_workflow
from panoscope.workflows import marker as marker_workflow

MARKERS = marker_workflow.WORKFLOW.uispec()
PATHWAYS = enrichment_workflow.WORKFLOW.uispec()

#: The demo unit with off-panel canonical markers. c2 is CAF: five of the
#: fibroblast canon (COL1A1, COL1A2, DCN, VIM, FAP) are not on the breast panel.
UNIT_WITH_ABSENCES = "c2"


@pytest.fixture(scope="module")
def rows():
    built = et.unit_build(UNIT_WITH_ABSENCES)
    assert built is not None, "the demo study must produce a record for c2"
    return et.rows_for(UNIT_WITH_ABSENCES, built)


# --------------------------------------------------------------------------- #
# The never-measured rows: the thing that previously did not render at all
# --------------------------------------------------------------------------- #
def test_a_never_measured_marker_is_a_row_not_a_blank_and_not_a_zero(rows):
    never = [r for r in rows if r.state is MeasuredState.NEVER_MEASURED]
    assert never, "c2's off-panel canonical markers must reach the surface as rows"
    html = surface.table_html(MARKERS, rows)
    for row in never:
        assert row.values["gene"] in html
        # No number, and no zero standing in for one.
        assert row.values["statistic"] is None
        assert row.values["pearson"] is None


def test_every_row_lands_in_exactly_one_block(rows):
    """The blocks partition the rows. Nothing is shown twice, nothing disappears.

    The second half is the one worth having. ``never_measured`` and ``undetected``
    select the same STATE and are separated only by ``frame_prefix``, so a row whose
    adapter forgot to set ``SurfaceRow.frame`` matches neither and silently vanishes
    from the table while the coverage ribbon still counts it. A defect that leaves
    the remaining numbers correct is the kind nobody finds.
    """
    counts = {b.key: len(surface.rows_for(b, rows)) for b in MARKERS.blocks}
    assert sum(counts.values()) == len(rows), (
        f"{len(rows)} rows but the blocks hold {sum(counts.values())}: {counts}"
    )


def test_off_panel_and_undetected_are_different_blocks_and_say_so(rows):
    """NEVER_MEASURED is two different sentences, and one header must not pick one.

    The marker table read "NEVER MEASURED - 49 off panel" for 49 genes that are all
    ON the 280-gene panel: they were probed and produced no usable spatial signal.
    Seven of the nine demo clusters showed it, and c1's genuinely off-panel count is
    zero. ``platforms/xenium.py`` names that collapse as the four-states-into-a-
    boolean mistake one layer down; this pins the split that prevents it.
    """
    by_key = {b.key: b for b in MARKERS.blocks}
    assert {"never_measured", "undetected"} <= set(by_key), (
        "the two NEVER_MEASURED frames must be separate blocks"
    )

    off_panel = surface.rows_for(by_key["never_measured"], rows)
    undetected = surface.rows_for(by_key["undetected"], rows)

    assert all(r.frame.startswith("panel:") for r in off_panel), (
        "the 'off panel' block holds a row refused by some other frame"
    )
    assert all(r.frame.startswith("detect:") for r in undetected), (
        "the 'no usable signal' block holds a row refused by some other frame"
    )
    assert "off panel" in by_key["never_measured"].label_template
    assert "off panel" not in by_key["undetected"].label_template
    assert undetected, "the demo study has undetectable genes; they must have a block"


def test_a_non_observed_row_still_carries_its_name_when_the_table_is_pinnable(rows):
    """A hatched row says WHICH entity was never measured, pin buttons or not.

    ``render_surface`` draws column 0 as a button for OBSERVED rows only. It used to
    pass ``skip_first=True`` for every other row, which dropped that column -- and
    the gene name with it -- putting nothing in its place: 196 of c1's 280 rows
    rendered as "never measured / never measured / --", the symbol surviving only
    inside an invisible ``data-claim`` attribute. "Was ESR1 tested?" became
    unanswerable on the surface whose whole job is to answer it.

    ``table_html`` never passes ``skip_first``, so the existing assertion above
    tested the one path the app does not take. This asserts the path it does.
    """
    src = inspect.getsource(surface.render_surface)
    assert "row_html(spec, row, skip_first=pinnable)" not in src, (
        "render_surface drops the first column -- the name -- on non-observed rows"
    )

    for row in rows:
        if row.state is MeasuredState.OBSERVED:
            continue
        html = surface.row_html(MARKERS, row)
        assert str(row.values["gene"]) in html, (
            f"{row.values['gene']} ({row.state.value}) renders with no name"
        )


def test_the_header_and_the_rows_use_the_same_grid(rows):
    """The column headings sit over the columns they name.

    Both are plain ``st.markdown`` into one container, styled by the same
    ``.pano-sf-head, .pano-sf-row { display:grid }`` rule, so their
    ``grid-template-columns`` must match track for track. They did not: the header
    was emitted with ``skip_first=pinnable`` (5 tracks) while non-observed rows
    carried all 6, which put GLM_COEF over the gene names and left the gene column
    unlabelled on 196 of c1's 280 rows.

    **What this canNOT check, and where that lives instead.** An earlier version
    compared the two ``grid-template-columns`` STRINGS. Both sides are
    ``_grid_style(spec.columns, skip_first=False)``, so it asserted a string equals
    itself and could not fail -- and it passed on a page where the same declaration
    resolved to three different track sets, because a bare ``1fr`` is
    ``minmax(auto, 1fr)`` and never shrinks below its own content. Alignment is a
    rendered property; only a browser settles it, and
    ``tests/test_ui_browser.py::test_the_column_headings_sit_over_their_columns``
    measures it at 1440 and 1024.

    What IS checkable here is the shape the renderer asks for, and both halves of
    it have shipped broken: the header skipping a column its rows did not, and the
    tracks being written so they could resolve differently per box.
    """
    import re

    header = surface.column_header_html(MARKERS)
    assert "minmax(0," in header, (
        "the header declares bare fr tracks, which resolve to each box's own "
        "min-content and so differ between the header and the rows"
    )
    for row in rows[:5]:
        assert "minmax(0," in surface.row_html(MARKERS, row), row.values["gene"]

    # The track COUNT must match too -- the 19ed6bd shape, where the header was one
    # column narrower than every non-observed row.
    n = len(re.findall(r"minmax\(", header))
    assert n == len(MARKERS.columns), f"header declares {n} of {len(MARKERS.columns)}"

    src = inspect.getsource(surface.render_surface)
    assert "column_header_html(spec, skip_first=pinnable)" not in src, (
        "render_surface emits a header one column narrower than its own rows"
    )


def test_the_never_measured_cells_hatch_and_name_the_frame_that_refused_them(rows):
    never = [r for r in rows if r.state is MeasuredState.NEVER_MEASURED]
    html = surface.table_html(MARKERS, rows)
    assert "pano-sf-hatch" in html
    from panoscope.platforms.xenium import XeniumAdapter

    declared = {f.name for f in XeniumAdapter().frames(et.cfg.current_study())}
    for row in never:
        # The frame is named, not implied: "never measured" without which frame
        # refused it is the blank row with extra words. WHICH frame refused it is
        # the point: `panel:` means the gene was never probed, `detect:` means it
        # was probed and produced no usable spatial signal, and those are different
        # facts to a biologist -- so this asserts a real declared frame is named,
        # not one hardcoded prefix.
        assert any(f"frame {name}" in row.reason for name in declared), row.reason
        assert "absence argues nothing" in row.reason
        assert row.reason in html  # it is the hover, on the hatch itself


def test_these_rows_are_absent_from_the_v1_view_the_page_used_to_render(rows):
    """The regression, pinned. ``ClusterVerdict.evidence`` is the compatibility view
    and carries measured rows only; rendering it is how 5 never-measured markers
    became 0 rows on screen."""
    from panoscope.ui import data_access as da

    v1_genes = {e.gene for e in da.verdict_for(UNIT_WITH_ABSENCES).evidence}
    never = {r.values["gene"] for r in rows
             if r.state is MeasuredState.NEVER_MEASURED}
    assert never and not (never & v1_genes)


def test_an_absence_block_renders_its_header_at_zero():
    """A block that disappears at zero is the panel-absence rule disappearing at the
    moment it had something to say.

    Driven from a filtered row set rather than from c1, which used to have no
    never-measured rows and now has 49: the method enumerates the whole panel table,
    so the detectability frame refuses the NoSig genes on every unit. The zero case
    is still reachable (a study whose panel carries every canonical marker and whose
    genes all gave signal), and it is the RENDERING of zero that this pins."""
    built = et.unit_build("c1")
    observed = [r for r in et.rows_for("c1", built)
                if r.state is not MeasuredState.NEVER_MEASURED]
    html = surface.table_html(MARKERS, observed)
    assert "NEVER MEASURED - 0 off panel" in html
    assert "every canonical marker for this label is on panel" in html


def test_the_ribbon_counts_every_state_against_the_unit_total(rows):
    html = surface.ribbon_html(MARKERS, rows)
    n_never = sum(1 for r in rows if r.state is MeasuredState.NEVER_MEASURED)
    assert f"{n_never} never measured" in html
    assert f"of {len(rows)}" in html


# --------------------------------------------------------------------------- #
# The band: ASSIGNMENT keeps it, and only ASSIGNMENT may have it
# --------------------------------------------------------------------------- #
def test_the_assignment_header_carries_the_band_its_label_space_licenses():
    html = surface.header_html(MARKERS, et._header_values("c1"))
    assert "<h1>Tumor</h1>" in html
    assert "Very High confidence" in html
    assert "Driven by" in html


def test_a_unit_not_yet_called_shows_no_band(monkeypatch):
    """No label, nothing to assess: the header carries no band chip."""
    from panoscope.data.study import NOT_YET_CALLED

    v = et.da.display_verdict_for("c1")
    monkeypatch.setattr(et.da, "display_verdict_for",
                        lambda c: dataclasses.replace(v, cell_type=NOT_YET_CALLED))
    html = surface.header_html(MARKERS, et._header_values("c1"))
    assert f"<h1>{NOT_YET_CALLED}</h1>" in html
    assert "confidence" not in html


def test_the_band_chip_carries_its_audit_and_the_verify_flag_is_read_not_chosen():
    """``verify`` comes from the deterministic rubric; the surface renders what the
    verdict says and has no way to decide it."""
    c9 = et._header_values("c9")
    assert c9["verify"] is True
    html = surface.header_html(MARKERS, c9)
    assert "re-check this" in html
    # The demotions that produced the band are the chip's tooltip, not a footnote.
    assert "fragile" in html


def test_the_same_header_is_a_category_error_on_the_pathways_surface():
    """The asymmetry, from the other end: CHARACTERISATION has no label space, so
    there is nowhere for the remaining probability mass to sit."""
    with pytest.raises(DegradedUpward):
        surface.label_band_html(PATHWAYS, et._header_values("c1"))


def test_the_markers_page_no_longer_builds_its_own_header():
    """The defect this conversion removes is a hand-built duplicate of a declared
    header, so it is asserted against the source."""
    import pathlib

    import panoscope

    src = inspect.getsource(et)
    # Against the imports, not the prose: the module docstring names the header it
    # replaced, which is the explanation, not the dependency.
    imports = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
    assert not [ln for ln in imports if "verdict_header" in ln]
    assert "verdict_header.render" not in src
    assert "confidence_chip" not in src
    # Read, not imported: importing app.py executes the Streamlit script.
    app_src = (pathlib.Path(panoscope.__file__).parent / "app.py").read_text()
    assert "verdict_header.render_verdict" not in app_src


# --------------------------------------------------------------------------- #
# Click-back, and the guard that refuses a number the state forbids
# --------------------------------------------------------------------------- #
def test_every_row_resolves_to_its_subject(rows):
    for row in rows:
        assert row.key.startswith("gene|g=gene:")
        assert f'data-claim="{row.key}"' in surface.row_html(MARKERS, row)


def test_a_row_with_nothing_to_click_raises_rather_than_rendering_plain(rows):
    with pytest.raises(surface.NothingToClick):
        surface.row_html(MARKERS, dataclasses.replace(rows[0], key=""))


def test_a_statistic_on_a_never_measured_row_is_refused_outright(rows):
    """Structural form of *statistic is None iff the state is not OBSERVED*: the
    renderer will not draw the number even if an adapter hands it one."""
    never = next(r for r in rows if r.state is MeasuredState.NEVER_MEASURED)
    forged = dataclasses.replace(never, values={**never.values, "statistic": 7.0})
    with pytest.raises(DegradedUpward):
        surface.row_html(MARKERS, forged)


# --------------------------------------------------------------------------- #
# What the adapter is allowed to put on screen
# --------------------------------------------------------------------------- #
def test_the_rows_are_the_records_items_and_none_are_filtered_out():
    built = et.unit_build(UNIT_WITH_ABSENCES)
    got = et.rows_for(UNIT_WITH_ABSENCES, built)
    assert len(got) == len(built.evidence.items)
    assert [r.state for r in got] == [i.measured.state for i in built.evidence.items]


def test_the_specificity_chip_still_reaches_the_screen():
    built = et.unit_build("c1")
    html = surface.table_html(MARKERS, et.rows_for("c1", built))
    assert et.FLAG_COPY["specificity:localises_elsewhere"] in html


def test_the_statistics_shown_are_the_records_own_numbers(rows):
    """Not recomputed and not re-read from a second artifact: the cell is the item's
    statistic, to the declared precision."""
    built = et.unit_build(UNIT_WITH_ABSENCES)
    by_gene = {i.subject.display(): i for i in built.evidence.items}
    for row in rows:
        assert row.values["statistic"] == by_gene[row.values["gene"]].statistic
