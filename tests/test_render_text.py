"""The second renderer, and the claim it exists to make checkable.

ARCHITECTURE §13 says the review surface is *declared by the workflow, not a UI
concern*. With exactly one renderer, that claim and "the surface is hand-built in
exactly one place" produce identical evidence: ``ui/surface.py`` could be reading
the declaration or it could be the declaration's only author, and nothing in the
suite can tell the difference.

``panoscope/render/text.py`` is a second renderer in a different medium that shares
no code with the first. What these tests buy, in descending order:

1. **Renderer-agnosticism, as a test.** The same ``UISpec`` and the *same row
   objects* through both renderers produce the same rows, in the same order, with
   the same measurement states and the same hatched cells -- while the two outputs
   share not one byte of each other. Anything the declaration failed to specify
   would show up here as a disagreement.
2. **The guards are properties of the declaration, not of HTML.** Each of the four
   is re-enforced in the text renderer independently and tested here against the
   text renderer, not inherited from ``ui.surface``.
3. **No wrapping.** The module imports nothing from ``panoscope.ui``, asserted
   against the source rather than trusted -- a renderer that delegated would prove
   the HTML renderer is reusable, which was never in question.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

from panoscope.render import text
from panoscope.types.measured import MeasuredState
from panoscope.types.uispec import (
    ColumnKind,
    ColumnSpec,
    DegradedUpward,
    HeaderKind,
    degraded_spec,
)
from panoscope.ui import surface
from panoscope.workflows import de, enrichment, marker

MARKER = marker.WORKFLOW.uispec()
PATHWAYS = enrichment.WORKFLOW.uispec()
CONTRASTS = de.WORKFLOW.uispec()
DEGRADED = degraded_spec("some_tool_output.csv", ("gene", "score", "note"))
ALL_SPECS = [MARKER, PATHWAYS, CONTRASTS, DEGRADED]

RENDER_TEXT = Path(text.__file__)

# The four state labels both renderers emit. Written here a third time on purpose:
# if either renderer renames one, these tests stop being able to read it.
LABELS = {
    "observed": MeasuredState.OBSERVED,
    "measured, not here": MeasuredState.TESTED_ABSENT,
    "too little signal": MeasuredState.UNDERPOWERED,
    "never measured": MeasuredState.NEVER_MEASURED,
}


def _rows() -> list[surface.SurfaceRow]:
    """One row per measurement state, with the numbers each state permits.

    Deliberately out of sorted order: the two renderers must agree on the order the
    *declaration* asks for, not on the order they were handed.
    """
    return [
        surface.SurfaceRow(
            key="gene|g=ESR1", state=MeasuredState.OBSERVED,
            values={"gene": "ESR1", "gene_set": "ESR1", "statistic": 1.02,
                    "pearson": (0.21, 0.30), "p_adj_bh": 0.04,
                    "coverage": "18/200", "detection": (61, 12),
                    "direction": "down"},
            reason="observed"),
        surface.SurfaceRow(
            key="gene|g=SLC1A3", state=MeasuredState.NEVER_MEASURED,
            values={"gene": "SLC1A3", "gene_set": "SLC1A3", "coverage": "0/200"},
            reason="not on the 280-gene panel"),
        surface.SurfaceRow(
            key="gene|g=ERBB2", state=MeasuredState.OBSERVED,
            values={"gene": "ERBB2", "gene_set": "ERBB2", "statistic": 4.21,
                    "pearson": (0.55, 0.12), "p_adj_bh": 0.001,
                    "coverage": "42/200", "detection": (88, 4),
                    "direction": "up"},
            reason="observed"),
        surface.SurfaceRow(
            key="gene|g=KRT14", state=MeasuredState.TESTED_ABSENT,
            values={"gene": "KRT14", "gene_set": "KRT14", "coverage": "30/200"},
            reason="on panel, peaked in c4"),
        surface.SurfaceRow(
            key="gene|g=MKI67", state=MeasuredState.UNDERPOWERED,
            values={"gene": "MKI67", "gene_set": "MKI67", "coverage": "2/200"},
            reason="below the detectability floor"),
    ]


def _header_values() -> dict:
    """One bag of values, enough for whichever header variant a spec declares."""
    return {
        "unit_scope": "sample1 / c1 - 4,812 cells",
        "label": "Tumor", "confidence": "High", "cell_type": "Tumor",
        "band_basis": ("driver glm_coef 4.21",), "demotions": (),
        "key_markers": ("ERBB2",), "runner_up": "Myoepithelial", "verify": True,
        "n_enriched": 1, "n_suggestive": 0, "n_not_supported": 1,
        "n_untestable": 2, "n_scored": 4, "gene_set": "HALLMARK_E2F_TARGETS",
        "statistic_name": "cgst_z", "statistic": "5.10", "panel_hits": 42,
        "set_size": 200, "top_key": "gene_set|set=HALLMARK_E2F_TARGETS",
        "panel_scope_caveat": "Panel-scoped: 280 of ~20,000 genes.",
        "gates": ("enriched: q < 0.05",),
        "n_samples": 1, "n_items": 5, "n_supported": 2, "n_tested": 3,
        "panel_size": 280, "scope_caveat": "Panel-scoped.",
    }


# --------------------------------------------------------------------------- #
# Semantic extraction: what each rendering SAYS, recovered from its own bytes.
# Neither renderer is asked what it meant -- that would be grading itself.
# --------------------------------------------------------------------------- #
_HTML_ROW = re.compile(
    r'data-claim="([^"]+)"><div class="pano-sf-row"[^>]*>(.*?)</div></span>',
    re.DOTALL)
_HTML_CELL = re.compile(r'<div class="pano-sf-cell[^"]*">(.*?)</div>', re.DOTALL)


def _state_in(blob: str) -> MeasuredState:
    found = [s for lab, s in LABELS.items() if lab in blob]
    assert len(found) == 1, f"expected exactly one state label in {blob!r}"
    return found[0]


def _semantics_html(out: str) -> list[tuple]:
    got = []
    for key, body in _HTML_ROW.findall(out):
        cells = _HTML_CELL.findall(body)
        hatched = tuple(i for i, c in enumerate(cells) if "pano-sf-hatch" in c)
        got.append((key, _state_in(body), hatched))
    return got


def _semantics_text(out: str) -> list[tuple]:
    got = []
    for line in out.splitlines():
        if text.ANCHOR not in line:
            continue
        body, key = line.rsplit(text.ANCHOR, 1)
        cells = [c.strip() for c in body.strip().split(text.SEP)]
        hatched = tuple(i for i, c in enumerate(cells)
                        if c.startswith("[") and c.endswith("]"))
        got.append((key, _state_in(body), hatched))
    return got


# --------------------------------------------------------------------------- #
# 1. The claim: one declaration, two media, the same meaning
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [MARKER, PATHWAYS, CONTRASTS],
                         ids=lambda s: s.workflow_id)
def test_both_renderers_report_the_same_rows_states_and_hatched_cells(spec):
    """THE test. Same spec, same row objects, two media -- one meaning.

    Rows present, their order, their measurement state and which cells carry no
    number. A declaration that under-specified any of those would show up here as a
    disagreement between two independent readings of it.
    """
    rows = _rows()
    assert (_semantics_text("\n".join(text.table_text(spec, rows)))
            == _semantics_html(surface.table_html(spec, rows)))


@pytest.mark.parametrize("spec", [MARKER, PATHWAYS, CONTRASTS],
                         ids=lambda s: s.workflow_id)
def test_the_two_renderings_agree_on_meaning_and_share_no_line(spec):
    """The other half: agreeing because they are the same output would prove
    nothing. Two media, no shared byte of layout."""
    rows = _rows()
    html = surface.table_html(spec, rows)
    txt = "\n".join(text.table_text(spec, rows))
    assert html != txt
    assert "<div" in html and "<div" not in txt
    assert not ({ln.strip() for ln in html.splitlines() if ln.strip()}
                & {ln.strip() for ln in txt.splitlines() if ln.strip()})


def test_both_renderers_sort_within_a_block_and_never_across_one():
    """Order is the declaration's (``BlockSpec.sort_by``), not the input's -- and
    a never-measured row stays in its own block rather than sinking to the bottom
    of a top-N nobody scrolls."""
    rows = _rows()
    expected = ["gene|g=ERBB2", "gene|g=ESR1", "gene|g=KRT14",
                "gene|g=MKI67", "gene|g=SLC1A3"]
    assert [k for k, _, _ in
            _semantics_text("\n".join(text.table_text(MARKER, rows)))] == expected
    assert [k for k, _, _ in
            _semantics_html(surface.table_html(MARKER, rows))] == expected


def test_both_renderers_say_why_a_cell_has_no_number():
    """The same reason, in the place each medium has for it: a ``title`` attribute
    in HTML, its own line in the terminal. The fact is declared; the hover is not."""
    rows = _rows()
    reason = "not on the 280-gene panel"
    assert reason in surface.table_html(MARKER, rows)
    assert reason in "\n".join(text.table_text(MARKER, rows))


def test_the_text_renderer_imports_nothing_from_the_html_one():
    """Otherwise the second renderer is a wrapper and proves nothing."""
    tree = ast.parse(RENDER_TEXT.read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    assert imported, "nothing imported at all -- the walk is wrong"
    assert not [m for m in imported if m.startswith("panoscope.ui")]
    assert not [m for m in imported
                if m.split(".")[0] not in ("panoscope", "typing", "__future__")], \
        "stdlib only, and no new dependency"


# --------------------------------------------------------------------------- #
# 2. The four guards, enforced in the text renderer independently
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [PATHWAYS, CONTRASTS, DEGRADED],
                         ids=lambda s: s.workflow_id)
def test_a_surface_with_no_label_space_cannot_render_an_assignment_header(spec):
    """``ui/enrichment_table.py``'s bug, at the second renderer. The rule is about
    the task kind, so it has to bite in every medium or it was a property of one."""
    with pytest.raises(DegradedUpward):
        text.label_band_text(spec, {"confidence": "High", "label": "Tumor"})


def test_the_assignment_surface_does_render_its_band():
    """The guard refuses the wrong task kind, not every caller."""
    out = "\n".join(text.label_band_text(MARKER, _header_values()))
    assert "[High]" in out and "Tumor" in out and "ERBB2" in out


def test_the_pathways_header_carries_no_band_anywhere():
    out = "\n".join(text.header_text(PATHWAYS, _header_values()))
    assert "High" not in out and "confidence" not in out.lower()
    assert "1 enriched" in out and "2 untestable" in out and "of 4 sets scored" in out


def test_a_registered_workflow_cannot_degrade_to_the_provenance_header():
    """Hiding a working interpretation behind a disclaimer is the mirror-image
    dishonesty, and just as wrong in a terminal."""
    with pytest.raises(DegradedUpward, match="undeclared"):
        text.provenance_text(PATHWAYS, {})


def test_a_number_on_a_row_with_no_measurement_is_refused():
    """``statistic is None`` iff TESTED_ABSENT or NEVER_MEASURED. The renderer does
    not quietly drop the number -- a caller that produced one has a bug."""
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    row = surface.SurfaceRow(key="gene|g=X", state=MeasuredState.NEVER_MEASURED,
                             values={"statistic": 4.2}, reason="off panel")
    with pytest.raises(DegradedUpward, match="only where something was observed"):
        text.cell_text(col, row)


def test_a_row_with_nothing_to_click_is_not_rendered():
    row = surface.SurfaceRow(key="", state=MeasuredState.OBSERVED,
                             values={"gene": "ERBB2", "statistic": 4.2}, reason="")
    with pytest.raises(text.NothingToClick):
        text.row_text(MARKER, row)


def test_a_header_sentence_with_nothing_to_click_is_not_rendered():
    """The census header's top-item line is a claim about a specific item, so it
    resolves to that item or it does not render."""
    values = _header_values()
    values.pop("top_key")
    with pytest.raises(text.NothingToClick):
        text.census_text(PATHWAYS, values)


def test_a_declaration_carrying_a_callable_is_refused():
    """A callable reaching a renderer means a renderer is about to decide what
    counts as measured."""
    bad = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    object.__setattr__(bad, "unit", lambda row: row.values["statistic"] * 2)
    spec = dataclasses.replace(MARKER, columns=(*MARKER.columns, bad))
    with pytest.raises(text.SpecCarriesCode):
        text.table_text(spec, _rows())


# --------------------------------------------------------------------------- #
# 3. A never-measured cell is neither blank nor zero -- in text too
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", list(MeasuredState))
def test_a_cell_with_no_number_carries_a_visible_marker_and_not_a_blank(state):
    if state is MeasuredState.OBSERVED:
        pytest.skip("observed carries a number; covered below")
    col = ColumnSpec("statistic", "glm_coef", ColumnKind.NUMBER)
    row = surface.SurfaceRow(key="gene|g=X", state=state, values={"statistic": None},
                             reason="3 of 200 set genes on panel (floor 5)")
    cell = text.cell_text(col, row)
    assert cell.startswith("[") and cell.endswith("]")   # not a blank
    assert "0.00" not in cell                            # not a zero
    assert LABELS and any(lab in cell for lab in LABELS)  # says which state


def test_an_observed_zero_is_a_zero_and_not_a_hatch():
    """The other half. A measured zero is a result and must read as one."""
    col = ColumnSpec("statistic", "z", ColumnKind.NUMBER)
    row = surface.SurfaceRow(key="gene|g=X", state=MeasuredState.OBSERVED,
                             values={"statistic": 0.0}, reason="")
    assert text.cell_text(col, row) == "0.00"


def test_the_never_measured_block_renders_its_header_at_zero():
    block = next(b for b in PATHWAYS.blocks
                 if MeasuredState.NEVER_MEASURED in b.statuses)
    out = "\n".join(text.block_text(PATHWAYS, block, [], ()))
    assert "NEVER SCORED - 0" in out and block.empty_text in out


def test_the_coverage_ribbon_labels_every_count():
    out = "\n".join(text.ribbon_text(MARKER, _rows()))
    assert "2 observed" in out and "1 never measured" in out and "of 5" in out


# --------------------------------------------------------------------------- #
# 4. Every declared surface renders, including the degraded one
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.workflow_id)
def test_a_declared_surface_renders_its_whole_table(spec):
    """Parameterised over the declarations: a fourth workflow proves conformance
    rather than merely asserting it."""
    rows = _rows()
    out = "\n".join(text.table_text(spec, rows))
    for block in spec.blocks:
        assert block.label_template.format(
            count=len(text.rows_for(block, rows))) in out
    for col in spec.columns:
        assert col.label[:text.MAX_WIDTH - 1] in out


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.workflow_id)
def test_the_declared_header_variant_is_the_one_rendered(spec):
    """No surface may borrow another task kind's header, in either direction."""
    expected = {
        HeaderKind.LABEL_BAND: text.label_band_text,
        HeaderKind.CENSUS: text.census_text,
        HeaderKind.CONFOUND_CHECKLIST: text.checklist_text,
        HeaderKind.PROVENANCE_ONLY: text.provenance_text,
    }[spec.header.kind]
    values = _header_values()
    assert text.header_text(spec, values) == expected(spec, values)


def test_the_degraded_surface_claims_nothing():
    out = text.render(DEGRADED, header={"provenance": ("sha256 9f3a",)},
                      rows=_rows())
    assert "Unregistered analysis type" in out
    assert "High" not in out and "[● observed]" not in out


def test_the_renderer_never_invents_a_column_the_spec_did_not_declare():
    """The declaration is the whole contract: a value with no column is not drawn."""
    row = surface.SurfaceRow(key="gene|g=ERBB2", state=MeasuredState.OBSERVED,
                             values={"gene": "ERBB2", "statistic": 4.2,
                                     "secret_field": "NOT_DECLARED"}, reason="")
    assert "NOT_DECLARED" not in "\n".join(text.table_text(MARKER, [row]))
