"""The review-surface elements interface-design §1-§5 names that are not on the page yet
(plan Stage 6b): one strict gap each, plus the shared shell they hang on.

Screen gaps read a real render (AppTest). The pages are rendered once per module by
whichever test asks first, inside that test's own per-test tree, and kept as plain
strings; nothing later depends on the environment. Plot and column gaps read the
workflow's declared surface, which is the contract a renderer draws from.
"""
from __future__ import annotations

import inspect
import json

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import evidence_rows, open_app, screen_text, widget_keys

_SCREENS: dict[str, dict] = {}


def _capture(at) -> dict:
    labels = [str(b.label) for b in at.button]
    return {"text": screen_text(at), "keys": widget_keys(at), "buttons": labels,
            "rows": len(evidence_rows(at))}


@pytest.fixture
def screens(app_tree) -> dict[str, dict]:
    """markers c1 and pathways c1 (c1 accepted), the Summary, and markers c9 (the one
    cluster verdicts.csv flags verify=TRUE, still OPEN)."""
    if not _SCREENS:
        from panoscope.pipeline import decisions as D
        from panoscope.ui import data_access as da

        D.accept("c1", da.verdict_for("c1").cell_type, by="tester", at="2026-11-09T00:00:00")
        _SCREENS["markers"] = _capture(open_app("markers", cluster="c1"))
        _SCREENS["pathways"] = _capture(open_app("pathways", cluster="c1"))
        _SCREENS["summary"] = _capture(open_app("summary"))
        _SCREENS["flagged"] = _capture(open_app("markers", cluster="c9"))
        _SCREENS["manifest"] = json.loads((app_tree / DEMO_ID / "manifest.json").read_text())
    return _SCREENS


# --------------------------------------------------------------------------- #
# §2.1 R1  One shell, three fillings
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.1-01")
def test_markers_and_pathways_share_one_shell_rail_stage_and_argument_pane(screens) -> None:
    """Both workflows fill the same three regions: the unit rail (one button per
    cluster in the manifest), the evidence stage (rows), and the argument pane (an ask
    box for this unit)."""
    rail = {f"rail_{c}" for c in screens["manifest"]["clusters"]}
    assert len(rail) == 9
    for page in ("markers", "pathways"):
        s = screens[page]
        assert rail <= s["keys"], (page, sorted(rail - s["keys"]))
        assert s["rows"] > 0, page
        assert any(k.endswith("ask_input_c1") for k in s["keys"]), page


# --------------------------------------------------------------------------- #
# Screen gaps
# --------------------------------------------------------------------------- #
@pytest.mark.req("UI-2.5-02", gap=True)
def test_the_band_chip_opens_a_rubric_card_with_the_band_edges(screens) -> None:
    assert screens["markers"]["rows"] > 0
    assert "rubric" in screens["markers"]["text"].lower()


@pytest.mark.req("UI-1.3-02", gap=True)
def test_the_page_footer_names_the_manifest_sha_of_what_it_renders(screens) -> None:
    sha = screens["manifest"]["artifacts"]["interp/records/assignment/c1.json"]["sha256"]
    assert sha[:8] in screens["markers"]["text"]


@pytest.mark.req("UI-5.3-04", gap=True)
def test_a_flagged_cluster_offers_a_one_tap_pressure_test(screens) -> None:
    buttons = " ".join(screens["flagged"]["buttons"]).lower()
    assert "accept proposal" in buttons                     # the page is the review page
    assert "pressure" in buttons or "second opinion" in buttons, buttons


@pytest.mark.req("UI-5.3-06", gap=True)
def test_the_summary_board_mounts_the_same_commit_bar(screens) -> None:
    assert "Accept proposal" in screens["flagged"]["buttons"]
    assert "Accept proposal" in screens["summary"]["buttons"], screens["summary"]["buttons"]


# --------------------------------------------------------------------------- #
# Declared-surface gaps: the column or plot the spec names is not declared
# --------------------------------------------------------------------------- #
def _surfaces():
    from panoscope.workflows import enrichment, marker

    return marker.WORKFLOW.uispec(), enrichment.WORKFLOW.uispec()


@pytest.mark.req("UI-2.5-03", gap=True)
def test_markers_declare_a_band_position_micro_bar_beside_the_specificity_pair() -> None:
    from panoscope.types.uispec import ColumnKind

    kinds = {c.kind for c in _surfaces()[0].columns}
    assert ColumnKind.SPECIFICITY_PAIR in kinds
    assert ColumnKind.BAND_MICROBAR in kinds


@pytest.mark.req("UI-2.5-05", gap=True)
def test_markers_declare_their_detail_plots() -> None:
    """Discrimination lollipop, the cross-unit marker x cluster matrix with hatched
    never-measured cells, and the shared-evidence graph."""
    from panoscope.types.uispec import PlotKind

    spec = _surfaces()[0]
    assert PlotKind.SHARED_EVIDENCE_GRAPH in {p.kind for p in spec.detail_plots}
    assert len(spec.detail_plots) >= 3


@pytest.mark.req("UI-2.6-02", gap=True)
def test_pathways_draw_k_of_n_as_a_coverage_bar() -> None:
    from panoscope.types.uispec import ColumnKind

    assert ColumnKind.COVERAGE_BAR in {c.kind for c in _surfaces()[1].columns}


@pytest.mark.req("UI-2.6-03", gap=True)
def test_the_pathways_primary_plot_is_the_testability_plane() -> None:
    from panoscope.types.uispec import PlotKind

    primary = _surfaces()[1].primary_plot
    assert primary is not None and primary.kind is PlotKind.TESTABILITY_PLANE


@pytest.mark.req("UI-2.6-04", gap=True)
def test_the_pathways_overview_is_a_categorical_tier_matrix() -> None:
    from panoscope.types.uispec import PlotKind

    assert PlotKind.TIER_MATRIX in {p.kind for p in _surfaces()[1].overview_plots}


@pytest.mark.req("UI-2.8-02", gap=True)
def test_the_density_panels_can_lock_one_shared_scale() -> None:
    """Today every panel says 'independent scale ... NOT comparable', which is the
    honest half; the lock that makes them comparable is missing."""
    from panoscope.ui import conditionality as cnd

    params = inspect.signature(cnd.gene_density_cond).parameters
    assert any("shared" in p or "lock" in p for p in params), list(params)


@pytest.mark.req("UI-2.9-01", gap=True)
def test_there_are_five_authorities_and_inference_needs_premises() -> None:
    from panoscope.types.claim import Authority

    assert {a.name for a in Authority} == {"MEASUREMENT", "LAB", "CURATED", "LITERATURE",
                                            "INFERENCE"}
