"""Grounding tests for the Summary page's data (annotation table + CSV export).

The Summary page renders ``all_verdicts`` as a table and offers ``verdict_csv``
as a download. These tests pin that every cluster produces a complete verdict and
that the CSV export carries the full documented column set — the same contract
the per-cluster output format promises.
"""
from __future__ import annotations

import html

from agent import verdict as agent_verdict
from agent.config import CLUSTER_ORDER
from ui import data_access as da
from ui import summary


def test_all_verdicts_returns_nine_complete_calls() -> None:
    verdicts = da.all_verdicts()
    assert len(verdicts) == len(CLUSTER_ORDER) == 9
    for v in verdicts:
        # the columns the summary table shows must be populated
        assert v.cell_type and v.cell_type_short
        assert v.confidence and v.category and v.lineage


def test_verdict_csv_has_full_header_and_nine_rows() -> None:
    csv = da.verdict_csv()
    lines = csv.strip().splitlines()
    header = lines[0]
    for col in agent_verdict.CSV_COLUMNS:
        assert col in header
    # header + one row per cluster
    assert len(lines) == 1 + len(CLUSTER_ORDER)


def test_exclude_note_flips_the_export_at_composition(monkeypatch) -> None:
    """An exclude note flips the exported exclude flag at composition, WITHOUT mutating
    the deterministic verdict on disk (docs/note-capture-design.md)."""
    # baseline: no exclude notes -> composed == deterministic
    monkeypatch.setattr(da, "_excluded_clusters", lambda: set())
    assert [v.exclude for v in da.composed_verdicts()] == [v.exclude for v in da.all_verdicts()]

    # an exclude note on c9 flips only c9, only in the composed view
    monkeypatch.setattr(da, "_excluded_clusters", lambda: {"c9"})
    composed = {v.cluster: v.exclude for v in da.composed_verdicts()}
    assert composed["c9"] is True
    assert composed["c1"] is False
    # the cached deterministic verdict for c9 is untouched (never mutated)
    assert da.all_verdicts()[8].exclude is False
    # and the CSV row for c9 differs from the un-excluded baseline
    monkeypatch.setattr(da, "_excluded_clusters", lambda: set())
    base_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    monkeypatch.setattr(da, "_excluded_clusters", lambda: {"c9"})
    excl_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    assert base_c9 != excl_c9 and "TRUE" in excl_c9.upper()


def test_celltype_override_reflects_at_composition(monkeypatch) -> None:
    """A confirmed celltype_override overlays the new call + lineage/category at
    composition (verify flagged only when the literature dissents), never mutating the
    deterministic verdict (docs/note-capture-design.md; user decision 2026-07-10)."""
    from agent import config as cfg
    from agent.types import Citation, Note, ScopeRef, Tension

    dissent = Citation(pmid="30000009", title="", authors="", year=2020, journal="",
                       stance="dissent", is_real=True)
    ov = Note(
        id="ov1", claim="c2 is CAF, not generic stroma", scope="cluster",
        scope_ref=ScopeRef(dataset=cfg.DATASET_ID, cluster="c2"), basis="own_validation",
        status="firm", subject_cell_type="CAF", subject_markers=(),
        tension=Tension(agree=(), dissent=(dissent,), thin=False, query="", looked_up_at=""),
        author="", created_at="2026-07-10T00:00:00+00:00", trigger="override", supersedes=None,
        type="celltype_override", subject_lineage="Fibroblast", subject_category="Stromal",
    )
    monkeypatch.setattr(da, "_override_notes", lambda: {"c2": ov})

    comp = {v.cluster: v for v in da.composed_verdicts()}
    assert comp["c2"].cell_type == "CAF"
    assert comp["c2"].lineage == "Fibroblast" and comp["c2"].category == "Stromal"
    assert comp["c2"].verify is True          # literature dissents -> flagged
    assert comp["c1"].cell_type != "CAF"       # other clusters untouched
    assert da.verdict_for("c2").cell_type == "Stromal"  # deterministic verdict never mutated

    info = da.override_info("c2")
    assert info["new_call"] == "CAF" and info["computed_call"] == "Stromal" and info["dissent"] == 1


def test_overview_table_column_order_and_grounding() -> None:
    """The overview merges the marker call and the enriched programs into one row
    per cluster: columns in order, and every value is a projection of the verdicts
    (+ enrichment records) — nothing invented."""
    verdicts = da.all_verdicts()
    try:
        enr_map = {ce.cluster: ce for ce in da.all_enrichments()}
    except Exception:  # noqa: BLE001 - no enrichment slice (fresh clone) -> marker-only overview
        enr_map = {}
    out = summary._overview_table_html(verdicts, enr_map)

    order = ["Cluster", "Cell type", "Conf.", "Key markers", "Enriched programs"]
    positions = [out.index(f">{label}<") for label in order]
    assert positions == sorted(positions), "overview columns are out of order"

    # Grounded: each cluster's cell type and key markers appear verbatim.
    for v in verdicts:
        assert html.escape(v.cell_type) in out
        for g in v.key_markers:
            assert html.escape(str(g)) in out

    # Enriched program names, when present, come from the enrichment records only.
    for ce in enr_map.values():
        for p in ce.enriched[:4]:
            assert html.escape(summary._short(p.gene_set)) in out


# --------------------------------------------------------------------------- #
# Sign-off board — the review cockpit that replaced the manuscript editor.
# --------------------------------------------------------------------------- #
def test_review_state_roundtrip(tmp_path) -> None:
    """The sign-off store persists and reads back ``{cluster: {at, note_id}}`` under a
    dataset tree, and a missing file reads as 'nothing signed off'."""
    from pipeline import store

    assert store.load_review_state(root=tmp_path) == {}
    store.save_review_state(
        {"c9": {"at": "2026-07-11T12:00:00", "note_id": "abc123"}},
        root=tmp_path, saved_at="2026-07-11T12:00:00",
    )
    got = store.load_review_state(root=tmp_path)
    assert got["c9"]["note_id"] == "abc123"


def test_sign_off_clears_verify_at_composition(monkeypatch) -> None:
    """Signing off a flagged call clears its ``verify`` at composition (and in the CSV),
    WITHOUT mutating the deterministic verdict on disk."""
    assert da.verdict_for("c9").verify is True  # c9 ships flagged for re-check
    # baseline: c9 still flagged in the composed view + CSV
    monkeypatch.setattr(da, "_signed_off_clusters", lambda: set())
    assert {v.cluster: v for v in da.composed_verdicts()}["c9"].verify is True

    # sign off c9 -> the re-check flag clears in the composed view
    monkeypatch.setattr(da, "_signed_off_clusters", lambda: {"c9"})
    comp = {v.cluster: v for v in da.composed_verdicts()}
    assert comp["c9"].verify is False
    assert comp["c1"].verify is False  # other clusters untouched
    # the cached deterministic verdict is never mutated
    assert da.verdict_for("c9").verify is True
    # and the CSV row for c9 now reads verify=FALSE
    c9_row = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    assert c9_row.strip().upper().endswith("FALSE")


def test_sign_off_wins_over_override_dissent_flag(monkeypatch) -> None:
    """A sign-off is the last word: it clears the verify flag an override's literature
    dissent would otherwise raise (the biologist adjudicated, with a note recorded)."""
    from agent import config as cfg
    from agent.types import Citation, Note, ScopeRef, Tension

    dissent = Citation(pmid="30000009", title="", authors="", year=2020, journal="",
                       stance="dissent", is_real=True)
    ov = Note(
        id="ov1", claim="c2 is CAF", scope="cluster",
        scope_ref=ScopeRef(dataset=cfg.DATASET_ID, cluster="c2"), basis="own_validation",
        status="firm", subject_cell_type="CAF", subject_markers=(),
        tension=Tension(agree=(), dissent=(dissent,), thin=False, query="", looked_up_at=""),
        author="", created_at="2026-07-10T00:00:00+00:00", trigger="override", supersedes=None,
        type="celltype_override", subject_lineage="Fibroblast", subject_category="Stromal",
    )
    monkeypatch.setattr(da, "_override_notes", lambda: {"c2": ov})
    # override alone -> dissent flags verify
    monkeypatch.setattr(da, "_signed_off_clusters", lambda: set())
    assert {v.cluster: v for v in da.composed_verdicts()}["c2"].verify is True
    # sign off the override -> flag clears
    monkeypatch.setattr(da, "_signed_off_clusters", lambda: {"c2"})
    assert {v.cluster: v for v in da.composed_verdicts()}["c2"].verify is False


def test_triage_bucket_routing() -> None:
    """A call routes to signed / needs-you (flagged, or a proposed refinement) / review."""
    v9 = da.verdict_for("c9")  # flagged for re-check
    v1 = da.verdict_for("c1")  # solid, Very High
    assert summary._triage_bucket(v9, {}, {}, {}) == summary._BUCKET_NEEDS
    assert summary._triage_bucket(v9, {"c9": {}}, {}, {}) == summary._BUCKET_SIGNED
    assert summary._triage_bucket(v1, {}, {}, {}) == summary._BUCKET_REVIEW
    # a proposed refinement pulls an otherwise-solid call into needs-you
    assert summary._triage_bucket(v1, {}, {"c1": object()}, {}) == summary._BUCKET_NEEDS


def test_evidence_strength_is_a_grounded_projection() -> None:
    """The row's evidence-strength line reads the top driver's gene + glm + pearson and
    the marker count straight from the verdict — nothing invented."""
    v = da.verdict_for("c1")
    top = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)[0]
    out = summary._evidence_strength_html(v)
    assert html.escape(top.gene) in out
    assert f"{top.glm_coef:.1f}" in out
    # the count is labelled "genes" (all genes jazzPanda assigned to the cluster), not
    # "markers" — those are not all drivers, and calling them markers overstates evidence.
    assert f"{len(v.evidence)} genes" in out


def test_reconciliation_items_are_grounded() -> None:
    """Every reconciliation cue is a projection of real evidence: any marker it names as
    'localizes better with another cluster' actually carries that caveat in some verdict."""
    verdicts = da.all_verdicts()
    try:
        themes = da.pathway_themes()
    except Exception:  # noqa: BLE001
        themes = None
    items = summary._reconciliation_items(verdicts, themes, {})
    assert isinstance(items, list)
    # collect markers that genuinely carry the spillover caveat
    real_spill = {
        e.gene
        for v in verdicts
        for e in v.evidence
        if e.gene in v.key_markers and any("localizes better" in str(c) for c in e.caveats)
    }
    joined = " ".join(body for _ic, body in items)
    if "localize better" in joined:
        assert real_spill, "claimed a spillover cue with no marker actually caveated"


def test_board_renders_headless() -> None:
    """The whole Summary board renders with no exception and offers a sign-off, a chat,
    and an evidence action per cluster (a runtime smoke over the real page)."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        import pytest

        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from ui import state as _state
        from ui import summary as _summary

        _state.init_state()
        _summary.render_summary_page()

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        import pytest

        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    labels = [b.label for b in at.button]
    assert sum("Sign off" in ln for ln in labels) >= 1
    md = "\n".join(m.value for m in at.markdown)
    assert "signed off" in md and "Reconcile across clusters" in md


def test_flag_reason_is_grounded_not_blanket_thin() -> None:
    """The flag reason and the contested-sign-off claim project the real verify cause —
    a fragile cluster reads 'rests on N markers', and NEITHER the reason nor the saved
    claim ever hardcodes 'thin' (which would misstate a specificity-demoted call)."""
    assert summary._flag_reason(da.verdict_for("c9")) == "rests on 2 markers"
    for v in da.all_verdicts():
        assert "thin" not in summary._flag_reason(v).lower()
        assert "thin" not in summary._signoff_claim(v, None).lower()
    # the claim names the top marker honestly (not 'the driver')
    assert "top marker CPA3" in summary._signoff_claim(da.verdict_for("c9"), None)


def test_two_pending_signoff_cards_render_without_crash() -> None:
    """Two contested sign-offs pending at once must not collide on a container key — the
    fail-soft rule (a normal interaction never crashes the board). Regression for the
    reproduced StreamlitDuplicateElementKey on a fixed 'sofcard' key."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        import pytest

        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from agent import memory
        from ui import state as _state
        from ui import summary as _summary

        _state.init_state()
        for c in ("c8", "c9"):
            d = memory.draft_note(claim=f"Confirmed {c}", scope="cluster", basis="convention",
                                  cluster=c, note_type="validation", literature_search=None)
            _state.set_pending_draft(f"signoff::{c}", d)
        _summary.render_summary_page()

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        import pytest

        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    assert sum("Confirm sign-off" in b.label for b in at.button) == 2
