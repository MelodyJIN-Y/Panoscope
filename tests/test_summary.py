"""Grounding tests for the Summary page's data (annotation table + CSV export).

The Summary page renders ``all_verdicts`` as a table and offers ``verdict_csv``
as a download. These tests pin that every cluster produces a complete verdict and
that the CSV export carries the full documented column set — the same contract
the per-cluster output format promises.
"""
from __future__ import annotations

from panoscope.agent import config as cfg
import html

import pytest

from panoscope.agent import compose
from panoscope.core.compat import SCORE_MAP
from panoscope.core.export import PROFILES
from panoscope.types.outputs import TaskKind
from panoscope.ui import data_access as da
from panoscope.ui import summary


# Every AppTest here renders a real study and the app writes as it renders, so
# render a per-test copy (conftest ``app_tree``); the committed tree stays read-only.
pytestmark = pytest.mark.usefixtures("app_tree")


def test_all_verdicts_returns_nine_complete_calls() -> None:
    verdicts = da.all_verdicts()
    assert len(verdicts) == len(cfg.CLUSTER_ORDER) == 9
    for v in verdicts:
        # the columns the summary table shows must be populated
        assert v.cell_type and v.cell_type_short
        assert v.confidence and v.category and v.lineage


def test_verdict_csv_has_full_header_and_nine_rows() -> None:
    csv = da.verdict_csv()
    lines = csv.strip().splitlines()
    header = lines[0]
    for col in PROFILES[TaskKind.ASSIGNMENT].names:
        assert col in header
    # header + one row per cluster
    assert len(lines) == 1 + len(cfg.CLUSTER_ORDER)


@pytest.mark.req("ARCH-18-03", gap=True)
def test_the_exported_csv_names_its_spec_version() -> None:
    """ARCH §18: spec_version on every artifact. The CSV export is an artifact and
    carries none today; where it goes (a column or a header line) is a product
    decision, since the 11-column contract is pinned by the golden output."""
    assert "spec_version" in da.verdict_csv()


def test_exclude_note_flips_the_export_at_composition() -> None:
    """An exclude note flips the exported exclude flag at composition, WITHOUT mutating
    the deterministic verdict on disk (docs/note-capture-design.md). Driven through a
    real note written to the (per-test) context store, not a patched helper."""
    from panoscope.agent import memory

    # baseline: no exclude notes -> composed == deterministic
    assert [v.exclude for v in da.composed_verdicts()] == [v.exclude for v in da.all_verdicts()]
    base_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]

    # an exclude note on c9 flips only c9, only in the composed view
    memory.create_note(claim="c9 is a doublet-rich cluster; exclude it", scope="cluster",
                       basis="own_validation", cluster="c9", note_type="exclude",
                       dataset=cfg.DATASET_ID, literature_search=lambda _q: ())
    composed = {v.cluster: v.exclude for v in da.composed_verdicts()}
    assert composed["c9"] is True
    assert composed["c1"] is False
    # the cached deterministic verdict for c9 is untouched (never mutated)
    assert da.all_verdicts()[8].exclude is False
    # and the CSV row for c9 differs from the un-excluded baseline
    excl_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    assert base_c9 != excl_c9 and "TRUE" in excl_c9.upper()


@pytest.mark.req("ARCH-8.2-01", "CLAUDE-MEM-01", "UI-5.1-01")
def test_celltype_override_reflects_at_composition() -> None:
    """A confirmed celltype_override overlays the new call + lineage/category at
    composition (literature dissent never flips verify: rubric only), never mutating the
    deterministic verdict (docs/note-capture-design.md; user decision 2026-07-10)."""
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.agent.types import Citation

    dissent = Citation(pmid="30000009", title="", authors="", year=2020, journal="",
                       stance="dissent", is_real=True)
    memory.create_note(
        claim="c2 is CAF, not generic stroma", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
        subject_lineage="Fibroblast", subject_category="Stromal",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (dissent,))

    comp = {v.cluster: v for v in da.composed_verdicts()}
    assert comp["c2"].cell_type == "CAF"
    assert comp["c2"].lineage == "Fibroblast" and comp["c2"].category == "Stromal"
    # Band + verify: the rubric on the biologist's label; dissent is a display flag only.
    b = da._biologist_assessment("c2", "CAF")
    assert (comp["c2"].confidence, comp["c2"].verify) == (b.band, b.verify)
    # The CSV's score anchor follows the band beside it, not the proposal's.
    assert comp["c2"].confidence_score == SCORE_MAP[b.band] != da.verdict_for("c2").confidence_score
    assert comp["c1"].cell_type != "CAF"       # other clusters untouched
    assert da.verdict_for("c2").cell_type == "Stromal"  # deterministic verdict never mutated

    info = da.override_info("c2")
    assert info["new_call"] == "CAF" and info["computed_call"] == "Stromal" and info["dissent"] == 1


# --------------------------------------------------------------------------- #
# Sign-off board — the review cockpit that replaced the manuscript editor.
# --------------------------------------------------------------------------- #
def test_review_state_roundtrip(tmp_path) -> None:
    """The sign-off store persists and reads back ``{cluster: {at, note_id}}`` under a
    dataset tree, and a missing file reads as 'nothing signed off'."""
    from panoscope.pipeline import store

    assert store.load_review_state(root=tmp_path) == {}
    store.save_review_state(
        {"c9": {"at": "2026-07-11T12:00:00", "note_id": "abc123"}},
        root=tmp_path, saved_at="2026-07-11T12:00:00",
    )
    got = store.load_review_state(root=tmp_path)
    assert got["c9"]["note_id"] == "abc123"


def test_sign_off_never_clears_verify(monkeypatch) -> None:
    """A reviewed tick is a mark, not a verdict: a signed-off cluster whose rubric
    ``verify`` is True still reads True composed, in the marker header, and in the CSV."""
    from panoscope.pipeline import store

    assert da.verdict_for("c9").verify is True  # c9 ships flagged for re-check
    monkeypatch.setattr(store, "load_review_state",
                        lambda *a, **k: {"c9": {"at": "2026-07-11T12:00:00", "note_id": None}})
    assert "c9" in da.signed_off()
    assert {v.cluster: v for v in da.composed_verdicts()}["c9"].verify is True
    assert da.display_verdict_for("c9").verify is True
    c9_row = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    assert c9_row.strip().upper().endswith("TRUE")


def test_triage_bucket_routing() -> None:
    """A call routes to signed / needs-you (flagged, or a proposed refinement) / review."""
    v9 = da.verdict_for("c9")  # flagged for re-check
    v1 = da.verdict_for("c1")  # solid, Very High
    assert summary._triage_bucket(v9, {}, {}, {}) == summary._BUCKET_NEEDS
    assert summary._triage_bucket(v9, {"c9": {}}, {}, {}) == summary._BUCKET_SIGNED
    assert summary._triage_bucket(v1, {}, {}, {}) == summary._BUCKET_REVIEW
    # a proposed refinement pulls an otherwise-solid call into needs-you
    assert summary._triage_bucket(v1, {}, {"c1": object()}, {}) == summary._BUCKET_NEEDS


def test_board_renders_headless() -> None:
    """The whole Summary board renders with no exception: the aligned HTML table with a
    row per cluster, the checklist tick links, and the key-markers column."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        import pytest

        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        _summary.render_summary_page()

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        import pytest

        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert 'class="ptbl"' in md          # the aligned HTML table rendered
    assert "once reviewed" in md and "(a mark, not a call)" in md and "Biology" in md
    assert 'href="?page=summary&sign=' in md   # a sign-off tick link (preserves the page)
    for ct in ("Tumor", "Stromal", "Mast Cells"):
        assert ct in md


def test_my_notes_section_says_what_it_withheld(tmp_path, monkeypatch) -> None:
    """The CALL SITE, not just the renderer: ``withheld_notes`` existed with zero
    production callers, so the board stayed silent about a note it had dropped.
    Driving the real page with the My-notes section open is what pins the wiring —
    a test that only called the renderer would stay green if the call were deleted.
    """
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        import pytest

        pytest.skip("streamlit.testing.v1 unavailable")

    import json

    from panoscope.agent import memory

    base = tmp_path / "context"
    corrections = base / "corrections"
    corrections.mkdir(parents=True, exist_ok=True)
    (corrections / "nostudy.json").write_text(json.dumps({
        "id": "nostudy",
        "claim": "c2 is CAF",
        "scope": "cluster",
        "scope_ref": {"cluster": "c2"},        # no dataset: the gate must drop it
        "basis": "own_validation",
        "status": "firm",
        "type": "celltype_override",
        "subject_cell_type": "CAF",
        "created_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    # The env var, not a module attribute: ``memory._default_base()`` resolves it
    # per call, which is what makes conftest's autouse isolation real.
    monkeypatch.setenv("PANOSCOPE_CONTEXT_DIR", str(base))

    def _run() -> None:
        import streamlit as _st

        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        _st.session_state[_summary._K_ACTIVE] = _summary._SEC_NOTES
        _summary.render_summary_page()

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        import pytest

        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "names no study" in md and "nostudy" in md and "c2 is CAF" in md


def test_table_html_is_grounded() -> None:
    """Every cell in the HTML table is a projection of the verdicts — cluster ids, cell
    types, confidence labels and the top marker all appear verbatim; nothing invented."""
    verdicts = da.all_verdicts()
    out = summary._table_html(verdicts, {}, {}, {})
    for v in verdicts:
        assert v.cluster in out
        assert html.escape(v.cell_type.replace("_", " ")) in out
        assert v.confidence in out
        top = sorted(v.evidence, key=lambda e: e.glm_coef, reverse=True)[0]
        assert html.escape(top.gene) in out


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


def test_a_reviewed_mark_never_reads_as_a_call() -> None:
    """The saved claim and the tick say 'reviewed', never 'accepted'; on an uncalled
    cluster neither pretends there is a call."""
    import dataclasses

    from panoscope.data.study import NOT_YET_CALLED

    v = da.verdict_for("c9")
    assert "accept" not in summary._signoff_claim(v, None).lower()
    assert "accept" not in summary._signoff_claim(v, {"dissent": 2}).lower()
    u = dataclasses.replace(v, cell_type=NOT_YET_CALLED, verify=False)
    claim = summary._signoff_claim(u, None)
    assert "c9 (no call yet)" in claim and "not yet called" not in claim
    assert "not a call: no call yet" in summary._table_html([u], {}, {}, {})


def test_pending_signoff_card_renders_on_drilldown() -> None:
    """A contested sign-off started from a cluster's drill-down renders its confirm card
    there (per-cluster container key, no crash). The Overall table shows only a ⚠, so the
    review action lives on the click-through — this checks the card still appears."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        import pytest

        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from panoscope.agent import memory
        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        import streamlit as _st

        _st.session_state["sum_active_section"] = "c9"      # drill into c9
        d = memory.draft_note(claim="Confirmed c9", scope="cluster", basis="convention",
                              cluster="c9", note_type="validation", literature_search=None)
        _state.set_pending_draft("signoff::c9", d)
        _summary.render_summary_page()

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        import pytest

        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    assert sum("Confirm reviewed" in b.label for b in at.button) == 1


def _summary_edits_path():
    """Where the Summary page's saved edits live for the active study."""
    from panoscope.pipeline import paths as pipeline_paths

    return pipeline_paths.summary_edits_json(cfg.current_study().id)


def _render_summary_headless(seed_edit: str = ""):
    """Run the real Summary page under AppTest, optionally with one region edited.

    ``seed_edit`` goes into the caveats region's canonical session key, which is what
    a keystroke in that editor would have set -- so the "an edit still saves" half
    exercises the same autosave branch the biologist does.
    """
    import pytest

    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    # The body runs as its own script (no closure over this frame), so the edit is
    # passed in through AppTest's kwargs rather than captured.
    def _run(edit: str = "") -> None:
        import streamlit as _st

        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        if edit:
            # the canonical key is scoped to the active study (``_skey``), which is
            # what a keystroke in that editor would set -- a bare ``wsval_caveats``
            # would be ignored by the page and quietly stop exercising the autosave.
            _st.session_state[_summary._skey("wsval_caveats")] = edit
        _summary.render_summary_page()

    at = AppTest.from_function(_run, kwargs={"edit": seed_edit})
    at.run(timeout=60)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    return at


def test_signing_off_does_not_freeze_the_stale_draft_as_the_biologists_prose() -> None:
    """A draft that changes under an untouched region is not an edit.

    The session seeds each region's text once; signing a flagged call off clears
    its verify flag, so ``default_cluster_summary`` and ``caveats_text`` regenerate
    WITHOUT "(flagged to re-check)" while the seeded value still carries it. The
    autosave compared value against the new default, saw a difference, and wrote
    the stale auto-draft into ``interp/summary_edits.json`` as the biologist's own
    text -- which then wins forever, because saved edits beat the draft.

    Four clicks produced a document saying "0 flagged for re-check" on page one and
    "Flagged for re-check - c9 Mast_Cells" later in the same file, with the
    contradiction persisted into the study tree and attributed to someone who never
    typed it.
    """
    import json

    import pytest

    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        pytest.skip("streamlit.testing.v1 unavailable")

    p = _summary_edits_path()
    if p.exists():
        if (json.loads(p.read_text(encoding="utf-8")).get("edits") or {}):
            pytest.skip("a real saved edit is on disk; not clobbering it")
        p.unlink()

    def _run() -> None:
        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        _summary.render_summary_page()

    try:
        at = AppTest.from_function(_run)
        at.run(timeout=90)
        if at.exception and "keyword argument 'key'" in str(at.exception[0].value):
            pytest.skip("Streamlit too old for st.container(key=...)")
        # The board dispatches a sign-off as a query param; then it reruns.
        at.query_params["sign"] = "c9"
        at.run(timeout=90)
        at.run(timeout=90)
        assert not at.exception, [str(e.value) for e in at.exception]

        assert not p.exists(), (
            "signing off wrote the stale auto-draft into the study tree as an edit: "
            + json.dumps(json.loads(p.read_text(encoding='utf-8')).get('edits', {}))[:300]
        )
    finally:
        if p.exists():
            p.unlink()
        rs = p.parent / "review_state.json"
        if rs.exists():
            rs.unlink()


def test_opening_the_summary_page_does_not_write_the_study_tree() -> None:
    """A read surface must not mutate the data tree. Opening the board used to write
    ``interp/summary_edits.json`` holding ``{"edits": {}}`` -- an empty write that says
    nothing and dirties every study a biologist merely looks at. The write now happens
    only when there is something to write (an edit, or a disk copy to clear)."""
    import json

    import pytest

    p = _summary_edits_path()
    if p.exists():
        on_disk = json.loads(p.read_text(encoding="utf-8")).get("edits") or {}
        if on_disk:
            pytest.skip("a real saved edit is on disk; not clobbering it")
        p.unlink()

    try:
        _render_summary_headless()
        assert not p.exists(), f"opening the Summary page wrote {p}"

        # the other half: a region the biologist actually edited still autosaves
        _render_summary_headless(seed_edit="EDITED BY THE TEST")
        assert p.exists()
        saved = json.loads(p.read_text(encoding="utf-8"))["edits"]
        assert saved.get("caveats") == "EDITED BY THE TEST"
    finally:
        if p.exists():
            p.unlink()


@pytest.mark.req("UI-5.6-01")
def test_accepting_every_unflagged_call_moves_no_score_and_shows_no_tally() -> None:
    """interface-design §5.6: N accepts leave confidence_score byte-identical, agreement
    is never aggregated as "8/9 confirmed", and an unflagged accept writes no note."""
    import re
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    from panoscope.agent import memory
    from panoscope.pipeline import decisions as D

    def scores():
        return [(v.cluster, v.confidence, v.confidence_score, v.verify)
                for v in da.composed_verdicts()]

    before, csv_before, notes_before = scores(), da.verdict_csv(), memory.read_notes()
    unflagged = [v for v in da.all_verdicts() if not v.verify]
    assert len(unflagged) >= 2, "the demo must carry several unflagged calls"
    for v in unflagged:
        D.accept(v.cluster, v.cell_type, by="you", at="2026-11-09T00:00:00")
    assert {c for c, d in da.decisions().items() if d["state"] == D.ACCEPTED} == {
        v.cluster for v in unflagged}

    assert scores() == before and da.verdict_csv() == csv_before
    assert memory.read_notes() == notes_before

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "panoscope" / "app.py"),
                           default_timeout=300)
    at.session_state["active_page"] = "summary"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    text = " ".join([m.value for m in at.markdown] + [c.value for c in at.caption])
    tally = re.compile(r"\b\d+\s*(?:/|of)\s*\d+\s+(?:confirmed|accepted|agree\w*|validated)",
                       re.IGNORECASE)
    assert not tally.search(text), tally.search(text)
