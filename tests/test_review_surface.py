"""The evidence-first annotation review, driven through the app on a fresh study with no
labels: evidence before the proposal, an override with both assessments, a deferral,
and the export refused until every cluster is decided."""
from __future__ import annotations

import shutil
from pathlib import Path

from panoscope.data.study import Study

APP = str(Path(__file__).resolve().parents[1] / "panoscope" / "app.py")


def _fresh(tmp_path: Path) -> Path:
    """demo_mini copied under ``tmp_path`` without its interp/ tree: no labels at all."""
    root = tmp_path / "data"
    shutil.copytree(Study.open("demo_mini").dir, root / "studies" / "fresh")
    shutil.rmtree(root / "studies" / "fresh" / "interp")
    return root


def _md(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_a_fresh_study_is_reviewed_evidence_first_and_exports_once_decided(
        tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from panoscope.data import layout
    from panoscope.ui import state

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    layout.clear_cache()
    root = _fresh(tmp_path)
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state[state.K_STUDY] = (str(root), "fresh")
    at.session_state["active_page"] = "markers"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]

    # Evidence first: markers with their states, the candidates' canonical lists with
    # an off-panel marker marked never measured, and the corpus line. No proposal yet.
    md = _md(at)
    assert "pano-sf-glyph s-never_measured" in md and "Cancer-Associated Fibroblast" in md
    assert "corpus not built — cannot settle" in md
    assert 'class="pano-verdict"' not in md
    at.toggle(key="rv_reveal_c1").set_value(True).run()
    assert '<div class="pano-verdict"><h1>not yet called</h1>' in _md(at)
    assert at.button(key="rv_accept_c1").disabled          # nothing to accept

    # Override c1 with a candidate: both assessments, the call's anchor on its note.
    at.toggle(key="rv_calling_c1").set_value(True).run()
    at.selectbox(key="rv_pick_c1").select("HER2+ Breast Tumor Epithelial").run()
    at.button(key="rv_save_c1").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]
    md = _md(at)
    assert md.count('class="pano-rv-assess"') == 2 and 'data-claim="note:' in md
    assert "decision &middot; overridden" in md
    assert "High &#9873; verify</div><div class=\"pano-sf-dim\">glm_coef direct · driver ERBB2" in md
    assert '<div class="pano-verdict"><h1>not yet called</h1>' in md   # still the proposal

    # Defer c2; c3 still OPEN, so the export is refused and names it.
    at.button(key="rail_c2").click().run()
    at.button(key="rv_defer_c2").click().run()
    at.session_state["active_page"] = "summary"
    at.run()
    assert not at.download_button
    assert "still OPEN: c3" in "\n".join(c.value for c in at.caption)
    at.session_state["active_page"] = "markers"
    at.run()
    at.button(key="rail_c3").click().run()
    at.button(key="rv_defer_c3").click().run()
    at.session_state["active_page"] = "summary"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert len(at.download_button) == 2


def test_a_shipped_label_stays_one_tap_away_in_the_rail_and_the_chat(
        tmp_path, monkeypatch) -> None:
    """The rail, the chat opening and Accept all wait for the tap: a shipped label is a proposal."""
    from streamlit.testing.v1 import AppTest

    from panoscope.data import layout
    from panoscope.ui import state

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    layout.clear_cache()
    root = tmp_path / "data"
    shutil.copytree(Study.open("demo_mini").dir, root / "studies" / "labelled")
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state[state.K_STUDY] = (str(root), "labelled")
    at.session_state["active_page"] = "markers"
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "HER2+" not in at.button(key="rail_c1").label
    assert 'class="conv-title"' not in _md(at) and at.button(key="rv_accept_c1").disabled

    at.toggle(key="rv_reveal_c1").set_value(True).run()
    assert not at.exception, [str(e.value) for e in at.exception]
    assert "HER2+ Breast Tumor Epithelial" in at.button(key="rail_c1").label
    assert 'class="conv-title"' in _md(at) and not at.button(key="rv_accept_c1").disabled
    assert "HER2+" not in at.button(key="rail_c2").label


def test_pathways_unlock_per_cluster_on_a_call_and_read_the_biologists_label(app_tree) -> None:
    """OPEN and DEFERRED say why and link back; ACCEPTED and OVERRIDDEN open on the call."""
    from streamlit.testing.v1 import AppTest

    from panoscope.agent import config as cfg
    from panoscope.pipeline import paths, store

    p = paths.review_state_json(cfg.BUNDLED_DEMO_ID)
    saved = p.read_bytes() if p.exists() else None
    store.save_decisions({"c2": {"state": "DEFERRED", "by": "t", "at": "t"},
                          "c3": {"state": "OVERRIDDEN", "label": "Fooblast", "by": "t",
                                 "at": "t"}}, cfg.BUNDLED_DEMO_ID)
    try:
        at = AppTest.from_file(APP, default_timeout=300)
        at.session_state["active_page"] = "pathways"
        for cluster, locked in (("c1", "open"), ("c2", "deferred"), ("c3", None)):
            at.session_state["selected_cluster"] = cluster
            at.run()
            assert not at.exception, [str(e.value) for e in at.exception]
            md = _md(at)
            if locked:
                assert f"Call this cluster first: {cluster} is {locked}" in md
                at.button(key=f"unlock_{cluster}").click().run()
                assert at.session_state["active_page"] == "markers"
                at.session_state["active_page"] = "pathways"
            else:
                assert "Call this cluster first" not in md and "Fooblast" in md
    finally:
        p.write_bytes(saved) if saved is not None else p.unlink()


def test_a_deferred_cluster_stays_locked_when_a_chat_override_note_exists(
        app_tree, monkeypatch) -> None:
    """DEFERRED never unlocks Pathways; the note still names what the cluster shows."""
    from panoscope.agent import compose
    from panoscope.agent import config as cfg
    from panoscope.pipeline import paths, store
    from panoscope.ui import data_access as da

    monkeypatch.setattr(compose, "override_call", lambda c: "Fooblast")
    p = paths.review_state_json(cfg.BUNDLED_DEMO_ID)
    saved = p.read_bytes() if p.exists() else None
    store.save_decisions({"c2": {"state": "DEFERRED", "by": "t", "at": "t"}},
                         cfg.BUNDLED_DEMO_ID)
    try:
        assert da.effective_call("c2") is None and da.display_cell_type("c2") == "Fooblast"
        assert da.effective_call("c1") == "Fooblast"
    finally:
        p.write_bytes(saved) if saved is not None else p.unlink()
