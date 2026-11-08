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
