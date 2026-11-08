"""The Summary board's sign-off, driven by real clicks (no ``?sign=`` shortcut)."""
from __future__ import annotations

import pytest


@pytest.mark.req("UI-5.2-04", "UI-5.3-07")
def test_a_contested_sign_off_made_by_clicking_keeps_its_validation_note(app_tree, monkeypatch):
    """c9 is flagged verify: Confirm & mark reviewed drafts a validation note, Confirm
    reviewed saves it, and the review record links that note."""
    from streamlit.testing.v1 import AppTest

    from panoscope.agent import memory, tools
    from panoscope.pipeline import store

    monkeypatch.setattr(tools, "_literature_search_fn", lambda: None)   # offline: thin

    def _run() -> None:
        import streamlit as st

        from panoscope.ui import state
        from panoscope.ui import summary

        state.init_state()
        st.session_state.setdefault("sum_active_section", "c9")
        summary.render_summary_page()

    at = AppTest.from_function(_run, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    at.button(key="ddsign_c9").click().run()
    assert "c9" not in store.load_review_state()           # nothing until confirmed
    at.button(key="sof_ok_c9").click().run()
    assert not at.exception, [str(e.value) for e in at.exception]

    rec = store.load_review_state()["c9"]
    notes = {n.id: n for n in memory.read_notes(tools.memory_base_dir())}
    assert rec["note_id"] in notes, (rec, sorted(notes))
    assert notes[rec["note_id"]].type == "validation"
