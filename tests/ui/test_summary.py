"""The Summary board, driven by real clicks (no ``?sign=`` shortcut): sign-off, the
drill-down, My notes and the export (plan Stage 5: 5.17-5.20)."""
from __future__ import annotations

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import open_app, run, screen_text, widget_keys


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

        from panoscope.ui import state, summary

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


# --------------------------------------------------------------------------- #
# Stage 5b: the drill-down by clicks, My notes, and the export (5.17-5.20)
# --------------------------------------------------------------------------- #


def _review() -> dict:
    from panoscope.pipeline import store

    return store.load_review_state()


def _notes() -> list:
    from panoscope.agent import memory, tools

    return memory.read_notes(tools.memory_base_dir())


def _drill(cluster: str):
    return open_app("summary", sum_active_section=cluster)


@pytest.fixture
def offline(monkeypatch):
    from panoscope.agent import tools

    monkeypatch.setattr(tools, "_LITERATURE_SEARCH_FN", lambda q: [])


@pytest.mark.req("UI-5.3-07")
def test_an_unflagged_cluster_is_marked_reviewed_and_reopened_by_clicks(app_tree, offline):
    """c1 is verify=FALSE in verdicts.csv: one tap marks it, with no note; the same
    button then reads as reopen and clears the mark."""
    at = _drill("c1")
    assert at.button(key="ddsign_c1").label == "Mark reviewed"
    at = run(at.button(key="ddsign_c1").click())
    assert _review()["c1"]["note_id"] is None and _notes() == []
    assert at.button(key="ddsign_c1").label == "✓ Reviewed — reopen"
    at = run(at.button(key="ddsign_c1").click())
    assert "c1" not in _review()
    assert at.button(key="ddsign_c1").label == "Mark reviewed"


@pytest.mark.req("UI-5.3-07", "UI-5.2-04")
def test_cancelling_a_contested_sign_off_writes_neither_note_nor_mark(app_tree, offline):
    at = _drill("c9")
    assert at.button(key="ddsign_c9").label == "Confirm & mark reviewed"
    at = run(at.button(key="ddsign_c9").click())
    assert "sof_ok_c9" in widget_keys(at)
    at = run(at.button(key="sof_no_c9").click())
    assert "c9" not in _review() and _notes() == []
    assert "sof_ok_c9" not in widget_keys(at)


_BASIS_WORDS = {"own_validation": "our own data", "paper": "a paper", "convention": "convention"}


@pytest.mark.req("SKM-S6-01")
def test_the_sign_off_card_offers_every_basis_and_saves_the_one_shown(app_tree, offline):
    """The basis vocabulary is closed at three (paper | own_validation | convention).
    The card offers all three, and Confirm saves exactly the option it shows picked:
    a proposed basis the card cannot show would be swapped silently for another."""
    at = run(_drill("c9").button(key="ddsign_c9").click())
    [bg] = [b for b in at.get("button_group") if (b.key or "").startswith("sof_basis_c9")]
    assert sorted(bg.options) == sorted(_BASIS_WORDS.values()), bg.options
    shown = bg.value
    run(at.button(key="sof_ok_c9").click())
    [note] = _notes()
    assert _BASIS_WORDS[note.basis] == shown, (note.basis, shown)


@pytest.mark.req("SKM-S6-01")
def test_my_notes_shows_the_basis_in_words_and_no_placeholder_author(app_tree, offline):
    from panoscope.agent import config as cfg
    from panoscope.agent import memory, tools

    memory.create_note(claim="c2 is CAF here", scope="cluster", basis="own_validation",
                       cluster="c2", subject_cell_type="CAF", dataset=cfg.DATASET_ID,
                       literature_search=lambda q: (), base_dir=tools.memory_base_dir())
    at = run(open_app("summary").button(key="snav_labnotes").click())
    text = screen_text(at)
    assert "c2 is CAF here" in text
    assert "basis: our own data" in text, text
    assert "own_validation" not in text and "you@example.com" not in text


def _decide_all() -> None:
    from panoscope.agent import config as cfg
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    for c in cfg.CLUSTER_ORDER:
        D.accept(c, da.verdict_for(c).cell_type, by="tester", at="2026-11-09T00:00:00")


@pytest.mark.req("UI-5.2-06", gap=True)
def test_export_asks_one_heavy_confirm_before_it_freezes(app_tree, offline):
    _decide_all()
    at = open_app("summary")
    labels = [str(w.label).lower() for k in ("button", "download_button") for w in at.get(k)]
    assert any("freeze" in lb or "confirm export" in lb for lb in labels), labels


@pytest.mark.req("UI-5.8.6-01", gap=True)
def test_the_summary_offers_a_csv_with_the_decision_columns_after_the_locked_eleven(
        app_tree, offline):
    """The locked 11 are the committed verdicts.csv header; the five decision columns
    follow them (interface-design §5.8.6)."""
    import csv

    _decide_all()
    with (app_tree / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        locked = next(csv.reader(f))
    assert len(locked) == 11
    at = open_app("summary")
    assert any("csv" in str(w.label).lower() for w in at.get("download_button"))
    from panoscope.ui import data_access as da

    header = next(csv.reader(da.verdict_csv().splitlines()))
    assert header == locked + ["decided_by", "decided_at", "decision", "reason_codes",
                               "decision_id"]
