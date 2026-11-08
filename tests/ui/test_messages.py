"""Refusal and state messages, read off the screen (plan Stage 5, 5.25).

Each one is the app saying what is missing instead of drawing nothing or a traceback.
None of these strings was asserted anywhere before; each test builds the state that
produces it and checks the biologist can read the reason.
"""
from __future__ import annotations

import json
import shutil

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import open_app, run, screen_text, widget_keys


@pytest.fixture
def no_model(isolated, monkeypatch):
    from panoscope.agent import loop as agent_loop

    def _refuse(*_a, **_k):
        raise RuntimeError("no model key")

    monkeypatch.setattr(agent_loop, "chat", _refuse)
    return isolated


def _folder(tmp_path, *extra: str):
    """A results folder with the demo's marker table and panel (no labels, no manifest)."""
    from panoscope.data.study import Study

    demo = Study.open(DEMO_ID).inputs
    folder = tmp_path / "results"
    folder.mkdir()
    for name in ("markers_top.csv", "panel.parquet"):
        shutil.copyfile(demo / name, folder / name)
    for name in extra:
        (folder / name).write_text("a,b\n1,2\n")
    return folder


@pytest.mark.req("PLAN-T5-01", "PLAN-P1.1-01")
def test_a_path_that_is_not_a_folder_is_named_and_an_unknown_file_is_listed_not_guessed(
        no_model, tmp_path):
    at = open_app("upload")
    run(at.text_input(key="ob_folder").input(str(tmp_path / "nope")))
    assert f"{tmp_path / 'nope'} is not a folder." in screen_text(at)
    run(at.text_input(key="ob_folder").input(str(_folder(tmp_path, "qc_notes.csv"))))
    text = screen_text(at)
    assert "qc_notes.csv: not recognised by any registered method" in text
    assert "markers_top.csv: jazzPanda" in text


@pytest.mark.req("PLAN-T5-01", "ARCH-6.3-02", "PLAN-P1.2-01")
def test_intake_says_what_blocks_the_build_and_offers_no_build_until_declared(
        no_model, tmp_path, monkeypatch):
    from panoscope.reference import sources as ref_sources

    at = open_app("upload")
    run(at.text_input(key="ob_folder").input(str(_folder(tmp_path))))
    run(at.selectbox(key="ob_platform").select("xenium"))
    warning = next(w.value for w in at.warning if w.value.startswith("Blocked until declared"))
    assert "tissue" in warning
    assert "ob_build" not in widget_keys(at)

    at.text_input(key="ob_f_tissue").input("human breast cancer")
    run(at.button(key="ob_apply").click())
    assert not [w for w in at.warning if w.value.startswith("Blocked until declared")]
    assert "ob_build" in widget_keys(at)
    # The optional ontology id: no match keeps the tissue words, and says so.
    monkeypatch.setattr(ref_sources, "uberon_candidates", lambda text: [])
    run(at.button(key="ob_uberon_find").click())
    assert "No UBERON term matched; the study keeps its tissue words." in screen_text(at)


@pytest.mark.req("PLAN-T5-01", "ARCH-13.1-02")
def test_a_registered_question_with_no_result_says_so_through_the_real_app(app_tree):
    text = screen_text(open_app("contrasts"))
    assert "No contrasts result for this study." in text
    assert "The question is registered; nothing in this study answers it yet." in text


@pytest.mark.req("PLAN-T5-01", "UI-3-02")
def test_before_a_pin_the_stage_says_what_to_do_and_shows_no_bin_control(app_tree):
    at = open_app("markers", cluster="c1")
    assert "select a marker in the evidence table to see its transcript density" in \
        screen_text(at)
    assert "bin_um_global" not in widget_keys(at)


@pytest.mark.req("PLAN-T5-01", "UI-3-02")
def test_a_view_the_study_never_prepared_names_what_would_produce_it(app_tree):
    """demo_mini ships no viz/: each view says it was not prepared, never an empty frame."""
    at = open_app("markers", cluster="c1", markers="ERBB2",
                  study=(str(app_tree.parent), "demo_mini"))
    text = screen_text(at)
    assert "not prepared for this study" in text and "would come from" in text
    assert not at.get("plotly_chart")


@pytest.mark.req("PLAN-T5-01", "CORP-A24", "ARCH-7.3-01")
def test_a_tissue_with_no_reference_says_there_are_no_candidates_never_borrows_breast(
        app_tree):
    study = app_tree / "demo_mini"
    (study / "inputs" / "reference" / "lab_markers.csv").unlink()
    m = json.loads((study / "manifest.json").read_text())
    m["tissue"] = "human kidney"
    (study / "manifest.json").write_text(json.dumps(m))
    text = screen_text(open_app("markers", cluster="c1",
                                study=(str(app_tree.parent), "demo_mini")))
    assert "no candidate cell types in reference data for this tissue" in text
    assert "Tumor" not in text.split("no candidate cell types")[1]


@pytest.mark.req("PLAN-T5-01")
def test_my_notes_with_nothing_saved_says_how_a_note_gets_there(app_tree):
    at = open_app("summary")
    run(at.button(key="snav_labnotes").click())
    assert "No notes yet. Override or confirm a call in the chat" in screen_text(at)


@pytest.mark.req("PLAN-T5-01", "UI-5.3-07")
def test_the_board_counts_reviewed_clusters_and_says_when_all_are_done(app_tree):
    from panoscope.pipeline import store

    n = len(json.loads((app_tree / DEMO_ID / "inputs" / "cluster_key.json").read_text()))
    store.save_review_state({}, DEMO_ID)
    assert f"0 of {n} reviewed" in screen_text(open_app("summary"))
    store.save_review_state({f"c{i}": {"at": "t", "note_id": None} for i in range(1, n + 1)},
                            DEMO_ID)
    text = screen_text(open_app("summary"))
    assert "All clusters marked reviewed." in text and f"of {n} reviewed" not in text
