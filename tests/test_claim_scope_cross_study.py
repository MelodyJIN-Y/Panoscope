"""Study scope on the note store, enforced fail-closed (CLAUDE.md).

The note store is ONE directory for every study (``context/corrections/``), and
cluster ids are ``c1..cN`` in every study, so "this note is about c2" names a
different cluster in each one. The composition readers in ``ui/data_access`` used to
match on type, scope and cluster and never on the study, so the demo's breast-cancer
``CAF`` override landed on the fixture's c2 — the biologist's own call, attributed to
a study they never made it in, reaching the Summary board, the exports and the
sign-off with click-back intact.

These tests exercise the PRODUCTION read, not a hand-built dict: a note is written
through ``memory.create_note`` under study A, study B is opened through
``cfg.use_study``, and ``da.composed_verdicts()`` is asked what it shows. Both
directions are pinned, because a gate that drops everything passes the easy half:

* study A's override still fires in study A (the half a blanket drop would break);
* it does not fire in study B (the defect);
* a note naming NO study fires nowhere (fail-closed), and ``withheld_notes`` is the
  sentence that says so instead of letting it disappear silently.

Hermetic: notes are written under ``tmp_path`` via ``memory._DEFAULT_BASE``, and the
two studies are only read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import memory
from panoscope.ui import data_access as da

STUDY_A = "xenium_hbreast_sample1"         # the demo: c1..c9
STUDY_B = "synthetic_multisample_fixture"  # the fixture: c1..c3, its own annotation


@pytest.fixture(autouse=True)
def notes_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the note store at a tmp dir. Nothing here may write into the repo."""
    base = tmp_path / "context"
    monkeypatch.setattr(memory, "_DEFAULT_BASE", base)
    return base


def _override(study: str, cluster: str, call: str) -> str:
    """Write a real celltype_override note under ``study``, through the real writer."""
    with cfg.use_study(study):
        note = memory.create_note(
            claim=f"{cluster} is {call} here",
            scope="cluster",
            basis="own_validation",
            cluster=cluster,
            subject_cell_type=call,
            note_type="celltype_override",
            literature_search=lambda _q: (),   # no lookup; tension stays empty
        )
    return note.id


def _calls(study: str) -> dict:
    with cfg.use_study(study):
        return {v.cluster: v.cell_type for v in da.composed_verdicts()}


def test_override_fires_in_the_study_it_was_made_in() -> None:
    """The half a blanket drop would break: the biologist's call still applies."""
    _override(STUDY_A, "c2", "CAF")
    assert _calls(STUDY_A)["c2"] == "CAF"


def test_override_does_not_cross_into_another_study() -> None:
    """The defect: study A's c2 call appearing on study B's c2."""
    _override(STUDY_A, "c2", "CAF")
    b = _calls(STUDY_B)
    assert b["c2"] != "CAF"
    # and it is the fixture's own annotation that is shown, not a blank
    assert b["c2"] == "Synthetic Type B"


def test_exclude_note_does_not_cross_into_another_study() -> None:
    """The other composition overlay reads the same ungated store."""
    with cfg.use_study(STUDY_A):
        memory.create_note(
            claim="c1 is a segmentation artefact here",
            scope="cluster",
            basis="own_validation",
            cluster="c1",
            note_type="exclude",
            literature_search=lambda _q: (),
        )
    with cfg.use_study(STUDY_B):
        assert [v.cluster for v in da.composed_verdicts() if v.exclude] == []
    with cfg.use_study(STUDY_A):
        assert "c1" in [v.cluster for v in da.composed_verdicts() if v.exclude]


def test_read_notes_shows_only_this_studys_notes() -> None:
    """The lab-note board and the exported report both go through ``da.read_notes``."""
    note_id = _override(STUDY_A, "c2", "CAF")
    with cfg.use_study(STUDY_A):
        assert [n.id for n in da.read_notes()] == [note_id]
    with cfg.use_study(STUDY_B):
        assert [n.id for n in da.read_notes()] == []


def test_lab_scoped_note_still_crosses_studies() -> None:
    """The lab rung means across studies — the gate must not flatten it."""
    with cfg.use_study(STUDY_A):
        note = memory.create_note(
            claim="POSTN is read as stromal bleed lab-wide",
            scope="lab",
            basis="own_validation",
            subject_markers=("POSTN",),
            note_type="marker_meaning",
            literature_search=lambda _q: (),
        )
    with cfg.use_study(STUDY_B):
        assert note.id in [n.id for n in da.read_notes()]


def test_note_naming_no_study_applies_nowhere_and_is_named(notes_base: Path) -> None:
    """Fail-closed on a scopeless note — and a sentence saying it was withheld.

    The file is written WITHOUT ``scope_ref.dataset`` and read back through the real
    loader, because the loader is where the bug was: it defaulted the missing dataset
    to whatever study was open, so the note became a note of that study.
    """
    corrections = notes_base / "corrections"
    corrections.mkdir(parents=True, exist_ok=True)
    (corrections / "nostudy.json").write_text(json.dumps({
        "id": "nostudy",
        "claim": "c2 is CAF",
        "scope": "cluster",
        "scope_ref": {"cluster": "c2"},        # no dataset
        "basis": "own_validation",
        "status": "firm",
        "type": "celltype_override",
        "subject_cell_type": "CAF",
        "created_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")

    for study in (STUDY_A, STUDY_B):
        assert _calls(study)["c2"] != "CAF", f"scopeless note fired in {study}"
        with cfg.use_study(study):
            assert [n.id for n in da.read_notes()] == []
            # the decision does not disappear silently: it is named, with its id, so
            # the sentence about it clicks back to the note like any other.
            assert [n.id for n in da.withheld_notes()] == ["nostudy"]


def test_withheld_notes_is_empty_when_every_note_names_its_study() -> None:
    """``withheld_notes`` reports the unscoped ones ONLY — a note belonging to another
    study is not withheld knowledge; it is shown in its own study."""
    _override(STUDY_A, "c2", "CAF")
    for study in (STUDY_A, STUDY_B):
        with cfg.use_study(study):
            assert da.withheld_notes() == []
