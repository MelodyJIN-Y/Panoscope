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

Hermetic: notes are written under ``tmp_path`` via ``PANOSCOPE_CONTEXT_DIR``, and the
two studies are only read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import memory
from panoscope.ui import data_access as da
from panoscope.ui import report
from panoscope.ui import summary

STUDY_A = "xenium_hbreast_sample1"         # the demo: c1..c9
STUDY_B = "synthetic_multisample_fixture"  # the fixture: c1..c3, its own annotation


@pytest.fixture(autouse=True)
def notes_base(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the note store at a tmp dir. Nothing here may write into the repo."""
    base = tmp_path / "context"
    # The env var, not a module attribute: ``memory._default_base()`` resolves it
    # per call, which is what makes conftest's autouse isolation real.
    monkeypatch.setenv("PANOSCOPE_CONTEXT_DIR", str(base))
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


@pytest.mark.req("ARCH-5.3-03")
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


# --------------------------------------------------------------------------- #
# The lab rung travels (above) — but it may not travel with its conditions
# stripped. These pin the other half: wherever it arrives, the surface names the
# study it was FORMED on.
# --------------------------------------------------------------------------- #
def _lab_note(study: str):
    """A lab-scoped marker-meaning note, written through the real writer under ``study``."""
    with cfg.use_study(study):
        return memory.create_note(
            claim="POSTN here reads as stromal bleed",
            scope="lab",
            basis="own_validation",
            subject_markers=("POSTN",),
            note_type="marker_meaning",
            literature_search=lambda _q: (),
        )


class _FakeSt:
    """Records ``st.markdown`` calls. The note cards are pure markdown, so this is the
    whole Streamlit surface they touch — no runtime needed."""

    def __init__(self) -> None:
        self.out: list[str] = []

    def markdown(self, body: str, **_kw: object) -> None:
        self.out.append(body)


def test_lab_note_rendered_in_another_study_names_the_study_that_formed_it() -> None:
    """The defect: a judgement formed on 280-gene FFPE breast arriving on another
    study's board as a bare "all datasets", under the biologist's own authority, with
    nothing on screen saying which tissue formed it.

    The HARD half is study B. Naming the formation study in study A would also be
    satisfied by a renderer that printed whatever study happens to be open, so study B
    is the input that separates the two: the span must name A and must NOT name B.
    """
    note = _lab_note(STUDY_A)

    with cfg.use_study(STUDY_B):
        # it really does arrive here — the lab rung is meant to travel
        arrived = [n for n in da.read_notes() if n.id == note.id]
        assert arrived, "the lab rung must still cross studies"

        span = report.note_scope_label(arrived[0])
        assert STUDY_A in span, span
        assert STUDY_B not in span, span
        # "all datasets" is still true and still said; it is no longer said alone
        assert "all datasets" in span

        # every surface the span ships through: the .docx/PDF text and the
        # Summary note board.
        line = report._note_line(arrived[0])
        assert STUDY_A in report._note_text(line)

        st = _FakeSt()
        summary._render_lab_note_cards(st, arrived)
        board = "\n".join(st.out)
        assert STUDY_A in board, board
        assert STUDY_B not in board, board


def test_lab_note_in_its_own_study_still_names_that_study() -> None:
    """The easy half, kept: the suffix is not a cross-study special case. A reader on
    study A sees the same sentence, so "formed on …" never has to mean "formed
    elsewhere" — the reader compares it with the study they are in."""
    note = _lab_note(STUDY_A)
    with cfg.use_study(STUDY_A):
        assert STUDY_A in report.note_scope_label(note)


def test_a_dataset_scoped_note_gains_no_formation_suffix() -> None:
    """Only the lab rung travels, so only the lab rung needs the suffix. A
    dataset-scoped note fires nowhere but the study it names, and "this dataset"
    already says which one that is — repeating it would be noise on every row."""
    with cfg.use_study(STUDY_A):
        note = memory.create_note(
            claim="this dataset is ER+ breast",
            scope="dataset",
            basis="own_validation",
            note_type="caveat",
            literature_search=lambda _q: (),
        )
        assert report.note_scope_label(note) == "this dataset"


def test_withheld_note_reaches_the_summary_board_as_a_sentence(notes_base: Path) -> None:
    """``withheld_notes`` had no production caller: the gate dropped the note and the
    board said nothing, which is a committed judgement vanishing without a sentence."""
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

    with cfg.use_study(STUDY_A):
        st = _FakeSt()
        summary._render_withheld_notes(st, da.withheld_notes())
        board = "\n".join(st.out)
        assert "nostudy" in board and "c2 is CAF" in board
        assert "names no study" in board
    # and nothing is drawn when there is nothing to say
    quiet = _FakeSt()
    summary._render_withheld_notes(quiet, [])
    assert quiet.out == []
