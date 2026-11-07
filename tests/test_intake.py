"""Invariants for study intake (ARCHITECTURE §6.2-§6.4).

Intake is where a guess would enter the *foundation* rather than an answer, and where
a twenty-question interview would lose the biologist before the tool has said anything.
So the tests come in two halves, one for each failure:

* **Nothing is asked that a file answers.** The real demo store resolves with an empty
  blocking list, and the only items left on its checklist are optional ones.
* **Nothing is recorded that nobody agreed to.** A ``Proposal`` cannot reach a
  ``Draft``, ``INFERRED`` has exactly one source line, and a blocking field cannot be
  declared unknown.

The third theme is the partial-state one: ``Conditions`` raises on a blocking unknown
and must keep doing so, which is precisely why ``Draft`` exists and why
``test_a_draft_holds_what_a_conditions_refuses_to`` asserts both halves in one test.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from panoscope.agent.intake import (
    CLUSTERING, FIELDS, ITEMS, MEASURED_SET, Draft, Proposal, Status, derive,
)
from panoscope.data.study import Study
from panoscope.platforms.xenium import XeniumAdapter
from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, DISEASE_STATE, UNKNOWN, Blocked, Condition,
    Conditions, Provenance,
)
from panoscope.types.contracts import UndeclaredClustering
from panoscope.types.measured import EntityKind

ADAPTER = XeniumAdapter()
INTAKE_SOURCE = Path(__file__).resolve().parent.parent / "panoscope" / "agent" / "intake.py"


def _store(root: Path, sid: str, **manifest) -> Study:
    """A minimal readable Xenium store. Written to tmp_path, never to the repo."""
    d = root / "studies" / sid
    (d / "inputs").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"dataset_id": sid, **manifest}))
    pd.DataFrame({"gene": ["CD8A", "PDCD1", "FOXP3"]}).to_parquet(
        d / "inputs" / "panel.parquet")
    pd.DataFrame([("CD8A", "c1", 1.0, 0.4), ("PDCD1", "NoSig", 0.0, 0.0)],
                 columns=["gene", "top_cluster", "glm_coef", "pearson"],
                 ).to_csv(d / "inputs" / "markers_top.csv", index=False)
    return Study.open(sid, data_root=root)


@pytest.fixture
def demo(demo_id: str, datasets_root: Path) -> Study:
    return Study.open(demo_id, data_root=datasets_root.parent).require()


@pytest.fixture
def multi_sample(tmp_path: Path) -> Study:
    """Three samples, no declared mode: §6.3's fourth blocker, which is the kidney
    study's shape."""
    return _store(tmp_path, "kidney_io", platform="Xenium", tissue="human kidney",
                  samples=["s1", "s2", "s3"], clusters=["c1", "c2"])


# --------------------------------------------------------------------------- #
# §6.4 step 1: zero questions for anything on disk
# --------------------------------------------------------------------------- #
def test_the_demo_study_asks_nothing_and_resolves(demo: Study) -> None:
    """The real store, end to end. Every blocker is on disk, so intake stops early.

    If this ever fails with a blocking item, intake has started interviewing a
    biologist about a file it could have read -- §6.4's named failure mode.
    """
    draft = derive(ADAPTER, demo)
    assert draft.missing == (), f"asked about something on disk: {draft.missing}"
    assert draft.complete

    conditions = draft.resolve()
    assert isinstance(conditions, Conditions)
    assert conditions.tissue.provenance is Provenance.READ
    assert conditions.assay.value == "Xenium"
    assert conditions.frames, "the panel IS the capability frame"


@pytest.mark.parametrize("damage", ["absent", "controls_only"])
def test_a_study_with_no_measured_set_is_blocked_not_ticked(tmp_path, demo, damage):
    """§6.3's structural blocker has to be able to fire.

    ``Draft.status`` tested the truthiness of the whole ``frames`` tuple, which
    mixes GENE, CELL_TYPE and UNIT refs -- so any store whose manifest declares
    ``clusters`` kept it non-empty and the blocker was unreachable on every
    readable store. A panel holding only control probes ticked "measured_set ...
    (read from the store)" under "nothing to ask", directly above a problem line
    saying the store declares no capability for any gene, and ``resolve()``
    returned a ``Conditions`` whose gene frame admits nothing.

    Two shapes, because they fail differently: a controls-only panel leaves an
    EMPTY ``panel:`` frame, while an absent file leaves no gene frame of that name
    at all and ``detect:`` alone would otherwise satisfy a size test.
    """
    import shutil

    root = tmp_path / "root"
    study_dir = root / "studies" / demo.id
    study_dir.parent.mkdir(parents=True)
    shutil.copytree(demo.dir, study_dir)
    pq = study_dir / "inputs" / "panel.parquet"
    if damage == "absent":
        pq.unlink()
    else:
        df = pd.read_parquet(pq).head(3).copy()
        df["gene"] = ["NegControlProbe_00001", "NegControlCodeword_0500",
                      "BLANK_0001"]
        df.to_parquet(pq, index=False)

    draft = derive(XeniumAdapter(), Study.open(demo.id, data_root=root))
    assert draft.status(MEASURED_SET) is Status.BLOCKING, (
        f"a study with no measured set ({damage}) ticked "
        f"{draft.status(MEASURED_SET).name}; frames="
        f"{[(f.name, f.size) for f in draft.frames]}"
    )
    assert MEASURED_SET in draft.missing, draft.missing
    with pytest.raises(Exception):
        draft.resolve()

    # ...and the healthy study is untouched by the stricter test.
    healthy = derive(XeniumAdapter(), demo)
    assert healthy.status(MEASURED_SET) is Status.KNOWN
    assert MEASURED_SET not in healthy.missing


def test_everything_derived_is_read_never_inferred(demo: Study) -> None:
    """Step 1 may only write ``READ``: nobody has proposed or confirmed anything yet.

    An ``INFERRED`` coming out of a pure file read would mean a confirmation that
    never happened, which is the one provenance that cannot be audited after the fact.
    """
    draft = derive(ADAPTER, demo)
    for field in FIELDS + (CLUSTERING,):
        cond = draft.condition(field)
        assert cond.provenance in (Provenance.READ, Provenance.UNKNOWN), field


def test_the_checklist_shows_three_columns_and_nothing_else(demo: Study) -> None:
    """What is known, what is missing and blocking, what is optional (§6.4 step 2)."""
    checklist = derive(ADAPTER, demo).checklist()
    rendered = checklist.render()

    assert checklist.blocking == ()
    assert {i.field for i in checklist.of(Status.KNOWN)} >= {MEASURED_SET, "assay",
                                                            "tissue"}
    # OPTIONAL is the declared-unknown-ok fields MINUS any the store actually
    # answered. The demo answers modality (it follows from platform=Xenium), so
    # it sits under KNOWN -- the checklist asks about what is missing, not about
    # every field that COULD be missing.
    optional = {i.field for i in checklist.of(Status.OPTIONAL)}
    known = {i.field for i in checklist.of(Status.KNOWN)}
    assert optional == set(DECLARED_UNKNOWN_OK) - known
    assert "modality" in known
    assert "Known from the files" in rendered
    assert "Optional" in rendered
    assert "BLOCKING" not in rendered, "a complete study must not show a blocking header"


def test_the_measured_set_line_is_derived_from_the_files(demo: Study) -> None:
    """Panel size is dynamic across studies; the checklist counts, never asserts.

    The 280 is read out of the frames in this test rather than typed in, so the second
    study's immuno-oncology panel changes the line instead of contradicting it.
    """
    draft = derive(ADAPTER, demo)
    panel = next(f for f in draft.frames if f.name.startswith("panel:"))
    item = next(i for i in draft.checklist().items if i.field == MEASURED_SET)
    assert f"({panel.size})" in (item.value or "")


# --------------------------------------------------------------------------- #
# The partial-state problem
# --------------------------------------------------------------------------- #
def test_a_draft_holds_what_a_conditions_refuses_to(tmp_path: Path) -> None:
    """Both halves, in one test, because weakening either is the tempting fix.

    ``Conditions`` must keep raising on a blocking unknown -- that is what lets every
    downstream check read it without a defensive branch. The checklist still has to
    render the study that is missing one. So intake carries a ``Draft`` and resolves
    it; it does not teach ``Conditions`` to be incomplete.
    """
    store = _store(tmp_path, "no_tissue", platform="Xenium", clusters=["c1"])
    draft = derive(ADAPTER, store)  # must not raise

    assert "tissue" in draft.missing
    assert any(i.status is Status.BLOCKING and i.field == "tissue"
               for i in draft.checklist().items)

    with pytest.raises(Blocked):
        Conditions(assay=draft.assay, modality=draft.modality, tissue=UNKNOWN,
                   method=draft.method, segmentation=draft.segmentation,
                   fixation=draft.fixation, frames=draft.frames)
    with pytest.raises(Blocked):
        draft.resolve()

    # The partial read, asserted: the blocking unknown costs intake the tissue and
    # NOTHING else. Before `PlatformAdapter.reads` existed this store yielded six
    # unknowns and a banner, including the assay the manifest plainly declares.
    assert draft.assay.known and draft.assay.value == "Xenium"
    assert draft.assay.provenance is Provenance.READ
    assert not draft.tissue.known
    assert not draft.unread_because, (
        "the adapter reported on every field; the could-not-read banner is for an "
        "adapter that refuses the store or reports fewer fields, not for a study "
        "that is merely incomplete"
    )
    assert "did not report on every field" not in draft.checklist().render()
    supplied = draft.stated("tissue", "UBERON:0002113").stated("assay", "Xenium")
    assert supplied.resolve().tissue.known


def test_resolve_names_every_gap_at_once(tmp_path: Path) -> None:
    """Not the first gap: §6.4 exists so a study is not fixed one round trip at a time.

    ``Conditions`` alone would report its own two and say nothing about clustering,
    which it deliberately cannot express.
    """
    store = _store(tmp_path, "two_gaps", platform="Xenium",
                   samples=["a", "b"], clusters=["c1"])
    with pytest.raises(Blocked) as exc:
        derive(ADAPTER, store).resolve()
    assert "tissue" in str(exc.value) and CLUSTERING in str(exc.value)


# --------------------------------------------------------------------------- #
# §6.3's fourth blocker: a checklist item, not a traceback
# --------------------------------------------------------------------------- #
def test_an_undeclared_clustering_mode_is_an_item_not_a_crash(multi_sample: Study) -> None:
    """The adapter raises; intake renders. A traceback is not a checklist.

    The line has to carry the adapter's own message, because the biologist's next
    action is to open that specific manifest, and "undeclared clustering mode" with
    no path is a stack trace wearing a checkbox.
    """
    with pytest.raises(UndeclaredClustering):
        ADAPTER.clustering_mode(multi_sample)

    draft = derive(ADAPTER, multi_sample)        # must not raise
    item = next(i for i in draft.checklist().items if i.field == CLUSTERING)
    assert item.status is Status.BLOCKING
    assert item.value is None
    assert "manifest.json" in item.why and "3 samples" in item.why
    assert draft.missing == (CLUSTERING,)


def test_a_declared_mode_unblocks_the_study(multi_sample: Study) -> None:
    draft = derive(ADAPTER, multi_sample).stated(CLUSTERING, "Joint")
    assert draft.clustering == Condition("joint", Provenance.STATED)
    assert draft.complete and isinstance(draft.resolve(), Conditions)
    assert "BLOCKING" not in draft.checklist().render()


def test_an_unrecognised_word_is_not_a_declaration(multi_sample: Study) -> None:
    """``"batch-corrected"`` is a word the writer hoped would be understood.

    Accepting it here would let an undeclared study pass the blocker by typo -- in
    the chat path rather than the manifest path, which is the same hole twice.
    """
    draft = derive(ADAPTER, multi_sample)
    with pytest.raises(UndeclaredClustering):
        draft.stated(CLUSTERING, "batch-corrected")


def test_a_single_sample_study_is_not_asked(demo: Study) -> None:
    """One sample makes the three modes the same statement; asking would be noise."""
    assert derive(ADAPTER, demo).status(CLUSTERING) is Status.KNOWN


# --------------------------------------------------------------------------- #
# A model guess is not a condition
# --------------------------------------------------------------------------- #
def test_a_proposal_cannot_be_stored_as_a_condition(demo: Study) -> None:
    """``INFERRED`` means proposed AND confirmed. An unconfirmed guess is not a
    weaker condition -- it is not in the object at all.

    Checked by type rather than by convention, because frozen dataclasses do not
    enforce their annotations and ``replace(draft, tissue=guess)`` is the exact line
    a future caller writes when the model has an answer and the biologist is slow.
    """
    guess = Proposal("tissue", "UBERON:0000310", "ERBB2 and ESR1 are on the panel")
    assert not isinstance(guess, Condition)
    assert not hasattr(guess, "provenance"), "a proposal has no provenance to read"

    draft = derive(ADAPTER, demo)
    with pytest.raises(TypeError):
        dataclasses.replace(draft, tissue=guess)
    with pytest.raises(TypeError):
        draft.confirm("UBERON:0000310")  # type: ignore[arg-type]


def test_confirmation_is_the_only_route_to_inferred() -> None:
    """One source line writes ``INFERRED``, and it is reached only through ``confirm``.

    A source assertion rather than a behavioural one on purpose: the invariant is
    that no *other* path exists, and no amount of calling the paths that do exist can
    demonstrate that.
    """
    source = INTAKE_SOURCE.read_text()
    constructions = source.count("Provenance.INFERRED)")  # the \u0029 excludes _HOW's key
    assert constructions == 1, "INFERRED is constructed somewhere other than confirm"
    assert "Provenance.INFERRED)" in source.split("def confirmed")[1].split("def ")[0]


def test_a_confirmed_proposal_is_inferred_and_a_stated_one_is_not(demo: Study) -> None:
    draft = derive(ADAPTER, demo)
    confirmed = draft.confirm(Proposal("fixation", "FFPE", "the manifest says FFPE-era"))
    assert confirmed.fixation == Condition("FFPE", Provenance.INFERRED)
    assert draft.stated("fixation", "FFPE").fixation.provenance is Provenance.STATED


# --------------------------------------------------------------------------- #
# Blocking means blocking; declared unknown names itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", BLOCKS_THE_STUDY)
def test_a_blocking_field_cannot_be_declared_unknown(demo: Study, field: str) -> None:
    """The decision §6.3 records, against the alternative of propagating it.

    An unknown tissue propagated as a caveat is a study whose literature is scoped to
    no organ, reported as a footnote nobody reads.
    """
    with pytest.raises(Blocked):
        derive(ADAPTER, demo).declare_unknown(field)


def test_an_undeclared_clustering_mode_cannot_be_declared_unknown(
        multi_sample: Study) -> None:
    """The conditional blocker blocks the same way the unconditional ones do."""
    with pytest.raises(Blocked):
        derive(ADAPTER, multi_sample).declare_unknown(CLUSTERING)


@pytest.mark.parametrize("field", DECLARED_UNKNOWN_OK)
def test_a_declared_unknown_propagates_and_names_itself(demo: Study, field: str) -> None:
    """"segmentation unknown" is actionable; "insufficient information" is not."""
    draft = derive(ADAPTER, demo).declare_unknown(field)
    item = next(i for i in draft.checklist().items if i.field == field)
    assert item.status is Status.OPTIONAL
    assert f'"{field} unknown"' in item.why
    assert field in draft.resolve().declared_unknown


def test_the_measured_set_is_not_a_chat_answer(demo: Study) -> None:
    """It IS the panel file. Typing it would make the capability frame prose."""
    with pytest.raises(ValueError):
        derive(ADAPTER, demo).stated(MEASURED_SET, "280 genes")


def test_a_blank_answer_is_not_an_answer(demo: Study) -> None:
    """A blank recorded as ``STATED`` reads as known to every applicability check,
    which then compares against nothing and passes."""
    with pytest.raises(ValueError):
        derive(ADAPTER, demo).stated("tissue", "   ")


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def test_the_checklist_covers_every_condition_field_exactly_once() -> None:
    """A seventh ``Conditions`` field must appear on the checklist or fail here.

    ``Conditions`` already forces a new field into one of §6.3's two columns; this
    forces it onto the surface the biologist actually sees, which is the half that
    would otherwise be filled in silently by whoever constructs the object.
    """
    conditions_fields = {f.name for f in dataclasses.fields(Conditions)} - {"frames"}
    assert set(FIELDS) == conditions_fields
    assert len(ITEMS) == len(set(ITEMS)) == len(FIELDS) + 2
    assert set(ITEMS) - {MEASURED_SET, CLUSTERING} == conditions_fields


def test_every_blocking_item_says_what_it_costs(tmp_path: Path) -> None:
    """A blocker with no reason reads as bureaucracy and gets guessed at."""
    store = _store(tmp_path, "bare", samples=["a", "b"], clusters=["c1"])
    draft = Draft(**{f: UNKNOWN for f in FIELDS}, frames=(), clustering=UNKNOWN,
                  clustering_blocked_because="two samples, nothing declared")
    assert set(draft.missing) == set(BLOCKS_THE_STUDY) | {MEASURED_SET, CLUSTERING}
    for item in draft.checklist().items:
        if item.status is Status.BLOCKING:
            assert len(item.why) > 20, item.field
    assert derive(ADAPTER, store).missing  # a real store takes the same path


# --------------------------------------------------------------------------- #
# Disease state and §6.4 background
# --------------------------------------------------------------------------- #
def test_an_unknown_disease_state_resolves_and_treatment_takes_no_guess(demo: Study) -> None:
    """Undeclared disease state is the starting state; "unknown" is not a treatment value."""
    draft = derive(ADAPTER, demo)
    assert set(DISEASE_STATE) <= set(draft.resolve().declared_unknown)
    assert draft.stated("disease_treatment", "Naive").disease_treatment.value == "naive"
    with pytest.raises(ValueError):
        draft.stated("disease_treatment", "unknown")


def test_an_unknown_tissue_blocks(tmp_path: Path) -> None:
    draft = derive(ADAPTER, _store(tmp_path, "no_tissue", platform="Xenium"))
    assert "tissue" in draft.missing
    with pytest.raises(Blocked):
        draft.resolve()


def test_background_persists_with_provenance_and_papers_are_identifiers(
        tmp_path: Path) -> None:
    draft = derive(ADAPTER, _store(tmp_path, "s", platform="Xenium", tissue="t"))
    draft = draft.stated("papers_of_interest", "pmid:1").stated("lab_conventions", "x")
    raw = json.loads(json.dumps(draft.background.to_manifest()))
    again = derive(ADAPTER, _store(tmp_path, "s2", platform="Xenium", tissue="t",
                                   background=raw))
    assert again.background == draft.background
    assert again.background.papers_of_interest[0].provenance is Provenance.STATED
    with pytest.raises(ValueError):
        draft.stated("papers_of_interest", "Smith et al. 2020, Nature")


def test_expected_cell_types_never_enter_the_label_space(tmp_path: Path) -> None:
    """A prior to test, not an input: it is in neither the frames nor the conditions."""
    stated = derive(ADAPTER, _store(tmp_path, "s0", platform="Xenium", tissue="t")
                    ).stated("expected_cell_types", "Zeta cell")
    store = _store(tmp_path, "s", platform="Xenium", tissue="t",
                   background=stated.background.to_manifest())
    (store.inputs / "cluster_key.json").write_text(json.dumps({"c1": {"cell_type": "T cell"}}))
    draft = derive(ADAPTER, store)
    assert draft.background.expected_cell_types[0].value == "Zeta cell"
    labels = {m for f in ADAPTER.frames(store) if f.kind is EntityKind.CELL_TYPE
              for m in f.members}
    assert labels == {"T cell"}
    assert "Zeta cell" not in repr(draft.resolve())
