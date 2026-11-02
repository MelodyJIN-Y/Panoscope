"""The note store on ``Claim`` (IMPLEMENTATION §10) — the migration's invariants.

The note files hold real lab judgement and are already on disk, so the first thing
these tests pin is that the bridge READS them and leaves them alone: a legacy-shape
note (written before ``type`` and the ``subject_*`` anchors existed) must still parse
and still bridge, and the bytes on disk must be identical afterwards. A migration that
needs the user's files rewritten is not a migration, it is data loss with a changelog
entry.

The other four are the rules CLAUDE.md says this layer exists for:

* **scope does not widen.** The note lattice has a cluster rung the claim lattice does
  not, so a cluster note carries its cluster in the SUBJECT. The test that matters is
  the negative one: the c2 note must not come back for c3.
* **the value is in the disagreement.** An override keeps the call AND the papers that
  argue with it, ordered by authority, never merged, with the dissent reachable by id.
* **the biologist decides.** Only the LAB claim is committable; the literature claims
  the same note produced are refused by ``write_claim``.
* **thin is three states, not two.** "We searched and found nothing" and "no lookup
  completed" are different sentences, derived from fields the note already persists.

Every PMID here is a placeholder digit string and is deliberately NOT a real
identifier — a test that hardcoded a real one would be a fabricated citation with a
green tick beside it.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from panoscope.agent import memory
from panoscope.agent.types import Citation, Note, ScopeRef, Tension
from panoscope.claims.notes import (
    UnanchoredNote, load_notes_as_claims, lookup_state, note_to_claims,
)
from panoscope.claims.store import ClaimStore, NotCommittable, OutOfScope, write_claim
from panoscope.corpus.store import Resolution
from panoscope.types.claim import Authority, UnformedLabClaim, by_authority
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.measured import FrameRef
from panoscope.types.subject import EntityKind

DATASET = "xenium_hbreast_sample1"
HERE = f"dataset:{DATASET}"
ELSEWHERE = "dataset:kidney_io_panel"


# --------------------------------------------------------------------------- #
# Fixtures. The Conditions are the study intake the note was formed under; the
# bridge never invents one, because Claim refuses a LAB claim without frames.
# --------------------------------------------------------------------------- #
PANEL = FrameRef(EntityKind.GENE, "panel:xenium_hbreast_280", "sha256:" + "a" * 8, 280)


def _conditions() -> Conditions:
    return Conditions(
        assay=Condition("Xenium", Provenance.READ),
        modality=Condition("rna", Provenance.READ),
        tissue=Condition("human breast", Provenance.STATED),
        method=Condition("jazzPanda", Provenance.READ),
        segmentation=UNKNOWN,
        fixation=UNKNOWN,
        frames=(PANEL,),
    )


def _tension(*, agree=(), dissent=(), looked_up_at="2026-07-13T10:00:00+00:00") -> Tension:
    return Tension(agree=tuple(agree), dissent=tuple(dissent),
                   thin=not (agree or dissent), query="q", looked_up_at=looked_up_at)


def _cite(pmid: str, stance: str, *, is_real: bool = True, title: str = "") -> Citation:
    return Citation(pmid=pmid, title=title or f"A paper {pmid}", authors="Doe J",
                    year=2021, journal="J Spatial", stance=stance, is_real=is_real)


def _note(**over) -> Note:
    base = dict(
        id="n0000000001",
        claim="c2 is CAF, not generic Stromal",
        scope="cluster",
        scope_ref=ScopeRef(dataset=DATASET, cluster="c2"),
        basis="own_validation",
        status="firm",
        subject_cell_type="CAF",
        subject_markers=(),
        tension=_tension(),
        author="bio@example.com",
        created_at="2026-07-13T10:34:59+00:00",
        trigger="override",
        supersedes=None,
        type="celltype_override",
    )
    return Note(**{**base, **over})


# --------------------------------------------------------------------------- #
# 1. The files already on disk. Read them; do not touch them.
# --------------------------------------------------------------------------- #
# A note as the store held it BEFORE docs/note-capture-design.md added `type` and the
# anchor fields. Written here as raw JSON rather than via create_note, because the
# point is that the loader tolerates the keys that are missing.
_LEGACY_NOTE_JSON = {
    "id": "legacy000001",
    "claim": "c2 is CAF, not generic Stromal",
    "scope": "cluster",
    "scope_ref": {"dataset": DATASET, "cluster": "c2"},
    "basis": "own_validation",
    "status": "firm",
    "subject_cell_type": "CAF",
    "subject_markers": ["LUM", "POSTN"],
    "tension": {"agree": [], "dissent": [], "thin": True, "query": "CAF c2",
                "looked_up_at": ""},
    "author": "bio@example.com",
    "created_at": "2026-07-13T10:34:59+00:00",
    "trigger": "override",
    "supersedes": None,
    # NO "type", NO "subject_gene_sets", NO "subject_clusters", NO lineage/category.
}


@pytest.fixture
def legacy_store(tmp_path: Path) -> Path:
    base = tmp_path / "context"
    path = memory.corrections_dir(base) / f"{_LEGACY_NOTE_JSON['id']}.json"
    path.write_text(json.dumps(_LEGACY_NOTE_JSON, indent=2) + "\n", encoding="utf-8")
    return base


def test_a_pre_typed_note_file_still_reads_and_bridges(legacy_store: Path):
    """The shape that is actually on disk — no ``type``, no anchors — bridges whole."""
    store = load_notes_as_claims(legacy_store, _conditions())
    lab = [c for c in store if c.authority is Authority.LAB]
    assert len(lab) == 1
    c = lab[0]
    assert c.id == "lab:legacy000001"
    assert c.subject.key == "unit|u=unit:c2"          # the cluster, from scope_ref
    assert c.predicate == "is_cell_type" and c.object == "CAF"
    assert c.source == "note:legacy000001"
    assert c.created == "2026-07-13T10:34:59+00:00"   # the note's clock, not ours
    assert c.author == "bio@example.com"
    # The markers the note named ride in depends_on, so evaluability gates them
    # against the panel frame instead of them becoming subjects of their own.
    assert {e.id for e in c.depends_on} == {"LUM", "POSTN", "CAF"}


def test_reading_never_rewrites_the_users_file(legacy_store: Path):
    """A read is a read. The persisted JSON is byte-identical afterwards."""
    path = memory.corrections_dir(legacy_store) / "legacy000001.json"
    before = path.read_bytes()
    load_notes_as_claims(legacy_store, _conditions())
    assert path.read_bytes() == before


def test_the_real_store_bridges_if_the_repo_has_one():
    """Whatever is in this checkout's ``context/corrections/`` must bridge.

    Skipped on a clean checkout. Not a substitute for the fixture above — it is the
    check that the shape a *person* actually wrote is covered, not only the shape a
    test invents. Read-only; it asserts structure, never content.
    """
    notes = memory.read_notes()
    if not notes:
        pytest.skip("no notes in this checkout's context/corrections/")
    store = load_notes_as_claims(None, _conditions())
    assert len(store) >= len(notes)
    for c in store:
        if c.authority is Authority.LAB:
            assert c.source.startswith("note:")
            assert c.conditions is not None and c.conditions.frames


# --------------------------------------------------------------------------- #
# 2. Scope does not widen. The negative case is the one that matters.
# --------------------------------------------------------------------------- #
def _store(note: Note) -> ClaimStore:
    return ClaimStore(note_to_claims(note, _conditions()))


def test_a_cluster_note_fires_for_its_cluster_and_no_other():
    store = _store(_note())
    assert [c.id for c in store.about("unit|u=unit:c2", HERE).value] == ["lab:n0000000001"]
    assert store.about("unit|u=unit:c3", HERE).value == ()


def test_a_cluster_note_never_crosses_datasets():
    store = _store(_note())
    assert store.about("unit|u=unit:c2", ELSEWHERE).value == ()


def test_out_of_scope_use_raises_not_warns():
    """Retrieval filters; *use* raises. Fail-closed is a property of the lattice."""
    store = _store(_note())
    assert store.cite("lab:n0000000001", HERE).id == "lab:n0000000001"
    with pytest.raises(OutOfScope):
        store.cite("lab:n0000000001", ELSEWHERE)


def test_a_dataset_note_anchored_to_a_gene_becomes_a_gene_claim():
    """"KRT7 is unreliable here" fires dataset-wide, so the gene IS the subject."""
    note = _note(type="marker_convention", scope="dataset",
                 scope_ref=ScopeRef(dataset=DATASET, cluster=None),
                 subject_cell_type=None, subject_markers=("KRT7",),
                 claim="KRT7 is unreliable on this panel; never trust it alone")
    store = _store(note)
    c = store.about("gene|g=gene:KRT7", HERE).value[0]
    assert c.predicate == "means_here"          # not a cell-type call
    assert c.object == note.claim
    assert c.scope == HERE


def test_a_lab_note_applies_downward_but_not_upward():
    note = _note(type="marker_convention", scope="lab",
                 scope_ref=ScopeRef(dataset=DATASET, cluster=None),
                 subject_cell_type=None, subject_markers=("KRT7",))
    c = note_to_claims(note, _conditions())[0]
    assert c.scope == "lab"
    assert c.applies_in(HERE) and c.applies_in(ELSEWHERE) and c.applies_in("lab")


def test_a_cross_cluster_note_fires_on_each_anchor_and_nowhere_else():
    note = _note(type="cross_cluster", scope="dataset",
                 scope_ref=ScopeRef(dataset=DATASET, cluster=None),
                 subject_cell_type=None, subject_clusters=("c5", "c7"),
                 claim="c5 and c7 are one population")
    store = _store(note)
    assert {c.id for c in store} == {"lab:n0000000001:c5", "lab:n0000000001:c7"}
    assert [c.id for c in store.about("unit|u=unit:c5", HERE).value] == ["lab:n0000000001:c5"]
    assert store.about("unit|u=unit:c6", HERE).value == ()


def test_a_note_about_nothing_raises_rather_than_disappearing():
    note = _note(scope="dataset", scope_ref=ScopeRef(dataset=DATASET, cluster=None),
                 subject_cell_type=None, subject_markers=(), claim="a thought")
    with pytest.raises(UnanchoredNote):
        note_to_claims(note, _conditions())


def test_a_lab_claim_without_its_conditions_is_refused():
    """The bridge cannot mint the frames a note was formed under, and does not try."""
    with pytest.raises((UnformedLabClaim, ValueError)):
        note_to_claims(_note(), dataclasses.replace(_conditions(), frames=()))


# --------------------------------------------------------------------------- #
# 3. The value is in the disagreement.
# --------------------------------------------------------------------------- #
def _contested() -> Note:
    return _note(tension=_tension(agree=(_cite("30000001", "agree"),),
                                  dissent=(_cite("30000002", "dissent"),)))


def test_an_override_keeps_the_call_and_the_dissent_together():
    store = _store(_contested())
    got = by_authority(store.about("unit|u=unit:c2", HERE).value)
    # The biologist's call first, then both papers. Nothing merged, nothing dropped.
    assert [c.authority for c in got] == [Authority.LAB,
                                          Authority.LITERATURE, Authority.LITERATURE]
    lab = got[0]
    assert lab.object == "CAF"                       # the call survives intact
    # ...and the paper that argues with it is reachable BY ID from the call.
    assert lab.tension == ("lab:n0000000001:lit:30000002",)
    dissent = store.get(lab.tension[0])
    assert dissent.source == "pmid:30000002"
    assert dissent.subject == lab.subject            # same subject: a rival reading


def test_agreement_is_kept_too_and_is_not_listed_as_tension():
    store = _store(_contested())
    agree = store.get("lab:n0000000001:lit:30000001")
    assert agree is not None and agree.authority is Authority.LITERATURE
    assert agree.id not in store.get("lab:n0000000001").tension


def test_a_fabricated_citation_becomes_nothing():
    note = _note(tension=_tension(dissent=(_cite("99999999", "dissent", is_real=False),)))
    store = _store(note)
    assert [c.authority for c in store] == [Authority.LAB]
    assert store.get("lab:n0000000001").tension == ()


def test_only_the_lab_claim_is_committable(tmp_path: Path):
    """The agent may propose; only the biologist commits. Both halves, one note."""
    claims = note_to_claims(_contested(), _conditions())
    lab = [c for c in claims if c.authority is Authority.LAB]
    lit = [c for c in claims if c.authority is Authority.LITERATURE]
    assert lab and lit
    for c in lit:
        with pytest.raises(NotCommittable):
            write_claim(c, tmp_path / "lab_store")
    assert write_claim(lab[0], tmp_path / "lab_store").exists()


def test_nothing_is_merged_or_deduped():
    """Two notes that say opposite things about one cluster both survive, ordered."""
    a = _note(id="n000000000a", claim="c2 is CAF", subject_cell_type="CAF")
    b = _note(id="n000000000b", claim="c2 is pericyte", subject_cell_type="Pericyte")
    store = ClaimStore(note_to_claims(a, _conditions()) + note_to_claims(b, _conditions()))
    got = store.about("unit|u=unit:c2", HERE).value
    assert {c.object for c in got} == {"CAF", "Pericyte"}


def test_status_is_kept_one_to_one():
    assert note_to_claims(_note(status="tentative"), _conditions())[0].status == "tentative"
    assert note_to_claims(_note(status="firm"), _conditions())[0].status == "firm"


# --------------------------------------------------------------------------- #
# 4. Thin literature is THREE states. Derived, never stored.
# --------------------------------------------------------------------------- #
def test_resolved_when_real_citations_came_back():
    assert lookup_state(_tension(agree=(_cite("30000001", "agree"),))) is Resolution.RESOLVED


def test_not_found_when_a_lookup_completed_and_found_nothing():
    assert lookup_state(_tension(looked_up_at="2026-07-13T10:00:00+00:00")) \
        is Resolution.NOT_FOUND


def test_not_looked_up_when_no_lookup_completed():
    assert lookup_state(_tension(looked_up_at="")) is Resolution.NOT_LOOKED_UP


def test_a_failed_connector_is_not_thin_literature():
    """The bug this closes: a lookup that RAISED used to read exactly like an empty one."""
    def boom(query: str):
        raise RuntimeError("connector down")

    def empty(query: str):
        return []

    note = _note()
    assert lookup_state(memory.reconcile(note, boom)) is Resolution.NOT_LOOKED_UP
    assert lookup_state(memory.reconcile(note, empty)) is Resolution.NOT_FOUND
    assert lookup_state(memory.reconcile(note, None)) is Resolution.NOT_LOOKED_UP


def test_the_three_states_render_as_three_different_sentences():
    down = memory.render_citation(_note(tension=_tension(looked_up_at="")))
    searched = memory.render_citation(_note(tension=_tension()))
    found = memory.render_citation(_contested())
    assert "no literature lookup completed" in down
    assert "is thin" not in down                 # the whole point: do not say thin
    assert "is thin" in searched and "searched" in searched
    assert "dissent" in found and "PMID:30000002" in found
    assert down != searched and searched != found


def test_the_derived_state_ignores_a_wrong_stored_boolean():
    """``thin`` is a value a writer could get wrong; the citations are the truth."""
    lying = Tension(agree=(_cite("30000001", "agree"),), dissent=(), thin=True,
                    query="q", looked_up_at="2026-07-13T10:00:00+00:00")
    assert lookup_state(lying) is Resolution.RESOLVED
