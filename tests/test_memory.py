"""Tests for agent/memory.py — the file-based reconciliation layer.

Covers the load-bearing invariants:
- Scope enforcement: a cluster-scoped note fires ONLY for its own cluster;
  dataset and lab notes fire for all.
- Reconciliation: an injected literature_search stub (no MCP) splits agree /
  dissent and the tension is attached to the note at creation.
- Cite-on-use: cite_note returns a Source (kind="mem").
- Disk round-trip: a note survives write -> read unchanged.
- Decision log writer under context/decisions/.

Every test points memory at a tmp base dir (no writes to the real context/).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import memory
from agent.types import Citation, Note, Source

DATASET = "xenium_hbreast_sample1"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def base(tmp_path: Path) -> Path:
    """A tmp base dir standing in for context/ (corrections/ + decisions/)."""
    return tmp_path / "context"


def _agree_one_dissent_one(query: str):
    """Stub literature_search: one agreeing + one dissenting real citation."""
    return [
        Citation(
            pmid="30000001",
            title="PDGFRB marks fibroblasts in breast stroma",
            authors="Doe J, Roe R",
            year=2021,
            journal="Nat Spatial",
            stance="agree",
            is_real=True,
        ),
        Citation(
            pmid="30000002",
            title="PDGFRB is a pericyte marker",
            authors="Smith A",
            year=2019,
            journal="Vasc Biol",
            stance="dissent",
            is_real=True,
        ),
    ]


def _make_cluster_note(base: Path, cluster: str = "c2") -> Note:
    return memory.create_note(
        claim="In our breast TME, PDGFRB marks CAFs here, not pericytes",
        scope="cluster",
        basis="own_validation",
        status="firm",
        cluster=cluster,
        subject_cell_type="Stromal",
        subject_markers=["PDGFRB"],
        dataset=DATASET,
        literature_search=_agree_one_dissent_one,
        base_dir=base,
    )


# --------------------------------------------------------------------------- #
# Scope enforcement (the headline invariant)
# --------------------------------------------------------------------------- #
def test_cluster_note_fires_only_for_its_cluster(base: Path):
    # Arrange
    note = _make_cluster_note(base, cluster="c2")

    # Act
    fires_c2 = memory.apply_notes("c2", dataset=DATASET, base_dir=base)
    fires_c1 = memory.apply_notes("c1", dataset=DATASET, base_dir=base)

    # Assert — cluster-scoped note fires for c2, NOT for c1
    assert note.id in {n.id for n in fires_c2}
    assert note.id not in {n.id for n in fires_c1}


def test_dataset_note_fires_for_all_clusters(base: Path):
    # Arrange
    ds_note = memory.create_note(
        claim="This dataset used the 280-gene Xenium breast panel",
        scope="dataset",
        basis="convention",
        status="firm",
        dataset=DATASET,
        base_dir=base,
    )

    # Act / Assert — a dataset note fires for every cluster in the dataset
    for cluster in ("c1", "c2", "c9"):
        fired = memory.apply_notes(cluster, dataset=DATASET, base_dir=base)
        assert ds_note.id in {n.id for n in fired}, f"dataset note should fire for {cluster}"


def test_lab_note_fires_across_datasets(base: Path):
    # Arrange
    lab_note = memory.create_note(
        claim="Our lab labels CAF subsets by convention X",
        scope="lab",
        basis="convention",
        status="firm",
        base_dir=base,
    )

    # Act / Assert — lab scope ignores dataset and cluster
    fired_here = memory.apply_notes("c2", dataset=DATASET, base_dir=base)
    fired_other = memory.apply_notes("c2", dataset="some_other_dataset", base_dir=base)
    assert lab_note.id in {n.id for n in fired_here}
    assert lab_note.id in {n.id for n in fired_other}


def test_dataset_note_does_not_fire_for_other_dataset(base: Path):
    # A dataset-scoped note must NOT leak into a different dataset (fail-closed).
    ds_note = memory.create_note(
        claim="dataset-scoped fact",
        scope="dataset",
        basis="convention",
        dataset=DATASET,
        base_dir=base,
    )
    fired = memory.apply_notes("c2", dataset="other_dataset", base_dir=base)
    assert ds_note.id not in {n.id for n in fired}


def test_cluster_note_requires_cluster(base: Path):
    with pytest.raises(ValueError):
        memory.create_note(
            claim="missing cluster",
            scope="cluster",
            basis="own_validation",
            cluster=None,
            base_dir=base,
        )


# --------------------------------------------------------------------------- #
# Reconciliation — the value is in the disagreement
# --------------------------------------------------------------------------- #
def test_reconcile_splits_agree_and_dissent(base: Path):
    # Arrange
    note = _make_cluster_note(base, cluster="c2")

    # Act — tension was attached at creation via the injected stub
    tension = note.tension

    # Assert — one agree, one dissent, both real, not thin
    assert len(tension.agree) == 1
    assert len(tension.dissent) == 1
    assert tension.agree[0].pmid == "30000001"
    assert tension.dissent[0].pmid == "30000002"
    assert tension.thin is False
    assert tension.query  # a query string was recorded


def test_reconcile_directly_returns_tension(base: Path):
    # reconcile() is callable standalone with an injected search.
    note = memory.create_note(
        claim="standalone reconcile check",
        scope="dataset",
        basis="paper",
        dataset=DATASET,
        base_dir=base,
    )
    tension = memory.reconcile(note, _agree_one_dissent_one)
    assert [c.stance for c in tension.agree] == ["agree"]
    assert [c.stance for c in tension.dissent] == ["dissent"]


def test_reconcile_drops_unreal_citations(base: Path):
    # A citation flagged is_real=False is never kept — fabricated refs are the
    # worst possible failure.
    def _has_fake(query: str):
        return [
            Citation(pmid="1", title="real", authors="", year=2020, journal="J",
                     stance="agree", is_real=True),
            Citation(pmid="99999999", title="fabricated", authors="", year=2020,
                     journal="J", stance="agree", is_real=False),
        ]

    note = memory.create_note(
        claim="drop-unreal check", scope="dataset", basis="paper",
        dataset=DATASET, base_dir=base,
    )
    tension = memory.reconcile(note, _has_fake)
    assert len(tension.agree) == 1
    assert tension.agree[0].pmid == "1"
    assert all(c.is_real for c in tension.agree)


def test_reconcile_thin_when_no_search(base: Path):
    note = memory.create_note(
        claim="thin check", scope="dataset", basis="paper",
        dataset=DATASET, base_dir=base,
    )
    tension = memory.reconcile(note, None)
    assert tension.thin is True
    assert tension.agree == ()
    assert tension.dissent == ()


# --------------------------------------------------------------------------- #
# Cite on use
# --------------------------------------------------------------------------- #
def test_cite_note_returns_mem_source(base: Path):
    # Arrange
    note = _make_cluster_note(base, cluster="c2")

    # Act
    src = memory.cite_note(note)

    # Assert
    assert isinstance(src, Source)
    assert src.kind == "mem"
    assert src.ref == note.id
    assert src.value == note.claim
    assert "cluster:c2" in src.detail


def test_render_citation_shows_tension(base: Path):
    note = _make_cluster_note(base, cluster="c2")
    rendered = memory.render_citation(note)
    assert f"[note:{note.id}]" in rendered
    assert "agree" in rendered and "dissent" in rendered
    assert "PMID:30000001" in rendered and "PMID:30000002" in rendered


# --------------------------------------------------------------------------- #
# Disk round-trip
# --------------------------------------------------------------------------- #
def test_note_round_trips_through_disk(base: Path):
    # Arrange — create writes the JSON file
    note = _make_cluster_note(base, cluster="c2")
    note_file = memory.corrections_dir(base) / f"{note.id}.json"
    assert note_file.exists()

    # Act — re-read from disk (fresh objects, no in-memory cache)
    reloaded = {n.id: n for n in memory.read_notes(base)}[note.id]

    # Assert — every field survives the trip
    assert reloaded == note
    assert reloaded.scope == "cluster"
    assert reloaded.scope_ref.cluster == "c2"
    assert reloaded.scope_ref.dataset == DATASET
    assert reloaded.subject_markers == ("PDGFRB",)
    assert len(reloaded.tension.agree) == 1
    assert len(reloaded.tension.dissent) == 1
    assert reloaded.tension.agree[0].pmid == "30000001"


def test_read_notes_empty_when_no_notes(base: Path):
    assert memory.read_notes(base) == []


# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #
def test_decision_log_written_under_decisions(base: Path):
    # create_note logs a note_created event; an explicit override event appends.
    note = _make_cluster_note(base, cluster="c2")
    memory.log_decision(
        kind="override_applied", cluster="c2", note_id=note.id,
        actor="you@example.com", detail="agent used biologist call",
        base_dir=base,
    )

    log_path = memory.decisions_dir(base) / "decision_log.jsonl"
    assert log_path.exists()

    events = memory.read_decisions(base)
    kinds = {e["kind"] for e in events}
    assert "note_created" in kinds
    assert "override_applied" in kinds
    override = next(e for e in events if e["kind"] == "override_applied")
    assert override["cluster"] == "c2"
    assert override["note_id"] == note.id


# --------------------------------------------------------------------------- #
# Immutability / supersede
# --------------------------------------------------------------------------- #
def test_supersede_creates_new_note_linked_to_old(base: Path):
    old = _make_cluster_note(base, cluster="c2")
    new = memory.supersede_note(
        old.id, base_dir=base, claim="Revised: PDGFRB marks a specific CAF subset",
    )
    assert new.id != old.id
    assert new.supersedes == old.id
    # old note is untouched on disk (immutability)
    still_there = memory.get_note(old.id, base)
    assert still_there == old


# --------------------------------------------------------------------------- #
# Draft-then-save (capture-at-override, two-tap confirm)
# --------------------------------------------------------------------------- #
def test_draft_note_reconciles_but_writes_nothing(base: Path):
    """draft_note attaches the tension yet persists nothing (biologist not asked)."""
    draft = memory.draft_note(
        claim="In our breast TME, PDGFRB marks CAFs here, not pericytes",
        scope="cluster",
        basis="own_validation",
        status="firm",
        cluster="c2",
        subject_markers=["PDGFRB"],
        dataset=DATASET,
        literature_search=_agree_one_dissent_one,
    )
    # Tension is computed (1 agree + 1 dissent) ...
    assert len(draft.tension.agree) == 1
    assert len(draft.tension.dissent) == 1
    # ... but nothing was written.
    assert memory.read_notes(base) == []


def test_save_draft_persists_with_confirmed_scope_and_kept_tension(base: Path):
    """The biologist re-scopes cluster->dataset at confirm; save keeps the tension
    (no second lookup) and drops the cluster so it cannot masquerade as cluster."""
    import dataclasses

    draft = memory.draft_note(
        claim="PDGFRB marks CAFs in this dataset",
        scope="cluster",
        basis="own_validation",
        cluster="c2",
        subject_markers=["PDGFRB"],
        dataset=DATASET,
        literature_search=_agree_one_dissent_one,
    )
    edited = dataclasses.replace(draft, scope="dataset", cluster=None)
    note = memory.save_draft(edited, base_dir=base)

    assert note.scope == "dataset"
    assert note.scope_ref.cluster is None
    assert len(note.tension.agree) == 1 and len(note.tension.dissent) == 1
    # Now on disk and firing dataset-wide.
    assert len(memory.read_notes(base)) == 1
    assert memory.apply_notes("c7", dataset=DATASET, base_dir=base)


def test_save_draft_cluster_scope_requires_cluster(base: Path):
    """A cluster-scoped draft with no cluster fails closed at save."""
    import dataclasses

    draft = memory.draft_note(
        claim="something", scope="dataset", basis="convention", dataset=DATASET
    )
    bad = dataclasses.replace(draft, scope="cluster", cluster=None)
    with pytest.raises(ValueError):
        memory.save_draft(bad, base_dir=base)


# --------------------------------------------------------------------------- #
# Typed / anchored notes (docs/note-capture-design.md)
# --------------------------------------------------------------------------- #
def test_typed_note_round_trips_with_anchors(base: Path):
    """A program_reinterpretation note keeps its type + gene-set anchor through disk."""
    note = memory.create_note(
        claim="This EMT program is co-infiltration from the tumor front, not the fibroblasts' own",
        scope="cluster",
        basis="paper",
        status="firm",
        cluster="c2",
        note_type="program_reinterpretation",
        subject_gene_sets=["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
        dataset=DATASET,
        base_dir=base,
    )
    (loaded,) = [n for n in memory.read_notes(base) if n.id == note.id]
    assert loaded == note
    assert loaded.type == "program_reinterpretation"
    assert loaded.subject_gene_sets == ("HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",)


def test_cross_cluster_note_fires_only_on_its_anchor_set(base: Path):
    """A cross_cluster note (dataset scope, anchored to c5+c7) fires on c5 and c7, never c1."""
    note = memory.create_note(
        claim="c5 and c7 are the same population, clustering over-split them",
        scope="dataset",
        basis="own_validation",
        status="firm",
        note_type="cross_cluster",
        subject_clusters=["c5", "c7"],
        dataset=DATASET,
        base_dir=base,
    )
    assert memory.note_in_scope(note, cluster="c5", dataset=DATASET)
    assert memory.note_in_scope(note, cluster="c7", dataset=DATASET)
    assert not memory.note_in_scope(note, cluster="c1", dataset=DATASET)  # not dataset-wide
    # and apply_notes routes through the same gate
    assert note.id in {n.id for n in memory.apply_notes("c7", dataset=DATASET, base_dir=base)}
    assert note.id not in {n.id for n in memory.apply_notes("c1", dataset=DATASET, base_dir=base)}


def test_gene_set_enters_the_reconcile_query(base: Path):
    """A program note's gene set drives the literature query (so it is cross-checked)."""
    seen = {}

    def _spy(query: str):
        seen["query"] = query
        return []

    draft = memory.draft_note(
        claim="co-infiltration, not intrinsic",
        scope="cluster",
        basis="paper",
        cluster="c2",
        note_type="program_reinterpretation",
        subject_gene_sets=["HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION"],
        dataset=DATASET,
        literature_search=_spy,
    )
    assert "epithelial mesenchymal transition" in seen["query"].lower()
    assert draft.tension.thin is True  # the spy returned nothing -> honestly thin, not faked


def test_note_never_carries_a_grounded_number():
    """Guardrail: the Note object has no numeric field a note could fabricate."""
    import dataclasses

    numeric_fields = {
        f.name for f in dataclasses.fields(Note)
        if f.type in ("float", "int") or "score" in f.name or "confidence" in f.name
    }
    assert numeric_fields == set(), f"Note must carry no grounded number; found {numeric_fields}"
