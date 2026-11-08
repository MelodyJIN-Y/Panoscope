"""Per-cluster annotation decisions: OPEN start, the three terminal calls, the export
gate, and the second deterministic assessment on the biologist's label."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from panoscope.agent import memory
from panoscope.agent.types import NoteDraft, Tension
from panoscope.core.measure import Reading, measure
from panoscope.core.verdict import assess
from panoscope.data.study import NOT_YET_CALLED
from panoscope.pipeline import decisions as D
from panoscope.pipeline import store
from panoscope.reference.graph import Edge, Graph
from panoscope.reference.view import View
from panoscope.types.measured import Coverage, Frame, MeasuredWorld
from panoscope.types.subject import EntityKind, cell_type, gene, subject

AT = "2026-11-08T10:00:00+00:00"
KW = dict(by="tester", at=AT, dataset_id="fresh")


def _view() -> View:
    def marks(g, ct):
        return Edge(subject=gene(g), predicate="marks", object=cell_type(ct),
                    source="test@1", citation="pmid:12345678", licence="CC-BY")
    return View(graph=Graph(edges=(marks("ERBB2", "Tumor"), marks("LUM", "Stromal")),
                            consulted=("test@1",)))


def _build():
    """Items for one cluster, and the proposal ("Tumor") assessed on them as built."""
    panel = Frame(EntityKind.GENE, "panel:t", frozenset({"ERBB2", "LUM", "ACTA2"}))
    world, reported = MeasuredWorld((panel,)), frozenset({"ERBB2", "LUM"})
    items = tuple(measure(Reading(subject("gene", g=gene(g)), s, "glm_coef"),
                          world, reported).value
                  for g, s in (("ERBB2", 21.4), ("LUM", 0.4), ("ACTA2", None)))
    return SimpleNamespace(evidence=SimpleNamespace(items=items),
                           verdict=assess(items, frozenset({"ERBB2"}),
                                          Coverage("replicates", 1, 4, 3)).value)


def test_every_cluster_starts_open_and_export_names_the_open_ones(tmp_path) -> None:
    for c in ("c1", "c2", "c3"):
        assert D.state(store.load_decisions("fresh", tmp_path), c) == D.OPEN
    D.accept("c1", "Tumor", root=tmp_path, **KW)
    D.defer("c3", root=tmp_path, **KW)
    got = store.load_decisions("fresh", tmp_path)
    assert D.export_refusal(("c1", "c2", "c3"), got).endswith("still OPEN: c2")
    assert D.call(got, "c1") == "Tumor" and D.call(got, "c3") is None
    assert D.export_refusal(("c1", "c3"), got) is None


def test_nothing_to_accept_without_a_proposal(tmp_path) -> None:
    with pytest.raises(ValueError, match="no proposal"):
        D.accept("c1", NOT_YET_CALLED, root=tmp_path, **KW)


def test_override_commits_a_lab_note_and_keeps_the_sign_offs(tmp_path) -> None:
    store.save_review_state({"c1": {"at": AT, "note_id": None}}, "fresh", tmp_path)
    draft = NoteDraft(claim="c2 is CAF by our IHC", scope="cluster", basis="own_validation",
                      status="firm", cluster="c2", subject_cell_type="CAF",
                      subject_markers=(), dataset="fresh",
                      tension=Tension((), (), True, "", ""))
    with pytest.raises(ValueError, match="scoped to 'c2'"):     # the note would fire on c2
        D.override("c3", draft, root=tmp_path, **KW)
    rec = D.override("c2", draft, root=tmp_path, **KW)
    note = memory.get_note(rec["note_id"])
    assert (note.author, note.subject_cell_type, note.basis) == ("tester", "CAF",
                                                                 "own_validation")
    got = store.load_decisions("fresh", tmp_path)
    assert (D.state(got, "c2"), D.call(got, "c2"), got["c2"]["scope"]) == (
        D.OVERRIDDEN, "CAF", "cluster")
    assert "c1" in store.load_review_state("fresh", tmp_path)
    assert [e["kind"] for e in memory.read_decisions()][-1] == "decision_overridden"


def test_the_biologist_label_gets_the_same_band_function_beside_the_proposal() -> None:
    b = _build()
    a = D.assessments(b, "Stromal", _view())
    assert a.proposal is b.verdict
    assert a.biologist.driver.subject.one("g").id == "LUM"      # its own canonical driver
    assert a.biologist.band != a.proposal.band
    assert a.biologist.replicate_coverage is b.verdict.replicate_coverage  # same measurement
    assert D.assessments(b, None, _view()).biologist is None


def test_a_label_with_no_reference_node_assesses_on_an_empty_canonical_set() -> None:
    a = D.assessments(_build(), "Zeta cell", _view())
    assert a.biologist.basis == "no canonical driver" and a.biologist.verify
    assert "not in reference data" in a.reads_as


def test_expected_types_sit_beside_candidates_never_alone() -> None:
    assert D.label_space(_view(), ("Zeta cell", "tumor")) == (("Tumor", "Stromal"),
                                                              ("Zeta cell",))
    assert D.label_space(View(graph=Graph()), ("Zeta cell",)) == ((), ())


def test_a_summary_sign_off_never_accepts_a_cluster_with_no_call(monkeypatch) -> None:
    from types import SimpleNamespace

    from panoscope.agent import compose
    from panoscope.ui import data_access as da

    monkeypatch.setattr(da, "signed_off", lambda: {"c1": {}, "c2": {}})
    monkeypatch.setattr(da, "verdict_for", lambda c: SimpleNamespace(
        cell_type="Tumor" if c == "c1" else NOT_YET_CALLED))
    monkeypatch.setattr(compose, "override_notes", lambda: {
        "c3": SimpleNamespace(subject_cell_type="Fooblast", id="n3")})
    monkeypatch.setattr(store, "load_decisions", lambda *a, **k: {})
    decs = da.decisions()
    assert D.call(decs, "c1") == "Tumor"
    assert D.export_refusal(["c1", "c2", "c3"], decs).endswith("still OPEN: c2")
    # One reader: the sign-off and the chat override unlock what the export counts.
    assert da.effective_call("c1") == "Tumor" and da.effective_call("c2") is None
    assert decs["c3"] == {"state": D.OVERRIDDEN, "label": "Fooblast", "note_id": "n3"}
