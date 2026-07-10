"""Tests for the interpretation report (``ui.report``).

Deterministic + offline: ``build_report`` is a pure function of the verdicts, the
lab notes, the cell-type notes, and the holistic review; the exporters serialize
that model. Nothing here fetches the network. Real verdicts (so cluster ids and
markers are real) are combined with synthetic notes.
"""
from __future__ import annotations

from io import BytesIO

import pytest

from agent import holistic as agent_holistic
from agent import verdict as agent_verdict
from agent.types import Citation, Note, ScopeRef, Tension
from ui import report as R


def _cite(pmid: str = "22264274") -> Citation:
    return Citation(pmid=pmid, title="t", authors="a", year=2012, journal="j")


def _note(*, cluster=None, scope="cluster", claim="", agree=(), dissent=(), thin=False) -> Note:
    return Note(
        id="n_" + (cluster or scope),
        claim=claim,
        scope=scope,
        scope_ref=ScopeRef(dataset="xenium_hbreast_sample1", cluster=cluster),
        basis="own_validation",
        status="firm",
        subject_cell_type=None,
        subject_markers=(),
        tension=Tension(agree=tuple(agree), dissent=tuple(dissent), thin=thin, query="q", looked_up_at=""),
        author="melody",
        created_at="2026-07-10T00:00:00",
        trigger="override",
        supersedes=None,
    )


@pytest.fixture
def report() -> R.ReportModel:
    notes = [
        _note(cluster="c1", claim="c1 is luminal tumor (our IHC)", agree=(_cite(),)),
        _note(cluster=None, scope="dataset", claim="this dataset is ER+ breast"),
    ]
    return R.build_report(
        verdicts=agent_verdict.all_verdicts(),
        celltype_notes={"c1": {"summary": "Malignant luminal epithelial cells.", "pmid": "22264274"}},
        notes=notes,
        holistic=agent_holistic.holistic_review(),
        panel_size=280,
        generated_at="2026-07-10",
    )


def test_build_report_shape(report):
    assert report.n_clusters == 9
    assert report.panel_size == 280
    assert report.dataset == "xenium_hbreast_sample1"
    for s in report.sections:
        assert s.cell_type and s.confidence


def test_cluster_note_lands_under_its_cluster_only(report):
    c1 = next(s for s in report.sections if s.cluster == "c1")
    assert any("luminal tumor" in n.claim for n in c1.notes)
    c2 = next(s for s in report.sections if s.cluster == "c2")
    assert all("luminal tumor" not in n.claim for n in c2.notes)


def test_dataset_note_buckets_separately(report):
    assert any("ER+ breast" in n.claim for n in report.dataset_notes)
    for s in report.sections:
        assert all("ER+ breast" not in n.claim for n in s.notes)


def test_cluster_note_tension_is_rendered(report):
    c1 = next(s for s in report.sections if s.cluster == "c1")
    note = next(n for n in c1.notes if "luminal tumor" in n.claim)
    assert "PMID:22264274" in note.tension


def test_biology_note_and_pmid_carried(report):
    c1 = next(s for s in report.sections if s.cluster == "c1")
    assert "Malignant luminal" in c1.biology
    assert c1.biology_pmid == "22264274"


def test_settle_line_on_mixed_cluster_flags_offpanel_without_experiments(report):
    settled = [s for s in report.sections if s.settle]
    assert settled, "expected at least one cluster with a settle line"
    joined = " ".join(s.settle for s in settled).lower()
    assert "off-panel" in joined or "never measured" in joined
    assert "ihc" not in joined and "experiment" not in joined


def test_clean_cluster_has_no_settle_line(report):
    c1 = next(s for s in report.sections if s.cluster == "c1")
    assert c1.settle == ""


def test_report_html_has_links_and_sections(report):
    html = R.report_html(report)
    assert "pubmed.ncbi.nlm.nih.gov" in html  # a PMID was linkified
    assert "Interpretation summary" in html
    assert "Cross-cluster review" in html


def test_docx_export_is_valid_and_has_content(report):
    from docx import Document

    data = R.report_to_docx(report)
    assert data[:2] == b"PK" and len(data) > 0
    doc = Document(BytesIO(data))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "ERBB2" in text
    assert "Cross-cluster review" in text


def test_pdf_export_is_valid(report):
    data = R.report_to_pdf(report)
    assert data[:4] == b"%PDF" and len(data) > 500


def test_holistic_none_is_handled():
    rep = R.build_report(
        verdicts=agent_verdict.all_verdicts(),
        celltype_notes={},
        notes=[],
        holistic=None,
        panel_size=280,
    )
    assert rep.coherence_notes == () and rep.refinements == ()
    assert R.report_to_pdf(rep)[:4] == b"%PDF"
