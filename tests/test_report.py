"""Tests for the interpretation report (``ui.report``).

Deterministic + offline: ``build_report`` is a pure function of the verdicts, the
lab notes, the cell-type notes, and the holistic review; the exporters serialize
that model. Nothing here fetches the network. Real verdicts (so cluster ids and
markers are real) are combined with synthetic notes.
"""
from __future__ import annotations

from io import BytesIO

import pytest

from panoscope.agent import holistic as agent_holistic
from panoscope.agent import verdict as agent_verdict
from panoscope.agent.types import Citation, Note, ScopeRef, Tension
from panoscope.ui import report as R


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
        author="reviewer",
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


# --------------------------------------------------------------------------- #
# The undecidables travel into the exported document (§9.10, concordance-travels)
#
# The replicate read reached ``panoscope explain``, the markers page and
# ``interp/records/assignment/``, and stopped. The document a biologist downloads
# -- the one that goes into a slide, a thesis chapter or a supplementary file --
# read "High confidence, driven by SYNG03" and never said the two sections read
# SYNG03 differently. These tests pin the carry, and they pin the two halves that
# a test asserting "the sentence appears somewhere" would leave open: that two
# different K-of-N do not render the same, and that a unit WITHOUT a divergence
# does not inherit its neighbour's.
# --------------------------------------------------------------------------- #
_DIVERGED = ("SYNG03 read differently across sampleA, sampleB (reference sampleA): "
             "batch effect, section quality or real biology, and 9.10 does not claim "
             "which")
_NOT_ASKABLE = ("replicate concordance (ARCHITECTURE 9.10) does not apply: the study "
                "holds one sample (sampleA), so there is no second reading")
_NO_WHY = ("replicate concordance (ARCHITECTURE 9.10) could not be assessed for this "
           "unit: sampleB reported no row under it")


def _report_with(undecidable: dict) -> R.ReportModel:
    """The same nine real verdicts, with undecidable sentences attached by unit."""
    return R.build_report(
        verdicts=agent_verdict.all_verdicts(),
        celltype_notes={},
        notes=[],
        holistic=None,
        panel_size=280,
        generated_at="2026-07-10",
        undecidable=undecidable,
    )


def _exported_text(model: R.ReportModel) -> str:
    """Everything the two document exporters would write, as one string."""
    return "\n".join(text for _, text in R._section_lines(model))


def test_a_divergence_reaches_the_exported_document_and_stays_on_its_own_unit():
    """The defect, in one assertion each way.

    The gene is named on c1 because c1's record says the sections disagreed about
    it, and it is absent from c2 because c2's does not. A carry that leaked the
    sentence onto every unit would read as a study-wide caveat and would be the
    same silence inverted: nothing in the document would separate the unit that
    diverged from the eight that did not.
    """
    model = _report_with({"c1": (_DIVERGED,)})
    text = _exported_text(model)

    assert "SYNG03 read differently across sampleA, sampleB" in text
    c1, c2 = (s for s in model.sections if s.cluster in ("c1", "c2"))
    assert c1.undecidable == (_DIVERGED,)
    assert c2.undecidable == ()
    # The on-page HTML and the editable working space carry it too: three exports,
    # and a caveat in one of them is a caveat the other two are silent about.
    assert "read differently" in R.report_html(model)
    assert "read differently" in R.default_cluster_summary(c1)
    assert "read differently" not in R.default_cluster_summary(c2)


def test_the_k_of_n_travels_rather_than_being_rounded_to_a_word():
    """"compared 1 of 12" and "compared 11 of 12" must not export the same bytes.

    This is the half a test that only greps for "concordance" would miss. Both
    units are not-divergent; the only thing separating a profile with one subject
    behind it from one with eleven is the denominator, and a document on which they
    read alike has thrown it away exactly where ``Concordance.coverage`` was added
    to keep it.
    """
    thin = "replicate concordance compared 1 of 12 subject(s) across sampleA, sampleB"
    thick = "replicate concordance compared 11 of 12 subject(s) across sampleA, sampleB"

    thin_text = _exported_text(_report_with({"c1": (thin,)}))
    thick_text = _exported_text(_report_with({"c1": (thick,)}))

    assert thin_text != thick_text
    assert "compared 1 of 12" in thin_text and "compared 1 of 12" not in thick_text
    assert "compared 11 of 12" in thick_text


def test_the_three_concordance_outcomes_stay_three_in_the_export():
    """Divergent / not comparable and the record cannot say why / never askable.

    Rendered as three distinct lines, in the record's own words. Collapsing any two
    into one sentence ("replicate check: not ok") is the four-states-into-two
    failure arriving in the file that leaves the building, where there is no hover
    to recover the difference from.
    """
    model = _report_with({"c1": (_DIVERGED,), "c2": (_NO_WHY,), "c3": (_NOT_ASKABLE,)})
    lines = [t for _, t in R._section_lines(model) if t.startswith("Not settled")]

    assert len(lines) == 3
    assert len({*lines}) == 3
    assert "read differently" in lines[0]
    assert "could not be assessed for this unit" in lines[1]
    assert "does not apply" in lines[2]


def test_the_sentences_are_carried_verbatim_and_nothing_is_derived():
    """The export may not shorten, re-word or re-count what the record said.

    ``ui/report.py`` renders; it does not interpret. A summarised divergence is a
    sentence nothing in the record backs, which is the one thing the confident
    floor forbids.
    """
    model = _report_with({"c1": (_DIVERGED, _NO_WHY)})
    said = [t for _, t in R._section_lines(model) if t.startswith("Not settled")]
    assert said == [f"Not settled by this measurement: {_DIVERGED}",
                    f"Not settled by this measurement: {_NO_WHY}"]


def test_the_document_bytes_actually_contain_it():
    """Through the real .docx writer, not only through the line list.

    ``_section_lines`` is shared by both exporters, but a style the writers do not
    handle would drop the line on the floor silently -- and the whole point of this
    stream is that a thing can be true on screen and missing from the file.
    """
    docx = pytest.importorskip("docx")
    model = _report_with({"c1": (_DIVERGED,)})
    doc = docx.Document(BytesIO(R.report_to_docx(model)))
    body = "\n".join(p.text for p in doc.paragraphs)
    assert "SYNG03 read differently across sampleA, sampleB" in body


def test_no_undecidable_leaves_the_report_exactly_as_it_was():
    """A study whose records carry nothing undecidable exports no extra line."""
    before = _exported_text(_report_with({}))
    assert "Not settled" not in before


# --------------------------------------------------------------------------- #
# The export reads the ACTIVE store, not one frozen at import
# --------------------------------------------------------------------------- #
def _record_store(root, dataset: str, unit: str, said: tuple[str, ...]) -> None:
    """Write the one file ``_dataset_undecidables`` reads, under ``root``."""
    import json

    from panoscope.core.record import RECORDS_DIR
    from panoscope.types.outputs import TaskKind

    d = root / "studies" / dataset / "interp" / RECORDS_DIR / TaskKind.ASSIGNMENT.value
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{unit}.json").write_text(
        json.dumps({"unit": unit, "undecidable": list(said)}), encoding="utf-8")


def test_the_undecidable_carry_follows_the_active_data_root(tmp_path, monkeypatch):
    """The replicate caveats in the exported document must come from THIS store.

    ``pipeline/paths`` used to freeze ``DEFAULT_DATASETS_ROOT`` at import, so this
    reader resolved whatever the environment said when the module first loaded while
    the verdicts beside it followed ``cfg.current_study()``. A study opened from
    outside ``data/studies/`` therefore exported with its replicate caveats missing;
    worse, with a same-named store in the frozen root, the OTHER store's sentences
    attached to these rows by unit id -- one store's concordance read printed into
    another store's document, which is the quiet half of a fabricated caveat.

    Two roots, one dataset id, different sentences: a reader that resolves at import
    returns the first root's answer for both halves, so the second assertion is the
    discriminator.
    """
    from panoscope.data import layout

    a, b = tmp_path / "A", tmp_path / "B"
    _record_store(a, "s", "c1", ("A says c1 could not be compared",))
    _record_store(b, "s", "c1", ("B says c1 diverged on GENEX",))

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(a))
    layout.clear_cache()
    assert R._dataset_undecidables("s") == {"c1": ("A says c1 could not be compared",)}

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(b))
    layout.clear_cache()
    assert R._dataset_undecidables("s") == {"c1": ("B says c1 diverged on GENEX",)}, (
        "the reader is still answering from the root it resolved first"
    )

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(tmp_path / "empty"))
    layout.clear_cache()
    assert R._dataset_undecidables("s") == {}, (
        "a store with no records must carry no caveat, not another store's"
    )
