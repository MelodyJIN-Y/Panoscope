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


# --------------------------------------------------------------------------- #
# The two things the export used to drop (found by the human-breast E2E run)
# --------------------------------------------------------------------------- #
def test_the_panel_absence_rule_reaches_the_document(report) -> None:
    """The worst omission the export had, and the reason it mattered.

    On screen, a Stromal call names COL1A1, COL1A2, DCN, VIM and FAP as never
    measured. The exported document named none of them, so a reader could not tell
    a call resting on a complete canonical set from one where five of its markers
    were never on the panel -- and the panel-absence rule is the thing this project
    is built around. ``ClusterVerdict.offpanel_notes`` had carried them the whole
    time; ``build_report`` never read the field.

    Asserted on the RENDERED text of all three outputs, not on the model: a field
    that is populated and never rendered is the same defect one layer down.
    """
    with_off = [s for s in report.sections if s.off_panel]
    assert with_off, "no section carries an off-panel marker; the fixture cannot show this"
    s = with_off[0]
    genes = [g for g, _m in s.off_panel]

    html = R.report_html(report)
    assert all(g in html for g in genes), f"{genes} missing from the HTML report"
    assert "never measured" in html.lower()

    text = R.default_cluster_summary(s)
    assert all(g in text for g in genes), f"{genes} missing from the working document"

    docx = bytes(R.report_to_docx(report))
    assert docx[:2] == b"PK"
    import zipfile
    from io import BytesIO
    body = zipfile.ZipFile(BytesIO(docx)).read("word/document.xml").decode("utf-8")
    assert all(g in body for g in genes), f"{genes} missing from the .docx"


def test_an_override_to_an_unannotated_call_keeps_the_panel_absence_block() -> None:
    """Renaming a call must not delete a true never-measured statement.

    Moving the off-panel set with the call was right; returning EMPTY when this
    study's annotation does not name the new one was not. Overriding c2 Stromal ->
    CAF -- a real override on the demo data -- emptied COL1A1, COL1A2, DCN, VIM and
    FAP out of the exported document while the markers table beside it still showed
    all five, because the table builds rows from the record's own evidence and not
    from the composed verdict. The headline rule vanished from the handed-on
    artifact for exactly the cluster the biologist had curated.

    Being off-panel is a fact about the PANEL, not the call, so the genes stay and
    only the sentence is re-worded.
    """
    from panoscope.agent import annotation, config as cfg
    from panoscope.agent import memory
    from panoscope.ui import data_access as da

    before = {v.cluster: v for v in da.all_verdicts()}["c2"]
    genes = [n.gene for n in before.offpanel_notes]
    assert genes, "c2 must carry off-panel canonical markers to begin with"
    assert annotation.offpanel_canonical("CAF") == (), (
        "this test needs a call the annotation does NOT name; pick another"
    )

    memory.create_note(
        claim="c2 is CAF here", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
    )

    after = {v.cluster: v for v in da.composed_verdicts()}["c2"]
    assert after.cell_type == "CAF", "the override did not reach the verdict"
    assert [n.gene for n in after.offpanel_notes] == genes, (
        f"the override deleted the panel-absence block: {genes} -> "
        f"{[n.gene for n in after.offpanel_notes]}"
    )
    assert all("CAF" in n.message for n in after.offpanel_notes), (
        "the notes still argue against the replaced call"
    )

    section = next(s for s in R.build_report_from_sources().sections
                   if s.cluster == "c2")
    assert [g for g, _ in (section.off_panel or ())] == genes, (
        "the document and the screen disagree about which markers were never measured"
    )


def test_the_cross_cluster_check_does_not_contradict_the_override() -> None:
    """One document, one claim per cluster.

    The overview section follows the override; the "Cross-cluster global check"
    read its calls straight off the annotation, so the SAME exported page said
    c2 is CAF and, a few paragraphs down, listed "Stromal (Stromal)" among the
    compartments and "distinct cell types (Tumor, Stromal, ...)" among the
    calls. The biologist's own call was being overruled in silence, in the
    document that leaves the lab.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.ui import data_access as da

    before = R.global_check_text(da.holistic(), None)
    assert "Stromal" in before, "this test needs c2 to start out called Stromal"

    memory.create_note(
        claim="c2 is CAF here", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", subject_category="Mesenchymal",
        note_type="celltype_override", dataset=cfg.DATASET_ID,
        literature_search=lambda _q: (),
    )

    after = R.global_check_text(da.holistic(), None)
    board_call = {v.cluster: v for v in da.composed_verdicts()}["c2"].cell_type
    assert board_call == "CAF"
    assert "CAF" in after, f"the cross-cluster check never adopted the call: {after}"
    assert "Stromal (Stromal)" not in after, (
        f"the cross-cluster check still groups c2 under the replaced call: {after}"
    )
    assert "Tumor, Stromal," not in after, (
        f"the cross-cluster check still lists the replaced call: {after}"
    )


def test_nothing_in_an_overridden_section_argues_the_replaced_call() -> None:
    """The WHOLE section follows the override, not one paragraph of it.

    Guarding only the celltype note left three other places arguing the call the
    biologist rejected, in the document that leaves the lab:

    * the enrichment paragraph, with four real PMIDs written about Macrophages;
    * "Never measured (1 off panel): CSF1R" -- the canonical MACROPHAGE marker,
      the panel-absence rule reported against a replaced call;
    * "What would settle it: To tell Macrophages from T Cells in c3: Macrophages
      is supported here by LYZ (glm_coef 11.84)...", carrying real statistics.

    Driven through the real note writer, so it exercises the path the app uses.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.ui import data_access as da

    old, new = "Macrophages", "Dendritic"
    memory.create_note(
        claim=f"c3 is {new} here",
        scope="cluster",
        basis="own_validation",
        cluster="c3",
        subject_cell_type=new,
        note_type="celltype_override",
        dataset=cfg.DATASET_ID,
        literature_search=lambda _q: (),   # no lookup; tension stays empty
    )

    rep = R.build_report_from_sources()
    section = next(s for s in rep.sections if s.cluster == "c3")
    assert section.cell_type == new, "the override did not reach the section"

    body = " ".join(str(x) for x in (
        section.biology, section.biology_pmid, section.enrichment,
        section.settle, section.driver_line,
    ))
    assert old not in body, (
        f"the replaced call is still argued in the c3 section: {body[:300]}"
    )
    for pmid in ("39223929", "38024532", "38342630", "38640930"):
        assert pmid not in body, f"a Macrophages PMID ({pmid}) survived the override"
    assert "CSF1R" not in " ".join(g for g, _ in (section.off_panel or ())), (
        "the Macrophage canonical off-panel set followed the Dendritic call"
    )


def test_the_generator_line_describes_the_artifacts_not_the_shell(monkeypatch) -> None:
    """§14 provenance must not be a function of the exporting environment.

    It read ``cfg.PRIMARY_MODEL`` -- itself ``getenv("PANOSCOPE_MODEL",
    "claude-sonnet-4-6")``, a default -- and appended "reachable" whenever
    ANTHROPIC_API_KEY was merely SET. Nothing is called when a report is built, so
    the header named a model that never ran and a reachability nobody tested, from
    a credential in whatever shell ran the export. A deliberately invalid key
    still printed "reachable".
    """
    def _generator() -> str:
        rep = R.build_report_from_sources()
        return next(ln for ln in R._header_meta(rep) if ln.lower().startswith("generator"))

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-definitely-not-a-real-key")
    monkeypatch.setenv("PANOSCOPE_MODEL", "some-model-that-never-ran-v9")
    with_junk = _generator()

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PANOSCOPE_MODEL", raising=False)
    without = _generator()

    assert with_junk == without, (
        f"the provenance header moved with the environment:\n  {with_junk}\n  {without}"
    )
    assert "reachable" not in with_junk, "reachability asserted without reaching anything"
    assert "some-model-that-never-ran-v9" not in with_junk, (
        "a model that never ran is named as the generator"
    )
    # §14 asks for a version and an access date; both come off the manifest.
    assert "pipeline" in with_junk and "built" in with_junk, with_junk


def test_an_overridden_call_does_not_inherit_the_old_calls_cited_biology() -> None:
    """A real PMID under the wrong cell type is worse than no PMID.

    ``celltype_notes`` records which ``cell_type`` each summary was written about,
    and nothing read it. Overriding c3 from Macrophages to Dendritic left the
    macrophage paragraph and its citation under a heading that said Dendritic --
    in the Summary table and in both exported documents. Sections come from
    ``composed_verdicts()``, which applies the override; the note did not.
    """
    from types import SimpleNamespace as NS

    notes = {"c3": {"cell_type": "Macrophages",
                    "summary": "Innate immune phagocytes infiltrating the tumour.",
                    "pmid": "38553005"}}

    # The call still reads Macrophages: the note is its own, and renders.
    same = R._celltype_note_for(notes, NS(cluster="c3", cell_type="Macrophages"))
    assert same.get("pmid") == "38553005"
    # Case and spacing are not a difference.
    assert R._celltype_note_for(notes, NS(cluster="c3", cell_type=" macrophages "))

    # The biologist overrode it: the macrophage biology must NOT follow the heading.
    moved = R._celltype_note_for(notes, NS(cluster="c3", cell_type="Dendritic"))
    assert moved == {}, f"stale biology followed the override: {moved}"

    # A note with no recorded cell_type is not retroactively suspect.
    legacy = {"c3": {"summary": "...", "pmid": "1"}}
    assert R._celltype_note_for(legacy, NS(cluster="c3", cell_type="Dendritic"))


def test_the_header_carries_the_provenance_section_14_asks_for() -> None:
    """ARCHITECTURE §14: "header: study, dataset, platform, frames + hashes,
    reference releases, corpus version + staleness, generator model + version +
    access date, spec version".

    Everything after ``platform`` was absent. Each line is emitted even when the
    answer is *not consulted*, because a document that simply omits the reference
    line reads as though reference data agreed -- which is the collapse the whole
    four-state lattice exists to prevent, arriving in the artifact that leaves the
    building.
    """
    rep = R.ReportModel(
        dataset="d", generated_at="", n_clusters=0, n_flagged=0, panel_size=1,
        sections=(), coherence_notes=(), refinements=(), set_is_coherent=True,
        dataset_notes=(), tissue="t", platform="p",
        frames=(("panel:d", "abc123def456"),), reference_releases=(),
        corpus_status="", generator="", spec_version="0.1",
    )
    header = " ".join(R._header_meta(rep))
    assert "panel:d" in header and "abc123def456"[:12] in header, "frames + hashes"
    # The absent cases must SAY so rather than vanish.
    assert "not consulted" in header, "reference line missing when nothing registered"
    assert "not built" in header, "corpus line missing when nothing built"
    assert "deterministic" in header, "generator line missing when no model ran"
    assert "spec version 0.1" in header


def test_the_enrichment_recheck_flag_reaches_the_document() -> None:
    """The screen said re-check this; the document said nothing.

    ``verify`` on the enrichment workflow is deterministic and per workflow, and
    it is True on 7 of the demo's 9 clusters. The Pathways header carried it and
    the exported .docx/PDF did not, so the file that leaves the lab was quieter
    than the screen it came from.
    """
    from panoscope.agent import config as cfg
    from panoscope.ui import data_access as _da

    flagged = {c for c in cfg.CLUSTER_ORDER
               if getattr(_da.enrichment_for(c), "verify", False)}
    assert flagged, "no cluster carries an enrichment re-check flag to test"

    rep = R.build_report_from_sources()
    for section in rep.sections:
        census = str(getattr(section, "enrichment_census", "") or "")
        if section.cluster in flagged:
            assert "re-check" in census.lower(), (
                f"{section.cluster} is flagged by the enrichment rubric and the "
                f"export says nothing: {census!r}")
        else:
            assert "re-check" not in census.lower(), (
                f"{section.cluster} is not flagged yet the export says so: {census!r}")


def test_a_pmid_is_never_split_across_a_pdf_line_break() -> None:
    """A citation broken by the wrap reads as a different, wrong identifier.

    ``FPDF.write`` wraps wherever the margin falls, so a PMID starting near the
    right edge is split: the reader sees "PMID:374797" ending one line and the
    remaining digits beginning the next, and only the first half carries the
    link. Measured over a sweep of line lengths, 5 of 40 tokens started with too
    little room to fit.
    """
    from fpdf import FPDF

    class Probe(FPDF):
        def __init__(self) -> None:
            super().__init__()
            self.cramped = 0
            self.tokens = 0

        def write(self, h, txt="", link=""):  # noqa: A003 - matching FPDF's API
            if txt.startswith("PMID:"):
                self.tokens += 1
                room = self.w - self.r_margin - self.get_x()
                if self.get_string_width(txt) > room + 0.01:
                    self.cramped += 1
            return super().write(h, txt, link)

    pdf = Probe()
    pdf.add_page()
    pdf.set_font("Helvetica", size=9)
    filler = ("Interferon gamma response drives classical macrophage activation "
              "and antigen presentation and ") * 2
    for n in range(40):
        R._pdf_write_linked(pdf, filler[:60 + n * 3] + " PMID:33352494 tail", 5)

    assert pdf.tokens == 40, pdf.tokens
    assert pdf.cramped == 0, (
        f"{pdf.cramped} of {pdf.tokens} citations start with too little room and "
        f"are split across the wrap")


def test_an_excluded_unit_says_so_in_the_document() -> None:
    """A call the biologist withdrew was exported as a confirmed one.

    ``composed_verdicts`` has carried ``exclude`` all along -- the Summary board
    reads it -- and the report builder never did, so a cluster the biologist had
    marked as an artifact kept a full section asserting its call, with nothing
    saying it had been set aside.

    The section is NOT dropped. A unit that silently vanishes from the document
    cannot be argued with, and the measurement is part of the record; the
    withdrawal is stated instead, before anything that describes the unit.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.ui import data_access as _da

    before = R.build_report_from_sources()
    assert not any(s.excluded for s in before.sections), "nothing starts excluded"

    memory.create_note(
        claim="c5 is an artifact, exclude it", scope="cluster",
        basis="own_validation", cluster="c5", note_type="exclude",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
    )
    assert {v.cluster: v for v in _da.composed_verdicts()}["c5"].exclude is True

    rep = R.build_report_from_sources()
    c5 = next(s for s in rep.sections if s.cluster == "c5")
    assert c5.excluded is True
    assert next(s for s in rep.sections if s.cluster == "c1").excluded is False

    lines = R._section_lines(rep)
    heads = [t for style, t in lines if style == "h2" and t.startswith("c5 ")]
    assert heads and "EXCLUDED" in heads[0], heads
    assert any("Excluded by the biologist" in t for _s, t in lines)

    # ...and no OTHER cluster picked up the banner.
    other_heads = [t for style, t in lines
                   if style == "h2" and not t.startswith("c5 ")]
    assert other_heads and not any("EXCLUDED" in t for t in other_heads)
