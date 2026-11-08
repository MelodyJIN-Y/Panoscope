"""Notes stage: the confident-floor guard on the deterministic fallback.

The adversarial review confirmed a grounding-floor violation: when the model's
prose is unusable and the note falls back to a deterministic clause, the note
must NOT keep the model's top search citation. Stapling a real PMID onto a
fallback clause the paper does not support is a mismatched citation — worse than
none, per the project's core principle. ``_resolve_note`` is the single point
that both note-builders (cell-type and per-gene) route through; these tests pin
its behavior so the violation cannot regress.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from panoscope.pipeline import run as pipeline_run
from panoscope.pipeline.stages.notes import (
    _balance_parens,
    _fallback_summary,
    _looks_meta,
    _resolve_note,
    _shorten,
)


def _cite(pmid: str = "12345"):
    return SimpleNamespace(pmid=pmid, title="A real paper", authors="Doe J", year=2021)


def test_first_person_process_prose_is_meta_not_a_finding():
    """The model narrating what it is about to do is never a note.

    All five of these were written into the committed pathway notes and two
    reached the exported document, where "Let me try one more targeted search."
    appeared as the cited biology of an enriched program with a PMID attached --
    on a sentence that says the papers do not support the claim.
    """
    for prose in (
        "I'll look up the literature on hypoxia in myoepithelial cells.",
        "Let me try one more targeted search.",
        "None of the resolved papers directly support a claim about this program.",
        "The fetched paper is about ROS, not directly about allograft rejection.",
        "I will search more specifically.",
    ):
        assert _looks_meta(prose), f"process prose accepted as a finding: {prose!r}"

    # ...and the guard stays narrow: a real negative finding is not meta.
    for finding in (
        "None of the driving genes are canonical myoepithelial markers.",
        "The Angiogenesis signal here is driven by stromal genes, not endothelial.",
    ):
        assert not _looks_meta(finding), f"a true finding was rejected: {finding!r}"


def test_no_committed_note_reads_as_process_prose():
    """The artifact, not just the guard. ``_looks_meta`` existed and these got in.

    The notes stage is a LIVE pass; it cannot be re-run offline, so a note that
    slips through is frozen into a committed artifact and exported from there.
    This reads the files a reviewer actually gets.
    """
    import json

    from panoscope.agent import config as cfg
    from panoscope.data.study import Study

    study = Study.open(cfg.BUNDLED_DEMO_ID)
    bad: list[str] = []

    pn = study.interp / "pathway_notes.json"
    if pn.is_file():
        for cl, sets in json.loads(pn.read_text()).items():
            if not isinstance(sets, dict):
                continue
            for s, n in sets.items():
                if isinstance(n, dict) and _looks_meta(str(n.get("summary") or "")):
                    bad.append(f"pathway_notes {cl}/{s}: {n.get('summary')!r}")

    cn = study.interp / "celltype_notes.json"
    if cn.is_file():
        for cl, n in json.loads(cn.read_text()).items():
            if isinstance(n, dict) and _looks_meta(str(n.get("summary") or "")):
                bad.append(f"celltype_notes {cl}: {n.get('summary')!r}")

    assert not bad, (
        "committed notes carry model meta-commentary, which the export renders as "
        "cited biology:\n  " + "\n  ".join(bad)
    )


def test_keeps_citation_when_model_text_survives():
    summary, pmid, citation, used_fb = _resolve_note(
        "KRT7 is a luminal epithelial keratin marking glandular identity.",
        _cite("33416175"),
        "fallback clause",
    )
    assert not used_fb
    assert summary.startswith("KRT7")
    assert pmid == "33416175"
    assert citation["pmid"] == "33416175"


def test_drops_citation_on_meta_fallback():
    # Meta-commentary about the citation choice -> deterministic fallback -> the
    # real-but-unrelated PMID must be dropped, not stapled to the fallback clause.
    fb = _fallback_summary("Stromal", ["LUM", "POSTN", "DCN"])
    summary, pmid, citation, used_fb = _resolve_note(
        "Neither paper is a strong fit; this citation resolves cleanly but is off-topic.",
        _cite("999"),
        fb,
    )
    assert used_fb
    assert summary == fb
    assert pmid is None
    assert citation is None


def test_drops_citation_on_empty_text():
    summary, pmid, citation, used_fb = _resolve_note("", _cite("111"), "Fallback.")
    assert used_fb and pmid is None and citation is None and summary == "Fallback."


def test_no_model_citation_is_fine():
    summary, pmid, citation, used_fb = _resolve_note(
        "EPCAM is an epithelial adhesion molecule marking carcinoma identity.",
        None,
        "fallback",
    )
    assert not used_fb and pmid is None and citation is None and summary.startswith("EPCAM")


def test_shorten_drops_dangling_paren():
    # The word cap can slice mid-parenthetical; the summary must not end on '('.
    out = _shorten(
        "Tissue-resident immune cells that store and release proteases (CPA3 is a "
        "mast-cell carboxypeptidase highly specific to this lineage)."
    )
    assert "(" not in out and out.endswith(".")
    assert out == "Tissue-resident immune cells that store and release proteases."


def test_balance_parens_leaves_matched_text_untouched():
    assert _balance_parens("expressing LYZ (an antimicrobial enzyme)") == (
        "expressing LYZ (an antimicrobial enzyme)"
    )
    assert _balance_parens("proteases (CPA3") == "proteases"


def test_looks_meta_catches_review_examples():
    assert _looks_meta("Neither paper is a strong fit for stromal identity.")
    assert _looks_meta("The PubMed connector is returning no hits.")
    assert not _looks_meta("LUM is a small leucine-rich proteoglycan in stroma.")


def test_manifest_cell_count_reads_the_built_tree():
    """n_cells in the manifest comes from the tree the run just built.

    Replaces a test for pipeline_run._read_tree_frame, a private workaround that
    existed only because agent.data froze its paths at import -- to the legacy
    location, before collect_viz moved the frames -- causing two confirmed
    first-run manifest bugs. The readers now resolve against the current study at
    call time, so the pipeline scopes itself to the tree it is writing and reads
    through the normal loader. What matters is the invariant, not the helper.
    """
    import json

    from panoscope.agent import config as cfg, data

    manifest = json.loads(
        (cfg.DATA_DIR_PATH / "studies" / cfg.BUNDLED_DEMO_ID / "manifest.json").read_text()
    )
    assert manifest["n_cells"] == len(data.load_cells())


@pytest.mark.req("SKM-O4-01", "SKE-O-01")
def test_a_note_restating_a_number_score_or_q_is_refused():
    """marker Output 4 / enrichment Outputs: no jazzPanda number, score or q restated.
    Such prose falls back, and its citation goes with it."""
    for prose, sentences in (
        ("ERBB2 marks HER2-amplified tumour cells here (glm 21.4).", 1),
        ("ERBB2 is a tumour marker with coefficient 21.40 in this cluster.", 1),
        ("Hypoxia response in tumour cells, q = 0.003. It reflects low oxygen.", 2),
        ("Interferon signalling in T cells with NES: 2 across the set.", 2),
        ("Angiogenesis program; its q-value marks it as strong.", 2),
    ):
        summary, pmid, citation, used_fb = _resolve_note(
            prose, _cite("33416175"), "Fallback.", max_sentences=sentences)
        assert used_fb and summary == "Fallback." and pmid is None and citation is None, prose
    # Narrow: gene names carrying digits are biology, not statistics.
    summary, pmid, _, used_fb = _resolve_note(
        "CD8A and p53 signalling mark cytotoxic T cells.", _cite("33416175"), "Fallback.")
    assert not used_fb and pmid == "33416175" and summary.startswith("CD8A")


@pytest.mark.req("SKM-O4-01", "SKE-O-01")
def test_a_note_never_carries_an_em_or_en_dash():
    for prose in ("KRT7 — a luminal keratin — marks glandular epithelium.",
                  "KRT7 – a luminal keratin – marks glandular epithelium."):
        summary, pmid, _, used_fb = _resolve_note(prose, _cite("1"), "Fallback.")
        assert not used_fb and pmid == "1"
        assert "—" not in summary and "–" not in summary and summary.startswith("KRT7")


@pytest.mark.req("SKE-S3-01")
def test_a_pathway_note_that_reinforces_the_call_falls_back(tmp_path, monkeypatch):
    """enrichment Step 3: concordant never 'reinforces' the call, in a new note (the
    pathway stage, model faked at its boundary) or in one written before the guard
    existed (the read-side ``without_lift``). A marker note is not a program: its
    biology may say a gene reinforces tissue and keep its citation."""
    from panoscope.pipeline.stages import notes as N
    from panoscope.pipeline.stages.notes import without_lift

    prose = "KIT is concordant with mast cell identity, reinforcing the c9 call."
    monkeypatch.setattr(N.agent_loop, "chat", lambda *a, **k: SimpleNamespace(
        text=prose, citations=[_cite("1")], verify=False))
    written = [n for sets in N.run_pathway_notes(root=tmp_path).values() for n in sets.values()]
    assert written, "the demo surfaces programs; nothing was generated"
    for n in written:
        assert "reinforc" not in n["summary"] and n["pmid"] is None and n["citation"] is None
    marker = "POSTN encodes a matrix protein that reinforces connective tissue architecture."
    summary, pmid, _, used_fb = _resolve_note(marker, _cite("1"), "Fallback.")
    assert not used_fb and pmid == "1" and summary == marker
    stored = {"gene_set": "HALLMARK_UV_RESPONSE_DN", "cell_type": "Mast_Cells",
              "leading_edge": ["CAV1", "KIT"], "summary": prose, "pmid": "1",
              "citation": {"pmid": "1"}}
    got = without_lift(stored)
    assert "reinforc" not in got["summary"] and "KIT" in got["summary"]
    assert got["pmid"] is None and got["citation"] is None
    plain = {**stored, "summary": "KIT sustains mast cell survival."}
    assert without_lift(plain) is plain
