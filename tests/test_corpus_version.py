"""Corpus rows left after the journeys (corpus requirements A12-A22, B5-B7; SKILL
corpus-building; ARCH §8.2, §18): the build's version stamp, what the literature may
never touch, back-off, and strict gaps for what is not built.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import io
import json
import urllib.error
from datetime import date
from pathlib import Path

import pytest

from panoscope.corpus import store as S
from panoscope.reference.sources import SourceUnavailable
from tests.conftest import DEMO_ID, clear_all_caches

REPO = Path(__file__).resolve().parents[1]
TODAY = date(2026, 11, 9)


# --------------------------------------------------------------------------- #
# ARCH §18 / corpus A13: a rate limit backs off and retries, it does not abort
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-18-01")
def test_a_429_backs_off_and_retries_and_a_404_does_not(monkeypatch) -> None:
    from panoscope.corpus import vocab

    sleeps: list[float] = []
    answers: list = [urllib.error.HTTPError("u", 429, "Too Many Requests", {}, None),
                     io.BytesIO(b'{"hitCount": 3}')]

    def urlopen(req, timeout):
        a = answers.pop(0)
        if isinstance(a, Exception):
            raise a
        return a

    monkeypatch.setattr(vocab.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(vocab.time, "sleep", sleeps.append)
    assert vocab._get("https://example.org/search", {"q": "ERBB2"}) == {"hitCount": 3}
    assert answers == [] and 1 in sleeps            # 2**0 s back-off, then the retry

    answers[:] = [urllib.error.HTTPError("u", 404, "Not Found", {}, None),
                  io.BytesIO(b"{}")]
    with pytest.raises(SourceUnavailable, match="404"):
        vocab._get("https://example.org/search", {"q": "ERBB2"})
    assert len(answers) == 1                        # a 404 is an answer, not retried


# --------------------------------------------------------------------------- #
# corpus-building SKILL: a refused query is split, and every part logged
# --------------------------------------------------------------------------- #
@pytest.mark.req("SKC-REF-01", gap=True)
def test_a_query_refused_as_too_long_is_split_alias_by_alias(monkeypatch, tmp_path) -> None:
    from panoscope.corpus import channels

    def search(q, *, pubmed, cap, probe):
        if " OR " in q:
            raise SourceUnavailable("HTTP Error 414: Request-URI Too Long")
        return 0, []

    monkeypatch.setattr(channels, "_search", search)
    run = channels._Run(S.Corpus(), tmp_path / "corpus.sqlite", 100, TODAY.isoformat())
    last = run.ladder(['("ERBB2" OR "HER2") AND breast'], "C1", "ERBB2", pubmed=False)
    assert last.status is not S.QueryStatus.FAILED, [q.query for q in run.corpus.queries]
    assert len(run.corpus.queries) >= 3                 # the refusal and both halves


@pytest.mark.req("SKC-WIN-01", gap=True)
def test_the_window_widens_one_rung_at_a_time_and_a_saturated_key_reads_thin() -> None:
    from panoscope.corpus import window

    assert "SATURATED" in S.QueryStatus.__members__
    assert callable(getattr(window, "widen", None))


# --------------------------------------------------------------------------- #
# corpus A14 / A22: what a record and a version carry
# --------------------------------------------------------------------------- #
@pytest.mark.req("CORP-A14")
def test_a_source_record_carries_its_display_fields() -> None:
    rec = S.SourceRecord(uri="pmid:101", title="ERBB2 in breast tumours",
                         authors="Smith J, Doe A", year=2020, journal="J Test",
                         abstract="ERBB2 marks tumour cells.", via="Europe PMC",
                         retrieved="2026-11-09", publication_type=("Retracted Publication",),
                         licence="CC-BY")
    assert (rec.title, rec.authors, rec.year, rec.journal, rec.via, rec.retrieved,
            rec.licence) == ("ERBB2 in breast tumours", "Smith J, Doe A", 2020, "J Test",
                             "Europe PMC", "2026-11-09", "CC-BY")
    assert rec.is_real and rec.retracted and not rec.redistributable   # abstract: store only
    assert not dataclasses.replace(rec, via="").is_real
    assert not dataclasses.replace(rec, publication_type=("Journal Article",)).retracted
    for part in ("Smith J, Doe A", "ERBB2 in breast tumours", "J Test", "2020", "pmid:101"):
        assert part in rec.citation


@pytest.mark.req("CORP-A14", gap=True)
def test_a_source_record_has_an_author_year_label() -> None:
    rec = S.SourceRecord(uri="pmid:101", title="t", authors="Smith J, Doe A", year=2020,
                         via="Europe PMC")
    labels = [getattr(rec, n) for n in ("label", "author_year") if hasattr(rec, n)]
    assert "Smith 2020" in labels, labels


@pytest.mark.req("CORP-A22")
def test_a_corpus_version_records_its_retrieval_and_no_embedding() -> None:
    v = S.CorpusVersion(DEMO_ID, (), (), "stub", "h")
    assert (v.retrieval, v.embedding) == ("BM25+aliases", "none")
    # Both are part of what the version IS: changing either is a different version.
    assert dataclasses.replace(v, embedding="dense").content_hash != v.content_hash
    assert dataclasses.replace(v, retrieval="BM25").content_hash != v.content_hash


# --------------------------------------------------------------------------- #
# corpus A17: the research background is a hash, never text
# --------------------------------------------------------------------------- #
_SECRET = "unpublished: our FAP+ CAF subset predicts relapse"


@pytest.mark.req("CORP-A17")
def test_the_research_background_is_kept_only_as_a_hash(app_tree, tmp_path) -> None:
    from panoscope.corpus import build as B
    from panoscope.data.study import Study

    mpath = app_tree / DEMO_ID / "manifest.json"
    m = json.loads(mpath.read_text())
    bg = {"research_background": {"value": _SECRET, "provenance": "stated"}}
    mpath.write_text(json.dumps({**m, "background": bg}))
    clear_all_caches()
    study = Study.open(DEMO_ID)
    assert study.manifest["background"] == bg                  # the build sees it
    v = B.version_for(study, B.context_for(study), S.Corpus(), "stub", "", TODAY)
    assert v.background_hash == hashlib.sha256(
        json.dumps(bg, sort_keys=True).encode()).hexdigest()[:16]
    assert _SECRET not in json.dumps(dataclasses.asdict(v))
    p = Path(S.save(S.Corpus(version=v), tmp_path / "corpus.sqlite"))
    assert _SECRET.encode() not in p.read_bytes()
    assert v.background_hash.encode() in p.read_bytes()


# --------------------------------------------------------------------------- #
# corpus A19 / plan P2: evals pin the reference release (and the corpus version)
# --------------------------------------------------------------------------- #
_HARNESSES = ("panel_absence", "refusal", "sycophancy")


def _assigned(name: str) -> dict[str, ast.expr]:
    tree = ast.parse((REPO / "evals" / name / "harness.py").read_text())
    return {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name)}


@pytest.mark.req("PLAN-P2C-01", "CORP-A19")
def test_each_eval_pins_the_reference_release_and_refuses_a_changed_one() -> None:
    from panoscope.agent import config as cfg
    from panoscope.reference import view

    now = dict(view.for_study(cfg.current_study()).releases)
    for h in _HARNESSES:
        pin = ast.literal_eval(_assigned(h)["REFERENCE_PIN"])
        assert pin == now, h                    # the pin matches the release in use
        view.check_pinned(cfg.current_study(), pin)
        moved = {**pin, next(iter(pin)): "0" * 12}
        with pytest.raises(SystemExit, match="reference release changed"):
            view.check_pinned(cfg.current_study(), moved)


@pytest.mark.req("CORP-A19", gap=True)
def test_each_eval_pins_the_corpus_version() -> None:
    missing = [h for h in _HARNESSES if "CORPUS_PIN" not in _assigned(h)]
    assert missing == []


# --------------------------------------------------------------------------- #
# corpus A12 / B6 / B7: version on the tension, tissue_match at read time
# --------------------------------------------------------------------------- #
@pytest.mark.req("CORP-A12", "CORP-B6", gap=True)
def test_a_tension_stores_the_corpus_version_it_was_read_against() -> None:
    from panoscope.agent.types import Tension

    assert "corpus_version" in {f.name for f in dataclasses.fields(Tension)}


@pytest.mark.req("CORP-B7", gap=True)
def test_tissue_match_is_computed_at_read_time_for_ordering() -> None:
    from panoscope.corpus import retrieve

    names = {f.name for f in dataclasses.fields(S.Statement)} | \
        {f.name for f in dataclasses.fields(retrieve.Hit)}
    assert "tissue_match" in names


@pytest.mark.req("ARCH-8.2-02", gap=True)
def test_a_cited_paper_shows_its_tissue_and_modality_mismatch_beside_the_citation() -> None:
    """ARCH §8.2: literature is not validation. A bulk-RNA lung paper cited for a Xenium
    breast study must say so where it is cited."""
    from panoscope.corpus import cards

    s = S.Statement(uri="pmid:101", gene="ERBB2", cell_type="Tumor", tissue="lung",
                    predicate="MARKS", stance="supports", span="ERBB2 marks tumour cells",
                    start=0, end=24, section="abstract", rung="W3", analyte="RNA",
                    resolution="BULK", modality_match="BULK")
    line = cards.cite(s)
    assert "pmid:101" in line
    assert "bulk" in line.lower() and "lung" in line.lower(), line


# --------------------------------------------------------------------------- #
# corpus B5: an extracted statement never changes what the panel can settle
# --------------------------------------------------------------------------- #
@pytest.mark.req("CORP-B5")
def test_imported_literature_never_changes_the_discriminating_genes(
        isolated, monkeypatch, tmp_path) -> None:
    """c3 vs T cells, before and after importing two papers on CD4 in T cells (one
    agrees, one dissents): settleable_on_panel and every marker bucket are unchanged."""
    from panoscope.agent.discriminate import discriminate
    from panoscope.corpus import exchange
    from panoscope.data.study import Study
    from tests.journeys.flow import literature

    before = discriminate("c3", "T_Cells")
    study = Study.open(DEMO_ID)
    path = literature(isolated, monkeypatch, study, tmp_path, [
        ("pmid:101", "CD4", "T_Cells", "agree"),
        ("pmcid:PMC7654321", "CD4", "T_Cells", "dissent")])
    corpus, _ = exchange.import_file(study, path, today=TODAY)
    assert {(s.gene, s.cell_type) for s in corpus.statements} == {("CD4", "T_Cells")}
    clear_all_caches()
    assert discriminate("c3", "T_Cells") == before
