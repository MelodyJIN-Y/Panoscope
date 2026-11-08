"""The corpus import from a biologist's own LLM client: schema fails closed by line, ids
resolved by code (a fabricated one rejected), spans verified verbatim, client facets
ignored and recomputed, versions kept. Synthetic rows and a stub resolver; no network."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date

import pytest

from panoscope.corpus import build as B
from panoscope.corpus import exchange as X
from panoscope.corpus.read import RELATIONS, SETTING_FIELDS
from panoscope.corpus.retrieve import key_state
from panoscope.corpus.store import (
    DISPOSITIONS, Corpus, CorpusUnavailable, CorpusVersion, SourceRecord, corpus_path, load, save,
)
from panoscope.data.study import Study
from tests.conftest import DEMO_ID
from tests.test_corpus_read import ABSTRACT, FULL, _ctx

TODAY = date(2026, 11, 8)
REC = SourceRecord(uri="pmid:1", title="POSTN in breast", abstract=ABSTRACT, full_text=FULL,
                   via="Europe PMC", year=2026)
SPAN = "POSTN marks fibroblasts in human breast cancer tissue."


@pytest.fixture
def study(monkeypatch):
    monkeypatch.setattr(B, "context_for", lambda s: _ctx(study=s.id))
    monkeypatch.setattr(B, "scope", lambda s, types: ("POSTN",))
    monkeypatch.setattr(X.vocab, "hgnc", lambda g: None)
    return Study.open(DEMO_ID)


def _resolver(pid, retrieved):
    return REC if pid == "pmid:1" else None          # every other id is fabricated


def _file(tmp_path, study, *rows, brief=None, name="c.jsonl"):
    head = {"kind": "header", "format_version": 1, "study_id": study.id,
            "brief_hash": brief or B.brief_hash(study), "client": "Claude Desktop",
            "provider": "Anthropic", "model": "claude-opus-5-5", "run_date": "2026-11-07"}
    rows = rows or (
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": SPAN, "section": "abstract"},
        {"kind": "query", "query": '"POSTN" AND breast', "source": "PubMed", "gene": "POSTN",
         "cell_type": "Fibroblasts", "total": 1, "retrieved": 1, "status": "OK"})
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in (head, *rows)) + "\n", encoding="utf-8")
    return p


def _import(study, path):
    return X.import_file(study, path, today=TODAY, resolver=_resolver)


@pytest.mark.req("CORP-A21", "IMPL-6.1-09")
def test_an_import_verifies_and_stamps_a_version_built_in_the_users_client(study, tmp_path):
    corpus, receipt = _import(study, _file(tmp_path, study))
    assert [s.span for s in corpus.statements] == [SPAN]
    assert corpus.statements[0].stance == "supports" and corpus.ledger[0].id == "pmid:1"
    v = corpus.version
    assert "claude-opus-5-5 (Anthropic) in Claude Desktop, built in user client" in v.generator
    assert (v.built_at, v.study) == ("2026-11-07", DEMO_ID)
    assert receipt[0] == "accepted 1 statement(s), rejected 0" and load(corpus_path(DEMO_ID)) == corpus


@pytest.mark.req("IMPL-6.1-02")
def test_a_fabricated_pmid_is_rejected_and_never_kept(study, tmp_path):
    corpus, receipt = _import(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:999", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:999", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": SPAN, "section": "abstract"}))
    assert not corpus.records and not corpus.statements and not corpus.ledger
    assert "  pmid:999: no record has this id" in receipt
    assert "  rejected pmid:999 POSTN: paper unresolved" in receipt


@pytest.mark.req("IMPL-6.1-11")
def test_the_receipt_states_every_section_for_a_mixed_import(study, tmp_path):
    """IMPL §6.1: the receipt lists accepted and rejected statements with reasons,
    unresolved ids and the rung distribution. One file with one good statement, one
    on a fabricated id and one altered span must show all of it."""
    good = {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
            "predicate": "MARKS", "span": SPAN, "section": "abstract"}
    corpus, receipt = _import(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "paper", "id": "pmid:999", "channel": "C1", "disposition": "READ:1:0"},
        good,
        {**good, "paper": "pmid:999"},
        {**good, "span": SPAN.replace("marks", "defines")}))
    assert receipt[0] == "accepted 1 statement(s), rejected 2"
    rejected = [ln for ln in receipt if ln.startswith("  rejected pmid:")]
    assert rejected[0] == "  rejected pmid:999 POSTN: paper unresolved"
    assert rejected[1].startswith("  rejected pmid:1 POSTN: ") and "not verbatim" in rejected[1]
    i = receipt.index("unresolved id(s): 1")
    assert receipt[i + 1] == "  pmid:999: no record has this id"
    # The distribution counts what was accepted, one rung per accepted statement.
    [s] = corpus.statements
    assert receipt[i + 2] == f"rungs: {s.rung or 'beyond the ladder'} 1"


@pytest.mark.req("IMPL-9-03")
def test_a_statement_naming_the_gene_only_by_its_hgnc_alias_is_retrieved(
        study, tmp_path, monkeypatch):
    """IMPL §9 alias retrieval. The paper says OSF2, never POSTN; the HGNC table (a
    fixture at the network boundary) lists OSF2 as POSTN's alias. The span is bound
    to POSTN and the POSTN x Fibroblasts card finds it."""
    span = "OSF2 marks fibroblasts in human breast cancer tissue."
    assert "postn" not in span.casefold()
    rec = replace(REC, abstract=span, full_text="")
    monkeypatch.setattr(X.vocab, "hgnc", lambda g: {"symbol": "POSTN",
                                                    "alias_symbol": ["OSF2"]})
    corpus, receipt = X.import_file(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": span, "section": "abstract"},
        {"kind": "query", "query": '"POSTN" AND breast', "source": "PubMed", "gene": "POSTN",
         "cell_type": "Fibroblasts", "total": 1, "retrieved": 1, "status": "OK"}),
        today=TODAY, resolver=lambda pid, r: rec if pid == "pmid:1" else None)
    assert receipt[0] == "accepted 1 statement(s), rejected 0", receipt
    got = key_state(corpus, "POSTN", "Fibroblasts")
    assert [(s.gene, s.span) for s in got.statements] == [("POSTN", span)]


@pytest.mark.req("IMPL-6.1-03")
def test_a_retracted_paper_is_closed_whatever_the_client_said(study, tmp_path, monkeypatch):
    gone = replace(REC, publication_type=("Retracted Publication",))
    monkeypatch.setattr(X, "resolve_id", lambda pid, r: gone)
    corpus, receipt = X.import_file(study, _file(tmp_path, study), today=TODAY)
    assert not corpus.statements and corpus.ledger[0].disposition == "RETRACTED"
    assert "  rejected pmid:1 POSTN: paper retracted" in receipt


@pytest.mark.req("IMPL-6.1-05")
def test_an_altered_span_is_rejected(study, tmp_path):
    corpus, receipt = _import(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": SPAN.replace("marks", "defines"), "section": "abstract"}))
    assert not corpus.statements
    assert any("not verbatim in abstract" in line for line in receipt)


@pytest.mark.req("IMPL-6.1-06")
def test_a_client_supplied_rung_and_stance_are_ignored_and_recomputed(study, tmp_path):
    paper = {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"}
    claimed = {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "",
               "predicate": "NOT_EXPRESSED_IN", "span": "Basal cells lacked POSTN expression.",
               "section": "abstract", "rung": "W0", "stance": "supports",
               "species": "NON_HUMAN"}
    corpus, receipt = _import(study, _file(tmp_path, study, paper, claimed))
    s, = corpus.statements
    assert s.stance == "argues_against" and s.rung != "W0" and s.species == "HUMAN_STATED"
    assert f"rungs: {s.rung or 'beyond the ladder'} 1" in receipt


def test_an_incomplete_client_search_reads_incomplete_never_thin(study, tmp_path):
    corpus, _ = _import(study, _file(
        tmp_path, study,
        {"kind": "query", "query": "q", "source": "PubMed", "gene": "ACTA2",
         "cell_type": "Fibroblasts", "total": 9, "retrieved": 3, "status": "OK"}))
    assert corpus.queries[-1].status.value == "truncated"       # the client's OK recomputed
    assert "incomplete search" in key_state(corpus, "ACTA2", "Fibroblasts").reads_as


def _logged(ch, cell="Fibroblasts", total=1, retrieved=1):
    return {"kind": "query", "query": f"{ch} POSTN", "source": "Europe PMC", "channel": ch,
            "gene": "POSTN", "cell_type": "" if ch in ("vocab", "C0") else cell,
            "total": total, "retrieved": retrieved, "status": "OK"}


FULL_LOG = ("vocab", "C1", "C3", "C4")


@pytest.mark.parametrize("drop, short, gap", [
    (None, None, None),
    ("C3", None, "client:C3 not run"),
    ("vocab", None, "vocab:hgnc not run"),
    (None, "C1", "client:C1:Europe PMC truncated"),     # OK with retrieved < total
])
@pytest.mark.req("IMPL-6.1-07")
def test_a_key_reads_complete_only_when_every_brief_line_is_logged_ok(
        study, tmp_path, drop, short, gap):
    rows = [_logged(c, total=2 if c == short else 1) for c in FULL_LOG if c != drop]
    corpus, _ = _import(study, _file(tmp_path, study, *rows))
    k = key_state(corpus, "POSTN", "Fibroblasts")
    assert k.gaps == ((gap,) if gap else ())
    # complete with nothing found reads what the in-app path reads: never thin
    assert "incomplete search" in k.reads_as and not re.search(r"\bthin\b", k.reads_as)
    assert ("no saturated stop yet" in k.reads_as) is (gap is None)


def test_a_confirmed_release_paper_requires_a_c0_line(study, tmp_path, monkeypatch):
    monkeypatch.setattr(B, "context_for", lambda s: _ctx(
        study=s.id, release_status="confirmed", release_pmid="123"))
    rows = [_logged(c) for c in FULL_LOG]
    corpus, _ = _import(study, _file(tmp_path, study, *rows))
    assert key_state(corpus, "POSTN", "Fibroblasts").gaps == ("client:C0 degraded",)
    corpus, _ = _import(study, _file(tmp_path, study, *rows, _logged("C0")))
    assert key_state(corpus, "POSTN", "Fibroblasts").gaps == ()


@pytest.mark.req("IMPL-6.1-10")
def test_earlier_versions_are_kept_never_overwritten(study, tmp_path):
    built = Corpus(version=CorpusVersion(DEMO_ID, (), (), "in-app", "h"))
    save(built, corpus_path(DEMO_ID))
    first, _ = _import(study, _file(tmp_path, study))
    second, _ = _import(study, _file(tmp_path, study, name="d.jsonl", brief="0" * 16))
    kept = {p.stem for p in (corpus_path(DEMO_ID).parent / "versions").iterdir()}
    assert kept == {built.version.content_hash, first.version.content_hash,
                    second.version.content_hash} and len(kept) == 3
    assert load(corpus_path(DEMO_ID)).version == second.version


def test_two_imports_that_differ_only_in_statements_are_two_versions(study, tmp_path):
    first, _ = _import(study, _file(tmp_path, study))
    paper, _, query = json.loads("[" + _file(tmp_path, study).read_text().replace(
        "\n", ",").rstrip(",") + "]")[1:]
    second, _ = _import(study, _file(tmp_path, study, paper, query, name="d.jsonl"))
    assert first.version.query_log_hash == second.version.query_log_hash
    assert first.version.content_hash != second.version.content_hash


def test_a_setting_line_is_checked_verbatim_and_feeds_the_rung(study, tmp_path):
    paper = {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"}
    st = {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Myoepithelial",
          "predicate": "NOT_EXPRESSED_IN", "span": "Basal cells lacked POSTN expression.",
          "section": "abstract"}
    tissue = {"kind": "setting", "paper": "pmid:1", "field": "tissue",
              "span": "breast tumour samples"}
    bare, _ = _import(study, _file(tmp_path, study, paper, st))
    corpus, receipt = _import(study, _file(tmp_path, study, paper, st, tissue,
                                           {**tissue, "field": "species", "span": "mice"},
                                           name="s.jsonl"))
    assert [(x.field, x.quote) for x in corpus.settings] == [("tissue", "breast tumour samples")]
    assert "settings: accepted 1, rejected 1" in receipt
    assert (bare.statements[0].rung, corpus.statements[0].rung) == ("W3", "W0*")


@pytest.mark.req("IMPL-6.1-04")
def test_full_text_is_fetched_only_for_cited_papers_and_a_failed_fetch_is_named(
        study, tmp_path, monkeypatch):
    oa = {u: replace(REC, uri=u, full_text="", open_access=True, pmcid="PMC1" + u[-1])
          for u in ("pmid:1", "pmid:2")}
    fetched = []
    monkeypatch.setattr(X, "full_text", lambda r: fetched.append(r.uri) or "")
    corpus, receipt = X.import_file(study, _file(
        tmp_path, study,
        *({"kind": "paper", "id": u, "channel": "C1", "disposition": "READ:1:0"} for u in oa),
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": "POSTN was a specific marker of fibroblasts",
         "section": "full_text"}), today=TODAY, resolver=lambda pid, d: oa.get(pid))
    assert fetched == ["pmid:1"]
    assert any("full text unavailable" in line for line in receipt)
    assert [q.status.value for q in corpus.queries if q.channel == "fulltext"] == ["failed"]


@pytest.mark.req("IMPL-6.1-08")
def test_a_stale_brief_warns_and_is_recorded(study, tmp_path):
    corpus, receipt = _import(study, _file(tmp_path, study, brief="0" * 16))
    assert any(line.startswith("WARNING: stale brief 0000") for line in receipt)
    assert (corpus.queries[0].channel, corpus.queries[0].status.value) == (
        "import:brief", "degraded")


@pytest.mark.req("IMPL-6.1-01")
def test_a_schema_error_fails_closed_by_line_and_writes_nothing(study, tmp_path):
    bad = _file(tmp_path, study,
                {"kind": "paper", "id": "12345", "channel": "C1", "disposition": "READ:1:0"},
                {"kind": "paper", "channel": "C1", "disposition": "READ:1:0"},
                {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "",
                 "predicate": "LOVES", "span": SPAN, "section": "abstract"})
    with pytest.raises(CorpusUnavailable) as err:
        _import(study, bad)
    msg = str(err.value)
    assert "line 2: id:" in msg and "line 4: predicate:" in msg
    assert "line 3: paper: 'id' is a required property" in msg
    assert "line 4: paper 'pmid:1' has no paper line" in msg
    assert not corpus_path(DEMO_ID).exists()
    for wrong in ('"run_date": "2026-11-07"', f'"study_id": "{DEMO_ID}"'):
        good = _file(tmp_path, study, name="g.jsonl")
        good.write_text(good.read_text().replace(wrong, wrong[:-4] + '99"', 1))
        with pytest.raises(CorpusUnavailable, match="run_date|is for"):
            _import(study, good)
    bad.write_text(bad.read_text().split("\n", 1)[1])
    with pytest.raises(CorpusUnavailable, match="line 1: the first line"):
        _import(study, bad)


@pytest.mark.req("SKC-R10-01")
def test_the_schema_vocabularies_match_the_code():
    defs = json.loads(X.SCHEMA.read_text())["$defs"]
    assert set(defs["statement"]["properties"]["predicate"]["enum"]) == set(RELATIONS)
    assert tuple(defs["setting"]["properties"]["field"]["enum"]) == SETTING_FIELDS
    rx = re.compile(defs["paper"]["properties"]["disposition"]["pattern"])
    assert all(rx.match(d + ("x" if d.endswith(":") else "")) for d in DISPOSITIONS)


def test_resolve_keeps_only_the_record_the_id_names(monkeypatch):
    hit = {"pmid": "2", "title": "Another paper", "pubYear": "2020"}
    monkeypatch.setattr(X, "_search", lambda q, **k: (1, [hit]))
    assert X.resolve_id("pmid:1", "2026-11-08") is None
    assert X.resolve_id("pmid:2", "2026-11-08").title == "Another paper"


def test_a_pmc_only_record_is_cited_as_pmcid_and_a_fabricated_pmcid_is_rejected(monkeypatch):
    hits = [{"pmcid": "PMC123", "title": "OA only", "pubYear": "2024"},
            {"pmcid": "PMC456", "pmid": "9", "title": "Indexed", "pubYear": "2024"}]
    monkeypatch.setattr(X, "_search", lambda q, **k: (2, hits))
    assert X.resolve_id("pmcid:PMC123", "2026-11-08").uri == "pmcid:PMC123"
    assert X.resolve_id("pmcid:PMC456", "2026-11-08").uri == "pmid:9"    # a PMID wins
    assert X.resolve_id("pmcid:PMC999", "2026-11-08") is None


def test_the_cli_imports_and_prints_the_receipt(study, tmp_path, monkeypatch, capsys):
    from panoscope import cli

    monkeypatch.setattr(X, "resolve_id", _resolver)
    monkeypatch.setattr(B, "refresh_notes", lambda s: None)         # never data/
    cli.main(["corpus", "import", "--study", DEMO_ID, str(_file(tmp_path, study))])
    out = capsys.readouterr().out
    assert "accepted 1 statement(s)" in out and "built 2026-11-07" in out


@pytest.mark.req("IMPL-6.1-05", "PLAN-P3.3-01")
def test_an_imported_span_that_never_names_the_gene_is_rejected(study, tmp_path):
    other = "VIM marks fibroblasts in human breast cancer tissue."
    rec = replace(REC, abstract=f"{ABSTRACT} {other}")
    corpus, receipt = X.import_file(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": other, "section": "abstract"}),
        today=TODAY, resolver=lambda pid, r: rec if pid == "pmid:1" else None)
    assert not corpus.statements
    assert any("does not name POSTN" in line for line in receipt), receipt


@pytest.mark.req("IMPL-6.1-02", "CLAUDE-HR-03")
def test_a_doi_whose_lookup_returns_a_different_paper_is_rejected(monkeypatch):
    """Europe PMC's first hit for a DOI query is not the DOI's paper by default."""
    hits = [{"doi": "10.1000/other", "pmid": "7", "title": "Wrong paper", "pubYear": "2020"},
            {"doi": "10.1000/ABC", "pmid": "8", "title": "Right paper", "pubYear": "2021"}]
    monkeypatch.setattr(X, "_search", lambda q, **k: (2, hits))
    assert X.resolve_id("doi:10.1000/abc", "2026-11-08").title == "Right paper"
    monkeypatch.setattr(X, "_search", lambda q, **k: (1, hits[:1]))
    assert X.resolve_id("doi:10.1000/abc", "2026-11-08") is None


@pytest.mark.req("IMPL-6.1-01")
def test_a_file_with_two_header_lines_is_refused_by_line(study, tmp_path):
    p = _file(tmp_path, study)
    head = p.read_text().splitlines()[0]
    p.write_text(p.read_text() + head + "\n")
    with pytest.raises(CorpusUnavailable, match="line 1: the first line must be the file's only header"):
        _import(study, p)
    assert not corpus_path(DEMO_ID).exists()
