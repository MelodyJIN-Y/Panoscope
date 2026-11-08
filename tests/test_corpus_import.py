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
from panoscope.corpus.read import RELATIONS
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


def test_an_import_verifies_and_stamps_a_version_built_in_the_users_client(study, tmp_path):
    corpus, receipt = _import(study, _file(tmp_path, study))
    assert [s.span for s in corpus.statements] == [SPAN]
    assert corpus.statements[0].stance == "supports" and corpus.ledger[0].id == "pmid:1"
    v = corpus.version
    assert "claude-opus-5-5 (Anthropic) in Claude Desktop, built in user client" in v.generator
    assert (v.built_at, v.study) == ("2026-11-07", DEMO_ID)
    assert receipt[0] == "accepted 1 statement(s), rejected 0" and load(corpus_path(DEMO_ID)) == corpus


def test_a_fabricated_pmid_is_rejected_and_never_kept(study, tmp_path):
    corpus, receipt = _import(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:999", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:999", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": SPAN, "section": "abstract"}))
    assert not corpus.records and not corpus.statements and not corpus.ledger
    assert "  pmid:999: no record has this id" in receipt
    assert "  rejected pmid:999 POSTN: paper unresolved" in receipt


def test_a_retracted_paper_is_closed_whatever_the_client_said(study, tmp_path, monkeypatch):
    gone = replace(REC, publication_type=("Retracted Publication",))
    monkeypatch.setattr(X, "resolve_id", lambda pid, r: gone)
    corpus, receipt = X.import_file(study, _file(tmp_path, study), today=TODAY)
    assert not corpus.statements and corpus.ledger[0].disposition == "RETRACTED"
    assert "  rejected pmid:1 POSTN: paper retracted" in receipt


def test_an_altered_span_is_rejected(study, tmp_path):
    corpus, receipt = _import(study, _file(
        tmp_path, study,
        {"kind": "paper", "id": "pmid:1", "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": "pmid:1", "gene": "POSTN", "cell_type": "Fibroblasts",
         "predicate": "MARKS", "span": SPAN.replace("marks", "defines"), "section": "abstract"}))
    assert not corpus.statements
    assert any("not verbatim in abstract" in line for line in receipt)


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


def test_earlier_versions_are_kept_never_overwritten(study, tmp_path):
    built = Corpus(version=CorpusVersion(DEMO_ID, (), (), "in-app", "h"))
    save(built, corpus_path(DEMO_ID))
    first, _ = _import(study, _file(tmp_path, study))
    second, _ = _import(study, _file(tmp_path, study, name="d.jsonl", brief="0" * 16))
    kept = {p.stem for p in (corpus_path(DEMO_ID).parent / "versions").iterdir()}
    assert kept == {built.version.content_hash, first.version.content_hash,
                    second.version.content_hash} and len(kept) == 3
    assert load(corpus_path(DEMO_ID)).version == second.version


def test_a_stale_brief_warns_and_is_recorded(study, tmp_path):
    corpus, receipt = _import(study, _file(tmp_path, study, brief="0" * 16))
    assert any(line.startswith("WARNING: stale brief 0000") for line in receipt)
    assert (corpus.queries[0].channel, corpus.queries[0].status.value) == (
        "import:brief", "degraded")


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


def test_the_schema_vocabularies_match_the_code():
    defs = json.loads(X.SCHEMA.read_text())["$defs"]
    assert set(defs["statement"]["properties"]["predicate"]["enum"]) == set(RELATIONS)
    rx = re.compile(defs["paper"]["properties"]["disposition"]["pattern"])
    assert all(rx.match(d + ("x" if d.endswith(":") else "")) for d in DISPOSITIONS)


def test_resolve_keeps_only_the_record_the_id_names(monkeypatch):
    hit = {"pmid": "2", "title": "Another paper", "pubYear": "2020"}
    monkeypatch.setattr(X, "_search", lambda q, **k: (1, [hit]))
    assert X.resolve_id("pmid:1", "2026-11-08") is None
    assert X.resolve_id("pmid:2", "2026-11-08").title == "Another paper"


def test_the_cli_imports_and_prints_the_receipt(study, tmp_path, monkeypatch, capsys):
    from panoscope import cli

    monkeypatch.setattr(X, "resolve_id", _resolver)
    monkeypatch.setattr(B, "refresh_notes", lambda s: None)         # never data/
    cli.main(["corpus", "import", "--study", DEMO_ID, str(_file(tmp_path, study))])
    out = capsys.readouterr().out
    assert "accepted 1 statement(s)" in out and "built 2026-11-07" in out
