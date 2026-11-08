"""The corpus build entry: D2 scope, fresh / resume / top-up versions, refusal without a
key, the header's staleness, and a literature-off build that is byte-identical.
Stubbed vocabulary, retrieval and reader; no network, no model."""

from __future__ import annotations

import json
from datetime import date

import pytest

from panoscope.corpus import build as B
from panoscope.corpus.channels import _Run
from panoscope.corpus.read import Receipt
from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, CorpusVersion, QueryRun, SourceRecord, corpus_path, load, save,
)
from panoscope.data.study import Study
from tests.conftest import DEMO_ID
from tests.test_corpus_read import ABSTRACT, FULL, MARK, POSTN, Stub, _ctx

TODAY = date(2026, 11, 8)


@pytest.fixture
def stubbed(monkeypatch):
    """The demo study read-only, with every network stage stubbed; returns the call log."""
    calls: list[str] = []
    monkeypatch.setattr(B, "context_for", lambda study: _ctx(study=study.id))
    monkeypatch.setattr(B, "scope", lambda study, types: ("POSTN",))

    def vocabulary(gene, tissue):
        calls.append(f"vocab {gene}")
        return POSTN, (QueryRun(f"hgnc {gene}", "vocab:hgnc", gene, total=1, retrieved=1),
                       QueryRun(f"census {gene}", "census", gene, total=1))

    def retrieve(ctx, gene, terms, *, today, tissue_terms, path, papers_of_interest):
        calls.append(f"retrieve {gene}")
        try:
            run = _Run(load(path), path, 0, today.isoformat())
        except CorpusUnavailable:
            run = _Run(Corpus(), path, 0, today.isoformat())
        if gene != "POSTN":
            run.log(QueryRun(f"q {gene}", "C1", gene, "Fibroblasts"))
            return run.corpus
        rec = SourceRecord(uri="pmid:1", title="POSTN in breast", abstract=ABSTRACT,
                           full_text=FULL, via="Europe PMC", year=2026)
        run.log(QueryRun("q POSTN", "C1", gene, "Fibroblasts", total=1, retrieved=1,
                         ids=(rec.uri,)), [rec])
        run.log(QueryRun("prefilter gate for POSTN", "gate", gene, total=1, retrieved=1,
                         ids=(rec.uri,)))
        return run.corpus

    monkeypatch.setattr(B.vocab, "vocabulary", vocabulary)
    monkeypatch.setattr(B, "retrieve", retrieve)
    monkeypatch.setattr(B, "refresh_notes", lambda study: calls.append("notes"))  # never data/
    return calls


def _build(**kw):
    receipt = Receipt()
    corpus = B.build(Study.open(DEMO_ID), today=TODAY, receipt=receipt,
                     reader=kw.pop("reader", Stub([MARK])), **kw)
    return corpus, receipt


def test_a_build_reads_its_scope_and_writes_a_dated_version_with_its_receipt(stubbed):
    lines: list[str] = []
    corpus, receipt = _build(progress=lambda i, n, line: lines.append(f"{i}/{n} {line}"))
    v = corpus.version
    assert [s.gene for s in corpus.statements] == ["POSTN"]
    assert (v.built_at, v.stale_after) == ("2026-11-08", "2027-05-07")
    assert v.cost == receipt.text and receipt.calls == 2 and v.study == DEMO_ID
    assert ("tissue_id", "UBERON:0000310") in v.conditions
    assert lines[-1].startswith("1/1 version ") and load(corpus_path(DEMO_ID)) == corpus


def test_the_tissue_block_carries_the_intake_uberon_label_not_only_the_prose(stubbed, monkeypatch):
    from panoscope.data import study as S

    real, seen = S._manifest, []
    monkeypatch.setattr(S, "_manifest", lambda s: {**real(s), "tissue_label": "breast"})
    monkeypatch.setattr(B, "context_for", lambda s: _ctx(study=s.id, tissue="human breast cancer"))
    monkeypatch.setattr(B, "retrieve", lambda *a, tissue_terms, **k: seen.append(tissue_terms))
    _build()
    assert seen == [("breast", "human breast cancer")]


@pytest.mark.req("PLAN-P3.1-01")
def test_resume_keeps_what_is_built_and_a_fresh_build_starts_over(stubbed):
    save(Corpus(queries=(QueryRun("old", "C1", "OLD", "Fibroblasts"),)), corpus_path(DEMO_ID))
    resumed, _ = _build(resume=True)
    assert any(q.gene == "OLD" for q in resumed.queries)
    fresh, _ = _build()
    assert not any(q.gene == "OLD" for q in fresh.queries)


@pytest.mark.req("ARCH-8.1-02")
def test_top_up_writes_a_new_version_and_refuses_without_a_built_corpus(stubbed):
    with pytest.raises(CorpusUnavailable, match="top-up refused"):
        _build(genes=("ACTA2",))
    first, _ = _build()
    topped, _ = _build(genes=("ACTA2",))
    assert topped.version.content_hash != first.version.content_hash
    assert {q.gene for q in topped.queries} >= {"POSTN", "ACTA2"}
    assert topped.statements == first.statements
    assert "retrieve ACTA2" in stubbed and stubbed.count("retrieve POSTN") == 1


def test_no_model_key_refuses_before_any_lookup(stubbed, monkeypatch):
    from panoscope import cli
    from panoscope.agent import loop

    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    with pytest.raises(SystemExit, match="no model key"):
        cli.main(["corpus", "build", "--study", DEMO_ID])
    assert stubbed == [] and not corpus_path(DEMO_ID).exists()


@pytest.mark.req("ARCH-8.3-01")
def test_the_header_says_not_built_incomplete_current_or_stale():
    assert B.header(DEMO_ID) == B.NOT_BUILT
    p = corpus_path(DEMO_ID)
    save(Corpus(), p)
    assert "incomplete" in B.header(DEMO_ID) and B.header(DEMO_ID).endswith("cannot settle")
    v = CorpusVersion(DEMO_ID, (), (), "stub", "h", built_at="2026-11-08",
                      stale_after="2027-05-07")
    save(Corpus(version=v), p)
    assert B.header(DEMO_ID, date(2027, 5, 6)).endswith("current until 2027-05-07")
    assert B.header(DEMO_ID, date(2027, 5, 7)).endswith("STALE since 2027-05-07")
    assert v.content_hash == CorpusVersion(DEMO_ID, (), (), "stub", "h").content_hash


def test_doctor_reports_each_studys_corpus(capsys):
    from panoscope import cli

    cli.main(["doctor"])
    assert B.NOT_BUILT in capsys.readouterr().out


@pytest.mark.req("ARCH-8.1-01", "ARCH-8.1-02")
def test_scope_is_the_canonical_markers_plus_each_clusters_top_ten():
    import pandas as pd

    from panoscope.reference import view as ref_view

    study = Study.open(DEMO_ID)
    labels = tuple(dict.fromkeys(v["cell_type"] for v in study.unit_labels.values()))
    genes = B.scope(study, labels)
    t = pd.read_csv(study.inputs / "markers_top.csv")
    c1 = t[t.top_cluster == "c1"].sort_values("glm_coef", ascending=False).gene.tolist()
    view = ref_view.for_study(study)
    canon = {g for lab in labels for g in view.canonical(lab)}
    assert canon and canon <= set(genes) and len(genes) == len(set(genes))
    assert set(c1[:10]) <= set(genes) and not (set(c1[10:]) - canon) & set(genes)


@pytest.mark.req("ARCH-11.2-01", "CORP-A3", "PLAN-P3C-01")
def test_a_literature_off_build_is_byte_identical_with_a_corpus_present(tmp_path):
    from panoscope.pipeline import run as pipeline_run

    def tree(root):
        out = {p.relative_to(root).as_posix(): p.read_bytes()
               for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        man = json.loads(next(root.rglob("manifest.json")).read_text())
        return out, man["artifacts"]

    pipeline_run.run(DEMO_ID, root=tmp_path / "off")
    save(Corpus(version=CorpusVersion(DEMO_ID, (), (), "stub", "h", built_at="B")),
         corpus_path(DEMO_ID))
    pipeline_run.run(DEMO_ID, root=tmp_path / "on")
    off, on = tree(tmp_path / "off"), tree(tmp_path / "on")
    assert off[0] and off == on


def test_a_summary_cell_with_no_note_says_cannot_settle_until_a_corpus_is_built():
    from panoscope.ui import data_access as da
    from panoscope.ui import summary

    v = da.all_verdicts()[0]
    assert B.NOT_BUILT in summary._biology_cell(v, {}, "")
    save(Corpus(version=CorpusVersion(DEMO_ID, (), (), "stub", "h")), corpus_path(DEMO_ID))
    assert summary._biology_cell(v, {}, "") == '<span class="bio-dim">—</span>'


@pytest.mark.req("ARCH-8.3-01", "CORP-B6")
def test_the_export_header_carries_the_corpus_staleness():
    from panoscope.ui import report

    save(Corpus(version=CorpusVersion(DEMO_ID, (), (), "stub", "h", stale_after="2000-01-01")),
         corpus_path(DEMO_ID))
    assert report._provenance(DEMO_ID)["corpus_status"].endswith("STALE since 2000-01-01")
