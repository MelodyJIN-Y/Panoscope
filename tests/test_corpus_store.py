"""The corpus store v2: SQLite round trip, span binding, ledger, version, per-key states.

All records here are SYNTHETIC ("1000000x" is not a PubMed id); nothing is fetched.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from panoscope.corpus.retrieve import Support, key_state
from panoscope.corpus.store import (
    Corpus, CorpusVersion, Disposition, QueryRun, QueryStatus, SourceRecord,
    Statement, UnboundSpan, load, log_hash, save,
)

ABSTRACT = "GENEX is expressed by fibroblasts in human breast tissue samples."
REC = SourceRecord(uri="pmid:10000001", title="A synthesised record", abstract=ABSTRACT,
                   via="FakeConnector", retrieved="2026-11-08",
                   publication_type=("Journal Article",))
SPAN = "GENEX is expressed by fibroblasts"
STMT = Statement(uri=REC.uri, gene="GENEX", cell_type="Fibroblasts", tissue="breast",
                 predicate="expressed_in", stance="supports", span=SPAN, start=0,
                 end=len(SPAN), section="abstract", rung="W0*", written_by="model/route")
RUN = QueryRun(query='"GENEX"[tiab] AND "breast"[tiab] AND "fibroblast"[tiab]',
               channel="C1", gene="GENEX", cell_type="Fibroblasts", total=1, retrieved=1,
               ids=(REC.uri,))
HGNC = QueryRun(query="hgnc symbol GENEX", channel="vocab:hgnc", gene="GENEX",
                total=1, retrieved=1)
CENSUS = QueryRun(query='"GENEX"[tiab] AND ("breast"[tiab])', channel="census",
                  gene="GENEX", total=12)
VERSION = CorpusVersion(study="s1", conditions=(("tissue", "breast"),),
                        reference=(("lab", "abc"),), generator="model/route",
                        query_log_hash=log_hash((RUN,)), built_at="2026-11-08")


def test_the_store_round_trips_every_table_through_sqlite(tmp_path):
    c = Corpus(records=(REC,), statements=(STMT,), queries=(CENSUS, RUN),
               ledger=(Disposition(REC.uri, "READ:1:0"),), version=VERSION, built="x")
    assert load(save(c, tmp_path / "corpus.sqlite")) == c


def test_a_statement_not_verbatim_at_its_offsets_is_refused():
    with pytest.raises(UnboundSpan):
        Corpus(records=(REC,), statements=(dataclasses.replace(STMT, start=1),))
    with pytest.raises(UnboundSpan):
        Corpus(statements=(STMT,))            # no record held


def test_the_ledger_names_every_returned_id_without_a_disposition():
    assert Corpus(records=(REC,), queries=(RUN,)).unclosed == (REC.uri,)
    assert Corpus(queries=(RUN,), ledger=(Disposition(REC.uri, "PREFILTER_NO_ALIAS"),)
                  ).unclosed == ()
    with pytest.raises(ValueError):
        Disposition(REC.uri, "LOOKED_FINE")
    with pytest.raises(ValueError):
        Corpus(ledger=(Disposition("a", "RETRACTED"), Disposition("a", "NON_TISSUE")))


def test_a_partial_retrieval_must_say_truncated_and_a_query_must_be_exact():
    with pytest.raises(ValueError):
        dataclasses.replace(RUN, total=300, retrieved=200)
    dataclasses.replace(RUN, total=300, retrieved=200, status=QueryStatus.TRUNCATED)
    with pytest.raises(ValueError):
        dataclasses.replace(RUN, query='"{GENE}"[tiab]')


def test_built_at_is_outside_the_version_content_hash():
    assert dataclasses.replace(VERSION, built_at="later").content_hash == VERSION.content_hash
    assert dataclasses.replace(VERSION, generator="other").content_hash != VERSION.content_hash


def test_full_text_is_licence_bearing_and_retraction_comes_from_publication_type():
    meta = SourceRecord(uri="pmid:10000002", title="t", via="X")
    assert meta.redistributable and meta.is_real and not meta.retracted
    assert not dataclasses.replace(meta, full_text="body").redistributable
    assert dataclasses.replace(meta, publication_type=("Retracted Publication",)).retracted


def test_each_key_resolves_to_one_of_three_states_and_reads_without_novelty():
    built = Corpus(records=(REC,), queries=(CENSUS, RUN), statements=(STMT,))
    found = key_state(built, "GENEX", "Fibroblasts")
    assert found.state is Support.FOUND and found.statements == (STMT,)

    rival = dataclasses.replace(RUN, cell_type="T cells", ids=())
    thin = key_state(Corpus(queries=(HGNC, CENSUS, rival)), "GENEX", "T cells")
    assert thin.state is Support.NONE_FOUND and "thin" in thin.reads_as

    failed = dataclasses.replace(CENSUS, status=QueryStatus.FAILED)
    partial = key_state(Corpus(queries=(HGNC, failed, rival)), "GENEX", "T cells")
    assert partial.state is Support.NONE_FOUND and partial.gaps == ("census failed",)
    assert "incomplete search" in partial.reads_as
    assert not re.search(r"\bthin\b", partial.reads_as)

    cut = dataclasses.replace(rival, total=300, retrieved=200, status=QueryStatus.TRUNCATED)
    assert key_state(Corpus(queries=(HGNC, CENSUS, cut)), "GENEX", "T cells"
                     ).gaps == ("C1 truncated",)
    assert key_state(Corpus(queries=(rival,)), "GENEX", "T cells"
                     ).gaps == ("vocab:hgnc not run", "census not run")

    unbuilt = key_state(Corpus(queries=(CENSUS,)), "GENEX", "B cells")
    assert unbuilt.state is Support.NOT_BUILT and "cannot settle" in unbuilt.reads_as

    for s in (found, thin, partial, unbuilt):
        assert not re.search(r"novel|not previously reported", s.reads_as, re.I)
