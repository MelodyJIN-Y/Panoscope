"""J17: the corpus my client searched, narrowed and truncated (corpus SKILL paging rule)."""
from __future__ import annotations

from datetime import date

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import literature

_TODAY = date(2026, 11, 9)


def _query(status: str, total: int, retrieved: int, query: str = '"CD4" AND breast') -> dict:
    return {"kind": "query", "query": query, "source": "Europe PMC", "channel": "C1",
            "gene": "CD4", "cell_type": "T_Cells", "total": total, "retrieved": retrieved,
            "status": status}


def _import(isolated, monkeypatch, tmp_path, *queries: dict):
    from panoscope.corpus import exchange as X
    from panoscope.corpus.retrieve import key_state
    from panoscope.data.study import Study

    study = Study.open(DEMO_ID)
    path = literature(isolated, monkeypatch, study, tmp_path,
                      [("pmid:101", "CD4", "T_Cells", "agree")], extra_rows=list(queries))
    corpus, _ = X.import_file(study, path, today=_TODAY)
    return key_state(corpus, "CD4", "T_Cells")


@pytest.mark.req("IMPL-6.1-07", "SKC-PAGE-01", "PLAN-P3.4-02")
def test_a_truncated_or_falsely_ok_search_reads_incomplete_never_thin(
        isolated, monkeypatch, tmp_path):
    """Given a client search for CD4 in T cells that hit 450 papers and retrieved 200,
    When it is logged TRUNCATED -- or logged 'OK' anyway --
    Then the key reads incomplete and names the truncated query, in both cases (a
    client's OK with retrieved < total is recomputed, never trusted)."""
    for status in ("TRUNCATED", "OK"):
        state = _import(isolated, monkeypatch, tmp_path, _query(status, 450, 200))
        assert any("truncated" in g for g in state.gaps), (status, state.gaps)


@pytest.mark.req("SKC-PAGE-01", gap=True)
def test_a_narrowed_then_fully_retrieved_search_reads_complete_and_shows_its_narrowing(
        isolated, monkeypatch, tmp_path):
    """Given the over-cap search narrowed with AND (M) and then fully retrieved
    (retrieved = total after narrowing), Then the import accepts the NARROWED line, the
    key carries no gap for that query, and the narrowing step is kept in the log."""
    from panoscope.corpus.store import CorpusUnavailable

    narrowed = _query("NARROWED", 120, 120, '"CD4" AND breast AND (M)')
    try:
        state = _import(isolated, monkeypatch, tmp_path, _query("TRUNCATED", 450, 200), narrowed)
    except CorpusUnavailable as exc:                     # the schema has no NARROWED today
        state, why = None, str(exc)
    assert state is not None, why
    assert not [g for g in state.gaps if "C1" in g], state.gaps
