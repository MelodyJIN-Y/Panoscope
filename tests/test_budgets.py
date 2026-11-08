"""IMPLEMENTATION §9 `test_budgets` / ARCH §18: retrieval < 300 ms, dataset load < 10 s,
full build < 5 min. Nightly only (`pytest -m budget`): timing is not a commit gate."""
from __future__ import annotations

import time

import pytest

from tests.conftest import DEMO_ID

pytestmark = [pytest.mark.budget, pytest.mark.req("ARCH-18-06")]


def _seconds(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def test_retrieval_over_a_large_corpus_is_under_300_ms() -> None:
    from panoscope.corpus import retrieve
    from panoscope.corpus.store import Corpus, SourceRecord

    records = tuple(SourceRecord(uri=f"pmid:{i}", title=f"gene {i % 97} in tumour {i}",
                                 abstract="ERBB2 KRT7 EPCAM " * (i % 5), via="Europe PMC")
                    for i in range(1, 5001))
    corpus = Corpus(records=records)
    assert retrieve.search(corpus, ["ERBB2", "tumour"]).value.hits   # it found something
    assert _seconds(lambda: retrieve.search(corpus, ["ERBB2", "tumour"])) < 0.3


def test_the_demo_dataset_loads_in_under_10_s() -> None:
    import pandas as pd

    from panoscope.data.study import Study

    def load():
        study = Study.open(DEMO_ID).require()
        pd.read_csv(study.dir / "viz" / "cells.csv")

    assert _seconds(load) < 10


def test_a_full_build_of_the_demo_is_under_5_min() -> None:
    """In memory: records.build_all is the build; nothing is written to data/studies."""
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    built: dict = {}
    assert _seconds(lambda: built.update(records.build_all(Study.open(DEMO_ID)))) < 300
    assert len(built) == 9
