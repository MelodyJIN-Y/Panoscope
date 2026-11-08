"""J3: the biologist overrides the call and the literature agrees and dissents (S3)."""
from __future__ import annotations

import re

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import call_it, go, import_corpus, literature, upload
from tests.ui.screen import plain, raw_markdown, run


def _notes():
    from panoscope.agent import memory, tools

    return memory.read_notes(tools.memory_base_dir())


def _tension_line(at) -> str:
    return next(plain(m.value) for m in at.markdown if "literature check on your call" in m.value)


@pytest.fixture
def with_corpus(isolated, monkeypatch, tmp_path):
    """The demo plus a corpus from the biologist's own client: CD4 in T cells (one paper
    agrees, a PMC paper dissents) and POSTN in stromal cells (one each way)."""
    from panoscope.data.study import Study

    study = Study.open(DEMO_ID)
    path = literature(isolated, monkeypatch, study, tmp_path, [
        ("pmid:101", "CD4", "T_Cells", "agree"),
        ("pmcid:PMC7654321", "CD4", "T_Cells", "dissent"),
        ("pmid:201", "POSTN", "Stromal", "agree"),
        ("pmid:202", "POSTN", "Stromal", "dissent"),
    ])
    at = import_corpus(upload(), DEMO_ID, path)
    receipt = at.code[0].value
    assert receipt.startswith("accepted 4 statement(s), rejected 0"), receipt
    return run(at.button(key="ob_corpus_skip").click())


def _states(cluster: str, label: str) -> dict:
    from panoscope.data.study import Study
    from panoscope.pipeline import records
    from panoscope.reference import view as V
    from panoscope.ui import review

    s = Study.open(DEMO_ID)
    return dict(review.gene_states(records.build(s, cluster), V.for_study(s).canonical(label)))


@pytest.mark.req("CLAUDE-MEM-01", "ARCH-8.2-01", "UI-5.1-01", "UI-2.9-05", "CORP-A2",
                 "PLAN-P3.4-02")
def test_an_override_shows_agreement_and_dissent_with_real_ids_even_with_no_marker_observed(
        with_corpus):
    """Given the demo and a corpus where one paper says CD4 marks T cells and a PMC
    paper says T cells lacked CD4, and the same split for POSTN in stromal cells,
    When the immunologist calls c3 T_Cells (CD4 is observed there),
    Then the proposal and their call sit side by side, the literature line reads
    1 agree / 1 dissent, verify is untouched, and both ids are links on Summary;
    When they call c4 Stromal, where no Stromal marker is observed,
    Then the corpus is still consulted for Stromal's markers: 1 agree / 1 dissent,
    never 'thin' while in-window statements exist."""
    import csv

    from panoscope.data.study import Study
    from panoscope.types.measured import MeasuredState

    with (Study.open(DEMO_ID).dir / "interp" / "verdicts.csv").open() as f:
        verify = {r["cluster"]: r["verify"] for r in csv.DictReader(f)}

    assert _states("c3", "T_Cells").get("CD4") is MeasuredState.OBSERVED
    at = call_it(with_corpus, "c3", pick="T_Cells")
    assert sum('class="pano-rv-assess"' in m.value for m in at.markdown) == 2
    assert "1 agree / 1 dissent" in _tension_line(at)
    (note,) = _notes()
    t = note.tension
    assert ([c.pmid for c in t.agree], [c.pmid for c in t.dissent]) == (["101"], [
        "pmcid:PMC7654321"])
    md = raw_markdown(go(at, "summary"))
    assert "europepmc.org/article/PMC/PMC7654321" in md, md[:200]

    c4 = _states("c4", "Stromal")
    assert MeasuredState.OBSERVED not in c4.values() and "POSTN" in c4
    at = call_it(at, "c4", pick="Stromal")
    line = _tension_line(at)
    assert not re.search(r"\bthin\b", line) and "1 agree / 1 dissent" in line, line
    with (Study.open(DEMO_ID).dir / "interp" / "verdicts.csv").open() as f:
        assert {r["cluster"]: r["verify"] for r in csv.DictReader(f)} == verify


@pytest.mark.req("UI-5.2-05", "CLAUDE-MEM-02", gap=True)
def test_an_override_lets_the_biologist_choose_dataset_or_lab_scope(isolated):
    """Given the demo, When the biologist calls c3 themselves,
    Then scope is one of two taps with this cluster, this dataset and the lab
    (interface §5.7), not fixed at this cluster."""
    at = go(upload(), "markers", "c3")
    run(at.toggle(key="rv_calling_c3").set_value(True))
    options = [str(o) for k in ("button_group", "radio", "selectbox") for w in at.get(k)
               if "scope" in (w.key or "") for o in (w.options or ())]
    assert any("lab" in o for o in options) and any("dataset" in o for o in options), options
