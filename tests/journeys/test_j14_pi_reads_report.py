"""J14: the PI reads the exported report, not the app."""
from __future__ import annotations

import re
from io import BytesIO

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import call_it, defer, go, import_corpus, literature, upload
from tests.ui.screen import run


def _text(data: bytes) -> str:
    import docx

    d = docx.Document(BytesIO(data))
    return "\n".join([p.text for p in d.paragraphs]
                     + [c.text for t in d.tables for r in t.rows for c in r.cells])


@pytest.fixture
def report(isolated, monkeypatch, tmp_path):
    """The demo with a corpus (CD4 in T cells: one paper agrees, one PMC paper dissents),
    c3 overridden to T_Cells, c5 deferred, the rest accepted; the docx the download serves."""
    from panoscope.data.study import Study
    from panoscope.ui import report as rep

    study = Study.open(DEMO_ID)
    path = literature(isolated, monkeypatch, study, tmp_path, [
        ("pmid:101", "CD4", "T_Cells", "agree"),
        ("pmcid:PMC7654321", "CD4", "T_Cells", "dissent")])
    at = run(import_corpus(upload(), DEMO_ID, path).button(key="ob_corpus_skip").click())
    got: dict = {}
    real = rep.working_docx

    def keep(*a, **k):
        got["docx"] = real(*a, **k)
        return got["docx"]

    monkeypatch.setattr(rep, "working_docx", keep)
    at = call_it(at, "c3", pick="T_Cells")
    at = defer(at, "c5")
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    for c in study.units:               # the rest accepted (the tap is J2's subject)
        if c not in ("c3", "c5"):
            D.accept(c, da.verdict_for(c).cell_type, by="pi", at="2026-11-09T00:00:00")
    assert len(go(at, "summary").download_button) == 2
    return study, _text(got["docx"])


@pytest.mark.req("UI-5.5-02", "ARCH-7.5-03", "CLAUDE-HR-03", "UI-5.2-01", "CLAUDE-MEM-01")
def test_the_report_carries_every_decision_and_only_real_citations(report):
    """Given the demo decided (c3 overridden to T_Cells against the literature, c5
    deferred, the rest accepted), When the PI reads the exported report,
    Then every cluster has its decision line, the override shows the biologist's call
    beside the proposal and its claim, no '% conditional' figure appears, and every
    PMID in the report is one the study's notes or corpus hold."""
    study, text = report
    decisions = text.split("Annotation decisions", 1)[1].split("Cross-cluster", 1)[0]
    for c in study.units:
        want = {"c3": "overridden", "c5": "deferred"}.get(c, "accepted")
        assert f"{c}: {want} · proposal" in decisions, c
    c3 = next(ln for ln in decisions.splitlines() if ln.startswith("c3:"))
    assert "your call T_Cells" in c3 and "override claim note:" in c3
    assert "PMC7654321" in c3 and "101" in c3, c3        # the tension, with real ids
    assert "% conditional" not in text.lower()
    known = set()
    for p in (study.dir / "interp").glob("*.json"):
        known |= set(re.findall(r"\d{6,9}", p.read_text()))
    cited = set(re.findall(r"PMID[:\s]*(\d+)", text))
    assert cited and cited <= known | {"101"}, cited - known


@pytest.mark.req("UI-5.2-03", gap=True)
def test_the_report_says_what_would_settle_a_deferred_cluster(report):
    """The deferred cluster's line says what would settle it (interface §5.2)."""
    _, text = report
    decisions = text.split("Annotation decisions", 1)[1].split("Cross-cluster", 1)[0]
    c5 = next(ln for ln in decisions.splitlines() if ln.startswith("c5:"))
    assert "settle" in c5 or "needs another assay" in c5, c5
