"""J10: the biologist does not know the fixation or the disease state (S10)."""
from __future__ import annotations

import re
from io import BytesIO

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import open_app


def _docx(monkeypatch) -> dict:
    from panoscope.ui import report

    got: dict = {}
    real = report.working_docx

    def keep(*a, **k):
        got["docx"] = real(*a, **k)            # the very bytes the download serves
        return got["docx"]

    monkeypatch.setattr(report, "working_docx", keep)
    return got


def _text(data: bytes) -> str:
    import docx

    d = docx.Document(BytesIO(data))
    return "\n".join([p.text for p in d.paragraphs]
                     + [c.text for t in d.tables for r in t.rows for c in r.cells])


@pytest.mark.req("ARCH-6.2-03", "ARCH-6.3-02", "ARCH-8.1-03")
def test_undeclared_fixation_and_disease_never_block_and_the_brief_says_not_declared(
        isolated, monkeypatch):
    """Given the demo, whose manifest declares neither fixation nor disease state,
    When the biologist reads the intake and exports the brief for their own LLM,
    Then nothing blocks, each skipped field propagates as '<field> unknown', and the
    brief says the disease state was not declared and the window is W0*."""
    from panoscope import platforms
    from panoscope.corpus import brief
    from panoscope.corpus import exchange as X
    from panoscope.data.study import Study
    from panoscope.agent.intake import derive

    def offline(url, *a, **k):              # what a blocked lookup raises, without retries
        raise X.SourceUnavailable(f"{url}: offline")

    monkeypatch.setattr(X.vocab, "_get", offline)
    study = Study.open(DEMO_ID)
    draft = derive(platforms.for_study(study), study)
    assert not draft.missing
    text = draft.checklist().render()
    for f in ("fixation", "disease_state"):
        assert f'propagates as "{f} unknown"' in text
    out = brief.write(study)
    md = (out / "BRIEF.md").read_text()
    assert "Disease state: not declared (window W0*)" in md


@pytest.mark.req("ARCH-6.2-03", "ARCH-6.2-01", gap=True)
def test_the_exported_report_names_every_unknown_condition(isolated, monkeypatch):
    """When the biologist decides every cluster and exports, Then the report lists each
    condition with its provenance, so 'unknown' cannot be read as 'FFPE, untreated'."""
    from panoscope.data.study import Study
    from panoscope.pipeline import decisions as D

    got = _docx(monkeypatch)
    for c in Study.open(DEMO_ID).units:
        D.defer(c, by="tester", at="2026-11-09T00:00:00")
    assert len(open_app("summary").download_button) == 2
    text = _text(got["docx"]).lower()
    assert all(re.search(rf"{f}\W+(unknown|not declared)", text)      # any wording, named
               for f in ("fixation", "disease.state")), [
        ln for ln in text.splitlines() if "unknown" in ln][:5]
