"""7.5: the report downloads, through the browser's real download, opened as files.

Export needs a terminal decision on every cluster (interface §5.2). The files that
leave the lab must carry each cluster's decision and the override's tension."""
from __future__ import annotations

import csv
import re
import zlib
from io import BytesIO

import pytest

from tests.browser.conftest import app_env, see, seed, serve, settle
from tests.conftest import DATASETS_ROOT, DEMO_ID

pytestmark = pytest.mark.browser

_CALLS = {r["cluster"]: r["cell_type"] for r in
          csv.DictReader((DATASETS_ROOT / DEMO_ID / "interp" / "verdicts.csv").open())}
_STATE = {"c2": "overridden", "c9": "deferred"}   # the rest accepted

_SEED = """
from panoscope.agent import tools
from panoscope.pipeline import decisions as D
b, by, at = tools.memory_base_dir(), 'you@example.com', '2026-11-09T00:00:00+00:00'
for c, label in {calls!r}.items():
    if c == 'c2':
        continue                     # overridden by mouse in the test, as a biologist would
    elif c == 'c9':
        D.defer(c, by=by, at=at, base_dir=b)
    else:
        D.accept(c, label, by=by, at=at, base_dir=b)
"""


@pytest.fixture(scope="module")
def decided_server(tmp_path_factory):
    env = app_env(tmp_path_factory.mktemp("decided"))
    seed(env, _SEED.format(calls=_CALLS))
    with serve(env) as base:
        yield base


def _download(page, label: str) -> bytes:
    with page.expect_download() as dl:
        page.get_by_role("button", name=label).click()
    return open(dl.value.path(), "rb").read()


def _docx_text(data: bytes) -> str:
    import docx

    d = docx.Document(BytesIO(data))
    return "\n".join([p.text for p in d.paragraphs]
                     + [c.text for t in d.tables for r in t.rows for c in r.cells])


def _pdf_text(data: bytes) -> str:
    """The text-showing strings of every page stream (fpdf2 deflates them)."""
    out = []
    for raw in re.findall(rb"stream\r?\n(.*?)\r?\nendstream", data, re.DOTALL):
        try:
            raw = zlib.decompress(raw)
        except zlib.error:
            pass
        out += [s.decode("latin-1") for s in re.findall(rb"\(((?:\\.|[^\\)])*)\)\s*Tj", raw)]
    return " ".join(out)


@pytest.mark.req("UI-5.2-01")
def test_no_download_while_a_cluster_is_open(server, page) -> None:
    page.goto(f"{server}/?page=summary")
    see(page, "still open: c1, c2, c3, c4, c5, c6, c7, c8, c9")
    settle(page)
    assert page.get_by_role("button", name=re.compile("Report \\((Word|PDF)\\)")).count() == 0


@pytest.mark.req("UI-5.5-02", "UI-5.2-01", "ARCH-13.8-01", "CLAUDE-MEM-01")
def test_the_downloaded_word_and_pdf_carry_every_decision_and_the_tension(
        decided_server, page) -> None:
    """Given every cluster decided (c9 deferred and the rest accepted on disk; c2
    overridden to CAF by mouse, with no corpus built),
    When I download the Word and the PDF report,
    Then each opens and names every cluster's decision against its proposal,
    And the override line carries my call and its literature check."""
    page.goto(f"{decided_server}/?page=markers&cluster=c2")
    see(page, "Call it yourself")
    page.get_by_text("Call it yourself").click()
    own = page.get_by_label("or your own label")
    own.fill("CAF")
    own.press("Enter")
    settle(page)
    page.get_by_role("button", name="Save call").click()
    see(page, "decision · overridden")

    page.goto(f"{decided_server}/?page=summary")
    see(page, "Report \\(Word\\)")
    settle(page)
    docx_text = _docx_text(_download(page, "⬇  Report (Word)"))
    pdf = _download(page, "⬇  Report (PDF)")
    assert pdf[:5] == b"%PDF-", pdf[:16]
    pdf_text = _pdf_text(pdf)

    for name, text in (("docx", docx_text), ("pdf", pdf_text)):
        flat = " ".join(text.split())
        for c, proposal in _CALLS.items():
            assert re.search(rf"{c}: {_STATE.get(c, 'accepted')} [·-] proposal {re.escape(proposal)}",
                             flat), (name, c, flat[:2000])
        c2 = flat.split("c2: overridden", 1)[1][:600]
        assert "your call CAF" in c2, (name, c2)
        # No corpus built: the line says no lookup happened, never that it is thin.
        line = c2.split(" c3:", 1)[0]
        assert "literature not checked" in line and "thin" not in line, (name, line)
