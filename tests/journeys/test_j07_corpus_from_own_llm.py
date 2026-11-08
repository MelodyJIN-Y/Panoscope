"""J7: a corpus built in the biologist's own LLM client, one file broken, one tampered (S7)."""
from __future__ import annotations

import json
import re

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import go, import_corpus, literature, upload
from tests.ui.screen import plain, run

_FABRICATED = "pmid:99999999"


@pytest.mark.req("IMPL-6.1-01", "IMPL-6.1-02", "IMPL-6.1-05", "IMPL-6.1-09", "IMPL-6.1-11",
                 "CLAUDE-HR-03", "PLAN-P3.4-02")
def test_a_broken_file_is_refused_by_line_and_a_tampered_one_rejects_only_its_lies(
        isolated, monkeypatch, tmp_path, capsys):
    """Given the demo and two files from the biologist's own client: one with a schema
    error on line 2, one with two real papers, a fabricated id and an altered quote,
    When they import the broken file, Then it is refused line by line and nothing written;
    When they import the tampered file, Then the receipt rejects the fabricated id as
    unresolved and the altered span as not verbatim, counts 1 unresolved id, matches the
    brief and stamps the version as built in their client; the header names the corpus;
    the T cell card reads 'search incomplete', never 'thin';
    And on the command line the receipt is the first thing printed, not buried under a
    line per refreshed note."""
    from panoscope import cli
    from panoscope.data.study import Study

    study = Study.open(DEMO_ID)
    good = literature(isolated, monkeypatch, study, tmp_path, [
        ("pmid:101", "CD4", "T_Cells", "agree"),
        ("pmcid:PMC7654321", "CD4", "T_Cells", "dissent")], extra_rows=[
        {"kind": "paper", "id": _FABRICATED, "channel": "C1", "disposition": "READ:1:0"},
        {"kind": "statement", "paper": _FABRICATED, "gene": "CD4", "cell_type": "T_Cells",
         "predicate": "MARKS", "span": "CD4 is the definitive T cell marker",
         "section": "abstract"},
        {"kind": "statement", "paper": "pmid:101", "gene": "CD4", "cell_type": "T_Cells",
         "predicate": "MARKS", "span": "CD4 exclusively marks T Cells", "section": "abstract"}])
    head = good.read_text().splitlines()[0]
    broken = tmp_path / "broken.jsonl"
    broken.write_text(head + "\n" + json.dumps({"kind": "statement", "paper": 5}) + "\n")

    at = import_corpus(upload(), DEMO_ID, broken)
    err = " ".join(e.value for e in at.error)
    assert "import refused, nothing written" in err and "line 2:" in err, err
    from panoscope.corpus.store import CorpusUnavailable, corpus_path, load

    with pytest.raises(CorpusUnavailable):
        load(corpus_path(DEMO_ID))

    run(at.text_input(key="ob_corpus_file").input(str(good)))
    at = run(at.button(key="ob_corpus_import").click())
    lines = at.code[0].value.splitlines()
    assert lines[0] == "accepted 2 statement(s), rejected 2"
    assert any(f"rejected {_FABRICATED} CD4: paper unresolved" in x for x in lines)
    assert any("pmid:101 CD4: span not verbatim" in x for x in lines), lines
    assert "unresolved id(s): 1" in lines
    assert "brief matches the study's current brief" in lines
    assert "some-model (Anthropic) in Claude Desktop, built in user client" in lines[-1]

    at = go(run(at.button(key="ob_corpus_skip").click()), "markers", "c3")
    text = " ".join(plain(m.value) for m in at.markdown)
    assert re.search(r"corpus \w+ built 2026-11-07", text), text[:400]
    card = next(plain(m.value) for m in at.markdown if "literature, per candidate type" in m.value)
    t_cells = card.split("T_Cells:", 1)[1].split("Tumor:", 1)[0] if "T_Cells:" in card else card
    assert "search incomplete" in t_cells and not re.search(r"\bthin\b", t_cells), t_cells

    capsys.readouterr()
    assert cli.main(["corpus", "import", "--study", DEMO_ID, str(good)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[panoscope] accepted 2 statement(s), rejected 2", out[:3]
    assert not [x for x in out if "no-cite" in x], len(out)
