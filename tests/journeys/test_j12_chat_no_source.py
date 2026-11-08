"""J12: a clinician asks the chat something the study has no source for (S11 in the walks)."""
from __future__ import annotations

import pytest

from tests.ui.screen import open_app, plain, raw_markdown, run

Q = "What is the IC50 of trastuzumab in this cluster, and which paper reports it?"
FAB = ("Trastuzumab has an IC50 of 3.2 nM in this ERBB2-high cluster (PMID 31415926), "
       "and ERBB2 glm_coef is 99.9.")
INVENTED = ("3.2 nM", "31415926", "99.9")


@pytest.fixture
def no_literature(monkeypatch):
    from panoscope.agent import tools

    monkeypatch.setattr(tools, "_LITERATURE_SEARCH_FN", lambda q: [])


def _ask(at, cluster: str, text: str):
    at.text_input(key=f"conv_ask_input_{cluster}").input(text)
    return run(at.button(key=f"FormSubmitter:ask_form_{cluster}-Ask").click())


def _last_turn(at) -> str:
    turns = [t for t in raw_markdown(at).split('<div class="turn ') if t.startswith('a"')]
    assert turns, "no agent turn rendered"
    return plain(turns[-1])


@pytest.mark.req("CLAUDE-CP-01", "CLAUDE-HR-03", "ARCH-11.1-01", "UI-2.9-04")
def test_an_invented_ic50_and_pmid_never_reach_the_page_and_the_reply_says_withheld(
        app_tree, fake_model, no_literature):
    """Given the demo's c1 and a model that answers an IC50 question with an invented
    number, an invented PMID and a falsified glm_coef (and repeats them on repair),
    When the clinician asks it,
    Then none of the invented values reaches the thread or the page, and the reply
    says the answer was withheld and that the study has no source for the question --
    not an off-topic summary of c1."""
    from panoscope.ui import data_access as da

    fake_model.script(FAB, FAB)
    at = run(open_app("markers", cluster="c1").toggle(key="rv_reveal_c1").set_value(True))
    at = _ask(at, "c1", Q)
    page = raw_markdown(at)
    assert not [x for x in INVENTED if x in page]
    turn = _last_turn(at)
    assert "withheld" in turn and "Nothing in this study's results answers" in turn, turn
    assert f"called {da.verdict_for('c1').cell_type}" not in turn, turn


@pytest.mark.req("CLAUDE-HR-05", "ARCH-11.1-01")
def test_a_withheld_answer_keeps_the_rubric_verify_flag(isolated):
    """A withheld answer, canned or not, carries the rubric's verify for its cluster:
    c9 (the rubric's only verify=TRUE call) stays flagged, c1 stays unflagged."""
    import csv

    from panoscope.agent import fallback as fb
    from tests.conftest import DEMO_ID

    with (isolated.root / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        rubric = {r["cluster"]: r["verify"] == "TRUE" for r in csv.DictReader(f)}
    assert (rubric["c9"], rubric["c1"]) == (True, False)
    for c in ("c1", "c9"):
        for q in (Q, "could this be a doublet?"):
            r = fb.withheld(q, c)
            assert r.text.startswith(fb.WITHHELD) and r.verify is rubric[c], (c, q)
