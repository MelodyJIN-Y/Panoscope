"""7.4 and 7.7 (auto-scroll): a chat turn in a real browser, through the REAL agent
loop with a scripted model client (tests/browser/launch.py patches it in before
Streamlit starts, so the app gains no test hook).

Only a browser runs the inline auto-scroll script and follows a citation link."""
from __future__ import annotations

import json

import pytest

from tests.browser.conftest import app_env, see, serve, settle

pytestmark = pytest.mark.browser

_PMID, _PMC = "12345678", "PMC7654321"
_ANSWERS = [f"c1 reads as tumour epithelium; see PMID:{_PMID} and {_PMC}.",
            "ERBB2 is the strongest marker of c1.",
            "Nothing further in this study's results bears on that."]


@pytest.fixture(scope="module")
def chat_server(tmp_path_factory):
    env = app_env(tmp_path_factory.mktemp("chat"))
    env["PANOSCOPE_FAKE_REPLIES"] = json.dumps(_ANSWERS)
    env["PANOSCOPE_FAKE_IDS"] = json.dumps([_PMID, _PMC])
    with serve(env, launcher="tests.browser.launch") as base:
        yield base


def _ask(page, text: str) -> None:
    box = page.get_by_placeholder("Ask, suggest another cell type, or override this call…")
    box.fill(text)
    page.get_by_role("button", name="Ask", exact=True).click()


def _thread_at_bottom(page) -> dict:
    return page.evaluate("""() => {
      const t = document.querySelector('[class*="st-key-conv_thread"]');
      return {top: t.scrollTop, room: t.clientHeight, height: t.scrollHeight};
    }""")


@pytest.mark.req("UI-2.9-05", "UI-2.1-01")
def test_an_answer_lands_at_the_bottom_with_its_citations_as_new_tab_links(
        chat_server, page) -> None:
    """Given c1 with its proposal shown and a model key set,
    When I ask three questions,
    Then each question shows as my bubble and each answer follows it,
    And the PMID and the PMCID in the first answer are links that open PubMed /
    PMC in a new tab,
    And after the last answer the thread is scrolled to the newest message."""
    page.set_viewport_size({"width": 1440, "height": 700})   # a thread that overflows
    page.goto(f"{chat_server}/?page=markers&cluster=c1")
    see(page, "Show the proposal")
    page.get_by_text("Show the proposal (call and band)").click()
    see(page, "Ask, suggest|override this call")
    settle(page)

    for q, a in zip(("what is c1?", "which marker leads?", "anything else?"), _ANSWERS):
        _ask(page, q)
        see(page, q)
        see(page, a.split(";")[0])
    settle(page)

    pmid = page.locator(f'a.pcite[href*="{_PMID}"]')
    pmc = page.locator(f'a.pcite[href*="{_PMC}"]')
    assert pmid.count() == 1 and pmc.count() == 1
    assert pmid.get_attribute("href") == f"https://pubmed.ncbi.nlm.nih.gov/{_PMID}/"
    assert pmid.get_attribute("target") == pmc.get_attribute("target") == "_blank"

    t = _thread_at_bottom(page)
    assert t["height"] > t["room"], f"the thread never overflowed, so nothing to scroll: {t}"
    assert t["top"] + t["room"] >= t["height"] - 4, f"the thread is not at its newest message: {t}"
