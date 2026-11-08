"""7.2: the evidence-first decision path on Markers, by mouse (interface §5.2-5.3).

The proposal waits behind the evidence; accepting needs it shown; defer and
override each take a click or two; the rail mark follows the decision; an
override sits beside the proposal's own assessment."""
from __future__ import annotations

import re

import pytest

from tests.browser.conftest import body, see, settle

pytestmark = pytest.mark.browser


def _rail(page) -> list[str]:
    return [" ".join(t.split()) for t in page.get_by_role("button").all_inner_texts()
            if t.strip()[:1] in "○✓✎…"]


def _open(page, server: str, cluster: str) -> None:
    page.goto(f"{server}/?page=markers&cluster={cluster}")
    see(page, "Show the proposal")
    settle(page)


@pytest.mark.req("UI-5.3-01", "UI-5.2-01")
def test_accept_needs_the_proposal_shown_and_marks_the_rail(server, page) -> None:
    """Given c4 opened on its evidence, the call hidden,
    Then Accept is disabled and the rail names no cell type for c4;
    When I show the proposal and accept,
    Then the decision reads accepted and the rail marks ✓ c4 Myoepithelial
    (c4's call in the study's verdicts.csv)."""
    _open(page, server, "c4")
    accept = page.get_by_role("button", name="Accept proposal")
    assert accept.is_disabled()
    assert "○ c4" in _rail(page)

    page.get_by_text("Show the proposal (call and band)").click()
    settle(page)
    assert accept.is_enabled()
    accept.click()
    see(page, "decision · accepted")
    assert "✓ c4 Myoepithelial" in _rail(page)


@pytest.mark.req("UI-5.2-01")
def test_defer_is_one_click_and_is_terminal(server, page) -> None:
    _open(page, server, "c5")
    page.get_by_role("button", name="Defer").click()
    see(page, "decision · deferred")
    assert any(r.startswith("… c5") for r in _rail(page)), _rail(page)
    assert page.get_by_role("button", name="Accept proposal").count() == 0


@pytest.mark.req("UI-5.1-01", "ARCH-13.8-01")
def test_an_own_call_sits_beside_the_proposal_with_its_literature_line(server, page) -> None:
    """When I call c6 myself ('Plasma cells') and save,
    Then the rail marks ✎ c6 Plasma cells, the decision reads overridden,
    And the proposal (B_Cells, c6's call in verdicts.csv) and my call are both
    assessed side by side, with the literature line on my call saying no corpus
    was built so it cannot settle this."""
    _open(page, server, "c6")
    page.get_by_text("Call it yourself").click()
    own = page.get_by_label("or your own label")
    own.wait_for()
    own.fill("Plasma cells")
    own.press("Enter")
    settle(page)
    page.get_by_role("button", name="Save call").click()
    see(page, "decision · overridden")

    assert any(r.startswith("✎ c6 Plasma cells") for r in _rail(page)), _rail(page)
    text = body(page)
    assert re.search(r"(?i)proposal B_Cells .*your call Plasma cells", text), text[-1500:]
    assert "literature check on your call: not looked up, so it cannot settle this" in text
