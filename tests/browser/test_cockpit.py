"""7.1: the Summary review table, clicked for real (its controls are HTML links,
which AppTest cannot click). A reviewed mark is the biologist's local state on disk,
so it must survive a hard reload and the browser's back/forward buttons."""
from __future__ import annotations

import re

import pytest

from tests.browser.conftest import body, see, settle

pytestmark = pytest.mark.browser


def _progress(page) -> str:
    m = re.search(r"(\d+) of (\d+) reviewed", body(page))
    assert m, "the board's lead line names no review progress"
    return m.group(0)


def _reviewed(page, cluster: str) -> bool:
    return page.locator(f'a[href*="undo={cluster}"]').count() == 1


@pytest.mark.req("UI-5.3-07")
def test_a_reviewed_mark_survives_reload_and_back_forward(server, page) -> None:
    """Given the board with nothing reviewed,
    When I tick c1, reload, then step back and forward through the history,
    Then the board reads 1 of 9 reviewed with c1 ticked every time,
    And unticking c1 takes it back to 0 of 9, which a reload also keeps."""
    page.goto(f"{server}/?page=summary")
    see(page, "0 of 9 reviewed")

    page.locator('a[href*="sign=c1"]').click()
    see(page, "1 of 9 reviewed")
    assert (_progress(page), _reviewed(page, "c1")) == ("1 of 9 reviewed", True)

    page.reload()
    settle(page)
    assert (_progress(page), _reviewed(page, "c1")) == ("1 of 9 reviewed", True)

    for step in (page.go_back, page.go_back, page.go_forward, page.go_forward):
        step()
        settle(page)
        # Wherever history lands, the board shows what is on disk, never a replay.
        assert "1 of 9 reviewed" in body(page), f"after {step.__name__}: {page.url}"

    page.goto(f"{server}/?page=summary")
    see(page, "1 of 9 reviewed")
    page.locator('a[href*="undo=c1"]').click()
    see(page, "0 of 9 reviewed")
    assert (_progress(page), _reviewed(page, "c1")) == ("0 of 9 reviewed", False)
    page.reload()
    settle(page)
    assert (_progress(page), _reviewed(page, "c1")) == ("0 of 9 reviewed", False)


@pytest.mark.req("UI-2.1-01")
def test_a_cluster_id_drills_in_and_all_calls_comes_back(server, page) -> None:
    """Clicking c3's id opens its evidence and write-up; '← All calls' returns to the
    board, and the URL never keeps the one-shot ``drill`` action (a reload of it would
    replay the click)."""
    page.goto(f"{server}/?page=summary")
    see(page, "of 9 reviewed")
    page.locator('a.cid[href*="drill=c3"]').click()
    see(page, "All calls")
    text = body(page)
    # c3's call and band from the study's verdicts.csv.
    assert "c3 Macrophages Very High" in text and "All calls" in text, text[:400]
    assert "drill=" not in page.url, page.url

    page.get_by_role("button", name="← All calls").click()
    see(page, r"\d+ of 9 reviewed")
