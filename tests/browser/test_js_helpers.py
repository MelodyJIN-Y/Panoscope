"""7.7: the inline JS helpers run. Both use ``st.components.v1.html``, which Streamlit
has deprecated ("will be removed after 2026-06-01"); an upgrade that drops it fails
here rather than silently. Auto-scroll is checked in test_chat.py."""
from __future__ import annotations

import pytest

from tests.browser.conftest import see, settle

pytestmark = pytest.mark.browser


@pytest.mark.req("UI-2.1-01")
def test_the_stale_page_hider_follows_a_tab_switch_and_then_lets_go(server, page) -> None:
    """The watcher in app._kill_cross_page_bleed is installed on the parent page, tracks
    the ``?page=`` it is on across a tab click, and never leaves the page hidden."""
    page.goto(f"{server}/?page=markers&cluster=c1")
    see(page, "never measured")
    settle(page)
    assert page.evaluate("() => window.__panoBleed && window.__panoBleed.last") == "markers"

    page.get_by_role("button", name="Summary", exact=True).click()
    see(page, "of 9 reviewed")
    settle(page)
    assert page.evaluate("() => window.__panoBleed.last") == "summary"
    page.wait_for_function(  # the 4 s hard cap: never hide forever
        "() => !document.documentElement.classList.contains('pano-killbleed')", timeout=6_000)
