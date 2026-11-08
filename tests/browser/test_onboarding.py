"""7.6: the onboarding error banners, typed into a real page (interface §6.4 intake).

A path that is not a folder is refused in words; a recognised folder whose tissue is
not declared shows what blocks it and offers no Build button."""
from __future__ import annotations

import pytest

from tests.browser.conftest import see, settle
from tests.journeys.flow import DEMO_INPUTS, folder_with

pytestmark = pytest.mark.browser


def _paste(page, path) -> None:
    box = page.get_by_label("Results folder")
    box.fill(str(path))
    box.press("Enter")


@pytest.mark.req("ARCH-6.4-01", "ARCH-6.3-02")
def test_a_non_folder_is_refused_and_an_undeclared_tissue_blocks_the_build(
        server, page, tmp_path) -> None:
    page.goto(f"{server}/?page=upload")
    see(page, "Bring your own results")
    settle(page)

    missing = tmp_path / "no_such_folder"
    _paste(page, missing)
    see(page, f"{missing} is not a folder.")
    assert page.get_by_role("alert").filter(has_text="is not a folder").count() == 1

    folder = folder_with(tmp_path, "my_breast_run", {
        "markers_top.csv": DEMO_INPUTS / "markers_top.csv",
        "panel.parquet": DEMO_INPUTS / "panel.parquet"})
    _paste(page, folder)
    see(page, "Platform \\(blocks the study until declared\\)")
    page.locator("[data-testid='stSelectbox']").filter(has_text="Platform").locator("input").click()
    page.get_by_role("option", name="xenium").click()
    see(page, "Blocked until declared: tissue")
    settle(page)
    assert page.get_by_role("alert").filter(has_text="Blocked until declared: tissue").count() == 1
    assert page.get_by_role("button", name="Build  →").count() == 0
