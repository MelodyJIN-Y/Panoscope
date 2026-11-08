"""7.3: hover and click-back, which only a browser can do (AppTest reads a ``title``
string but cannot hover, and cannot click HTML).

Every status glyph says its state in words and gives the frame on hover; every
figure caption gives its full conditionality record on hover; a sentence's
``data-claim`` anchor should open its claim (CLAUDE.md "Click-back")."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.browser.conftest import body, see, settle
from tests.conftest import DATASETS_ROOT, DEMO_ID

pytestmark = pytest.mark.browser

# The four-state vocabulary (CLAUDE.md "Measurement status"); UNDERPOWERED does not
# occur on a marker unit of the demo.
_WORDS = {"observed": "observed", "tested_absent": "measured, not here",
          "never_measured": "never measured"}


def _hover(page, loc) -> dict:
    loc.scroll_into_view_if_needed()
    loc.hover()
    return loc.evaluate("e => ({hovered: e.matches(':hover'), title: e.title,"
                        " text: e.innerText.trim()})")


@pytest.fixture
def c1(server, page):
    page.set_viewport_size({"width": 1440, "height": 1000})
    page.goto(f"{server}/?page=markers&cluster=c1")
    see(page, "never measured")
    settle(page)
    return page


@pytest.mark.req("UI-2.3-01", "ARCH-13.4-01")
@pytest.mark.parametrize("state", sorted(_WORDS))
def test_hovering_a_status_glyph_reaches_it_and_gives_the_reason(c1, state) -> None:
    got = _hover(c1, c1.locator(f".pano-sf-glyph.s-{state}").first)
    assert got["hovered"], f"the {state} glyph is covered: hover does not reach it"
    assert _WORDS[state] in got["text"], got
    assert got["title"].startswith(f"{state} in frame "), got["title"]


@pytest.mark.req("ARCH-13.2-01", "ARCH-13.2-02")
def test_every_figure_carries_a_caption_whose_hover_is_the_full_record(c1) -> None:
    """One conditionality caption per drawn figure, and its hover names what is drawn,
    the denominator (the panel size read from the study's panel file, never a
    constant), the transform, what it is not, and its provenance."""
    panel = pd.read_parquet(DATASETS_ROOT / DEMO_ID / "inputs" / "panel.parquet")["gene"].nunique()
    figures = c1.locator(".js-plotly-plot").count()
    captions = c1.locator(".plegend")
    assert figures > 0 and captions.count() == figures, (figures, captions.count())
    for i in range(figures):
        got = _hover(c1, captions.nth(i))
        assert got["hovered"], f"caption {i} is covered: hover does not reach it"
        for field in ("drawn:", "denominator:", "transform:", "not this:", "provenance:"):
            assert field in got["title"], (field, got["title"])
        assert f"{panel}-gene targeted panel" in got["title"], got["title"]


@pytest.mark.req("ARCH-13.5-01")
def test_every_claim_anchor_is_on_screen(c1) -> None:
    """The half of click-back that is built: each ``data-claim`` anchor is a real,
    laid-out element (none hidden or zero-sized), so there is something to click. A
    gene row's anchor is an empty marker span, so its row is what must be laid out."""
    boxes = c1.evaluate("""() => [...document.querySelectorAll('[data-claim]')]
        .map(e => { const r = (e.innerText.trim() ? e : e.parentElement).getBoundingClientRect();
                    return {claim: e.dataset.claim, w: r.width, h: r.height}; })""")
    assert len(boxes) > 50, len(boxes)
    flat = [b["claim"] for b in boxes if b["w"] < 1 or b["h"] < 1]
    assert not flat, f"{len(flat)} claim anchors take no space: {flat[:5]}"


@pytest.mark.req("ARCH-13.5-01", gap=True)
def test_clicking_a_sentence_opens_its_claim_with_authority_and_conditions(c1) -> None:
    """Not built: ``data-claim`` is an attribute with no handler, popover or hover."""
    before = body(c1).lower().count("authority")
    c1.locator(".pano-ev-conc[data-claim]").first.click()
    settle(c1)
    assert body(c1).lower().count("authority") > before, "the click opened nothing"
