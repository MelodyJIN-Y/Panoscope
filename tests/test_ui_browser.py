"""Layer 3: a REAL browser. The only layer that can see the cockpit at all.

``test_ui_crawl.py`` drives widgets through ``AppTest``, which renders NO HTML.
So three things were structurally invisible to every test in this repo:

1. **The review cockpit.** Its table is pure HTML and every action arrives as a
   query parameter, so ``test_interactive_journey.py`` FAKES the click by setting
   ``?sign=c1`` directly. That assumes the ``<a>`` exists. Nothing had ever
   checked. (It does -- but the same read said ``?confirm=`` and ``?accept=`` are
   dispatched by ``summary._handle_table_actions`` and emitted by nothing, which
   this now observes rather than argues.)
2. **JavaScript errors.** The first run of this found one: the chat auto-scroll
   emitted ``}}`` where it meant ``}`` -- a plain string segment that was never
   the f-string whose escaping it copied -- so the browser said "missing ) after
   argument list" and the feature had never once worked. ``except Exception``
   could not catch it; the failure was in the browser, not in Python.
3. **Layout.** Horizontal overflow at a narrower viewport.

Skipped unless ``playwright`` is importable, so a normal ``pytest`` run is
unaffected. To enable::

    pip install playwright && playwright install chromium

It starts its OWN Streamlit on a free ephemeral port and tears it down, so it
never collides with a server the biologist is already running.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api",
                    reason="pip install playwright && playwright install chromium")

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "panoscope" / "app.py"
PAGES = ("summary", "markers", "pathways", "contrasts")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory, _studies_copy):
    """Our own app on an ephemeral port, against a COPY of the study tree.

    The copy is not optional. This app PERSISTS as it renders -- opening a chat
    writes ``interp/chat_threads.json``, signing off writes
    ``interp/review_state.json`` -- and a browser run clicks sign-off for real.
    Pointed at ``data/`` it leaves runtime files in the committed store, which is
    what ``_studies_copy`` exists to prevent and what the tree guard catches.
    """
    port = _free_port()
    env = {**os.environ,
           "PANOSCOPE_DATA_ROOT": str(_studies_copy),
           "PANOSCOPE_DATASET": "xenium_hbreast_sample1",
           # Onboarding's "Analyze my data" writes the research profile; keep a
           # browser run away from the biologist's own context/ for the same
           # reason test_ui_crawl.py does.
           "PANOSCOPE_CONTEXT_DIR": str(tmp_path_factory.mktemp("ctx"))}
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(APP),
         "--server.port", str(port), "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                pytest.fail("the app exited while starting")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            pytest.fail(f"the app never listened on {port}")
        time.sleep(3)   # streamlit accepts connections before it can serve
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _settle(page, timeout: int = 45_000) -> None:
    """Streamlit streams over a websocket -- ``load`` fires long before content."""
    page.wait_for_selector("[data-testid='stAppViewContainer']", timeout=timeout)
    page.wait_for_function(
        "() => document.body.innerText.replace(/\\s+/g,' ').trim().length > 300",
        timeout=timeout)
    page.wait_for_timeout(2500)


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        b = pw.chromium.launch()
        yield b
        b.close()


# --------------------------------------------------------------------------- #
def test_no_javascript_errors_on_any_page(server, browser) -> None:
    """The invariant that caught a feature which had never worked.

    Inline scripts fail silently: Streamlit shows nothing, Python raises nothing,
    and the only witness is the browser console.
    """
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(f"{m.type}: {m.text}")
            if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))

    for p in PAGES:
        page.goto(f"{server}/?page={p}")
        _settle(page)

    real = [e for e in errors if "favicon" not in e.lower()]
    ctx.close()
    assert not real, "JavaScript errors:\n  " + "\n  ".join(real[:8])


def test_the_cockpit_emits_the_links_the_dispatcher_handles(server, browser) -> None:
    """The gap the journey could not close: do the anchors EXIST?

    ``summary._handle_table_actions`` dispatches five actions. Three are emitted
    as links; ``confirm`` and ``accept`` are reachable only through real
    ``st.button``s, so their query-param branches are dead. Asserted as it IS,
    so that a change in either direction shows up here.
    """
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(f"{server}/?page=summary")
    _settle(page)

    hrefs = page.evaluate(
        """() => Array.from(document.querySelectorAll('a[href]'))
             .map(a => a.getAttribute('href'))
             .filter(h => /[?&](sign|undo|drill|confirm|accept)=/.test(h))""")
    kinds = {m.group(1) for h in hrefs
             for m in [re.search(r"[?&](sign|undo|drill|confirm|accept)=", h)] if m}
    ctx.close()

    assert "sign" in kinds, f"no ?sign= link on an unsigned board: {hrefs[:5]}"
    assert "drill" in kinds, f"no ?drill= link: {hrefs[:5]}"
    # undo appears only once something IS signed -- see the next test.
    assert "confirm" not in kinds and "accept" not in kinds, (
        "confirm/accept are now emitted as links; the dispatcher branches in "
        "summary._handle_table_actions are no longer dead -- update this.")


def test_signing_off_in_a_real_browser_changes_the_board(server, browser) -> None:
    """Click the anchor for real, then reopen. Never a fabricated query param."""
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto(f"{server}/?page=summary")
    _settle(page)

    sign = page.query_selector('a[href*="sign="]')
    assert sign is not None, "no sign-off anchor to click"
    box = sign.bounding_box()
    assert box and box["width"] >= 4 and box["height"] >= 4, (
        f"the sign-off link has no clickable area: {box}")

    before = page.inner_text("body")
    sign.click()
    _settle(page)
    assert page.inner_text("body") != before, "clicking sign-off changed nothing"

    undo = page.query_selector('a[href*="undo="]')
    assert undo is not None, "signing produced no reopen link"
    undo.click()
    _settle(page)
    assert page.query_selector('a[href*="sign="]') is not None, (
        "after reopening there is no way to sign off again")
    ctx.close()


@pytest.mark.parametrize("gene", ["CD8B", "NOTAGENE", "ADIPOQ", "ERBB2,CD8B,NOTAGENE"])
def test_a_gene_in_the_url_never_takes_the_page_down(server, browser, gene: str) -> None:
    """``?m=`` is restored from the URL with no validation, and one bad value used
    to abort the whole script.

    Streamlit derives a chart's element id from its params, and the feature UMAP's
    honest fallback re-drew the cluster UMAP with the SAME figure, layout, config
    and conditionality as Row 1. So any gene with no exported expression column
    raised StreamlitDuplicateElementId -- which stops the script, losing the dot
    plot and the entire chat pane below it. The documented fallback had never once
    been able to execute.

    CD8B is off-panel, ADIPOQ is on-panel with no usable signal, NOTAGENE does not
    exist, and the list mixes a real gene with both.
    """
    ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = ctx.new_page()
    page.goto(f"{server}/?page=markers&cluster=c1&m={gene}")
    _settle(page)
    state = page.evaluate("""() => ({
      exc: [...document.querySelectorAll("[data-testid='stException']")]
             .map(e => e.innerText.split('\\n').slice(0, 2).join(' ')),
      charts: document.querySelectorAll('.js-plotly-plot').length,
      ask: !!document.querySelector("textarea, input[type='text']"),
    })""")
    ctx.close()

    assert not state["exc"], f"?m={gene} raised: {state['exc']}"
    assert state["charts"] > 0, f"?m={gene} rendered no figure at all"
    # The chat pane sits BELOW the figures, so it is the thing an aborted script
    # silently removes -- and its absence is how this defect stayed invisible.
    assert state["ask"], f"?m={gene} lost the chat pane: the script stopped early"


@pytest.mark.parametrize("width", [1440, 1024])
def test_the_column_headings_sit_over_their_columns(server, browser, width: int) -> None:
    """Measured geometry, because every cheaper check here has been wrong.

    Three attempts at this bug shipped. The first dropped the gene name from
    non-observed rows. The second gave the header one fewer track than the rows,
    putting GLM_COEF over the gene names. The third emitted matching track COUNTS
    whose identical declaration still resolved to different WIDTHS, because a bare
    ``1fr`` is ``minmax(auto, 1fr)`` and never shrinks below its own content -- so
    "observed" rendered under the SPATIAL SPECIFICITY heading with no overlap of
    its own, and the cited biology under CAVEATS.

    Each of those passed a test comparing the ``grid-template-columns`` STRING. The
    strings were identical every time; the pixels were not. Only a browser can tell
    the difference, so this asks one.

    Tolerance is 12px. A full-width row matches the header exactly; the pin-button
    row carries a structural 3-8px offset because ``st.columns`` gaps at 16px while
    the grid gaps at 10px. 12px admits that and still catches a one-column slip,
    which is never smaller than a whole track (~50px).
    """
    ctx = browser.new_context(viewport={"width": width, "height": 1000})
    page = ctx.new_page()
    page.goto(f"{server}/?page=markers")
    _settle(page)
    geom = page.evaluate("""() => {
      const head = document.querySelector('.pano-sf-head');
      if (!head) return {err: 'no column header rendered'};
      const n = head.children.length;
      const left = el => Math.round(el.getBoundingClientRect().left * 10) / 10;
      const labels = [...head.children].map(c => c.innerText.trim().split('\\n')[0]);
      const heads = [...head.children].map(left);
      const rows = [...document.querySelectorAll('.pano-sf-row')];
      const full = rows.find(r => r.children.length === n);
      const inner = rows.find(r => r.children.length === n - 1);
      // Worst overflow per column, over the header and every full-width row.
      const box = el => ({w: Math.round(el.clientWidth), sw: Math.round(el.scrollWidth)});
      const cells = [...head.children].map(box);
      for (const r of rows) {
        if (r.children.length !== n) continue;
        [...r.children].forEach((c, i) => {
          const b = box(c);
          if (b.sw - b.w > cells[i].sw - cells[i].w) cells[i] = b;
        });
      }
      return {labels, heads, cells, nrows: rows.length,
              full: full ? [...full.children].map(left) : null,
              inner: inner ? [...inner.children].map(left) : null};
    }""")
    ctx.close()

    assert not geom.get("err"), geom.get("err")
    assert geom["nrows"] > 0, "the markers table rendered no rows"

    bad: list[str] = []
    if geom["full"]:
        for label, h, c in zip(geom["labels"], geom["heads"], geom["full"]):
            if abs(c - h) > 12:
                bad.append(f"{label!r}: heading at {h}, content at {c} ({c - h:+.1f}px)")
    if geom["inner"]:
        for label, h, c in zip(geom["labels"][1:], geom["heads"][1:], geom["inner"]):
            if abs(c - h) > 12:
                bad.append(f"{label!r} (pin row): heading {h}, content {c} ({c - h:+.1f}px)")

    assert not bad, (
        f"column content is not under its heading at {width}px:\n  " + "\n  ".join(bad)
    )

    # ...and no cell spills INTO the next column. Checking left edges alone was not
    # enough: the first alignment fix made every track honour its fraction, which
    # aligned the origins and then let the content overflow them -- headings drawn
    # on top of the next heading, measured at 13-36px. A column that starts in the
    # right place and paints over its neighbour is not a fixed column.
    spill = [f"{label!r}: content {c['sw']}px in a {c['w']}px track"
             for label, c in zip(geom["labels"], geom["cells"])
             if c["sw"] - c["w"] > 1]
    assert not spill, (
        f"cell content overflows its column at {width}px:\n  " + "\n  ".join(spill)
    )


@pytest.mark.parametrize("width", [1440, 1024])
def test_no_horizontal_overflow(server, browser, width: int, tmp_path) -> None:
    """A page that scrolls sideways is broken in a way no assertion on text sees."""
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    page = ctx.new_page()
    bad: list[str] = []
    for p in ("summary", "markers"):
        page.goto(f"{server}/?page={p}")
        _settle(page)
        scroll, client = page.evaluate(
            "() => [document.documentElement.scrollWidth,"
            " document.documentElement.clientWidth]")
        page.screenshot(path=str(tmp_path / f"{p}_{width}.png"), full_page=True)
        if scroll > client + 2:
            bad.append(f"{p}: content {scroll}px > viewport {client}px")
    ctx.close()
    assert not bad, f"horizontal overflow at {width}px:\n  " + "\n  ".join(bad)


@pytest.mark.parametrize("width", [1440, 1024])
def test_no_inline_element_paints_outside_its_cell(server, browser, width: int) -> None:
    """The overflow the cell itself cannot report.

    ``test_the_column_headings_sit_over_their_columns`` checks each cell's
    ``scrollWidth`` against its ``clientWidth``, which is how the hatch pill was
    caught. It does not catch the same defect one column over: an inline-block
    whose max-content width exceeds the track PAINTS past the cell without
    extending its scroll box, so the cell reports no overflow at all and the
    cheaper check passes. ``.pano-sf-chip`` spilled "low spatial specificity"
    7.5px out of a 64px CAVEATS cell at 1024 while every existing geometry
    assertion stayed green.

    So this measures the CHILD against its cell, not the cell against itself.
    """
    ctx = browser.new_context(viewport={"width": width, "height": 1000})
    page = ctx.new_page()
    page.goto(f"{server}/?page=markers")
    _settle(page)
    spill = page.evaluate("""() => {
      const bad = [];
      const sels = ['.pano-sf-chip', '.pano-sf-prose', '.pano-sf-hatch',
                    '.pano-sf-cite', '.pano-sf-num', '.pano-sf-text',
                    '.pano-sf-dir', '.pano-sf-glyph'];
      for (const sel of sels) {
        for (const el of document.querySelectorAll(sel)) {
          const cell = el.closest('.pano-sf-cell');
          if (!cell) continue;
          const e = el.getBoundingClientRect(), c = cell.getBoundingClientRect();
          const over = Math.round((e.right - c.right) * 10) / 10;
          if (over > 0.5) bad.push(
            `${sel} +${over}px past a ${Math.round(c.width)}px cell: `
            + JSON.stringify(el.innerText.slice(0, 32)));
        }
      }
      return [...new Set(bad)];
    }""")
    ctx.close()
    assert not spill, (
        f"{len(spill)} element(s) paint outside their cell at {width}px:\n  "
        + "\n  ".join(spill))
