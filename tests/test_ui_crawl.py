"""The interactive surface, crawled and fuzzed -- what nobody thought to assert.

``test_interactive_journey.py`` clicks the widgets I chose and asserts the 31
things I thought of. By construction it cannot find breakage I did not predict.
This file does the complement: drive EVERY widget, and after each interaction
check a handful of rules that must hold no matter what was clicked.

**The invariants are the artifact, not the crawler.** The first version of this
reported 311 findings and every one was the regex, not the app: ``\\bNone\\b``
matched ``flex: none`` in inline CSS, and the control-probe rule fired on the
plot's own conditionality note ("negcontrolprobe/.../antisense probes removed"),
which CLAUDE.md REQUIRES. An invariant that fires on correct behaviour is worse
than no invariant, because it teaches you to skim the report. Every rule below
survived that triage, which is why they are narrow and why the HTML is stripped
before any of them reads the text.

Two layers, because each sees what the other cannot:

* **crawl** -- each widget from a clean page. Finds breakage in one click.
* **sequence** -- seeded chains in ONE session, where ``session_state`` persists.
  Finds order-dependent breakage (pin -> switch unit -> pin again) that no
  single-click test can reach.

Neither renders HTML, so neither can see layout, CSS, or whether the review
cockpit's ``?sign=cN`` links exist at all -- that table is pure HTML and the
journey fakes the click with a query param. That gap needs a browser and is
not covered here.
"""

from __future__ import annotations

import os
import random
import re

import pytest

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "panoscope", "app.py")

KINDS = ("button", "checkbox", "selectbox", "radio", "slider", "multiselect",
         "toggle", "text_input", "text_area", "number_input", "select_slider")

#: Interactions per page in the default suite. The marker pane carries ~100
#: widgets, ~90 of them homogeneous ``sfsel_gene|...`` evidence rows, so a sample
#: buys nearly the same coverage for a fraction of the time. The cap is REPORTED
#: rather than silent -- a bounded sweep that reads as exhaustive is how a gap
#: hides. Set PANOSCOPE_UI_CRAWL_FULL=1 for every widget.
MAX_PER_PAGE = 10

TAG = re.compile(r"<[^>]*>")
STYLE = re.compile(r"<style[^>]*>.*?</style>", re.S | re.I)
#: Case-SENSITIVE: ``flex: none`` is CSS, ``None`` is a leaked Python value.
LEAK = re.compile(r"\bNone\b|\bnan\b|\bNaN\b|Traceback|object at 0x|<built-in|\{\}")
CONTROL = re.compile(r"NegControl|Blank ?Codeword|Unassigned ?Codeword", re.I)
VIEW_CONTROL = re.compile(r"bin|view|toggle|sample|scale|zoom|size|mode|layer")
NUM = re.compile(r"-?\d+\.\d{2,4}")
#: "None of the resolved papers support..." is English, not a leaked value.
PROSE_NONE = re.compile(r"\bNone of\b")


@pytest.fixture(autouse=True)
def _isolated_context(tmp_path_factory, monkeypatch):
    """Keep the biologist's own ``context/`` out of reach of a crawl.

    The onboarding page's "Analyze my data" button calls ``agent/profile.save()``,
    and this file clicks every button there is. ``profile`` resolves its path from
    ``PANOSCOPE_CONTEXT_DIR`` or the CWD, so without this a suite run from the repo
    root would overwrite the real research profile with whatever the crawl typed --
    a test that silently destroys user data to prove the UI works.
    """
    monkeypatch.setenv("PANOSCOPE_CONTEXT_DIR",
                       str(tmp_path_factory.mktemp("context")))


def _at(page: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["active_page"] = page
    at.run()
    return at


def _plain(at) -> str:
    """Rendered text as a READER sees it -- style blocks and tags removed.

    Stripping is load-bearing: every false positive in the first run came from a
    rule reading raw HTML that no biologist ever sees.
    """
    raw = " ".join(m.value for m in at.markdown)
    for kind in ("header", "subheader", "title", "caption", "info", "warning",
                 "success", "error"):
        raw += " " + " ".join(str(getattr(w, "value", "") or "")
                              for w in getattr(at, kind, []))
    return re.sub(r"\s+", " ", TAG.sub(" ", STYLE.sub(" ", raw)))


def _widgets(at) -> list[tuple[str, object]]:
    return [(k, w) for k in KINDS for w in getattr(at, k, [])]


def _drive(kind: str, w) -> bool:
    """Interact the way a user would. False when the widget offers no change."""
    if getattr(w, "disabled", False):   # a browser user cannot press it either
        return False
    if kind == "button":
        w.click(); return True
    if kind in ("checkbox", "toggle"):
        w.set_value(not w.value); return True
    if kind in ("selectbox", "radio", "select_slider"):
        opts = list(getattr(w, "options", []) or [])
        nxt = next((o for o in opts if o != getattr(w, "value", None)), None)
        if nxt is None:
            return False
        w.set_value(nxt); return True
    if kind == "multiselect":
        opts = list(getattr(w, "options", []) or [])
        if not opts:
            return False
        w.set_value(opts[:1]); return True
    if kind == "slider":
        try:
            if w.min == w.max:
                return False
            w.set_value(w.max if w.value != w.max else w.min); return True
        except Exception:  # noqa: BLE001 - a slider we cannot drive is not a defect
            return False
    if kind == "number_input":
        try:
            w.set_value((w.value or 0) + 1); return True
        except Exception:  # noqa: BLE001
            return False
    if kind in ("text_input", "text_area"):
        w.set_value("which markers support this call?"); return True
    return False


def _violations(at, key: str, before: str) -> list[str]:
    """Every rule that must hold after ANY interaction. Empty list means clean."""
    out: list[str] = []
    if at.exception:
        return [f"raised {str(at.exception[0].value)[:200]}"]
    txt = _plain(at)
    if len(txt.strip()) < 40:
        return [f"page went blank ({len(txt.strip())} chars rendered)"]

    # A control probe may be NAMED in a conditionality note -- "negcontrolprobe/
    # ...probes removed" is the plot declaring what it excluded, which CLAUDE.md
    # requires. It may never be a CLICKABLE EVIDENCE ROW, and the widget keys are
    # where that distinction actually shows.
    for _k, w in _widgets(at):
        if CONTROL.search(getattr(w, "key", None) or ""):
            out.append(f"control probe is a clickable evidence row: {w.key}")

    for m in LEAK.finditer(txt):
        if m.group(0) == "None" and PROSE_NONE.match(txt, m.start()):
            continue
        out.append(f"leaked {m.group(0)!r}: ...{txt[max(0, m.start() - 50):m.end() + 50]}...")
        break

    # CLAUDE.md: a viewing control changes the picture, never a value.
    if VIEW_CONTROL.search(key):
        b, a = set(NUM.findall(before)), set(NUM.findall(txt))
        gone = b - a
        if gone and (b & a):
            out.append(f"a viewing control moved values: {sorted(gone)[:4]} vanished")

    # The headline rule: statistic is None IFF never measured. A never-measured
    # item rendered WITH a number is that invariant failing where it is visible.
    for m in re.finditer(r"never measured[^.]{0,120}", txt, re.I):
        if re.search(r"\b\d+\.\d+\b", m.group(0)):
            out.append(f"a never-measured item carries a statistic: {m.group(0)[:110]}")
            break
    return out


def _pages() -> list[str]:
    from panoscope import workflows
    from panoscope.ui import nav

    return (list(nav.ONBOARDING_PAGES) + [nav.SUMMARY_PAGE]
            + sorted(w.uispec().workflow_id for w in workflows.WORKFLOWS.values()))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", _pages())
def test_every_widget_holds_the_invariants(page: str, app_tree, capsys) -> None:
    """Click each widget from a clean page; the rules must survive all of them."""
    base = _at(page)
    assert not _violations(base, "", ""), f"{page} is broken before any click"
    before = _plain(base)

    ws = _widgets(base)
    full = os.getenv("PANOSCOPE_UI_CRAWL_FULL") == "1"
    chosen = ws if full else ws[:MAX_PER_PAGE]
    if len(chosen) < len(ws):
        with capsys.disabled():
            print(f"\n  [{page}] {len(chosen)} of {len(ws)} widgets "
                  f"(PANOSCOPE_UI_CRAWL_FULL=1 for all)")

    broken: list[str] = []
    for kind, w in chosen:
        key = getattr(w, "key", None) or f"<{kind}>"
        at = _at(page)
        target = next((x for k2, x in _widgets(at)
                       if k2 == kind and (getattr(x, "key", None) or f"<{k2}>") == key),
                      None)
        if target is None:
            continue
        try:
            if not _drive(kind, target):
                continue
            at.run()
        except Exception as e:  # noqa: BLE001 - an interaction that throws IS the finding
            broken.append(f"{key}: {type(e).__name__}: {e}")
            continue
        broken += [f"{key}: {v}" for v in _violations(at, key, before)]

    assert not broken, f"{page}:\n  " + "\n  ".join(broken)


@pytest.mark.parametrize("page", ["summary", "markers", "pathways"])
def test_click_sequences_hold_the_invariants(page: str, app_tree) -> None:
    """Seeded chains in ONE session, where state persists between clicks.

    Streamlit keeps ``session_state`` across ``.run()`` within one AppTest
    instance, so a chain here is what a real session is -- and order is where UIs
    actually break. Seeded so a failure replays exactly.
    """
    broken: list[str] = []
    for seed in range(3):
        rng = random.Random(seed)
        at = _at(page)
        before = _plain(at)
        path: list[str] = []
        for _ in range(6):
            ws = [(k, w) for k, w in _widgets(at) if getattr(w, "key", None)]
            if not ws:
                break
            kind, w = rng.choice(ws)
            key = w.key
            try:
                if not _drive(kind, w):
                    continue
                at.run()
            except Exception as e:  # noqa: BLE001
                broken.append(f"seed={seed} {path + [key]}: {type(e).__name__}: {e}")
                break
            path.append(key)
            bad = _violations(at, key, before)
            if bad:
                broken.append(f"seed={seed} chain={path}: {bad[0]}")
                break
    assert not broken, f"{page}:\n  " + "\n  ".join(broken)


def test_a_repeated_gene_in_the_url_does_not_abort_the_page(app_tree):
    """A selection is a SET: a gene chosen twice is the same gene.

    ``?m=ERBB2,ERBB2`` was stored verbatim, the small-multiples emitted two
    charts with the same element key, and Streamlit raised
    ``StreamlitDuplicateElementKey`` -- which aborts the script, so everything
    below it, the chat pane included, silently disappeared. Reproduced:
    ``key='pano_density_gene_ERBB2_25'``.
    """
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.query_params["page"] = "markers"
    at.query_params["cluster"] = "c1"
    at.query_params["m"] = "ERBB2,ERBB2,KRT7"
    at.run()

    assert not at.exception, [str(e.value)[:160] for e in at.exception]
    # ...and the page is whole, not truncated at the abort point.
    assert len(at.markdown) > 300, len(at.markdown)
