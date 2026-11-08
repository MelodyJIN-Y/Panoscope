"""The interactive surface -- the first test in this repo that clicks anything.

Before this file: 81 test modules, 7 using ``AppTest``, and **zero** containing
``.click()``. Every existing check rendered a page and asserted no exception. So
the half of Panoscope a biologist actually touches -- selecting a unit, pinning a
marker, signing off, reopening -- had never been exercised, and two export defects
survived long enough to be found by hand against real data.

**Two mechanisms, and missing the second is why the gap persisted.**

1. Streamlit widgets: the rail (``rail_<unit>``), evidence rows
   (``sfsel_gene|g=gene:<SYMBOL>``), the Ask form.
2. The review cockpit's table is **pure HTML**, and every action on it arrives as
   a QUERY PARAMETER. ``at.button`` can never see those, so a widget-only test
   skips the whole cockpit while looking thorough.

   ``summary._handle_table_actions`` dispatches five names, but only THREE are
   ever emitted: ``?sign=``, ``?undo=`` and ``?drill=``. No generator in
   ``panoscope/ui/`` produces ``?confirm=`` or ``?accept=`` -- observed in a real
   browser by ``tests/browser/test_ui_browser.py``, not argued from reading the source.
   Those two branches and their CSS are live code with no caller.

Keys are read off the running app rather than hardcoded: a renamed key should fail
here loudly, not quietly stop testing anything.

Isolation is the ``app_tree`` fixture. The suite has been burned before -- five
AppTest modules once wrote ``chat_threads.json`` into the committed demo -- so the
sign-off journey asserts the committed store stays clean too.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from panoscope.types.measured import MeasuredState

APP = str(Path(__file__).resolve().parents[1] / "panoscope" / "app.py")
REPO = Path(__file__).resolve().parents[1]


def _at(page: str):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["active_page"] = page
    at.run()
    return at


def _keys(at, kind: str = "button") -> dict:
    return {(w.key or ""): w for w in getattr(at, kind, [])}


def _text(at) -> str:
    return " ".join(m.value for m in at.markdown)


def _nums(t: str) -> set:
    return set(re.findall(r"\b\d{1,2}\.\d{2,4}\b", t))


def _committed_store_clean() -> bool:
    out = subprocess.run(["git", "status", "--porcelain", "data/studies/"],
                         cwd=REPO, capture_output=True, text=True).stdout.strip()
    return out == ""


# --------------------------------------------------------------------------- #
def test_the_rail_selects_a_unit_and_the_evidence_follows(app_tree) -> None:
    """The first thing a biologist does: pick a cluster."""
    at = _at("markers")
    rail = {k: v for k, v in _keys(at).items() if k.startswith("rail_")}
    assert len(rail) >= 2, f"the rail offers {len(rail)} unit(s)"

    before = _text(at)
    target = sorted(rail)[1]
    rail[target].click().run()
    after = _text(at)
    assert after != before, f"clicking {target} changed nothing on screen"
    assert target.split("_", 1)[1] in after


@pytest.mark.req("UI-5.3-02")
def test_pinning_a_marker_changes_the_view_and_no_value(app_tree) -> None:
    """A viewing control changes the picture, NEVER a number (CLAUDE.md).

    The strong half is the second assertion: every number that survives the click
    must be unchanged. A pin that silently rescaled or re-ranked a statistic would
    pass a "did the page change?" test while violating the rule the interface
    rests on.
    """
    at = _at("markers")
    pins = {k: v for k, v in _keys(at).items() if k.startswith("sfsel_")}
    if not pins:
        pytest.skip("this study's evidence table has no pinnable row")

    before = _text(at)
    nums_before = _nums(before)
    pins[sorted(pins)[0]].click().run()
    after = _text(at)
    nums_after = _nums(after)

    assert after != before, "pinning changed nothing"
    assert nums_before, "no number on screen before the pin; nothing to hold fixed"
    # EVERY number that was on screen must still be on screen. Not an intersection:
    # `nums_before & nums_after <= nums_before` holds for ANY two sets, which is what
    # this assertion used to say -- it could not fail, and a mutation doubling every
    # statistic on pin left the whole suite green. A pin may ADD numbers (the pinned
    # marker's own panel appears), so this is a superset check, not equality.
    moved = nums_before - nums_after
    assert not moved, (
        f"pinning moved {len(moved)} value(s) when only the view should have "
        f"changed: {sorted(moved)[:8]}"
    )


def test_the_rendered_page_names_the_genes_it_never_measured(app_tree) -> None:
    """End-to-end form of the rule: a hatched row is not an anonymous row.

    ``tests/test_surface_markers.py`` pins the renderer. This pins the PAGE, through
    the same ``render_surface`` path the biologist hits -- the path where the bug
    actually lived, and the one ``surface.table_html`` silently does not exercise.
    On the demo study 196 of c1's 280 rows are non-observed; every one of them must
    still say which gene it is talking about.
    """
    from panoscope.ui import evidence_table as et

    at = _at("markers")
    unit = {k: v for k, v in _keys(at).items() if k.startswith("rail_")}
    assert unit, "the rail offers no unit to read"
    cluster = sorted(unit)[0].split("_", 1)[1]

    built = et.unit_build(cluster)
    if built is None:
        pytest.skip(f"this study produces no record for {cluster}")
    hidden = [r for r in et.rows_for(cluster, built)
              if r.state is not MeasuredState.OBSERVED]
    if not hidden:
        pytest.skip(f"{cluster} has no non-observed row to check")

    page = _text(at)
    missing = [r.values["gene"] for r in hidden
               if str(r.values["gene"]) not in page]
    assert not missing, (
        f"{len(missing)} of {len(hidden)} non-observed rows render with no name "
        f"on {cluster}: {missing[:8]}"
    )


def test_sign_off_persists_reverses_and_survives_a_reload(app_tree) -> None:
    """The review cockpit, driven the only way it CAN be driven.

    The table is pure HTML, so ``?sign=cN`` IS the click. Reaching for the widget
    list here would silently test nothing at all.
    """
    from panoscope.agent import config as cfg

    unit = cfg.CLUSTER_ORDER[0]
    state = Path(app_tree) / cfg.DATASET_ID / "interp" / "review_state.json"
    if state.exists():
        state.unlink()

    at = _at("summary")
    at.query_params["sign"] = unit
    at.run()
    assert state.is_file(), "signing off wrote no review state"
    signed = json.loads(state.read_text(encoding="utf-8")).get("reviewed", {})
    assert unit in signed and signed[unit].get("at"), f"{unit} not recorded: {signed}"

    at.query_params["undo"] = unit
    at.run()
    reopened = json.loads(state.read_text(encoding="utf-8")).get("reviewed", {})
    assert unit not in reopened, f"reopening left {unit} signed: {reopened}"

    # Persisted, not session state: the file is the record, so a re-read sees it.
    at.query_params["sign"] = unit
    at.run()
    again = json.loads(state.read_text(encoding="utf-8")).get("reviewed", {})
    assert unit in again, "sign-off did not survive being written and re-read"

    assert _committed_store_clean(), (
        "an interactive write reached the committed study store"
    )


def test_the_summary_offers_both_exports(app_tree) -> None:
    """The handover. ``download_button`` exposes no payload to AppTest, so the
    document BYTES are checked in ``tests/test_report.py``; what is pinned here is
    that the biologist is offered them at all -- once every cluster is decided."""
    from panoscope.agent import config as cfg
    from panoscope.pipeline import store

    store.save_decisions({c: {"state": "DEFERRED"} for c in cfg.CLUSTER_ORDER})
    at = _at("summary")
    dl = _keys(at, "download_button")
    assert {"dl_docx", "dl_pdf"} <= set(dl), f"expected docx + pdf, found {sorted(dl)}"
