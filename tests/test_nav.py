"""The tab strip is the registry's, not a list in ``app.py``.

Both content surfaces already rendered from their declarations; the navigation still
asserted which workflows existed, so a fourth registered workflow got a declared
surface nobody could reach. These tests pin the half that closes it:

* the strip's pages, titles and order are read off ``WORKFLOWS``;
* a workflow registered **in this process** gets a tab with no edit to ``app.py``
  -- that is the whole claim, so it is asserted against app.py's source text, not
  against a mock;
* a registered workflow with no result in this study keeps its tab and its page
  states the absence, because a hidden tab and an unsupported question look
  identical to a biologist and are not the same fact;
* the ``?page=`` vocabulary is unchanged -- the surface ids already spell it.
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

import panoscope
from panoscope import workflows
from panoscope.types.outputs import TaskKind
from panoscope.types.uispec import degraded_spec
from panoscope.ui import nav


@pytest.fixture(autouse=True)
def _render_against_a_copy(app_tree):
    """Every AppTest in this module renders a real study, and the app writes as it
    renders (``chat_threads.json``, ``review_state.json``). Render a copy, so the
    committed tree stays the read-only input the suite says it is."""

APP_SRC = (pathlib.Path(panoscope.__file__).parent / "app.py").read_text()

# The real wiring, restated. These tests must not import app.py: it is a Streamlit
# script whose module body runs the app.
WIRED = {"markers": (lambda: None, lambda: True),
         "pathways": (lambda: None, lambda: True)}


def _strip(wired=None):
    return nav.tabs(WIRED if wired is None else wired, lambda: None)


def _workflow_for(page: str):
    return next(w for w in workflows.WORKFLOWS.values()
                if w.uispec().workflow_id == page)


@pytest.fixture
def isolated_workflows():
    """Snapshot and restore the registry, whatever the test registered into it.

    A leaked stub would make a later duplicate-registration test raise for the wrong
    reason -- the same hazard ``test_registries.isolated_registries`` names.
    """
    snapshot = dict(workflows.WORKFLOWS)
    try:
        yield
    finally:
        workflows.WORKFLOWS.clear()
        workflows.WORKFLOWS.update(snapshot)


@dataclasses.dataclass(frozen=True)
class _Spike:
    """A throwaway workflow carrying only the four attributes navigation reads."""

    name: str
    surface_id: str
    title: str
    task_kind: TaskKind = TaskKind.COMPARISON
    question: str = "Does this spike reach a surface?"

    def uispec(self):
        return dataclasses.replace(
            degraded_spec(self.title, ("x",)),
            workflow_id=self.surface_id, title=self.title,
        )


# --------------------------------------------------------------------------- #
# The strip comes from the registry
# --------------------------------------------------------------------------- #
def test_every_registered_workflow_gets_a_tab():
    pages = [t.page for t in _strip()]
    for wf in workflows.WORKFLOWS.values():
        assert wf.uispec().workflow_id in pages


def test_title_and_route_are_the_declaration_not_a_label_in_the_app():
    """The tab reads what the workflow called its surface and routes on the id it
    declared, so renaming a surface is a one-line change inside the workflow."""
    by_page = {t.page: t for t in _strip()}
    for wf in workflows.WORKFLOWS.values():
        spec = wf.uispec()
        assert by_page[spec.workflow_id].title == spec.title
        assert by_page[spec.workflow_id].question == wf.question


def test_summary_leads_and_the_rest_sit_in_task_kind_order():
    """Assignment, then characterisation, then comparison: *what is this*, *what is
    running*, *does this make sense*. ``TaskKind`` is frozen at three, so the order
    is total and no new workflow needs a line added anywhere."""
    strip = _strip()
    assert strip[0].page == nav.SUMMARY_PAGE
    kinds = [nav._TASK_ORDER[_workflow_for(t.page).task_kind] for t in strip[1:]]
    assert kinds == sorted(kinds)
    # and it is NOT merely registry order, which is sorted-by-name (enrichment first)
    assert [t.page for t in strip[1:]][:2] == ["markers", "pathways"]


def test_the_app_does_not_list_the_tabs_any_more():
    """The defect this closes is a hardcoded strip, so it is asserted against the
    source: app.py names a surface id in its wiring, but it no longer builds the
    strip out of one button call per tab."""
    assert "nav.tabs(" in APP_SRC
    assert '"Marker genes"' not in APP_SRC
    assert 'key="nav_summary"' not in APP_SRC
    assert "_PAGE_PATHWAYS" not in APP_SRC and "_PAGE_SUMMARY" not in APP_SRC


# --------------------------------------------------------------------------- #
# The claim: a fourth workflow reaches the strip with no edit to app.py
# --------------------------------------------------------------------------- #
def test_registering_a_workflow_adds_a_tab_with_no_edit_to_the_app(isolated_workflows):
    assert "spikes" not in [t.page for t in _strip()]
    workflows.register(_Spike(name="spike-interpretation", surface_id="spikes",
                              title="Spikes"))
    after = _strip()
    spike = nav.find(after, "spikes")
    assert spike is not None and spike.title == "Spikes"
    assert "spikes" in nav.pages(after)          # and ?page=spikes now resolves
    # the whole claim: nothing in app.py knows this workflow exists
    assert "spike" not in APP_SRC.lower()


def test_a_comparison_workflow_sorts_after_the_others(isolated_workflows):
    workflows.register(_Spike(name="spike-interpretation", surface_id="spikes",
                              title="Spikes", task_kind=TaskKind.COMPARISON))
    assert [t.page for t in _strip()][-1] == "spikes"


# --------------------------------------------------------------------------- #
# Registered, but no result in this study
# --------------------------------------------------------------------------- #
def test_a_workflow_with_no_result_keeps_its_tab_and_is_marked(isolated_workflows):
    """Hiding it would make *this study has no spike result* indistinguishable from
    *Panoscope cannot interpret spikes*. The second is a much bigger claim."""
    workflows.register(_Spike(name="spike-interpretation", surface_id="spikes",
                              title="Spikes"))
    spike = nav.find(_strip(), "spikes")
    assert spike is not None                 # the tab is there
    assert spike.available() is False        # marked, not silently empty
    assert spike.render is None


def test_a_wired_surface_whose_study_has_no_data_is_unavailable_not_broken():
    strip = _strip({"markers": (lambda: None, lambda: False)})
    assert nav.find(strip, "markers").available() is False


def test_an_availability_check_that_raises_is_no_data_not_a_traceback():
    """A missing file empties one tab; it never breaks the nav bar on every page."""
    def boom() -> bool:
        raise FileNotFoundError("enrichment.parquet")

    strip = _strip({"markers": (lambda: None, boom)})
    assert nav.find(strip, "markers").available() is False


def test_building_the_strip_reads_no_result():
    """The landing page resolves a route before a study is open, so availability is
    a call and not a field -- building the strip must not load a parquet."""
    calls: list[str] = []
    strip = _strip({"markers": (lambda: None,
                                lambda: calls.append("read") or True)})
    assert calls == []
    nav.find(strip, "markers").available()
    assert calls == ["read"]


def test_the_absence_page_names_the_question_rather_than_apologising():
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from panoscope.ui import nav as _nav

        _nav.render_unavailable(_nav.Tab(page="spikes", title="Spikes",
                                         question="Does this spike reach a surface?"))

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception, [str(e.value) for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "No spikes result for this study" in md
    assert "Does this spike reach a surface?" in md


# --------------------------------------------------------------------------- #
# The ?page= contract
# --------------------------------------------------------------------------- #
def test_every_page_name_the_app_shipped_with_still_resolves():
    """The URL vocabulary did not change: the surface ids already spelled it, and a
    link into a specific page is how a result gets shared today."""
    pages = nav.pages(_strip())
    for page in ("welcome", "upload", "markers", "pathways", "summary"):
        assert page in pages


def test_an_unregistered_page_name_is_not_valid():
    """Fail-closed, like the registries: an unknown id is never defaulted to another
    workflow's surface, which would answer a question nobody asked.

    The example used to be ``contrasts``, which is now a real surface -- the DE
    workflow registered once there was an artifact to answer it against and a root
    to run it. Having to replace it is the nav working rather than breaking: a
    registered workflow brings its page with it, and no page exists for a question
    nobody declared.
    """
    assert "spatial-stats" not in nav.pages(_strip())
    assert nav.find(_strip(), "spatial-stats") is None


# --------------------------------------------------------------------------- #
# The app itself still runs, on every page
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("page", ["welcome", "upload", "markers", "pathways",
                                  "summary", "not-a-page"])
def test_the_app_renders_with_no_exception_on_every_page(page):
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        pytest.skip("streamlit.testing.v1 unavailable")

    app = str(pathlib.Path(panoscope.__file__).parent / "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["active_page"] = page
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]


def test_the_rendered_strip_is_the_declared_one():
    """End to end: the buttons the app actually draws carry the registry's titles."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001
        pytest.skip("streamlit.testing.v1 unavailable")

    app = str(pathlib.Path(panoscope.__file__).parent / "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["active_page"] = "markers"
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]
    drawn = [b.label for b in at.button if b.key and b.key.startswith("nav_")]
    expected = ["Summary"] + [
        w.uispec().title for w in sorted(
            workflows.WORKFLOWS.values(),
            key=lambda w: (nav._TASK_ORDER[w.task_kind], w.uispec().workflow_id))
    ]
    assert drawn == expected
