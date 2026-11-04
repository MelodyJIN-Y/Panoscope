"""The method's declared limits reach the biologist, not just the registry.

``types/contracts.py`` annotates ``does_not_measure`` "MANDATORY, from the paper,
rendered in the UI" and ``IMPLEMENTATION.md`` says the same -- and for the life of
the field nothing in ``panoscope/ui/`` or ``panoscope/agent/`` read it. The
registry's docstring names what that costs: "a ``glm_coef`` with no
``does_not_measure`` beside it is indistinguishable from a fold change, and the
confidence assessment silently stops being an audit of someone else's output."

So the test that matters is not that the builder *can* produce the block -- the
previous state of the repo would pass that bar, because the field was always there
to read. It is that the renderer actually emits it.
``test_the_limits_are_wired_into_the_renderer_not_just_available`` is the one that
fails if the call in :func:`surface.render_surface` is deleted and everything else
is left alone.

Five existing tests name this field (``test_workflow_enrichment``,
``test_workflow_de``, ``test_types_uispec``, ``test_registries``,
``test_types_contracts_protocols``); every one of them checks the *declaration*.
None checked that anybody reads it.
"""

from __future__ import annotations

import sys

import pytest

from panoscope import methods
from panoscope.ui import surface
from panoscope.workflows import enrichment, marker

MARKER = marker.WORKFLOW.uispec()
PATHWAYS = enrichment.WORKFLOW.uispec()

#: The surface id and the skill id are deliberately different strings
#: (``workflows/marker.py:51``), so a method is looked up through the workflow
#: that owns the surface, never through the surface id.
_OWNERS = {"markers": marker.WORKFLOW, "pathways": enrichment.WORKFLOW}


@pytest.mark.parametrize("spec", [MARKER, PATHWAYS], ids=["markers", "pathways"])
def test_each_surface_carries_what_its_method_does_not_measure(spec):
    got = surface.method_limits_html(spec)
    registered = methods.for_workflow(_OWNERS[spec.workflow_id].name)
    assert registered, f"no method registered into {spec.workflow_id!r}"
    for m in registered:
        # The sentence itself, not a paraphrase of it. A surface that restated the
        # limits in its own words would be a second copy to drift from the paper.
        head = str(m.does_not_measure).strip()[:40]
        assert head in got, f"{m.name}: does_not_measure not on the {spec.workflow_id} surface"
        assert m.statistic_name in got, f"{m.name}: the statistic it qualifies is unnamed"


def test_the_glm_coef_is_not_left_readable_as_a_fold_change():
    """The specific failure ``methods/jazzpanda.py`` says the module exists to prevent.

    A biologist reading ``LUM glm_coef 18.00`` reads it as expression unless the
    surface says otherwise, and the misreading is invisible: both quantities are
    large and positive on a gene that genuinely is a c2 marker.
    """
    got = surface.method_limits_html(MARKER).lower()
    assert "fold change" in got
    assert "co-locate" in got or "co-location" in got


def test_the_limits_are_wired_into_the_renderer_not_just_available(monkeypatch):
    """Drive ``render_surface`` with a fake Streamlit and watch for the block.

    Deleting the ``method_limits_html`` call from ``render_surface`` leaves every
    other test in this file green, which is exactly how the field stayed unrendered
    for as long as it did.
    """
    emitted: list[str] = []

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _FakeSt:
        def markdown(self, body, **kw): emitted.append(str(body))
        def container(self, **kw): return _Ctx()
        def columns(self, *a, **kw): return (_Ctx(), _Ctx())
        def button(self, *a, **kw): return False
        def rerun(self): raise AssertionError("no rerun in a render test")

    monkeypatch.setitem(sys.modules, "streamlit", _FakeSt())
    surface.render_surface(MARKER, header={}, rows=[])

    blob = "".join(emitted)
    assert "What produced these numbers" in blob, \
        "render_surface drew the table without the method's declared limits"
    head = str(methods.get("jazzpanda_spatial_glm").does_not_measure).strip()[:40]
    assert head in blob


def test_the_assumptions_travel_with_the_limits():
    """Both fields are mandatory, so surfacing one and dropping the other would
    re-open half the gap: a coefficient is only what it says under its conditions."""
    got = surface.method_limits_html(MARKER)
    m = methods.get("jazzpanda_spatial_glm")
    assert m.assumptions, "jazzPanda declares no assumptions"
    for a in m.assumptions:
        assert str(a).strip()[:30] in got


def test_a_surface_with_no_registered_method_says_nothing_rather_than_lying():
    """``degraded_spec`` is someone else's CSV: no method, so no limits to state.

    An empty string is right here. A placeholder ("no limits declared") would read
    as a method that declared it has none, which is the one thing ``register``
    refuses to let a method do.
    """
    from panoscope.types.uispec import degraded_spec

    spec = degraded_spec("some_tool_output.csv", ("gene", "score"))
    assert surface.method_limits_html(spec) == ""
