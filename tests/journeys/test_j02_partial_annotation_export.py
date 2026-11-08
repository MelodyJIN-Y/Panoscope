"""J2: partial annotation, a flagged accept, a defer, then export (S2)."""
from __future__ import annotations

import csv

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import accept, defer, go
from tests.ui.screen import open_app, run


@pytest.mark.req("UI-5.2-01", "UI-5.2-04", "UI-5.2-03", "UI-5.2-06", gap=True)
def test_a_flagged_accept_drafts_a_note_a_defer_keeps_its_reason_and_export_confirms(isolated):
    """Given the demo with every cluster OPEN, and c9 the rubric's only verify=TRUE call,
    When the PI opens Summary, Then there is no download and it names every OPEN cluster;
    When they accept c9, Then (because it is flagged) a validation note with a literature
    reconciliation is drafted (interface §5.6) -- the first missing step today;
    When they defer c5, Then the defer stores a reason the report prints;
    And export is one heavy confirm (unresolved list, freeze, report id, manifest sha)."""
    with (isolated.root / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["cluster"] for r in rows if r["verify"] == "TRUE"] == ["c9"]
    clusters = [r["cluster"] for r in rows]

    at = open_app("summary")
    assert not at.download_button
    assert f"still OPEN: {', '.join(clusters)}" in "\n".join(c.value for c in at.caption)

    from panoscope.agent import memory, tools

    at = accept(at, "c9")
    assert len(memory.read_notes(tools.memory_base_dir())) == 1     # GAP: 0 today

    at = defer(at, "c5")
    from panoscope.pipeline import store

    assert store.load_decisions(DEMO_ID)["c5"].get("reason")
    for c in clusters:
        if c not in ("c5", "c9"):
            at = accept(at, c)
    at = go(at, "summary")
    labels = [str(b.label).lower() for b in at.button]
    assert any("freeze" in lb or "confirm export" in lb for lb in labels), labels
    run(at)
