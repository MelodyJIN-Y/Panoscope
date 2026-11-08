"""J18: defer a cluster, then come back to it."""
from __future__ import annotations

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import defer, go
from tests.ui.screen import open_app, widget_keys


@pytest.mark.req("UI-5.2-03", "UI-5.2-02", "UI-5.3-07", gap=True)
def test_a_deferred_cluster_keeps_its_reason_and_can_be_reopened_and_accepted(isolated):
    """Given the demo, When the biologist defers c6,
    Then c6 is DEFERRED (terminal: it no longer blocks export) and the defer carries the
    pre-filled reason from the discrimination check ('the panel can settle it with GENE'
    or 'needs another assay') -- the first missing step today;
    When they reopen c6 and accept it, Then the ledger reads defer -> reopen -> accept,
    and review_state rebuilt from the ledger equals the stored one."""
    from panoscope.pipeline import decisions as D
    from panoscope.pipeline import store
    from panoscope.ui import data_access as da

    at = defer(open_app("markers", cluster="c6"), "c6")
    assert D.state({"c6": da.decision("c6")}, "c6") == D.DEFERRED
    from panoscope.data.study import Study

    rest = ", ".join(c for c in Study.open(DEMO_ID).units if c != "c6")
    assert f"still OPEN: {rest}" in "\n".join(c.value for c in go(at, "summary").caption)
    reason = store.load_decisions(DEMO_ID)["c6"].get("reason", "")
    assert "settle" in reason or "another assay" in reason, store.load_decisions(DEMO_ID)["c6"]
    at = go(at, "markers", "c6")
    assert "rv_reopen_c6" in widget_keys(at)
