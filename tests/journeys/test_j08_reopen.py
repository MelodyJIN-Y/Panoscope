"""J8: reopening a call after new evidence (S8)."""
from __future__ import annotations

import pytest

from tests.journeys.flow import accept, go
from tests.ui.screen import open_app, widget_keys


@pytest.mark.req("UI-5.2-02", "UI-5.3-07", "CLAUDE-MEM-03", gap=True)
def test_a_pi_reopens_an_accepted_call_and_export_is_blocked_again(isolated):
    """Given the demo, where the PI accepted c9's proposal,
    When the IHC says basophils and they open c9,
    Then one tap reopens it: c9 is OPEN again, the ledger appends decision_reopened
    superseding the accept, and Summary refuses export naming c9."""
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    at = accept(open_app("markers", cluster="c9"), "c9")
    assert D.state({"c9": da.decision("c9")}, "c9") == D.ACCEPTED
    at = go(at, "markers", "c9")
    assert "rv_reopen_c9" in widget_keys(at), sorted(widget_keys(at))   # GAP: none today
    at.button(key="rv_reopen_c9").click().run()
    assert D.state({"c9": da.decision("c9")}, "c9") == D.OPEN
    assert "c9" in "\n".join(c.value for c in go(at, "summary").caption)
