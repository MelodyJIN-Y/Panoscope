"""The cluster rail, rendered: what the biologist sees down the left side."""
from __future__ import annotations

import csv

import pytest

from tests.conftest import DEMO_ID


def _rail(at_tree):
    from streamlit.testing.v1 import AppTest

    def _run() -> None:
        from panoscope.ui import cluster_rail, state

        state.init_state()
        cluster_rail.render_rail()

    at = AppTest.from_function(_run, default_timeout=60)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
    return {b.key.removeprefix("rail_"): b.label for b in at.button if b.key.startswith("rail_")}


@pytest.mark.req("SKM-3d-03", "UI-5.2-04")
def test_the_rail_flags_re_check_on_exactly_the_clusters_whose_record_says_verify(app_tree):
    """Expected from the build's verdicts.csv, not from the rail's own reader."""
    with (app_tree / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        verify = {r["cluster"]: r["verify"] == "TRUE" for r in csv.DictReader(f)}
    assert set(verify.values()) == {True, False}          # the demo has both
    labels = _rail(app_tree)
    assert set(labels) == set(verify)
    assert {c: "⚑" in label for c, label in labels.items()} == verify
