"""J13: Pathways after the biologist's call -- the second workflow end to end."""
from __future__ import annotations

import csv
import re

import pytest

from tests.conftest import DEMO_ID
from tests.journeys.flow import accept, go
from tests.ui.screen import evidence_rows, open_app, screen_text, widget_keys


def _enrichment(root, cluster: str) -> list[dict]:
    with (root / DEMO_ID / "interp" / "enrichment.csv").open() as f:
        return [r for r in csv.DictReader(f) if r["cluster"] == cluster]


@pytest.mark.req("PLAN-P4.4-01", "UI-2.6-01", "CLAUDE-TK-02", "CLAUDE-TK-04", "SKE-S2-01",
                 "SKE-S1-02")
def test_pathways_open_after_a_call_with_counts_and_never_touch_the_call(isolated):
    """Given the demo's c1 still OPEN, Pathways is locked and says to call c1 first;
    When the biologist accepts c1, Then Pathways opens on c1 with a census whose counts
    match the enrichment result, a panel-scoped (not genome-wide) caveat, an untestable
    row for every untestable set, and no band word; c1's label and band are unchanged;
    And a question half-typed into c1's pathway chat does not follow them to c2."""
    from panoscope.ui import data_access as da

    rows = _enrichment(isolated.root, "c1")
    n_untestable = int(rows[0]["cluster_n_untestable"])
    n_enriched = int(rows[0]["cluster_n_supported"])

    at = go(open_app("pathways", cluster="c1"), "pathways", "c1")
    assert "Call this cluster first" in screen_text(at)
    before = da.verdict_for("c1")

    at = go(accept(at, "c1"), "pathways", "c1")
    text = screen_text(at)
    assert re.search(rf"\b{n_enriched} enriched · \d+ suggestive · \d+ not enriched · "
                     rf"{n_untestable} untestable of \d+ sets", text), text[:800]
    assert "not genome-wide" in text
    under = [r for r in evidence_rows(at).values() if "s-underpowered" in r["html"]]
    assert len(under) == n_untestable
    assert not re.search(r"\b(Very High|High|Medium|Low)\b", text)
    da.verdict_for.clear()
    after = da.verdict_for("c1")
    assert (after.cell_type, after.confidence, after.verify) == (
        before.cell_type, before.confidence, before.verify)

    assert "pwconv_ask_input_c1" in widget_keys(at)
    at.text_input(key="pwconv_ask_input_c1").input("does EMT mean CAF contamination?")
    at = go(accept(at, "c2"), "pathways", "c2")
    assert "pwconv_ask_input_c1" not in widget_keys(at)
    assert at.text_input(key="pwconv_ask_input_c2").value == ""


@pytest.mark.req("SKE-S5-01", "SKE-S3-01", gap=True)
def test_a_program_recurring_across_clusters_is_stated_as_a_count(isolated):
    """When a program is enriched in several clusters, the cluster's page states the recurrence
    as a count ('also enriched in N other clusters'), never a stronger per-cluster claim;
    a cross-lineage program reads as a tension, never a re-typing."""
    with (isolated.root / DEMO_ID / "interp" / "enrichment.csv").open() as f:
        enriched = [(r["cluster"], r["gene_set"]) for r in csv.DictReader(f)
                    if r["tier"] == "enriched"]
    counts = {g: sum(1 for _, x in enriched if x == g) for _, g in enriched}
    cluster, gs = next((c, g) for c, g in enriched if counts[g] > 1)
    at = go(accept(open_app("markers", cluster=cluster), cluster), "pathways", cluster)
    text = screen_text(at)
    assert f"also enriched in {counts[gs] - 1} other cluster" in text, (gs, text[:600])
