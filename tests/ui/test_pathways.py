"""The Pathways screen (CHARACTERISATION): a census with denominators, never a band
(plan Stage 5: 5.29). Counts are checked against enrichment.csv's own census columns
and against the per-item rows the page draws."""
from __future__ import annotations

import csv
import re

import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import evidence_rows, open_app, screen_text

_CENSUS = re.compile(r"(\d+) enriched · (\d+) suggestive · (\d+) not enriched · "
                     r"(\d+) untestable of (\d+) sets")


def _called(cluster: str):
    """Pathways open once the cluster has the biologist's call (Given: accepted)."""
    from panoscope.pipeline import decisions as D
    from panoscope.ui import data_access as da

    D.accept(cluster, da.verdict_for(cluster).cell_type, by="tester", at="2026-11-09T00:00:00")
    return open_app("pathways", cluster=cluster)


@pytest.mark.req("CLAUDE-TK-04", "UI-2.6-01", "IMPL-10-01", "SKE-S1-02")
@pytest.mark.parametrize("cluster", ["c1", "c2"])
def test_the_census_matches_the_rows_and_the_input_with_its_denominator(app_tree, cluster):
    with (app_tree / DEMO_ID / "interp" / "enrichment.csv").open() as f:
        row = next(r for r in csv.DictReader(f) if r["cluster"] == cluster)
    want = (int(row["cluster_n_supported"]), int(row["cluster_n_suggestive"]),
            int(row["cluster_n_untestable"]))
    at = _called(cluster)
    m = _CENSUS.search(screen_text(at))
    assert m, screen_text(at)[:800]
    enriched, suggestive, not_enriched, untestable, total = map(int, m.groups())
    assert (enriched, suggestive, untestable) == want
    assert enriched + suggestive + not_enriched + untestable == total

    rows = evidence_rows(at)
    states = [next(s for s in ("observed", "underpowered", "tested_absent", "never_measured")
                   if f"s-{s}" in r["html"]) for r in rows.values()]
    assert states.count("observed") == enriched + suggestive, states
    assert states.count("underpowered") == untestable, states
    assert states.count("tested_absent") == not_enriched, states
    assert len(rows) - states.count("never_measured") == total


@pytest.mark.req("CLAUDE-TK-04", "UI-2.1-02")
@pytest.mark.parametrize("cluster", ["c1", "c2"])
def test_no_band_word_appears_anywhere_on_the_pathways_page(app_tree, cluster):
    """The band vocabulary is the verdicts.csv confidence column: none of it, and no
    'confidence' chip, may appear on a characterisation page."""
    with (app_tree / DEMO_ID / "interp" / "verdicts.csv").open() as f:
        bands = {r["confidence"] for r in csv.DictReader(f)}
    assert {"Very High", "High", "Medium"} <= bands
    text = screen_text(_called(cluster))
    found = [b for b in bands if re.search(rf"(?<![\w-]){re.escape(b)}(?![\w-])", text)]
    assert found == [], found
    assert not re.search(r"\bconfidence\b", text, re.IGNORECASE), text
