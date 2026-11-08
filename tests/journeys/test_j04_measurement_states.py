"""J4: never measured vs tested absent vs underpowered, on the demo's c1 (S4).

The expected counts come from the raw inputs (the panel file and the method's own
table), never from the CLI or the app, which share the code under test."""
from __future__ import annotations

import re

import pandas as pd
import pytest

from tests.conftest import DEMO_ID
from tests.ui.screen import evidence_rows, open_app


@pytest.mark.req("CLAUDE-MS-01", "CLAUDE-MS-02", "CLAUDE-MS-03", "ARCH-9.5-04",
                 "ARCH-13.4-01", "ARCH-18-04")
def test_a_reviewer_sees_why_a_gene_is_missing_and_no_unmeasured_gene_carries_a_number(
        isolated, capsys):
    """Given the demo's c1 and its raw inputs (280-gene panel; jazzPanda names each
    detected gene's top cluster and marks 49 genes NoSig),
    When the reviewer runs `panoscope explain c1` and opens the Markers table,
    Then observed = c1's genes, tested-absent = the detected genes not in c1 (absence
    IS evidence), never measured = every NoSig gene plus canonical markers off the panel,
    each named with its frame; no non-observed item carries a number; and no
    never-measured row on the screen shows a value."""
    from panoscope import cli
    from panoscope.data.study import Study
    from panoscope.pipeline import records
    from panoscope.types.measured import MeasuredState

    inputs = isolated.root / DEMO_ID / "inputs"
    panel = set(pd.read_parquet(inputs / "panel.parquet")["gene"])
    table = pd.read_csv(inputs / "markers_top.csv")
    observed = set(table.loc[table["top_cluster"] == "c1", "gene"])
    nosig = set(table.loc[table["top_cluster"] == "NoSig", "gene"])
    detected = panel - nosig
    assert (len(panel), len(observed), len(detected - observed), len(nosig)) == (280, 84, 147, 49)

    assert cli.main(["explain", "c1", "--study", DEMO_ID]) == 0
    out = capsys.readouterr().out
    evidence = out.split("evidence (measured", 1)[1].split("tested, not reported", 1)[0]
    assert set(re.findall(r"^  [ *] (\S+)\s+glm_coef=", evidence, re.M)) == observed
    assert f"tested, not reported here ({len(detected - observed)}) — absence IS evidence" in out
    never = dict(re.findall(r"^  - (\S+): outside (\w+):", out.split("never measured (", 1)[1],
                            re.M))
    assert {g for g, f in never.items() if f == "detect"} == nosig
    off_panel = {g for g, f in never.items() if f == "panel"}
    assert off_panel and not off_panel & panel                  # e.g. KRT18
    assert f"never measured ({len(never)}) — absence is NOT evidence" in out

    built = records.build(Study.open(DEMO_ID), "c1")
    for item in built.evidence.items:
        assert (item.statistic is None) == (item.measured.state is not MeasuredState.OBSERVED)

    rows = evidence_rows(open_app("markers", cluster="c1"))
    shown = {k.rsplit(":", 1)[1]: r for k, r in rows.items() if k.startswith("gene|")}
    for g in sorted(nosig | off_panel):
        assert g in shown, g                                    # a row, never a blank
        assert not re.search(r"\d", " ".join(shown[g]["cells"][1:3])), shown[g]["cells"]
