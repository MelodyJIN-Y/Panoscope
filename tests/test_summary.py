"""Grounding tests for the Summary page's data (annotation table + CSV export).

The Summary page renders ``all_verdicts`` as a table and offers ``verdict_csv``
as a download. These tests pin that every cluster produces a complete verdict and
that the CSV export carries the full documented column set — the same contract
the per-cluster output format promises.
"""
from __future__ import annotations

from agent import verdict as agent_verdict
from agent.config import CLUSTER_ORDER
from ui import data_access as da


def test_all_verdicts_returns_nine_complete_calls() -> None:
    verdicts = da.all_verdicts()
    assert len(verdicts) == len(CLUSTER_ORDER) == 9
    for v in verdicts:
        # the columns the summary table shows must be populated
        assert v.cell_type and v.cell_type_short
        assert v.confidence and v.category and v.lineage


def test_verdict_csv_has_full_header_and_nine_rows() -> None:
    csv = da.verdict_csv()
    lines = csv.strip().splitlines()
    header = lines[0]
    for col in agent_verdict.CSV_COLUMNS:
        assert col in header
    # header + one row per cluster
    assert len(lines) == 1 + len(CLUSTER_ORDER)
