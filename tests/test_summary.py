"""Grounding tests for the Summary page's data (annotation table + CSV export).

The Summary page renders ``all_verdicts`` as a table and offers ``verdict_csv``
as a download. These tests pin that every cluster produces a complete verdict and
that the CSV export carries the full documented column set — the same contract
the per-cluster output format promises.
"""
from __future__ import annotations

import html

from agent import verdict as agent_verdict
from agent.config import CLUSTER_ORDER
from ui import data_access as da
from ui import summary


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


def test_summary_table_column_order_and_full_summary() -> None:
    """The table headers appear in the biologist's requested order, and the full
    (untruncated) cell-type summary is rendered — not clipped like st.dataframe."""
    verdicts = da.all_verdicts()
    out = summary._table_html(verdicts)

    order = [
        "Cluster",
        "Lineage",
        "Cell type",
        "Key markers",
        "Cell-type summary",
        "Confidence",
        "Re-check",
    ]
    positions = [out.index(f">{label}<") for label in order]
    assert positions == sorted(positions), "summary columns are out of the requested order"

    # The complete cell-type summary text is present (wraps in full, never clipped).
    c1_summary = da.celltype_summary("c1")
    assert c1_summary, "expected a grounded cell-type summary for c1"
    assert html.escape(c1_summary) in out
