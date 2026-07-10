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


def test_exclude_note_flips_the_export_at_composition(monkeypatch) -> None:
    """An exclude note flips the exported exclude flag at composition, WITHOUT mutating
    the deterministic verdict on disk (docs/note-capture-design.md)."""
    # baseline: no exclude notes -> composed == deterministic
    monkeypatch.setattr(da, "_excluded_clusters", lambda: set())
    assert [v.exclude for v in da.composed_verdicts()] == [v.exclude for v in da.all_verdicts()]

    # an exclude note on c9 flips only c9, only in the composed view
    monkeypatch.setattr(da, "_excluded_clusters", lambda: {"c9"})
    composed = {v.cluster: v.exclude for v in da.composed_verdicts()}
    assert composed["c9"] is True
    assert composed["c1"] is False
    # the cached deterministic verdict for c9 is untouched (never mutated)
    assert da.all_verdicts()[8].exclude is False
    # and the CSV row for c9 differs from the un-excluded baseline
    monkeypatch.setattr(da, "_excluded_clusters", lambda: set())
    base_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    monkeypatch.setattr(da, "_excluded_clusters", lambda: {"c9"})
    excl_c9 = [ln for ln in da.verdict_csv().splitlines() if ln.startswith("c9,")][0]
    assert base_c9 != excl_c9 and "TRUE" in excl_c9.upper()


def test_overview_table_column_order_and_grounding() -> None:
    """The overview merges the marker call and the enriched programs into one row
    per cluster: columns in order, and every value is a projection of the verdicts
    (+ enrichment records) — nothing invented."""
    verdicts = da.all_verdicts()
    try:
        enr_map = {ce.cluster: ce for ce in da.all_enrichments()}
    except Exception:  # noqa: BLE001 - no enrichment slice (fresh clone) -> marker-only overview
        enr_map = {}
    out = summary._overview_table_html(verdicts, enr_map)

    order = ["Cluster", "Cell type", "Conf.", "Key markers", "Enriched programs"]
    positions = [out.index(f">{label}<") for label in order]
    assert positions == sorted(positions), "overview columns are out of order"

    # Grounded: each cluster's cell type and key markers appear verbatim.
    for v in verdicts:
        assert html.escape(v.cell_type) in out
        for g in v.key_markers:
            assert html.escape(str(g)) in out

    # Enriched program names, when present, come from the enrichment records only.
    for ce in enr_map.values():
        for p in ce.enriched[:4]:
            assert html.escape(summary._short(p.gene_set)) in out
