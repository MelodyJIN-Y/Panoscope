"""Grounding tests for the redesign's data-access additions.

The spatial redesign precomputes ALL panel genes (not just the 26 demo markers),
so any pinned marker drives the density / feature-UMAP / violin. These tests pin
that contract: every panel gene has expression, and the violin accessor
(``expr_by_cluster``) returns per-cell values joined to the authoritative cluster
labels across all nine clusters. Skips gracefully if the committed parquet is
absent (so a partial checkout doesn't fail the suite spuriously).
"""
from __future__ import annotations

import pytest

from ui import data_access as da


def _expr_available() -> bool:
    try:
        return da.marker_expr_df().shape[1] > 1
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _expr_available(), reason="marker_expr parquet not available"
)


def test_available_expr_markers_covers_full_panel() -> None:
    genes = da.available_expr_markers()
    # widened from the 26 demo markers to the full analyzed panel
    assert len(genes) >= 200
    assert "cell_id" not in genes


def test_marker_expr_col_resolves_a_non_demo_gene() -> None:
    # ESR1 is not among the original 26 demo markers — it must resolve now that
    # expression is exported for every panel gene.
    col = da.marker_expr_col("ESR1")
    assert col is not None
    assert list(col.columns) == ["cell_id", "value"]
    assert len(col) > 0


def test_expr_by_cluster_covers_all_nine_clusters() -> None:
    df = da.expr_by_cluster("ERBB2")
    assert df is not None
    assert {"value", "cluster"}.issubset(df.columns)
    clusters = set(df["cluster"].astype(str).unique())
    assert {f"c{i}" for i in range(1, 10)}.issubset(clusters)


def test_expr_by_cluster_none_for_unknown_gene() -> None:
    assert da.expr_by_cluster("NOT_A_REAL_GENE_XYZ") is None
