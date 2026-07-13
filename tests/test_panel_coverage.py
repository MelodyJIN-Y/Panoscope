"""Panel-coverage guarantee — the "8 of 200 genes" penalty, tested WITHOUT the private
jazzPanda enrichment result.

``test_enrichment.py`` pins the same engine but is SKIPPED on a fresh clone / CI (it is
gated on the gitignored jazzPanda CSV existing), so on the public repo the flagship
panel-coverage rule would otherwise have no test coverage at all. These tests drive the
deterministic logic with synthetic gene-set rows, so they run everywhere: a set with almost
none of its genes on the panel earns the low-coverage caveat AND a one-band confidence
demotion; a well-covered set does not; and the penalty never underflows below Low.

The sibling guarantee — the marker-level panel-ABSENCE rule (an off-panel gene's absence is
not evidence against a cell type, and never moves the band) — lives in ``test_verdict.py``,
which is not data-gated.
"""
from __future__ import annotations

from agent import enrichment as E

# Confidence order (best -> worst), mirroring agent.enrichment._band_cluster.
_ORDER = ("Very High", "High", "Medium-High", "Medium", "Low")
# The method _band_cluster scores against (jazzPanda enrichment bands live under this key).
_METHOD = E._JZ_SCORE_KIND  # "jazzpanda_enrichment"


def _row(*, geneset_size: int, n_overlap: int, n_leading_edge: int = 10,
         test_statistic: float = 15.0, q: float = 0.001) -> dict:
    """A synthetic jazzPanda result row (only the fields ``_pathway_from_jz_row`` reads).

    ``n_overlap`` = set genes measured on the panel; coverage = n_overlap / geneset_size.
    ``n_leading_edge`` kept >= 5 by default so the *coverage* demotion is isolated from the
    separate tiny-leading-edge demotion.
    """
    genes = ",".join(f"G{i}" for i in range(n_leading_edge))
    return {
        "gene_set": "HALLMARK_TEST",
        "geneset_size": geneset_size,
        "n_overlap": n_overlap,
        "genes_selected": genes,
        "test_statistic": test_statistic,
        "p_value": 0.001,
        "p_adj_bh": q,
        "gc_corr": 0.80,
    }


def test_low_coverage_earns_caveat_and_exact_fraction():
    # 8 of 200 genes measured -> 4% coverage, below the 5% floor.
    pe = E._pathway_from_jz_row(_row(geneset_size=200, n_overlap=8))
    assert pe.panel_coverage == 8 / 200
    assert "very low panel coverage" in pe.caveats
    # The deterministic panel-scoped caveat names the real fraction — never invented.
    assert pe.panel_scope_caveat.startswith("Panel-scoped:")
    assert "8 of 200" in pe.panel_scope_caveat


def test_adequate_coverage_is_not_flagged():
    pe = E._pathway_from_jz_row(_row(geneset_size=40, n_overlap=20))  # 50%
    assert pe.panel_coverage == 0.5
    assert "very low panel coverage" not in pe.caveats


def test_low_panel_coverage_demotes_confidence_one_band():
    # Same strong score (15 -> Very High); ONLY coverage differs. Both keep 10 leading-edge
    # genes so the leading-edge demotion never fires — isolating the coverage penalty.
    well = E._pathway_from_jz_row(_row(geneset_size=40, n_overlap=20, n_leading_edge=10))   # 50%
    thin = E._pathway_from_jz_row(_row(geneset_size=200, n_overlap=8, n_leading_edge=10))   # 4%
    band_well, dem_well = E._band_cluster(well, _METHOD)
    band_thin, dem_thin = E._band_cluster(thin, _METHOD)
    assert band_well == "Very High"                                   # clears the top bar
    assert band_thin == "High"                                        # one band weaker
    assert _ORDER.index(band_thin) == _ORDER.index(band_well) + 1
    assert any("panel coverage" in d for d in dem_thin)
    assert not any("panel coverage" in d for d in dem_well)


def test_coverage_penalty_never_underflows_below_low():
    # A base band already at Low stays Low — the demotion is clamped, never wraps around.
    weak = E._pathway_from_jz_row(
        _row(geneset_size=200, n_overlap=8, test_statistic=1.0, n_leading_edge=10)
    )
    band, _ = E._band_cluster(weak, _METHOD)
    assert band == "Low"
