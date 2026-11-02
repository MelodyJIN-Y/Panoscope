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

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E

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


# --------------------------------------------------------------------------- #
# The TIER gates — the band/q-value logic that otherwise only runs locally.
#
# ``test_enrichment.py`` exercises these against the real jazzPanda result, but
# that whole module is skipped on a fresh clone / CI (the result CSV is
# gitignored), and the golden hash cannot see two of them on the demo data:
#
#   * ``ENRICH_Q_MAX`` / ``SUGGESTIVE_Q_MAX`` — the demo study has no gene set in
#     the suggestive window, so the suggestive branch never executes there
#   * ``MIN_PANEL_HITS`` — ``n_leading_edge <= panel_hits`` always holds (the
#     leading edge is a SUBSET of the measured set genes), so while
#     ``MIN_LEADING_EDGE >= MIN_PANEL_HITS`` the leading-edge gate subsumes the
#     panel-hits gate and nothing real can trip it on its own
#
# Synthetic rows can hold the two apart, which is the only way to pin each gate
# independently. These run everywhere.
# --------------------------------------------------------------------------- #
def test_panel_hits_gate_is_what_makes_a_set_untestable():
    """Below MIN_PANEL_HITS -> untestable, however good the q and the leading edge.

    The row is deliberately impossible in real data (a leading edge larger than the
    measured overlap) so the panel-hits gate is isolated from the leading-edge one.
    """
    too_thin = E._pathway_from_jz_row(
        _row(geneset_size=200, n_overlap=cfg.MIN_PANEL_HITS - 1, n_leading_edge=10, q=0.0)
    )
    assert too_thin.panel_hits < cfg.MIN_PANEL_HITS
    assert too_thin.tier == "untestable"

    just_enough = E._pathway_from_jz_row(
        _row(geneset_size=200, n_overlap=cfg.MIN_PANEL_HITS, n_leading_edge=10, q=0.0)
    )
    assert just_enough.tier == "enriched"


def test_leading_edge_gate_is_independent_of_panel_hits():
    """Plenty measured, too few driving -> untestable. The 1-gene-artifact rule."""
    thin_edge = E._pathway_from_jz_row(
        _row(geneset_size=200, n_overlap=50, n_leading_edge=cfg.MIN_LEADING_EDGE - 1, q=0.0)
    )
    assert thin_edge.panel_hits >= cfg.MIN_PANEL_HITS
    assert thin_edge.tier == "untestable"


def test_q_value_tiers_at_both_thresholds():
    """enriched < ENRICH_Q_MAX <= suggestive <= SUGGESTIVE_Q_MAX < untestable.

    Both boundaries are pinned, including which side of each is inclusive — a
    silent flip of ``<`` to ``<=`` would promote a set the floor refuses.
    """
    def tier(q: float) -> str:
        return E._pathway_from_jz_row(
            _row(geneset_size=200, n_overlap=50, n_leading_edge=10, q=q)
        ).tier

    assert tier(0.0) == "enriched"
    assert tier(cfg.ENRICH_Q_MAX / 2) == "enriched"
    # the enriched bar is strict: q == ENRICH_Q_MAX is NOT enriched
    assert tier(cfg.ENRICH_Q_MAX) == "suggestive"
    assert tier(cfg.SUGGESTIVE_Q_MAX) == "suggestive"
    # above the suggestive bar nothing is surfaced
    assert tier(cfg.SUGGESTIVE_Q_MAX + 0.01) == "untestable"


def test_missing_q_value_is_untestable_never_assumed_significant():
    """No q -> untestable. The floor never promotes a set on a missing number."""
    row = _row(geneset_size=200, n_overlap=50, n_leading_edge=10)
    row["p_adj_bh"] = ""
    pe = E._pathway_from_jz_row(row)
    assert pe.q_value is None
    assert pe.tier == "untestable"


def test_untestable_sets_are_never_surfaced_and_force_verify():
    """A cluster whose only sets are untestable: nothing surfaced, verify TRUE.

    The CHARACTERISATION workflow carries no unit-level band, so "Low + verify"
    is how an untestable cluster must read — not an empty, confident-looking row.
    """
    pathways = (
        E._pathway_from_jz_row(
            _row(geneset_size=200, n_overlap=50, n_leading_edge=1, q=0.0)
        ),
        E._pathway_from_jz_row(
            _row(geneset_size=200, n_overlap=50, n_leading_edge=10, q=0.99)
        ),
    )
    ce = E._cluster_enrichment("c1", pathways, _METHOD)
    assert ce.enriched == () and ce.suggestive == ()
    assert ce.all_tested == pathways          # the audit keeps every tested set
    assert ce.top_theme is None
    assert ce.confidence == "Low"
    assert ce.verify is True


def test_enrichment_bands_are_per_method_and_ordered():
    """The band comes from the method's own scale; an unknown method never guesses."""
    bands = cfg.ENRICHMENT_BANDS[_METHOD]
    seen = [
        E._band_cluster(
            E._pathway_from_jz_row(
                _row(geneset_size=40, n_overlap=20, n_leading_edge=10, test_statistic=s)
            ),
            _METHOD,
        )[0]
        for s in sorted(bands.values(), reverse=True)
    ]
    # Monotonic: a weaker score is never a stronger band.
    assert [_ORDER.index(b) for b in seen] == sorted(_ORDER.index(b) for b in seen)
    assert cfg.band_for_enrichment(1e9, "no_such_method") == "Low"
