"""Grounding tests for the Tier-A enrichment engine (the jazzPanda method).

These pin the confident floor for the second workflow: the panel-coverage caveat
is on every pathway, the leading-edge gate refuses to surface 1-gene artifacts,
every enriched set clears the full gate, and confidence bands are per-method.
"""

from __future__ import annotations

import pytest

from agent import config as cfg
from agent import enrichment as E
from agent.types import ClusterEnrichment

# The jazzPanda enrichment result is unpublished + gitignored, so it may be absent
# on a fresh clone / CI. Skip the whole module then rather than fail.
pytestmark = pytest.mark.skipif(
    not E._JZ_ENRICHMENT_CSV.exists(),
    reason="jazzPanda enrichment result not present (gitignored)",
)


def test_all_clusters_produce_enrichment():
    ces = E.all_enrichments()
    assert len(ces) == len(cfg.CLUSTER_ORDER) == 9
    for ce in ces:
        assert isinstance(ce, ClusterEnrichment)
        assert ce.cell_type == cfg.CLUSTER_KEY[ce.cluster]["cell_type"]
        assert ce.method == "jazzpanda_enrichment"


def test_panel_scope_caveat_on_every_pathway():
    for ce in E.all_enrichments():
        for p in ce.all_tested:
            assert p.panel_scope_caveat.startswith("Panel-scoped:")
            assert str(p.set_size_full) in p.panel_scope_caveat
            assert 0.0 <= p.panel_coverage <= 1.0
            assert p.panel_hits <= p.set_size_full


def test_leading_edge_gate_kills_one_gene_artifacts():
    # c9 MTORC1 is driven by a single selected gene (FGL2) -> must be untestable,
    # never surfaced as enriched, even though its q is tiny.
    ce = E.enrichment_for_cluster("c9")
    mtorc1 = next(p for p in ce.all_tested if p.gene_set == "HALLMARK_MTORC1_SIGNALING")
    assert mtorc1.n_leading_edge < cfg.MIN_LEADING_EDGE
    assert mtorc1.tier == "untestable"
    assert mtorc1 not in ce.enriched and mtorc1 not in ce.suggestive


def test_enriched_sets_clear_the_full_gate():
    for ce in E.all_enrichments():
        for p in ce.enriched:
            assert p.q_value is not None and p.q_value < cfg.ENRICH_Q_MAX
            assert p.n_leading_edge >= cfg.MIN_LEADING_EDGE
            assert p.panel_hits >= cfg.MIN_PANEL_HITS
            assert p.tier == "enriched"
        # enriched sorted by score descending
        scores = [p.score for p in ce.enriched]
        assert scores == sorted(scores, reverse=True)


def test_suggestive_sets_are_verify_and_below_enriched():
    for ce in E.all_enrichments():
        for p in ce.suggestive:
            assert p.tier == "suggestive"
            assert p.q_value is not None and cfg.ENRICH_Q_MAX <= p.q_value <= cfg.SUGGESTIVE_Q_MAX
        # a cluster with no enriched set must ask for a re-check
        if not ce.enriched:
            assert ce.verify is True


def test_band_for_enrichment_is_per_method_and_monotonic():
    jz = "jazzpanda_enrichment"
    assert cfg.band_for_enrichment(20, jz) == "Very High"
    assert cfg.band_for_enrichment(0.5, jz) == "Low"
    assert cfg.band_for_enrichment(None, jz) == "Low"
    # the same numeric score bands differently under the two methods (different scales)
    assert cfg.band_for_enrichment(6.0, jz) == "Medium-High"
    assert cfg.band_for_enrichment(6.0, "ora_neg_log10_q") == "High"
