"""Grounding tests for the Tier-A enrichment engine (the jazzPanda method).

These pin the confident floor for the second workflow: the panel-coverage caveat
is on every pathway, the leading-edge gate refuses to surface 1-gene artifacts,
every enriched set clears the full gate, and confidence bands are per-method.

**This module is SKIPPED on a fresh clone / CI** — it drives the real jazzPanda
result, which is gitignored (``data/jazzpanda/*``). That is not a coverage gap any
more; the same guarantees run everywhere from two ungated siblings:

  * ``tests/test_panel_coverage.py`` — the panel-coverage penalty, the panel-hits
    and leading-edge gates, both q-value tier boundaries, the missing-q rule, the
    untestable-cluster verify rule, and per-method bands, all on synthetic rows
  * ``tests/test_golden.py`` — the demo study's real per-cluster enrichment, its
    CSV and the cross-cluster themes, recomputed from the COMMITTED
    ``interp/enrichment/`` records and pinned byte-for-byte

What stays local-only is this module's agreement with the raw CSV itself (that the
adapter reads the real file correctly, and that c9's MTORC1 row really is the
single-gene artifact the gate must kill). Committing
``data/datasets/<id>/inputs/enrichment.csv`` would un-gate that too.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as E
from panoscope.agent.types import ClusterEnrichment

# The jazzPanda enrichment result is unpublished + gitignored, so it may be absent
# on a fresh clone / CI. Skip the whole module then rather than fail — but name the
# ungated siblings in the reason, so `pytest -rs` never reads as "this guarantee is
# untested here" when it is tested, just elsewhere.
pytestmark = pytest.mark.skipif(
    not E._jz_enrichment_csv().exists(),
    reason=(
        f"jazzPanda enrichment result not present (gitignored): "
        f"{E._jz_enrichment_csv()}. The band/tier/coverage guarantees still run in "
        f"tests/test_panel_coverage.py (synthetic) and tests/test_golden.py (the "
        f"committed demo records)."
    ),
)


def test_all_clusters_produce_enrichment():
    ces = E.all_enrichments()
    assert len(ces) == len(cfg.CLUSTER_ORDER) == 9
    for ce in ces:
        assert isinstance(ce, ClusterEnrichment)
        assert ce.cell_type == cfg.CLUSTER_KEY[ce.cluster]["cell_type"]
        assert ce.method == "jazzpanda_enrichment"


@pytest.mark.req("SKE-S2-01")
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


@pytest.mark.req("SKE-S1-01")
def test_suggestive_sets_are_verify_and_below_enriched():
    for ce in E.all_enrichments():
        for p in ce.suggestive:
            assert p.tier == "suggestive"
            assert p.q_value is not None and cfg.ENRICH_Q_MAX <= p.q_value <= cfg.SUGGESTIVE_Q_MAX
        # a cluster with no enriched set must ask for a re-check
        if not ce.enriched:
            assert ce.verify is True


def test_band_for_enrichment_is_monotonic():
    jz = "jazzpanda_enrichment"
    assert cfg.band_for_enrichment(20, jz) == "Very High"
    assert cfg.band_for_enrichment(6.0, jz) == "Medium-High"
    assert cfg.band_for_enrichment(0.5, jz) == "Low"
    assert cfg.band_for_enrichment(None, jz) == "Low"
    # an unknown method has no bands -> Low (never guesses a scale)
    assert cfg.band_for_enrichment(9.9, "unknown_method") == "Low"
