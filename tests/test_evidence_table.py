"""Evidence table: the jazzPanda-derived specificity badge.

The badge must be driven off the verdict's own ``ev.caveats`` (Tier A), not the
LLM biology note — so it is grounded and shows even when a gene has no note. It
fires only for ``max_gc_corr > pearson`` ("localizes better with another
cluster"), never for other caveats. These tests pin that behavior.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent import verdict as V
from ui import evidence_table as et

_CAVEAT = "localizes better with another cluster"


def test_caveat_badge_present_and_absent():
    assert "also marks another cluster" in et._caveat_badge(SimpleNamespace(caveats=(_CAVEAT,)))
    assert et._caveat_badge(SimpleNamespace(caveats=())) == ""
    # A different caveat must NOT trigger the specificity badge.
    assert et._caveat_badge(SimpleNamespace(caveats=("spatial pattern not unique",))) == ""


def test_bio_html_shows_badge_for_flagged_marker():
    v = V.verdict_for_cluster("c1")
    flagged = [e for e in v.evidence if _CAVEAT in e.caveats]
    assert flagged, "expected at least one specificity-flagged marker in c1"
    out = et._bio_html("c1", flagged[0])
    assert "pano-bio-caveat" in out and "also marks another cluster" in out


def test_bio_html_no_badge_for_clean_marker():
    v = V.verdict_for_cluster("c1")
    clean = [e for e in v.evidence if _CAVEAT not in e.caveats]
    assert clean
    out = et._bio_html("c1", clean[0])
    assert "pano-bio-caveat" not in out
