"""Tests for the deterministic discriminator (``agent.discriminate``).

No network, no LLM: every assertion follows from the precomputed jazzPanda marker
table + the canonical-marker map. Ground truth used below (verified against the
demo data): c1 is Tumor with ERBB2/KRT7/EPCAM/... as its own top markers; the
Myoepithelial program (ACTA2/MYH11/MYLK/KRT14/KRT5/OXTR) localizes to c4; TP63 is
off-panel and was never measured.
"""
from __future__ import annotations

import pytest

from agent import data
from agent import discriminate as dsc
from agent.verdict import CANONICAL_MARKERS


def _genes(markers) -> set[str]:
    return {m.gene.upper() for m in markers}


def test_named_alt_buckets_markers_for_c1_vs_myoepithelial():
    d = dsc.discriminate("c1", "Myoepithelial")
    assert d.cluster == "c1"
    assert d.call_A == "Tumor"
    assert d.alt_B == "Myoepithelial"

    # ERBB2 is a c1 top marker canonical for Tumor -> supports A, WITH a real number.
    assert "ERBB2" in _genes(d.supporting_A)
    erbb2 = next(m for m in d.supporting_A if m.gene.upper() == "ERBB2")
    assert erbb2.glm_coef is not None and erbb2.glm_coef > 0
    assert erbb2.top_cluster == "c1"

    # ACTA2 is a Myoepithelial marker that localizes to c4 -> b_elsewhere, NO number here.
    elsewhere = {m.gene.upper(): m for m in d.b_elsewhere}
    assert "ACTA2" in elsewhere
    assert elsewhere["ACTA2"].glm_coef is None
    assert elsewhere["ACTA2"].top_cluster == "c4"
    assert elsewhere["ACTA2"].on_panel is True

    # TP63 is off-panel -> flagged, never measured, no number.
    off = {m.gene.upper(): m for m in d.offpanel_absent}
    assert "TP63" in off
    assert off["TP63"].on_panel is False
    assert off["TP63"].top_cluster is None
    assert off["TP63"].glm_coef is None

    # Six of seven Myoepithelial markers are on the panel -> settleable from data.
    assert d.settleable_on_panel is True


def test_offpanel_and_elsewhere_invariants():
    d = dsc.discriminate("c1", "Myoepithelial")
    assert all(not data.panel_contains(m.gene) for m in d.offpanel_absent)
    assert all(m.on_panel is False for m in d.offpanel_absent)
    assert all(data.panel_contains(m.gene) for m in d.b_elsewhere)
    assert all(m.top_cluster not in (None, "c1") for m in d.b_elsewhere)


def test_source_trace_quotes_numbers_only_for_own_top_markers():
    """The grounding guarantee: no glm_coef/pearson is ever attributed to a gene
    that is not one of THIS cluster's own top markers."""
    d = dsc.discriminate("c1", "Myoepithelial")
    own = {g.upper() for g in data.get_cluster_markers("c1")["gene"]}
    for entry in d.source_trace:
        if entry.startswith("jz:") and ("glm_coef=" in entry or "pearson=" in entry):
            gene = entry.split(":")[1]
            assert gene.upper() in own, f"number attributed to non-own gene: {entry}"


def test_derived_alt_none_for_clean_cluster():
    """c1's markers are all Tumor-canonical, so no rival is derivable; supporting_A
    is still populated and nothing raises."""
    d = dsc.discriminate("c1")
    assert d.call_A == "Tumor"
    assert d.alt_B is None
    assert len(d.supporting_A) > 0
    assert d.b_here == () and d.b_elsewhere == () and d.offpanel_absent == ()
    assert d.settleable_on_panel is False


def test_derived_alt_is_valid_type_when_present():
    """If any cluster derives a rival, it is a real canonical type != the call, and
    every cluster is handled without raising."""
    for c in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"):
        d = dsc.discriminate(c)
        if d.alt_B is not None:
            assert d.alt_B in CANONICAL_MARKERS
            assert d.alt_B != d.call_A


def test_unknown_alt_type_is_handled_not_crashed():
    d = dsc.discriminate("c1", "NotACellType")
    assert d.alt_B is None
    assert d.b_here == () and d.b_elsewhere == () and d.offpanel_absent == ()
    assert "NotACellType" in d.reason or "unknown" in d.reason.lower()


def test_alt_equal_to_call_is_noop():
    d = dsc.discriminate("c1", "Tumor")
    assert d.alt_B is None  # a type cannot be discriminated from itself
    assert d.refinement is False  # naming the same type is a no-op, not a refinement


def test_within_lineage_subtype_is_a_refinement_not_a_rival():
    # c2 is Stromal; CAF / fibroblast are subtypes of the stromal lineage, not distinct
    # cell types the panel can discriminate -> a refinement, answered honestly.
    for alt in ("cancer-associated fibroblast", "CAF", "matrix-remodelling CAF", "fibroblast"):
        d = dsc.discriminate("c2", alt)
        assert d.refinement is True, alt
        assert d.alt_B is None, alt
        low = d.reason.lower()
        assert "subtype" in low and "off-panel" in low
        # the on-panel drivers still ground the call
        assert any(m.gene.upper() == "LUM" for m in d.supporting_A)


def test_refinement_summary_is_grounded_and_honest():
    d = dsc.discriminate("c2", "cancer-associated fibroblast")
    s = dsc.settle_summary(d)
    low = s.lower()
    assert "subtype" in low and "tissue-context" in low
    assert "never measured" in low or "off-panel" in low
    # names the off-panel canonical markers without a number, and no bench recommendation
    assert "FAP" in s and "ihc" not in low and "experiment" not in low
    # every stated number still traces to source
    from agent.grounding_check import GroundingChecker
    assert GroundingChecker(literature_verifier=lambda _i: True).check(s).ok


def test_settle_summary_is_grounded_and_flags_offpanel_without_bench():
    d = dsc.discriminate("c1", "Myoepithelial")
    s = dsc.settle_summary(d)
    assert isinstance(s, str) and s
    assert "Tumor" in s and "Myoepithelial" in s
    assert "TP63" in s
    low = s.lower()
    assert "off-panel" in low or "never measured" in low
    # Biologist instruction: only FLAG off-panel genes, never recommend experiments.
    assert "ihc" not in low
    assert "experiment" not in low


def test_settle_summary_for_none_alt_is_nonempty():
    d = dsc.discriminate("c1")
    s = dsc.settle_summary(d)
    assert isinstance(s, str) and s
    assert "Tumor" in s


def test_unknown_cluster_raises():
    with pytest.raises(KeyError):
        dsc.discriminate("c999", "Tumor")


def test_normalize_cell_type_variants():
    assert dsc._normalize_cell_type("myoepithelial") == "Myoepithelial"
    assert dsc._normalize_cell_type("T cells") == "T_Cells"
    assert dsc._normalize_cell_type("t_cells") == "T_Cells"
    assert dsc._normalize_cell_type("Mast cells") == "Mast_Cells"
    assert dsc._normalize_cell_type("nonsense") is None
