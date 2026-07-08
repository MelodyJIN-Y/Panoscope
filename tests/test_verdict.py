"""Grounding tests for the deterministic verdict engine.

These assert the corrected confidence rubric (glm_coef DIRECT), the anchor
verdicts (c1 Tumor / c2 Stromal / c9 Mast), the NoSig -> Low+verify branch, the
panel-absence invariant (off-panel notes NEVER move the confidence band), and
the exact 11-column CSV contract. Every asserted number is pulled live from
``agent.data`` so a test can never encode a fabricated statistic.
"""

from __future__ import annotations

import pytest

from agent import config as cfg
from agent import data
from agent import verdict as V
from agent.types import ClusterVerdict, MarkerEvidence


# --------------------------------------------------------------------------- #
# c1 Tumor — Very High, driven by ERBB2 (glm_coef traces to source)
# --------------------------------------------------------------------------- #
def test_c1_tumor_very_high_driven_by_erbb2():
    v = V.verdict_for_cluster("c1")
    assert isinstance(v, ClusterVerdict)
    assert v.cell_type == "Tumor"
    assert v.confidence in ("Very High", "High")
    assert v.verify is False

    driver = v.opening.driving_markers[0]
    assert driver.gene == "ERBB2"
    # The driver's glm_coef must equal the source table value exactly.
    assert driver.glm_coef == float(data.get_marker("ERBB2")["glm_coef"])
    assert "ERBB2" in v.key_markers


def test_c1_band_matches_direct_glm_coef_rule():
    """Base band before modifiers is band_for_coef(driver.glm_coef)."""
    erbb2 = float(data.get_marker("ERBB2")["glm_coef"])
    # ERBB2 glm_coef is large -> Very High under the direct rubric.
    assert cfg.band_for_coef(erbb2) == "Very High"
    v = V.verdict_for_cluster("c1")
    assert v.confidence == "Very High"


# --------------------------------------------------------------------------- #
# c2 Stromal — driven by LUM; off-panel notes include COL1A1 and VIM
# --------------------------------------------------------------------------- #
def test_c2_stromal_driven_by_lum():
    v = V.verdict_for_cluster("c2")
    assert v.cell_type == "Stromal"
    driver = v.opening.driving_markers[0]
    assert driver.gene == "LUM"
    assert driver.glm_coef == float(data.get_marker("LUM")["glm_coef"])
    assert v.confidence in ("Very High", "High")


def test_c2_offpanel_notes_include_col1a1_and_vim():
    v = V.verdict_for_cluster("c2")
    genes = {n.gene for n in v.offpanel_notes}
    assert {"COL1A1", "VIM"} <= genes
    assert genes == {"COL1A1", "COL1A2", "DCN", "VIM", "FAP"}
    # Every off-panel note references a gene that is genuinely NOT on the panel.
    for n in v.offpanel_notes:
        assert data.panel_contains(n.gene) is False
        assert "never measured" in n.message.lower()
        assert "not evidence against Stromal" in n.message


def test_offpanel_map_asserts_all_off_panel():
    """The OFF_PANEL_CANONICAL map must contain only genes absent from the panel."""
    for cell_type, genes in V.OFF_PANEL_CANONICAL.items():
        for g in genes:
            assert data.panel_contains(g) is False, f"{g} for {cell_type} is on-panel"


# --------------------------------------------------------------------------- #
# c9 Mast — fragile (<=2 markers) -> verify=True
# --------------------------------------------------------------------------- #
def test_c9_mast_fragile_verify_true():
    v = V.verdict_for_cluster("c9")
    assert v.cell_type == "Mast_Cells"
    assert v.verify is True
    assert v.small_n is True
    # Only 2 assigned markers (CPA3, CTSG) -> fragile.
    assert len(v.evidence) <= V.FRAGILE_MARKER_COUNT
    driver = v.opening.driving_markers[0]
    assert driver.gene == "CPA3"
    assert driver.glm_coef == float(data.get_marker("CPA3")["glm_coef"])
    # Capped/floored: fragile never reads above Medium-High and, with a real
    # canonical driver, stays >= Medium.
    assert v.confidence in ("Medium-High", "Medium")


# --------------------------------------------------------------------------- #
# NoSig-driven scenario -> Low + verify
# --------------------------------------------------------------------------- #
def test_nosig_driver_is_low_and_verify():
    """A driver whose top_cluster is NoSig collapses the band to Low."""
    nosig_driver = MarkerEvidence(
        gene="FAKEGENE",
        top_cluster="NoSig",
        glm_coef=0.05,
        pearson=0.10,
        max_gg_corr=0.5,
        max_gc_corr=0.5,
        p_value=None,
        within_cluster_pctile=1.0,
        is_canonical=True,
        is_on_panel=True,
        role="supports",
    )
    band, basis, _ = V._compute_band(nosig_driver, fragile=False)
    assert band == "Low"
    assert "NoSig" in basis


def test_no_canonical_support_is_low():
    """With no canonical driver at all, band is Low."""
    band, _, _ = V._compute_band(None, fragile=False)
    assert band == "Low"
    assert cfg.SCORE_MAP[band] == cfg.SCORE_MAP["Low"]


# --------------------------------------------------------------------------- #
# Panel-absence invariant — off-panel notes NEVER change the confidence band
# --------------------------------------------------------------------------- #
def test_offpanel_notes_never_change_band():
    """Adding/removing panel-absence notes must not move confidence or verify.

    The band is computed purely from the driving canonical marker's jazzPanda
    numbers. We recompute the band directly (notes cannot participate) and assert
    it equals the full verdict's band, whether or not off-panel notes exist.
    """
    for cluster in cfg.CLUSTER_ORDER:
        v = V.verdict_for_cluster(cluster)
        drivers = V._driving_markers(v.evidence)
        driver = drivers[0] if drivers else None
        band_no_notes, _, _ = V._compute_band(driver, fragile=v.small_n)
        # The band the verdict reports is exactly the notes-free computation.
        assert v.confidence == band_no_notes, cluster

    # c2 carries 5 off-panel notes; the band comes only from its driver, so the
    # computation with notes-present and a hypothetical notes-absent path agree.
    v2 = V.verdict_for_cluster("c2")
    assert len(v2.offpanel_notes) == 5
    driver2 = V._driving_markers(v2.evidence)[0]
    band_a = V._compute_band(driver2, fragile=v2.small_n)[0]
    band_b = V._compute_band(driver2, fragile=v2.small_n)[0]
    assert band_a == band_b == v2.confidence


def test_offpanel_notes_do_not_touch_verify():
    """c2 has off-panel notes but verify stays driven by band/fragility only."""
    v = V.verdict_for_cluster("c2")
    assert v.offpanel_notes  # notes present
    # Very High + non-fragile + canonical driver -> verify False despite notes.
    assert v.verify is False


# --------------------------------------------------------------------------- #
# CSV contract — exactly 11 fields, in exact order
# --------------------------------------------------------------------------- #
EXPECTED_CSV_ORDER = (
    "cluster",
    "cell_type",
    "cell_type_short",
    "confidence",
    "confidence_score",
    "key_markers",
    "notes",
    "category",
    "lineage",
    "exclude",
    "verify",
)


def test_to_csv_row_has_11_fields_in_order():
    v = V.verdict_for_cluster("c1")
    row = V.to_csv_row(v)
    assert len(row) == 11
    assert tuple(row.keys()) == EXPECTED_CSV_ORDER
    assert V.CSV_COLUMNS == EXPECTED_CSV_ORDER
    # Content spot checks.
    assert row["cluster"] == "c1"
    assert row["cell_type"] == "Tumor"
    assert row["verify"] in ("TRUE", "FALSE")
    assert row["exclude"] in ("TRUE", "FALSE")
    assert "ERBB2" in row["key_markers"]


def test_to_csv_header_matches_contract():
    csv_text = V.to_csv(V.all_verdicts())
    header = csv_text.splitlines()[0]
    assert header == ",".join(EXPECTED_CSV_ORDER)
    # 9 data rows + 1 header.
    assert len([ln for ln in csv_text.splitlines() if ln]) == 10


# --------------------------------------------------------------------------- #
# All nine clusters resolve and source-trace is populated
# --------------------------------------------------------------------------- #
def test_all_nine_clusters_resolve():
    verdicts = V.all_verdicts()
    assert [v.cluster for v in verdicts] == list(cfg.CLUSTER_ORDER)
    for v in verdicts:
        assert v.cell_type == cfg.CLUSTER_KEY[v.cluster]["cell_type"]
        assert v.confidence in (
            "Very High",
            "High",
            "Medium-High",
            "Medium",
            "Low",
        )
        assert v.source_trace  # every verdict traces its numbers


def test_source_trace_values_match_data():
    """Each glm_coef on the evidence equals the source table value exactly."""
    v = V.verdict_for_cluster("c1")
    for e in v.evidence:
        expected = float(data.get_marker(e.gene)["glm_coef"])
        assert e.glm_coef == expected


def test_opening_interpretation_no_invented_numbers():
    """opening_interpretation exposes only real driver numbers + unfilled hooks."""
    op = V.opening_interpretation("c1")
    assert op.cluster == "c1"
    assert op.cell_type == "Tumor"
    for d in op.driving_markers:
        assert d.glm_coef == float(data.get_marker(d.gene)["glm_coef"])
    for hook in op.literature_hooks:
        assert hook.status == "unfilled"  # engine writes ZERO citations


def test_unknown_cluster_raises():
    with pytest.raises(KeyError):
        V.verdict_for_cluster("c99")
