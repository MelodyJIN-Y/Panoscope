"""Grounding tests for the deterministic verdict engine.

These assert the corrected confidence rubric (glm_coef DIRECT), the anchor
verdicts (c1 Tumor / c2 Stromal / c9 Mast), the NoSig -> Low+verify branch, the
panel-absence invariant (off-panel notes NEVER move the confidence band), and
the exact 11-column CSV contract. Every asserted number is pulled live from
``agent.data`` so a test can never encode a fabricated statistic.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import verdict as V
from panoscope.agent.types import ClusterVerdict, MarkerEvidence
from panoscope.types.measured import Coverage

#: A canonical-coverage count that clears its floor, for the hand-built drivers
#: below. ``_compute_band`` requires one -- there is deliberately no default, so a
#: caller cannot get the pre-coverage band back by omitting it.
_COVERED = Coverage(V.CANONICAL_COVERAGE_WHAT, 1, 1, V.coverage_floor(1))


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
    """Every gene named as off-panel canonical must really be absent from the panel.

    This was an import-time assertion in agent/verdict.py. It has been moved here for
    two reasons: it did filesystem I/O at import, and it guarded a duplicated map that
    nothing read. It now guards the map the runtime actually uses -- the one
    ``annotation.offpanel_canonical()`` resolves -- so a wrong entry fails loudly
    instead of passing against a dead copy.
    """
    from panoscope.agent import annotation

    live = annotation.OFF_PANEL_CANONICAL_FALLBACK
    assert live, "the off-panel canonical map is empty; the invariant would pass vacuously"
    for cell_type, genes in live.items():
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
    band, basis, _ = V._compute_band(nosig_driver, fragile=False,
                                     coverage=_COVERED)
    assert band == "Low"
    assert "NoSig" in basis


def test_no_canonical_support_is_low():
    """With no canonical driver at all, band is Low."""
    band, _, _ = V._compute_band(None, fragile=False, coverage=_COVERED)
    assert band == "Low"
    assert cfg.SCORE_MAP[band] == cfg.SCORE_MAP["Low"]


# --------------------------------------------------------------------------- #
# Panel-absence invariant — off-panel notes NEVER change the confidence band
# --------------------------------------------------------------------------- #
def test_offpanel_notes_never_change_band(monkeypatch):
    """Take the notes away entirely; nothing about the number may move.

    This used to recompute the band from the same inputs and assert the two agreed,
    which cannot fail: the notes were never passed to the computation it compared
    against. It now REMOVES them -- every off-panel note, for every cluster -- and
    asserts band, score and verify are byte-identical. That is the claim the
    headline rule actually makes, and it is the one worth being able to break.
    """
    before = {c: V.verdict_for_cluster(c) for c in cfg.CLUSTER_ORDER}
    assert sum(len(v.offpanel_notes) for v in before.values()), (
        "no cluster carries an off-panel note; this test would prove nothing")

    monkeypatch.setattr(V, "offpanel_notes", lambda cell_type: ())
    for cluster, was in before.items():
        now = V.verdict_for_cluster(cluster)
        assert now.offpanel_notes == ()
        assert now.confidence == was.confidence, cluster
        assert now.confidence_score == was.confidence_score, cluster
        assert now.verify == was.verify, cluster
        assert now.demotions == was.demotions, cluster


def test_a_per_gene_absence_is_not_an_input_but_the_per_label_count_is():
    """The distinction the coverage demote rests on, stated rather than implied.

    A panel-absence NOTE says "ERBB2 was never measured, so its absence argues
    nothing about this call". That is a statement about one gene and it may never
    move a number -- it is not evidence against anything, and the test above proves
    it reaches nothing.

    Canonical COVERAGE is a different quantity: how much of the label's curated
    support the measurement was capable of speaking to AT ALL. It does not argue
    against the call either -- it conditions how much the call's own number is
    worth, and it can only subtract. Both halves are asserted here, because the
    two are easy to collapse and the collapse is a rule violation in whichever
    direction it happens.
    """
    # The notes exist and are per-gene; the coverage is per-label and is a K of N.
    v = V.verdict_for_cluster("c2")
    assert {n.gene for n in v.offpanel_notes} == {"COL1A1", "COL1A2", "DCN",
                                                  "VIM", "FAP"}
    cov = V.canonical_coverage(v.cell_type)
    assert (cov.k, cov.n) == (4, 6)

    # Coverage NEVER promotes: full coverage on a weak driver changes nothing.
    weak = V.MarkerEvidence(
        gene="LUM", top_cluster="c2", glm_coef=1.5, pearson=0.5, max_gg_corr=0.5,
        max_gc_corr=0.0, p_value=None, within_cluster_pctile=1.0,
        is_canonical=True, is_on_panel=True, role="supports", caveats=(),
    )
    full = Coverage(V.CANONICAL_COVERAGE_WHAT, 9, 9, V.coverage_floor(9))
    thin = Coverage(V.CANONICAL_COVERAGE_WHAT, 1, 9, V.coverage_floor(9))
    band_full, _, changes_full = V._compute_band(weak, fragile=False,
                                                 coverage=full)
    band_thin, _, changes_thin = V._compute_band(weak, fragile=False,
                                                 coverage=thin)
    assert band_full == "Medium" and changes_full == ()
    assert band_thin == "Low", "it subtracts"
    assert len(changes_thin) == 1


def test_offpanel_notes_do_not_touch_verify():
    """c2 has off-panel notes but verify stays driven by the rubric's conditions.

    c2's canonical coverage is 4/6, which clears the floor, so the fifth condition
    does not fire here -- the notes themselves reach ``verify`` through nothing at
    all, which is what ``test_offpanel_notes_never_change_band`` proves by removing
    them.
    """
    v = V.verdict_for_cluster("c2")
    assert v.offpanel_notes  # notes present
    assert V.canonical_coverage(v.cell_type).ok
    # Very High + non-fragile + canonical driver + covered -> verify False.
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
