"""Grounding tests for the deterministic verdict engine.

These assert the corrected confidence rubric (glm_coef DIRECT), the anchor
verdicts (c1 Tumor / c2 Stromal / c9 Mast), the NoSig -> Low+verify branch, the
panel-absence invariant (off-panel notes NEVER move the confidence band), and
the verdict object the chat, the table and the CSV all read
(``pipeline.v1_verdicts``, built through ``core/verdict.py``). Every asserted number
is pulled live from ``agent.data`` so a test can never encode a fabricated statistic.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent.types import ClusterVerdict
from panoscope.core import verdict as C
from panoscope.pipeline import records, v1_verdicts as V
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Measured, MeasuredState
from panoscope.types.subject import gene, subject


def _built() -> dict:
    return records.build_all(cfg.current_study())


# --------------------------------------------------------------------------- #
# c1 Tumor — Very High, driven by ERBB2 (glm_coef traces to source)
# --------------------------------------------------------------------------- #
def test_c1_tumor_very_high_driven_by_erbb2():
    v = V.verdict_for("c1")
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
    """Base band before modifiers is the ladder read off driver.glm_coef."""
    erbb2 = float(data.get_marker("ERBB2")["glm_coef"])
    # ERBB2 glm_coef is large -> Very High under the direct rubric.
    assert C._band_for_coef(erbb2) == "Very High"
    v = V.verdict_for("c1")
    assert v.confidence == "Very High"


# --------------------------------------------------------------------------- #
# c2 Stromal — driven by LUM; off-panel notes include COL1A1 and VIM
# --------------------------------------------------------------------------- #
def test_c2_stromal_driven_by_lum():
    v = V.verdict_for("c2")
    assert v.cell_type == "Stromal"
    driver = v.opening.driving_markers[0]
    assert driver.gene == "LUM"
    assert driver.glm_coef == float(data.get_marker("LUM")["glm_coef"])
    assert v.confidence in ("Very High", "High")


def test_c2_offpanel_notes_include_col1a1_and_vim():
    v = V.verdict_for("c2")
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

    # The map now lives in the demo's lab list (reference view); its order is the
    # note order the golden output pins.
    live = annotation.offpanel_canonical("Stromal")
    assert live == ("COL1A1", "COL1A2", "DCN", "VIM", "FAP")
    for g in live:
        assert data.panel_contains(g) is False, f"{g} for Stromal is on-panel"


# --------------------------------------------------------------------------- #
# c9 Mast — fragile (<=2 markers) -> verify=True
# --------------------------------------------------------------------------- #
@pytest.mark.req("SKM-3d-02")
def test_c9_mast_fragile_verify_true():
    v = V.verdict_for("c9")
    assert v.cell_type == "Mast_Cells"
    assert v.verify is True
    assert v.small_n is True
    # Only 2 assigned markers (CPA3, CTSG) -> fragile.
    assert len(v.evidence) <= C.FRAGILE_MARKER_COUNT
    driver = v.opening.driving_markers[0]
    assert driver.gene == "CPA3"
    assert driver.glm_coef == float(data.get_marker("CPA3")["glm_coef"])
    # Capped/floored: fragile never reads above Medium-High and, with a real
    # canonical driver, stays >= Medium.
    assert v.confidence in ("Medium-High", "Medium")


# --------------------------------------------------------------------------- #
# Panel-absence invariant — off-panel notes NEVER change the confidence band
# --------------------------------------------------------------------------- #
@pytest.mark.req("SKM-3b-01")
def test_offpanel_notes_never_change_band():
    """Take the never-measured items away entirely; nothing about the number may move.

    The off-panel notes are rendered from the NEVER_MEASURED items. This REMOVES
    them -- every one, for every cluster -- re-runs the band engine on what is left,
    and asserts band, verify and the adjustments are identical. That is the claim
    the headline rule actually makes, and it is the one worth being able to break.
    """
    built = _built()
    assert sum(len(V.verdict_for(u).offpanel_notes) for u in built), (
        "no cluster carries an off-panel note; this test would prove nothing")
    for unit, b in built.items():
        kept = tuple(i for i in b.evidence.items
                     if i.measured.state is not MeasuredState.NEVER_MEASURED)
        now = C.assess(kept, b.canonical).value
        assert now.band == b.verdict.band, unit
        assert now.verify == b.verdict.verify, unit
        assert now.demotions == b.verdict.demotions, unit


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
    v = V.verdict_for("c2")
    assert {n.gene for n in v.offpanel_notes} == {"COL1A1", "COL1A2", "DCN",
                                                  "VIM", "FAP"}
    cov = _built()["c2"].verdict.coverage
    assert (cov.k, cov.n) == (4, 6)

    # Coverage NEVER promotes: full coverage on a weak driver changes nothing.
    # Three rows with a number, so the fragile cap/floor stays out of it.
    weak = tuple(EvidenceItem(
        subject=subject("gene", g=gene(g)), statistic=coef,
        statistic_name="glm_coef",
        measured=Measured(MeasuredState.OBSERVED, "panel:test"),
        corroborating=(("pearson", 0.5),),
    ) for g, coef in (("LUM", 1.5), ("N1", 0.5), ("N2", 0.4)))
    full = C.assess(weak, frozenset({"LUM"})).value
    thin = C.assess(weak, frozenset({"LUM", *(f"X{i}" for i in range(8))})).value
    assert full.band == "Medium" and full.demotions == ()
    assert thin.band == "Low", "it subtracts"
    assert len(thin.demotions) == 1


def test_offpanel_notes_do_not_touch_verify():
    """c2 has off-panel notes but verify stays driven by the rubric's conditions.

    c2's canonical coverage is 4/6, which clears the floor, so the fifth condition
    does not fire here -- the notes themselves reach ``verify`` through nothing at
    all, which is what ``test_offpanel_notes_never_change_band`` proves by removing
    them.
    """
    v = V.verdict_for("c2")
    assert v.offpanel_notes  # notes present
    assert _built()["c2"].verdict.coverage.ok
    # Very High + non-fragile + canonical driver + covered -> verify False.
    assert v.verify is False


# --------------------------------------------------------------------------- #
# All nine clusters resolve and source-trace is populated
# --------------------------------------------------------------------------- #
def test_all_nine_clusters_resolve():
    verdicts = [V.verdict_for(c) for c in cfg.CLUSTER_ORDER]
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
    v = V.verdict_for("c1")
    for e in v.evidence:
        expected = float(data.get_marker(e.gene)["glm_coef"])
        assert e.glm_coef == expected


def test_opening_interpretation_no_invented_numbers():
    """opening_interpretation exposes only real driver numbers + unfilled hooks."""
    op = V.verdict_for("c1").opening
    assert op.cluster == "c1"
    assert op.cell_type == "Tumor"
    for d in op.driving_markers:
        assert d.glm_coef == float(data.get_marker(d.gene)["glm_coef"])
    for hook in op.literature_hooks:
        assert hook.status == "unfilled"  # engine writes ZERO citations


def test_unknown_cluster_raises():
    with pytest.raises(KeyError):
        V.verdict_for("c99")
