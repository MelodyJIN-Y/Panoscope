"""Calibration set — proves the verdict engine commits on clean calls AND
flags shaky ones, without rubber-stamping (everything "confident") or crying
wolf (everything "verify").

Each case pins the ACTUAL output of ``verdict_for_cluster`` for a hand-picked
cluster: the cell type, the confidence band, and the verify flag. The expected
values here were derived by RUNNING the engine, not invented — see the
calibration table printed by ``scripts/calibration_table.py`` (also injected
into the README).

The set spans the confidence spectrum on purpose:

  CONFIDENT, verify=False
    c1 Tumor        Very High   (ERBB2 glm_coef 21.44 — the flagship clean call)
    c2 Stromal      Very High   (LUM glm_coef 18.00; COL1A1/VIM off-panel spine)
    c3 Macrophages  Very High   (LYZ glm_coef 11.84)

  MODERATE but still committed, verify=False (the anti-cry-wolf cases: lower
  glm_coef so a lower band, yet clean enough spatially that the engine does not
  demand a re-check)
    c5 T_Cells      Medium-High
    c6 B_Cells      Medium-High

  CAUTIOUS, verify=True (the anti-rubber-stamp case: a real fragile call the
  engine refuses to over-sell)
    c9 Mast_Cells   Medium      (CPA3 only; 2 assigned markers -> fragile; the
                                 driver also localizes better with another
                                 cluster)

Invariants at the bottom guarantee the set stays MEANINGFUL: at least one
verify=True and several verify=False, and both a top band and a lower band are
represented. If someone tunes the rubric, these break loudly instead of silently
becoming a rubber stamp.
"""

from __future__ import annotations

import pytest

from agent import verdict as V
from agent.types import ClusterVerdict

# --------------------------------------------------------------------------- #
# The calibration set: (cluster, cell_type, confidence, verify).
# Expected values were produced by RUNNING verdict_for_cluster — do not edit by
# hand; regenerate via scripts/calibration_table.py if the rubric changes.
# --------------------------------------------------------------------------- #
CALIBRATION_SET: tuple[tuple[str, str, str, bool], ...] = (
    # cluster, cell_type,       confidence,     verify
    ("c1", "Tumor", "Very High", False),   # clean flagship call
    ("c2", "Stromal", "Very High", False),   # clean, off-panel spine intact
    ("c3", "Macrophages", "Very High", False),   # clean myeloid call
    ("c5", "T_Cells", "Medium-High", False),   # moderate but committed
    ("c6", "B_Cells", "Medium-High", False),   # moderate but committed
    ("c9", "Mast_Cells", "Medium", True),    # fragile -> flagged for re-check
)

_CONFIDENT_CLUSTERS = frozenset({"c1", "c2", "c3"})  # expected Very High
_CAUTIOUS_CLUSTERS = frozenset({"c9"})               # expected verify=True


@pytest.mark.parametrize(
    ("cluster", "cell_type", "confidence", "verify"),
    CALIBRATION_SET,
    ids=[case[0] for case in CALIBRATION_SET],
)
def test_calibration_case(cluster: str, cell_type: str, confidence: str, verify: bool) -> None:
    """Each calibrated cluster resolves to its expected type / band / verify."""
    v = V.verdict_for_cluster(cluster)
    assert isinstance(v, ClusterVerdict)
    assert v.cell_type == cell_type, (
        f"{cluster}: expected cell_type {cell_type!r}, got {v.cell_type!r}"
    )
    assert v.confidence == confidence, (
        f"{cluster}: expected confidence {confidence!r}, got {v.confidence!r}"
    )
    assert v.verify is verify, (
        f"{cluster}: expected verify={verify}, got verify={v.verify}"
    )
    # A cluster we call Very High must never simultaneously ask for a re-check —
    # that would be an internally contradictory verdict.
    if v.confidence == "Very High":
        assert v.verify is False, f"{cluster}: Very High call must not carry verify=True"


def test_confident_clusters_commit() -> None:
    """The clean anchors land at the top band with no re-check (not cautious)."""
    for cluster in sorted(_CONFIDENT_CLUSTERS):
        v = V.verdict_for_cluster(cluster)
        assert v.confidence == "Very High", (
            f"{cluster}: expected the engine to COMMIT (Very High), got {v.confidence!r}"
        )
        assert v.verify is False, f"{cluster}: a clean call should not be flagged for re-check"


def test_cautious_clusters_are_flagged() -> None:
    """The shaky anchors are flagged for re-check, never rubber-stamped."""
    for cluster in sorted(_CAUTIOUS_CLUSTERS):
        v = V.verdict_for_cluster(cluster)
        assert v.verify is True, (
            f"{cluster}: expected the engine to FLAG this call (verify=True), "
            f"got verify={v.verify}"
        )
        # A flagged call must not also be sold as top confidence.
        assert v.confidence != "Very High", (
            f"{cluster}: a re-check call should not read as Very High"
        )


def test_calibration_set_is_meaningful() -> None:
    """The set must exercise BOTH commit and caution — not all one way.

    This is the anti-rubber-stamp / anti-cry-wolf guard: if the rubric ever
    collapses to a single behaviour, this fails.
    """
    verdicts = {c: V.verdict_for_cluster(c) for c, *_ in CALIBRATION_SET}

    verify_true = [c for c, v in verdicts.items() if v.verify is True]
    verify_false = [c for c, v in verdicts.items() if v.verify is False]

    assert len(verify_true) >= 1, "calibration set must include >=1 verify=True case"
    assert len(verify_false) >= 3, "calibration set must include several verify=False cases"

    bands = {v.confidence for v in verdicts.values()}
    assert "Very High" in bands, "set must include at least one top-confidence call"
    assert len(bands) >= 2, "set must span more than one confidence band"


def test_calibration_expectations_match_live_engine() -> None:
    """Every hard-coded expectation still matches the live engine output.

    Redundant with the parametrized test by design: it makes the calibration
    table a single, auditable contract and fails as one clear signal if the
    rubric drifts away from the documented calibration.
    """
    mismatches: list[str] = []
    for cluster, cell_type, confidence, verify in CALIBRATION_SET:
        v = V.verdict_for_cluster(cluster)
        if (v.cell_type, v.confidence, v.verify) != (cell_type, confidence, verify):
            mismatches.append(
                f"{cluster}: expected ({cell_type}, {confidence}, verify={verify}) "
                f"but got ({v.cell_type}, {v.confidence}, verify={v.verify})"
            )
    assert not mismatches, "calibration drift:\n" + "\n".join(mismatches)
