"""Grounding tests for the holistic cross-cluster review (agent.holistic).

These tests enforce the Confident floor for the holistic pass:
- coherence_notes are non-empty and mention the immune compartment;
- every cell count named in the proportions note equals a REAL
  ``len(data.get_cluster_cells(...))`` (no fabricated numbers);
- exactly ONE refinement, c8 Dendritic -> a pDC/plasmacytoid subtype;
- the refinement's evidence_markers are the ones READ from data
  (include LILRA4 and TCL1A).
"""

from __future__ import annotations

import re
from dataclasses import FrozenInstanceError

import pytest

from agent import config as cfg
from agent import data
from agent.holistic import HolisticReview, Refinement, holistic_review


@pytest.fixture(scope="module")
def review() -> HolisticReview:
    return holistic_review()


# --------------------------------------------------------------------------- #
# Shape + immutability
# --------------------------------------------------------------------------- #
def test_returns_holistic_review(review: HolisticReview) -> None:
    assert isinstance(review, HolisticReview)


def test_dataclasses_are_frozen(review: HolisticReview) -> None:
    with pytest.raises(FrozenInstanceError):
        review.set_is_coherent = False  # type: ignore[misc]
    ref = review.refinements[0]
    with pytest.raises(FrozenInstanceError):
        ref.to_call = "something else"  # type: ignore[misc]


def test_set_is_coherent_true(review: HolisticReview) -> None:
    # All expected lineages present, no redundancy, no mis-call -> coherent.
    assert review.set_is_coherent is True


# --------------------------------------------------------------------------- #
# Coherence notes: non-empty, mention immune compartment, real cell counts
# --------------------------------------------------------------------------- #
def test_coherence_notes_non_empty(review: HolisticReview) -> None:
    assert len(review.coherence_notes) > 0
    assert all(isinstance(n, str) and n.strip() for n in review.coherence_notes)


def test_coherence_notes_mention_immune_compartment(review: HolisticReview) -> None:
    joined = " ".join(review.coherence_notes).lower()
    assert "immune" in joined


def test_coherence_notes_name_the_real_largest_and_rarest_counts(
    review: HolisticReview,
) -> None:
    """Every count in the proportions note must equal a real cell count.

    We recompute the ground-truth largest/rarest from data and assert the note
    reports exactly those numbers — proving no number was fabricated.
    """
    counts = {c: len(data.get_cluster_cells(c)) for c in cfg.CLUSTER_ORDER}
    largest = max(counts, key=counts.__getitem__)
    rarest = min(counts, key=counts.__getitem__)

    joined = " ".join(review.coherence_notes)

    # Counts are rendered with thousands separators in the note.
    assert f"{counts[largest]:,}" in joined
    assert f"{counts[rarest]:,}" in joined
    # And they are indeed len(get_cluster_cells(...)), not some constant.
    assert counts[largest] == len(data.get_cluster_cells(largest))
    assert counts[rarest] == len(data.get_cluster_cells(rarest))


def test_no_fabricated_numbers_in_coherence_notes(review: HolisticReview) -> None:
    """Any integer with >=3 digits in a note must be a real cell count.

    Every multi-digit number the holistic notes emit is a per-cluster cell count;
    this test extracts them and asserts each matches some real
    len(get_cluster_cells(...)), catching any invented statistic.
    """
    real_counts = {len(data.get_cluster_cells(c)) for c in cfg.CLUSTER_ORDER}
    joined = " ".join(review.coherence_notes)
    # Strip thousands separators, then find integer literals of length >= 3.
    numbers = re.findall(r"\d{3,}", joined.replace(",", ""))
    for token in numbers:
        assert int(token) in real_counts, (
            f"number {token} in coherence_notes is not a real cell count "
            f"{sorted(real_counts)}"
        )


# --------------------------------------------------------------------------- #
# Refinement: exactly one, c8 Dendritic -> pDC, grounded evidence markers
# --------------------------------------------------------------------------- #
def test_exactly_one_refinement(review: HolisticReview) -> None:
    assert len(review.refinements) == 1
    assert isinstance(review.refinements[0], Refinement)


def test_refinement_is_c8_dendritic_to_pdc(review: HolisticReview) -> None:
    ref = review.refinements[0]
    assert ref.cluster == "c8"
    assert ref.from_call == "Dendritic"
    to_lower = ref.to_call.lower()
    assert "pdc" in to_lower or "plasmacytoid" in to_lower


def test_refinement_evidence_markers_read_from_data(review: HolisticReview) -> None:
    """Evidence markers must be the top-3 of c8 pulled from data, not hard-coded."""
    ref = review.refinements[0]
    expected = tuple(
        str(g) for g in data.get_cluster_markers("c8")["gene"].head(3).tolist()
    )
    assert ref.evidence_markers == expected


def test_refinement_evidence_includes_lilra4_and_tcl1a(review: HolisticReview) -> None:
    ref = review.refinements[0]
    upper = {m.upper() for m in ref.evidence_markers}
    assert "LILRA4" in upper
    assert "TCL1A" in upper


def test_refinement_has_lit_query_for_live_citation(review: HolisticReview) -> None:
    # The pDC reading is a literature direction cited live; a query must be present.
    ref = review.refinements[0]
    assert ref.lit_query.strip()
    assert "plasmacytoid" in ref.lit_query.lower()


def test_refinement_is_subtype_not_miscall(review: HolisticReview) -> None:
    """c8 stays in the immune/dendritic lineage — a sharpening, not a lineage fix.

    The from_call is the cluster-key cell type, confirming the refinement does
    not change the underlying jazzPanda-driven lineage assignment.
    """
    ref = review.refinements[0]
    assert ref.from_call == cfg.CLUSTER_KEY["c8"]["cell_type"]
    assert cfg.CLUSTER_KEY["c8"]["category"] == "Immune"
