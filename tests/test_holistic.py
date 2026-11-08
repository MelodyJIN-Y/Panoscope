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

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent.holistic import HolisticReview, Refinement, holistic_review


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


# --------------------------------------------------------------------------- #
# The two guards added when the refinement moved onto NormalisedEvidence.
# Both are sabotage checks: they break the thing the guard protects and assert
# the guard fires, because an invariant never seen to fail is not evidence.
# --------------------------------------------------------------------------- #
def test_refinement_refuses_to_cite_unmeasured_markers(monkeypatch) -> None:
    """Fewer OBSERVED markers than the refinement cites must RAISE.

    c8 has exactly 4 OBSERVED items, so asking for 5 is the smallest sabotage
    that makes the four-state rule bite. The failure mode this forbids is a
    silently shorter ``evidence_markers`` tuple -- or, worse, a TESTED_ABSENT
    gene printed as the evidence for a subtype call.
    """
    import panoscope.agent.holistic as holistic_mod

    monkeypatch.setattr(holistic_mod, "_REFINE_TOP_N", 5)
    with pytest.raises(AssertionError, match="OBSERVED"):
        holistic_review()


def test_proportions_note_does_not_swallow_a_real_failure(monkeypatch) -> None:
    """Only a MISSING FILE may become "no per-cell coordinates provided".

    A renamed column raises KeyError, which is not "no coordinates were
    provided" -- emitting that sentence for it would be a false grounded claim
    produced by a silent catch.
    """
    def boom(_cluster: str):
        raise KeyError("renamed column 'cluster'")

    monkeypatch.setattr(data, "get_cluster_cells", boom)
    with pytest.raises(KeyError):
        holistic_review()


# --------------------------------------------------------------------------- #
# Arbitrary-study honesty. Everything below is about a study that is NOT the
# breast demo: the module used to crash on one, and to describe the other two in
# breast prose. Each test is a sabotage check -- it was watched to fail against
# the old behaviour before being kept.
# --------------------------------------------------------------------------- #
_SHARED_CALL_FIXTURE = "synthetic_multisample_fixture"


@pytest.fixture
def shared_call_review() -> HolisticReview:
    """The holistic review of a study whose c1 and c3 carry the same cell type.

    Joint clustering over several samples produces exactly this, which is why the
    fixture has it. Opened via ``use_study`` so the process-global caches are
    scoped to the block (``tests/conftest.py`` clears them either side).
    """
    from panoscope.data.study import Study

    if not Study.open(_SHARED_CALL_FIXTURE).exists:
        pytest.skip(f"study {_SHARED_CALL_FIXTURE!r} is not on disk")
    with cfg.use_study(_SHARED_CALL_FIXTURE):
        return holistic_review()


def test_two_clusters_sharing_a_cell_type_is_reported_not_raised(
    shared_call_review: HolisticReview,
) -> None:
    """The defect that killed ``panoscope build`` on any jointly clustered study.

    ``_redundancy_note`` raised AssertionError here, inside
    ``pipeline.run._write_deterministic_interp``. Shared calls are a cross-cluster
    FINDING -- the note must name both clusters and the label they share.
    """
    joined = " ".join(shared_call_review.coherence_notes)
    assert "Shared calls" in joined
    assert "c1" in joined and "c3" in joined
    assert "Synthetic Type A" in joined


@pytest.mark.req("SKM-S4-01")
def test_set_is_coherent_is_computed_not_asserted(
    shared_call_review: HolisticReview,
) -> None:
    """``set_is_coherent`` was hardcoded True; a study with a shared call is not.

    Paired with ``test_set_is_coherent_true`` (the demo, genuinely coherent), this
    is what proves the flag tracks the set rather than being a constant: one study
    must come back False or the field carries no information at all.
    """
    assert shared_call_review.set_is_coherent is False


def test_no_tissue_claim_or_plausibility_verdict(review: HolisticReview) -> None:
    """No note asserts a tissue, or that what it measured is plausible.

    The proportions note opened "Proportions are plausible for a breast tumour"
    and closed "(a rare, small population as expected)". Both are unsourced
    judgments, written into ``holistic.json`` -- hashed into the manifest and
    rendered in the Summary tab -- and on a kidney study the first is a fabricated
    tissue. The real cluster sizes stay (asserted above); the verdict on them goes.
    """
    joined = " ".join(review.coherence_notes).lower()
    for banned in ("plausible", "breast", "tumour", "as expected"):
        assert banned not in joined, f"the holistic notes still assert {banned!r}"


def test_uncalled_clusters_are_counted_not_treated_as_a_shared_call(monkeypatch) -> None:
    """An un-annotated study: every cluster "Unknown" -- the annotate-failure shape.

    ``stages/annotate.py`` writes cell_type "Unknown" for every cluster whose call
    fails, so this is the state a real study lands in when annotation does not
    resolve. Two Unknowns are the ABSENCE of two calls, not two clusters sharing
    one, and a grouping whose every category is "Unknown" is not a compartment
    reading. The review must say that, and must not be coherent.
    """
    from panoscope.agent import annotation as ann

    blank = {
        c: {"cluster": c, "cell_type": "Unknown", "cell_type_short": "Unknown",
            "category": "Unknown", "lineage": "Unknown",
            "canonical_markers": [], "offpanel_canonical": []}
        for c in cfg.CLUSTER_ORDER
    }
    monkeypatch.setattr(ann, "load_annotation", lambda *a, **k: blank)

    out = holistic_review()
    joined = " ".join(out.coherence_notes)
    assert "No compartment grouping" in joined
    assert f"{len(cfg.CLUSTER_ORDER)} of {len(cfg.CLUSTER_ORDER)} clusters carry no cell-type call" in joined
    assert "Shared calls" not in joined, "Unknown was counted as a shared cell type"
    assert "No redundancy" not in joined, "uncalled clusters were claimed to be distinct calls"
    assert out.set_is_coherent is False


def test_redundancy_note_covers_all_nine_demo_units(review: HolisticReview) -> None:
    """All nine, never one: every demo cluster's call appears in the note."""
    joined = " ".join(review.coherence_notes)
    assert len(cfg.CLUSTER_ORDER) == 9
    for cluster in cfg.CLUSTER_ORDER:
        assert cfg.CLUSTER_KEY[cluster]["cell_type"] in joined, cluster
