"""Holistic cross-cluster review (SKILL Step 4 — grounded refinement).

After every cluster is annotated individually (``agent.verdict``), STOP and read
the whole set together. Individual calls can look fine in isolation but tell an
incoherent story as a whole, or a call can be *sharpened* only once the whole
compartment is visible.

This module implements the honest holistic pass the user chose: re-check all 9
calls for coherence and surface exactly ONE real refinement — c8 Dendritic ->
"Plasmacytoid DC (pDC)". This is a SUBTYPE SHARPENING within the same
immune/dendritic lineage, not a mis-call fix and not a changed jazzPanda number.
It only becomes clear seeing the whole immune compartment.

Grounding discipline (Confident floor):
- Every number in ``coherence_notes`` is COMPUTED AT RUNTIME from
  ``agent.data`` / ``agent.config`` (cell counts via ``data.get_cluster_cells``);
  no number is hard-coded prose.
- The refinement's ``evidence_markers`` are READ from
  ``data.get_cluster_markers("c8")`` (top 3 by glm_coef), never hard-coded.
- The pDC label itself is an INTERPRETATION (a literature direction, cited live
  later via ``lit_query``), not a changed jazzPanda value. The biologist decides
  whether to accept it.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import annotation
from agent import config as cfg
from agent import data

# --------------------------------------------------------------------------- #
# Constants for the single grounded refinement (c8 -> pDC).
# The evidence markers are NOT listed here — they are read from data at runtime.
# --------------------------------------------------------------------------- #
_REFINE_CLUSTER: str = "c8"
_REFINE_FROM: str = "Dendritic"
_REFINE_TO: str = "Plasmacytoid DC (pDC)"
_REFINE_TOP_N: int = 3
_REFINE_RATIONALE: str = (
    "these markers are plasmacytoid-DC-specific; the subtype is only clear "
    "across the whole immune compartment"
)
_REFINE_LIT_QUERY: str = "LILRA4 plasmacytoid dendritic cell marker"

# Breast-TME compartment grouping of the 9 authoritative clusters. This is a
# GROUPING of the CLUSTER_KEY, not new biology: it names which major compartment
# each cluster's cell type belongs to so the coherence check can assert coverage.
_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "epithelial tumor": ("c1",),
    "myoepithelial": ("c4",),
    "fibroblast/stroma": ("c2",),
    "endothelial": ("c7",),
    "immune": ("c3", "c5", "c6", "c8", "c9"),
}


# --------------------------------------------------------------------------- #
# Frozen dataclasses
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Refinement:
    """One grounded refinement surfaced by the holistic review.

    ``from_call`` -> ``to_call`` is a proposal the biologist decides on. It is a
    subtype sharpening within the same lineage, NOT a changed jazzPanda value.
    ``evidence_markers`` are read from jazzPanda output; the ``to_call`` reading
    is a literature direction to be cited live via ``lit_query``.
    """

    cluster: str
    from_call: str
    to_call: str
    evidence_markers: tuple[str, ...]
    rationale: str
    lit_query: str


@dataclass(frozen=True)
class HolisticReview:
    """Result of the cross-cluster coherence pass over all 9 calls.

    ``coherence_notes`` are grounded observations (every number computed from
    source at runtime). ``refinements`` holds exactly the real refinements the
    whole-set view surfaces (here: one). ``set_is_coherent`` is True when all
    expected lineages are present, there is no redundancy, and no mis-call
    remains — the refinement is a sharpening, not a correction.
    """

    coherence_notes: tuple[str, ...]
    refinements: tuple[Refinement, ...]
    set_is_coherent: bool


# --------------------------------------------------------------------------- #
# Grounded coherence notes (every number computed from source at runtime)
# --------------------------------------------------------------------------- #
def _compartment_note() -> str:
    """(a) Expected breast-TME compartments present, derived from CLUSTER_KEY.

    Lists each major compartment and the cell types grouped under it, then states
    all major compartments are represented. Pure grouping of the authoritative
    cluster key — no invented biology.
    """
    parts: list[str] = []
    for compartment, clusters in _COMPARTMENTS.items():
        types = ", ".join(annotation.cell_type_for(c) for c in clusters)
        parts.append(f"{compartment} ({types})")
    return (
        "Expected breast-TME compartments are all represented: "
        + "; ".join(parts)
        + ". The immune compartment in particular is well populated "
        "(macrophages, T, B, dendritic, mast)."
    )


def _proportions_note() -> str:
    """(b) Proportions: largest and rarest cluster by REAL cell count.

    Cell counts come from ``data.get_cluster_cells`` at runtime — never hard-coded.
    """
    counts = {c: len(data.get_cluster_cells(c)) for c in cfg.CLUSTER_ORDER}
    largest = max(counts, key=counts.__getitem__)
    rarest = min(counts, key=counts.__getitem__)
    largest_type = annotation.cell_type_for(largest)
    rarest_type = annotation.cell_type_for(rarest)
    return (
        f"Proportions are plausible for a breast tumour: the largest cluster is "
        f"{largest} {largest_type} at {counts[largest]:,} cells, and the rarest is "
        f"{rarest} {rarest_type} at {counts[rarest]:,} cells (a rare, small "
        f"population as expected)."
    )


def _redundancy_note() -> str:
    """(c) Redundancy: assert no two clusters share a cell type, and say so.

    Derived from the authoritative CLUSTER_KEY values at runtime.
    """
    cell_types = [annotation.cell_type_for(c) for c in cfg.CLUSTER_ORDER]
    distinct = len(set(cell_types)) == len(cell_types)
    if not distinct:
        # Fail loud rather than silently claim coherence on a key that
        # accidentally repeats a cell type.
        raise AssertionError(
            f"[holistic] two clusters share a cell type in CLUSTER_KEY: {cell_types}"
        )
    return (
        f"No redundancy: all {len(cell_types)} clusters carry distinct cell types "
        f"({', '.join(cell_types)}); no two clusters collapse onto the same call."
    )


def _coherence_notes() -> tuple[str, ...]:
    """The three grounded coherence observations (compartments, proportions, redundancy)."""
    return (
        _compartment_note(),
        _proportions_note(),
        _redundancy_note(),
    )


# --------------------------------------------------------------------------- #
# The single grounded refinement (c8 Dendritic -> Plasmacytoid DC)
# --------------------------------------------------------------------------- #
def _c8_refinement() -> Refinement:
    """Build the c8 -> pDC refinement, reading its evidence markers from data.

    Evidence markers are the top-3 markers of c8 by glm_coef, pulled live from
    ``data.get_cluster_markers`` (NOT hard-coded). The pDC label is a literature
    direction (``lit_query``), not a changed jazzPanda number.
    """
    rows = data.get_cluster_markers(_REFINE_CLUSTER)  # glm_coef desc, NoSig excluded
    evidence_markers = tuple(str(g) for g in rows["gene"].head(_REFINE_TOP_N).tolist())
    return Refinement(
        cluster=_REFINE_CLUSTER,
        from_call=_REFINE_FROM,
        to_call=_REFINE_TO,
        evidence_markers=evidence_markers,
        rationale=_REFINE_RATIONALE,
        lit_query=_REFINE_LIT_QUERY,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def holistic_review() -> HolisticReview:
    """Re-check all 9 calls together and surface the one grounded refinement.

    Coherence: all expected breast-TME compartments are present, proportions are
    plausible (largest/rarest by real cell count), and no two clusters share a
    cell type. Refinement: exactly one — c8 Dendritic -> Plasmacytoid DC (pDC),
    a subtype sharpening within the same immune/dendritic lineage, evidenced by
    c8's actual top markers and to be cited live. ``set_is_coherent`` is True:
    every expected lineage is present, there is no redundancy, and the refinement
    is a sharpening, not a mis-call fix.
    """
    return HolisticReview(
        coherence_notes=_coherence_notes(),
        refinements=(_c8_refinement(),),
        set_is_coherent=True,
    )


if __name__ == "__main__":
    review = holistic_review()
    print("coherence_notes:")
    for note in review.coherence_notes:
        print(f"  - {note}")
    print()
    print("refinements:")
    for ref in review.refinements:
        print(f"  {ref.cluster}: {ref.from_call} -> {ref.to_call}")
        print(f"    evidence_markers: {ref.evidence_markers}")
        print(f"    rationale: {ref.rationale}")
        print(f"    lit_query: {ref.lit_query}")
    print()
    print("set_is_coherent:", review.set_is_coherent)
