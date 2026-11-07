"""Holistic cross-cluster review (SKILL Step 4 — grounded refinement).

After every cluster is annotated individually (``pipeline.v1_verdicts``), STOP and read
the whole set together. Individual calls can look fine in isolation but tell an
incoherent story as a whole, or a call can be *sharpened* only once the whole
compartment is visible.

The bundled demo's one refinement -- c8 Dendritic -> "Plasmacytoid DC (pDC)", a
subtype sharpening, not a changed number -- is gated on the demo.

Study-agnostic: notes come from THIS study's annotation and cell counts. Nothing
here names a tissue or judges what a tissue ought to have.

Grounding discipline (Confident floor):
- Every number in ``coherence_notes`` is COMPUTED AT RUNTIME from
  ``agent.data`` / ``agent.config`` (cell counts via ``data.get_cluster_cells``);
  no number is hard-coded prose.
- The refinement's ``evidence_markers`` are READ from the unit's
  ``NormalisedEvidence`` (``pipeline.records.build_all``), top 3 by the method's
  statistic and filtered to OBSERVED items, never hard-coded.
- The pDC label itself is an INTERPRETATION (a literature direction, cited live
  later via ``lit_query``), not a changed jazzPanda value. The biologist decides
  whether to accept it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.data.study import NOT_YET_CALLED

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

#: Cell-type values that mean "no call was made". Two uncalled clusters are NOT two
#: clusters sharing a cell type, and an uncalled cluster is not a compartment.
_UNCALLED: frozenset[str] = frozenset({"", "Unknown", "None", NOT_YET_CALLED})


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
    """Result of the cross-cluster coherence pass over this study's calls.

    ``coherence_notes`` are grounded observations (every number computed from
    source at runtime). ``refinements`` holds the real refinements the whole-set
    view surfaces. ``set_is_coherent`` is True when every cluster carries a call
    and no two clusters carry the SAME call. NOT a claim that the tissue's expected
    compartments are present.
    """

    coherence_notes: tuple[str, ...]
    refinements: tuple[Refinement, ...]
    set_is_coherent: bool


# --------------------------------------------------------------------------- #
# Grounded coherence notes (every number computed from source at runtime)
# --------------------------------------------------------------------------- #
def _metas(metas: Mapping[str, Mapping[str, str]] | None) -> dict[str, Mapping[str, str]]:
    """Per-cluster ``{"cell_type", "category"}``, defaulting to this study's annotation.

    Display passes overridden calls in; the default keeps the built artifact (and
    the golden) byte-stable.
    """
    out = {c: annotation.meta_for(c) for c in cfg.CLUSTER_ORDER}
    for c, meta in (metas or {}).items():
        if c in out:
            out[c] = meta
    return out


def _compartment_note(metas: Mapping[str, Mapping[str, str]]) -> str:
    """(a) Which compartments this study's own annotation groups its clusters into.

    A pure grouping of the ``category`` field: what IS represented, never coverage.
    """
    by_cat: dict[str, list[str]] = {}
    for c in cfg.CLUSTER_ORDER:
        meta = metas[c]
        by_cat.setdefault(str(meta.get("category") or "Unknown"), []).append(meta["cell_type"])
    if not by_cat or set(by_cat) <= _UNCALLED:
        return (
            f"No compartment grouping: this study's annotation assigns no category to "
            f"any of its {len(cfg.CLUSTER_ORDER)} clusters, so the set cannot be read "
            f"compartment by compartment."
        )
    parts = [f"{cat} ({', '.join(types)})" for cat, types in by_cat.items()]
    return "Compartments represented across the dataset: " + "; ".join(parts) + "."


def _proportions_note(metas: Mapping[str, Mapping[str, str]]) -> str:
    """(b) Proportions: largest and rarest cluster by REAL cell count.

    Cell counts come from ``data.get_cluster_cells`` at runtime — never hard-coded.
    """
    try:
        counts = {c: len(data.get_cluster_cells(c)) for c in cfg.CLUSTER_ORDER}
    except FileNotFoundError:
        # ONLY a missing file: the sentence below is grounded for this error alone.
        return ("Cell-count proportions are unavailable for this dataset "
                "(no per-cell coordinates provided).")
    largest = max(counts, key=counts.__getitem__)
    rarest = min(counts, key=counts.__getitem__)
    largest_type = metas[largest]["cell_type"]
    rarest_type = metas[rarest]["cell_type"]
    # Counts only: no unsourced tissue expectation about the spread.
    return (
        f"Cluster sizes: the largest is {largest} {largest_type} at "
        f"{counts[largest]:,} cells, the smallest {rarest} {rarest_type} at "
        f"{counts[rarest]:,} cells. Whether that spread is expected is a question "
        f"for the tissue, and this review has no grounded basis to answer it."
    )


def _redundancy_note(metas: Mapping[str, Mapping[str, str]]) -> tuple[str, bool]:
    """(c) Redundancy: do two clusters carry the same call, and is any cluster uncalled?

    Returns the note and whether the set is internally coherent. A shared call is a
    reported finding, not an error; uncalled clusters are counted separately.
    """
    calls = {c: str(metas[c]["cell_type"]) for c in cfg.CLUSTER_ORDER}
    if not calls:
        return ("This study declares no clusters, so there is no set to review.", False)

    uncalled = [c for c, t in calls.items() if t in _UNCALLED]
    named = {c: t for c, t in calls.items() if t not in _UNCALLED}
    by_type: dict[str, list[str]] = {}
    for c, t in named.items():
        by_type.setdefault(t, []).append(c)
    shared = {t: cs for t, cs in by_type.items() if len(cs) > 1}

    parts: list[str] = []
    if shared:
        parts.append(
            "Shared calls: "
            + "; ".join(f"{', '.join(cs)} all carry {t}" for t, cs in sorted(shared.items()))
            + " — distinct clusters landing on one label is a finding for the "
              "biologist to resolve, not a merge this review performs"
        )
    elif named:
        parts.append(
            f"No redundancy: all {len(named)} called clusters carry distinct cell types "
            f"({', '.join(named.values())}); no two clusters collapse onto the same call"
        )
    if uncalled:
        parts.append(
            f"{len(uncalled)} of {len(calls)} clusters carry no cell-type call "
            f"({', '.join(uncalled)}); nothing is claimed about them"
        )
    return "; ".join(parts) + ".", not shared and not uncalled


def _coherence_notes(
    metas: Mapping[str, Mapping[str, str]],
) -> tuple[tuple[str, ...], bool]:
    """The three grounded coherence observations, and whether the set is coherent."""
    redundancy, coherent = _redundancy_note(metas)
    return (_compartment_note(metas), _proportions_note(metas), redundancy), coherent


# --------------------------------------------------------------------------- #
# The single grounded refinement (c8 Dendritic -> Plasmacytoid DC)
# --------------------------------------------------------------------------- #
def _c8_refinement() -> Refinement:
    """Build the c8 -> pDC refinement, reading its evidence markers from evidence.

    Evidence markers are c8's top markers by the method's statistic, read off
    ``NormalisedEvidence`` (statistic-descending) and NOT hard-coded. The pDC
    label is a literature direction (``lit_query``), not a changed jazzPanda
    number. Only OBSERVED items are citable; fewer than ``_REFINE_TOP_N`` raises.
    """
    from panoscope.pipeline import records
    from panoscope.types.measured import MeasuredState

    evidence = records.build_all(cfg.current_study())[_REFINE_CLUSTER].evidence
    observed = [i for i in evidence.items if i.measured.state is MeasuredState.OBSERVED]
    if len(observed) < _REFINE_TOP_N:
        raise AssertionError(
            f"[holistic] {_REFINE_CLUSTER} has {len(observed)} OBSERVED marker(s), "
            f"fewer than the {_REFINE_TOP_N} the {_REFINE_TO} refinement cites; "
            "refusing to emit a refinement evidenced by unmeasured markers"
        )
    evidence_markers = tuple(i.subject.one("g").id for i in observed[:_REFINE_TOP_N])
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
def holistic_review(
    metas: Mapping[str, Mapping[str, str]] | None = None,
) -> HolisticReview:
    """Re-check this study's calls together and surface any grounded refinement.

    Coherence notes: the compartments this study's own annotation names, the
    real cluster sizes, and whether any two clusters share a call or carry none.
    ``set_is_coherent`` is COMPUTED from the last of those, not asserted.

    Refinements are study-specific content: the bundled demo's one c8 Dendritic
    -> Plasmacytoid DC (pDC) sharpening is gated on the demo and nothing else is
    emitted for any other study, so the list is empty rather than borrowed.
    """
    # Demo content: gated on the bundled demo AND the CURRENT call, so an
    # already-made sharpening is not offered again.
    refinements: tuple[Refinement, ...] = ()
    resolved = _metas(metas)
    if cfg.DATASET_ID == cfg.BUNDLED_DEMO_ID and \
            _REFINE_CLUSTER in cfg.KNOWN_CLUSTERS and \
            resolved[_REFINE_CLUSTER]["cell_type"] == _REFINE_FROM:
        refinements = (_c8_refinement(),)
    notes, coherent = _coherence_notes(resolved)
    return HolisticReview(
        coherence_notes=notes,
        refinements=refinements,
        set_is_coherent=coherent,
    )
