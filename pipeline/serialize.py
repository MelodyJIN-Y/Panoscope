"""Faithful ``ClusterVerdict`` <-> JSON-safe dict conversion.

The verdict engine is deterministic, so a persisted verdict must reload to an
object EQUAL to the freshly computed one (``==`` on the frozen dataclasses). That
means every tuple field has to reload as a tuple, not a list — so the loaders are
explicit per type rather than a generic ``**d`` splat. ``tests/test_pipeline.py``
asserts the round-trip for all clusters; that equality is the correctness gate
for reading verdicts off disk instead of recomputing them.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from agent.holistic import HolisticReview, Refinement
from agent.types import (
    ClusterVerdict,
    LiteratureHook,
    MarkerEvidence,
    OffPanelNote,
    OpeningInterpretation,
)

_DEFAULT_EVIDENCE_SOURCE = "jazzpanda:top_result"
_DEFAULT_OFFPANEL_SOURCE = "panel:absence"


# --------------------------------------------------------------------------- #
# Serialize (dataclass -> JSON-safe dict). asdict handles the nesting; tuples
# become lists, which is fine for JSON.
# --------------------------------------------------------------------------- #
def verdict_to_dict(v: ClusterVerdict) -> dict[str, Any]:
    """Return a JSON-safe dict for a ClusterVerdict (nested, tuples -> lists)."""
    return dataclasses.asdict(v)


# --------------------------------------------------------------------------- #
# Deserialize (dict -> dataclass). Explicit per type; lists -> tuples so the
# reloaded object compares EQUAL to the computed one.
# --------------------------------------------------------------------------- #
def _evidence_from_dict(d: dict[str, Any]) -> MarkerEvidence:
    return MarkerEvidence(
        gene=d["gene"],
        top_cluster=d["top_cluster"],
        glm_coef=d["glm_coef"],
        pearson=d["pearson"],
        max_gg_corr=d["max_gg_corr"],
        max_gc_corr=d["max_gc_corr"],
        p_value=d["p_value"],
        within_cluster_pctile=d["within_cluster_pctile"],
        is_canonical=d["is_canonical"],
        is_on_panel=d["is_on_panel"],
        role=d["role"],
        caveats=tuple(d.get("caveats", ())),
        source=d.get("source", _DEFAULT_EVIDENCE_SOURCE),
    )


def _offpanel_from_dict(d: dict[str, Any]) -> OffPanelNote:
    return OffPanelNote(
        gene=d["gene"],
        cell_type=d["cell_type"],
        message=d["message"],
        source=d.get("source", _DEFAULT_OFFPANEL_SOURCE),
    )


def _hook_from_dict(d: dict[str, Any]) -> LiteratureHook:
    return LiteratureHook(
        claim=d["claim"],
        marker=d["marker"],
        cell_type=d["cell_type"],
        query_terms=tuple(d.get("query_terms", ())),
        status=d.get("status", "unfilled"),
    )


def _opening_from_dict(d: dict[str, Any]) -> OpeningInterpretation:
    return OpeningInterpretation(
        cluster=d["cluster"],
        cell_type=d["cell_type"],
        confidence=d["confidence"],
        headline=d["headline"],
        driving_markers=tuple(_evidence_from_dict(x) for x in d["driving_markers"]),
        offpanel_notes=tuple(_offpanel_from_dict(x) for x in d["offpanel_notes"]),
        literature_hooks=tuple(_hook_from_dict(x) for x in d["literature_hooks"]),
        verify=d["verify"],
    )


def verdict_from_dict(d: dict[str, Any]) -> ClusterVerdict:
    """Rebuild a ClusterVerdict from :func:`verdict_to_dict` output.

    Every tuple field is restored as a tuple so the result compares EQUAL to a
    freshly computed verdict (frozen-dataclass equality).
    """
    return ClusterVerdict(
        cluster=d["cluster"],
        cell_type=d["cell_type"],
        cell_type_short=d["cell_type_short"],
        confidence=d["confidence"],
        confidence_score=d["confidence_score"],
        key_markers=tuple(d["key_markers"]),
        notes=d["notes"],
        category=d["category"],
        lineage=d["lineage"],
        exclude=d["exclude"],
        verify=d["verify"],
        small_n=d["small_n"],
        evidence=tuple(_evidence_from_dict(x) for x in d["evidence"]),
        offpanel_notes=tuple(_offpanel_from_dict(x) for x in d["offpanel_notes"]),
        opening=_opening_from_dict(d["opening"]),
        band_basis=d["band_basis"],
        demotions=tuple(d["demotions"]),
        source_trace=tuple(d["source_trace"]),
    )


# --------------------------------------------------------------------------- #
# HolisticReview <-> JSON-safe dict (Step 4, deterministic Tier A). Same
# tuple-faithful round-trip discipline as the verdict: a persisted review must
# reload EQUAL to the freshly computed one.
# --------------------------------------------------------------------------- #
def holistic_to_dict(r: HolisticReview) -> dict[str, Any]:
    """Return a JSON-safe dict for a HolisticReview (tuples -> lists)."""
    return dataclasses.asdict(r)


def _refinement_from_dict(d: dict[str, Any]) -> Refinement:
    return Refinement(
        cluster=d["cluster"],
        from_call=d["from_call"],
        to_call=d["to_call"],
        evidence_markers=tuple(d.get("evidence_markers", ())),
        rationale=d["rationale"],
        lit_query=d["lit_query"],
    )


def holistic_from_dict(d: dict[str, Any]) -> HolisticReview:
    """Rebuild a HolisticReview from :func:`holistic_to_dict` output.

    Every tuple field is restored as a tuple so the result compares EQUAL to a
    freshly computed review (frozen-dataclass equality).
    """
    return HolisticReview(
        coherence_notes=tuple(d.get("coherence_notes", ())),
        refinements=tuple(_refinement_from_dict(x) for x in d.get("refinements", ())),
        set_is_coherent=d["set_is_coherent"],
    )
