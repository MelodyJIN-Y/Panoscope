"""Deterministic, pre-baked, GROUNDED fallbacks so a slow or failed API never
breaks the demo.

Every response here is a pure function of :mod:`agent.verdict` (which is itself a
pure function of the precomputed jazzPanda output and the panel list). Nothing is
invented: each number traces to a jazzPanda column, each off-panel line traces to
the panel-absence primitive, and no live citation is asserted (the fallback path
never has a network, so it states no PMID/DOI — it says the literature would be
fetched live, and stays purely jazzPanda-grounded).

Because every response is built from the verdict, every response carries the same
``sources`` / ``verify`` / ``grounding`` envelope as a live answer and therefore
passes :class:`agent.grounding_check.GroundingChecker` with no literature verifier
needed (no identifiers to resolve).

Public API
----------
- :func:`fallback_opening` — the pre-baked opening interpretation for a cluster.
- :func:`fallback_answer` — a canned answer for a recognized demo-beat question,
  or ``None`` when nothing matches (the caller then uses the generic fallback).
- :func:`generic_fallback` — the last-resort grounded template for any cluster.

Design rules honoured
---------------------
- Confident floor: no fabricated marker, number, or citation. Numbers are rendered
  with two decimals (well within the grounding checker's tolerance) straight from
  the verdict's evidence.
- Panel-absence rule: off-panel canonical markers are surfaced as context that does
  NOT down-weight the call, and never carry a jazzPanda statistic (stating a number
  for an off-panel gene would be fabrication and the grounding gate rejects it).
- Viewing controls never change a value: fallbacks state numbers, never recompute.
- No overclaiming: uncertain clusters keep their ``verify = True`` and say so.
"""

from __future__ import annotations

from typing import Optional

from agent.types import (
    AgentResponse,
    ClusterVerdict,
    GroundingSidecar,
    MarkerEvidence,
    OffPanelNote,
    Source,
)
from agent.verdict import opening_interpretation, verdict_for_cluster

# Number formatting: two decimals is inside the grounding checker's absolute
# tolerance (1e-2) for these stats, so a rendered "18.00" grounds to LUM's real
# 17.9978... . Kept as one constant so prose and sidecar never drift.
_FMT = "{:.2f}"

# How many driving markers to name in prose (avoid a wall of numbers).
_MAX_DRIVERS_IN_PROSE: int = 3

# Above this driver pearson, the spatial signal reads as coherent (used only to
# phrase the doublet answer honestly; never invents a doublet score).
_COHERENT_PEARSON: float = 0.5


# --------------------------------------------------------------------------- #
# Grounding envelope builders (shared by every fallback)
# --------------------------------------------------------------------------- #
def _marker_sources(markers: tuple[MarkerEvidence, ...]) -> tuple[Source, ...]:
    """One ``jz`` Source chip per named marker, valued by its glm_coef."""
    return tuple(
        Source(
            kind="jz",
            ref=m.gene,
            value=_FMT.format(m.glm_coef),
            detail=f"glm_coef {_FMT.format(m.glm_coef)}, pearson {_FMT.format(m.pearson)}",
        )
        for m in markers
    )


def _offpanel_sources(notes: tuple[OffPanelNote, ...]) -> tuple[Source, ...]:
    """One ``panel`` Source chip per off-panel canonical gene (absence context)."""
    return tuple(
        Source(
            kind="panel",
            ref=n.gene,
            value="off-panel",
            detail=n.message,
        )
        for n in notes
    )


def _sidecar(
    markers: tuple[MarkerEvidence, ...],
    notes: tuple[OffPanelNote, ...],
) -> GroundingSidecar:
    """Machine-readable manifest of exactly the numbers/markers a response used.

    Off-panel genes are listed as markers-mentioned (so the checker recognises the
    token) but contribute NO number — their absence is not a statistic. No PMIDs
    and no notes are used on the fallback path.
    """
    numbers: list[tuple[str, str, float]] = []
    for m in markers:
        numbers.append((m.gene, "glm_coef", float(m.glm_coef)))
        numbers.append((m.gene, "pearson", float(m.pearson)))
    marker_names = tuple(m.gene for m in markers) + tuple(n.gene for n in notes)
    return GroundingSidecar(
        numbers=tuple(numbers),
        markers=marker_names,
        pmids=(),
        notes_used=(),
    )


def _driver_phrase(markers: tuple[MarkerEvidence, ...]) -> str:
    """Render 'GENE (glm_coef X.XX, pearson Y.YY)' for the leading drivers."""
    bits = [
        f"{m.gene} (glm_coef {_FMT.format(m.glm_coef)}, "
        f"pearson {_FMT.format(m.pearson)})"
        for m in markers[:_MAX_DRIVERS_IN_PROSE]
    ]
    return ", ".join(bits)


def _offpanel_line(notes: tuple[OffPanelNote, ...], cell_type: str) -> str:
    """A single grounded sentence naming the off-panel canonical markers.

    Names the genes without any number (absence is not a statistic) and states the
    panel-absence rule explicitly so the reader does not read a missing canonical
    marker as evidence against the call.
    """
    if not notes:
        return ""
    genes = ", ".join(n.gene for n in notes)
    return (
        f" Note on absence: {genes} are canonical {cell_type.replace('_', ' ')} "
        f"markers that are off-panel (never measured here), so their absence is "
        f"not evidence against the call."
    )


# --------------------------------------------------------------------------- #
# Opening interpretation fallback
# --------------------------------------------------------------------------- #
def fallback_opening(cluster: str) -> AgentResponse:
    """Pre-baked opening interpretation for ``cluster`` (call, confidence, drivers).

    Built from :func:`agent.verdict.opening_interpretation` +
    :func:`agent.verdict.verdict_for_cluster`. States the cell-type call, the
    confidence band, the driving canonical markers with their real glm_coef /
    pearson, and — where present — the panel-absence context (c2 Stromal). Cites no
    live literature (the fallback path has no network); the live loop fills PMIDs.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    op = opening_interpretation(cluster)
    drivers = op.driving_markers
    notes = op.offpanel_notes

    if drivers:
        lead = (
            f"{op.cluster} reads as {op.cell_type} — {op.confidence} confidence, "
            f"driven by {_driver_phrase(drivers)}."
        )
    else:
        lead = (
            f"{op.cluster} reads as {op.cell_type} — {op.confidence} confidence; "
            f"no canonical marker drives the call, so re-check this."
        )

    verify_line = (
        " verify=TRUE — the signal is thin, re-check this call."
        if op.verify
        else ""
    )
    offpanel_line = _offpanel_line(notes, op.cell_type)

    text = lead + offpanel_line + verify_line

    sources = _marker_sources(drivers) + _offpanel_sources(notes)
    sidecar = _sidecar(drivers, notes)
    pin = drivers[0].gene if drivers else None

    return AgentResponse(
        text=text,
        sources=sources,
        verify=op.verify,
        grounding=sidecar,
        pin_marker=pin,
        citations=(),
        note_written=None,
        used_fallback=True,
        opening=True,
    )


# --------------------------------------------------------------------------- #
# Canned demo-beat Q&A (answered from verdict/data only)
# --------------------------------------------------------------------------- #
def _defines_answer(v: ClusterVerdict) -> AgentResponse:
    """'What defines this cluster' — the driving markers and their numbers."""
    drivers = v.opening.driving_markers
    notes = v.offpanel_notes
    if drivers:
        body = (
            f"{v.cluster} is called {v.cell_type} ({v.confidence} confidence). "
            f"The call is defined by {_driver_phrase(drivers)}."
        )
    else:
        body = (
            f"{v.cluster} is called {v.cell_type} ({v.confidence} confidence), but "
            f"no canonical marker defines it here — re-check this."
        )
    body += _offpanel_line(notes, v.cell_type)
    if v.verify:
        body += " verify=TRUE — re-check this call."
    return AgentResponse(
        text=body,
        sources=_marker_sources(drivers) + _offpanel_sources(notes),
        verify=v.verify,
        grounding=_sidecar(drivers, notes),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


def _doublet_answer(v: ClusterVerdict) -> AgentResponse:
    """'Could this be a doublet' — answered honestly from spatial specificity.

    Grounded in the driver's pearson (spatial specificity) only; no invented
    doublet score. High driver pearson argues the signal is spatially coherent
    rather than a mix; a low pearson / verify flag is stated as the caveat.
    """
    drivers = v.opening.driving_markers
    if not drivers:
        body = (
            f"{v.cluster} has no canonical driver, so a doublet cannot be ruled in "
            f"or out from the markers alone — re-check this."
        )
        return AgentResponse(
            text=body,
            sources=(),
            verify=True,
            grounding=_sidecar((), ()),
            used_fallback=True,
        )
    d = drivers[0]
    coherent = d.pearson >= _COHERENT_PEARSON
    if coherent and not v.verify:
        body = (
            f"Unlikely to be a pure doublet: the driver {d.gene} "
            f"(glm_coef {_FMT.format(d.glm_coef)}, pearson {_FMT.format(d.pearson)}) "
            f"is spatially coherent, which argues for a real {v.cell_type} "
            f"population rather than a mix. Confidence {v.confidence}."
        )
    else:
        body = (
            f"Cannot rule out a doublet from the markers alone: {d.gene} "
            f"(glm_coef {_FMT.format(d.glm_coef)}, pearson {_FMT.format(d.pearson)}) "
            f"is the driver, but the spatial signal is not strong enough to be sure. "
            f"verify=TRUE — re-check this."
        )
    return AgentResponse(
        text=body,
        sources=_marker_sources((d,)),
        verify=v.verify or not coherent,
        grounding=_sidecar((d,), ()),
        pin_marker=d.gene,
        used_fallback=True,
    )


def _confidence_answer(v: ClusterVerdict) -> AgentResponse:
    """'How confident / why this confidence' — the band and what drives it."""
    drivers = v.opening.driving_markers
    if drivers:
        d = drivers[0]
        body = (
            f"{v.cluster} {v.cell_type}: {v.confidence} confidence. The band comes "
            f"from the driving marker {d.gene} (glm_coef {_FMT.format(d.glm_coef)}, "
            f"pearson {_FMT.format(d.pearson)})."
        )
    else:
        body = (
            f"{v.cluster} {v.cell_type}: {v.confidence} confidence — no canonical "
            f"marker drives it, so re-check this."
        )
    if v.small_n:
        body += " This is a fragile cluster (<=2 assigned markers)."
    if v.verify:
        body += " verify=TRUE — re-check this call."
    return AgentResponse(
        text=body,
        sources=_marker_sources(drivers),
        verify=v.verify,
        grounding=_sidecar(drivers, ()),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


def _offpanel_answer(v: ClusterVerdict) -> Optional[AgentResponse]:
    """'What about <missing canonical>' — the panel-absence explanation.

    Only meaningful where the cluster carries off-panel canonical markers (c2). For
    a cluster with no off-panel notes, returns None (caller uses generic fallback).
    """
    notes = v.offpanel_notes
    if not notes:
        return None
    genes = ", ".join(n.gene for n in notes)
    drivers = v.opening.driving_markers
    body = (
        f"{genes} are canonical {v.cell_type.replace('_', ' ')} markers that are "
        f"off-panel — they were never measured in this panel, so their absence is "
        f"not evidence against the {v.cell_type} call. The call stands on the "
        f"on-panel markers: {_driver_phrase(drivers)}."
    )
    return AgentResponse(
        text=body,
        sources=_offpanel_sources(notes) + _marker_sources(drivers),
        verify=v.verify,
        grounding=_sidecar(drivers, notes),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


# Intent -> builder. Each builder takes a ClusterVerdict and returns an
# AgentResponse (or None to defer to the generic fallback).
_INTENT_BUILDERS = {
    "defines": _defines_answer,
    "doublet": _doublet_answer,
    "confidence": _confidence_answer,
    "offpanel": _offpanel_answer,
}

# Normalized-intent keyword table. First matching intent wins (checked in order),
# so more specific beats sit ahead of generic ones.
_INTENT_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("doublet", ("doublet", "mixture", "mixed cell", "two cell", "contamination")),
    (
        "offpanel",
        ("off-panel", "off panel", "missing marker", "not measured", "absent",
         "col1a1", "why no", "where is"),
    ),
    (
        "defines",
        ("what defines", "define", "what markers", "which markers", "what drives",
         "what is this", "characteri"),
    ),
    (
        "confidence",
        ("how confident", "confidence", "how sure", "why very high", "why high",
         "why medium", "why low", "how strong", "trust"),
    ),
)


def _classify(query: str) -> Optional[str]:
    """Map a free-text query to a canned intent, or None if nothing matches."""
    q = (query or "").strip().lower()
    if not q:
        return None
    for intent, keywords in _INTENT_KEYWORDS:
        for kw in keywords:
            if kw in q:
                return intent
    return None


def fallback_answer(query: str, cluster: str) -> Optional[AgentResponse]:
    """Canned, grounded answer for a recognized demo-beat question, else ``None``.

    Recognizes a small set of intents (what defines this cluster, could this be a
    doublet, how confident, and — for c2 — the off-panel-absence question) and
    answers each purely from the verdict/data. Returns ``None`` when the query does
    not match a canned intent, OR when the matched builder has nothing to say for
    this cluster (e.g. the off-panel intent on a cluster with no off-panel notes),
    so the caller can fall back to :func:`generic_fallback`.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    intent = _classify(query)
    if intent is None:
        return None
    v = verdict_for_cluster(cluster)
    builder = _INTENT_BUILDERS[intent]
    return builder(v)


# --------------------------------------------------------------------------- #
# Generic grounded fallback (last resort)
# --------------------------------------------------------------------------- #
def generic_fallback(cluster: str) -> AgentResponse:
    """Last-resort grounded response: the call, confidence, and top markers.

    Always returns a fully grounded :class:`AgentResponse` for any known cluster —
    the demo never shows a spinner or an error. Numbers trace to jazzPanda via the
    verdict; off-panel context (c2) is included; verify is preserved.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    v = verdict_for_cluster(cluster)
    drivers = v.opening.driving_markers
    notes = v.offpanel_notes

    if drivers:
        body = (
            f"{v.cluster} is called {v.cell_type} at {v.confidence} confidence, "
            f"driven by {_driver_phrase(drivers)}."
        )
    else:
        body = (
            f"{v.cluster} is called {v.cell_type} at {v.confidence} confidence; no "
            f"canonical marker drives the call."
        )
    body += _offpanel_line(notes, v.cell_type)
    if v.verify:
        body += " verify=TRUE — re-check this call."

    return AgentResponse(
        text=body,
        sources=_marker_sources(drivers) + _offpanel_sources(notes),
        verify=v.verify,
        grounding=_sidecar(drivers, notes),
        pin_marker=drivers[0].gene if drivers else None,
        citations=(),
        note_written=None,
        used_fallback=True,
        opening=False,
    )


# --------------------------------------------------------------------------- #
# FallbackStore — a thin, stateless facade over the three fallback layers.
# --------------------------------------------------------------------------- #
class FallbackStore:
    """Facade the loop calls: opening, canned-answer match, or generic fallback.

    Stateless and deterministic. Holds no data of its own — every response is
    (re)built from the verdict engine on demand, so it can never go stale relative
    to the jazzPanda output the way a frozen JSON blob could.
    """

    def opening(self, cluster: str) -> AgentResponse:
        """Pre-baked opening interpretation for ``cluster``."""
        return fallback_opening(cluster)

    def match(self, query: str, cluster: str) -> AgentResponse:
        """Best available fallback for a free-text query on ``cluster``.

        Tries the canned demo-beat answers first; on no match, returns the generic
        grounded fallback so the caller ALWAYS gets a valid, grounded response.
        """
        canned = fallback_answer(query, cluster)
        if canned is not None:
            return canned
        return generic_fallback(cluster)

    def generic(self, cluster: str) -> AgentResponse:
        """The last-resort grounded fallback for ``cluster``."""
        return generic_fallback(cluster)


__all__ = [
    "fallback_opening",
    "fallback_answer",
    "generic_fallback",
    "FallbackStore",
]
