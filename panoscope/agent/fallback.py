"""Deterministic, pre-baked, GROUNDED fallbacks so a slow or failed API never
breaks the demo.

Every response is a pure function of :mod:`pipeline.v1_verdicts`: each number traces
to a jazzPanda column, each off-panel line to the panel-absence primitive, and no
PMID/DOI is ever stated (no network). Each carries the live answer's
``sources`` / ``verify`` / ``grounding`` envelope, so it passes the grounding checker.

Public API
----------
- :func:`fallback_opening`, the pre-baked opening interpretation for a cluster.
- :func:`fallback_answer`, a canned answer for a recognized demo-beat question,
  or ``None`` when nothing matches (the caller then uses the generic fallback).
- :func:`generic_fallback`, the last-resort grounded template for any cluster.

Off-panel canonical markers are context that never down-weights the call and never
carries a statistic; uncertain clusters keep ``verify = True`` and say so.
"""

from __future__ import annotations

import re

from typing import Optional

from panoscope.agent import discriminate
from panoscope.agent.types import (
    AgentResponse,
    ClusterVerdict,
    GroundingSidecar,
    MarkerEvidence,
    OffPanelNote,
    Source,
)
from panoscope.agent.compose import composed_verdict_for, computed_call
from panoscope.data.study import NOT_YET_CALLED

# Two decimals is inside the grounding checker's 1e-2 tolerance. One constant so
# prose and sidecar never drift.
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
    token) but contribute NO number, their absence is not a statistic. No PMIDs
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
    # Number agreement: most units carry ONE off-panel canonical marker.
    one = len(notes) == 1
    return (
        f" Note on absence: {genes} {_canonical_clause(cell_type, one)} "
        f"off-panel (never measured here), so "
        f"{'its' if one else 'their'} absence is not evidence against the call."
    )


def _computed_call_prefix(cluster: str, shown: str) -> str:
    """One sentence saying the rival check reads the COMPUTED call, when it does.

    After an override, ``agent.discriminate`` still answers about the computed call
    (the annotation indexes canonical sets by it); say so rather than relabel.
    """
    from panoscope.agent import annotation

    computed = computed_call(cluster)
    if not shown or shown == computed:
        return ""
    why = ("" if annotation.canonical_markers(shown)
           else f", for which this study's annotation carries no canonical "
                f"marker set")
    return (f"Your call for {cluster} is {shown}{why}. The rival check below "
            f"reads the computed call, {computed}. ")


def _whose_call(cluster: str, shown: str) -> str:
    """" -- your call, replacing the computed X" when the biologist overrode it.

    Empty when the shown call IS the computed one, so an unedited cluster reads
    exactly as it always did.
    """
    computed = computed_call(cluster)
    if not shown or shown == computed:
        return ""
    return f" (your call, replacing the computed {computed})"


def _not_called(v: ClusterVerdict, opening: bool = False) -> AgentResponse:
    """A unit with no band: no confidence word and no re-check -- there is no claim."""
    who = ("is not yet called" if v.cell_type == NOT_YET_CALLED
           else f"is your call, {v.cell_type}, which the build had no label to assess")
    return AgentResponse(
        text=(f"{v.cluster} {who}: there is no band and nothing to re-check. The "
              "markers table lists what was measured here; the call is yours."),
        sources=(), verify=False, grounding=_sidecar((), ()),
        used_fallback=True, opening=opening,
    )


# --------------------------------------------------------------------------- #
# Opening interpretation fallback
# --------------------------------------------------------------------------- #
def fallback_opening(cluster: str) -> AgentResponse:
    """Pre-baked opening interpretation for ``cluster`` (call, confidence, drivers).

    Built from :func:`agent.compose.composed_verdict_for` (override applied). Cites
    no literature; the live loop fills PMIDs.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    v = composed_verdict_for(cluster)
    if not v.confidence:
        return _not_called(v, opening=True)
    op = v.opening
    drivers = op.driving_markers
    notes = op.offpanel_notes
    # The biologist's call WITH the tension visible: say whose call it is and what
    # the method produced, never a silent swap.
    whose = _whose_call(cluster, op.cell_type)

    if drivers:
        lead = (
            f"{op.cluster} reads as {op.cell_type}{whose}, {op.confidence} "
            f"confidence, driven by {_driver_phrase(drivers)}."
        )
    else:
        lead = (
            f"{op.cluster} reads as {op.cell_type}{whose}, {op.confidence} "
            f"confidence; no canonical marker drives the call, so re-check this."
        )

    verify_line = (
        " verify=TRUE, the signal is thin, re-check this call."
        if op.verify
        else ""
    )
    offpanel_line = _offpanel_line(notes, op.cell_type)

    text = lead + offpanel_line + verify_line

    sources = _marker_sources(drivers) + _offpanel_sources(notes)
    sidecar = _sidecar(drivers, notes)

    # Proactive "what would settle it": append only when there is a genuine rival
    # to name — a shaky call (verify) or real mixed signal (a rival marker peaks
    # here). Deterministic + grounded, so it survives with no network.
    d = discriminate.discriminate(cluster)
    if d.alt_B is not None and (op.verify or d.b_here):
        text += discriminate.settle_line(d)
        d_sources, d_numbers, d_markers = _discrimination_grounding(d, include_support=False)
        sources = sources + d_sources
        sidecar = GroundingSidecar(
            numbers=sidecar.numbers + d_numbers,
            markers=sidecar.markers + d_markers,
            pmids=sidecar.pmids,
            notes_used=sidecar.notes_used,
        )

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
def _call_answer(v: ClusterVerdict, lead: str) -> AgentResponse:
    """``lead`` about the call, plus its off-panel context and verify suffix, grounded
    in the driving markers' numbers."""
    drivers = v.opening.driving_markers
    notes = v.offpanel_notes
    body = lead + _offpanel_line(notes, v.cell_type)
    if v.verify:
        body += " verify=TRUE, re-check this call."
    return AgentResponse(
        text=body,
        sources=_marker_sources(drivers) + _offpanel_sources(notes),
        verify=v.verify,
        grounding=_sidecar(drivers, notes),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


def _defines_answer(v: ClusterVerdict) -> AgentResponse:
    """'What defines this cluster', the driving markers and their numbers."""
    drivers = v.opening.driving_markers
    if drivers:
        return _call_answer(v, (
            f"{v.cluster} is called {v.cell_type} ({v.confidence} confidence). "
            f"The call is defined by {_driver_phrase(drivers)}."
        ))
    return _call_answer(v, (
        f"{v.cluster} is called {v.cell_type} ({v.confidence} confidence), but "
        f"no canonical marker defines it here, re-check this."
    ))


def _doublet_answer(v: ClusterVerdict) -> AgentResponse:
    """'Could this be a doublet', answered honestly from spatial specificity.

    Grounded in the driver's pearson (spatial specificity) only; no invented
    doublet score. High driver pearson argues the signal is spatially coherent
    rather than a mix; a low pearson / verify flag is stated as the caveat.
    """
    drivers = v.opening.driving_markers
    if not drivers:
        body = (
            f"{v.cluster} has no canonical driver, so a doublet cannot be ruled in "
            f"or out from the markers alone, re-check this."
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
            f"verify=TRUE, re-check this."
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
    """'How confident / why this confidence', the band and what drives it."""
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
            f"{v.cluster} {v.cell_type}: {v.confidence} confidence, no canonical "
            f"marker drives it, so re-check this."
        )
    if v.small_n:
        body += " This is a fragile cluster (<=2 assigned markers)."
    if v.verify:
        body += " verify=TRUE, re-check this call."
    return AgentResponse(
        text=body,
        sources=_marker_sources(drivers),
        verify=v.verify,
        grounding=_sidecar(drivers, ()),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


def _canonical_clause(cell_type: str, one: bool) -> str:
    """"are canonical Stromal markers that are" -- or just "are" when it cannot.

    Canonicity is an annotation claim; after an override the annotation may not name
    the call, so only the canonicity is dropped (off-panel stays true).
    """
    from panoscope.agent import annotation

    verb = "is" if one else "are"
    if not annotation.canonical_markers(cell_type):
        return verb
    return (f"{verb}{' a' if one else ''} canonical "
            f"{cell_type.replace('_', ' ')} {'marker' if one else 'markers'} "
            f"that {verb}")


def _offpanel_answer(v: ClusterVerdict) -> Optional[AgentResponse]:
    """'What about <missing canonical>', the panel-absence explanation.

    Only meaningful where the cluster carries off-panel canonical markers (c2). For
    a cluster with no off-panel notes, returns None (caller uses generic fallback).
    """
    notes = v.offpanel_notes
    if not notes:
        return None
    genes = ", ".join(n.gene for n in notes)
    drivers = v.opening.driving_markers
    one = len(notes) == 1          # see _offpanel_line: most units carry exactly one
    body = (
        f"{genes} {_canonical_clause(v.cell_type, one)} off-panel, "
        f"{'it was' if one else 'they were'} never measured in this panel, so "
        f"{'its' if one else 'their'} absence is not evidence against the "
        f"{v.cell_type} call. The call stands on the on-panel markers: "
        f"{_driver_phrase(drivers)}."
    )
    return AgentResponse(
        text=body,
        sources=_offpanel_sources(notes) + _marker_sources(drivers),
        verify=v.verify,
        grounding=_sidecar(drivers, notes),
        pin_marker=drivers[0].gene if drivers else None,
        used_fallback=True,
    )


# --------------------------------------------------------------------------- #
# "What would settle it" — the discriminator, as a grounded fallback answer
# --------------------------------------------------------------------------- #
# Free-text cell-type hints -> canonical map key. Ordered: 'myoepithelial' is
# checked before generic terms so it wins over a bare 'epithelial'-like match.
_TYPE_HINTS: tuple[tuple[str, str], ...] = (
    ("myoepithelial", "Myoepithelial"),
    ("tumor", "Tumor"),
    ("tumour", "Tumor"),
    ("fibroblast", "Stromal"),
    ("stromal", "Stromal"),
    ("caf", "Stromal"),
    ("macrophage", "Macrophages"),
    ("tam", "Macrophages"),
    ("t cell", "T_Cells"),
    ("t-cell", "T_Cells"),
    ("b cell", "B_Cells"),
    ("b-cell", "B_Cells"),
    ("endothelial", "Endothelial"),
    ("endothelium", "Endothelial"),
    ("dendritic", "Dendritic"),
    ("mast", "Mast_Cells"),
)


def _alts_from_query(query: str) -> list[str]:
    """All canonical cell types named in a free-text question, in table order.

    The fallback has no LLM to parse the query, so it scans for cell-type names the
    biologist may have named (e.g. 'is this tumor or myoepithelial'). The caller
    drops the cluster's own call and takes the rest as the alternative; when none
    remains, the discriminator derives the rival from the cluster's own markers.
    """
    # Word boundaries, not substrings: "t cell" is inside "mast cells".
    q = (query or "").lower()
    out: list[str] = []
    for hint, key in _TYPE_HINTS:
        if key in out:
            continue
        if _names_type(q, hint):
            out.append(key)
    return out


def _names_type(q: str, hint: str) -> bool:
    """Whether lower-cased ``q`` names ``hint`` as a word (optional plural). Shared by every caller."""
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(hint)}s?(?![a-z0-9])", q))


# "c1 is Hepatocytes, not Tumor": the rival named in a pushback, so an unknown rival
# is declined by name instead of the recognised CALL being taken as the alternative.
_CONTRAST_RE = re.compile(
    # One optional word between modal and "be" ("could THIS be X rather than Y").
    r"\b(?:is|are|looks?\s+like|(?:could|might|must)\s+(?:\w+\s+)?be)\s+"
    # {2,} = three-character minimum, so "CAF" matches; two-letter names miss on purpose.
    r"(?:an?\s+|the\s+)?(?P<rival>[A-Za-z][A-Za-z0-9-]{2,}(?:\s+cells?)?)"
    r"\s*,?\s+(?:not|rather\s+than|instead\s+of)\b",
    re.IGNORECASE,
)

# Words the contrast pattern can capture that are not a cell type.
_NOT_A_TYPE: frozenset[str] = frozenset({
    "this", "that", "them", "they", "there", "here", "definitely", "clearly",
    "really", "probably", "actually", "just", "more", "very", "still", "also",
    "cell", "cells", "something", "anything", "likely", "certainly", "obviously",
    # Three-letter words the looser rival pattern can now reach.
    "not", "the", "one", "two", "its", "all", "any", "now", "too", "yet", "but",
    "and", "for", "was", "are", "has", "had", "its", "our", "out", "off",
})


def _contrasted_rival(query: str) -> Optional[str]:
    """The rival named in an "X, not Y" pushback, or None.

    Returned verbatim so ``discriminate`` -- which owns every decision about
    what a cell-type name resolves to -- can answer it, including by declining.
    """
    m = _CONTRAST_RE.search(query or "")
    if m is None:
        return None
    rival = " ".join(m.group("rival").split())
    return None if rival.lower() in _NOT_A_TYPE else rival


def _discrimination_grounding(d, include_support: bool = True):
    """Grounded (sources, numbers, marker-vocab) for a Discrimination.

    Numbers are attached ONLY to markers that peak in this cluster (supporting_A /
    b_here); elsewhere / off-panel markers contribute vocabulary but no statistic.
    ``include_support=False`` omits supporting_A (used when the caller already
    grounded the drivers, e.g. the opening).
    """
    numbered = (*(d.supporting_A if include_support else ()), *d.b_here)
    numbers: list[tuple[str, str, float]] = []
    for m in numbered:
        numbers.append((m.gene, "glm_coef", float(m.glm_coef)))
        numbers.append((m.gene, "pearson", float(m.pearson)))
    marker_names = tuple(
        m.gene
        for m in (
            *(d.supporting_A if include_support else ()),
            *d.b_here,
            *d.b_elsewhere,
            *d.offpanel_absent,
        )
    )
    sources = tuple(
        Source(
            kind="jz",
            ref=m.gene,
            value=_FMT.format(m.glm_coef),
            detail=f"glm_coef {_FMT.format(m.glm_coef)}, pearson {_FMT.format(m.pearson)}",
        )
        for m in numbered
    ) + tuple(
        Source(kind="panel", ref=m.gene, value="off-panel", detail="off-panel — never measured")
        for m in d.offpanel_absent
    )
    return sources, tuple(numbers), marker_names


def _settle_answer(v: ClusterVerdict, query: str) -> AgentResponse:
    """Grounded 'what would settle it' answer: name the discriminating markers.

    Derives the rival from the biologist's wording if present, else from the
    cluster's own markers. Quotes numbers only for markers that peak here, flags
    off-panel genes (never recommends an experiment), and always clears the floor.
    """
    q = (query or "").lower()
    named = [(hint, key) for hint, key in _TYPE_HINTS if _names_type(q, hint)]
    distinct = [key for hint, key in named if key != v.cell_type]
    if distinct:
        alt = distinct[0]
    else:
        # An unrecognised contrasted rival is declined BY NAME, ahead of the
        # synonym branch below.
        alt = _contrasted_rival(query)
    if alt is None:
        # the biologist may have named a SUBTYPE/synonym of the call itself (e.g. CAF for
        # Stromal): pass their word so the discriminator answers it as a refinement, not
        # by auto-deriving "no competing hypothesis".
        same = [hint for hint, key in named if key == v.cell_type]
        alt = same[0] if same else None
    d = discriminate.discriminate(v.cluster, alt)
    sources, numbers, marker_names = _discrimination_grounding(d, include_support=True)
    sidecar = GroundingSidecar(numbers=numbers, markers=marker_names, pmids=(), notes_used=())
    pin = d.supporting_A[0].gene if d.supporting_A else None
    return AgentResponse(
        text=_computed_call_prefix(v.cluster, v.cell_type)
             + discriminate.settle_summary(d),
        sources=sources,
        verify=v.verify,
        grounding=sidecar,
        pin_marker=pin,
        citations=(),
        note_written=None,
        used_fallback=True,
        opening=False,
        tools_used=("discriminate_call",),  # so the UI shows the rival-test fired offline too
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
        "settle",
        ("settle", "could this be", "could it be", "might it be", "might this be",
         "distinguish", "differentiate", "tell apart", "tell it from", "rule out",
         "rule in", "versus", " vs ", "confused with", "what separates",
         "what would decide", "how do i tell", "how to tell"),
    ),
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
    # No canned intent matched, but if the biologist named a specific cell type,
    # read it as a "what would settle it" comparison against that alternative.
    if _alts_from_query(q) or _contrasted_rival(query):
        return "settle"
    return None


def fallback_answer(query: str, cluster: str) -> Optional[AgentResponse]:
    """Canned, grounded answer for a recognized demo-beat question, else ``None``.

    Recognizes a small set of intents (what defines this cluster, could this be a
    doublet, how confident, and, for c2, the off-panel-absence question) and
    answers each purely from the verdict/data. Returns ``None`` when the query does
    not match a canned intent, OR when the matched builder has nothing to say for
    this cluster (e.g. the off-panel intent on a cluster with no off-panel notes),
    so the caller can fall back to :func:`generic_fallback`.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    intent = _classify(query)
    if intent is None:
        return None
    v = composed_verdict_for(cluster)
    if not v.confidence:
        return _not_called(v)
    if intent == "settle":
        # The settle answer needs the query text (to pick up a named alternative),
        # so it is built here rather than through the ClusterVerdict-only table.
        return _settle_answer(v, query)
    builder = _INTENT_BUILDERS[intent]
    return builder(v)


# --------------------------------------------------------------------------- #
# Generic grounded fallback (last resort)
# --------------------------------------------------------------------------- #
def generic_fallback(cluster: str) -> AgentResponse:
    """Last-resort grounded response: the call, confidence, and top markers.

    Always returns a fully grounded :class:`AgentResponse` for any known cluster,
    the demo never shows a spinner or an error. Numbers trace to jazzPanda via the
    verdict; off-panel context (c2) is included; verify is preserved.

    KeyError if ``cluster`` is unknown (propagated from the verdict engine).
    """
    v = composed_verdict_for(cluster)
    if not v.confidence:
        return _not_called(v)
    drivers = v.opening.driving_markers
    if drivers:
        return _call_answer(v, (
            f"{v.cluster} is called {v.cell_type} at {v.confidence} confidence, "
            f"driven by {_driver_phrase(drivers)}."
        ))
    return _call_answer(v, (
        f"{v.cluster} is called {v.cell_type} at {v.confidence} confidence; no "
        f"canonical marker drives the call."
    ))


# --------------------------------------------------------------------------- #
# match: canned answer if one fits, else the generic grounded fallback.
# --------------------------------------------------------------------------- #
def match(query: str, cluster: str) -> AgentResponse:
    """Best available fallback for a free-text query on ``cluster``.

    Tries the canned demo-beat answers first; on no match, returns the generic
    grounded fallback so the caller ALWAYS gets a valid, grounded response.
    """
    canned = fallback_answer(query, cluster)
    if canned is not None:
        return canned
    return generic_fallback(cluster)


__all__ = [
    "fallback_opening",
    "fallback_answer",
    "generic_fallback",
    "match",
]
