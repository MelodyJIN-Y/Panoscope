"""ARCHITECTURE §9.8's named failure modes, as an INDEX of what already fires.

:data:`MODES` maps a biologist-facing failure mode onto the registered ids (rule ids in
``types/trace.py::RULES``, flag ids in ``types/evidence.py::FLAGS``) that already emit
it; :func:`detect` reads a unit's evidence and the caller's traces. It detects nothing
itself: a second implementation of an unmeasured display flag (§9.9) would be a second
firing rate.

- :data:`UNDETECTED` names the two §9.8 modes with no detector (spatially incoherent
  cells; a tissue region rather than a cell type): no method reports a per-cell
  dispersion statistic, and an unemitted id would be dead vocabulary.
- ``detected_by`` spans both registries, because a mode can be noticed by either layer;
  ``test_every_detector_id_is_registered`` pins each id to exactly one registry.
- No decisions of its own; results are in :data:`MODES` declaration order, not firing
  order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from panoscope.types.evidence import FLAGS, NormalisedEvidence
from panoscope.types.trace import RULES, Traced


@dataclass(frozen=True, slots=True)
class FailureMode:
    """One named way a reading can be wrong, and what already detects it.

    ``what`` and ``ask`` are human-readable and are NEVER matched on -- the same
    split ``Flag.what`` and ``Rule.what`` make. ``id`` is what code branches on.

    ``ask`` is the half §9.8 means by "more actionable than a low band": it names
    what the biologist should look at, not how worried to be. A mode that only said
    "this unit may be incoherent" would be a low band wearing a longer sentence.
    """

    id: str
    what: str
    ask: str
    detected_by: tuple[str, ...]   # REGISTERED ids, from RULES or from FLAGS


def _modes(*modes: FailureMode) -> dict[str, FailureMode]:
    return {m.id: m for m in modes}


# --------------------------------------------------------------------------- #
# The index. Every id on the right already fires somewhere in this package; see
# the module docstring for why nothing here computes one.
# --------------------------------------------------------------------------- #
MODES: dict[str, FailureMode] = _modes(
    FailureMode(
        "unit:transcript_bleed",
        "a marker supporting this unit also localises in a unit interdigitated with "
        "it -- ARCHITECTURE 9.8's transcript bleed from a neighbour, which only "
        "spatial data can tell apart from a genuine specificity problem",
        "look at the segmentation along the boundary between the two units before "
        "reading this marker as this unit's",
        ("adjacency:bleed_candidate",),
    ),
    FailureMode(
        "unit:mixed_profile",
        "a named alternative's canonical markers were ALSO observed here, so the "
        "unit carries genuine signal for two accounts at once -- 9.8's mixed "
        "profile, which over-segmentation and a real doublet population both produce",
        "check whether the unit splits at a finer clustering resolution before "
        "reading either account as the unit's identity",
        ("rival:ambiguous",),
    ),
    FailureMode(
        "unit:thin_profile",
        "the call rests on two or fewer markers that carry a number, so the band was "
        "capped and floored rather than read off the measurement alone",
        "look at the markers individually; one more assigned marker would move this "
        "call more than any re-reading of the ones that are here",
        ("band:fragile_cap", "band:fragile_floor"),
    ),
    FailureMode(
        "unit:unsupported_call",
        "nothing canonical carries the call -- either no canonical marker reported a "
        "number, or the leading one was measured and found absent here",
        "check whether the curated marker list for this label is thin before "
        "concluding the unit is not this type; the two read identically in a band",
        ("band:no_canonical_driver", "band:driver_nosig"),
    ),
    FailureMode(
        "marker:nonspecific",
        "a marker's spatial pattern is shared -- it tracks another lineage better, "
        "another gene shares its pattern, its specificity is weak, or it also "
        "localises in a unit that does not neighbour this one",
        "read the marker as supporting a programme rather than an identity until a "
        "more specific one is found",
        ("specificity:localises_elsewhere", "specificity:not_unique",
         "specificity:low", "adjacency:specificity"),
    ),
    FailureMode(
        "unit:replicate_divergence",
        "the replicates read a subject under this unit differently -- batch effect, "
        "section quality or real biology, and 9.10 does not claim which",
        "compare the diverging subject across sections before reading the pooled "
        "number; agreement would not have raised confidence either",
        ("concordance:divergent",),
    ),
    FailureMode(
        "result:composition_confounded",
        "this unit's share of cells differs across the arms being compared, so a "
        "gene reads as up when the cells expressing it merely became more numerous",
        "get a within-cell-type comparison; no re-reading of this table separates "
        "the two claims",
        ("de:composition_confounded",),
    ),
    FailureMode(
        "result:cross_method_tension",
        "two results pull against each other about one subject -- a real conflict "
        "where they ask the same question, a tension where they do not (9.11)",
        "read which method measured what; neither side's silence is on the table "
        "here, because a side that could not see the subject lands on "
        "not-comparable instead",
        ("reconcile:disagree", "reconcile:tension", "de:coherence_tension"),
    ),
)

#: §9.8's modes this system does **not** detect, named rather than silently absent.
#:
#: Both need a per-cell spatial statistic -- dispersion of a unit's cells, or the
#: shape of the region they occupy -- that no registered method module reports. They
#: are listed here so a reviewer can see which of §9.8's questions the apparatus does
#: not ask, instead of inferring from an empty result that it asked and found nothing.
UNDETECTED: Final[tuple[str, ...]] = (
    "spatially incoherent cells: no method module reports per-cell dispersion, so "
    "whether a unit's cells sit together is never asked",
    "a tissue region rather than a cell type: distinguishing the two needs the shape "
    "of the region a unit occupies, which no registered method reports",
)


class UnregisteredDetector(KeyError):
    """Raised for a ``detected_by`` id that is in neither ``RULES`` nor ``FLAGS``.

    Never guessed at and never tolerated: a mode pointing at an id nothing emits is
    a failure the review surface would advertise and the system could never raise,
    which is worse than not naming it at all.
    """


def _known(detector: str) -> bool:
    return detector in RULES or detector in FLAGS


def detect(ev: NormalisedEvidence,
           *traced: Traced[Any]) -> Traced[tuple[FailureMode, ...]]:
    """Which named failure modes this unit's evidence and decisions already raised.

    ``*traced`` is whatever the caller already holds back from ``core/`` -- the
    verdict, the rival tests, the adjacency checks, the reconciliation. They are read
    for their ``Decision.rule`` ids only; nothing is recomputed and no number is
    read, so passing more traces can add a mode and can never change one.

    The evidence is read for ``EvidenceItem.flags``, which is the other half: a
    non-specific marker is condemned by the method layer at load, before any ``core``
    function has seen it, and a traversal over decisions alone would miss every unit
    whose specificity problem was caught there.

    Returns ``Traced`` with no decisions. See the module docstring: every mode here
    restates a decision the record already carries, and a second line saying the same
    thing is the copy that drifts.
    """
    seen = {f for i in ev.items for f in i.flags}
    seen |= {d.rule for t in traced for d in t.decisions}
    fired = tuple(m for m in MODES.values()
                  if any(d in seen for d in m.detected_by))
    return Traced(fired)


def explain(mode_id: str) -> Traced[FailureMode]:
    """One mode by id, with its detectors checked against the live registries.

    Lookup rather than construction, and it raises twice over. An unregistered mode
    id raises :class:`UnregisteredDetector` for the reason ``workflows.get`` raises
    ``UnregisteredWorkflow`` rather than defaulting: a near match would name a
    failure the biologist is not looking at. A registered mode whose detector left
    ``RULES`` or ``FLAGS`` raises for the reason above it -- a mode nothing can emit
    is a promise the surface would make and the system could never keep.
    """
    mode = MODES.get(mode_id)
    if mode is None:
        raise UnregisteredDetector(
            f"no failure mode registered as {mode_id!r}; known: {sorted(MODES)}. "
            f"Undetected by design: {len(UNDETECTED)} of ARCHITECTURE 9.8's modes"
        )
    orphans = tuple(d for d in mode.detected_by if not _known(d))
    if orphans:
        raise UnregisteredDetector(
            f"{mode_id} names detector(s) {list(orphans)} that are in neither RULES "
            "nor FLAGS; a failure mode nothing emits is a badge nobody can test"
        )
    return Traced(mode)
