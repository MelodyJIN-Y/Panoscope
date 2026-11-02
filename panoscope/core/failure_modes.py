"""ARCHITECTURE §9.8's named failure modes -- as an INDEX of what already fires.

§9.8 asks for the modes to be "named, computed, more actionable than a low band".
Three of those four words were already true before this module existed. The system
computes these conditions today, in ``core/adjacency.py``, ``core/verdict.py``,
``core/rival.py``, ``core/concordance.py``, ``core/reconcile.py`` and
``workflows/de.py``, and every one of them is already *registered* -- as a rule id in
``types/trace.py::RULES`` or as a flag id in ``types/evidence.py::FLAGS``. What was
missing was only the naming: a biologist reading a record sees
``adjacency:bleed_candidate`` next to ``band:fragile_cap`` next to
``reconcile:tension`` and has no way to know that the first is §9.8's transcript
bleed, the second is a thin profile, and the third is not a unit-coherence question
at all.

**So this module detects nothing.** :data:`MODES` maps a biologist-facing failure
mode onto the registered ids that already emit it, and :func:`detect` reads a unit's
evidence and the traces a caller already holds. That is a deliberate refusal to write
the obvious alternative. A second implementation -- re-deriving "is this unit
spatially coherent" from the items here -- would be a second place a flag is decided,
and §9.9 is explicit that these are display flags whose firing rate has not been
measured; two implementations of an unmeasured flag is two firing rates.

**Two of §9.8's four modes have NO detector, and they are named rather than
quietly absent.** :data:`UNDETECTED` carries them. "Spatially incoherent cells" and
"a tissue region rather than a cell type" need a per-cell spatial dispersion
statistic that no method module in this repo reports, and inventing ids for them here
would be exactly the dead vocabulary ``FLAGS`` and ``RULES`` both have a stated rule
against -- a registry nobody emits is a traversal nobody can test. Naming them as
undetected is the honest half: a reviewer can see that the apparatus does not ask
these two, which is the same discipline the four-state lattice applies to a
measurement.

**``detected_by`` mixes two registries on purpose.** A failure mode is a fact about
the reading, and the reading is condemned by whichever layer noticed: a non-specific
marker is flagged on the ``EvidenceItem`` by the method layer
(``specificity:localises_elsewhere``) *and* by the spatial check in
``core/adjacency.py`` (``adjacency:specificity``), and a mode that listed only one of
them would go quiet whenever the other was the one that fired.
``test_every_detector_id_is_registered`` pins that every id lands in exactly one of
the two registries, so a rule renamed in ``RULES`` cannot leave a mode pointing at
nothing.

**No decision of its own, and no ordering of its own.** Every mode this module
reports is a re-read of a decision already in the record; noting it again would be
the duplicate that drifts, and ``core/record.py`` returns ``Traced`` with empty
decisions for the same reason. The result is in :data:`MODES` declaration order
rather than in firing order, because firing order depends on which trace a caller
passed first, which is a property of the call site and not of the study.
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
