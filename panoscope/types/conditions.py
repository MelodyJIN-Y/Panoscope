"""Study conditions: what was true of the experiment, and *how we know it*.

Intake writes this object (ARCHITECTURE §6.2) and every applicability check reads it, so
a value never travels alone: it is a :class:`Condition`, value plus how it was obtained.

The §6.3 line:

- ``BLOCKS_THE_STUDY`` -- tissue and platform/assay, plus the measured set. Unknown here
  is a validation failure.
- ``DECLARED_UNKNOWN_OK`` -- fixation, segmentation, ``modality`` and ``method``. These
  propagate: a check reading one returns *cannot be settled* and names it.

Every ``Condition`` field is in exactly one tuple
(``test_every_condition_field_is_classified_exactly_once``).

Clustering mode, §6.3's fourth blocker, is not a field here: it blocks only for
multi-sample studies, and the sample count belongs to the platform adapter
(``PlatformAdapter.samples`` / ``clustering_mode``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from panoscope.types.measured import EntityKind, FrameRef


class Provenance(StrEnum):
    """How a condition was obtained. Four states, for the same reason §9.5 has four."""

    STATED = "stated"      # the biologist said it
    READ = "read"          # parsed from a file -- the panel, the manifest, the header
    INFERRED = "inferred"  # the model proposed it and the biologist CONFIRMED it
    UNKNOWN = "unknown"    # explicitly not known -- never a silent default


class Blocked(ValueError):
    """Intake is not finished: a blocking condition is unknown (§6.3).

    Distinct from a plain ``ValueError`` because it is not a programming error. It is
    the expected outcome of an incomplete study, and §6.4's checklist -- not a
    traceback -- is its user-facing form, so intake has to be able to catch exactly
    this and render the missing items.
    """


@dataclass(frozen=True, slots=True)
class Condition:
    """A condition value plus HOW it was obtained. Intake writes these (§6.2).

    Named ``Condition`` rather than ``Field`` as IMPLEMENTATION §2.7 has it: every
    sibling module in this package imports ``dataclasses.field``, and a contract type
    that shadows it is a collision waiting for the first module that needs both.
    """

    value: str | None
    provenance: Provenance

    def __post_init__(self) -> None:
        """``value is None`` **iff** ``provenance is UNKNOWN``.

        The same structural trick as ``statistic is None`` iff unmeasured: it makes
        the two incoherent states unrepresentable instead of merely wrong. A value
        carried under ``UNKNOWN`` is a guess wearing a disclaimer -- downstream reads
        the provenance, drops the field, and the guess survives in the serialised
        record anyway. A ``None`` under ``STATED`` is the mirror: it reads as *known*
        to every applicability check, which then compares against nothing and passes.
        """
        if (self.value is None) is not (self.provenance is Provenance.UNKNOWN):
            raise ValueError(
                f"incoherent condition {self!r}: a value exists iff it is known. "
                "Use Provenance.UNKNOWN with value=None, or give a value."
            )

    @property
    def known(self) -> bool:
        return self.provenance is not Provenance.UNKNOWN


UNKNOWN: Final = Condition(None, Provenance.UNKNOWN)
"""The one legitimate spelling of a declared unknown. Frozen, so sharing is safe."""


# --------------------------------------------------------------------------- #
# The §6.3 table, as two constants. A new field must land in one of them.
# `frames` is in neither: it is not a Condition and has no provenance to check --
# it blocks structurally, by being empty, because it *is* the panel file.
# --------------------------------------------------------------------------- #
BLOCKS_THE_STUDY: Final[tuple[str, ...]] = ("assay", "tissue")
DECLARED_UNKNOWN_OK: Final[tuple[str, ...]] = (
    "modality", "method", "segmentation", "fixation",
)


def gene_frames_admit(frames: tuple[FrameRef, ...]) -> bool:
    """True when there is a GENE frame and EVERY one admits something.

    The GENE frames, not the tuple's truthiness: a store declaring any ``clusters:``
    frame keeps the tuple non-empty beside an empty or absent gene frame. And every
    one, because a gene is in frame only when all of them admit it
    (``measured._state_for``): one empty gene frame empties the measured set, every
    absence then resolves to NEVER_MEASURED and the panel-absence rule discriminates
    nothing -- §6.3's structural blocker.
    """
    genes = [f for f in frames if f.kind is EntityKind.GENE]
    return bool(genes) and all(f.size for f in genes)


@dataclass(frozen=True, slots=True)
class Conditions:
    """The conditions a study was produced under. No defaults, by design.

    A default would be a silent one, and §6.2's whole point is that no condition is
    ever filled in without a provenance saying who filled it in. Construct it with
    ``UNKNOWN`` where that is the honest answer.
    """

    assay: Condition
    modality: Condition
    tissue: Condition          # an ontology id, not prose -- it is the scoping key
    method: Condition
    segmentation: Condition
    fixation: Condition
    frames: tuple[FrameRef, ...]   # replaces panel_hash + panel_genes

    def __post_init__(self) -> None:
        if not gene_frames_admit(self.frames):
            empty = [f.name for f in self.frames
                     if f.kind is EntityKind.GENE and not f.size]
            raise Blocked(
                "the measured set is structural (§6.3): "
                + (f"the gene frame(s) {', '.join(empty)} admit nothing"
                   if empty else "Conditions declares no gene capability frame")
                + ", so every absence under it resolves to NEVER_MEASURED and the "
                  "panel-absence rule discriminates nothing"
            )
        missing = [n for n in BLOCKS_THE_STUDY if not getattr(self, n).known]
        if missing:
            raise Blocked(
                f"study blocked: {', '.join(missing)} unknown (§6.3). Ask the "
                "biologist; do not record a guess as inferred."
            )

    @property
    def declared_unknown(self) -> tuple[str, ...]:
        """Which fields are explicitly not known, in declaration order.

        The propagating half of "``unknown`` is first-class". A check that reads one
        of these must return *cannot be settled* and NAME what is missing. Names
        rather than a flag is what makes the caveat actionable: "segmentation
        unknown" is something the biologist can go and fix, "insufficient
        information" is not.
        """
        return tuple(n for n in DECLARED_UNKNOWN_OK if not getattr(self, n).known)
