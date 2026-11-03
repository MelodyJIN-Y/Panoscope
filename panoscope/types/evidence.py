"""Evidence: the INPUT side of an interpretation, and the IFF that grounds it.

An ``EvidenceItem`` is what a method *reported*, before anything interprets it.
There is deliberately no tier, no band and no ``verify`` here -- those are the
interpreter's output, and keeping them off the input is what stops a method from
grading its own homework.

**The one rule this module exists to enforce.** A statistic may be present only
where the subject was actually observed::

    statistic is None  IFF  not measured.carries_number

The check is derived from :attr:`Measured.carries_number` rather than from a
re-listed set of states, because the two drift. ``IMPLEMENTATION.md`` §2.2 and
``CLAUDE.md`` both used to state the rule as ``{TESTED_ABSENT, NEVER_MEASURED}``,
omitting ``UNDERPOWERED`` -- which was wrong: "in frame, below the floor" is a
state where absence is not evidence either, and a number quoted from it is a
number the measurement could not support. ``measured.py`` always had it right and
both documents now say so. Spelling the states out a second time here would have
made that spec bug permanent, and the second list is the one nobody updates.

It is enforced in ``__post_init__`` and it *raises*. Quoting a number for a thing
that was never measured is the single worst failure this project can produce, and
a failure of that size must be unrepresentable, not merely tested for.

``flags`` are REGISTERED ids, never prose, for the same reason ``Decision.rule`` is
(``trace.py``): the UI decided whether to draw a specificity badge by matching the
exact English phrase "localizes better with another cluster"
(``ui/evidence_table.py``), so a copy edit in the analysis layer silently stopped
the badge rendering -- and the test asserted the same literal, so it stayed green.
Ids are what code matches on; English lives in ``FLAGS[...].what`` and in the UI's
``FLAG_COPY``, where editing it is harmless.

``corroborating`` is a SORTED tuple of pairs rather than the dict the spec sketch
shows. A dict field on a frozen dataclass is unhashable, and its insertion order
flows through ``dataclasses.asdict`` into the serialised bytes -- so the golden
record would differ by the order a reader happened to add corroborators. Sorting at
construction makes byte-determinism a property of the type rather than a habit of
every caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.types.conditions import Conditions
from panoscope.types.measured import Frame, Measured
from panoscope.types.subject import Subject


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class Flag:
    """A caveat the system is allowed to raise, declared once.

    ``what`` is human-readable and is NEVER matched on -- it is the explanation a
    reviewer reads, not the token a branch tests.
    """

    id: str
    what: str


def _flags(*flags: Flag) -> dict[str, Flag]:
    return {f.id: f for f in flags}


# --------------------------------------------------------------------------- #
# The registry. Each entry is a caveat the analysis layer actually emits today
# (agent/verdict.py::_caveats_for) -- a transcription of current behaviour, not a
# wish list. A flag nobody emits is a badge nobody can test.
# --------------------------------------------------------------------------- #
FLAGS: dict[str, Flag] = _flags(
    Flag("specificity:localises_elsewhere",
         "transcripts correlate better with another cluster's spatial pattern "
         "(max_gc_corr > pearson) -- the marker also marks another lineage"),
    Flag("specificity:not_unique",
         "another gene shares this spatial pattern (max_gg_corr at or above the floor)"),
    Flag("specificity:low",
         "weak spatial specificity (pearson below the floor)"),
)


class UnregisteredFlag(KeyError):
    """Raised for a flag id that is not in ``FLAGS``. Never guessed at."""


class UngroundedStatistic(ValueError):
    """Raised when a statistic and a measurement status disagree.

    Both directions are errors: a number where nothing was observed is a fabricated
    value, and a missing number where something *was* observed means the reader
    dropped the measurement on the floor, so the item would read as untestable when
    it is not -- a refusal the biologist cannot tell from a real one.
    """


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """What one method reported about one subject. INPUT side: no tier, no verify."""

    subject: Subject
    statistic: float | None
    statistic_name: str
    measured: Measured
    corroborating: tuple[tuple[str, float], ...] = ()  # SORTED pairs, never a dict
    direction: Direction | None = None
    flags: tuple[str, ...] = ()        # REGISTERED ids, never prose
    shared_with: tuple[str, ...] = ()  # other units carrying this subject

    def __post_init__(self) -> None:
        for f in self.flags:
            if f not in FLAGS:
                raise UnregisteredFlag(
                    f"unregistered evidence flag {f!r}; register it in FLAGS first. "
                    f"Known: {sorted(FLAGS)}"
                )
        if (self.statistic is not None) is not self.measured.carries_number:
            raise UngroundedStatistic(
                f"{self.subject.key}: {self.statistic_name}={self.statistic!r} "
                f"with measured state {self.measured.state.value!r} "
                f"(frame {self.measured.frame!r}); a statistic exists if and only "
                "if the subject was OBSERVED"
            )
        # Byte-determinism: the serialised record must not depend on the order a
        # reader happened to add corroborators.
        object.__setattr__(self, "corroborating", tuple(sorted(self.corroborating)))


@dataclass(frozen=True, slots=True)
class NormalisedEvidence:
    """Every method's output reduced to one shape, for ONE unit.

    ``unit_measured`` carries the unit-level capability verdict -- "too few
    replicates", "fewer cells than the floor" -- because it is a property of the
    unit, not of any marker in it. Pushing it down onto the items duplicated it once
    per item and let two items disagree about the same design.
    """

    unit: str
    unit_kind: str          # open vocabulary; only TaskKind is frozen
    unit_subject: Subject
    unit_measured: Measured
    module: str
    items: tuple[EvidenceItem, ...]
    reported: tuple[Frame, ...]
    conditions: Conditions
