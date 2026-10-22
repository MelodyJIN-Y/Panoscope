"""Why a value came out the way it did: ``Traced[T]`` and the local-only ``Trace``.

Eight signatures in the pre-refactor code returned ``tuple[R, tuple[Decision, ...]]``
by hand. This is that, typed -- and the reason it is worth a module is the *direction*
rule it enforces:

**Decisions flow out, never in.** A ``Trace`` is constructed INSIDE a function and
returned as frozen data. It is never a parameter. That is what keeps ``core/``
un-injectable: there is no seam a caller could reach through to supply a recorder, so
a function is pure in its declared arguments and ``test_record_not_generated`` does
not have to rest on nobody exploiting one.

Three alternatives, each rejected for a stated reason:

- a bare ``tuple[R, tuple[Decision, ...]]`` return -- untyped, and unreadable once
  calls nest;
- a recorder passed **in** -- invites exactly the caller-injected recorder above;
- a ``ContextVar`` -- an *ambient* seam, which is the same injection point wearing a
  disguise, and the one hardest to see in a review.

``Decision.rule`` is a REGISTERED id rather than prose. The code this replaces decided
whether to draw a UI badge by string-matching the exact English phrase "localizes
better with another cluster" (``ui/evidence_table.py:53``), so a copy edit in the
analysis layer silently stopped a badge rendering -- and the test asserted the same
literal, so it stayed green. Ids are what tests and the UI match on; English lives in
``RULES`` and in ``ui``'s copy tables, where editing it is harmless.

Determinism: decision order is **code order** -- append, splice on ``absorb``, never
sort. ``because`` strings are built with ``fmt_num`` at fixed precision and carry no
clock, because they are hashed into the golden record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from panoscope.types.subject import Subject

T = TypeVar("T")
X = TypeVar("X")


@dataclass(frozen=True, slots=True)
class Rule:
    """A decision the system is allowed to make, declared once.

    ``step`` lives here rather than on ``Decision`` because it is a property of the
    rule, not of the occasion: "coverage_floor" is always part of measuring. Putting
    it on each Decision let two emitters of one rule disagree about which step they
    were in, which made grouping the record by step unreliable.
    """

    id: str
    step: str   # "measure" | "confidence" | ...
    what: str   # human-readable; NEVER matched on


def _rules(*rules: Rule) -> dict[str, Rule]:
    return {r.id: r for r in rules}


# --------------------------------------------------------------------------- #
# The registry. Each entry is a decision some core function actually emits --
# this is a transcription of today's behaviour, not a wish list.
# --------------------------------------------------------------------------- #
RULES: dict[str, Rule] = _rules(
    # -- step: measure (panoscope.types.measured.resolve) --
    Rule("measure:never_measured", "measure",
         "a constituent lies outside the capability frame for its kind"),
    Rule("measure:tested_absent", "measure",
         "in frame and tested, but not reported -- absence IS evidence here"),
    Rule("measure:observed", "measure", "reported with a statistic"),
    Rule("measure:coverage_floor", "measure",
         "K of N fell below a declared floor; demoted to underpowered"),
    # -- step: confidence (agent.verdict._compute_band) --
    Rule("band:no_canonical_driver", "confidence",
         "no canonical marker supports the call"),
    Rule("band:driver_nosig", "confidence", "the driving marker is NoSig"),
    Rule("band:from_coef", "confidence", "base band read off the driver's glm_coef"),
    Rule("band:promote_pearson_high", "confidence",
         "strong spatial specificity on the driver; +1 band"),
    Rule("band:demote_pearson_low", "confidence",
         "weak spatial specificity on the driver; -1 band"),
    Rule("band:demote_localises_elsewhere", "confidence",
         "driver localises better with another cluster; -1 band"),
    Rule("band:demote_not_unique", "confidence",
         "driver's spatial pattern is not unique; -1 band"),
    Rule("band:fragile_cap", "confidence",
         "fragile cluster (<=2 markers) cannot read above Medium-High"),
    Rule("band:fragile_floor", "confidence",
         "fragile cluster with a canonical driver stays at or above Medium"),
)


class UnregisteredRule(KeyError):
    """Raised for a decision whose rule id is not in ``RULES``. Never guessed at."""


@dataclass(frozen=True, slots=True)
class Decision:
    """One recorded step. Frozen, hashable, clock-free, and hashed into the record."""

    rule: str
    outcome: str
    because: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()   # ref() strings: what this decision read

    @property
    def step(self) -> str:
        return RULES[self.rule].step

    @property
    def what(self) -> str:
        return RULES[self.rule].what

    def __str__(self) -> str:
        return f"{self.rule} -> {self.outcome}"


@dataclass(frozen=True, slots=True)
class Traced(Generic[T]):
    """A value with the decisions that produced it. The return type of every core fn."""

    value: T
    decisions: tuple[Decision, ...] = ()

    def steps(self, step: str) -> tuple[Decision, ...]:
        return tuple(d for d in self.decisions if d.step == step)


class Trace:
    """A LOCAL, mutable builder. Construct inside a function; return ``done()``.

    Never accept one as a parameter -- ``test_no_core_function_accepts_a_trace``
    enforces that, and it is the whole point of the design.
    """

    __slots__ = ("_decisions",)

    def __init__(self) -> None:
        self._decisions: list[Decision] = []

    def note(self, rule: str, outcome: str,
             because: tuple[str, ...] = (), refs: tuple[str, ...] = ()) -> None:
        """Record one decision. Fails loudly on an unregistered rule id.

        Unregistered means the record would carry a decision nothing can explain,
        render or test against -- which is how prose crept back in last time.
        """
        if rule not in RULES:
            raise UnregisteredRule(
                f"unregistered decision rule {rule!r}; register it in RULES first. "
                f"Known: {sorted(RULES)}"
            )
        self._decisions.append(
            Decision(rule=rule, outcome=outcome, because=tuple(because), refs=tuple(refs))
        )

    def absorb(self, t: Traced[X]) -> X:
        """Splice a callee's decisions in call order and unwrap its value."""
        self._decisions.extend(t.decisions)
        return t.value

    def done(self, value: T) -> Traced[T]:
        return Traced(value=value, decisions=tuple(self._decisions))


# --------------------------------------------------------------------------- #
# Deterministic formatting. ONE home, because these strings are hashed.
# --------------------------------------------------------------------------- #
_PRECISION = 4   # matches the provenance strings already emitted (tools.py:193)


def fmt_num(x: float | int | None) -> str:
    """Fixed-precision, locale-free, platform-stable. The only way to put a number
    into a ``because`` or a ``ref``.

    ``None`` renders as "na" rather than "None" so a missing statistic reads the same
    whether it was never measured or tested-absent -- the distinction belongs in the
    Measured state, not in a formatting accident.
    """
    if x is None:
        return "na"
    if isinstance(x, int) and not isinstance(x, bool):
        return str(x)
    f = float(x)
    if f != f:
        return "nan"
    if f in (float("inf"), float("-inf")):
        return "inf" if f > 0 else "-inf"
    s = f"{f:.{_PRECISION}f}"
    return "0." + "0" * _PRECISION if s == "-" + "0." + "0" * _PRECISION else s


def ref(anchor: Subject | str, stat: str = "", value: float | int | None = None) -> str:
    """A click-back anchor: what a decision read, in a form that resolves.

    Anchored on ``Subject.key`` because that is the one identity the rest of the
    system joins on. A prettier anchor (``gene:ERBB2``) would not resolve back to the
    subject for any shape with more than one role, so the interface could not click
    through from a ligand-receptor sentence to its evidence.
    """
    head = anchor.key if isinstance(anchor, Subject) else anchor
    if not stat:
        return head
    return f"{head}#{stat}={fmt_num(value)}"
