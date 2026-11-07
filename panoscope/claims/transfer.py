"""Condition deltas, and the larger set of claims that must never get a verdict.

The second half of ARCHITECTURE §7.5's split. §7.4 measured the conditional-validity
check at 0.000 bits over 280 curated vendor-panel edges: a constant. So the
verdict-bearing half runs only where conditions came from a manifest, and
:func:`manifest_sourced` is that gate.

- Database-curated and literature claims get a provenance stamp, not a verdict, so
  §7.5's class-level banner renders once; the measurement adjudicates them
  (:mod:`panoscope.claims.evaluability`, §7.6).
- Frames are not a delta: digest comparison would fire on nearly every claim.
  Capability is answered per entity, per frame, by ``evaluability``.
- The return carries named deltas. ``action`` is a display instruction (§10.3: a blocked
  claim is surfaced with its reason), not a score; never aggregate it.

Pure: no IO, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panoscope.types.claim import Authority, Claim
from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, DISEASE_STATE, Conditions,
)
from panoscope.types.trace import Trace, Traced, ref

MISMATCH: Final = "mismatch"
CANNOT_BE_SETTLED: Final = "cannot_be_settled"

FIELDS: Final[tuple[str, ...]] = BLOCKS_THE_STUDY + DECLARED_UNKNOWN_OK
"""Every ``Condition`` field, derived from §6.3's two tables rather than re-listed.

``conditions.py`` already pins that each field sits in exactly one of those tuples, so
a seventh field lands here automatically. A second hand-written list in this module is
the copy nobody updates -- the bug ``evidence.py`` documents from the other direction.
"""

_BLOCKING: Final[frozenset[str]] = frozenset(BLOCKS_THE_STUDY)


@dataclass(frozen=True, slots=True)
class Delta:
    """One named condition difference. IMPLEMENTATION §5's shape, unchanged."""

    field: str
    claim_value: str | None
    context_value: str | None
    severity: str                        # MISMATCH | CANNOT_BE_SETTLED

    def __str__(self) -> str:
        return (f"{self.field}: {self.claim_value or 'unknown'} -> "
                f"{self.context_value or 'unknown'} ({self.severity})")


@dataclass(frozen=True, slots=True)
class TransferResult:
    """Deltas, plus how to surface them. ``stamp`` is this module's one addition.

    IMPLEMENTATION §5 gives ``action`` and ``deltas`` only, but §7.5 requires that
    database-curated and literature claims receive "a **provenance stamp**, not a
    verdict" -- and with §5's two fields the only way to express that is an empty
    delta tuple, which is indistinguishable from *checked and matched everywhere*.
    That collapse is the §7.4 failure in miniature, so the stamp is a field. It is
    empty exactly when a verdict was computed.
    """

    action: str                          # see ACTIONS
    deltas: tuple[Delta, ...] = ()
    stamp: str = ""

    @property
    def checked(self) -> bool:
        """True iff this claim was eligible for a verdict at all."""
        return not self.stamp


RETURN: Final = "return"
WITH_CONDITIONS: Final = "return_with_conditions"
BLOCKED: Final = "return_blocked_with_reason"
ACTIONS: Final[tuple[str, ...]] = (RETURN, WITH_CONDITIONS, BLOCKED)


def manifest_sourced(claim: Claim) -> bool:
    """Did this claim's conditions come from a manifest? §7.5's eligibility rule.

    LAB and MEASUREMENT claims always: both are formed while looking at one study's
    own files, and ``Claim`` already refuses to store a LAB claim without them.

    CURATED claims only when **dataset- or sample-scoped** -- §7.5's "dataset-scoped
    curated sets such as a vendor panel annotation". The scope string is the honest
    discriminator and it is already validated fail-closed by ``Claim``: a curated edge
    that came out of a database is scoped ``lab`` or ``study`` because it is not about
    any one dataset, while a vendor panel annotation is scoped to the dataset whose
    manifest produced it. Inferring this from the source URI instead would mean
    teaching this module which schemes are "databasey", which is a second opinion
    about provenance that drifts from the first.

    LITERATURE never: a claim read off a PMID has no study intake behind it and
    ``conditions`` is legitimately ``None``.
    """
    if claim.authority in (Authority.LAB, Authority.MEASUREMENT):
        return True
    if claim.authority is Authority.CURATED:
        return claim.scope.startswith(("dataset:", "sample:"))
    return False


def _delta(field: str, claim_c: Conditions, context: Conditions) -> Delta | None:
    mine, here = getattr(claim_c, field), getattr(context, field)
    if not mine.known or not here.known:
        # "`UNKNOWN`-provenance fields never silently apply" (§5). An unknown can
        # neither match nor differ, and treating it as a match is how a lab caveat
        # about one FFPE block leaks into fresh-frozen.
        return Delta(field, mine.value, here.value, CANNOT_BE_SETTLED)
    if mine.value != here.value:
        return Delta(field, mine.value, here.value, MISMATCH)
    return None


def transfer_check(claim: Claim, context: Conditions) -> Traced[TransferResult]:
    """Named condition deltas, for the claims whose conditions came from a manifest.

    Runs AFTER the scope gate and after ``evaluability`` (IMPLEMENTATION §5's stated
    order); ``store.assess`` is where that order is kept, because an ordering rule
    enforced by each caller remembering it is not enforced.

    A mismatch on a *blocking* condition (§6.3: assay, tissue) blocks the claim --
    with its reason, never silently. Everything else is returned with the conditions
    attached, which is the form the interface renders as a caveat the biologist can
    click.
    """
    t = Trace()
    anchor = ref(f"claim:{claim.id}")

    if not manifest_sourced(claim):
        stamp = (f"{claim.authority.name.lower()} claim from {claim.source}; "
                 "formation conditions are not manifest-sourced, so no transfer "
                 "verdict is computed (ARCHITECTURE §7.4/§7.5)")
        t.note("transfer:no_verdict", "provenance stamp", (stamp,), (anchor,))
        return t.done(TransferResult(RETURN, (), stamp=stamp))

    if claim.conditions is None:
        because = ("the claim carries no formation conditions, so none of them can "
                   "be compared with this context",)
        t.note("transfer:cannot_be_settled", "no conditions", because, (anchor,))
        return t.done(TransferResult(
            WITH_CONDITIONS,
            (Delta("conditions", None, None, CANNOT_BE_SETTLED),),
        ))

    # The qualifiers are subordinate to the disease state: an unsettled value is one delta.
    ds = _delta(DISEASE_STATE[0], claim.conditions, context)
    skip = DISEASE_STATE[1:] if ds and ds.severity == CANNOT_BE_SETTLED else ()
    deltas = tuple(d for f in FIELDS if f not in skip
                   and (d := _delta(f, claim.conditions, context)) is not None)

    if not deltas:
        t.note("transfer:transferable", RETURN,
               ("every condition matches: " + ", ".join(FIELDS),), (anchor,))
        return t.done(TransferResult(RETURN, ()))

    for d in deltas:
        # One decision per delta, unlike evaluability's single meet: there is no
        # meet here. "tissue differs" and "segmentation unknown" are two separate
        # things the biologist must see, and reducing them to a worst-of would be
        # the verdict §7.4 retracted, rebuilt one layer up.
        if d.severity == CANNOT_BE_SETTLED:
            t.note("transfer:cannot_be_settled", d.field, (str(d),), (anchor,))
        else:
            t.note("transfer:delta", d.field, (str(d),), (anchor,))

    blocking = tuple(d for d in deltas
                     if d.severity == MISMATCH and d.field in _BLOCKING)
    action = BLOCKED if blocking else WITH_CONDITIONS
    return t.done(TransferResult(action, deltas))
