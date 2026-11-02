"""Condition deltas -- and the far larger set of claims that must never get a verdict.

This is the second half of ARCHITECTURE §7.5's split, and it is defined as much by
what it REFUSES to run on as by what it computes.

**The retraction.** §7.4 measured the v4 conditional-validity check and it returned
``0.000 bits`` over 280 curated vendor-panel edges: rules 3, 4, 6 and 7 fired on
100.0% of them, rule 5 on 0.0%, and ``H(mismatch-tuple)`` was also 0.000 -- the same
four strings every time. A counterfactual sweep across Xenium/breast/FFPE,
fresh-frozen, CosMx/lung, Visium, IHC and matched scRNA-seq returned ``conditional``
for five of six, the exception being the degenerate case where the context *is* the
claim's formation conditions. And the verdict predicted nothing: AUROC undefined
because constant, while the measurement's own ``1 - pearson`` managed 0.667.

A check that cannot vary is not a safety property, it is a banner printed per row. So
the fix is not a better threshold -- it is to stop running the verdict-bearing half on
the class where it is constant. :func:`manifest_sourced` is that gate, and it is the
most important function in the module.

**Database-curated and literature claims get a provenance stamp, not a verdict.** The
stamp says where the knowledge came from and lets §7.5's class-level banner be
rendered once -- *"curated edges in this corpus were established in dissociated
scRNA-seq on fresh tissue; you are in FFPE in-situ"* -- rather than repeating the same
four deltas on every claim. Those claims are still adjudicated; just by the
measurement (:mod:`panoscope.claims.evaluability`, §7.6), which is the only thing that
ever could adjudicate them.

**Frames are deliberately not a delta.** Comparing the claim's ``FrameRef`` digests
against the context's would fire on nearly every claim and reintroduce exactly the
constant §7.4 caught -- one extra negative-control probe reads as a different world.
The capability question is answered per entity, per frame, by ``evaluability``, which
is strictly more informative: it names the gene and the frame that refused it instead
of announcing that two digests differ.

**Never report a verdict distribution.** The return carries the named deltas. The
``action`` exists because §10.3 requires a blocked claim to be SURFACED with its
reason, so something must say "surfaced, but blocked" -- it is a display instruction,
not a score, and aggregating it is the mistake §7.4 documents.

Pure: no IO, no clock.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from panoscope.types.claim import Authority, Claim
from panoscope.types.conditions import BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, Conditions
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

    deltas = tuple(d for f in FIELDS
                   if (d := _delta(f, claim.conditions, context)) is not None)

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
