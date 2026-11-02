"""Claims: one object type for measurement, lab, curated and literature knowledge.

Four very different things -- a GLM coefficient, a biologist's note, a reference
edge, a sentence from a paper -- are the same shape here, differing only in who may
change them and how much weight they carry (ARCHITECTURE §10.1). That is the whole
design: the moment a lab note and a curated edge are *different types*, some code
path handles one and forgets the other, and the forgotten one is always the lab's.

**Authority is an ``IntEnum`` because claims are ORDERED and NEVER MERGED.** Ordering
needs a comparison; merging needs a resolution rule, and there is none. Conflicts are
grouped and displayed (§10.2). A ``StrEnum`` would have made ``sorted(claims, ...)``
sort alphabetically -- curated, lab, literature, measurement -- which is exactly
backwards for three of the four, and silently.

Three fields are REGISTERED or VALIDATED rather than free text, each for a bug that
was real:

* ``predicate`` is an id from :data:`PREDICATES`, the same pattern as ``trace.py``'s
  ``RULES`` and ``evidence.py``'s ``FLAGS``. A free-text predicate means retrieval
  matches on English, and a copy edit silently empties a result set.
* ``source`` is a URI whose scheme must be known. Claims do not carry abstracts --
  the URI resolves to a ``SourceRecord`` in a reference store -- which is what makes
  ``licence`` meaningful and what makes "citations must be real and looked up live"
  checkable at all. A claim whose source is the string "Smith et al." is unresolvable
  and therefore uncitable, and the grounding gate cannot tell it from a fabrication.
* ``scope`` is parsed and enforced **fail-closed**. See :meth:`Claim.applies_in`.

**``conditions`` is optional, and that is a deviation from IMPLEMENTATION §2.7.**
§2.7 types it ``Conditions`` outright, but ``conditions.py`` blocks construction of a
``Conditions`` with no frames and with an unknown assay or tissue -- so under §2.7 as
written, *every* ``Claim`` would already carry frames and §2.7's own stated invariant
("a LAB claim with no frame reference fails validation") could never fire. An
invariant that cannot fail is a comment. Worse, a literature claim read off a PMID
has no study intake behind it: its assay and tissue are genuinely unknown, so it
cannot have a ``Conditions`` at all, and §2.7 as written makes imported literature
unstorable. ``None`` is therefore the honest spelling of "this claim was not formed
under a study", the LAB invariant fires against it, and both halves work.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from panoscope.types.conditions import Conditions
from panoscope.types.subject import Entity, Subject


class Authority(IntEnum):
    """Who is entitled to be believed, and in what order. Never merged, never blurred.

    The numbers are the ordering from ARCHITECTURE §10.2, and the ordering is the
    only operation: there is deliberately no rule for combining two claims of
    different authority into one. "If a lab claim renders as a curated fact, or a
    curated edge as evidence, the design collapses."
    """

    LITERATURE = 1   # a paper said it
    CURATED = 2      # a reference database asserts it
    LAB = 3          # this lab's own judgement, committed by the biologist
    MEASUREMENT = 4  # this study's own method output


@dataclass(frozen=True, slots=True)
class Predicate:
    """A relation a claim is allowed to assert, declared once.

    ``what`` is human-readable and is NEVER matched on. ``object_is`` names what the
    claim's ``object`` holds when it holds anything, so a reader knows whether to
    expect an entity id or free text without guessing from the value it happens to
    find.
    """

    id: str
    what: str
    object_is: str = ""   # "" means the predicate takes no object


def _predicates(*preds: Predicate) -> dict[str, Predicate]:
    return {p.id: p for p in preds}


# --------------------------------------------------------------------------- #
# The registry. Exactly the three predicates IMPLEMENTATION §2.7 names, because
# each has a writer today: the deterministic engine writes `is_cell_type` (the
# verdict), the bundled map at agent/annotation.py holds ~66 `marks`, and the
# note-capture path writes `means_here`. A fourth would be a predicate nothing
# emits -- a retrieval mask nobody can test, which is the FLAGS lesson.
# --------------------------------------------------------------------------- #
PREDICATES: dict[str, Predicate] = _predicates(
    Predicate("is_cell_type",
              "the subject (a unit, usually a cluster) IS this cell type",
              object_is="cell type id"),
    Predicate("marks",
              "the subject (a gene) is a marker for this cell type -- canonical "
              "whether or not it is on the panel; the frame decides measurability, "
              "the claim does not",
              object_is="cell type id"),
    Predicate("means_here",
              "what the subject means UNDER THESE CONDITIONS -- the local reading a "
              "biologist commits when the general one does not hold here",
              object_is="free text"),
)


# --------------------------------------------------------------------------- #
# Source URIs. A claim points at a source record; it never inlines one.
# --------------------------------------------------------------------------- #
SOURCE_SCHEMES: frozenset[str] = frozenset({"pmid", "doi", "note", "panel", "file"})
"""The known schemes. ``file:`` is content-addressed in practice (``file:sha256:…``)
but that shape is not re-checked here: the digest belongs to whoever minted the URI,
and a second, weaker opinion about its spelling in this module would only drift."""


# --------------------------------------------------------------------------- #
# Scope. Enforced fail-closed (CLAUDE.md), which has to be a property of the
# LATTICE rather than of each caller remembering to check.
# --------------------------------------------------------------------------- #
_SCOPE_RANK: dict[str, int] = {"sample": 0, "dataset": 1, "study": 2, "lab": 3}
_KEYED_SCOPES: frozenset[str] = frozenset({"sample", "dataset"})


def _parse_scope(scope: str) -> tuple[str, str] | None:
    """``(kind, id)``; ``id`` is "" for the unkeyed scopes. ``None`` if malformed."""
    if scope in ("study", "lab"):
        return scope, ""
    kind, sep, ident = scope.partition(":")
    # A colon inside the id ("sample:s1:extra") is a typo'd nested scope, not an id.
    # Left alone it parses, stores, and then matches no context ever -- which is the
    # invisible failure the fail-closed gate makes it the write's job to catch.
    if sep and kind in _KEYED_SCOPES and ident and ":" not in ident:
        return kind, ident
    return None


class UnregisteredPredicate(KeyError):
    """Raised for a predicate id not in :data:`PREDICATES`. Never guessed at."""


class UnknownSource(ValueError):
    """Raised for a source URI whose scheme is not in :data:`SOURCE_SCHEMES`.

    An unresolvable source is worse than no source: it renders as a citation, it
    survives into the exported report, and nothing downstream can distinguish it
    from one that was looked up live.
    """


class BadScope(ValueError):
    """Raised for a scope that does not parse.

    Fail-closed makes an unparseable scope invisible rather than loud: the claim
    would be filtered out of every retrieval, silently, long after it was written.
    So the failure is moved to the write.
    """


class UnformedLabClaim(ValueError):
    """Raised for a LAB claim with no capability frame behind it (§2.7).

    The biologist's judgement is the thing this system exists to accumulate, and it
    is formed while looking at one measurement of one tissue. Stored without the
    frames it was formed under it cannot be transfer-checked against a later study,
    so it either applies everywhere -- and the lab's hard-won caveat about one FFPE
    block leaks into fresh-frozen -- or it is dropped. Neither is storable.
    """


@dataclass(frozen=True, slots=True)
class Claim:
    """One claim, whatever its authority. ARCHITECTURE §10.1, IMPLEMENTATION §2.7.

    ``status`` (firm | tentative) is orthogonal to ``authority`` and both are kept:
    authority says *who* is entitled to be believed, status says *how firmly they
    said it*. A tentative measurement and a firm literature claim are different
    things, and collapsing them loses the distinction a reviewer most wants.
    """

    id: str
    subject: Subject
    predicate: str                     # REGISTERED: see PREDICATES
    object: str | None
    authority: Authority
    source: str                        # URI: see SOURCE_SCHEMES
    conditions: Conditions | None = None
    depends_on: tuple[Entity, ...] = ()
    #: ``sample:<id> | dataset:<id> | study | lab``. **No usable default**: ``""``
    #: does not parse, so ``__post_init__`` raises :class:`BadScope` naming the
    #: claim and the four legal spellings. It defaulted to ``"study"``, which is an
    #: UNKEYED rung and applies downward into every study -- so a claim constructed
    #: without a scope silently became a claim about all of them, which is the
    #: widening the fail-closed gate exists to refuse. ``claims/store._from_dict``
    #: already refuses it on the read path; this is the same rule for the rows this
    #: process builds itself. A default rather than a required field only because
    #: making it required would reorder the positional arguments of a dataclass
    #: eight defaulted fields deep.
    scope: str = ""
    status: str = "firm"               # firm | tentative
    valid: str = "active"              # active | superseded | retracted
    superseded_by: str | None = None
    created: str = ""                  # an ISO string someone else minted; no clock here
    author: str | None = None
    tension: tuple[str, ...] = ()      # ids of claims this one is in tension with
    licence: str | None = None
    spans: tuple[str, ...] = ()
    spec_version: str = "0.1"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError(
                "a claim needs an id: it is what a rendered sentence clicks back to, "
                "and 'a sentence with nothing to click should not have been rendered'"
            )
        if self.predicate not in PREDICATES:
            raise UnregisteredPredicate(
                f"unregistered predicate {self.predicate!r}; register it in "
                f"PREDICATES first. Known: {sorted(PREDICATES)}"
            )
        scheme, sep, rest = self.source.partition(":")
        if not sep or scheme not in SOURCE_SCHEMES or not rest:
            raise UnknownSource(
                f"{self.id}: source {self.source!r} is not a resolvable URI; "
                f"known schemes: {sorted(SOURCE_SCHEMES)}"
            )
        if _parse_scope(self.scope) is None:
            raise BadScope(
                f"{self.id}: scope {self.scope!r} does not parse. Use 'lab', "
                "'study', 'dataset:<id>' or 'sample:<id>'."
            )
        if self.status not in ("firm", "tentative"):
            raise ValueError(
                f"{self.id}: status must be firm | tentative, got {self.status!r}"
            )
        if self.valid not in ("active", "superseded", "retracted"):
            raise ValueError(
                f"{self.id}: valid must be active | superseded | retracted, "
                f"got {self.valid!r}"
            )
        # Supersession is a POINTER, and it has to point both ways or neither. A
        # claim marked superseded with nothing to follow is a dead end the biologist
        # cannot act on; a pointer left on a still-active claim reads as active to
        # retrieval and as replaced to the reader. Same trick as Condition's
        # value/provenance iff: make the incoherent pair unrepresentable.
        if (self.valid == "superseded") is not (self.superseded_by is not None):
            raise ValueError(
                f"{self.id}: valid='superseded' iff superseded_by names the "
                f"replacement (valid={self.valid!r}, "
                f"superseded_by={self.superseded_by!r})"
            )
        if self.authority is Authority.LAB and not (
            self.conditions is not None and self.conditions.frames
        ):
            raise UnformedLabClaim(
                f"{self.id}: a LAB claim needs the conditions it was formed under, "
                "including at least one FrameRef. Without them it cannot be "
                "transfer-checked, so it would apply everywhere or nowhere."
            )

    @property
    def what(self) -> str:
        """The registered predicate's English. Never matched on."""
        return PREDICATES[self.predicate].what

    def applies_in(self, context: str) -> bool:
        """Fail-closed scope gate: true only where this claim PROVABLY applies.

        The gate runs before retrieval, as a filter rather than a downgrade
        (IMPLEMENTATION §5), so a false here means the claim is never seen -- which
        is exactly why the default answer is no.

        Two rules:

        * An unkeyed scope (``lab``, ``study``) applies downward, never upward: a
          lab-wide claim holds in one dataset; a study claim must not be reused
          lab-wide. Rank decides.
        * A keyed scope (``dataset:X``, ``sample:Y``) applies only where the context
          names the same key. Membership is NOT derivable from a scope string --
          nothing in ``"sample:s1"`` says which dataset s1 belongs to -- so a caller
          that knows s1 is in d1 asks about both contexts. Inferring the hierarchy
          from the strings is precisely how a note about one FFPE block ends up
          qualifying a different one.

        A malformed context is false rather than an error: contexts arrive from the
        session, and a typo must narrow what is shown, never widen it.
        """
        here = _parse_scope(context)
        if here is None:
            return False
        mine = _parse_scope(self.scope)
        if mine is None:            # unreachable via __post_init__; still fail closed
            return False
        kind, ident = mine
        if kind in _KEYED_SCOPES:
            return (kind, ident) == here
        return _SCOPE_RANK[kind] >= _SCOPE_RANK[here[0]]


def by_authority(claims: tuple[Claim, ...]) -> tuple[Claim, ...]:
    """Highest authority first, ties in the order given. GROUPING, not merging.

    Stable on purpose: within one authority, the store's order is the order the
    biologist sees, and re-sorting by anything else -- recency, a relevance score --
    is how the field's default, resolve-by-recency, creeps back in disguised as a
    display detail.
    """
    return tuple(sorted(claims, key=lambda c: -int(c.authority)))
