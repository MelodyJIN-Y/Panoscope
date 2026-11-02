"""The reference graph: nodes, typed edges, and the one rule the whole module exists
to make structural.

> Reference data determines **which question is asked of the measurement**. The
> measurement alone determines **the answer**. Reference data may select and order
> evidence, generate rivals, supply canonical lists and carry citations; it may never
> supply a value, count as agreement, or corroborate a measurement.

``evidence.py`` makes a statistic-without-an-observation *unrepresentable* rather than
merely tested for. This module does the same thing one level up, and it needs three
separate guards because a number can reach the interpretation layer by three routes:

1. **A typed field.** ``Node`` and ``Edge`` have no ``int``/``float`` field at all.
   There is no slot to put a statistic in, so no source can return one. This is not a
   convention: ``test_no_numeric_field_on_the_graph`` walks the dataclass fields.
   PanglaoDB is the live demonstration -- its marker table carries a ubiquitousness
   index and four sensitivity/specificity columns, and ``sources.py`` has nowhere to
   put any of them.

2. **The one open channel.** ``Edge.qualifiers`` is the only free-text key/value
   channel on the graph, and it is exactly where a determined caller would smuggle
   ``("specificity", "0.87")``. So it **rejects a numeric-looking value** at
   construction (:class:`ReferenceValueLeak`). The cost is honest and small: a
   legitimately textual qualifier that happens to read as a number must be spelled so
   it does not -- ``("release", "v2020")``, never ``("release", "2020")``.

3. **Authority.** ``Edge.to_claim()`` takes no authority argument and hard-codes
   :attr:`Authority.CURATED`. ``by_authority`` groups and never merges, so an edge
   that cannot be minted above CURATED can never appear in the MEASUREMENT band, and
   "counts as agreement" has no code path. A ``MEASUREMENT`` reference claim is
   unrepresentable rather than forbidden by review.

**A missing node reads as "not in reference data", never as "does not exist".** That
is the panel-absence rule applied to the graph (ARCHITECTURE §7.3), and it has the
same failure mode: collapse it to a boolean and silence starts arguing. So a lookup
returns three states, not two --

=========================  ====================================================
``IN_REFERENCE``           the consulted sources have a node for this entity
``NOT_IN_REFERENCE``       every source was read, and none of them mentions it
``NO_SOURCE``              nothing could be read -- offline, or never fetched
=========================  ====================================================

``NO_SOURCE`` is the half that makes offline degradation honest. An unfetched source
returning an empty graph would render as ``NOT_IN_REFERENCE`` for every gene in the
panel, which reads as "the literature knows nothing about any of this" -- a confident
claim built out of a failed download. The third state is the difference between "we
looked and found nothing" and "we did not look", and :attr:`Lookup.absence_argues` is
a constant ``False`` for all three, because reference silence never argues under any
of them.

**Relations are a registry, data not types** -- the ``SHAPES``/``RULES``/``FLAGS``
pattern. ``RELATIONS`` holds exactly one entry because exactly one source emits one
today; a relation nothing emits is a traversal nobody can test, which is the FLAGS
lesson. Note what is *not* registerable: ``is_cell_type`` is the verdict, written by
the deterministic engine from the measurement. A reference source able to assert it
would be reference data supplying the answer, so there is no relation for it and
``Edge`` refuses the predicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from panoscope.types.claim import (
    SOURCE_SCHEMES, Authority, Claim, UnknownSource,
)
from panoscope.types.subject import Entity, EntityKind, Subject, subject
from panoscope.types.trace import Trace, Traced


# --------------------------------------------------------------------------- #
# The relation registry. DATA, not types.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Relation:
    """A relation a reference edge is allowed to assert, declared once.

    ``subject_kind``/``object_kind`` are checked at construction, so an edge whose
    endpoints are the wrong kind cannot exist -- a ``cell_type --marks--> gene`` edge
    would traverse backwards and hand the caller a canonical list of cell types for a
    gene, silently.

    ``claim_predicate`` is the ``claim.PREDICATES`` id this relation mints as, or
    ``""`` for a relation that is graph structure only (an alias link is navigation,
    not an assertion anyone should cite). It is the join between the graph's
    vocabulary and the claim store's, written down once instead of guessed at the
    call site.
    """

    id: str
    what: str                      # human-readable; NEVER matched on
    subject_kind: EntityKind
    object_kind: EntityKind
    claim_predicate: str = ""      # "" means this relation does not mint a Claim


def _relations(*rels: Relation) -> dict[str, Relation]:
    return {r.id: r for r in rels}


RELATIONS: dict[str, Relation] = _relations(
    Relation(
        "marks",
        "the gene is a curated marker for the cell type -- canonical whether or not "
        "it is on any panel; the frame decides measurability, the edge does not",
        EntityKind.GENE, EntityKind.CELL_TYPE,
        claim_predicate="marks",
    ),
    # Deliberately absent: `is_cell_type`. That is the verdict, and it is the
    # measurement's to make. See the module docstring.
)


class UnregisteredRelation(KeyError):
    """Raised for an edge predicate not in :data:`RELATIONS`. Never guessed at."""


class ReferenceValueLeak(ValueError):
    """Raised when a qualifier value reads as a number.

    The worst failure this module can produce is a reference statistic that renders
    beside a measured one. ``Edge`` has no numeric field, so the only way in is the
    qualifier channel -- and this closes it at construction rather than at review.
    """


class NotClaimable(ValueError):
    """Raised when an edge whose relation mints no claim is asked for one."""


def _reads_as_number(value: str) -> bool:
    """True for anything ``float()`` accepts -- which includes ``nan`` and ``inf``,
    and should: a qualifier spelled "nan" is a dropped measurement, not a word."""
    try:
        float(value)
    except ValueError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The graph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Node:
    """One entity as a reference ontology holds it.

    Keyed on :class:`Entity`, which is the join key the rest of the system already
    uses: ``Subject.key`` is built from entities, so a node joins to a subject without
    a second identity scheme to keep in sync. ``label`` is display-only on ``Entity``
    and equality is on ``(kind, id)``, so two spellings of one gene cannot become two
    nodes.

    ``aliases`` is a ``frozenset`` rather than a tuple because it is a set and
    ordering it would be a lie that the serialised bytes then depend on. It answers
    §7.2's "is this symbol an alias" and nothing else: an alias is navigation, never
    an assertion, so it carries no citation.
    """

    entity: Entity
    source: str                              # "<source name>@<release>"
    aliases: frozenset[str] = frozenset()

    @property
    def key(self) -> str:
        """``kind:id`` -- the same spelling ``Subject.key`` embeds."""
        return str(self.entity)

    def __str__(self) -> str:
        return self.key


@dataclass(frozen=True, slots=True)
class Edge:
    """One typed, cited, provenanced relation between two reference entities.

    Note what is **not** here: no score, no weight, no confidence, no statistic, no
    rank. The type has no numeric field, which is guard 1 of the three in the module
    docstring -- a reference source has no channel through which a value could reach
    the interpretation layer, so "never supplies a value" is a property of the shape
    rather than a rule somebody has to remember.

    ``citation`` is a resolvable URI from ``claim.SOURCE_SCHEMES`` for the same reason
    ``Claim.source`` is: a citation that is the string "Smith et al." renders, exports
    and survives, and the grounding gate cannot tell it from a fabrication.

    ``qualifiers`` are SORTED pairs rather than a dict, for ``evidence.py``'s reason:
    a dict field is unhashable and its insertion order flows through ``asdict`` into
    the serialised bytes, so the graph would differ by the order a reader happened to
    add them.
    """

    subject: Entity
    predicate: str                           # REGISTERED: see RELATIONS
    object: Entity
    source: str                              # "<source name>@<release>"
    citation: str                            # URI: see claim.SOURCE_SCHEMES
    licence: str                             # travels with the edge; §12 wants it per claim
    qualifiers: tuple[tuple[str, str], ...] = ()   # SORTED; values are NEVER numeric

    def __post_init__(self) -> None:
        rel = RELATIONS.get(self.predicate)
        if rel is None:
            raise UnregisteredRelation(
                f"unregistered reference relation {self.predicate!r}; register it in "
                f"RELATIONS first. Known: {sorted(RELATIONS)}"
            )
        if self.subject.kind is not rel.subject_kind:
            raise ValueError(
                f"{self.predicate!r} takes a {rel.subject_kind.value} subject, got "
                f"{self.subject.kind.value} ({self.subject.id})"
            )
        if self.object.kind is not rel.object_kind:
            raise ValueError(
                f"{self.predicate!r} takes a {rel.object_kind.value} object, got "
                f"{self.object.kind.value} ({self.object.id})"
            )
        scheme, sep, rest = self.citation.partition(":")
        if not sep or scheme not in SOURCE_SCHEMES or not rest:
            raise UnknownSource(
                f"{self.subject.id} {self.predicate} {self.object.id}: citation "
                f"{self.citation!r} is not a resolvable URI; known schemes: "
                f"{sorted(SOURCE_SCHEMES)}"
            )
        for name, value in self.qualifiers:
            if _reads_as_number(value):
                raise ReferenceValueLeak(
                    f"{self.subject.id} {self.predicate} {self.object.id}: qualifier "
                    f"{name!r}={value!r} reads as a number. Reference data may select "
                    "and order evidence; it may never supply a value. Drop it, or "
                    "spell it so it cannot be read as a measurement."
                )
        object.__setattr__(self, "qualifiers", tuple(sorted(self.qualifiers)))

    @property
    def what(self) -> str:
        """The registered relation's English. Never matched on."""
        return RELATIONS[self.predicate].what

    def qualifier(self, name: str) -> str | None:
        return next((v for n, v in self.qualifiers if n == name), None)

    def as_subject(self) -> Subject:
        """The edge's subject in canonical ``Subject`` form -- the join key.

        Reconciliation is a join on ``Subject.key`` across task kinds; an edge that
        kept its own identity scheme could not participate in it.
        """
        return subject(self.subject.kind.value,
                       **{_SUBJECT_ROLE[self.subject.kind]: self.subject})

    def to_claim(self, claim_id: str) -> Claim:
        """Mint this edge as a CURATED claim. Guard 3: the authority is not an argument.

        There is no parameter through which a caller could ask for ``MEASUREMENT`` or
        ``LAB``, so a reference edge cannot enter the band where the measurement's own
        output lives, and cannot be grouped as the lab's judgement either. That is the
        "never counts as agreement" half of the rule, made structural: claims are
        ordered and never merged, so an edge that cannot be minted above CURATED can
        only ever be displayed *beside* a measurement, never as one.

        ``conditions`` is deliberately left ``None``: this edge's formation conditions
        belong to its source, and ARCHITECTURE §7.5 is explicit that a
        database-curated claim receives a provenance stamp, not a transfer verdict.
        """
        rel = RELATIONS[self.predicate]
        if not rel.claim_predicate:
            raise NotClaimable(
                f"relation {self.predicate!r} mints no claim: it is graph structure, "
                "not an assertion. See Relation.claim_predicate."
            )
        return Claim(
            id=claim_id,
            subject=self.as_subject(),
            predicate=rel.claim_predicate,
            object=self.object.id,
            authority=Authority.CURATED,     # NOT a parameter. See the docstring.
            source=self.citation,
            licence=self.licence,
        )

    def __str__(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


# The single-entity shapes in ``SHAPES`` and the role each one names. Written out
# rather than derived, because deriving "the shape with one role" from the registry
# would silently pick a different shape the moment a second single-role shape lands.
_SUBJECT_ROLE: dict[EntityKind, str] = {
    EntityKind.GENE: "g",
    EntityKind.CELL_TYPE: "ct",
}


# --------------------------------------------------------------------------- #
# Lookup: three states, because silence has to be distinguishable from ignorance
# --------------------------------------------------------------------------- #
class Presence(StrEnum):
    IN_REFERENCE = "in_reference"            # a node exists
    NOT_IN_REFERENCE = "not_in_reference"    # read, and not mentioned -> NOT absence
    NO_SOURCE = "no_source"                  # nothing could be read -> says nothing


@dataclass(frozen=True, slots=True)
class Graph:
    """The consulted subgraph. Nodes from different sources are NEVER merged.

    ``consulted`` and ``unavailable`` are both recorded, and that pair is the whole
    offline story: a graph built with every source unavailable has empty ``nodes``
    *and* a non-empty ``unavailable``, so :func:`lookup` can tell "nothing is known"
    from "nothing was read" instead of reporting the download failure as a finding.
    """

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    consulted: tuple[str, ...] = ()          # sources that were successfully read
    unavailable: tuple[str, ...] = ()        # sources that could not be read

    def incident(self, entity: Entity) -> tuple[Edge, ...]:
        # ponytail: linear scan. The per-study reference view is hundreds of edges,
        # not millions; index it if a source ever makes that false.
        return tuple(e for e in self.edges
                     if e.subject == entity or e.object == entity)


@dataclass(frozen=True, slots=True)
class Lookup:
    """What the reference graph says about one entity, and what that is worth.

    :attr:`absence_argues` is a constant ``False`` with no branch in it, mirroring
    ``Measured.absence_is_evidence``. The difference is the point: a *measured*
    absence is the productive half of the lattice and does argue, but a reference
    source's silence is only ever coverage, so there is no state in which it may be
    read as evidence against anything.
    """

    entity: Entity
    presence: Presence
    consulted: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    edges: tuple[Edge, ...] = ()

    @property
    def absence_argues(self) -> bool:
        """Never. Reference coverage is uneven; silence is not a negative result."""
        return False

    @property
    def reads_as(self) -> str:
        """The sentence a reviewer sees. Never "does not exist"."""
        if self.presence is Presence.IN_REFERENCE:
            return (f"{self.entity} is in reference data "
                    f"({', '.join(self.consulted)})")
        if self.presence is Presence.NOT_IN_REFERENCE:
            return (f"{self.entity} is not in reference data "
                    f"({', '.join(self.consulted)}) -- this says nothing about "
                    "whether it exists or occurs here")
        return (f"no reference source could be read for {self.entity} "
                f"({', '.join(self.unavailable) or 'none registered'}) -- "
                "reference data was not consulted")

    def objects(self, predicate: str) -> tuple[Entity, ...]:
        """Forward traversal: what this entity points at under ``predicate``."""
        return tuple(e.object for e in self.edges
                     if e.predicate == predicate and e.subject == self.entity)

    def subjects(self, predicate: str) -> tuple[Entity, ...]:
        """Reverse traversal: what points at this entity under ``predicate``.

        This is the canonical-marker list -- ``lookup(g, fibroblast).subjects("marks")``
        -- and the rival generator. Both *select* evidence; neither supplies one.
        """
        return tuple(e.subject for e in self.edges
                     if e.predicate == predicate and e.object == self.entity)


def lookup(graph: Graph, entity: Entity) -> Traced[Lookup]:
    """What the graph says about ``entity``, with the decision that settled it.

    Fail-closed on a partial read: if the entity was not found and *any* source was
    unavailable, the answer is ``NO_SOURCE``, not ``NOT_IN_REFERENCE``. Saying "not in
    reference data" while a source failed to load would be reporting a download error
    as knowledge, and it is invisible -- the sentence reads identically either way.
    """
    t = Trace()
    found = tuple(n for n in graph.nodes if n.entity == entity)
    incident = graph.incident(entity)
    anchor = str(entity)

    if found:
        state = Presence.IN_REFERENCE
        t.note("reference:in_reference", state.value,
               because=(f"{len(found)} node(s) in {', '.join(graph.consulted)}",),
               refs=(anchor,))
    elif graph.unavailable or not graph.consulted:
        state = Presence.NO_SOURCE
        t.note("reference:no_source", state.value,
               because=(
                   f"unavailable: {', '.join(graph.unavailable) or 'none registered'}",
               ),
               refs=(anchor,))
    else:
        state = Presence.NOT_IN_REFERENCE
        t.note("reference:not_in_reference", state.value,
               because=(f"read {', '.join(graph.consulted)}; no node",),
               refs=(anchor,))

    return t.done(Lookup(
        entity=entity, presence=state,
        consulted=graph.consulted, unavailable=graph.unavailable,
        edges=incident,
    ))
