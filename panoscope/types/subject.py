"""What an evidence item is *about*: one key space shared by every task kind.

The decisive argument for this abstraction is reconciliation. Asking whether the
marker result, the DE result and a ligand-receptor result agree about ERBB2 is a
**join across task kinds on the item key**. That join only works if every workflow
keys its items identically. Under the shape the v2 spec proposed it could not:
assignments keyed on a bare gene string, characterisations on an ``Item.name``, held
together by convention.

A tagged union was rejected for a different reason. A union is a *closed* type, so
each of the unbuilt analysis families -- ligand-receptor, niche, neighbourhood,
co-localisation, differential abundance, co-expression -- would force it open again
and reopen every exhaustive match over it. That is a scheduled rewrite, not a frozen
contract. Here the closed thing is the *registry*, which is data: adding a family is
a dict entry, and no type, serializer or consumer changes.

``Subject.key`` is the join key. It is stable, sortable and canonical, which is why
symmetric role groups are sorted at construction -- an unordered cell-type pair must
be one subject however the reader happened to emit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class EntityKind(StrEnum):
    """What a thing *is*. Each kind has exactly one capability frame (see measured.py)."""

    GENE = "gene"
    GENE_SET = "gene_set"
    CELL_TYPE = "cell_type"
    UNIT = "unit"            # a cluster / niche / region id the METHOD produced
    CONDITION = "condition"  # a design level: KO, WT, DrugA
    CONTRAST = "contrast"    # an ordered pair of conditions
    SAMPLE = "sample"


@dataclass(frozen=True, slots=True, order=True)
class Entity:
    """A canonical, resolvable thing.

    ``id`` is the identifier its capability frame is keyed on -- an HGNC symbol, an
    annotation label, a gene-set name. ``label`` is display only and is NEVER
    load-bearing: it sorts last and equality is effectively on ``(kind, id)``, so two
    spellings of the same thing can never become two subjects.
    """

    kind: EntityKind
    id: str
    label: str = ""

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.id}"


class Gate(StrEnum):
    """How a role participates in the measurement-status computation (measured.py)."""

    REQUIRED = "required"  # in the MEET: absent -> the whole subject is unmeasurable
    MEMBER = "member"      # in COVERAGE: K of N against a floor
    CONTEXT = "context"    # neither; recorded for provenance only


@dataclass(frozen=True, slots=True)
class Role:
    name: str
    kind: EntityKind
    gate: Gate = Gate.REQUIRED
    many: bool = False


@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    roles: tuple[Role, ...]
    symmetric: tuple[str, ...] = ()  # role names that are interchangeable
    display: str = "{}"

    def role(self, name: str) -> Role | None:
        return next((r for r in self.roles if r.name == name), None)


# --------------------------------------------------------------------------- #
# The registry. DATA, not types -- this is where a new analysis family lands.
# --------------------------------------------------------------------------- #
SHAPES: dict[str, Shape] = {
    # ---- built ----
    "gene": Shape("gene", (Role("g", EntityKind.GENE),), display="{g}"),
    "gene_set": Shape(
        "gene_set",
        (Role("set", EntityKind.GENE_SET),
         Role("member", EntityKind.GENE, Gate.MEMBER, many=True)),
        display="{set}",
    ),
    "cell_type": Shape("cell_type", (Role("ct", EntityKind.CELL_TYPE),), display="{ct}"),
    # A method-produced unit (a cluster, a region) as a subject in its own right.
    # Added after the marker workflow had to borrow "niche" to key a cluster:
    # structurally right, but a cluster is not a niche and the name misleads.
    # Spelling a cluster as a cell_type is worse -- for an ASSIGNMENT the label is
    # the thing being decided, so that subject would be circular.
    "unit": Shape("unit", (Role("u", EntityKind.UNIT),), display="{u}"),
    # ---- declared, not built: each of these is one dict entry away ----
    "cell_type_pair": Shape(                       # co-localisation (unordered)
        "cell_type_pair",
        (Role("a", EntityKind.CELL_TYPE), Role("b", EntityKind.CELL_TYPE)),
        symmetric=("a", "b"), display="{a} <-> {b}",
    ),
    "adjacency": Shape(                            # neighbourhood (ordered)
        "adjacency",
        (Role("centre", EntityKind.CELL_TYPE), Role("neighbour", EntityKind.CELL_TYPE)),
        display="{centre} -> {neighbour}",
    ),
    "interaction": Shape(                          # ligand-receptor
        "interaction",
        (Role("ligand", EntityKind.GENE), Role("receptor", EntityKind.GENE),
         Role("source", EntityKind.CELL_TYPE), Role("target", EntityKind.CELL_TYPE)),
        display="{ligand}->{receptor} ({source}->{target})",
    ),
    "gene_pair": Shape(                            # co-expression edge
        "gene_pair",
        (Role("a", EntityKind.GENE), Role("b", EntityKind.GENE)),
        symmetric=("a", "b"), display="{a} - {b}",
    ),
    "module": Shape(                               # hub gene + its module
        "module",
        (Role("hub", EntityKind.GENE),
         Role("member", EntityKind.GENE, Gate.MEMBER, many=True)),
        display="{hub} module",
    ),
    "niche": Shape(                                # a niche and its composition
        "niche",
        (Role("niche", EntityKind.UNIT),
         Role("component", EntityKind.CELL_TYPE, many=True)),
        display="{niche}",
    ),
}


class UnknownShape(KeyError):
    """Raised for a shape that is not registered. Never guessed at."""


@dataclass(frozen=True, slots=True)
class Subject:
    """A subject in canonical form.

    Construct with :func:`subject`, never directly -- the constructor is what
    enforces canonical ordering, and a hand-built Subject can silently fail to
    compare equal to the same thing built properly.
    """

    shape: str
    parts: tuple[tuple[str, Entity], ...]  # (role, entity), canonical order

    @property
    def key(self) -> str:
        """Stable, sortable, join-safe identity. The one thing consumers may rely on.

        MEMBER roles are EXCLUDED. They are coverage, not identity: HALLMARK_EMT is
        HALLMARK_EMT whether 22 or 180 of its genes fell inside the capability frame.
        Including them made a subject's identity depend on the apparatus, so the same
        gene set read off a 280-gene panel and a 5,000-gene panel produced two keys
        that could never join -- and reconciliation is a join across task kinds on
        exactly this key, so it would have under-reported in silence.

        This also makes the rule consistent with ``resolve()``, where MEMBER roles are
        already excluded from the meet and contribute only via ``Coverage``.
        """
        sh = SHAPES.get(self.shape)
        parts = self.parts if sh is None else tuple(
            (r, e) for r, e in self.parts
            if (ro := sh.role(r)) is None or ro.gate is not Gate.MEMBER
        )
        return self.shape + "|" + "|".join(f"{r}={e}" for r, e in parts)

    def one(self, role: str) -> Entity | None:
        return next((e for r, e in self.parts if r == role), None)

    def all(self, role: str) -> tuple[Entity, ...]:
        return tuple(e for r, e in self.parts if r == role)

    def entities(self) -> tuple[Entity, ...]:
        return tuple(e for _, e in self.parts)

    def roles_with_gate(self, gate: Gate) -> tuple[tuple[str, Entity], ...]:
        sh = SHAPES[self.shape]
        return tuple((r, e) for r, e in self.parts
                     if (ro := sh.role(r)) is not None and ro.gate is gate)

    def display(self) -> str:
        sh = SHAPES.get(self.shape)
        if sh is None:
            return self.key
        vals = {r: (e.label or e.id) for r, e in self.parts}
        try:
            return sh.display.format(**vals)
        except (KeyError, IndexError):
            return self.key

    def __str__(self) -> str:
        return self.key


def subject(shape: str, **roles: Entity | Iterable[Entity]) -> Subject:
    """The ONE constructor. Validates against the registry and canonicalises.

    Raises rather than guessing: an unregistered shape, an unknown role, a missing
    required role, or an entity of the wrong kind are all errors. A subject that
    cannot be built correctly must not be built at all -- a silently malformed one
    would corrupt the join key that reconciliation depends on.
    """
    sh = SHAPES.get(shape)
    if sh is None:
        raise UnknownShape(
            f"unregistered subject shape {shape!r}; known: {sorted(SHAPES)}"
        )
    declared = {r.name for r in sh.roles}
    unknown = set(roles) - declared
    if unknown:
        raise ValueError(f"shape {shape!r} has no role(s) {sorted(unknown)}; "
                         f"declared: {sorted(declared)}")

    parts: list[tuple[str, Entity]] = []
    for role in sh.roles:                       # declaration order, not kwargs order
        if role.name not in roles:
            if role.gate is Gate.REQUIRED and not role.many:
                raise ValueError(f"shape {shape!r} requires role {role.name!r}")
            continue
        val = roles[role.name]
        ents = list(val) if isinstance(val, (list, tuple, set, frozenset)) else [val]
        if not role.many and len(ents) != 1:
            raise ValueError(f"role {role.name!r} of shape {shape!r} takes one entity, "
                             f"got {len(ents)}")
        for e in ents:
            if not isinstance(e, Entity):
                raise TypeError(f"role {role.name!r} expects Entity, got {type(e).__name__}")
            if e.kind is not role.kind:
                raise ValueError(f"role {role.name!r} of shape {shape!r} expects "
                                 f"{role.kind.value}, got {e.kind.value} ({e.id})")
        if role.many:
            ents.sort()                          # a member set has no meaningful order
        parts.extend((role.name, e) for e in ents)

    # Canonicalise symmetric role groups: an unordered pair must be ONE subject
    # however the reader emitted it.
    if sh.symmetric:
        sym = [p for p in parts if p[0] in sh.symmetric]
        if sym:
            ordered = sorted((e for _, e in sym))
            names = sorted(sh.symmetric)
            replacement = dict(zip(names, ordered))
            out, seen = [], set()
            for name, ent in parts:
                if name in sh.symmetric:
                    if name in seen:
                        continue
                    seen.add(name)
                    out.append((name, replacement[name]))
                else:
                    out.append((name, ent))
            parts = out
    return Subject(shape=shape, parts=tuple(parts))


def gene(symbol: str, label: str = "") -> Entity:
    return Entity(EntityKind.GENE, symbol, label)


def cell_type(name: str, label: str = "") -> Entity:
    return Entity(EntityKind.CELL_TYPE, name, label)


def unit(uid: str, label: str = "") -> Entity:
    return Entity(EntityKind.UNIT, uid, label)


def gene_set(name: str, label: str = "") -> Entity:
    return Entity(EntityKind.GENE_SET, name, label)
