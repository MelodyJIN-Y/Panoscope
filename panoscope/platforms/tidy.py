"""The method-agnostic reader: a tidy table in, a ``ReadingRecord`` and rows for
``core.measure`` out.

One row per unit x entity. That is the only shape this module knows, and knowing one
shape is what makes it method-agnostic: the column *names* are parameters, so
jazzPanda's ``gene, top_cluster, glm_coef`` and a DE result's
``gene, contrast, log2fc`` are the same read with different arguments. Nothing here
knows what a gene is, what a cluster means, or which statistic is primary.

**Why the reader, and not the method, counts the drops.** ``rows_in`` minus
``rows_dropped`` is the only place a biologist can see that part of a table never
reached the interpretation, and a method that drops its own rows reports a clean run
over whatever survived. Every drop is counted under a reason string, and the reasons
are a sorted tuple of pairs (``ReadingRecord.__post_init__`` re-sorts, so this does
not depend on anyone remembering): two runs that dropped the same rows for the same
reasons must hash identically, and a dict would make the bytes depend on which
malformed row a reader happened to meet first.

**``enumerates_absence`` is the one judgement call, and it defaults to the cautious
side.** ``TESTED_ABSENT`` -- the productive half of the lattice -- exists only if
something can say "this entity was tested here and not reported". For a table that
names every entity it tested, the entities a unit does *not* claim are exactly that,
and enumerating them is what turns the table into evidence against a rival. For a
table truncated to the top 100 rows it is a fabrication: the unnamed genes were not
tested-absent, they were cut off. The reader cannot tell the two apart from the
bytes, so the caller declares it, and the default is ``False`` -- under-reporting a
measured absence is a weaker answer, while inventing one is an overclaim, and this
project's floor is confidence.

**The entity universe includes rows that were dropped.** A row dropped because its
unit column held a sentinel (jazzPanda's ``NoSig``) still tells us the method *looked
at* that entity, so it belongs to the universe absence is enumerated over. This is
load-bearing for the demo: the 49 undetectable panel genes reach every cluster's
evidence as readings with no number, where the detectability frame resolves them to
``NEVER_MEASURED`` -- visible, rather than a blank row.

**Statistics are carried in unadjudicated.** ``Reading`` takes whatever the file said,
including a number for an entity the frame will refuse; ``core.measure`` is what
withholds it and records the withholding. A reader that pre-filtered would delete the
difference between "the method reported nothing" and "the method reported something
the apparatus could not support", which is the whole thing being audited.

**On the returned ``Traced``: its decision tuple is always empty, and that is a
contract gap, not an oversight.** ``Trace.note`` refuses any rule id absent from
``trace.RULES``, and ``RULES`` has no ``read:*`` step -- reading is not one of the
registered steps. The reader's reasoning is therefore carried entirely by the typed
``ReadingRecord``, which is arguably where it belongs (it is hashed, diffable and
click-backable in a way a prose decision is not). ``Traced`` is returned anyway
because every value crossing out of this layer is expected to be one. If reading
should narrate itself, ``RULES`` needs read-step entries, and that is a reviewed
contract change.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from panoscope.core.measure import Reading
from panoscope.platforms.base import digest
from panoscope.types.contracts import UnrecognisedResult
from panoscope.types.measured import Frame, MeasuredWorld
from panoscope.types.record import ReadingRecord
from panoscope.types.subject import Entity, EntityKind, subject
from panoscope.types.trace import Traced

_SINGLE_ROLE_SHAPE: Final[dict[EntityKind, tuple[str, str]]] = {
    EntityKind.GENE: ("gene", "g"),
    EntityKind.CELL_TYPE: ("cell_type", "ct"),
    EntityKind.GENE_SET: ("gene_set", "set"),
}
"""Entity kind -> the registered one-entity subject shape and its role name.

Derived from ``subject.SHAPES`` by hand rather than by introspection: a shape is
single-entity by intent, not by arity, and a rule like "exactly one REQUIRED role"
would silently pick up ``module`` and ``niche`` the day their member roles changed
gate. A kind absent from this map has no one-entity shape, and the reader refuses
rather than building a ``Subject`` directly -- a hand-built Subject can fail to
compare equal to the same thing built through ``subject()``, which breaks the one
join key reconciliation depends on.
"""


@dataclass(frozen=True, slots=True)
class UnitRead:
    """One unit's rows, in the two shapes ``core.measure.measure_unit`` wants.

    ``reported`` is a tuple of ``Frame`` rather than a flat id set because that is
    what ``NormalisedEvidence`` carries and what ``measure_unit`` takes; re-deriving
    it from a flat set downstream would mean guessing which entity kind each id
    belonged to.
    """

    unit: str
    readings: tuple[Reading, ...]
    reported: tuple[Frame, ...]


@dataclass(frozen=True, slots=True)
class TidyRead:
    """What one read of one file produced: the record, and the rows per unit.

    ``units`` is a tuple sorted by unit id, not a dict, for the reason every other
    frozen carrier in this system avoids one: a dict field is unhashable and its
    insertion order leaks into ``dataclasses.asdict`` and from there into the
    serialised bytes.
    """

    record: ReadingRecord
    units: tuple[UnitRead, ...]

    def for_unit(self, unit: str) -> UnitRead | None:
        return next((u for u in self.units if u.unit == unit), None)


def _text(value: object) -> str:
    """A cell as a trimmed string; empty for anything null.

    One helper rather than per-column handling because a missing cell arrives as
    ``None``, ``float('nan')`` or ``pd.NA`` depending on the dtype the parser
    inferred, and the string ``"nan"`` as a gene id is a row that looks valid.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _read_table(path: Path) -> pd.DataFrame:
    """Parquet by suffix, CSV otherwise. Fails loudly, naming the file."""
    try:
        if path.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised as the contract's own error
        raise UnrecognisedResult(f"{path} could not be read as a table: {exc}") from exc


def read_tidy(
    path: str | Path,
    *,
    unit_column: str,
    entity_column: str,
    entity_kind: EntityKind,
    statistic_column: str,
    unit_kind: str,
    world: MeasuredWorld,
    clustering_mode: str,
    corroborating_columns: tuple[str, ...] = (),
    not_units: frozenset[str] = frozenset(),
    unit_detection_because: tuple[str, ...] = (),
    enumerates_absence: bool = False,
    reference_ids: frozenset[str] | None = None,
    record_path: str | None = None,
) -> Traced[TidyRead]:
    """Read one tidy result file into a record plus per-unit readings.

    ``record_path`` is what lands in ``ReadingRecord.path``, defaulting to ``path``.
    A caller passes a store-relative name here so the record does not embed an
    absolute path: the record is hashed into the golden artifact, and
    ``/Users/someone/...`` in it makes byte-stability a property of whose laptop
    produced it.

    ``reference_ids`` is the annotated vocabulary this study's entities are checked
    against. ``None`` means no reference was consulted, and
    ``entities_not_in_reference`` is then empty -- which reads identically to "every
    entity is annotated". That ambiguity is in the type, not here: ``ReadingRecord``
    has no way to say "not checked". Noted rather than papered over.

    Raises ``UnrecognisedResult`` when a named column is absent, or when
    ``entity_kind`` has no registered one-entity subject shape. Never guesses a
    column: a wrong unit column reinterprets every row beneath it.
    """
    p = Path(path)
    if entity_kind not in _SINGLE_ROLE_SHAPE:
        raise UnrecognisedResult(
            f"no one-entity subject shape is registered for {entity_kind.value!r}; "
            f"known: {sorted(k.value for k in _SINGLE_ROLE_SHAPE)}"
        )
    shape, role = _SINGLE_ROLE_SHAPE[entity_kind]

    df = _read_table(p)
    wanted = (unit_column, entity_column, statistic_column, *corroborating_columns)
    missing = [c for c in wanted if c not in df.columns]
    if missing:
        raise UnrecognisedResult(
            f"{p} has no column(s) {missing}; it has {list(df.columns)}. The unit and "
            "statistic columns are read from the schema and never guessed."
        )

    rows_in = int(len(df))
    units_raw = [_text(v) for v in df[unit_column]]
    ents_raw = [_text(v) for v in df[entity_column]]
    stats = pd.to_numeric(df[statistic_column], errors="coerce").tolist()
    corr = {c: pd.to_numeric(df[c], errors="coerce").tolist()
            for c in corroborating_columns}

    drops: Counter[str] = Counter()
    universe: list[str] = []           # file order; the absence-enumeration domain
    in_universe: set[str] = set()
    seen: set[tuple[str, str]] = set()
    kept: dict[tuple[str, str], Reading] = {}
    units_present: set[str] = set()

    for i in range(rows_in):
        ent = ents_raw[i]
        if not ent:
            drops[f"{entity_column} is empty"] += 1
            continue
        # Before the unit checks, deliberately: a row dropped for its unit column
        # still proves the method looked at this entity, and that is what makes it
        # eligible for a tested/never-measured verdict under every other unit.
        if ent not in in_universe:
            in_universe.add(ent)
            universe.append(ent)

        unit = units_raw[i]
        if not unit:
            drops[f"{unit_column} is empty"] += 1
            continue
        if unit in not_units:
            drops[f"{unit_column} is {unit!r}, declared not a unit"] += 1
            continue
        stat = stats[i]
        if stat is None or pd.isna(stat):
            drops[f"{statistic_column} is not a number"] += 1
            continue
        if (unit, ent) in seen:
            drops[f"duplicate {unit_column} x {entity_column} row"] += 1
            continue
        seen.add((unit, ent))
        units_present.add(unit)
        kept[(unit, ent)] = Reading(
            subject=subject(shape, **{role: Entity(entity_kind, ent)}),
            statistic=float(stat),
            statistic_name=statistic_column,
            corroborating=tuple(
                (c, float(corr[c][i])) for c in corroborating_columns
                if corr[c][i] is not None and not pd.isna(corr[c][i])
            ),
        )

    unit_reads: list[UnitRead] = []
    for unit in sorted(units_present):
        reported_ids = frozenset(e for (u, e) in kept if u == unit)
        readings: list[Reading] = []
        for ent in universe:                       # file order, for every unit alike
            hit = kept.get((unit, ent))
            if hit is not None:
                readings.append(hit)
            elif enumerates_absence:
                readings.append(Reading(
                    subject=subject(shape, **{role: Entity(entity_kind, ent)}),
                    statistic=None,
                    statistic_name=statistic_column,
                ))
        unit_reads.append(UnitRead(
            unit=unit,
            readings=tuple(readings),
            reported=(Frame(entity_kind, f"reported:{unit_kind}:{unit}", reported_ids),),
        ))

    kind_frames = world.for_kind(entity_kind)
    outside = tuple(sorted(
        e for e in universe
        if not all(f.contains(Entity(entity_kind, e)) for f in kind_frames)
    )) if kind_frames else tuple(sorted(universe))

    record = ReadingRecord(
        path=record_path if record_path is not None else str(p),
        sha256=digest(p),
        statistic_column=statistic_column,
        unit_column=unit_column,
        unit_kind=unit_kind,
        unit_detection_because=tuple(unit_detection_because),
        rows_in=rows_in,
        rows_dropped=int(sum(drops.values())),
        drop_reasons=tuple(drops.items()),       # re-sorted by ReadingRecord
        entities_outside_frame=outside,
        entities_not_in_reference=() if reference_ids is None
        else tuple(sorted(e for e in universe if e not in reference_ids)),
        frames=world.refs(),
        clustering_mode=clustering_mode,
    )
    # Empty decisions by construction -- see the module docstring. The record is the
    # read's audit trail; RULES has no read step to note one under.
    return Traced(TidyRead(record=record, units=tuple(unit_reads)))
