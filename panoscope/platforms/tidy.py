"""The method-agnostic reader: a tidy table in, a ``ReadingRecord`` out.

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

**The entity universe includes rows that were dropped.** A row dropped because its
unit column held a sentinel (jazzPanda's ``NoSig``) still tells us the method *looked
at* that entity, so it belongs to the universe ``entities_outside_frame`` is computed
over: the demo's 49 undetectable panel genes are named there rather than vanishing.

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
from pathlib import Path
from typing import Final

import pandas as pd

from panoscope.platforms.base import digest
from panoscope.types.contracts import UnrecognisedResult
from panoscope.types.measured import MeasuredWorld
from panoscope.types.record import ReadingRecord
from panoscope.types.subject import Entity, EntityKind
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
    not_units: frozenset[str] = frozenset(),
    not_subjects: frozenset[str] = frozenset(),
    unit_detection_because: tuple[str, ...] = (),
    record_path: str | None = None,
) -> Traced[ReadingRecord]:
    """Read one tidy result file into its ``ReadingRecord`` (counts, drops, frames).

    ``record_path`` is what lands in ``ReadingRecord.path``, defaulting to ``path``.
    A caller passes a store-relative name here so the record does not embed an
    absolute path: the record is hashed into the golden artifact, and
    ``/Users/someone/...`` in it makes byte-stability a property of whose laptop
    produced it.

    ``not_subjects`` names entities that are measured but are not subjects of
    interpretation -- on Xenium, the negative control probes. They are dropped
    BEFORE the universe is built, which is the whole difference: an entity in the
    universe is one the method looked at, so every frame then gets asked about it
    and the ones that refuse it put it in ``entities_outside_frame``. A control
    refused by both gene frames is therefore counted and NAMED there, among the
    genes this panel could not speak to -- and it is not a gene the panel missed,
    it is a probe doing its job. The count is reported ("11 entity(s) the method
    named that no frame admits"), so six probes inflate a number a biologist reads
    as how much of the method's output the capability frames had to refuse.

    Dropped with a REASON, like every other drop here, so ``panoscope explain``
    prints "gene is a control probe, not a subject of interpretation: 6" rather
    than six rows silently disappearing between the file and the record.

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

    df = _read_table(p)
    wanted = (unit_column, entity_column, statistic_column)
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

    drops: Counter[str] = Counter()
    universe: list[str] = []           # file order; every entity the method looked at
    in_universe: set[str] = set()
    seen: set[tuple[str, str]] = set()

    for i in range(rows_in):
        ent = ents_raw[i]
        if not ent:
            drops[f"{entity_column} is empty"] += 1
            continue
        # Before the universe, deliberately -- see ``not_subjects`` in the docstring.
        if ent.strip().upper() in not_subjects:
            drops[f"{entity_column} is a control probe, not a subject of "
                  f"interpretation"] += 1
            continue
        # Before the unit checks, deliberately: a row dropped for its unit column
        # still proves the method looked at this entity, and that is what makes it
        # eligible for ``entities_outside_frame`` under every unit.
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
        entities_not_in_reference=(),  # no reference vocabulary is consulted here
        frames=world.refs(),
        clustering_mode=clustering_mode,
    )
    # Empty decisions by construction -- see the module docstring. The record is the
    # read's audit trail; RULES has no read step to note one under.
    return Traced(record)
