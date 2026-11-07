"""CSV export profiles, keyed on ``TaskKind``. The one writer for both CSVs.

Each :class:`Column` pairs a header name with its getter, so a renamed column cannot
shift every cell one to the left.

**Byte-identity is the contract.** The 11-column verdict CSV is pinned by
``tests/test_golden.py`` and ``skills/marker-interpretation/SKILL.md`` Output 2 and read
by downstream R, so the profiles reproduce the existing formatting exactly:

- the verdict table formats nothing (``confidence_score`` 0.70 lands as ``0.7``);
- the enrichment table pre-formats: ``.4f`` for ``score`` and ``panel_coverage``,
  ``.3e`` for ``q_value``, ``""`` for a missing one.

``fmt_num`` is deliberately not used; these cells are an established on-disk interface.

Every public function returns ``Traced[str]`` with no decisions: an export projects an
interpretation already made.

The characterisation profile carries no band (no label space, IMPLEMENTATION.md §10):
``cluster_n_supported``, ``cluster_n_suggestive``, ``cluster_n_untestable`` (from
``ClusterEnrichment.census``) and ``cluster_verify``. A characterisation row is a
``(unit, item)`` pair, keeping the per-item grain.

Getters are duck-typed: ``core/`` does not import ``agent/``.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from panoscope.types.outputs import TaskKind
from panoscope.types.trace import Traced


class NoExportProfile(KeyError):
    """Raised for a ``TaskKind`` with no registered profile.

    Fails closed rather than emitting a header-only file. A silently empty export is
    the worst outcome available: it looks like a run that found nothing, so the
    biologist reads "no pathways cleared the bar" where the truth is "this task kind
    has no table yet".
    """


@dataclass(frozen=True, slots=True)
class Column:
    """One CSV column: the header cell and the thing that fills it, together.

    Kept as one object because the two drifting apart is the failure in the module
    docstring, and because a header tuple beside a row builder gives no way to say
    "these are the same list" that a test can check.
    """

    name: str
    get: Callable[[Any], object]


@dataclass(frozen=True, slots=True)
class Profile:
    """The CSV contract for one task kind."""

    task_kind: TaskKind
    columns: tuple[Column, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _flag(b: object) -> str:
    """R's ``read.csv`` reads TRUE/FALSE as logical and ``True``/``False`` as factor,
    so the spelling is part of the contract, not a style choice."""
    return "TRUE" if b else "FALSE"


# --------------------------------------------------------------------------- #
# ASSIGNMENT -- the 11-column marker verdict.
# Frozen at 11 columns in exact order: IMPLEMENTATION.md §10.
# --------------------------------------------------------------------------- #
ASSIGNMENT_COLUMNS: tuple[Column, ...] = (
    Column("cluster", lambda v: v.cluster),
    Column("cell_type", lambda v: v.cell_type),
    Column("cell_type_short", lambda v: v.cell_type_short),
    Column("confidence", lambda v: v.confidence),
    # Raw float, NOT formatted -- see the module docstring. 0.70 -> "0.7".
    Column("confidence_score", lambda v: v.confidence_score),
    Column("key_markers", lambda v: ";".join(v.key_markers)),
    Column("notes", lambda v: v.notes),
    Column("category", lambda v: v.category),
    Column("lineage", lambda v: v.lineage),
    Column("exclude", lambda v: _flag(v.exclude)),
    Column("verify", lambda v: _flag(v.verify)),
)

# --------------------------------------------------------------------------- #
# CHARACTERISATION -- 14 columns, one row per surfaced pathway
# (the committed enrichment.csv). The row is a (unit, item) pair:
# untestable sets are not exported, because they were never a surfaced call.
# --------------------------------------------------------------------------- #
CHARACTERISATION_COLUMNS: tuple[Column, ...] = (
    Column("cluster", lambda r: r[0].cluster),
    Column("cell_type", lambda r: r[0].cell_type),
    Column("gene_set", lambda r: r[1].gene_set),
    Column("tier", lambda r: r[1].tier),
    Column("score", lambda r: f"{r[1].score:.4f}"),
    Column("score_kind", lambda r: r[1].score_kind),
    Column("q_value", lambda r: "" if r[1].q_value is None else f"{r[1].q_value:.3e}"),
    Column("panel_hits", lambda r: str(r[1].panel_hits)),
    Column("set_size_full", lambda r: str(r[1].set_size_full)),
    Column("panel_coverage", lambda r: f"{r[1].panel_coverage:.4f}"),
    Column("n_leading_edge", lambda r: str(r[1].n_leading_edge)),
    Column("leading_edge", lambda r: ";".join(r[1].leading_edge)),
    # The census: per-item counts, the header a task kind with no label space
    # permits. These replace cluster_confidence -- see the docstring.
    Column("cluster_n_supported", lambda r: str(r[0].census[0])),
    Column("cluster_n_suggestive", lambda r: str(r[0].census[1])),
    Column("cluster_n_untestable", lambda r: str(r[0].census[2])),
    Column("cluster_verify", lambda r: _flag(r[0].verify)),
)

# A module-level registry, the shape IMPLEMENTATION.md §6 specifies: a table, not a
# second dataclass per task kind. COMPARISON has no committed CSV yet and therefore
# no profile -- inventing one would pin a contract nothing has agreed to.
PROFILES: dict[TaskKind, Profile] = {
    TaskKind.ASSIGNMENT: Profile(TaskKind.ASSIGNMENT, ASSIGNMENT_COLUMNS),
    TaskKind.CHARACTERISATION: Profile(TaskKind.CHARACTERISATION, CHARACTERISATION_COLUMNS),
}


def to_csv(task_kind: TaskKind, rows: Iterable[Any],
           header: bool = True) -> Traced[str]:
    """Render ``rows`` through the profile registered for ``task_kind``.

    ``header=False`` exists so a caller can append to an open file; it is the only
    knob, because every other difference between the two tables is in the profile
    where a reviewer can diff it against the committed CSV.
    """
    try:
        profile = PROFILES[task_kind]
    except KeyError:
        raise NoExportProfile(
            f"no CSV export profile for task kind {task_kind!r}; "
            f"registered: {sorted(k.value for k in PROFILES)}"
        ) from None

    buf = io.StringIO()
    w = csv.writer(buf)
    if header:
        w.writerow(profile.names)
    for row in rows:
        w.writerow(tuple(c.get(row) for c in profile.columns))
    return Traced(value=buf.getvalue())
