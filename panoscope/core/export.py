"""CSV export profiles, keyed on ``TaskKind``. One home for tables written twice.

Two CSV contracts already exist and were written by hand, independently:
``agent/verdict.py`` (11 columns, the marker verdict) and ``agent/enrichment.py``
(14 columns, one row per surfaced pathway). They are the same object -- a projection
of an interpretation onto a flat, R-importable table -- and nothing said so, so the
column name and the value that fills it lived in two places per table, four in all.
That is the failure this module exists to prevent: a column renamed in the header
tuple and not in the row builder shifts every cell one to the left, the file still
parses, and downstream R reads ``lineage`` out of the ``category`` column. Pairing
the name with its getter in one :class:`Column` makes that particular drift
unrepresentable rather than merely unlikely.

**Byte-identity is the contract, not a nicety.** The 11-column verdict CSV is pinned
by ``tests/test_golden.py`` and by ``skills/marker-interpretation/SKILL.md`` Output 2,
and it is consumed by downstream R. So the profiles below reproduce what the two
exporters do *today*, including their inconsistencies, and ``tests/test_core_export.py``
proves it by string equality against those functions rather than by inspection:

- the verdict table formats nothing. ``confidence_score`` goes to ``csv`` as a raw
  float, so the anchor 0.70 lands as ``0.7`` and not ``0.70``. Writing ``f"{x:.2f}"``
  here would read as a tidy-up and would move the golden hash;
- the enrichment table pre-formats: ``.4f`` for ``score`` and ``panel_coverage``,
  ``.3e`` for ``q_value``, ``""`` for a missing one.

``fmt_num`` is deliberately **not** used. It is the one home for numbers that go into
*decisions*, where a drifting precision corrupts the record; these cells are an
established on-disk interface whose precisions predate it, and unifying the two would
be a breaking change to a committed artifact disguised as consistency.

**Both csv spellings are preserved.** The verdict exporter uses ``csv.DictWriter``
with dict rows, the enrichment one ``csv.writer`` with tuple rows. The two agree on
output today -- ``DictWriter`` is a thin shim over a ``writer`` with the same dialect
-- but ``dict_rows`` keeps each profile pinned to the API its committed file was
written with, so a future dialect or quoting change made for one table cannot reach
the other by having silently become shared.

**Why no decisions.** Every public function here returns ``Traced[str]`` with an
empty decision tuple, which is not an oversight. An export makes no judgement: it
projects an interpretation that has already been made, and the decisions that
produced it belong to the step that made them. A ``note`` emitted here would claim
the export chose something, and the record would carry a rule that explains a
formatting choice as if it were an inference.

**Nothing calls this yet, and that is correct.** Rewiring ``verdict.py`` and
``enrichment.py`` through these profiles is the method refactor (IMPLEMENTATION.md
§10), a later task with a visible surface.

**``cluster_confidence`` is gone, and this is the deletion §10 sequenced.** The
characterisation profile used to end with ``cluster_confidence`` and
``cluster_verify``. A band on a characterisation row is exactly the shape the no-band
rule exists to catch -- there is no label space for a pathway finding, so a unit-level
band over it is undefined, not merely unhelpful. The defence previously written here
was that it is the *parent cluster assignment's* band riding along on the row; that
defence does not survive contact with the file, because the column is named
``cluster_confidence`` on a row whose other thirteen cells are about a gene set, and
downstream R reading it beside ``gene_set`` and ``tier`` reads "how sure are we this
pathway is running". A column cannot carry a caveat.

Three count columns replace it -- ``cluster_n_supported``, ``cluster_n_suggestive``,
``cluster_n_untestable`` -- which is the header IMPLEMENTATION.md §10 specifies for a
task kind with no label space, and is strictly more informative than the band it
replaces: the untestable count is the only thing in the file that says how many of
this cluster's scored programs the panel could not ask the question of at all, since
those rows are not exported. ``cluster_verify`` stays, since "a human should look" is
well-defined without a label space.

All four come from ``agent/enrichment.ClusterEnrichment``: ``.census`` for the counts,
``.verify`` for the flag. The counts are properties over the row lists rather than
stored fields, so a count here cannot disagree with the rows beneath it.

Row shapes differ between profiles and must. An assignment row is one unit; a
characterisation row is one *item*, so the unit it belongs to has to travel with it
as a ``(unit, item)`` pair. Flattening both to "a row is a unit" would lose the
per-item grain that is the whole header for a no-band task kind.

The getters are duck-typed on purpose: ``core/`` does not import ``agent/``, and
annotating the row as ``ClusterVerdict`` would invert the layering to buy a type
check that the byte-equality tests already make redundant.
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
    """The CSV contract for one task kind.

    ``dict_rows`` records which ``csv`` API the committed file was written with; see
    the module docstring for why that is preserved rather than normalised.
    """

    task_kind: TaskKind
    columns: tuple[Column, ...]
    dict_rows: bool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _flag(b: object) -> str:
    """R's ``read.csv`` reads TRUE/FALSE as logical and ``True``/``False`` as factor,
    so the spelling is part of the contract, not a style choice."""
    return "TRUE" if b else "FALSE"


# --------------------------------------------------------------------------- #
# ASSIGNMENT -- the 11-column marker verdict (agent/verdict.py CSV_COLUMNS).
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
# (agent/enrichment.py ENRICHMENT_CSV_COLUMNS). The row is a (unit, item) pair:
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
    TaskKind.ASSIGNMENT: Profile(TaskKind.ASSIGNMENT, ASSIGNMENT_COLUMNS, dict_rows=True),
    TaskKind.CHARACTERISATION: Profile(
        TaskKind.CHARACTERISATION, CHARACTERISATION_COLUMNS, dict_rows=False
    ),
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
    if profile.dict_rows:
        w = csv.DictWriter(buf, fieldnames=list(profile.names))
        if header:
            w.writeheader()
        for row in rows:
            w.writerow({c.name: c.get(row) for c in profile.columns})
    else:
        tw = csv.writer(buf)
        if header:
            tw.writerow(profile.names)
        for row in rows:
            tw.writerow(tuple(c.get(row) for c in profile.columns))
    return Traced(value=buf.getvalue())
