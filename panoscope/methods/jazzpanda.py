"""jazzPanda's spatial GLM, declared: what its paper says it measures, and does not.

Almost all *declaration*. ``measures``, ``does_not_measure``, ``assumptions`` and
``statistic_name`` are transcribed from
``panoscope/skills/marker-interpretation/SKILL.md`` (itself from the jazzPanda paper's
Methods and the package docs).

**The failure this prevents:** a ``glm_coef`` read as a fold change. jazzPanda asks
whether a gene's transcripts fall in the same places as a cluster's cells, not
whether it is higher on average there; ``does_not_measure`` carries that question
with the number.

**Where the rubric stops.** This module owns :data:`BAND_EDGES` (its own statistic's
magnitude -> band) and the specificity thresholds that raise ``specificity:*`` flags.
The promote, demotes, cap and floor are ``core/verdict.py``'s confidence policy, so
each method maps its own statistic through its own edges (ARCHITECTURE §9.4).

**``load`` reads; it never recomputes.** ``top_result`` is a consumed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
from panoscope.core.verdict import CONFIDENCE_BANDS
from panoscope.methods import _table
# ``is_no_signal`` from the adapter, not a fifth private copy of ``!= NOSIG``.
# A method importing a concrete platform is the wrong direction and is debt:
# the sentinel is jazzPanda's own (``NOSIG`` below) while the predicate lives
# beside ``NO_SIGNAL`` in the adapter, where this codebase already put it and
# where ``agent/grounding_check`` already imports it from. One answer in the
# process matters more than which module holds it; re-homing it is a move,
# not a fix, and would touch every importer.
from panoscope.platforms.xenium import is_no_signal
from panoscope.types.contracts import PlatformAdapter, UnitSpec, UnrecognisedResult
from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.measured import Frame, Measured, MeasuredState, MeasuredWorld
from panoscope.types.subject import EntityKind, Subject, gene, subject
from panoscope.types.subject import unit as unit_entity
from panoscope.types.trace import Trace, Traced, fmt_num, ref

# --------------------------------------------------------------------------- #
# The schema this method can read. A table missing any of these is not a
# jazzPanda top_result, and guessing at one routes the rows to an interpretive
# question they do not answer -- every answer underneath confidently wrong.
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: tuple[str, ...] = ("gene", "top_cluster", "glm_coef")
CORROBORATING_COLUMNS: tuple[str, ...] = ("pearson", "max_gc_corr", "max_gg_corr")

# jazzPanda's own sentinel: the top coefficient fell below ``coef_cutoff`` (default
# 0.05) or its best match was a background term. In frame, tested, not assigned
# here -- which is TESTED_ABSENT exactly, and is why these rows are not units.
NOSIG: str = "NoSig"

UNIT_KIND: str = "cluster"

# --------------------------------------------------------------------------- #
# The method's own rubric: magnitude of ITS statistic -> band. The edges live in
# core/verdict.CONFIDENCE_BANDS (the engine that applies them); this is the same
# ladder as (label, edge) pairs, descending, floor last.
# --------------------------------------------------------------------------- #
BAND_EDGES: tuple[tuple[str, float], ...] = tuple(CONFIDENCE_BANDS.items())

# The three specificity conditions, with the thresholds the SKILL rubric states.
# Raised ONCE, here, as registered flag ids; core/verdict.py branches on the ids.
LOW_PEARSON: float = 0.30       # weak spatial specificity on this gene
GG_NOT_UNIQUE: float = 0.98     # another gene shares this spatial pattern
EPS: float = 1e-9               # max_gc_corr > pearson must be a real difference


def _flags_for(pearson: float, max_gc_corr: float, max_gg_corr: float) -> tuple[str, ...]:
    """Registered ``specificity:*`` ids for one row. Order is code order, hashed.

    Ids, never prose, so the interface matches on a stable id rather than an
    English sentence a copy edit could change.
    """
    out: list[str] = []
    if max_gc_corr > pearson + EPS:
        out.append("specificity:localises_elsewhere")
    if max_gg_corr >= GG_NOT_UNIQUE:
        out.append("specificity:not_unique")
    if pearson < LOW_PEARSON:
        out.append("specificity:low")
    return tuple(out)


#: Columns that carry a measurement, and so must be finite when present.
_STAT_COLUMNS = ("glm_coef", "pearson", "max_gg_corr", "max_gc_corr")


def _rows(table: Any) -> tuple[dict[str, Any], ...]:
    """Rows as plain dicts, in the table's own order.

    The file's order is not statistic-descending (``markers_top.csv`` is
    alphabetical); ``load`` sorts per unit, which the driver choice depends on.
    """
    return _normalised(_table.rows(table))


class MalformedMethodOutput(ValueError):
    """The method's own table cannot be read as a measurement.

    Raised rather than worked around. This is a trust boundary: the table is
    produced by someone else, and each shape below produced a confident wrong
    answer instead of a refusal.
    """


def _normalised(raw: tuple[dict, ...]) -> tuple[dict, ...]:
    """Strip symbols, and refuse a table that cannot mean one thing.

    * **Whitespace** is stripped (normalisation, not invention).
    * **A statistic that is not a finite number** is refused: ``statistic is None
      iff NOT OBSERVED`` cannot hold for a NaN.
    * **Two rows for one (unit, gene) with different numbers** are refused.

    An exactly duplicated row is collapsed.
    """
    seen: dict[tuple[str, str], Any] = {}
    out: list[dict] = []
    for r in raw:
        row = dict(r)
        for col in ("gene", "top_cluster"):
            if col in row and isinstance(row[col], str):
                row[col] = row[col].strip()
        gene, unit = str(row.get("gene", "")), str(row.get("top_cluster", ""))

        for col, value in row.items():
            if col in _STAT_COLUMNS and value is not None:
                try:
                    f = float(value)
                except (TypeError, ValueError):
                    continue
                if f != f or f in (float("inf"), float("-inf")):
                    raise MalformedMethodOutput(
                        f"[jazzpanda] {unit}/{gene}: {col} is {value!r}, which is "
                        f"not a number. A statistic the method could not produce "
                        f"must not arrive as one -- fix or drop the row in "
                        f"inputs/markers_top.csv."
                    )

        key = (unit, gene)
        if key in seen:
            prior = seen[key]
            if prior == row:
                continue
            differing = sorted(c for c in set(prior) | set(row)
                               if prior.get(c) != row.get(c))
            raise MalformedMethodOutput(
                f"[jazzpanda] {unit}/{gene} appears more than once with "
                f"different values ({', '.join(differing)}). One measurement "
                f"cannot carry two numbers; fix inputs/markers_top.csv."
            )
        seen[key] = row
        out.append(row)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class JazzPandaSpatialGLM:
    """The declaration. Every field is a sentence from the paper, not from us.

    A frozen dataclass rather than a module of constants so that the registry holds
    one object whose attributes cannot be rebound after registration: a declaration
    that a later import could edit is a declaration nothing downstream can trust.
    """

    name: str = "jazzpanda_spatial_glm"
    #: How the package names ITSELF, for anywhere a person reads it.
    display_name: str = "jazzPanda"
    workflow: str = "marker-interpretation"
    statistic_name: str = "glm_coef"

    measures: str = (
        "whether a gene's transcript detections fall in the same tissue tiles as a "
        "cluster's cells: the lasso/GLM coefficient on that cluster's binned-count "
        "vector, with sample/batch and negative-control background vectors as "
        "covariates. Larger means a stronger spatial association."
    )

    # MANDATORY. The half a reader cannot reconstruct from the number itself.
    does_not_measure: str = (
        "mean expression, fold change or any cell-level differential expression "
        "(no pct.1/pct.2, no avg_log2FC); per-cell abundance; cell-type identity; "
        "causation or regulation; and nothing about a gene that is not on the panel. "
        "A high glm_coef does not mean a gene is highly expressed, only that its "
        "transcripts co-locate with the cluster."
    )

    # MANDATORY. Each is a condition under which the coefficient means what it says.
    assumptions: tuple[str, ...] = (
        "transcript counts are used on their native scale (no normalisation), so the "
        "gene-vector/cluster-vector relationship stays linear",
        "the analysis grid (bin_param, e.g. 40x40 tiles) was chosen so that the "
        "average is about one cell per tile; it is baked into the precomputed result "
        "and is not a viewing control",
        "cluster labels are fixed and were produced upstream; the model conditions on "
        "them and cannot question them",
        "negative-control and sample vectors are present as covariates, so background "
        "and batch are adjusted for rather than assumed absent",
        "each gene is assigned to at most ONE cluster (maximum coefficient, minimum "
        "p-value), so a gene's coefficient exists only for its winning cluster",
        "power is lower for small or rare clusters, and their marker ranking is less "
        "stable",
        "only panel genes can ever be assigned; absence of an off-panel gene is not a "
        "measurement",
    )

    # Rivals come from reference data (a curated cell-type marker list), never from
    # the data: jazzPanda assigns one cluster per gene, so the table itself cannot
    # propose an alternative account of a cluster.
    rival_space: Literal["reference", "data", "none"] = "reference"

    # MANDATORY-ADJACENT: makes TESTED_ABSENT reachable. jazzPanda's top_result is
    # not truncated: it enumerates the whole analysed panel, one row per gene
    # (``NoSig`` for genes assigned nowhere), so a gene that peaked in c4 was
    # tested in c1 and did not come up there.
    enumerates_absence: bool = True

    # Looked up live (Bioconductor, 2026-11-02), not written from memory. The
    # package DOI is the citation that resolves today; the Methods pages the SKILL
    # transcribes are cited there as page numbers, and no PMID is recorded in-repo.
    citation: str = (
        "Jin M, Phipson B. jazzPanda: finding spatially relevant marker genes in "
        "image based spatial transcriptomics data. Bioconductor R package. "
        "doi:10.18129/B9.bioc.jazzPanda "
        "(https://bioconductor.org/packages/jazzPanda)"
    )

    # ----------------------------------------------------------------------- #
    def detect_unit(self, table: Any) -> UnitSpec:
        """Read the unit kind and ids FROM THE SCHEMA, never from a caller's say-so.

        ``NoSig`` (*tested here, assigned nowhere*) is not a unit; it is excluded in
        ANY case, with blank and NaN, through the adapter's ``is_no_signal`` so
        there is one answer to "is this the sentinel".
        """
        cols = _table.columns(
            table, "jazzPanda's top_result is a table with columns "
                   f"{list(REQUIRED_COLUMNS)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not a jazzPanda top_result -- missing column(s) "
                f"{missing}; saw {list(cols)}"
            )
        seen = {str(r["top_cluster"]).strip() for r in _rows(table)
                if not is_no_signal(r["top_cluster"])}
        return UnitSpec(
            unit_kind=UNIT_KIND,
            unit_column="top_cluster",
            # Sorted, not first-seen: the unit list is hashed into the record, and
            # row order in a CSV is not a fact about the study.
            units=tuple(sorted(seen)),
            because=(
                f"columns {list(REQUIRED_COLUMNS)} present; units read off "
                f"'top_cluster' with {NOSIG!r} (in any case, and blank) excluded "
                "as jazzPanda's tested-but-unassigned sentinel",
            ),
        )

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> Traced[dict[str, NormalisedEvidence]]:
        """Precomputed ``top_result`` in, normalised evidence out, keyed by unit.

        Reads, never recomputes. Capability frames and standing conditions come from
        the adapter, never derived here. Returns ``Traced`` because loading
        DECIDES: it drops rows, applies frames and withholds statistics the frame
        refused.
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        # Control rows leave before anything here calls them a subject. The adapter
        # owns the rule (``PlatformAdapter.is_control``) because it is platform
        # knowledge, and this method is a declaration of what jazzPanda measures,
        # not of what 10x ships in the probe table. Dropped from the READER, never
        # from the file: a control's count is the assay's noise floor.
        table = table[~table["gene"].astype(str).map(
            lambda g: adapter.is_control(store, g))].reset_index(drop=True)
        spec = self.detect_unit(table)
        world = MeasuredWorld(adapter.frames(store))
        conditions = adapter.standing_conditions(store)
        rows = _rows(table)
        # Every gene the table names, first-appearance order, INCLUDING the rows
        # whose unit column is the NoSig sentinel. A gene jazzPanda assigned nowhere
        # was still looked at, so it belongs to the domain absence is enumerated
        # over; the capability frames, not this method, decide what each absence is
        # (the panel and detectability frames turn the NoSig genes into
        # NEVER_MEASURED, and everything else into a tested absence that argues).
        universe: tuple[str, ...] = tuple(dict.fromkeys(str(r["gene"]) for r in rows))

        out: dict[str, NormalisedEvidence] = {}
        for uid in spec.units:
            # Sort by descending statistic, as agent/data.get_cluster_markers
            # already does. core/verdict._drivers takes the FIRST eligible
            # canonical marker in caller order, so the driver -- and the band read
            # off it -- is decided here. A row with no statistic sorts last: it
            # cannot drive a band it carries no number for.
            mine = tuple(sorted(
                (r for r in rows if str(r["top_cluster"]) == uid),
                key=lambda r: (r.get(self.statistic_name) is not None,
                               r.get(self.statistic_name) or 0.0),
                reverse=True,
            ))
            # The absences follow every assigned row, in the file's own gene order,
            # which is the same order for every unit. They carry no statistic, so
            # core/verdict._drivers skips them and the band is still read off the
            # first eligible canonical row -- but putting them anywhere other than
            # last would make the driver depend on where an unnumbered row landed.
            assigned = frozenset(str(r["gene"]) for r in mine)
            readings = tuple(self._reading(r) for r in mine) + (tuple(
                Reading(subject=subject("gene", g=gene(g)), statistic=None,
                        statistic_name=self.statistic_name)
                for g in universe if g not in assigned
            ) if self.enumerates_absence else ())
            reported = Frame(
                EntityKind.GENE,
                f"reported:{self.name}:{uid}",
                frozenset(str(r["gene"]) for r in mine),
            )
            out[uid] = tr.absorb(measure_unit(
                unit=uid,
                unit_kind=spec.unit_kind,
                unit_subject=subject("unit", u=unit_entity(uid)),
                module=self.name,
                readings=readings,
                world=world,
                reported=(reported,),
                conditions=conditions,
            ))
        return tr.done(out)

    def _reading(self, row: dict[str, Any]) -> Reading:
        """One top_result row, as the method claims it -- before the frame rules.

        The corroborators travel as declared pairs rather than as extra statistics:
        ``pearson`` corroborates ``glm_coef``, it is not a second answer, and
        ``core/measure.py`` withholds them together so a pearson can never survive
        into an item whose coefficient the frame refused.
        """
        pearson = float(row.get("pearson", 0.0) or 0.0)
        max_gc = float(row.get("max_gc_corr", 0.0) or 0.0)
        max_gg = float(row.get("max_gg_corr", 0.0) or 0.0)
        return Reading(
            subject=subject("gene", g=gene(str(row["gene"]))),
            statistic=float(row["glm_coef"]),
            statistic_name=self.statistic_name,
            corroborating=(("max_gc_corr", max_gc), ("max_gg_corr", max_gg),
                           ("pearson", pearson)),
            flags=_flags_for(pearson, max_gc, max_gg),
        )

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        """What jazzPanda's silence about this subject MEANS, in jazzPanda's terms.

        One sentence, deterministic, no clock and no model. The four states are
        generic; what a tested absence *argues* is not -- for a spatial method it
        argues "measured, and it localises somewhere else", which is the half of the
        lattice a rival can actually be ruled out with.
        """
        who = subject.display()
        if measured.state is MeasuredState.NEVER_MEASURED:
            return (f"{who} is not on the panel ({measured.frame}); jazzPanda could "
                    "not have detected it here, so its absence argues nothing")
        if measured.state is MeasuredState.UNDERPOWERED:
            return (f"{who} is on the panel but below the floor in {measured.frame}; "
                    "jazzPanda had too little signal to speak about it")
        if measured.state is MeasuredState.TESTED_ABSENT:
            return (f"{who} was on the panel and tested, and its transcripts did not "
                    "co-locate with this cluster -- measured absence, which argues "
                    "against any account that needs it here")
        return (f"{who} was observed: its transcripts co-locate with this cluster's "
                "cells")

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        """Base band off THIS method's statistic, with the decision that produced it.

        No promote, demote, cap or floor: those are ``core/verdict.py``'s policy.
        ``abs()``: a strongly negative coefficient is a strong measurement; direction
        rides on the ``EvidenceItem``.
        """
        t = Trace()
        lead = next((i for i in ev.items if i.measured.carries_number), None)
        floor = BAND_EDGES[-1][0]
        if lead is None:
            # No number at all. The band for "nothing was observed here" is the
            # floor, and it is recorded as such: a unit whose band simply did not
            # appear is indistinguishable from one the rubric never ran on.
            t.note("band:no_canonical_driver", floor,
                   (f"{len(ev.items)} item(s), none carrying a {self.statistic_name}",),
                   (ref(ev.unit_subject),))
            return t.done(floor)

        c = abs(float(lead.statistic or 0.0))
        band = next((lab for lab, edge in BAND_EDGES if c >= edge), floor)
        t.note("band:from_coef", band,
               (f"{lead.statistic_name}={fmt_num(lead.statistic)} on "
                f"{lead.subject.display()}, at or above {fmt_num(_edge(band))}",),
               (ref(lead.subject, lead.statistic_name, lead.statistic),))
        return t.done(band)

    def traversals(self) -> tuple[tuple[str, str], ...]:
        """Reference-graph edges this method's evidence licenses following.

        A spatial co-location coefficient licenses "what is this gene a canonical
        marker of" and "what else marks that type" -- nothing about regulation or
        pathway membership.
        """
        return (("marker_of", "forward"), ("marker_of", "reverse"))


def _edge(band: str) -> float:
    return next(e for lab, e in BAND_EDGES if lab == band)


#: The registered instance. One object, built once, attributes frozen.
MODULE = JazzPandaSpatialGLM()
