"""jazzPanda's competitive gene-set test, declared: what it measures, and what it
does not -- which on a targeted panel is almost everything a reader will assume.

Registers into a *different* workflow from ``jazzpanda.py`` with no change to
``types/`` or ``core/``. The task kind is CHARACTERISATION (*what is RUNNING here*),
which has no label space, so there is no band ladder and :meth:`JazzPandaCGST.rubric`
raises.

**The failure this prevents:** a within-panel over-representation (ten of a
200-gene set on the panel) read as a genome-wide claim. ``does_not_measure`` writes
the gap down; :class:`~panoscope.types.measured.Coverage` enforces it (below the
floor the set is UNDERPOWERED and its statistic withheld).

**Panel hits are counted against the frames, not read off ``n_overlap``**: a set's
``genes_tested`` must be admitted by EVERY gene frame (panel and detectability), so
capability is declared by the apparatus, not by the method.

**Two bars, two owners.** ``Q_REPORTED`` is this method's reporting bar (below it a
scored set is TESTED_ABSENT); the *enriched versus suggestive* split is lab policy
in ``workflows/enrichment.py``.

**``load`` reads; it never recomputes.**
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
from panoscope.methods import _table
from panoscope.types.contracts import PlatformAdapter, UnitSpec, UnrecognisedResult
from panoscope.types.evidence import NormalisedEvidence
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.subject import EntityKind, Subject, gene, gene_set, subject
from panoscope.types.subject import unit as unit_entity
from panoscope.types.trace import Traced, Trace, fmt_num

# --------------------------------------------------------------------------- #
# The schema this method can read. A table missing any of these is not a jazzPanda
# competitive gene-set result, and guessing routes its rows to an interpretive
# question they do not answer.
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: tuple[str, ...] = (
    "gene_set", "cluster", "test_statistic", "p_adj_bh", "geneset_size",
    "genes_tested", "genes_selected",
)
CORROBORATING_COLUMNS: tuple[str, ...] = ("gc_corr",)

UNIT_KIND: str = "cluster"
UNIT_COLUMN: str = "cluster"

#: How a collection names its sets -> the collection. DERIVED from the result's
#: own set names (provenance), never asserted. A deliberate second copy of
#: ``agent/enrichment.py``'s table: ``methods/`` does not import from ``agent/``.
COLLECTION_BY_SET_PREFIX: tuple[tuple[str, str], ...] = (
    ("HALLMARK_", "MSigDB_Hallmark"),
)

#: What a result whose sets match no entry above was scored against. ``unknown`` is
#: first-class here and propagates; a wrong provenance does not. Add a prefix entry,
#: never a default.
UNKNOWN_COLLECTION: str = "unknown"


def _collection(sets: frozenset[str]) -> str:
    """The one collection this result was scored against, or ``unknown``.

    One frame covers the whole table, so a table whose sets do not all derive the
    same collection gets ``unknown`` rather than half an answer.
    """
    derived = {next((c for p, c in COLLECTION_BY_SET_PREFIX
                     if s.strip().upper().startswith(p)), UNKNOWN_COLLECTION)
               for s in sets}
    return derived.pop() if len(derived) == 1 else UNKNOWN_COLLECTION

# --------------------------------------------------------------------------- #
# The method's own bars. Each is a statement about THIS statistic, which is the
# only kind of threshold a method may own.
# --------------------------------------------------------------------------- #
#: The reporting bar. A scored set above this q is not reported for the unit --
#: TESTED_ABSENT, not "unknown". jazzPanda's NoSig, in the enrichment vocabulary.
Q_REPORTED: float = 0.25

#: Capability floors. Below either one the set is UNDERPOWERED: the question could
#: not be asked of it on this panel, so its absence argues nothing.
MIN_PANEL_HITS: int = 3      # set genes inside every gene frame
MIN_LEADING_EDGE: int = 3    # genes actually driving the enrichment


def _to_float(v: Any) -> float | None:
    """A cell as a float, or ``None`` for anything that is not one -- NaN included.

    NaN is not a small number here. A q of NaN is a test that returned nothing, and
    letting it compare False against the bar would file the set as a measured
    absence, which claims the opposite of what happened.
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _split(cell: Any) -> tuple[str, ...]:
    """A comma-joined gene cell, as symbols in the table's own order."""
    if cell is None:
        return ()
    s = str(cell).strip()
    if not s or s.lower() in ("nan", "na"):
        return ()
    return tuple(g.strip() for g in s.split(",") if g.strip())


def _in_frame(symbols: tuple[str, ...], world: MeasuredWorld) -> int:
    """How many of a set's genes EVERY gene frame admits.

    Every frame, not any: panel and detectability both. No gene frame at all counts
    as nothing measurable, matching the lattice's reading of a missing frame.
    """
    frames = world.for_kind(EntityKind.GENE)
    if not frames:
        return 0
    return sum(1 for s in symbols if all(s in f.members for f in frames))


class NoBandForThisTaskKind(ValueError):
    """Raised when a characterisation method is asked for a confidence band.

    Not a refusal to compute something hard. There is no label space, so the number
    being asked for does not exist -- and a method that answered with one would be
    handing the interface a figure it could only have invented.
    """


@dataclass(frozen=True, slots=True)
class JazzPandaCGST:
    """The declaration. Every field is a sentence about the method, not about us."""

    name: str = "jazzpanda_cgst"
    #: How the package names ITSELF, for anywhere a person reads it.
    display_name: str = "jazzPanda CGST"
    workflow: str = "enrichment-interpretation"
    statistic_name: str = "cgst_z"

    measures: str = (
        "over-representation of a gene set's PANEL-RESIDENT genes among the genes "
        "whose spatial vectors track this cluster: the set's lasso-selected genes "
        "are weighted by inverse standard error and compared, in a one-sided "
        "z-test, against the mean of all other genes on the same panel. The "
        "reported statistic is that z (column 'test_statistic'); larger means more "
        "over-represented relative to the rest of the panel."
    )

    # MANDATORY. Everything a reader will otherwise supply from the set's name.
    does_not_measure: str = (
        "pathway activity, flux, or whether a program is running: it is a "
        "competitive over-representation test, not an activity assay. Not "
        "genome-wide -- the universe is this panel, so a 200-gene Hallmark set "
        "enters as however many of its genes the panel carries, and a large z on a "
        "3-gene footprint is still three genes. Not expression level, not fold "
        "change, not per-cell abundance, not causation or regulation, and nothing "
        "at all about a set's genes that are off panel. It does not re-derive, "
        "confirm or question the cluster's cell-type call."
    )

    # MANDATORY. Each is a condition under which the z means what it says.
    assumptions: tuple[str, ...] = (
        "the universe is the analysed panel, not the transcriptome; the competitive "
        "background is the other panel genes, so enrichment is relative to a set of "
        "genes chosen for a particular biological question",
        "panel composition is itself biased toward the question the panel was "
        "designed for, so cross-set comparison of z is comparison within that bias",
        "cluster labels are fixed and were produced upstream; the test conditions on "
        "them and cannot question them",
        "the per-gene coefficients entering the test come from the same spatial GLM "
        "as the marker workflow, so the two methods are not independent witnesses",
        "the consumed table may be a per-cluster top-N truncation of the scored "
        "collection; a set absent from a cluster's rows did not clear this unit's "
        "reporting bar, which is not the same as having been scored and reported",
        "set membership is the collection release's; genes the release does not list "
        "are outside the set whatever their biology",
        "power falls with the set's panel footprint, and a set below the floors is "
        "reported as untestable rather than as not enriched",
    )

    # Rivals are other PROGRAMS, and the space of them is the collection -- reference
    # data. The table cannot propose an alternative account of a cluster: it scores
    # each set independently and never ranks accounts against one another.
    rival_space: Literal["reference", "data", "none"] = "reference"

    # Looked up live (Bioconductor, 2026-11-02), not written from memory. The
    # competitive gene-set test ships in the same package as the spatial GLM; no
    # separate PMID is recorded in-repo for it.
    citation: str = (
        "Jin M, Phipson B. jazzPanda: finding spatially relevant marker genes in "
        "image based spatial transcriptomics data. Bioconductor R package. "
        "doi:10.18129/B9.bioc.jazzPanda "
        "(https://bioconductor.org/packages/jazzPanda)"
    )

    # ----------------------------------------------------------------------- #
    def detect_unit(self, table: Any) -> UnitSpec:
        """Read the unit kind and ids FROM THE SCHEMA, never from a caller's say-so."""
        cols = _table.columns(
            table, "the jazzPanda gene-set result is a table with columns "
                   f"{list(REQUIRED_COLUMNS)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not a jazzPanda gene-set result -- missing column(s) "
                f"{missing}; saw {list(cols)}"
            )
        seen = {str(r[UNIT_COLUMN]).strip() for r in _table.rows(table)}
        return UnitSpec(
            unit_kind=UNIT_KIND,
            unit_column=UNIT_COLUMN,
            # Sorted, not first-seen: the unit list is hashed into the record, and
            # row order in a CSV is not a fact about the study.
            units=tuple(sorted(u for u in seen if u)),
            because=(
                f"columns {list(REQUIRED_COLUMNS)} present; units read off "
                f"{UNIT_COLUMN!r}, one per cluster the collection was scored against",
            ),
        )

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> Traced[dict[str, NormalisedEvidence]]:
        """Precomputed gene-set result in, normalised evidence out, keyed by unit.

        Reads, never recomputes. Gene frames and standing conditions come from the
        adapter; the GENE_SET frame is the method's (:meth:`scored_frame`).
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        spec = self.detect_unit(table)
        rows = _table.rows(table)

        world = MeasuredWorld(
            tuple(adapter.frames(store)) + (self.scored_frame(table),))
        conditions = adapter.standing_conditions(store)

        out: dict[str, NormalisedEvidence] = {}
        for uid in spec.units:
            mine = tuple(r for r in rows if str(r[UNIT_COLUMN]).strip() == uid)
            readings = tuple(self._reading(r, world) for r in mine)
            reported = Frame(
                EntityKind.GENE_SET, f"reported:{self.name}:{uid}",
                frozenset(
                    str(r["gene_set"]).strip() for r in mine
                    if (q := _to_float(r.get("p_adj_bh"))) is not None
                    and q <= Q_REPORTED
                ),
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

    def scored_frame(self, table: Any) -> Frame:
        """The GENE_SET capability frame: every set this result scored, anywhere.

        Public: only the method can declare it, and without it a caller cannot tell
        TESTED_ABSENT (scored elsewhere, not reported here) from NEVER_MEASURED.
        ``load`` uses this, so there is one spelling of the frame name.
        """
        scored = frozenset(str(r["gene_set"]).strip() for r in _table.rows(table))
        return Frame(EntityKind.GENE_SET,
                     f"collection:{_collection(scored)}", scored)

    def _reading(self, row: dict[str, Any], world: MeasuredWorld) -> Reading:
        """One result row, as the method claims it -- before the frame rules.

        Both K-of-N numbers travel as ``Coverage``, which the lattice demotes on,
        rather than as corroborating statistics.
        """
        tested = _split(row.get("genes_tested"))
        selected = _split(row.get("genes_selected"))
        q = _to_float(row.get("p_adj_bh"))

        coverage = (
            Coverage("panel_hits", _in_frame(tested, world),
                     int(float(row["geneset_size"])), MIN_PANEL_HITS),
            Coverage("leading_edge", len(selected), len(tested), MIN_LEADING_EDGE),
        )
        if q is None:
            # The test returned no q for this set. Not a measured absence: a set we
            # could not look at, which is what UNDERPOWERED means.
            coverage += (Coverage("q_reported", 0, 1, 1),)

        corroborating = tuple(
            (k, v) for k, v in (("p_adj_bh", q),
                                ("gc_corr", _to_float(row.get("gc_corr"))))
            if v is not None
        )
        return Reading(
            subject=subject("gene_set",
                            set=gene_set(str(row["gene_set"]).strip()),
                            member=[gene(g) for g in tested]),
            statistic=float(row["test_statistic"]),
            statistic_name=self.statistic_name,
            coverage=coverage,
            corroborating=corroborating,
        )

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        """What this method's silence about a set MEANS, in its own terms.

        The four states are generic; what a tested absence *argues* is not. For a
        competitive over-representation test it argues "the set's panel genes are
        not over-represented among this cluster's spatial signal" -- which is a
        statement about the panel footprint and never about the pathway.
        """
        who = subject.display()
        if measured.state is MeasuredState.NEVER_MEASURED:
            return (f"{who} was not scored in {measured.frame}; the collection this "
                    "result was run against does not contain it, so its absence "
                    "argues nothing")
        if measured.state is MeasuredState.UNDERPOWERED:
            k = ", ".join(str(c) for c in measured.coverage if not c.ok)
            return (f"{who} is in the collection but below a floor ({k}); too little "
                    "of it was measurable on this panel to ask the question")
        if measured.state is MeasuredState.TESTED_ABSENT:
            return (f"{who} was scored against this cluster and did not clear the "
                    f"reporting bar (q <= {fmt_num(Q_REPORTED)}) -- measured "
                    "absence, which argues against any account that needs this "
                    "program here, within the panel's coverage of the set")
        return (f"{who} was observed: its panel-resident genes are over-represented "
                "among the genes tracking this cluster")

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        """There is no band here, and asking for one is the error, not the answer.

        CHARACTERISATION has no label space, so a unit-level band is *undefined*.
        """
        raise NoBandForThisTaskKind(
            f"{self.name} answers a CHARACTERISATION ({self.workflow}): there is no "
            f"label space over {ev.unit}, so a unit-level band is undefined. The "
            "header for this workflow is per-item counts; see "
            "panoscope.workflows.enrichment.EnrichmentInterpretation.interpret."
        )

    def traversals(self) -> tuple[tuple[str, str], ...]:
        """Reference-graph edges this method's evidence licenses following.

        Empty: an over-representation z licenses nothing about ``marks`` (gene ->
        cell type), and set membership is not a registered relation.
        """
        return ()


#: The registered instance. One object, built once, attributes frozen.
MODULE = JazzPandaCGST()
