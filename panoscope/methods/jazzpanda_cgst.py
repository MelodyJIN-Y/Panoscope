"""jazzPanda's competitive gene-set test, declared: what it measures, and what it
does not -- which on a targeted panel is almost everything a reader will assume.

The second method module, and the one that proves the registry claim is structural
rather than a story told once. It registers into a *different* workflow from
``jazzpanda.py`` with no change to ``types/`` or ``core/``, and the difference that
matters is not the statistic: it is the task kind. Markers ask *what IS this unit*,
which has a label space and therefore a band. This asks *what is RUNNING here*,
which has none -- so this module declares no band ladder at all, and
:meth:`JazzPandaCGST.rubric` raises rather than returning one. A ``BAND_EDGES`` here
would be the same category error ``UISpec`` already refuses to render.

**The failure this module exists to prevent.** "c2 is enriched for interferon
gamma response." The set has 200 genes; ten of them are on this panel; the test
compared those ten against the other panel genes. The sentence a reader forms from
a z of 11.2 is a genome-wide claim, and the measurement is an over-representation
*within the measured set*. ``does_not_measure`` is where that gap is written down,
and :class:`~panoscope.types.measured.Coverage` is where it is enforced: K of N
against a floor, and below the floor the set is UNDERPOWERED, which means its
absence argues nothing and its statistic is withheld rather than printed small.

**Why the panel hits are counted against the frame and not read off the table.**
The result carries ``n_overlap``, the method's own count of set genes on the panel.
This module instead counts how many of ``genes_tested`` every GENE frame in the
study's world admits -- which on the demo study is the panel frame *and* the
detectability frame, so the 49 of 280 genes that yield no usable spatial signal do
not count toward a set's footprint. Reading ``n_overlap`` would take the method's
word for its own capability; the whole point of the audit is that capability is
declared by the apparatus, before the result is read.

**Two bars, two owners.** ``Q_REPORTED`` is this method's reporting bar on its own
statistic -- the analog of jazzPanda's ``NoSig`` sentinel -- and a scored set that
does not clear it is TESTED_ABSENT: in frame, asked, and not reported here. That is
the productive half of the lattice and it belongs to the method, because only the
method knows what its own q means. The *enriched versus suggestive* split is a
question of how cautious this lab has decided to be and lives in
``workflows/enrichment.py``, for exactly the reason jazzPanda's promote/demote
ladder lives in ``core/verdict.py`` and not beside its ``BAND_EDGES``.

**``load`` reads; it never recomputes.** The enrichment result is a consumed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
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

#: The collection the sets are drawn from. Carried so the frame says WHICH universe
#: a set was scored in; it is not a count and nothing is derived from it.
COLLECTION: str = "MSigDB_Hallmark"

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


def _columns(table: Any) -> tuple[str, ...]:
    """Column names off anything table-shaped, without importing a table library.

    Deliberately a second copy of ``jazzpanda.py``'s reader rather than an import of
    it. A method module is an independent declaration -- the registries have no
    shared ancestor on purpose -- and importing another method's private schema
    reader would mean a change to jazzPanda's table handling silently changing what
    this method reads.
    """
    cols = getattr(table, "columns", None)
    if cols is not None:
        return tuple(str(c) for c in cols)
    if isinstance(table, (list, tuple)) and table and isinstance(table[0], dict):
        return tuple(str(k) for k in table[0])
    raise UnrecognisedResult(
        f"cannot read column names off a {type(table).__name__}; the jazzPanda "
        f"gene-set result is a table with columns {list(REQUIRED_COLUMNS)}"
    )


def _rows(table: Any) -> tuple[dict[str, Any], ...]:
    """Rows as plain dicts, in the table's own order. Never re-sorted.

    The order is the method's (``test_statistic`` descending within a cluster) and
    the census header reads the first item as the top program; re-sorting here would
    have to place the sets carrying no statistic somewhere arbitrary.
    """
    to_dict = getattr(table, "to_dict", None)
    if to_dict is not None:
        return tuple(to_dict(orient="records"))
    return tuple(dict(r) for r in table)


def _in_frame(symbols: tuple[str, ...], world: MeasuredWorld) -> int:
    """How many of a set's genes EVERY gene frame admits.

    Every frame, not any: the panel says a gene could have been detected and the
    detectability frame says it produced usable signal, and a set whose footprint
    is three undetectable genes was never measurable however the panel was designed.
    No gene frame at all counts as nothing measurable, matching the lattice's
    reading of a missing frame -- an empty answer here would say the apparatus
    measured nothing, which is the same verdict with a false reason behind it.
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
        cols = _columns(table)
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not a jazzPanda gene-set result -- missing column(s) "
                f"{missing}; saw {list(cols)}"
            )
        seen = {str(r[UNIT_COLUMN]).strip() for r in _rows(table)}
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

        Reads, never recomputes. The gene frames and standing conditions come from
        the adapter; the GENE_SET frame is the method's, because which sets were
        scored at all is a fact about this result and about nothing else. Without it
        every set would resolve as NEVER_MEASURED -- there is one capability frame
        per entity kind, and nobody else can declare this one.
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        spec = self.detect_unit(table)
        rows = _rows(table)

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

        Public, and it has to be. Nobody else can declare this frame -- the adapter
        knows panels and annotations, not which collection a result was run against
        -- and a caller that cannot get it cannot tell "scored for another unit and
        not reported here" (TESTED_ABSENT, which argues) from "never scored at all"
        (NEVER_MEASURED, which argues nothing). That is exactly the collapse the four
        states exist to stop, and it would happen in the rival test, where it matters
        most. ``load`` uses this, so there is one spelling of the frame name.
        """
        return Frame(
            EntityKind.GENE_SET, f"collection:{COLLECTION}",
            frozenset(str(r["gene_set"]).strip() for r in _rows(table)),
        )

    def _reading(self, row: dict[str, Any], world: MeasuredWorld) -> Reading:
        """One result row, as the method claims it -- before the frame rules.

        Both K-of-N numbers travel as ``Coverage`` rather than as corroborating
        statistics. That is not tidiness: ``Coverage`` is what the lattice demotes
        on, so a set below a floor comes back UNDERPOWERED with its z withheld,
        whereas a panel-hit count carried as a statistic would sit in a column
        beside a z that the apparatus could not support.
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

        CHARACTERISATION has no label space, so there is no alternative for the
        remaining probability mass to sit on and a unit-level band is *undefined*
        rather than merely unhelpful. The Protocol requires the attribute; what it
        cannot require is a number that does not exist. Raising makes the category
        error loud at the one moment it is cheap, which is the same move
        ``UISpec.__post_init__`` makes for the header and ``Finding`` makes by
        having no field to fill.
        """
        raise NoBandForThisTaskKind(
            f"{self.name} answers a CHARACTERISATION ({self.workflow}): there is no "
            f"label space over {ev.unit}, so a unit-level band is undefined. The "
            "header for this workflow is per-item counts; see "
            "panoscope.workflows.enrichment.EnrichmentInterpretation.interpret."
        )

    def traversals(self) -> tuple[tuple[str, str], ...]:
        """Reference-graph edges this method's evidence licenses following.

        Empty, and deliberately so. ``reference/graph.py`` registers exactly one
        relation today -- ``marks`` (gene -> cell type) -- and an over-representation
        z licenses nothing about it: the set is not a marker of anything, and
        traversing to the cell type its genes mark would let the enrichment result
        argue about identity, which ``does_not_measure`` says it cannot. Set
        membership is not a registered relation; when it becomes one, it belongs
        here and nowhere else.
        """
        return ()


#: The registered instance. One object, built once, attributes frozen.
MODULE = JazzPandaCGST()
