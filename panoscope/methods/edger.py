"""edgeR: a condition contrast on PSEUDOBULK counts, with the replicates modelled.

The second method in ``de-interpretation``; nothing in ``core/``, ``workflows/de.py``
or the composition root knows this file exists (ARCHITECTURE §4).

**What differs from ``methods/seurat_de.py`` is the DECLARATION.** edgeR fits a
negative binomial GLM over samples nested in conditions, so its p-value is a
condition-level statement given enough replicates; that is declared in
:attr:`assumptions`. It does not make §9.7's check 2 settle:
``workflows/de.replicate_confound`` refuses on every branch, since a ``topTags`` table
has one number per gene per unit.

This module reads a table and declares what its numbers mean; the checks, the
lattice and the join live elsewhere.

**Band edges** equal ``seurat_de.BAND_EDGES`` (both are log2 fold changes) and are
**not calibrated against an edgeR calibration set**.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
from panoscope.methods import _table
from panoscope.types.contracts import PlatformAdapter, UnitSpec, UnrecognisedResult
from panoscope.types.evidence import Direction, NormalisedEvidence
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.subject import EntityKind, Subject, gene, subject
from panoscope.types.subject import unit as unit_entity
from panoscope.types.trace import Trace, Traced, fmt_num, ref
from panoscope.workflows.de import REPLICATE_COVERAGE, REPLICATE_FLOOR, WORKFLOW_ID

# --------------------------------------------------------------------------- #
# The schema this method can read: ``topTags()`` output, one block per cluster,
# concatenated with the cluster it was fitted for. ``logCPM`` and ``F`` are
# corroborators rather than requirements -- an exactTest table carries ``logCPM``
# but no ``F``, and refusing it for that would refuse a real edgeR result over
# which test was run.
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: tuple[str, ...] = ("gene", "cluster", "logFC", "FDR")

#: Carried as declared corroborators, never as second answers. ``logCPM`` is the
#: average abundance the fold change was estimated at -- a low-abundance gene with
#: a large logFC is the classic unstable call -- and ``F``/``LR``/``PValue`` are the
#: test statistic and its unadjusted p. None of them is the effect.
CORROBORATING_COLUMNS: tuple[str, ...] = ("logCPM", "F", "LR", "PValue")

UNIT_KIND: str = "cluster"

# --------------------------------------------------------------------------- #
# This method's OWN ladder, on its OWN statistic. See the module docstring: the
# edges are fold-change arithmetic, not a calibration, and they are declared here
# rather than imported from ``seurat_de`` so that tuning one cannot move the other.
# Descending, floor last, as every other method's are.
# --------------------------------------------------------------------------- #
BAND_EDGES: tuple[tuple[str, float], ...] = (
    ("Very High", 2.0),      # 4x
    ("High", 1.0),           # 2x
    ("Medium-High", 0.585),  # 1.5x
    ("Medium", 0.263),       # 1.2x
    ("Low", 0.0),
)


@dataclass(frozen=True, slots=True)
class EdgeR:
    """The declaration. What edgeR's paper says it measures, and what it does not."""

    name: str = "edger"
    #: How the package names ITSELF, for anywhere a person reads it.
    display_name: str = "edgeR"
    workflow: str = WORKFLOW_ID
    statistic_name: str = "logFC"

    measures: str = (
        "the log2 fold change in a gene's PSEUDOBULK abundance between two "
        "conditions within one cluster, fitted by a negative binomial GLM over a "
        "design in which biological samples are nested in conditions, with the "
        "dispersion moderated across genes by empirical Bayes"
    )

    # MANDATORY. What the paper does not claim.
    does_not_measure: str = (
        "anything spatial -- counts are summed over a sample's cells before the "
        "model is fitted, so where a transcript sat is discarded before edgeR sees "
        "it; and anything about a single cell, since within-sample heterogeneity is "
        "summed away. It does not test whether the clustering is right, and it says "
        "nothing about a gene that is not on the panel."
    )

    # MANDATORY. Each is a condition under which the fold change means what it says.
    assumptions: tuple[str, ...] = (
        "cluster labels are fixed and were produced upstream; the model conditions "
        "on them and cannot question them",
        "counts were SUMMED per sample per cluster before fitting, so the unit of "
        "replication is the sample and not the cell -- this is what makes the "
        "p-value a condition-level statement, and it is also why too few samples "
        "makes it nothing at all",
        "the design matrix is correct: samples are assigned to the conditions they "
        "actually came from, which no result table can confirm",
        "dispersion is estimated across genes and moderated toward a common trend, "
        "so a gene whose variance is genuinely unlike the rest is shrunk toward a "
        "value that does not describe it",
        "library sizes were normalised (TMM or equivalent) before fitting; an "
        "un-normalised pseudobulk profile makes composition look like expression",
        "segmentation assigned transcripts to the right cells; a mis-segmented cell "
        "carries its neighbour's transcripts into this cluster's pseudobulk sum",
        "genes are filtered by expression before testing, so a gene absent from the "
        "table may have been filtered rather than tested",
        "only panel genes can ever appear; absence of an off-panel gene is not a "
        "measurement",
    )

    #: Whether a panel gene missing from the table was tested. edgeR's topTags
    #: reference: n is the "maximum number of genes/tags to return" (default 10) and
    #: p.value keeps "Only tags with adjusted p-values equal or lower than specified";
    #: genes removed by expression filtering before fitting were never tested.
    tests_whole_frame: bool = False
    tested_set: str = (
        "genes filtered by expression before fitting were never tested, and topTags "
        "returns at most n genes under its p.value cutoff, so a panel gene missing "
        "here cannot be read as tested and unchanged"
    )

    # ARCHITECTURE §9.6: for a COMPARISON the rival is a CONFOUND, and the space of
    # confounds comes from the DATA and the design, never from a reference list.
    rival_space: Literal["reference", "data", "none"] = "data"

    # Looked up live 2026-11-03 against the authors' institutional record
    # (publications.wehi.edu.au/Publications/10970) and the package's own
    # Bioconductor citation block. The DOI below appears verbatim on the first.
    # NO PMID is recorded, because neither page displayed one and a PMID written
    # from memory is the worst failure this project can produce.
    citation: str = (
        "Robinson MD, McCarthy DJ, Smyth GK. edgeR: a Bioconductor package for "
        "differential expression analysis of digital gene expression data. "
        "Bioinformatics 26(1):139-140 (2010). doi:10.1093/bioinformatics/btp616 "
        "(package: doi:10.18129/B9.bioc.edgeR)"
    )

    # ----------------------------------------------------------------------- #
    def detect_unit(self, table: Any) -> UnitSpec:
        """Read the unit kind and ids FROM THE SCHEMA (§9.2), never from a caller.

        One ``topTags`` block per cluster, concatenated with its ``cluster``. A table
        with no ``cluster`` column (a whole-object result) is refused.
        """
        cols = _table.columns(
            table, "edgeR's topTags output is a table with columns "
                   f"{list(REQUIRED_COLUMNS)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not an edgeR topTags table keyed on a cluster -- "
                f"missing column(s) {missing}; saw {list(cols)}. A contrast fitted "
                "on something other than a cluster asks a different question and "
                "must register its own method rather than be read by this one."
            )
        seen = {str(r["cluster"]) for r in _table.rows(table)}
        return UnitSpec(
            unit_kind=UNIT_KIND,
            unit_column="cluster",
            # Sorted, not first-seen: the unit list is hashed into the record.
            units=tuple(sorted(seen)),
            because=(
                f"columns {list(REQUIRED_COLUMNS)} present; units read off "
                "'cluster', so each block is a condition contrast fitted within "
                "one cluster",
            ),
        )

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> Traced[dict[str, NormalisedEvidence]]:
        """Precomputed ``topTags`` output in, normalised evidence out, by unit.

        Reads, never recomputes. The capability frames and the **sample count**
        come from the adapter; the count lands as a unit-level :class:`Coverage`
        against :data:`~panoscope.workflows.de.REPLICATE_FLOOR`, under the same name
        ``seurat_de`` writes.
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        spec = self.detect_unit(table)
        world = MeasuredWorld(adapter.frames(store))
        conditions = adapter.standing_conditions(store)
        samples = tuple(adapter.samples(store))
        rows = _table.rows(table)

        replicates = Coverage(REPLICATE_COVERAGE, k=len(samples), n=len(samples),
                              floor=REPLICATE_FLOOR)

        out: dict[str, NormalisedEvidence] = {}
        for uid in spec.units:
            mine = tuple(r for r in rows if str(r["cluster"]) == uid)
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
                readings=tuple(self._reading(r) for r in mine),
                world=world,
                reported=(reported,),
                conditions=conditions,
                unit_coverage=(replicates,),
            ))
        return tr.done(out)

    def _reading(self, row: dict[str, Any]) -> Reading:
        """One ``topTags`` row, as edgeR reports it -- before the frame rules.

        ``direction`` is a field, not folded into the magnitude (§9.1). ``flags`` is
        empty: the flag registry holds only spatial ``specificity:*`` ids.
        """
        fc = float(row["logFC"])
        return Reading(
            subject=subject("gene", g=gene(str(row["gene"]))),
            statistic=fc,
            statistic_name=self.statistic_name,
            direction=Direction.UP if fc >= 0 else Direction.DOWN,
            corroborating=tuple((c, _table.as_float(row, c))
                                for c in CORROBORATING_COLUMNS if c in row),
        )

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        """What this method's silence about a subject MEANS, in its own terms.

        Narrower than a spatial or Seurat absence: this gene's summed abundance in
        this cluster did not move **between the conditions compared**.
        """
        who = subject.display()
        if measured.state is MeasuredState.NEVER_MEASURED:
            return (f"{who} is not on the panel ({measured.frame}); edgeR could not "
                    "have tested it here, so its absence argues nothing")
        if measured.state is MeasuredState.UNDERPOWERED:
            return (f"{who} is on the panel but below the floor in "
                    f"{measured.frame}; there was too little to test")
        if measured.state is MeasuredState.TESTED_ABSENT:
            return (f"{who} was on the panel and tested, and its pseudobulk "
                    "abundance in this cluster did not change between the "
                    "conditions compared -- measured absence, which argues against "
                    "any account needing it to move in THIS contrast, and says "
                    "nothing about any other contrast or about where it sits")
        return (f"{who} was observed: its pseudobulk abundance in this cluster "
                "differs between the conditions compared, with no claim about "
                "where those cells are")

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        """Per-item EFFECT SIZE off this method's own statistic. **Never a band.**

        COMPARISON has no label space. What a magnitude of *its* statistic is worth
        is the method's to say (§9.4). ``abs()``: direction rides on the
        ``EvidenceItem``.
        """
        t = Trace()
        lead = next((i for i in ev.items if i.measured.carries_number), None)
        floor = BAND_EDGES[-1][0]
        if lead is None:
            t.note("band:no_effect_reported", floor,
                   (f"{len(ev.items)} item(s), none carrying a "
                    f"{self.statistic_name}",),
                   (ref(ev.unit_subject),))
            return t.done(floor)

        c = abs(float(lead.statistic or 0.0))
        label = next((lab for lab, edge in BAND_EDGES if c >= edge), floor)
        t.note("band:from_effect_size", label,
               (f"{lead.statistic_name}={fmt_num(lead.statistic)} on "
                f"{lead.subject.display()}, at or above {fmt_num(_edge(label))}",),
               (ref(lead.subject, lead.statistic_name, lead.statistic),))
        return t.done(label)

    def traversals(self) -> tuple[tuple[str, str], ...]:
        """Reference-graph edges this method's evidence licenses following.

        None, by declaration: rivals come from the data (``rival_space="data"``).
        """
        return ()


def _edge(label: str) -> float:
    return next(e for lab, e in BAND_EDGES if lab == label)


#: The registered instance. One object, built once, attributes frozen.
MODULE = EdgeR()
