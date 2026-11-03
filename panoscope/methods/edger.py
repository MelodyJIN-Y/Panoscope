"""edgeR: a condition contrast on PSEUDOBULK counts, with the replicates modelled.

The second method registering into ``de-interpretation``, and it is here to make
ARCHITECTURE §4's claim checkable: *the core does not change when a member is
added*. Nothing in ``core/``, in ``workflows/de.py`` or in the composition root
knows this file exists. The workflow reads a unit-level ``Coverage`` and never asks
which package wrote it.

**What differs from ``methods/seurat_de.py`` is the DECLARATION, not the plumbing,
and it does not make the replicate check settle.** Seurat's test treats cells as
independent observations, which they are not, so its p-value is a cell-level
statement that does not generalise to a condition. edgeR sums counts to one profile
per sample per cluster and fits a negative binomial GLM over a design in which
samples are nested in conditions, so its p-value *is* a condition-level statement
when the design carries enough biological replicates. That difference is real and
is declared in :attr:`assumptions`, where a biologist reads it.

It changes nothing about ARCHITECTURE §9.7's check 2, and claiming otherwise would
be the overclaim this project exists to refuse. ``workflows/de.replicate_confound``
returns *cannot be settled* on every branch -- including when the replicate floor is
cleared -- because a ``topTags`` table reports one number per gene for the whole
unit, so which replicate drives it is not in the file whoever fitted it. Both
methods are refused for the same reason, and the refusal names the same missing
thing: the contrast re-run within each sample.

**What this module does NOT own.** Not the four checks (``workflows/de.py``), not
the lattice (``types/measured.py``), not the join (``core/reconcile.py``). It reads
a table and declares what the table's numbers mean. If it ever computes a
statistic, the layering has inverted.

**On the band edges.** They are the same numbers ``seurat_de.BAND_EDGES`` carries,
and that is not a shared ladder smuggled in: both statistics are log2 fold changes,
so "1.0 is a doubling" is arithmetic and not a tuned threshold. They are declared
here, in this module, because §9.4 says a rubric belongs to the method -- and they
are **not calibrated against an edgeR calibration set**, which is stated rather than
implied. The demo's bands were tuned against jazzPanda's ``glm_coef``; nothing in
this repository has tuned a fold-change ladder, and an uncalibrated constant that
says so is honest in a way that a borrowed one is not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
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


def _columns(table: Any) -> tuple[str, ...]:
    """Column names off anything table-shaped, without importing a table library.

    The same two shapes ``seurat_de._columns`` reads, and the same refusal for the
    rest: duck-typing further is how a wrong answer gets produced confidently.
    """
    cols = getattr(table, "columns", None)
    if cols is not None:
        return tuple(str(c) for c in cols)
    if isinstance(table, (list, tuple)) and table and isinstance(table[0], dict):
        return tuple(str(k) for k in table[0])
    raise UnrecognisedResult(
        f"cannot read column names off a {type(table).__name__}; edgeR's topTags "
        f"output is a table with columns {list(REQUIRED_COLUMNS)}"
    )


def _rows(table: Any) -> tuple[dict[str, Any], ...]:
    """Rows as plain dicts, in the table's own order. Never re-sorted.

    ``topTags`` orders by adjusted p-value, and that order is hashed into the
    record downstream. Row order in a CSV is not a fact about the study, but
    re-sorting it here would be a second, invented one.
    """
    to_dict = getattr(table, "to_dict", None)
    if to_dict is not None:
        return tuple(to_dict(orient="records"))
    return tuple(dict(r) for r in table)


def _f(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


@dataclass(frozen=True, slots=True)
class EdgeR:
    """The declaration. What edgeR's paper says it measures, and what it does not."""

    name: str = "edger"
    #: How the package names ITSELF, for anywhere a person reads it.
    #: The registry id is an identifier, and the study chip was printing it
    #: verbatim ("jazzpanda_spatial_glm markers").
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

        edgeR is run per cluster here, one ``topTags`` block each, concatenated with
        the cluster it was fitted for. A table with no ``cluster`` column is an
        edgeR result for the whole object -- a different unit, and a different
        question -- and is refused rather than read as though every gene belonged to
        one unnamed unit.
        """
        cols = _columns(table)
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not an edgeR topTags table keyed on a cluster -- "
                f"missing column(s) {missing}; saw {list(cols)}. A contrast fitted "
                "on something other than a cluster asks a different question and "
                "must register its own method rather than be read by this one."
            )
        seen = {str(r["cluster"]) for r in _rows(table)}
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

        Reads, never recomputes. Two things come from the adapter and nowhere else:
        the capability frames, and the **sample count**, which ``Conditions``
        deliberately cannot express.

        The sample count lands as a unit-level :class:`Coverage` against
        :data:`~panoscope.workflows.de.REPLICATE_FLOOR` -- the same coverage name
        ``seurat_de`` writes, so ``workflows/de.replicate_confound`` reads one field
        and never branches on which package produced it. What differs between the
        two methods is not the number but what the design does with it, and that
        difference is declared in :attr:`assumptions` rather than coded here.
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        spec = self.detect_unit(table)
        world = MeasuredWorld(adapter.frames(store))
        conditions = adapter.standing_conditions(store)
        samples = tuple(adapter.samples(store))
        rows = _rows(table)

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

        ``direction`` is carried as a field rather than smuggled into the magnitude:
        a comparison's answer is "a ranked annotated set, **with direction**"
        (§9.1), and a down-regulated gene is a result, not a weak one.

        ``flags`` is empty and must be. The flag registry in ``types/evidence.py``
        holds only ``specificity:*`` ids, which belong to a spatial statistic; the
        confounds a pseudobulk contrast is subject to are unit-level and live on the
        workflow's check results, where they can name what is missing.
        """
        fc = float(row["logFC"])
        return Reading(
            subject=subject("gene", g=gene(str(row["gene"]))),
            statistic=fc,
            statistic_name=self.statistic_name,
            direction=Direction.UP if fc >= 0 else Direction.DOWN,
            corroborating=tuple((c, _f(row, c)) for c in CORROBORATING_COLUMNS
                                if c in row),
        )

    def absence_reads_as(self, subject: Subject, measured: Measured) -> str:
        """What this method's silence about a subject MEANS, in its own terms.

        A spatial method's tested absence argues "it localises somewhere else";
        Seurat's argues "its mean in these cells is not different enough from the
        rest". edgeR's argues something narrower than either: this gene's summed
        abundance in this cluster did not move **between the conditions compared**.
        It is silent about every other contrast, and about place entirely.
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

        The workflow this method registers into is a ``COMPARISON`` and has no label
        space, so there is nothing for a unit-level band to be confidence *in*, and
        ``Finding`` has no field to put one in. What this returns is the one thing
        only the method can know: what a given magnitude of *its* statistic is
        worth -- which is what §9.4's divergence invariant compares against
        jazzPanda's, and the reason a ``glm_coef`` of 3.0 and a ``logFC`` of 3.0
        must not read the same.

        ``abs()`` matches every other method's: a strongly negative fold change is a
        strong measurement, and direction is carried on the ``EvidenceItem`` rather
        than read as weakness.
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

        None, deliberately, and that is a declaration rather than a gap. This
        method's rivals come from the data (``rival_space="data"``): a fold change
        between two groups of samples licenses no traversal of a curated graph,
        because the question it answers -- does this result make sense? -- is
        settled by the design and the measurement, and a reference edge can neither
        supply a value nor count as agreement.
        """
        return ()


def _edge(label: str) -> float:
    return next(e for lab, e in BAND_EDGES if lab == label)


#: The registered instance. One object, built once, attributes frozen.
MODULE = EdgeR()
