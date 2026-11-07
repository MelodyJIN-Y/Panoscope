"""Seurat's cluster-vs-rest differential expression, declared: what it measures, and
what it does not.

Almost all *declaration*: **this method's ``does_not_measure`` is jazzPanda's
``measures``, and the reverse**, so a reader can tell which question produced a number.

**The failure this prevents:** an ``avg_log2FC`` read as spatial organisation.
Seurat compares per-cell expression and never looks at a coordinate.

**COMPARISON has no band** (§9.1). :meth:`SeuratDE.rubric` exists because
``MethodModule`` requires it and §9.4's rubric-divergence invariant compares this
ladder against jazzPanda's: :data:`BAND_EDGES` is a per-item **effect-size** ladder,
never put on a unit. The edges are the conventional log2 fold-change breaks (4x, 2x,
1.5x, 1.2x), declared as conventions, not tuned: there is no calibration set for
this statistic.

**``load`` reads; it never recomputes.** ``FindAllMarkers`` output is a consumed input.
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
# The schema this method can read. A ``FindAllMarkers`` frame, as Seurat writes
# it. Missing any of these and the table is not one -- and guessing routes the
# rows to an interpretive question they do not answer.
# --------------------------------------------------------------------------- #
REQUIRED_COLUMNS: tuple[str, ...] = ("gene", "cluster", "avg_log2FC", "p_val_adj")

#: Carried as declared corroborators, never as second answers. ``pct.1``/``pct.2``
#: are detection rates -- the fraction of cells in each group with a non-zero count
#: -- and they are the one lever in the table that speaks to composition at all.
CORROBORATING_COLUMNS: tuple[str, ...] = ("p_val", "pct.1", "pct.2")

UNIT_KIND: str = "cluster"

# --------------------------------------------------------------------------- #
# This method's OWN ladder, on its OWN statistic. Declared here, never shared.
# A log2FC of 3.0 is an eight-fold change and a large effect; a glm_coef of 3.0
# is a middling spatial association. One shared ladder in core/ would force the
# two to agree, which is the method-blindness ARCHITECTURE §9.4 forbids.
# Descending, floor last, as jazzPanda's are.
# --------------------------------------------------------------------------- #
BAND_EDGES: tuple[tuple[str, float], ...] = (
    ("Very High", 2.0),      # 4x
    ("High", 1.0),           # 2x
    ("Medium-High", 0.585),  # 1.5x
    ("Medium", 0.263),       # 1.2x
    ("Low", 0.0),
)


@dataclass(frozen=True, slots=True)
class SeuratDE:
    """The declaration. Every field is a sentence from the method's docs, not ours.

    Frozen, like ``JazzPandaSpatialGLM``, so the registry holds one object whose
    attributes cannot be rebound after registration.
    """

    name: str = "seurat_de"
    #: How the package names ITSELF, for anywhere a person reads it.
    display_name: str = "Seurat"
    workflow: str = WORKFLOW_ID
    statistic_name: str = "avg_log2FC"

    measures: str = (
        "the log2 ratio of a gene's mean expression in one cluster's cells against "
        "all other cells in the object, with a per-gene two-sample test (Wilcoxon "
        "rank sum by default) over cells, reported with the fraction of cells in "
        "each group in which the gene was detected (pct.1, pct.2)."
    )

    # MANDATORY. The half a reader cannot reconstruct from the number itself, and
    # the exact mirror of jazzPanda's: this one is blind to everything spatial.
    does_not_measure: str = (
        "nothing spatial at all: no coordinate, no tile and no neighbourhood enters "
        "this statistic, so a gene uniformly raised across every cell of a dispersed "
        "cluster scores identically to one whose transcripts concentrate in a single "
        "region. It also does not measure cell-type identity, abundance or "
        "composition -- a gene can read as up because the cells expressing it became "
        "more numerous, not because expression per cell rose -- nor causation or "
        "regulation. The cells are not independent samples, so its p-value is not a "
        "condition-level p-value, and it says nothing about a gene that is not on "
        "the panel."
    )

    # MANDATORY. Each is a condition under which the fold change means what it says.
    assumptions: tuple[str, ...] = (
        "cluster labels are fixed and were produced upstream; the test conditions on "
        "them and cannot question them",
        "counts were normalised and log-transformed before the ratio was taken, so "
        "the statistic is a difference of log means and not a ratio of raw counts",
        "segmentation assigned transcripts to the right cells; a mis-segmented cell "
        "carries its neighbour's transcripts into this cluster's mean",
        "cells are treated as independent observations, which they are not -- they "
        "are nested in samples, so this is a cell-level test and its p-value does "
        "not generalise to a condition without biological replicates",
        "the comparison group is 'all other cells in this object', so the statistic "
        "moves when the composition of the rest moves, with no change in this "
        "cluster at all",
        "genes are pre-filtered by detection rate and minimum effect size before "
        "testing, so a gene absent from the table may have been filtered rather "
        "than tested",
        "only panel genes can ever appear; absence of an off-panel gene is not a "
        "measurement",
    )

    # ARCHITECTURE §9.6: for a COMPARISON the rival is a CONFOUND, and the space of
    # confounds comes from the DATA and the design -- per-sample consistency, effect
    # within cell type, driving-cell concentration -- never from a reference list.
    # The contract allows a module to say this, and this module says it.
    rival_space: Literal["reference", "data", "none"] = "data"

    # Looked up live (CRAN's Seurat citation page, 2026-11-02), not written from
    # memory. The v5 methods paper is the citation that resolves today; the test
    # itself is FindMarkers/FindAllMarkers, documented in the package reference.
    citation: str = (
        "Hao Y, Stuart T, Kowalski MH, et al. Dictionary learning for integrative, "
        "multimodal and scalable single-cell analysis. Nature Biotechnology (2023). "
        "doi:10.1038/s41587-023-01767-y "
        "(citation list: https://cran.r-project.org/web/packages/Seurat/citation.html)"
    )

    # ----------------------------------------------------------------------- #
    def detect_unit(self, table: Any) -> UnitSpec:
        """Read the unit kind and ids FROM THE SCHEMA (§9.2), never from a caller.

        DE can be keyed on a cluster, on a cell type, or on a contrast between
        conditions, and only the table says which. This method reads the
        cluster-vs-rest shape -- a ``cluster`` column -- and refuses anything else,
        rather than guessing that a ``condition`` column means the same thing.
        """
        cols = _table.columns(
            table, "Seurat's FindAllMarkers output is a table with columns "
                   f"{list(REQUIRED_COLUMNS)}")
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not a Seurat FindAllMarkers table -- missing "
                f"column(s) {missing}; saw {list(cols)}. A DE table keyed on "
                "something other than a cluster asks a different question and must "
                "register its own method rather than be read by this one."
            )
        seen = {str(r["cluster"]) for r in _table.rows(table)}
        return UnitSpec(
            unit_kind=UNIT_KIND,
            unit_column="cluster",
            # Sorted, not first-seen: the unit list is hashed into the record.
            units=tuple(sorted(seen)),
            because=(
                f"columns {list(REQUIRED_COLUMNS)} present; units read off "
                "'cluster', so this is a cluster-vs-rest contrast and not a "
                "condition contrast",
            ),
        )

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> Traced[dict[str, NormalisedEvidence]]:
        """Precomputed ``FindAllMarkers`` output in, normalised evidence out, by unit.

        Reads, never recomputes. The capability frames and the **sample count** come
        from the adapter; the count lands as a unit-level :class:`Coverage` against
        :data:`REPLICATE_FLOOR`, which makes too few replicates an UNDERPOWERED
        **unit** (§9.5) for the workflow's replicate check.
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
        """One FindAllMarkers row, as Seurat claims it -- before the frame rules.

        ``direction`` is a field, not folded into the magnitude (§9.1). ``flags`` is
        empty: the flag registry holds only spatial ``specificity:*`` ids.
        """
        fc = float(row["avg_log2FC"])
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

        Here a tested absence argues "its mean expression in this cluster's cells is
        not different enough from the rest" -- magnitude in cells, not place.
        """
        who = subject.display()
        if measured.state is MeasuredState.NEVER_MEASURED:
            return (f"{who} is not on the panel ({measured.frame}); Seurat could not "
                    "have tested it here, so its absence argues nothing")
        if measured.state is MeasuredState.UNDERPOWERED:
            return (f"{who} is on the panel but below the floor in {measured.frame}; "
                    "there was too little to test")
        if measured.state is MeasuredState.TESTED_ABSENT:
            return (f"{who} was on the panel and tested, and its mean expression in "
                    "these cells was not reported as different from the rest -- "
                    "measured absence, which argues against any account that needs "
                    "it raised here, and says nothing about where it sits")
        return (f"{who} was observed: its mean expression differs from the rest of "
                "the cells, with no claim about where those cells are")

    def rubric(self, ev: NormalisedEvidence) -> Traced[str]:
        """Per-item EFFECT SIZE off this method's own statistic. **Never a band.**

        COMPARISON has no label space; ``workflows/de.py`` never calls this for a
        unit. What a magnitude of *its* statistic is worth is the method's to say
        (§9.4). ``abs()``: direction rides on the ``EvidenceItem``.
        """
        t = Trace()
        lead = next((i for i in ev.items if i.measured.carries_number), None)
        floor = BAND_EDGES[-1][0]
        if lead is None:
            t.note("band:no_effect_reported", floor,
                   (f"{len(ev.items)} item(s), none carrying an "
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
#: No study here carries a FindAllMarkers table, so it is registered and, by
#: schema, never selected today.
MODULE = SeuratDE()
