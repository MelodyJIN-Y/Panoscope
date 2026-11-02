"""jazzPanda's spatial GLM, declared: what its paper says it measures, and does not.

This file is deliberately almost all *declaration*. Adding a method to Panoscope
means reading the paper and writing down the contract -- not writing interpretation
logic -- and this is the first module that has to prove that claim is real rather
than aspirational. The four fields that carry weight (``measures``,
``does_not_measure``, ``assumptions``, ``statistic_name``) are transcribed from
``panoscope/skills/marker-interpretation/SKILL.md``, which is itself transcribed from
the jazzPanda paper's Methods and the package docs. Nothing here paraphrases upward.

**The failure this module exists to prevent.** A ``glm_coef`` read as a fold change.
jazzPanda does not ask "is this gene higher on average in this cluster's cells?" --
it asks "do this gene's transcripts fall in the same places on the tissue as this
cluster's cells?" Those two questions disagree, loudly, on a gene that is uniformly
higher everywhere in a cluster but not spatially concentrated. A reader who has only
the number cannot tell which question produced it, so the number must arrive carrying
the question. ``does_not_measure`` is that carrying.

**Where the rubric stops.** This module declares :data:`BAND_EDGES` -- the mapping
from the magnitude of *its own* statistic to a band label -- and nothing else about
confidence. The promote, the three demotes, the fragile cap and the fragile floor all
live in ``core/verdict.py``, because they are the project's confidence *policy* and
apply to whatever statistic a method hands up. A method knows what its own
``glm_coef`` means; it does not know how cautious this lab has decided to be. The
split matters most for the second method: a mean-expression rubric must map its own
``log2FC`` through its own edges, and if the edges lived in ``core/`` the two methods
would be forced to agree, which is precisely the method-blindness ARCHITECTURE §9.4
calls the inversion of the value proposition.

The three specificity thresholds below are *also* the method's, for the same reason,
and they are the only place ``specificity:*`` flags are raised. ``core/verdict.py``
branches on the flag ids and never re-derives them from the raw correlations -- one
owner for one rule, so there is no second copy to drift.

**``load`` reads; it never recomputes.** The precomputed ``top_result`` table is a
consumed input. That is the whole basis of the audit: if this module ran the analysis
it would be grading its own homework, and the confidence assessment would stop being
an assessment of someone else's output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from panoscope.core.measure import Reading, measure_unit
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
# The method's own rubric: magnitude of ITS statistic -> band. Declared here,
# APPLIED in core/verdict.py. Edges are tuned against the calibration set
# (data/studies/*/interp/calibration.md), descending, floor last.
# --------------------------------------------------------------------------- #
BAND_EDGES: tuple[tuple[str, float], ...] = (
    ("Very High", 10.0),
    ("High", 5.0),
    ("Medium-High", 2.5),
    ("Medium", 1.0),
    ("Low", 0.0),
)

# The three specificity conditions, with the thresholds the SKILL rubric states.
# Raised ONCE, here, as registered flag ids; core/verdict.py branches on the ids.
LOW_PEARSON: float = 0.30       # weak spatial specificity on this gene
GG_NOT_UNIQUE: float = 0.98     # another gene shares this spatial pattern
EPS: float = 1e-9               # max_gc_corr > pearson must be a real difference


def _flags_for(pearson: float, max_gc_corr: float, max_gg_corr: float) -> tuple[str, ...]:
    """Registered ``specificity:*`` ids for one row. Order is code order, hashed.

    Ids, never prose: the interface used to decide whether to draw a badge by
    matching the English sentence "localizes better with another cluster", so a copy
    edit here silently stopped a badge rendering while the test, asserting the same
    literal, stayed green.
    """
    out: list[str] = []
    if max_gc_corr > pearson + EPS:
        out.append("specificity:localises_elsewhere")
    if max_gg_corr >= GG_NOT_UNIQUE:
        out.append("specificity:not_unique")
    if pearson < LOW_PEARSON:
        out.append("specificity:low")
    return tuple(out)


def _columns(table: Any) -> tuple[str, ...]:
    """Column names off anything table-shaped, without importing a table library.

    ``MethodModule.detect_unit`` takes ``Any`` on purpose -- a method registered out
    of tree may hand up a polars frame, a list of dicts or an arrow table -- so this
    reads the two shapes that actually occur and refuses the rest rather than
    duck-typing its way into a wrong answer.
    """
    cols = getattr(table, "columns", None)
    if cols is not None:
        return tuple(str(c) for c in cols)
    if isinstance(table, (list, tuple)) and table and isinstance(table[0], dict):
        return tuple(str(k) for k in table[0])
    raise UnrecognisedResult(
        f"cannot read column names off a {type(table).__name__}; jazzPanda's "
        f"top_result is a table with columns {list(REQUIRED_COLUMNS)}"
    )


def _rows(table: Any) -> tuple[dict[str, Any], ...]:
    """Rows as plain dicts, in the table's own order. Never re-sorted.

    The order is jazzPanda's (``glm_coef`` descending within a cluster) and it is
    load-bearing downstream: ``core/verdict.py`` takes the first eligible canonical
    marker as the driver and does not re-sort, because re-sorting on the statistic
    would have to place items with no statistic somewhere arbitrary.
    """
    to_dict = getattr(table, "to_dict", None)
    if to_dict is not None:
        return tuple(to_dict(orient="records"))
    return tuple(dict(r) for r in table)


@dataclass(frozen=True, slots=True)
class JazzPandaSpatialGLM:
    """The declaration. Every field is a sentence from the paper, not from us.

    A frozen dataclass rather than a module of constants so that the registry holds
    one object whose attributes cannot be rebound after registration: a declaration
    that a later import could edit is a declaration nothing downstream can trust.
    """

    name: str = "jazzpanda_spatial_glm"
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

        ``NoSig`` is excluded from the units on purpose: it is jazzPanda's sentinel
        for *tested here, assigned nowhere*, not a cluster. Admitting it would create
        a unit whose every item is a tested absence and whose band would then be
        computed as though something had been found.
        """
        cols = _columns(table)
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise UnrecognisedResult(
                f"{self.name}: not a jazzPanda top_result -- missing column(s) "
                f"{missing}; saw {list(cols)}"
            )
        seen = {str(r["top_cluster"]) for r in _rows(table)} - {NOSIG}
        return UnitSpec(
            unit_kind=UNIT_KIND,
            unit_column="top_cluster",
            # Sorted, not first-seen: the unit list is hashed into the record, and
            # row order in a CSV is not a fact about the study.
            units=tuple(sorted(seen)),
            because=(
                f"columns {list(REQUIRED_COLUMNS)} present; units read off "
                f"'top_cluster' with {NOSIG!r} excluded as jazzPanda's "
                "tested-but-unassigned sentinel",
            ),
        )

    def load(self, path: str, adapter: PlatformAdapter,
             store: Any) -> Traced[dict[str, NormalisedEvidence]]:
        """Precomputed ``top_result`` in, normalised evidence out, keyed by unit.

        Reads, never recomputes. The capability frames and the standing conditions
        come from the adapter -- this method has no idea what a Xenium panel is, and
        a method that derived one would be asserting a panel size rather than
        reading it, which is how a hardcoded 280 survives into a study with a
        different panel.

        Returns ``Traced``: loading DECIDES -- it drops rows, applies frames and
        withholds statistics the frame refused -- and a statistic silently dropped
        during load is indistinguishable from a method that reported nothing. The
        Protocol originally returned a bare dict, which discarded exactly those
        decisions at the registry boundary; it was widened for this reason.
        """
        import pandas as pd  # local: keeps `import panoscope.methods` free of pandas

        tr = Trace()
        table = pd.read_csv(path)
        spec = self.detect_unit(table)
        world = MeasuredWorld(adapter.frames(store))
        conditions = adapter.standing_conditions(store)
        rows = _rows(table)

        out: dict[str, NormalisedEvidence] = {}
        for uid in spec.units:
            mine = tuple(r for r in rows if str(r["top_cluster"]) == uid)
            readings = tuple(self._reading(r) for r in mine)
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

        The ladder stops here. No promote, no demote, no cap, no floor -- those are
        ``core/verdict.py``'s, because they are the project's confidence policy and
        would otherwise be re-stated (and eventually re-tuned, differently) by every
        method that registers. What a method owns is the one thing only it can know:
        what a given magnitude of its own statistic is worth.

        ``abs()`` matches the live config: a strongly negative coefficient is a
        strong measurement, and direction is carried on the ``EvidenceItem`` rather
        than smuggled into the band as a weakness.
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

        Deliberately short. A spatial co-location coefficient licenses asking "what
        is this gene a canonical marker of" and "what else marks that type" -- it
        licenses nothing about regulation or pathway membership, because the
        measurement says nothing about either. The predicate vocabulary lands with
        ``panoscope/reference/graph.py``; these are the two edges the marker workflow
        already follows today.
        """
        return (("marker_of", "forward"), ("marker_of", "reverse"))


def _edge(band: str) -> float:
    return next(e for lab, e in BAND_EDGES if lab == band)


#: The registered instance. One object, built once, attributes frozen.
MODULE = JazzPandaSpatialGLM()
