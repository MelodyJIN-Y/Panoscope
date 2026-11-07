"""Tier-A gene-set enrichment engine (deterministic).

Turns a per-cluster gene-set enrichment RESULT into a grounded ``ClusterEnrichment``
per cluster: the enriched pathways, a two-tier split (``enriched`` vs
``suggestive · verify``), a per-item census, and a ``verify`` flag — every value
tracing to the enrichment result or to the study's capability frames. No network,
no LLM.

The method wired here is the biologist's own jazzPanda competitive gene-set test
(``_load_jazzpanda_rows`` parses its result CSV). The schema keeps a ``score_kind``
so the record stays self-describing, but this workflow interprets one method.

**No band.** CHARACTERISATION has no label space, so a unit-level confidence is
undefined. The header is ``ClusterEnrichment.census_line``; ``verify`` comes from the
deterministic rubric in :func:`_cluster_enrichment`.

Confident floor (enrichment edition):
- Never invent a pathway, a score, a q-value, or a leading-edge gene — all come
  from the result.
- **Panel-coverage rule:** a Hallmark set has ~200 genes but only ``panel_hits``
  were usably measured here; enrichment is only measured over those. Every row
  carries ``panel_hits``/``panel_coverage`` + a deterministic "panel-scoped, not
  genome-wide" caveat. A set with too few measured genes, or a tiny leading edge,
  is ``untestable`` and never surfaced.
- **Two tiers:** ``enriched`` (q < ENRICH_Q_MAX, leading edge >= MIN_LEADING_EDGE,
  panel_hits >= MIN_PANEL_HITS); ``suggestive`` (same gate but q in
  [ENRICH_Q_MAX, SUGGESTIVE_Q_MAX]) carries ``verify = TRUE``; everything else is
  ``untestable``.
"""

from __future__ import annotations

import csv

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent.types import ClusterEnrichment, PathwayEvidence
from panoscope.methods.jazzpanda_cgst import _in_frame, _split, _to_float
from panoscope.types.subject import EntityKind

# The dataset's jazzPanda enrichment result: prefer the per-dataset tree
# (inputs/enrichment.csv), else the bundled legacy demo file. Absent -> no
# Pathways slice (fail-soft in the pipeline).
def _jz_enrichment_csv():
    """The active study's jazzPanda enrichment result, resolved at CALL time.

    The bundled demo file is a fallback ONLY for the demo.
    """
    legacy = (
        cfg.DATA_DIR_PATH / "jazzpanda" / "hbc_sp1_top10_hallmark_test_statistic.csv"
        if cfg.DATASET_ID == cfg.BUNDLED_DEMO_ID
        else cfg.current_study().inputs / "enrichment.csv"
    )
    return cfg._active_input("enrichment.csv", legacy)
_JZ_SCORE_KIND = "jazzpanda_enrichment"

#: Set-name prefix -> collection. Derived per ROW from the set's own name, so a
#: record rebuilt by ``tests/golden_projection`` re-derives it with no file read.
_COLLECTION_BY_SET_PREFIX: tuple[tuple[str, str], ...] = (
    ("HALLMARK_", "MSigDB_Hallmark"),
)

#: A set matching no prefix. A guessed name would be false provenance (and could
#: name a restrictively-licensed collection): add a prefix entry, never a default.
UNKNOWN_COLLECTION: str = "unknown"


def _collection_for(gene_set: str) -> str:
    """The collection a set was drawn from, read off the set's own name."""
    name = gene_set.strip().upper()
    for prefix, collection in _COLLECTION_BY_SET_PREFIX:
        if name.startswith(prefix):
            return collection
    return UNKNOWN_COLLECTION


# --------------------------------------------------------------------------- #
# Panel size (the enrichment universe) — read from the panel file, per study.
# --------------------------------------------------------------------------- #
@data._per_study
def _panel_size() -> int:
    """The analysed panel's gene count, derived per study (the panel-coverage denominator)."""
    return int(len(data.load_panel()))


def _panel_scope_caveat(gene_set: str, panel_hits: int, set_size_full: int) -> str:
    """The mandatory, deterministic panel-scoped caveat (not model-generated).

    Says *usably measured*: ``panel_hits`` counts genes every gene frame admits.
    """
    return (
        f"Panel-scoped: only {panel_hits} of {set_size_full} genes in {gene_set} "
        f"were usably measured on this {_panel_size()}-gene panel (on-panel and "
        f"producing spatial signal); over-representation among measured genes, not "
        f"genome-wide."
    )


# --------------------------------------------------------------------------- #
# jazzPanda enrichment result adapter (the FIRST method).
# --------------------------------------------------------------------------- #
#: The frame-checked panel-hit count :func:`_load_jazzpanda_rows` installs; the ONLY
#: one downstream may read. Named apart from the result's own ``n_overlap``.
PANEL_HITS_IN_FRAME: str = "panel_hits_in_frame"


@data._per_study
def _gene_world():
    """The study's capability frames, as a ``MeasuredWorld``. Declared, not counted here."""
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld

    study = cfg.current_study()
    world = MeasuredWorld(tuple(XeniumAdapter().frames(study)))
    if not world.for_kind(EntityKind.GENE):
        raise ValueError(
            f"[enrichment] study {study.id!r} declares no gene capability frame, so "
            "no set's panel footprint can be counted. Every panel hit would come "
            "back 0 and every program would read as untestable — a study-wide "
            "refusal with a false reason behind it. Fix inputs/panel.parquet (and "
            "inputs/markers_top.csv for detectability) before interpreting."
        )
    return world


@data._per_study
def _load_jazzpanda_rows() -> tuple[dict, ...]:
    """Parse the jazzPanda enrichment result CSV into per-(cluster, set) rows.

    Columns consumed: gene_set, geneset_size, test_statistic, p_value, p_adj_bh,
    gc_corr, genes_tested, genes_selected, cluster. The honest driving genes are
    ``genes_selected`` (the lasso sig_features).

    ``n_overlap`` is never read: it would credit genes the measurement never usably
    saw (panel-absence rule). Each footprint is recounted against every gene frame and
    installed as :data:`PANEL_HITS_IN_FRAME`, using the same ``_in_frame`` rule as
    ``methods/jazzpanda_cgst`` so the two engines stay comparable.
    """
    path = _jz_enrichment_csv()
    if not path.exists():
        raise FileNotFoundError(
            f"[enrichment] jazzPanda enrichment result missing: {path}. "
            f"Place the per-cluster top-set result there."
        )
    world = _gene_world()
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    for row in rows:
        if row.get("genes_tested") is None:
            raise KeyError(
                f"[enrichment] {path} has no 'genes_tested' column. Without the "
                "set's tested symbols the panel footprint cannot be checked against "
                "the capability frames, and falling back to 'n_overlap' would take "
                "the method's word for its own capability."
            )
        row[PANEL_HITS_IN_FRAME] = _in_frame(_split(row["genes_tested"]), world)
    return tuple(rows)


def _pathway_from_jz_row(row: dict) -> PathwayEvidence:
    """Build a grounded PathwayEvidence from one jazzPanda result row.

    The leading edge goes through the same frames as ``panel_hits``: a gene the frames
    reject is not a driver. The filter is idempotent, so golden-projection round-trips
    hold; rejected counts are reported by ``pipeline/stages/enrichment._report_divergence``.
    """
    gene_set = str(row["gene_set"]).strip()
    set_size_full = int(float(row["geneset_size"]))
    if PANEL_HITS_IN_FRAME not in row:
        raise KeyError(
            f"[enrichment] row for {gene_set!r} carries no {PANEL_HITS_IN_FRAME!r}. "
            "That key is installed by _load_jazzpanda_rows after checking the set's "
            "tested genes against the study's capability frames; a row that skipped "
            "it has an unaudited footprint, and reading 'n_overlap' instead would "
            "silently restore the count this change exists to replace."
        )
    panel_hits = int(row[PANEL_HITS_IN_FRAME])   # set genes EVERY gene frame admits
    leading_edge = tuple(
        g for g in _split(row.get("genes_selected", ""))
        if _in_frame((g,), _gene_world())
    )
    n_leading_edge = len(leading_edge)
    score = _to_float(row.get("test_statistic")) or 0.0
    p_value = _to_float(row.get("p_value"))
    q_value = _to_float(row.get("p_adj_bh"))
    gc_corr = _to_float(row.get("gc_corr"))
    coverage = (panel_hits / set_size_full) if set_size_full else 0.0

    # Tier: the report gate + two tiers (untestable never surfaced as a theme).
    if panel_hits < cfg.MIN_PANEL_HITS or n_leading_edge < cfg.MIN_LEADING_EDGE:
        tier = "untestable"
    elif q_value is not None and q_value < cfg.ENRICH_Q_MAX:
        tier = "enriched"
    elif q_value is not None and q_value <= cfg.SUGGESTIVE_Q_MAX:
        tier = "suggestive"
    else:
        tier = "untestable"

    caveats: list[str] = []
    if coverage < 0.05:
        caveats.append("very low panel coverage")
    if n_leading_edge < 5:
        caveats.append("tiny leading edge")
    if gc_corr is not None and gc_corr < 0.30:
        caveats.append("low spatial specificity")

    return PathwayEvidence(
        gene_set=gene_set,
        gene_set_collection=_collection_for(gene_set),
        score=score,
        score_kind=_JZ_SCORE_KIND,
        p_value=p_value,
        q_value=q_value,
        set_size_full=set_size_full,
        panel_hits=panel_hits,
        panel_coverage=coverage,
        leading_edge=leading_edge,
        n_leading_edge=n_leading_edge,
        gc_corr=gc_corr,
        tier=tier,
        panel_scope_caveat=_panel_scope_caveat(gene_set, panel_hits, set_size_full),
        caveats=tuple(caveats),
        source="jazzpanda:enrichment",
    )


# --------------------------------------------------------------------------- #
# The per-cluster call (two-tier + census + verify), method-agnostic. No band:
# CHARACTERISATION has no label space (IMPLEMENTATION.md §10).
# --------------------------------------------------------------------------- #
def _cluster_enrichment(
    cluster: str, pathways: tuple[PathwayEvidence, ...], method: str
) -> ClusterEnrichment:
    """Assemble a ClusterEnrichment from that cluster's PathwayEvidence rows."""
    cell_type = annotation.cell_type_for(cluster)
    enriched = tuple(
        sorted((p for p in pathways if p.tier == "enriched"), key=lambda p: p.score, reverse=True)
    )
    suggestive = tuple(
        sorted((p for p in pathways if p.tier == "suggestive"), key=lambda p: p.score, reverse=True)
    )
    top = enriched[0] if enriched else None
    # The deterministic verify rubric: nothing clears the enriched bar, or the top
    # call rests on too little of its set. The model is never offered the choice.
    verify = (
        not enriched
        or (top is not None and (top.panel_coverage < 0.05 or top.n_leading_edge < cfg.MIN_LEADING_EDGE))
    )
    source_trace = tuple(
        f"{p.gene_set}: score={p.score:.2f} q={p.q_value} K={p.panel_hits}/{p.set_size_full} LE={p.n_leading_edge}"
        for p in pathways
    )
    return ClusterEnrichment(
        cluster=cluster,
        cell_type=cell_type,
        method=method,
        enriched=enriched,
        suggestive=suggestive,
        all_tested=pathways,
        top_theme=top.gene_set if top else None,
        verify=bool(verify),
        source_trace=source_trace,
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def enrichment_for_cluster(cluster: str) -> ClusterEnrichment:
    """Return the jazzPanda-method enrichment verdict for one cluster.

    Raises KeyError for an unknown cluster id. Reads the biologist's enrichment
    result; computes nothing new — tiers the reported sets and counts them.
    """
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(f"[enrichment] unknown cluster {cluster!r}")
    rows = [r for r in _load_jazzpanda_rows() if str(r.get("cluster")) == cluster]
    pathways = tuple(_pathway_from_jz_row(r) for r in rows)
    return _cluster_enrichment(cluster, pathways, _JZ_SCORE_KIND)


def load_or_compute(cluster: str) -> ClusterEnrichment:
    """The persisted enrichment record for ``cluster`` if the tree has one, else
    computed live by :func:`enrichment_for_cluster`."""
    from panoscope.pipeline import store  # lazy: pipeline imports agent

    persisted = store.load_enrichment(cluster)
    return persisted if persisted is not None else enrichment_for_cluster(cluster)


def all_enrichments() -> list[ClusterEnrichment]:
    """Enrichment verdicts for every cluster, in cluster order."""
    return [enrichment_for_cluster(c) for c in cfg.CLUSTER_ORDER]
