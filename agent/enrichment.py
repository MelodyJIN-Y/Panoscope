"""Tier-A gene-set enrichment engine (deterministic) — the enrichment mirror of
``agent.verdict``.

Turns a per-cluster gene-set enrichment RESULT into a grounded ``ClusterEnrichment``
per cluster: the enriched pathways, a two-tier split (``enriched`` vs
``suggestive · verify``), a confidence, and a ``verify`` flag — every value tracing
to the enrichment result or the panel. No network, no LLM.

The method wired here is the biologist's own jazzPanda competitive gene-set test
(``_load_jazzpanda_rows`` parses its result CSV). The schema keeps a ``score_kind``
so the record stays self-describing, but this workflow interprets one method.

Confident floor (enrichment edition):
- Never invent a pathway, a score, a q-value, or a leading-edge gene — all come
  from the result.
- **Panel-coverage rule:** a Hallmark set has ~200 genes but only ``panel_hits``
  are on this ~280-gene panel; enrichment is only measured over those. Every row
  carries ``panel_hits``/``panel_coverage`` + a deterministic "panel-scoped, not
  genome-wide" caveat. A set with too few panel genes, or a tiny leading edge, is
  ``untestable`` and never surfaced.
- **Two tiers:** ``enriched`` (q < ENRICH_Q_MAX, leading edge >= MIN_LEADING_EDGE,
  panel_hits >= MIN_PANEL_HITS); ``suggestive`` (same gate but q in
  [ENRICH_Q_MAX, SUGGESTIVE_Q_MAX]) carries ``verify = TRUE``; everything else is
  ``untestable``.
"""

from __future__ import annotations

import csv
from functools import lru_cache

from agent import annotation
from agent import config as cfg
from agent import data
from agent.types import ClusterEnrichment, PathwayEvidence

# The dataset's jazzPanda enrichment result: prefer the per-dataset tree
# (inputs/enrichment.csv), else the bundled legacy demo file. Absent -> no
# Pathways slice (fail-soft in the pipeline).
_JZ_ENRICHMENT_CSV = cfg._active_input(
    "enrichment.csv",
    cfg.DATA_DIR_PATH / "jazzpanda" / "hbc_sp1_top10_hallmark_test_statistic.csv",
)
_JZ_SCORE_KIND = "jazzpanda_enrichment"
_JZ_COLLECTION = "MSigDB_Hallmark"


# --------------------------------------------------------------------------- #
# Panel size (the enrichment universe) — read once from the panel file.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _panel_size() -> int:
    return int(len(data.load_panel()))


def _panel_scope_caveat(gene_set: str, panel_hits: int, set_size_full: int) -> str:
    """The mandatory, deterministic panel-scoped caveat (not model-generated)."""
    return (
        f"Panel-scoped: only {panel_hits} of {set_size_full} genes in {gene_set} "
        f"are on this {_panel_size()}-gene panel; over-representation among measured "
        f"genes, not genome-wide."
    )


# --------------------------------------------------------------------------- #
# jazzPanda enrichment result adapter (the FIRST method).
# --------------------------------------------------------------------------- #
def _split_genes(cell: str) -> tuple[str, ...]:
    if not cell:
        return ()
    return tuple(g.strip() for g in str(cell).split(",") if g.strip())


@lru_cache(maxsize=1)
def _load_jazzpanda_rows() -> tuple[dict, ...]:
    """Parse the jazzPanda enrichment result CSV into per-(cluster, set) rows.

    Columns consumed: gene_set, geneset_size, n_overlap, test_statistic, p_value,
    p_adj_bh, gc_corr, n_sig_features, genes_selected, cluster, anno. The honest
    driving genes are ``genes_selected`` (the lasso sig_features), NOT n_overlap.
    """
    path = _JZ_ENRICHMENT_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"[enrichment] jazzPanda enrichment result missing: {path}. "
            f"Place the per-cluster top-set result there."
        )
    with open(path, newline="", encoding="utf-8") as fh:
        return tuple(csv.DictReader(fh))


def _to_float(v) -> float | None:
    try:
        f = float(v)
        return f if f == f else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _pathway_from_jz_row(row: dict) -> PathwayEvidence:
    """Build a grounded PathwayEvidence from one jazzPanda result row."""
    gene_set = str(row["gene_set"]).strip()
    set_size_full = int(float(row["geneset_size"]))
    panel_hits = int(float(row["n_overlap"]))           # set genes on panel (tested)
    leading_edge = _split_genes(row.get("genes_selected", ""))
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
        gene_set_collection=_JZ_COLLECTION,
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
# The per-cluster verdict (band + two-tier + verify), method-agnostic.
# --------------------------------------------------------------------------- #
def _band_cluster(top: PathwayEvidence | None, method: str) -> tuple[str, tuple[str, ...]]:
    """Confidence band from the top enriched set's score, with demotions.

    Mirrors verdict._compute_band: base band on the score, then demote one band
    each for very low panel coverage or a tiny leading edge. No enriched set -> Low.
    """
    if top is None:
        return "Low", ()
    band = cfg.band_for_enrichment(top.score, method)
    order = ("Very High", "High", "Medium-High", "Medium", "Low")
    idx = order.index(band)
    demotions: list[str] = []
    if top.panel_coverage < 0.05:
        idx = min(idx + 1, len(order) - 1)
        demotions.append("top set very low panel coverage -> -1 band")
    if top.n_leading_edge < 5:
        idx = min(idx + 1, len(order) - 1)
        demotions.append("top set tiny leading edge -> -1 band")
    return order[idx], tuple(demotions)


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
    confidence, demotions = _band_cluster(top, method)
    # verify when nothing clears the enriched bar, or the top call is shaky.
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
        confidence=confidence,
        confidence_score=cfg.SCORE_MAP[confidence],
        verify=bool(verify),
        demotions=demotions,
        source_trace=source_trace,
    )


# --------------------------------------------------------------------------- #
# Public API (mirrors agent.verdict)
# --------------------------------------------------------------------------- #
def enrichment_for_cluster(cluster: str) -> ClusterEnrichment:
    """Return the jazzPanda-method enrichment verdict for one cluster.

    Raises KeyError for an unknown cluster id. Reads the biologist's enrichment
    result; computes nothing new — bands + tiers the reported sets.
    """
    if cluster not in cfg.KNOWN_CLUSTERS:
        raise KeyError(f"[enrichment] unknown cluster {cluster!r}")
    rows = [r for r in _load_jazzpanda_rows() if str(r.get("cluster")) == cluster]
    pathways = tuple(_pathway_from_jz_row(r) for r in rows)
    return _cluster_enrichment(cluster, pathways, _JZ_SCORE_KIND)


def all_enrichments() -> list[ClusterEnrichment]:
    """Enrichment verdicts for all nine clusters (c1..c9), in order."""
    return [enrichment_for_cluster(c) for c in cfg.CLUSTER_ORDER]


# --------------------------------------------------------------------------- #
# CSV export — one row per surfaced pathway (enriched + suggestive), the
# portable analog of verdicts.csv. `leading_edge` is ";"-joined so the cell is
# atomic; untestable sets are not exported (they are never a surfaced call).
# --------------------------------------------------------------------------- #
ENRICHMENT_CSV_COLUMNS: tuple[str, ...] = (
    "cluster",
    "cell_type",
    "gene_set",
    "tier",
    "score",
    "score_kind",
    "q_value",
    "panel_hits",
    "set_size_full",
    "panel_coverage",
    "n_leading_edge",
    "leading_edge",
    "cluster_confidence",
    "cluster_verify",
)


def _csv_row(ce: ClusterEnrichment, p: PathwayEvidence) -> tuple[str, ...]:
    return (
        ce.cluster,
        ce.cell_type,
        p.gene_set,
        p.tier,
        f"{p.score:.4f}",
        p.score_kind,
        "" if p.q_value is None else f"{p.q_value:.3e}",
        str(p.panel_hits),
        str(p.set_size_full),
        f"{p.panel_coverage:.4f}",
        str(p.n_leading_edge),
        ";".join(p.leading_edge),
        ce.confidence,
        "TRUE" if ce.verify else "FALSE",
    )


def to_csv(enrichments: list[ClusterEnrichment], header: bool = True) -> str:
    """Render enrichment verdicts to CSV (enriched then suggestive per cluster)."""
    import csv
    import io

    buf = io.StringIO()
    w = csv.writer(buf)
    if header:
        w.writerow(ENRICHMENT_CSV_COLUMNS)
    for ce in enrichments:
        for p in (*ce.enriched, *ce.suggestive):
            w.writerow(_csv_row(ce, p))
    return buf.getvalue()
