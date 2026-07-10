"""Holistic pathway themes across clusters (enrichment Step 4).

After annotating each cluster's enrichment on its own, read the whole set
together: which biological PROGRAMS recur across clusters, and do the enriched
pathways cohere with the cell-type calls. This is the enrichment mirror of
``agent.holistic`` — every number is COMPUTED AT RUNTIME from the enrichment
records + the authoritative ``CLUSTER_KEY``; nothing is invented.

Grounding discipline: a "recurring" program is one that clears the enriched gate
in >= 2 clusters (never a coincidence of one panel gene — the per-set leading-edge
gate already ran). The compartment summary is a GROUPING of the cluster key, not
new biology.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent import config as cfg
from agent import enrichment as agent_enrichment
from agent.types import ClusterEnrichment

_RECUR_MIN_CLUSTERS = 2   # enriched in >= 2 clusters to count as a recurring theme
_TOP_PER_CLUSTER = 3      # top enriched programs named per cluster


def _short(gene_set: str) -> str:
    return gene_set.replace("HALLMARK_", "").replace("_", " ").title()


@dataclass(frozen=True)
class RecurringTheme:
    """A program enriched across multiple clusters (grounded count)."""
    gene_set: str
    clusters: tuple[str, ...]
    cell_types: tuple[str, ...]
    n_clusters: int
    max_score: float


@dataclass(frozen=True)
class ClusterThemeLine:
    """One cluster's headline enrichment (top programs + its confidence)."""
    cluster: str
    cell_type: str
    category: str
    top_themes: tuple[str, ...]      # short names of top enriched sets
    confidence: str
    verify: bool


@dataclass(frozen=True)
class PathwayThemes:
    """Cross-cluster enrichment review: coherence notes + recurring + per-cluster."""
    method: str
    n_clusters_with_enrichment: int
    coherence_notes: tuple[str, ...]
    recurring: tuple[RecurringTheme, ...]
    per_cluster: tuple[ClusterThemeLine, ...]


# --------------------------------------------------------------------------- #
# Grounded builders (every number from the enrichment records)
# --------------------------------------------------------------------------- #
def _per_cluster_lines(enrichments: list[ClusterEnrichment]) -> tuple[ClusterThemeLine, ...]:
    lines = []
    for ce in enrichments:
        top = tuple(_short(p.gene_set) for p in ce.enriched[:_TOP_PER_CLUSTER])
        lines.append(
            ClusterThemeLine(
                cluster=ce.cluster,
                cell_type=ce.cell_type,
                category=cfg.CLUSTER_KEY[ce.cluster]["category"],
                top_themes=top,
                confidence=ce.confidence,
                verify=ce.verify,
            )
        )
    return tuple(lines)


def _recurring_themes(enrichments: list[ClusterEnrichment]) -> tuple[RecurringTheme, ...]:
    by_set: dict[str, list[ClusterEnrichment]] = {}
    scores: dict[str, float] = {}
    for ce in enrichments:
        for p in ce.enriched:
            by_set.setdefault(p.gene_set, []).append(ce)
            scores[p.gene_set] = max(scores.get(p.gene_set, 0.0), p.score)
    themes = [
        RecurringTheme(
            gene_set=gs,
            clusters=tuple(ce.cluster for ce in ces),
            cell_types=tuple(ce.cell_type for ce in ces),
            n_clusters=len(ces),
            max_score=scores[gs],
        )
        for gs, ces in by_set.items()
        if len(ces) >= _RECUR_MIN_CLUSTERS
    ]
    themes.sort(key=lambda t: (t.n_clusters, t.max_score), reverse=True)
    return tuple(themes)


def _coherence_notes(
    enrichments: list[ClusterEnrichment],
    recurring: tuple[RecurringTheme, ...],
) -> tuple[str, ...]:
    notes: list[str] = []

    # (a) coverage
    with_enr = [ce for ce in enrichments if ce.enriched]
    without = [ce.cluster for ce in enrichments if not ce.enriched]
    cov = (
        f"{len(with_enr)} of {len(enrichments)} clusters have at least one enriched "
        f"pathway (q<{cfg.ENRICH_Q_MAX}, >={cfg.MIN_LEADING_EDGE} driving genes on panel)."
    )
    if without:
        cov += f" No enriched program clears the bar for {', '.join(without)}."
    notes.append(cov)

    # (b) recurring programs (grounded counts)
    if recurring:
        parts = [
            f"{_short(t.gene_set)} across {', '.join(t.clusters)} ({', '.join(dict.fromkeys(t.cell_types))})"
            for t in recurring[:3]
        ]
        notes.append("Programs shared across clusters: " + "; ".join(parts) + ".")
    else:
        notes.append("No Hallmark program is enriched in more than one cluster.")

    # (c) compartment summary (a grouping of the cluster key, not new biology)
    by_cat: dict[str, list[str]] = {}
    for ce in enrichments:
        if ce.enriched:
            cat = cfg.CLUSTER_KEY[ce.cluster]["category"]
            by_cat.setdefault(cat, []).append(f"{ce.cluster} {ce.cell_type}")
    if by_cat:
        seg = "; ".join(f"{cat} ({', '.join(cls)})" for cat, cls in by_cat.items())
        notes.append("Enriched clusters by compartment: " + seg + ".")

    return tuple(notes)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def pathway_themes() -> PathwayThemes:
    """Read all cluster enrichments together and surface the cross-cluster themes.

    Deterministic (Tier A): recurring programs, per-cluster headlines, and grounded
    coherence notes — every count from the enrichment records.
    """
    enrichments = agent_enrichment.all_enrichments()
    recurring = _recurring_themes(enrichments)
    return PathwayThemes(
        method=enrichments[0].method if enrichments else "jazzpanda_enrichment",
        n_clusters_with_enrichment=sum(1 for ce in enrichments if ce.enriched),
        coherence_notes=_coherence_notes(enrichments, recurring),
        recurring=recurring,
        per_cluster=_per_cluster_lines(enrichments),
    )
