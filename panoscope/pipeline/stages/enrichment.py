"""Enrichment stage — compute + persist the per-cluster enrichment (deterministic).

The enrichment mirror of ``stages/verdicts.py``: runs the deterministic engine
(``agent.enrichment.all_enrichments``) and writes each verdict to
``interp/enrichment/c{n}.json`` (full object: enriched + suggestive + all_tested,
confidence, audit) plus ``interp/enrichment.csv`` (one row per surfaced pathway).
Computes nothing new — it persists what the engine produced from the biologist's
enrichment result so the UI reads it off disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent import config as cfg
from agent import enrichment as agent_enrichment
from agent.types import ClusterEnrichment

from pipeline import paths
from pipeline.serialize import enrichment_to_dict


def run_enrichment(
    dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None
) -> list[ClusterEnrichment]:
    """Compute all enrichment verdicts, write per-cluster JSON + enrichment.csv."""
    enrichments = agent_enrichment.all_enrichments()

    edir = paths.enrichment_dir(dataset_id, root)
    edir.mkdir(parents=True, exist_ok=True)
    for e in enrichments:
        out = paths.enrichment_cluster_json(dataset_id, e.cluster, root)
        out.write_text(
            json.dumps(enrichment_to_dict(e), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    csv_path = paths.enrichment_csv(dataset_id, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(agent_enrichment.to_csv(enrichments), encoding="utf-8")

    return enrichments
