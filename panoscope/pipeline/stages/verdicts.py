"""Stage 4 — compute + persist the per-cluster verdicts (deterministic).

Runs the existing deterministic engine (``agent.verdict.all_verdicts``) and
writes each verdict to ``interp/clusters/c{n}.json`` (full object: call,
confidence, evidence, opening, audit trail) plus the canonical 11-column
``interp/verdicts.csv`` — the exact R-importable export. Computes nothing new;
it persists what the engine already produces so the UI can read it off disk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent import config as cfg
from agent import verdict as agent_verdict
from agent.types import ClusterVerdict

from pipeline import paths
from pipeline.serialize import verdict_to_dict


def run_verdicts(
    dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None
) -> list[ClusterVerdict]:
    """Compute all verdicts, write per-cluster JSON + verdicts.csv, return them."""
    verdicts = agent_verdict.all_verdicts()

    cdir = paths.clusters_dir(dataset_id, root)
    cdir.mkdir(parents=True, exist_ok=True)
    for v in verdicts:
        out = paths.cluster_json(dataset_id, v.cluster, root)
        out.write_text(
            json.dumps(verdict_to_dict(v), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    csv_path = paths.verdicts_csv(dataset_id, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(agent_verdict.to_csv(verdicts), encoding="utf-8")

    return verdicts
