"""Read persisted per-dataset artifacts off the pipeline tree.

The reader for what the pipeline wrote. It is fail-soft by design: a missing or
malformed artifact returns ``None`` so callers fall back to the live engine and
the app never breaks on a partial tree. A returned verdict is byte-faithful to
the computed one (guaranteed by ``tests/test_pipeline.py``'s round-trip gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from agent import config as cfg
from agent.types import ClusterVerdict

from pipeline import paths
from pipeline.serialize import verdict_from_dict


def load_verdict(
    cluster: str,
    dataset_id: str = cfg.DATASET_ID,
    root: Optional[Path] = None,
) -> Optional[ClusterVerdict]:
    """Return the persisted verdict for ``cluster``, or None if absent/unreadable."""
    p = paths.cluster_json(dataset_id, cluster, root)
    if not p.exists():
        return None
    try:
        return verdict_from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - fail soft; caller recomputes live
        return None


def load_all_verdicts(
    dataset_id: str = cfg.DATASET_ID,
    root: Optional[Path] = None,
) -> Optional[list[ClusterVerdict]]:
    """Return all persisted verdicts in cluster order, or None if any is missing.

    All-or-nothing: a partial tree returns None so the caller recomputes the full
    set live rather than mixing persisted and computed calls.
    """
    out: list[ClusterVerdict] = []
    for cluster in cfg.CLUSTER_ORDER:
        v = load_verdict(cluster, dataset_id, root)
        if v is None:
            return None
        out.append(v)
    return out


def load_celltype_notes(
    dataset_id: str = cfg.DATASET_ID,
    root: Optional[Path] = None,
) -> dict:
    """Return the per-cluster cell-type notes map, or {} if absent/unreadable.

    Shape: ``{cluster: {cell_type, summary, pmid|null, citation|null, verify}}``.
    Fail-soft: a missing file returns an empty dict so callers show a plain
    fallback rather than crashing.
    """
    p = paths.interp_dir(dataset_id, root) / "celltype_notes.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001 - fail soft
        return {}
