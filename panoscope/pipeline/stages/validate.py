"""Stage 0 — validate a dataset's raw inputs before any compute (fail closed).

Reads the inputs through ``agent.data`` (exactly what the app will see) and
asserts they are present and coherent: the cluster key covers every cluster, the
marker table's ``top_cluster`` labels are known, every cluster has at least one
assigned marker, and the panel is non-empty. A precise ValueError here beats a
confusing failure three stages later.
"""

from __future__ import annotations

from agent import config as cfg
from agent import data


def validate(dataset_id: str = cfg.DATASET_ID) -> None:
    """Assert the dataset's inputs are present and well-formed. Raises on any gap."""
    markers = data.load_markers()
    panel = data.load_panel()

    # cluster_key.json is optional: cell types are the marker-skill annotation
    # (interp/annotation.json). If a key IS provided it must cover every cluster.
    try:
        key = data.load_cluster_key()
    except Exception:  # noqa: BLE001 - absent key is fine; annotate assigns the types
        key = None
    if key is not None:
        missing_key = [c for c in cfg.CLUSTER_ORDER if c not in key]
        if missing_key:
            raise ValueError(f"[pipeline] cluster_key is missing clusters: {missing_key}")

    allowed = set(cfg.KNOWN_CLUSTERS) | {"NoSig"}
    seen = set(markers["top_cluster"].astype(str))
    unknown = sorted(seen - allowed)
    if unknown:
        raise ValueError(
            f"[pipeline] markers_top has unknown top_cluster labels: {unknown} "
            f"(allowed: c1..cN + NoSig)"
        )

    empty = [c for c in cfg.CLUSTER_ORDER if data.get_cluster_markers(c).empty]
    if empty:
        raise ValueError(f"[pipeline] clusters with no assigned markers: {empty}")

    if panel.empty:
        raise ValueError("[pipeline] panel is empty; the absence primitive needs it")
