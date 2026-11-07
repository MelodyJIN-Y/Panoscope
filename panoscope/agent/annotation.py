"""Per-dataset cell-type annotation, read from ``interp/annotation.json`` in the tree.

cluster -> cell type and lineage/category. Absent the file, the bundled demo's cluster
key applies to the bundled demo only; any other study reads "not yet called".
Canonical markers come from the study's reference view (``reference/view.py``), never
from the annotation or from code.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from panoscope.agent import config as cfg
from panoscope.data.study import NOT_YET_CALLED

def _annotation_path(dataset_id: str, root: Path) -> Path:
    return root / "studies" / dataset_id / "interp" / "annotation.json"


def load_annotation(dataset_id: Optional[str] = None) -> dict[str, dict]:
    """Resolve root and study BEFORE caching, so two roots' same-id studies never share an entry."""
    return _load_annotation_cached(str(cfg.DATA_DIR_PATH), dataset_id or cfg.DATASET_ID)


@lru_cache(maxsize=8)
def _load_annotation_cached(root: str, dataset_id: str) -> dict[str, dict]:
    """cluster -> {cell_type, cell_type_short, category, lineage}.

    Reads the per-dataset annotation (``interp/annotation.json``) when present and
    fills every cluster it does not name: the demo's cluster key, else not yet called.
    """
    path = _annotation_path(dataset_id, Path(root))
    stored: dict[str, dict] = {}
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            clusters = raw.get("clusters", raw)
            if isinstance(clusters, dict):
                stored = clusters
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # fall through to the bundled map
    # Clusters the file does not name: the bundled demo map supplies known types
    # ONLY for the bundled demo dataset; any OTHER dataset reads "not yet called"
    # until the biologist (or the skill's proposal) names it, so it never inherits
    # demo cell types (even if it happens to reuse c1/c2/... ids).
    is_bundled_demo = dataset_id == cfg.BUNDLED_DEMO_ID
    out: dict[str, dict] = dict(stored)
    for c in cfg.CLUSTER_ORDER:
        if c in out:
            continue
        meta = cfg.CLUSTER_KEY.get(c) if is_bundled_demo else None
        if meta is None:
            out[c] = {
                "cluster": c, "cell_type": NOT_YET_CALLED,
                "cell_type_short": NOT_YET_CALLED, "category": NOT_YET_CALLED,
                "lineage": NOT_YET_CALLED,
            }
        else:
            out[c] = {
                "cluster": c, "cell_type": meta["cell_type"],
                "cell_type_short": meta["cell_type_short"],
                "category": meta["category"], "lineage": meta["lineage"],
            }
    return out


def meta_for(cluster: str, dataset_id: Optional[str] = None) -> dict:
    """The full annotation record for ``cluster`` (KeyError if unknown)."""
    return load_annotation(dataset_id)[cluster]


def cell_type_for(cluster: str, dataset_id: Optional[str] = None) -> str:
    return load_annotation(dataset_id)[cluster]["cell_type"]


def _view(dataset_id: Optional[str] = None):
    """This study's reference view."""
    from panoscope.data.study import Study
    from panoscope.reference import view

    study = (Study(id=dataset_id, data_root=Path(cfg.DATA_DIR_PATH)) if dataset_id
             else cfg.current_study())
    return view.for_study(study)


def canonical_markers(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """Canonical markers for ``cell_type`` in THIS study's reference view; empty if none.

    No fallback across tissues: these markers feed the band.
    """
    return _view(dataset_id).canonical(cell_type)


def offpanel_canonical(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """The reference markers for ``cell_type`` that are not on the panel file
    (panel-absence note source). No readable panel: say nothing."""
    from panoscope.agent import data as _data

    out: list[str] = []
    for g in _view(dataset_id).markers(cell_type):
        try:
            if _data.panel_contains(g):
                continue
        except Exception:  # noqa: BLE001 - no readable panel: say nothing, not "off-panel"
            return ()
        out.append(g)
    return tuple(out)


def all_canonical(dataset_id: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    """cell_type -> canonical markers, over every annotated cluster (for discrimination)."""
    view = _view(dataset_id)
    return {str(m["cell_type"]): view.canonical(str(m["cell_type"]))
            for m in load_annotation(dataset_id).values()}


# Keep the public cache_clear() working; pipeline/stages/annotate.py calls it after
# writing a fresh annotation.json.
load_annotation.cache_clear = _load_annotation_cached.cache_clear  # type: ignore[attr-defined]
