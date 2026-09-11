"""Per-dataset cell-type annotation — the marker-gene skill's Output 2, read from the tree.

The annotation (cluster -> cell type + lineage/category + the canonical markers that
drive the panel-absence rule) is the marker-gene skill's *output*, not a hardcoded
key. A processed dataset ships ``interp/annotation.json`` (written by the annotate
stage, or snapshotted for the bundled demo) and it is read directly. When the file is
absent, we fall back to the built-in demo map so the bundled dataset works out of the
box, and so a fresh checkout / CI has a deterministic annotation without a live run.

This is what makes the cell type come FROM THE DATA (the skill) rather than a literal
in ``config.py`` / ``verdict.py``. The canonical-marker knowledge lives here (moved out
of ``verdict.py``) so the annotation artifact can carry per-cell-type canonical markers
for any dataset, and so ``verdict`` can import it without a cycle.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

from agent import config as cfg

# --------------------------------------------------------------------------- #
# Fallback / bootstrap domain knowledge (the bundled demo map). A processed
# dataset's interp/annotation.json supersedes this per cluster.
# --------------------------------------------------------------------------- #
CANONICAL_MARKERS_FALLBACK: dict[str, tuple[str, ...]] = {
    "Tumor": ("ERBB2", "EPCAM", "KRT8", "KRT7", "FOXA1", "GATA3", "KRT18"),
    "Stromal": ("LUM", "POSTN", "PDGFRA", "PDGFRB", "DCN", "COL1A1"),
    "Macrophages": ("LYZ", "CD68", "CD163", "ITGAX", "FCGR3A", "FCER1G", "CSF1R"),
    "Myoepithelial": ("MYLK", "ACTA2", "KRT14", "KRT5", "MYH11", "OXTR", "TP63"),
    "T_Cells": ("IL7R", "PTPRC", "TRAC", "CD3E", "CD3D", "CD8A", "CD4"),
    "B_Cells": ("MS4A1", "CD79A", "CD79B", "CD19", "BANK1", "MZB1"),
    "Endothelial": ("PECAM1", "VWF", "CD93", "AQP1", "CLDN5", "CDH5", "FLT1"),
    "Dendritic": ("LILRA4", "TCL1A", "SPIB", "PLD4", "IL3RA", "CLEC4C", "IRF7"),
    "Mast_Cells": ("CPA3", "TPSAB1", "KIT", "CTSG", "MS4A2", "TPSB2"),
}
OFF_PANEL_CANONICAL_FALLBACK: dict[str, tuple[str, ...]] = {
    "Stromal": ("COL1A1", "COL1A2", "DCN", "VIM", "FAP"),
}


def _annotation_path(dataset_id: Optional[str] = None) -> Path:
    return (
        cfg.DATA_DIR_PATH / "datasets" / (dataset_id or cfg.DATASET_ID)
        / "interp" / "annotation.json"
    )


@lru_cache(maxsize=8)
def load_annotation(dataset_id: Optional[str] = None) -> dict[str, dict]:
    """cluster -> {cell_type, cell_type_short, category, lineage, canonical_markers, offpanel_canonical}.

    Reads the per-dataset skill annotation (``interp/annotation.json``) when present,
    else builds the fallback from ``cfg.CLUSTER_KEY`` + the bundled canonical maps.
    """
    path = _annotation_path(dataset_id)
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            clusters = raw.get("clusters", raw)
            if isinstance(clusters, dict) and clusters:
                return clusters
        except (OSError, json.JSONDecodeError, ValueError):
            pass  # fall through to the bundled map
    # No annotation file yet: derive over the dataset's real clusters. The bundled
    # demo map supplies known types ONLY for the bundled demo dataset; any OTHER
    # dataset reads "Unknown" until the annotate stage writes annotation.json, so it
    # never inherits demo cell types (even if it happens to reuse c1/c2/... ids).
    is_bundled_demo = (dataset_id or cfg.DATASET_ID) == cfg.BUNDLED_DEMO_ID
    out: dict[str, dict] = {}
    for c in cfg.CLUSTER_ORDER:
        meta = cfg.CLUSTER_KEY.get(c) if is_bundled_demo else None
        if meta is None:
            out[c] = {
                "cluster": c, "cell_type": "Unknown", "cell_type_short": "Unknown",
                "category": "Unknown", "lineage": "Unknown",
                "canonical_markers": [], "offpanel_canonical": [],
            }
        else:
            ct = meta["cell_type"]
            out[c] = {
                "cluster": c, "cell_type": ct,
                "cell_type_short": meta["cell_type_short"],
                "category": meta["category"], "lineage": meta["lineage"],
                "canonical_markers": list(CANONICAL_MARKERS_FALLBACK.get(ct, ())),
                "offpanel_canonical": list(OFF_PANEL_CANONICAL_FALLBACK.get(ct, ())),
            }
    return out


def meta_for(cluster: str, dataset_id: Optional[str] = None) -> dict:
    """The full annotation record for ``cluster`` (KeyError if unknown)."""
    return load_annotation(dataset_id)[cluster]


def cell_type_for(cluster: str, dataset_id: Optional[str] = None) -> str:
    return load_annotation(dataset_id)[cluster]["cell_type"]


def _by_cell_type(field: str, dataset_id: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for meta in load_annotation(dataset_id).values():
        out.setdefault(str(meta["cell_type"]), tuple(meta.get(field) or ()))
    return out


def canonical_markers(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """Canonical markers for ``cell_type`` (from the annotation, else the fallback map)."""
    return _by_cell_type("canonical_markers", dataset_id).get(
        cell_type, CANONICAL_MARKERS_FALLBACK.get(cell_type, ())
    )


def offpanel_canonical(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """Off-panel canonical markers for ``cell_type`` (panel-absence note source)."""
    return _by_cell_type("offpanel_canonical", dataset_id).get(
        cell_type, OFF_PANEL_CANONICAL_FALLBACK.get(cell_type, ())
    )


def all_canonical(dataset_id: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    """cell_type -> canonical markers, over every annotated cluster (for discrimination)."""
    return _by_cell_type("canonical_markers", dataset_id)
