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

from panoscope.agent import config as cfg

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
        cfg.DATA_DIR_PATH / "studies" / (dataset_id or cfg.DATASET_ID)
        / "interp" / "annotation.json"
    )


def load_annotation(dataset_id: Optional[str] = None) -> dict[str, dict]:
    """Resolve ``None`` to the current study BEFORE caching.

    Caching on the raw argument meant ``load_annotation(None)`` shared one entry
    across every study, so switching study silently served the previous study's
    annotation -- and that annotation is what drives cell-type labels and canonical
    markers. The public signature is unchanged.
    """
    return _load_annotation_cached(dataset_id or cfg.DATASET_ID)


@lru_cache(maxsize=8)
def _load_annotation_cached(dataset_id: str) -> dict[str, dict]:
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


def _is_bundled_demo(dataset_id: Optional[str] = None) -> bool:
    """Whether we are reading the bundled demo study.

    The fallback marker maps below are BREAST maps. They belong to the bundled demo
    and to nothing else.
    """
    return (dataset_id or cfg.DATASET_ID) == cfg.BUNDLED_DEMO_ID


def canonical_markers(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """Canonical markers for ``cell_type`` in THIS study.

    The bundled breast fallback fires only for the bundled demo. It used to fire for
    any study: a kidney or lung dataset whose annotation happened to name a cluster
    "Tumor" silently inherited ERBB2/EPCAM/KRT8/KRT7/FOXA1/GATA3/KRT18, and those
    markers feed ``verdict._is_canonical`` -> driver selection -> the confidence band.
    Wrong-tissue curated knowledge moving a band is the precise failure this project
    exists to prevent, so it is guarded structurally rather than by convention.

    A study with no annotated markers for a cell type returns empty. Empty means
    "this study says nothing", which the caller reports honestly; it never means
    "use another tissue's markers".
    """
    found = _by_cell_type("canonical_markers", dataset_id).get(cell_type)
    if found:
        return found
    if _is_bundled_demo(dataset_id):
        return CANONICAL_MARKERS_FALLBACK.get(cell_type, ())
    return ()


def offpanel_canonical(cell_type: str, dataset_id: Optional[str] = None) -> tuple[str, ...]:
    """Off-panel canonical markers for ``cell_type`` (panel-absence note source).

    Same guard as ``canonical_markers``: the fallback is a breast map and fires only
    for the bundled demo. An off-panel note naming another tissue's genes would be a
    fabricated caveat.
    """
    stored = _by_cell_type("offpanel_canonical", dataset_id).get(cell_type) or ()
    if not stored and _is_bundled_demo(dataset_id):
        stored = OFF_PANEL_CANONICAL_FALLBACK.get(cell_type, ())

    # DERIVED as well as read. The stored field is written once and never
    # recomputed (``pipeline/stages/annotate.py`` reads an existing
    # annotation.json rather than regenerating it), and in the committed demo it
    # is empty for eight of nine units while their canonical lists DO name
    # off-panel genes -- KRT18 for Tumor, CD3D for T_Cells, CLDN5/CDH5/FLT1 for
    # Endothelial. The panel-absence rule therefore demonstrated on one cluster.
    #
    # Membership is asked of the panel file, never asserted, so a study whose
    # annotation names no such type still gets ``()`` and no other tissue's genes
    # can leak in. Stored entries come first and win ties, because a study may
    # curate absences its canonical list does not name (the demo's Stromal entry
    # carries COL1A2, VIM and FAP, none of which are in its canonical set).
    from panoscope.agent import data as _data

    out: list[str] = []
    for g in (*stored, *canonical_markers(cell_type, dataset_id)):
        name = str(g)
        if name in out:
            continue
        try:
            if _data.panel_contains(name):
                continue
        except Exception:  # noqa: BLE001 - no readable panel: say nothing, not "off-panel"
            return tuple(stored)
        out.append(name)
    return tuple(out)


def all_canonical(dataset_id: Optional[str] = None) -> dict[str, tuple[str, ...]]:
    """cell_type -> canonical markers, over every annotated cluster (for discrimination)."""
    return _by_cell_type("canonical_markers", dataset_id)


# Keep the public cache_clear() working; pipeline/stages/annotate.py calls it after
# writing a fresh annotation.json.
load_annotation.cache_clear = _load_annotation_cached.cache_clear  # type: ignore[attr-defined]
