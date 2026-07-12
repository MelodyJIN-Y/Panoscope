"""The per-dataset annotation layer: read the tree, else a safe fallback.

The cell type is the marker skill's output, read from ``interp/annotation.json``.
These assert (a) the bundled demo reads its file, (b) canonical / off-panel markers
resolve, and (c) a NON-demo dataset never inherits the demo's cell types.
"""
from __future__ import annotations

from agent import annotation
from agent import config as cfg


def test_reads_annotation_file_for_the_demo():
    assert annotation.cell_type_for("c2") == "Stromal"
    meta = annotation.meta_for("c2")
    assert meta["lineage"] == "Mesenchymal"
    assert meta["cell_type_short"] == "Str_Fib"


def test_canonical_and_offpanel_resolve():
    assert "LUM" in annotation.canonical_markers("Stromal")
    # COL1A1 / VIM are canonical fibroblast markers that are off this panel
    off = annotation.offpanel_canonical("Stromal")
    assert "COL1A1" in off and "VIM" in off


def test_all_canonical_covers_every_annotated_type():
    by_type = annotation.all_canonical()
    for c in cfg.CLUSTER_ORDER:
        assert annotation.cell_type_for(c) in by_type


def test_nondemo_dataset_never_inherits_demo_types():
    # No annotation.json for this id -> fallback; a non-demo dataset reads Unknown,
    # never the bundled demo's cell types (regression: the fallback used to leak them).
    ann = annotation.load_annotation("some_unprepared_dataset_xyz")
    assert ann, "fallback should still enumerate the clusters"
    assert all(v["cell_type"] == "Unknown" for v in ann.values())
