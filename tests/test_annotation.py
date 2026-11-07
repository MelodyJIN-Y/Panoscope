"""The per-dataset annotation layer: read the tree, else a safe fallback.

The cell type is the marker skill's output, read from ``interp/annotation.json``.
These assert (a) the bundled demo reads its file, (b) canonical / off-panel markers
resolve, and (c) a NON-demo dataset never inherits the demo's cell types.
"""
from __future__ import annotations

from panoscope.agent import annotation
from panoscope.agent import config as cfg


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
    # No annotation.json for this id -> fallback; a non-demo dataset reads not yet called,
    # never the bundled demo's cell types (regression: the fallback used to leak them).
    ann = annotation.load_annotation("some_unprepared_dataset_xyz")
    assert ann, "fallback should still enumerate the clusters"
    assert all(v["cell_type"] == "not yet called" for v in ann.values())


def test_the_cache_keys_on_the_data_root_as_well_as_the_study_id(tmp_path):
    """Two roots can hold a study with the same id; neither may serve the other's labels."""
    import json

    from panoscope.data import layout

    for root, label in (("a", "Tumor"), ("b", "Stromal")):
        p = tmp_path / root / "studies" / "same" / "interp"
        p.mkdir(parents=True)
        (p / "annotation.json").write_text(json.dumps({"c1": {"cell_type": label}}))
    try:
        layout.set_session_root(tmp_path / "a")
        assert annotation.load_annotation("same")["c1"]["cell_type"] == "Tumor"
        layout.set_session_root(tmp_path / "b")
        assert annotation.load_annotation("same")["c1"]["cell_type"] == "Stromal"
    finally:
        layout.set_session_root(None)
