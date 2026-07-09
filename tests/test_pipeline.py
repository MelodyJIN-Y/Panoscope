"""Pipeline slice 1: verdict serialization round-trip + a tmp-tree smoke test.

The round-trip equality is the correctness gate for reading verdicts off disk
instead of recomputing them: a persisted verdict must reload to an object EQUAL
to the freshly computed one. The smoke test runs the whole pipeline into a tmp
dir (never the real data/datasets tree) and checks the artifacts + faithful read.
"""

from __future__ import annotations

import json

from agent import config as cfg
from agent import verdict as agent_verdict

from pipeline import paths
from pipeline import run as pipeline_run
from pipeline import store
from pipeline.serialize import verdict_from_dict, verdict_to_dict
from pipeline.stages.validate import validate


def test_verdict_roundtrip_equals_computed():
    for c in cfg.CLUSTER_ORDER:
        v = agent_verdict.verdict_for_cluster(c)
        assert verdict_from_dict(verdict_to_dict(v)) == v


def test_verdict_roundtrip_through_json():
    for c in cfg.CLUSTER_ORDER:
        v = agent_verdict.verdict_for_cluster(c)
        again = verdict_from_dict(json.loads(json.dumps(verdict_to_dict(v))))
        assert again == v


def test_validate_passes_on_real_inputs():
    validate()  # must not raise on the committed demo inputs


def test_run_writes_tree_and_store_reads_faithfully(tmp_path):
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)

    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert man["dataset_id"] == cfg.DATASET_ID
    assert man["clusters"] == list(cfg.CLUSTER_ORDER)
    assert "markers_top.csv" in man["inputs"]
    assert man["artifacts"]  # non-empty

    csv_text = paths.verdicts_csv(cfg.DATASET_ID, tmp_path).read_text()
    assert csv_text.startswith("cluster,cell_type")
    assert len(csv_text.strip().splitlines()) == 1 + len(cfg.CLUSTER_ORDER)

    for c in cfg.CLUSTER_ORDER:
        loaded = store.load_verdict(c, cfg.DATASET_ID, root=tmp_path)
        assert loaded == agent_verdict.verdict_for_cluster(c)

    assert store.load_all_verdicts(cfg.DATASET_ID, root=tmp_path) == agent_verdict.all_verdicts()


def test_store_returns_none_when_absent(tmp_path):
    assert store.load_verdict("c1", cfg.DATASET_ID, root=tmp_path) is None
    assert store.load_all_verdicts(cfg.DATASET_ID, root=tmp_path) is None
