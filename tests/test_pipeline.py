"""Pipeline slice 1: verdict serialization round-trip + a tmp-tree smoke test.

The round-trip equality is the correctness gate for reading verdicts off disk
instead of recomputing them: a persisted verdict must reload to an object EQUAL
to the freshly computed one. The smoke test runs the whole pipeline into a tmp
dir (never the real data/datasets tree) and checks the artifacts + faithful read.
"""

from __future__ import annotations

import json

import pytest

from agent import config as cfg
from agent import enrichment as agent_enrichment
from agent import enrichment_themes as agent_enrichment_themes
from agent import holistic as agent_holistic
from agent import verdict as agent_verdict

from pipeline import paths
from pipeline import run as pipeline_run
from pipeline import store
from pipeline.serialize import (
    enrichment_from_dict,
    enrichment_to_dict,
    holistic_from_dict,
    holistic_to_dict,
    pathway_themes_from_dict,
    pathway_themes_to_dict,
    verdict_from_dict,
    verdict_to_dict,
)
from pipeline.stages.validate import validate

_HAS_ENRICHMENT = agent_enrichment._JZ_ENRICHMENT_CSV.exists()


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
    assert store.load_holistic(cfg.DATASET_ID, root=tmp_path) is None


def test_holistic_roundtrip_equals_computed():
    review = agent_holistic.holistic_review()
    assert holistic_from_dict(holistic_to_dict(review)) == review


def test_holistic_roundtrip_through_json():
    review = agent_holistic.holistic_review()
    again = holistic_from_dict(json.loads(json.dumps(holistic_to_dict(review))))
    assert again == review


@pytest.mark.skipif(not _HAS_ENRICHMENT, reason="enrichment result not present")
def test_enrichment_roundtrip_equals_computed():
    for e in agent_enrichment.all_enrichments():
        assert enrichment_from_dict(enrichment_to_dict(e)) == e
        again = enrichment_from_dict(json.loads(json.dumps(enrichment_to_dict(e))))
        assert again == e


@pytest.mark.skipif(not _HAS_ENRICHMENT, reason="enrichment result not present")
def test_run_writes_enrichment_faithfully(tmp_path):
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)

    assert paths.enrichment_csv(cfg.DATASET_ID, tmp_path).exists()
    loaded = store.load_all_enrichments(cfg.DATASET_ID, root=tmp_path)
    assert loaded == agent_enrichment.all_enrichments()

    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert "interp/enrichment.csv" in man["artifacts"]
    assert "interp/enrichment/c1.json" in man["artifacts"]


@pytest.mark.skipif(not _HAS_ENRICHMENT, reason="enrichment result not present")
def test_pathway_themes_roundtrip_and_persisted(tmp_path):
    themes = agent_enrichment_themes.pathway_themes()
    assert pathway_themes_from_dict(pathway_themes_to_dict(themes)) == themes

    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    assert store.load_pathway_themes(cfg.DATASET_ID, root=tmp_path) == themes


def test_run_writes_holistic_and_calibration(tmp_path):
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)

    # Holistic review persisted and reads back byte-faithful to the computed one.
    hp = paths.holistic_json(cfg.DATASET_ID, tmp_path)
    assert hp.exists()
    assert store.load_holistic(cfg.DATASET_ID, root=tmp_path) == agent_holistic.holistic_review()

    # Calibration table: header + one row per cluster, and it names the flagged call.
    cal = paths.calibration_md(cfg.DATASET_ID, tmp_path).read_text()
    assert cal.startswith("| Cluster")
    assert len([ln for ln in cal.splitlines() if ln.startswith("| c")]) == len(cfg.CLUSTER_ORDER)
    assert "TRUE" in cal  # at least one verify=TRUE call is present (anti-rubber-stamp)

    # Both are recorded as manifest artifacts (self-describing tree).
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert "interp/holistic.json" in man["artifacts"]
    assert "interp/calibration.md" in man["artifacts"]
