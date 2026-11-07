"""Pipeline slice 1: verdict serialization round-trip + a tmp-tree smoke test.

The round-trip equality is the correctness gate for reading verdicts off disk
instead of recomputing them: a persisted verdict must reload to an object EQUAL
to the freshly computed one. The smoke test runs the whole pipeline into a tmp
dir (never the real data/studies tree) and checks the artifacts + faithful read.
"""

from __future__ import annotations

import csv
import json

import pandas as pd

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import enrichment as agent_enrichment
from panoscope.agent import enrichment_themes as agent_enrichment_themes
from panoscope.agent import holistic as agent_holistic
from panoscope.pipeline import v1_verdicts

from panoscope.pipeline import paths
from panoscope.pipeline import run as pipeline_run
from panoscope.pipeline import store
from panoscope.pipeline.serialize import (
    enrichment_from_dict,
    enrichment_to_dict,
    holistic_from_dict,
    holistic_to_dict,
    pathway_themes_from_dict,
    pathway_themes_to_dict,
    verdict_from_dict,
    verdict_to_dict,
)
from panoscope.pipeline.stages.validate import validate

_HAS_ENRICHMENT = agent_enrichment._jz_enrichment_csv().exists()


def test_verdict_roundtrip_equals_computed():
    for c in cfg.CLUSTER_ORDER:
        v = v1_verdicts.verdict_for(c)
        assert verdict_from_dict(verdict_to_dict(v)) == v


def test_verdict_roundtrip_through_json():
    for c in cfg.CLUSTER_ORDER:
        v = v1_verdicts.verdict_for(c)
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
        assert loaded == v1_verdicts.verdict_for(c)

    assert store.load_all_verdicts(cfg.DATASET_ID, root=tmp_path) == [v1_verdicts.verdict_for(c) for c in cfg.CLUSTER_ORDER]


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


def _fresh_study(tmp_path, monkeypatch) -> None:
    """A data root holding one study with the demo's tidy inputs and no labels."""
    import shutil

    from panoscope.data import layout

    demo = paths.inputs_dir(cfg.BUNDLED_DEMO_ID)
    study = tmp_path / "studies" / "fresh"
    (study / "inputs").mkdir(parents=True)
    shutil.copyfile(demo / "panel.parquet", study / "inputs" / "panel.parquet")
    # No cluster_key, no annotation, and no cell_type column: that column is a label too.
    pd.read_csv(demo / "markers_top.csv").drop(columns="cell_type").to_csv(
        study / "inputs" / "markers_top.csv", index=False)
    (study / "manifest.json").write_text(json.dumps(
        {"dataset_id": "fresh", "tissue": "human breast cancer", "platform": "Xenium"}))
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path))
    layout.clear_cache()


def _no_band_anywhere(cluster: str) -> None:
    """Chat, Summary and report say "not yet called" with no confidence word and no re-check."""
    from panoscope.agent import fallback, skeptic
    from panoscope.ui import report, summary

    for r in (fallback.fallback_opening(cluster), fallback.match("how confident?", cluster),
              fallback.generic_fallback(cluster)):
        assert "not yet called" in r.text and not r.verify, r.text
        assert "confidence" not in r.text and "re-check this" not in r.text, r.text
    v = summary.da.composed_verdicts()[0]
    head = summary._ws_head_html(v.cluster, v.cell_type, v.confidence, v.verify)
    assert "re-check" not in head and 'class="cf' not in head and summary._conf_badge("") == ""
    rep = report.build_report_from_sources()
    text = report.default_cluster_summary(rep.sections[0]) + "".join(
        t for k, t in report._section_lines(rep) if k != "meta")  # meta: study-wide counts
    assert "not yet called" in text
    assert "confidence" not in text.lower() and "re-check" not in text
    edited = report._working_lines([(v.cluster, v.cell_type, v.confidence, v.verify, "x")],
                                   dataset="d", generated_at="")
    assert [t for k, t in edited if k == "h2"] == [f"{v.cluster} · {v.cell_type}"]
    sk = skeptic.second_opinion(cluster)  # the live tool: nothing to challenge, no verify
    assert not sk.challenges and "no band" in sk.verdict_line, sk.verdict_line


def test_a_study_with_no_labels_builds_with_every_cluster_not_yet_called(tmp_path, monkeypatch):
    """No label set is the starting state: the build passes and the biologist calls."""
    from panoscope.agent import loop as agent_loop
    from panoscope.data.study import NOT_YET_CALLED, Study
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.subject import EntityKind

    _fresh_study(tmp_path, monkeypatch)
    asked: list[str] = []

    def no_model(msg, cluster=None, **_):
        asked.append(cluster)
        raise RuntimeError("no model key")

    monkeypatch.setattr(agent_loop, "chat", no_model)
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: True)
    pipeline_run.run("fresh")
    clusters = json.loads(paths.manifest_json("fresh").read_text())["clusters"]
    assert len(clusters) == 9 and len(asked) == 9
    for c in clusters:
        v = store.load_verdict(c, "fresh")
        assert v.cell_type == NOT_YET_CALLED, c
        # No label, no claim: no band, no score anchor, nothing to re-check.
        assert (v.confidence, v.confidence_score, v.verify) == ("", None, False), c
        assert "confidence" not in v.notes.lower() and "re-check this" not in v.notes
    rows = list(csv.DictReader(paths.verdicts_csv("fresh").open()))
    assert {(r["confidence"], r["confidence_score"], r["verify"]) for r in rows} == {
        ("", "", "FALSE")}
    recs = list((paths.interp_dir("fresh") / "records" / "assignment").glob("*.json"))
    assert len(recs) == 9 and all('"band:not_called"' in p.read_text() for p in recs)
    assert store.load_review_state("fresh") == {}  # nothing signed off: every unit OPEN
    assert EntityKind.CELL_TYPE not in {f.kind for f in XeniumAdapter().frames(Study.open("fresh"))}
    with cfg.use_study("fresh"):  # the chat's opening reads the label; none is not a crash
        assert data.cell_type_for("c1") == NOT_YET_CALLED
        _no_band_anywhere("c1")

    pipeline_run.run("fresh")  # the proposal already ran; a rebuild does not re-ask
    assert len(asked) == 9


def test_a_proposal_written_during_the_build_is_what_that_build_reads(tmp_path, monkeypatch):
    """Validate caches the (empty) label set first; the build must not serve it stale."""
    from types import SimpleNamespace

    from panoscope.agent import loop as agent_loop

    _fresh_study(tmp_path, monkeypatch)
    body = json.dumps({"cell_type": "T cell", "cell_type_short": "T", "category": "Immune",
                       "lineage": "Lymphoid", "canonical_markers": ["CD3E"]})
    monkeypatch.setattr(agent_loop, "chat", lambda *a, **k: SimpleNamespace(text=body))
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: True)
    pipeline_run.run("fresh")
    clusters = json.loads(paths.manifest_json("fresh").read_text())["clusters"]
    assert {store.load_verdict(c, "fresh").cell_type for c in clusters} == {"T cell"}


def test_with_no_model_key_the_build_asks_nothing_until_a_key_exists(tmp_path, monkeypatch):
    """No key: no proposal call, and nothing written that would stop a later build asking."""
    from panoscope.agent import loop as agent_loop

    _fresh_study(tmp_path, monkeypatch)
    asked: list[str] = []
    monkeypatch.setattr(agent_loop, "chat", lambda m, cluster=None, **_: asked.append(cluster))
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: False)
    pipeline_run.run("fresh")
    assert asked == []
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: True)
    pipeline_run.run("fresh")
    assert len(asked) == 9


def test_a_de_only_folder_builds_without_marker_inputs(tmp_path, monkeypatch):
    """No marker table: only the DE workflow is available, and only it builds."""
    import shutil

    import pandas as pd

    from panoscope.data import layout

    demo = paths.inputs_dir(cfg.BUNDLED_DEMO_ID)
    inputs = tmp_path / "studies" / "deonly" / "inputs"
    inputs.mkdir(parents=True)
    shutil.copyfile(demo / "panel.parquet", inputs / "panel.parquet")
    g1, g2 = pd.read_parquet(demo / "panel.parquet")["gene"][:2]
    (inputs / "de_seurat.csv").write_text(
        f"gene,cluster,avg_log2FC,p_val_adj\n{g1},c1,2.0,1e-4\n{g2},c2,1.1,1e-3\n")
    (inputs.parent / "manifest.json").write_text(json.dumps(
        {"dataset_id": "deonly", "tissue": "human breast cancer", "platform": "Xenium"}))
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path))
    layout.clear_cache()

    pipeline_run.run("deonly")
    man = json.loads(paths.manifest_json("deonly").read_text())
    assert man["workflows"] == ["de-interpretation"] and man["clusters"] == ["c1", "c2"]
    assert man["inputs"]["de_seurat.csv"]["file"] == "de_seurat.csv"
    recs = paths.interp_dir("deonly") / "records"
    assert sorted(p.name for p in recs.rglob("*.json")) == ["c1.json", "c2.json"]
    assert (recs / "comparison" / "de_seurat").is_dir()
    assert not paths.verdicts_csv("deonly").exists()


def test_an_enrichment_only_folder_builds_with_its_clusters_not_yet_called(tmp_path, monkeypatch):
    """No marker table and no labels: the enrichment slice still builds."""
    import shutil

    from panoscope.data import layout

    src = layout.studies_root() / "synthetic_multisample_fixture" / "inputs"
    inputs = tmp_path / "studies" / "eo" / "inputs"
    inputs.mkdir(parents=True)
    for name in ("panel.parquet", "enrichment.csv"):
        shutil.copyfile(src / name, inputs / name)
    (inputs.parent / "manifest.json").write_text(json.dumps(
        {"dataset_id": "eo", "tissue": "t", "platform": "Xenium"}))
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path))
    layout.clear_cache()

    pipeline_run.run("eo")
    man = json.loads(paths.manifest_json("eo").read_text())
    assert man["workflows"] == ["enrichment-interpretation"]
    assert man["clusters"] == ["c1", "c2", "c3"]
