"""Declarations survive a build, and nothing is defaulted in their place.

The manifest holds two kinds of field. Derived ones a build recomputes; declared ones
only the biologist can supply. Four of the declared ones block the study (§6.3), and
this module exists because every one of them had a path to a silently wrong value:

* ``tissue`` / ``platform`` were MODULE CONSTANTS in ``pipeline/run.py``
  (``"human breast cancer"``, ``"Xenium"``) handed straight to ``build_manifest``.
  Tissue is the corpus scoping key, so the kidney study would have been built, scored
  and literature-scoped as breast, with a manifest that said so confidently.
* ``samples`` / ``clustering_mode`` had no route through ``build_manifest`` at all and
  were rebuilt away on every run. Declaring a clustering mode therefore could not
  work: the first build ate the declaration and the second raised
  ``UndeclaredClustering`` against the file it had just emptied. Not declaring it was
  worse -- ``samples()`` fell through to one sample named after the store, and a
  single-sample store takes ``"independent"`` as a tautology, so a jointly clustered
  study got a clustering mode stamped on every record that nobody had declared.

The tests below are written so that each one fails for exactly one reason. The
preservation tests go through ``pipeline.run`` rather than calling ``build_manifest``
with a dict, because the defect was in the *round trip* -- build_manifest alone was
never the thing that lost the declaration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.pipeline import manifest as manifest_mod
from panoscope.pipeline import paths
from panoscope.pipeline import run as pipeline_run
from panoscope.pipeline.stages.validate import validate
from panoscope.platforms.base import UndeclaredSamples
from panoscope.platforms.xenium import XeniumAdapter
from panoscope.types.contracts import UndeclaredClustering

from tests.test_platforms import _store

ADAPTER = XeniumAdapter()

_BASE: dict = dict(
    dataset_id="s", generated_at="2026-01-01T00:00:00Z",
    tissue="human kidney", platform="Xenium", panel_gene_count=3, n_cells=10,
    clusters=["c1"], views_available={}, inputs={}, artifacts={},
)

_MARKER_CSV = "gene,top_cluster,glm_coef,pearson\nAAA,u1,1.0,0.5\n"


def _seed(root: Path, dataset_id: str, **declarations) -> Path:
    """Write a declaration-only manifest into the output tree, as a biologist would."""
    mp = paths.manifest_json(dataset_id, root)
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_text(json.dumps({"dataset_id": dataset_id, **declarations}, indent=2))
    return mp


def _with_sample_tables(store, *samples: str):
    """Give a store per-sample marker tables -- the store saying, in files, how many."""
    for s in samples:
        d = store.inputs / "samples" / s
        d.mkdir(parents=True)
        (d / "markers_top.csv").write_text(_MARKER_CSV)
    return store


# --------------------------------------------------------------------------- #
# 1. build_manifest carries the §6.3 declarations
# --------------------------------------------------------------------------- #
def test_samples_and_clustering_mode_travel_through_build_manifest() -> None:
    """They had no parameter at all, so a caller could not pass them if it tried."""
    man = manifest_mod.build_manifest(
        **_BASE, samples=["sampleA", "sampleB"], clustering_mode="joint")
    assert man["samples"] == ["sampleA", "sampleB"]
    assert man["clustering_mode"] == "joint"


def test_an_undeclared_sample_set_is_not_written_as_one_sample() -> None:
    """A build must not invent the declaration it failed to find.

    Writing ``samples: ["<study id>"]`` from a fallback would turn a guess into a
    record, and the next read of that manifest would find a declared single-sample
    study -- which takes ``"independent"`` clustering as a tautology.
    """
    man = manifest_mod.build_manifest(**_BASE)
    assert "samples" not in man
    assert "clustering_mode" not in man


def test_a_rebuild_preserves_every_declared_field() -> None:
    """Derived keys are recomputed; everything else in the old manifest survives."""
    previous = {
        "dataset_id": "s", "generated_at": "1999-01-01T00:00:00Z",
        "tissue": "stale", "platform": "stale",
        "samples": ["sampleA", "sampleB"], "clustering_mode": "joint",
        "fixation": "FFPE", "segmentation": "nucleus_expansion",
        "panel_gene_count": 99999, "n_cells": 1, "clusters": ["gone"],
        "artifacts": {"stale.json": {"sha256": "0"}},
    }
    man = manifest_mod.build_manifest(
        **_BASE, samples=previous["samples"],
        clustering_mode=previous["clustering_mode"], declared=previous)

    # Declared, and not derivable from any file: carried forward untouched.
    assert man["fixation"] == "FFPE"
    assert man["segmentation"] == "nucleus_expansion"
    assert man["samples"] == ["sampleA", "sampleB"]
    assert man["clustering_mode"] == "joint"

    # Derived: recomputed, and the stale values are gone rather than merged.
    assert man["panel_gene_count"] == 3
    assert man["clusters"] == ["c1"]
    assert man["artifacts"] == {}
    assert man["generated_at"] == "2026-01-01T00:00:00Z"
    assert man["tissue"] == "human kidney"
    assert set(manifest_mod.DERIVED_KEYS).isdisjoint(
        {"samples", "clustering_mode", "tissue", "platform"})


@pytest.mark.parametrize("field", ["tissue", "platform"])
def test_a_manifest_cannot_be_built_without_the_blocking_declarations(field) -> None:
    """The replacement for the module constant is a refusal, not another default."""
    with pytest.raises(ValueError, match=f"declares no {field}"):
        manifest_mod.build_manifest(**{**_BASE, field: ""})


# --------------------------------------------------------------------------- #
# 2. The round trip: a declaration survives an actual build (all nine units)
# --------------------------------------------------------------------------- #
def test_a_build_does_not_eat_the_biologists_declarations(tmp_path: Path) -> None:
    """The deadlock, reproduced end to end and then not reproduced.

    Declare the §6.3 pair, build, and they must still be there -- otherwise the next
    build raises ``UndeclaredClustering`` against a file this build emptied, which is
    the deadlock half of the defect, and removing the declaration to escape it is the
    silent half.
    """
    _seed(tmp_path, cfg.DATASET_ID, tissue="human breast cancer", platform="Xenium",
          samples=["sampleA", "sampleB"], clustering_mode="joint", fixation="FFPE")

    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())

    assert man["samples"] == ["sampleA", "sampleB"]
    assert man["clustering_mode"] == "joint"
    assert man["fixation"] == "FFPE"

    # ... and it is a real build of all nine units, not an early return.
    assert man["clusters"] == list(cfg.CLUSTER_ORDER) and len(man["clusters"]) == 9
    assert sum(1 for k in man["artifacts"] if k.startswith("interp/clusters/")) == 9
    assert man["panel_gene_count"] == 280

    # A SECOND build still has something to preserve -- the half that deadlocked.
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    again = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert again["clustering_mode"] == "joint"
    assert again["samples"] == ["sampleA", "sampleB"]


def test_tissue_comes_from_the_declaration_and_not_from_a_constant(tmp_path: Path) -> None:
    """The kidney check, run against the demo's own machinery.

    Everything else about this build is the breast demo. Only the declaration says
    kidney, and the manifest must agree with the declaration -- a constant would
    report breast here with no gap anywhere for a reader to notice.
    """
    _seed(tmp_path, cfg.DATASET_ID, tissue="human kidney", platform="Xenium",
          method="immuno-oncology panel")

    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())

    assert man["tissue"] == "human kidney"
    assert "breast" not in man["tissue"]
    assert man["method"] == "immuno-oncology panel"


def test_an_unseeded_build_reads_the_studys_own_declaration(tmp_path: Path) -> None:
    """With no manifest in the output tree, the study's own one is the declaration.

    This is the path every existing test takes, and it must still produce the demo's
    real tissue -- the fix removes a constant, it does not remove the value.
    """
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert man["tissue"] == "human breast cancer"
    assert man["platform"] == "Xenium"


# --------------------------------------------------------------------------- #
# 3. samples() refuses rather than defaulting
# --------------------------------------------------------------------------- #
def test_per_sample_tables_with_no_declared_samples_is_a_loud_failure(tmp_path: Path) -> None:
    """The store says two in files and the manifest says nothing. One is not an answer."""
    store = _with_sample_tables(
        _store(tmp_path, "multi",
               manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"]},
               panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)]),
        "sampleA", "sampleB")

    with pytest.raises(UndeclaredSamples) as exc:
        ADAPTER.samples(store)
    msg = str(exc.value)
    assert "sampleA, sampleB" in msg
    assert "manifest.json" in msg, "the biologist's next action is to open that file"
    assert "independent" in msg, "the message must name the wrong answer it refuses"

    # The blocker it is: callers that already render the fourth blocker see it.
    assert isinstance(exc.value, UndeclaredClustering)
    with pytest.raises(UndeclaredClustering):
        ADAPTER.clustering_mode(store)
    assert any("per-sample marker tables" in p for p in ADAPTER.validate(store))


def test_declaring_the_samples_unblocks_it(tmp_path: Path) -> None:
    """The refusal is conditional on the contradiction, not on having sample dirs."""
    store = _with_sample_tables(
        _store(tmp_path, "multi",
               manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"],
                         "samples": ["sampleA", "sampleB"],
                         "clustering_mode": "joint"},
               panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)]),
        "sampleA", "sampleB")
    assert ADAPTER.samples(store) == ["sampleA", "sampleB"]
    assert ADAPTER.clustering_mode(store) == "joint"
    assert not [p for p in ADAPTER.validate(store)
                if "sample" in p or "cluster" in p]


def test_a_store_with_no_per_sample_tables_still_counts_one(tmp_path: Path) -> None:
    """A flat store holding one sample is a count, and must not become a blocker.

    The guard fires on a contradiction; with nothing contradicting it the tautology
    stands, or every single-sample study in the repo stops dead.
    """
    store = _store(tmp_path, "flat",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"]},
                   panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)])
    assert ADAPTER.samples(store) == ["flat"]
    assert ADAPTER.clustering_mode(store) == "independent"


def test_sample_inputs_reads_the_directory_not_the_declared_list(tmp_path: Path) -> None:
    """It says WHERE the readings are, so the filesystem settles it.

    A manifest that under-declares its samples must not hide a table that is there;
    ``validate`` cross-checks the two, and it can only do that if they are two
    independent reads.
    """
    store = _with_sample_tables(
        _store(tmp_path, "multi",
               manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"],
                         "samples": ["sampleA"], "clustering_mode": "joint"},
               panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)]),
        "sampleA", "sampleB")
    assert set(ADAPTER.sample_inputs(store)) == {"sampleA", "sampleB"}
    assert ADAPTER.samples(store) == ["sampleA"]


# --------------------------------------------------------------------------- #
# 4. Stage 0 actually calls the adapter
# --------------------------------------------------------------------------- #
def test_stage_zero_runs_the_adapters_blockers(tmp_path: Path) -> None:
    """``XeniumAdapter.validate`` had no caller anywhere in the repo.

    Every §6.3 blocker it checks could therefore fail while the build ran to
    completion and wrote a manifest. This asserts Stage 0 stops instead, and names
    more than one problem -- a stage that reported the first and hid the rest is how
    a study gets fixed one round trip at a time.
    """
    store = _store(tmp_path, "blocked",
                   manifest={"platform": "Xenium", "clusters": ["u1"]},
                   markers=[("AAA", "u1", 1.0, 0.5)])  # no panel, no tissue

    with cfg.use_study(store):
        with pytest.raises(ValueError) as exc:
            validate("blocked")
    msg = str(exc.value)
    assert "no measured set" in msg, "the panel blocker must reach the build"
    assert "tissue is not declared" in msg
    assert msg.count("\n  - ") >= 2, "a checklist names every gap at once"


def test_stage_zero_validates_the_study_it_was_given(tmp_path: Path, demo_id) -> None:
    """``panoscope build --study X`` scopes nothing (``cli.py``), so this must.

    Every reader in this stage resolves the ACTIVE study and none has ever looked at
    the argument, so a build of one study could be cleared by a validation of
    another. Here the active study is a broken one and the argument names the clean
    demo: passing proves the argument won, and the broken active store was not what
    got checked.
    """
    broken = _store(tmp_path, "blocked",
                    manifest={"platform": "Xenium", "clusters": ["u1"]},
                    markers=[("AAA", "u1", 1.0, 0.5)])  # no panel, no tissue
    with cfg.use_study(broken):
        validate(demo_id)                      # the demo, not the broken active store
        with pytest.raises(ValueError, match="blocked"):
            validate("blocked")                # and the broken one still fails


def test_stage_zero_passes_on_the_real_demo(demo_id: str) -> None:
    """The gate must not fail closed on the study it ships with."""
    from panoscope import platforms
    from panoscope.data.study import Study

    store = Study.open(demo_id)
    assert platforms.for_study(store).validate(store) == []   # no blocking problem
    assert validate(demo_id) is None


# --------------------------------------------------------------------------- #
# 5. The enrichment stage's failures are not swallowed
# --------------------------------------------------------------------------- #
def test_a_failure_inside_the_enrichment_stage_fails_the_build(tmp_path, monkeypatch) -> None:
    """The v2 record composition runs inside that stage and raises this exact class.

    ``read_tidy``, ``adapter.frames`` and ``record.write`` all raise
    ``FileNotFoundError``, and the build used to catch it alongside "this study has no
    enrichment result" -- so a study whose records could not be written reported
    success with ``interp/records/`` absent.
    """
    if not pipeline_run._has_enrichment_result():
        pytest.skip("enrichment result not present")

    def boom(*a, **k):
        raise FileNotFoundError("the v2 record could not be written")

    monkeypatch.setattr(pipeline_run, "run_enrichment", boom)
    with pytest.raises(FileNotFoundError, match="v2 record"):
        pipeline_run.run(cfg.DATASET_ID, root=tmp_path)


def test_a_study_with_no_enrichment_result_still_builds(tmp_path, monkeypatch) -> None:
    """The other half: absence of the INPUT is a condition, not a failure."""
    monkeypatch.setattr(pipeline_run, "_has_enrichment_result", lambda: False)
    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)

    assert not paths.enrichment_csv(cfg.DATASET_ID, tmp_path).exists()
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    assert len(man["clusters"]) == 9  # the marker half built all nine regardless


def test_the_enrichment_records_are_written_and_hashed(tmp_path: Path) -> None:
    """What the swallowed exception made invisible: one v2 record per unit, hashed."""
    if not pipeline_run._has_enrichment_result():
        pytest.skip("enrichment result not present")

    pipeline_run.run(cfg.DATASET_ID, root=tmp_path)
    man = json.loads(paths.manifest_json(cfg.DATASET_ID, tmp_path).read_text())
    recorded = [k for k in man["artifacts"] if k.startswith("interp/records/")]
    assert len(recorded) >= 9, recorded
    for c in cfg.CLUSTER_ORDER:
        assert any(k.endswith(f"/{c}.json") for k in recorded), c


def test_every_file_that_can_move_a_band_is_hashed() -> None:
    """Provenance that omits the input the band comes from records nothing.

    ``interp/annotation.json`` is read-if-present -- ``_ensure_annotation``
    regenerates it only when absent -- so on the bundled demo, and on every
    rebuild after the first, it is an INPUT. ``records.build_all`` turns its
    ``canonical_markers`` into the ``canonical`` set ``core_verdict.assess``
    reads, and emptying one cluster's list moves c1 from Very High /
    "glm_coef direct" / verify=False to Low / "no canonical driver" /
    verify=True.

    It was hashed in neither ``inputs`` nor ``artifacts``, while
    ``inputs/cluster_key.json`` -- the FALLBACK source for the same labels --
    was.
    """
    mf = json.loads(
        (paths.dataset_dir(cfg.DATASET_ID) / "manifest.json").read_text()
    )
    recorded = set(mf.get("inputs", {})) | set(mf.get("artifacts", {}))

    ann = paths.interp_dir(cfg.DATASET_ID) / "annotation.json"
    assert ann.is_file(), "the demo study has no annotation.json to hash"
    assert "interp/annotation.json" in recorded, (
        "the file the band comes from is hashed nowhere in the manifest"
    )

    # ...and the hash is of the file that is actually there.
    assert (mf["artifacts"]["interp/annotation.json"]["sha256"]
            == manifest_mod.sha256_file(ann)), (
        "the recorded hash does not match the annotation on disk"
    )
