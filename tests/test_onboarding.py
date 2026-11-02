"""The loading screen states numbers, so the numbers have to be DERIVED, and this
study's.

``ui/onboarding._run_progress`` ended on the literal
``"Interpretation ready. 9 clusters, 280-gene panel."`` -- the demo's cluster count
and the demo's panel size, shown on every study's hand-off. That literal was replaced
by a read of the study's ``manifest.json`` (``clusters`` / ``panel_gene_count``),
which is per-study and still an assertion: a stored count is a CLAIM about the files,
written once at build time, and a manifest built before the control-probe filter says
13 on a study whose panel file holds 12 -- while every caption under every plot on the
same screen says 12. CLAUDE.md: panel size is dynamic across studies, derive it from
the file, never assert a constant.

Four things are pinned, in descending order of what they buy:

1. **A stored count does not get to speak.** The manifest is made to disagree with
   the files, and the line must follow the FILES. This is the half that bites: every
   study on disk has a manifest that agrees with its own files, so a test using only
   those would pass just as well on the version this replaces.
2. The two real studies' lines are each their own, checked against their input files
   read directly here -- not against the expression the implementation uses.
3. The two lines differ, which the literal would also have failed.
4. An underivable number is left out of the sentence rather than defaulted, down to
   the bare "Interpretation ready."; and the count is never borrowed from the bundled
   demo's marker table by a study that has none of its own.

Every tree this file writes is built under ``tmp_path``; nothing is written under
``data/studies/``.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from panoscope.agent import config as cfg
from panoscope.agent.data import is_control
from panoscope.data.study import Study
from panoscope.ui import onboarding

DEMO = "xenium_hbreast_sample1"
FIXTURE = "synthetic_multisample_fixture"


def _ready_for(study: Study | str) -> str:
    with cfg.use_study(study):
        return onboarding._ready_line()


def _ground_truth(store: Study) -> tuple[int, int]:
    """(clusters, panel genes) read straight off this study's input files.

    Deliberately NOT ``cfg.CLUSTER_ORDER`` / ``da.panel_names()``: asserting the
    implementation against the expression the implementation evaluates would pin
    nothing. These are the files, read here.

    Control probes are dropped through the platform adapter's own list, because a
    panel gene count that counts them is the error ``agent/data.load_panel`` exists
    to fix -- and the fixture's ``panel.parquet`` holds exactly one such row, which
    is what makes a manifest "built before the control filter" reproducible here.
    """
    markers = pd.read_csv(store.inputs / "markers_top.csv")
    clusters = markers.loc[markers["top_cluster"] != "NoSig", "top_cluster"]
    panel = pd.read_parquet(store.inputs / "panel.parquet")["gene"].astype(str)
    return clusters.nunique(), int((~panel.map(is_control)).sum())


@pytest.mark.parametrize("study_id", [DEMO, FIXTURE])
def test_the_ready_line_carries_the_open_studys_own_numbers(study_id: str) -> None:
    store = Study.open(study_id)
    if not store.exists:
        pytest.skip(f"study {study_id!r} is not on disk")
    n_clusters, n_panel = _ground_truth(store)
    assert _ready_for(study_id) == (
        f"Interpretation ready. {n_clusters} clusters, {n_panel}-gene panel."
    )


def test_the_two_studies_do_not_get_the_same_line() -> None:
    """The half the literal failed: one sentence, shown on both studies."""
    for study_id in (DEMO, FIXTURE):
        if not Study.open(study_id).exists:
            pytest.skip(f"study {study_id!r} is not on disk")
    assert _ready_for(DEMO) != _ready_for(FIXTURE)


def _study_with_manifest(tmp_path: Path, manifest_patch: dict, *, inputs=True) -> Study:
    """A copy of the fixture under ``tmp_path``, with its manifest edited.

    Copied rather than pointed at, so the edited manifest never touches the repo;
    ``Study(id=..., data_root=tmp_path)`` carries the root explicitly, so no
    environment variable and no ``layout`` cache is involved.
    """
    src = Study.open(FIXTURE)
    dst = tmp_path / "studies" / "edited"
    shutil.copytree(src.dir, dst)
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(manifest_patch)
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if not inputs:
        shutil.rmtree(dst / "inputs")
    return Study(id="edited", data_root=tmp_path)


def test_a_stored_count_does_not_get_to_speak(tmp_path: Path) -> None:
    """THE discriminating input: the manifest disagrees with the files.

    A manifest built before the control-probe filter says 13 where the panel file
    holds 12, and a manifest written before a cluster was merged over-counts the
    clusters. A reader of ``panel_gene_count`` / ``clusters`` prints 13 and 5 here --
    numbers no file on disk holds, under a screen whose own plot captions say 12.

    13 is not an arbitrary wrong number: the fixture's ``panel.parquet`` really does
    hold 13 rows, one of them a control probe, so a manifest written by a run that
    counted rows rather than genes says 13 -- the exact shape this guards.
    """
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(
        tmp_path, {"panel_gene_count": 13, "clusters": ["c1", "c2", "c3", "c4", "c5"]}
    )
    n_clusters, n_panel = _ground_truth(store)
    assert (n_clusters, n_panel) == (3, 12)  # the files, and they differ from the claim
    assert _ready_for(store) == "Interpretation ready. 3 clusters, 12-gene panel."


def test_an_underivable_number_is_left_out_rather_than_defaulted(tmp_path: Path) -> None:
    """``unknown`` is first-class and the sentence shrinks around it.

    With no inputs of its own the study can derive neither number, and the line must
    shrink to the bare statement -- NOT borrow the bundled demo's 9 clusters and 280
    genes, which is what a reader that falls back to the legacy flat files does.
    """
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(
        tmp_path, {"panel_gene_count": 13, "clusters": ["c1", "c2", "c3"]}, inputs=False
    )
    assert _ready_for(store) == "Interpretation ready."


def test_one_unreadable_file_costs_only_its_own_half_of_the_sentence(tmp_path: Path) -> None:
    """The two numbers are read independently: a missing panel file must not take the
    cluster count down with it, and vice versa."""
    if not Study.open(FIXTURE).exists:
        pytest.skip(f"study {FIXTURE!r} is not on disk")
    store = _study_with_manifest(tmp_path, {})
    (store.inputs / "panel.parquet").unlink()
    assert _ready_for(store) == "Interpretation ready. 3 clusters."
