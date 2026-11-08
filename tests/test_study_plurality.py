"""The plurality gate: two studies, one process, no leakage between them.

This is the cheap half of the architecture's decision point, reduced to assertions
that run in CI. Before this landed, none of it was possible: ``config.py`` read three
dataset files at import and raised, and the path it used for the cluster key is
gitignored and absent, so a second study could not be opened at all.

Each assertion below corresponds to a bug that was real and verified, not a
hypothetical:

- the readers resolved every path at import, so the second study served the first
  study's frames
- the zero-argument readers were ``lru_cache(maxsize=1)`` -- one slot, shared
- ``load_annotation(None)`` cached on the literal ``None``, so switching study
  returned the previous study's annotation
- ``canonical_markers`` fell back to a BREAST marker map for any study, and those
  markers feed driver selection and therefore the confidence band
- the holistic review emitted a breast-specific pDC refinement for any study with a
  unit called Dendritic
- ``load_cluster_key`` required exactly nine entries

The demo and the fixture study are both breast-derived, so tissue is deliberately
NOT used as a discriminator here -- unit counts, labels and markers are.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from panoscope.agent import annotation, config as cfg, data, holistic
from panoscope.data.study import Study

REPO_ROOT = Path(__file__).resolve().parent.parent

DEMO = "xenium_hbreast_sample1"
MINI = "demo_mini"


@pytest.fixture(scope="module", autouse=True)
def _both_studies_present():
    for sid in (DEMO, MINI):
        if not Study.open(sid).exists:
            pytest.skip(f"study {sid!r} is not on disk")


@pytest.mark.req("ARCH-4-03")
def test_both_studies_open_in_one_process():
    a, b = Study.open(DEMO), Study.open(MINI)
    assert len(a.units) == 9, a.units
    assert len(b.units) == 3, b.units
    assert a != b and hash(a) != hash(b)


def test_readers_are_not_shared_between_studies():
    """The maxsize=1 bug: the second study used to receive the first study's frames."""
    demo_markers = data.load_markers()
    with cfg.use_study(MINI):
        mini_markers = data.load_markers()
        assert len(mini_markers) != len(demo_markers)
        assert set(mini_markers["top_cluster"]) <= {"c1", "c2", "c3"}
    assert len(data.load_markers()) == len(demo_markers), "switching back must restore"


def test_unit_order_follows_the_active_study():
    with cfg.use_study(MINI):
        assert len(cfg.CLUSTER_ORDER) == 3
        assert cfg.DATASET_ID == MINI
    assert len(cfg.CLUSTER_ORDER) == 9
    assert cfg.DATASET_ID == DEMO


@pytest.mark.req("ARCH-4-03")
def test_no_breast_marker_leak_into_another_study():
    """The most consequential leak: wrong-tissue curated knowledge moving a band.

    ``canonical_markers`` feeds ``verdict._is_canonical`` -> driver selection -> the
    confidence band. A study that happens to name a unit "Tumor" must not inherit
    ERBB2/EPCAM/KRT8/... from the bundled breast map. Empty means "this study says
    nothing", which is honest; it must never mean "use another tissue's markers".
    """
    assert annotation.canonical_markers("Tumor"), "the demo itself must still resolve"
    with cfg.use_study(MINI):
        assert annotation.canonical_markers("Tumor") == ()
        assert annotation.offpanel_canonical("Stromal") == ()


def test_no_demo_refinement_leak_into_another_study():
    """The holistic review's pDC refinement is breast content, gated on the demo."""
    assert len(holistic.holistic_review().refinements) == 1
    with cfg.use_study(MINI):
        assert holistic.holistic_review().refinements == ()


def test_annotation_is_not_cached_across_studies():
    demo_labels = {c: m.get("cell_type") for c, m in annotation.load_annotation().items()}
    with cfg.use_study(MINI):
        mini_labels = {c: m.get("cell_type") for c, m in annotation.load_annotation().items()}
        assert mini_labels != demo_labels
        assert len(mini_labels) == 3
    assert {c: m.get("cell_type") for c, m in annotation.load_annotation().items()} == demo_labels


def test_cluster_key_does_not_require_nine_entries():
    """The hard blocker. It also hid a second bug: validate.py caught the resulting
    ValueError as a bare Exception, so a 3-unit study silently skipped validation."""
    man = json.loads((Study.open(MINI).dir / "manifest.json").read_text(encoding="utf-8"))
    with cfg.use_study(MINI):
        assert sorted(cfg.CLUSTER_KEY) == sorted(man["clusters"]) and len(man["clusters"]) == 3


def test_palette_covers_any_unit_count():
    from panoscope.ui import format as fmt

    with cfg.use_study(MINI):
        palette = {c: fmt.cluster_color(c) for c in cfg.CLUSTER_ORDER}
        assert len(palette) == 3
        assert len(set(palette.values())) == 3, "units must not collapse to one colour"


def test_study_scope_is_restored_on_exception():
    """use_study must not leak its study when the block raises."""
    before = cfg.DATASET_ID
    with pytest.raises(RuntimeError):
        with cfg.use_study(MINI):
            raise RuntimeError("boom")
    assert cfg.DATASET_ID == before


def test_every_module_imports_under_a_switched_study():
    """Import the whole package in a fresh process with a non-demo study active.

    This exists because a real regression slipped past everything else. A module
    constant in agent/enrichment.py froze the study at import AND referenced a
    private config attribute removed during the Study refactor. The expression
    short-circuited under the bundled demo, so the demo-only suite stayed green
    while the entire Pathways slice was unimportable for every other study.

    The import-surface test imports each module once, under the demo. That is not
    enough: a study-conditional expression is only exercised when the study differs.

    A subprocess rather than importlib.reload -- reloading config rebinds its
    ContextVar, so tokens from the old one cannot be reset, and a reloaded module
    graph is not what a real run does anyway.
    """
    import subprocess
    import sys

    child = (
        "import importlib, pkgutil, json, sys\n"
        "import panoscope\n"
        "fails = []\n"
        "for m in pkgutil.walk_packages(panoscope.__path__, 'panoscope.'):\n"
        "    if '.app' in m.name:\n"
        "        continue\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "    except Exception as e:\n"
        "        fails.append(f'{m.name}: {type(e).__name__}: {e}')\n"
        "print(json.dumps(fails))\n"
    )
    env = {**os.environ, "PANOSCOPE_DATASET": MINI}
    proc = subprocess.run(
        [sys.executable, "-c", child],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"child failed:\n{proc.stderr[-2000:]}"
    failures = json.loads(proc.stdout.strip().splitlines()[-1])
    assert not failures, (
        f"{len(failures)} module(s) fail to import under study {MINI!r}:\n  "
        + "\n  ".join(failures)
    )
