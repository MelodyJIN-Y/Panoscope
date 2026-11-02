"""Path resolution is study-local: no study ever reads the bundled demo's files.

The companion to ``tests/test_study_plurality.py``, for the one leak that file does
not watch. Both assertions below correspond to a bug that was real and reproduced,
not hypothetical:

- ``agent.data`` resolved every artifact tree-first with an UNCONDITIONAL fall back
  to the flat legacy files under ``data/``. Those files are the bundled demo's. A
  pipeline-built study that has no ``inputs/cluster_key.json`` silently read the
  demo's, and on a machine where the prep scripts have run it also read the demo's
  cells, UMAP, expression matrix and hexbins -- every plot, with a conditionality
  record naming the wrong study.
- ``data.load_cluster_key`` was a SECOND resolver for a fact ``Study.unit_labels``
  already owned, and it knew only one of that resolver's two sources. So
  ``data.cell_type_for`` handed another study's breast label to
  ``discriminate.call_A`` (-> rival generation -> the band) and to the model via
  ``tools.py``. ``test_study_plurality.py`` stayed green because it asserts on
  ``cfg.CLUSTER_KEY``, which takes the ``Study.unit_labels`` route.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from panoscope.agent import annotation, config as cfg, data
from panoscope.data.study import Study
from panoscope.ui import data_access as da

DEMO = "xenium_hbreast_sample1"
MINI = "demo_mini"

# Every zero-arg path helper that resolves against the active study.
_RESOLVERS = (
    data._markers_top,
    data._panel_parquet,
    data._cluster_key_json,
    data._cells_parquet,
    data._cells_csv,
    data._umap_parquet,
    data._umap_csv,
    data._density_dir,
    da._marker_expr_parquet,
    da._marker_expr_csv,
    da._density_index,
)


@pytest.fixture(scope="module", autouse=True)
def _both_studies_present():
    for sid in (DEMO, MINI):
        if not Study.open(sid).exists:
            pytest.skip(f"study {sid!r} is not on disk")


def test_no_study_resolves_into_another_studys_tree():
    """A second study's artifacts resolve under ITS OWN directory, present or not.

    A missing file must be a loud miss naming the study's own path, never a silent
    hit on the bundled demo's. Only the demo may use the flat legacy files.
    """
    with cfg.use_study(MINI):
        mini_dir = cfg.current_study().dir
        for fn in _RESOLVERS:
            got = fn()
            assert isinstance(got, Path)
            assert mini_dir in got.parents, f"{fn.__name__} -> {got} (outside {mini_dir})"


def test_cluster_key_is_the_studys_own_on_every_unit():
    """One fact, one resolver. Checked on EVERY unit of BOTH studies, not one."""
    for sid in (DEMO, MINI):
        with cfg.use_study(sid):
            study = cfg.current_study()
            units = cfg.CLUSTER_ORDER
            assert units, sid
            for unit in units:
                assert data.cell_type_for(unit) == study.label_for(unit) == \
                    annotation.cell_type_for(unit), f"{sid}/{unit}"


def test_cluster_key_entry_count_follows_the_study():
    with cfg.use_study(DEMO):
        assert len(data.load_cluster_key()) == 9
    with cfg.use_study(MINI):
        assert len(data.load_cluster_key()) == 3
