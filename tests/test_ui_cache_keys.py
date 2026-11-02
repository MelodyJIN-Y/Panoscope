"""The UI read layer must never serve one study's values under another study.

``agent/data`` was keyed per study (``_per_study``); ``ui/data_access`` was not, so
its ``st.cache_data`` readers keyed the demo's 280-gene panel and the second study's
12-gene panel to the SAME cache entry. Whichever study was opened first won for the
life of the session:

    A: xenium_hbreast_sample1        panel 280   c1 "Tumor"
    B: synthetic_multisample_fixture panel 280   c1 "Tumor"      <- A's values

Nothing in the app can switch studies today (there is no picker), which is exactly
why this file exists: the defect is invisible until the picker lands, and then it is
a measured value from the wrong study under another study's plot. These tests fail
if that ever comes back.

``test_study_plurality.py`` pins the same class of bug one layer down, for the pure
loaders; nothing pinned it for the cached UI readers the pages actually call.

The fixture is the second study in shape only -- a dozen synthetic genes and made-up
labels -- which is all that is needed here: its numbers differ from the demo's, and
that difference is the whole assertion.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data as agent_data
from panoscope.data.study import Study
from panoscope.ui import data_access as da

DEMO = "xenium_hbreast_sample1"
FIXTURE = "synthetic_multisample_fixture"


@pytest.fixture(scope="module", autouse=True)
def _studies_present() -> None:
    for sid in (DEMO, FIXTURE):
        if not Study.open(sid).exists:
            pytest.skip(f"study {sid!r} is not on disk")


def test_panel_and_markers_are_not_shared_across_studies() -> None:
    """The headline bug: B's panel scope must be B's, and A must still be A's after."""
    with cfg.use_study(DEMO):
        a_panel, a_markers = da.panel_names(), len(da.markers_df())
    with cfg.use_study(FIXTURE):
        b_panel, b_markers = da.panel_names(), len(da.markers_df())
        # ground truth is the uncached loader reading THIS study's files
        assert b_panel == sorted(agent_data.load_panel()["gene"].astype(str))
        assert b_markers == len(agent_data.load_markers())
    assert b_panel != a_panel and b_markers != a_markers
    # and the switch BACK is a cache hit on A's own entry, not B's leftovers
    with cfg.use_study(DEMO):
        assert da.panel_names() == a_panel and len(da.markers_df()) == a_markers


def test_verdict_for_is_not_keyed_on_the_cluster_id_alone() -> None:
    """``verdict_for("c1")`` was keyed on "c1" and nothing else, so the second study's
    c1 came back as the first study's call -- a cell-type label from another tissue."""
    with cfg.use_study(DEMO):
        a = da.verdict_for("c1").cell_type
    with cfg.use_study(FIXTURE):
        b = da.verdict_for("c1").cell_type
    assert a and b and a != b
    with cfg.use_study(DEMO):
        assert da.verdict_for("c1").cell_type == a


def test_cache_shim_keys_on_the_study_tree_not_its_id(tmp_path) -> None:
    """The key is the study's DIRECTORY: two stores can hold the same study id under
    different data roots (the pipeline's own copy-to-tmp tests do exactly that), and
    keying on the id alone would merge them."""
    calls: list[str] = []

    @da._cache_data(show_spinner=False)
    def reader() -> str:
        calls.append(cfg.current_study().id)
        return str(cfg.current_study().dir)

    one = Study(id="same_id", data_root=tmp_path / "root_one")
    two = Study(id="same_id", data_root=tmp_path / "root_two")
    with cfg.use_study(one):
        first = reader()
    with cfg.use_study(two):
        second = reader()
    assert first != second
    assert len(calls) == 2  # a real miss on the second store, not a shared hit
    with cfg.use_study(one):
        assert reader() == first and len(calls) == 2  # still cached per store


def test_each_cached_reader_keeps_its_own_cache_namespace() -> None:
    """Streamlit derives a cache namespace from ``__module__``/``__qualname__`` plus the
    function source, so the shim's inner wrapper must carry the wrapped function's
    identity. Without that, every reader in ``data_access`` shares ONE namespace and
    ``panel_df()`` can return ``markers_df()``'s frame -- the same bug, one level worse."""

    @da._cache_data(show_spinner=False)
    def one() -> str:
        return "one"

    @da._cache_data(show_spinner=False)
    def two() -> str:
        return "two"

    assert (one(), two()) == ("one", "two")
    assert one.__name__ == "one" and two.__name__ == "two"
