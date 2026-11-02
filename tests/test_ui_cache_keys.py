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


@pytest.fixture(autouse=True)
def _render_against_a_copy(app_tree):
    """Every AppTest in this module renders a real study, and the app writes as it
    renders (``chat_threads.json``, ``review_state.json``). Render a copy, so the
    committed tree stays the read-only input the suite says it is."""

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


def test_unit_build_is_keyed_on_the_study_tree_not_its_id(tmp_path, monkeypatch) -> None:
    """``evidence_table._builds`` held an ``lru_cache`` keyed on the study ID, so two
    stores holding the same id under different data roots shared ONE entry: the second
    tree's rows would render the first tree's builds, and a build is where every row's
    MEASUREMENT STATE comes from -- the difference between "tested and not reported,
    absence IS evidence here" and "below its floor, absence argues nothing". Keyed on
    the tree path, the two stores are two entries."""
    from panoscope.pipeline import records
    from panoscope.ui import evidence_table as et

    seen: list[str] = []

    def _fake_build_all(study):
        seen.append(str(study.dir))
        return {"c1": f"build from {study.dir}"}

    monkeypatch.setattr(records, "build_all", _fake_build_all)
    one = Study(id="same_id", data_root=tmp_path / "root_one")
    two = Study(id="same_id", data_root=tmp_path / "root_two")
    et._builds.cache_clear()
    try:
        with cfg.use_study(one):
            a = et.unit_build("c1")
        with cfg.use_study(two):
            b = et.unit_build("c1")
        assert a and b and a != b
        assert len(seen) == 2  # a real miss on the second store, not a shared hit
        with cfg.use_study(one):
            assert et.unit_build("c1") == a and len(seen) == 2  # still cached per store
    finally:
        et._builds.cache_clear()  # the cache is module-global; leave it as found


def _summary_edits_path(study_id: str):
    """Where the Summary page's saved edits live for ``study_id``."""
    from panoscope.pipeline import paths as pipeline_paths

    return pipeline_paths.summary_edits_json(study_id)


def test_switching_studies_does_not_carry_summary_prose_into_the_next_tree() -> None:
    """The worst shape in this area, pinned on the REAL switch path.

    The Summary page's working-space keys (``_sum_seeded``, ``wsval_*``,
    ``_sum_saved_snapshot``) were study-agnostic. Switch study inside one session and
    the seeding block is skipped (``_sum_seeded`` is already set), so every region
    still holds study A's prose; every region then differs from study B's defaults,
    and the autosave persists **A's text into B's tree as biologist-authored edits**.
    Not a display bug: one study's prose written into another study's record.

    The switch is driven the way a picker would drive it -- a second script run of the
    same session with a different active study -- because the keys only collide across
    RERUNS, which is exactly what a hand-built session dict would fail to reproduce.

    Both halves are asserted: B's region must hold B's own draft (the visible half),
    and B's tree must still have no ``summary_edits.json`` (the half that persists).
    """
    import pytest

    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    from panoscope.ui import summary

    b_edits = _summary_edits_path(FIXTURE)
    if b_edits.exists():
        pytest.skip("a real saved edit is on the second study; not clobbering it")

    # The body runs as its own script (no closure over this frame): the active study
    # comes from session state, which is how an in-app picker would set it.
    def _run(first: str = "") -> None:
        import streamlit as _st

        from panoscope.agent import config as _cfg
        from panoscope.ui import state as _state
        from panoscope.ui import summary as _summary

        _state.init_state()
        with _cfg.use_study(_st.session_state.get("picked_study") or first):
            _summary.render_summary_page()

    at = AppTest.from_function(_run, kwargs={"first": DEMO})
    at.run(timeout=180)
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]

    with cfg.use_study(DEMO):
        a_key = summary._skey("wsval_caveats")
    with cfg.use_study(FIXTURE):
        b_key = summary._skey("wsval_caveats")
    planted = "STUDY A PROSE — MUST NOT REACH STUDY B"
    at.session_state[a_key] = planted          # as a keystroke in A's editor would
    at.session_state["picked_study"] = FIXTURE  # the switch

    try:
        at.run(timeout=180)
        assert not at.exception, [str(e.value) for e in at.exception]
        # the half that persists, asserted FIRST: nothing A's biologist typed may
        # land in B's tree as B's edits.
        assert not b_edits.exists(), f"the switch wrote {b_edits}"
        b_text = at.session_state[b_key]
        assert planted not in b_text
        # and it is B's OWN draft, not merely "not A's": the ground truth is what a
        # FRESH session opened straight on B seeds, read off the same page rather than
        # re-derived here (a re-derivation would be this page's logic written twice).
        fresh = AppTest.from_function(_run, kwargs={"first": FIXTURE})
        fresh.run(timeout=180)
        assert not fresh.exception, [str(e.value) for e in fresh.exception]
        assert b_text == fresh.session_state[b_key]
        assert at.session_state[a_key] == planted  # A's own edit is still A's
    finally:
        if b_edits.exists():
            b_edits.unlink()


def test_a_plot_caption_never_carries_another_studys_conditionality() -> None:
    """The same bug one layer up, where it reaches pixels instead of a number.

    ``conditionality._manifest`` and ``._density_index`` were zero-argument
    ``lru_cache(maxsize=1)`` readers: one slot for every study. Draw a figure on the
    demo, switch, and the fixture's plots carried the DEMO's artifact digest and the
    demo's qv floor and control-probe exclusions -- for an artifact the fixture does
    not have and a density index it never produced. ``_panel_scope()`` stayed correct
    per study (12 vs 280), which made it worse rather than better: the caption read as
    internally consistent, so nothing in it invited doubt. CLAUDE.md: a plot with no
    conditionality record refuses to draw -- this one drew with someone else's.

    The order is the one that breaks it (demo first, fixture second) and the switch is
    the real one, with no cache cleared by hand: clearing by hand is precisely what
    production does not do.
    """
    from panoscope.ui import conditionality as cnd

    with cfg.use_study(DEMO):
        demo_sha = cnd._sha("viz/cells.csv")
        demo_excl = cnd._density_exclusions()
        demo_scope = cnd._panel_scope()
    # the demo really does carry both, or the switch below proves nothing
    assert "sha256" in demo_sha and "qv >=" in demo_excl

    with cfg.use_study(FIXTURE):
        assert cnd._panel_scope() != demo_scope
        # ground truth: the fixture's own manifest names no artifacts and its
        # viz/hexbin holds no _index.json, so both fragments must be empty
        assert cnd._sha("viz/cells.csv") == ""
        assert cnd._density_exclusions() == ""
        # and on the records the stages actually build, not just the fragments
        cell_map = cnd.cell_map_cond(
            cluster="c1", n_total=12, n_selected=4, n_bg_drawn=8, n_bg_total=8
        )
        density = cnd.gene_density_cond(gene="SYNG01", hb=None, bin_um=50)
        assert cell_map.provenance == "" and density.exclusions == ""
        assert demo_sha not in "\n".join(cell_map.caption())
        assert demo_excl not in "\n".join(density.caption())

    # and the switch back is the demo's own values, not the fixture's leftovers
    with cfg.use_study(DEMO):
        assert cnd._sha("viz/cells.csv") == demo_sha
        assert cnd._density_exclusions() == demo_excl
