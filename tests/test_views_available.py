"""The spatial stage honours ``manifest.json``'s ``views_available``.

A study whose viz stage never ran -- ``demo_mini``, the multi-sample fixture, and the
kidney study when it arrives -- declares every view false in its manifest. Nothing in
the UI read that declaration, so the Markers page called ``cells_df()``, took the
loader's named ``FileNotFoundError`` and rendered a Streamlit traceback over the whole
page, evidence table included. The defect was loud rather than silently wrong, which
is why it is a surface defect and not a grounding one: a traceback is not an
interpretation, and Markers is the primary review surface.

What these pin, in descending order of what they buy:

1. **The guard only ever subtracts.** An absent key, an empty ``views_available`` and
   an unreadable manifest all read as *available*, exactly as
   ``PlatformAdapter.available_views`` has it: a manifest may veto a view whose files
   exist, it may never grant one whose files do not. A guard that could grant would
   let a manifest claim a figure into existence, and the loader would stop being the
   backstop.
2. **The absence reaches the pixel, and it names a producer.** Headless, through the
   real entry point, with a marker selected so the gene rows draw too -- a panel that
   says only "unavailable" teaches a biologist the tool is broken when it is being
   honest. An absent view is not an empty one: not a blank, not a zeroed plot.
3. **One unavailable view does not blank the page.** The stage still runs end to end
   and every panel states its own absence.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import panoscope
from panoscope.agent import config as cfg
from panoscope.agent import data as agent_data
from panoscope.data.study import Study
from panoscope.ui import data_access as da
from panoscope.ui import spatial_stage as stage

ALL_FALSE = {"cell_map": False, "umap": False, "density": False, "expr": False}


def _unprepped(root: Path, views: dict) -> Study:
    """A study that is nothing but a manifest -- no ``viz/`` tree at all.

    Built here rather than copied from ``data/studies/`` so the test owns its own
    tree: the real fixture must stay byte-identical on disk, and a study whose only
    file is its manifest is exactly the shape this guard exists for.
    """
    d = root / "studies" / "unprepped"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"dataset_id": "unprepped", "views_available": views}),
        encoding="utf-8",
    )
    return Study(id="unprepped", data_root=root)


# --------------------------------------------------------------------------- #
# 1. The guard only ever subtracts
# --------------------------------------------------------------------------- #
_DAMAGE = {
    "no_markers": lambda d: (d / "inputs" / "markers_top.csv").unlink(),
    "header_only_markers": lambda d: (d / "inputs" / "markers_top.csv").write_text(
        "gene,top_cluster,glm_coef,pearson,max_gc_corr,max_gg_corr\n"),
    "no_panel": lambda d: (d / "inputs" / "panel.parquet").unlink(),
    "no_manifest": lambda d: (d / "manifest.json").unlink(),
    "corrupt_manifest": lambda d: (d / "manifest.json").write_text("{ not json"),
}


@pytest.mark.parametrize("damage", sorted(_DAMAGE))
@pytest.mark.parametrize("page", ["summary", "markers", "pathways"])
def test_a_damaged_study_renders_a_message_not_a_traceback(tmp_path, monkeypatch,
                                                           damage, page):
    """A missing input is something to SAY, not something to crash on.

    Measured before this was fixed: with no ``inputs/markers_top.csv`` every page
    was 1075 characters whose entire content was a ``FileNotFoundError`` traceback
    -- no brand, no nav strip, nothing. ``cfg.CLUSTER_ORDER`` raised inside
    ``app._restore_selection_from_url``, which runs before any body container
    exists. A header-only marker table did the same through ``IndexError``.
    Separately, a missing ``panel.parquet`` or an absent/corrupt ``manifest.json``
    killed Summary outright and put a traceback on Markers from inside the very
    guard whose docstring says it "must never crash the app".

    ``nav.Tab.available`` already states the contract: "A missing or unreadable
    result is *no data*, never a traceback in the nav bar."

    This is the shape a half-finished ``panoscope build`` leaves behind, so it is
    also the first thing a new study hits.
    """
    from streamlit.testing.v1 import AppTest

    src = Path(cfg.DATA_DIR_PATH) / "studies" / cfg.BUNDLED_DEMO_ID
    if not src.is_dir():
        pytest.skip("the bundled demo study is not on disk")

    root = tmp_path / "root"
    study = root / "studies" / cfg.BUNDLED_DEMO_ID
    study.parent.mkdir(parents=True)
    shutil.copytree(src, study)
    _DAMAGE[damage](study)

    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    monkeypatch.setenv("PANOSCOPE_DATASET", cfg.BUNDLED_DEMO_ID)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    at = AppTest.from_file(
        str(Path(panoscope.__file__).parent / "app.py"), default_timeout=300)
    at.session_state["active_page"] = page
    at.run()

    assert not at.exception, (
        f"{page} raised on a study with {damage}: "
        f"{[str(e.value)[:160] for e in at.exception]}"
    )
    assert len(" ".join(m.value for m in at.markdown)) > 200, (
        f"{page} rendered essentially nothing for {damage}; it must SAY what is missing"
    )


def test_a_view_declared_false_is_named_and_a_declared_true_one_draws(tmp_path):
    with cfg.use_study(_unprepped(tmp_path, {"cell_map": False, "umap": True})):
        assert stage._unavailable("cell_map") == "cell_map"
        assert stage._unavailable("umap") is None
        # The dotplot needs two views; the first missing one is what gets named.
        assert stage._unavailable("umap", "cell_map") == "cell_map"


def test_an_undeclared_view_is_not_a_veto(tmp_path):
    """An absent key is silence, not a refusal -- the loader stays the backstop."""
    with cfg.use_study(_unprepped(tmp_path, {})):
        assert stage._unavailable("density") is None


def test_an_unreadable_manifest_is_not_a_veto(tmp_path):
    """``Study.views_available`` is ``{}`` when the manifest will not parse, and ``{}``
    grants nothing and vetoes nothing. A guard that failed closed here would hide every
    figure in a study whose manifest merely has a typo."""
    d = tmp_path / "studies" / "unprepped"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text("{ not json", encoding="utf-8")
    with cfg.use_study(Study(id="unprepped", data_root=tmp_path)):
        assert stage._unavailable("cell_map") is None


# --------------------------------------------------------------------------- #
# 2 + 3. The absence reaches the pixel, and the page survives it
# --------------------------------------------------------------------------- #
def test_the_stage_states_every_absent_view_instead_of_raising(tmp_path):
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    _unprepped(tmp_path, ALL_FALSE)

    def _run(root: str) -> None:
        import pathlib

        import streamlit as st

        from panoscope.agent import config as _cfg
        from panoscope.data.study import Study as _Study
        from panoscope.ui import spatial_stage as _stage

        # Set inside the script: the active study is a ContextVar, and the AppTest
        # script runs in its own thread.
        _cfg.set_study(_Study(id="unprepped", data_root=pathlib.Path(root)))
        # A marker selected, so the per-gene density + feature-UMAP rows and the
        # across-cluster dotplot all render rather than the "pick a marker" state.
        st.session_state["selected_markers"] = {"c1": ["GENE_A"]}
        _stage.render_spatial_stage("c1")

    at = AppTest.from_function(_run, default_timeout=120, args=(str(tmp_path),))
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]

    md = "\n".join(m.value for m in at.markdown)
    for label in ("cell map", "UMAP", "transcript density", "per-cell expression"):
        assert f"{label} not prepared for this study" in md, label
    # What would produce it -- the same prep step the loader's own error names.
    for producer in ("scripts/prep_data.R", "scripts/precompute_density.py",
                     "scripts/prep_expr.py"):
        assert producer in md, producer
    # A viewing control for pictures that do not exist is a dead knob.
    assert not [r for r in at.radio if r.key == "bin_um_global"]


# --------------------------------------------------------------------------- #
# 4. The half-prepped study: every view true except the one Panoscope adds
# --------------------------------------------------------------------------- #
# The shape above -- nothing prepared at all -- is the easy half. The likely arrival
# is the HALF-prepped one: ``prep_data.R`` ran (cells + UMAP exist) and
# ``prep_expr.py`` did not, because the expression matrix is Panoscope's own extra
# step rather than something a collaborator hands over. That manifest reads
# ``{cell_map: true, umap: true, expr: false}``, and a guard that asked only about
# ``umap`` let the feature-UMAP branch past it and into the loader -- the same
# full-page traceback, on the study most likely to turn up.
#
# Driven through ``panoscope/app.py`` rather than ``render_spatial_stage``, because
# the crash entered through ``app.py``'s ``_examine_body``: the renderer is reached
# only after the evidence table has drawn, and a traceback there takes the evidence
# table with it. Testing the renderer alone would report the page as covered while
# the page is what breaks.
def _half_prepped(root: Path) -> Path:
    """A study with cells + UMAP on disk and no exported expression.

    Its inputs (markers, panel) are copied from the bundled ``demo_mini`` so the
    Markers page has a result to draw and actually reaches the spatial stage; the
    ``viz/`` frames are written here, tiny and synthetic, because what is being
    pinned is which views exist, not what is in them. Nothing under
    ``data/studies/`` is touched -- the copy is one-way, into ``tmp_path``.
    """
    from panoscope.data import layout

    src = layout.studies_root() / "demo_mini"
    d = root / "studies" / "half_prepped"
    shutil.copytree(src, d)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "half_prepped"
    manifest["views_available"] = {
        "cell_map": True, "umap": True, "density": False, "expr": False,
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    viz = d / "viz"
    viz.mkdir()
    cluster = ["c1", "c2", "c3"] * 4
    xy = [float(i) for i in range(12)]
    pd.DataFrame(
        {"cell_id": range(12), "cluster": cluster, "x": xy, "y": xy}
    ).to_parquet(viz / "cells.parquet")
    pd.DataFrame(
        {"cell_id": range(12), "umap_1": xy, "umap_2": xy, "cluster": cluster}
    ).to_parquet(viz / "umap.parquet")
    return d


def test_a_prepared_umap_with_no_exported_expression_does_not_kill_the_markers_page(
    tmp_path, monkeypatch
):
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    _half_prepped(tmp_path)
    # Through the environment, not ``set_study``: ``AppTest.from_file`` runs the real
    # app script in its own thread, where the study ContextVar is back at its default
    # and the study is resolved from the environment exactly as in production.
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PANOSCOPE_DATASET", "half_prepped")
    # ``layout.resolve_data_root`` is an lru_cache, so setting the variable is not
    # enough: without this the app thread resolves the REAL data root and writes
    # ``data/studies/half_prepped/interp/chat_threads.json`` into the repository --
    # gitignored, so ``git status`` stays silent about a study id that does not
    # exist, holding the demo's interpretation. conftest's autouse clear runs before
    # the setenv, not after it, which is why this one is local.
    from panoscope.data import layout as _layout
    _layout.clear_cache()

    app = str(Path(panoscope.__file__).parent / "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["active_page"] = "markers"
    at.session_state["selected_cluster"] = "c1"
    # A marker selected, so the gene rows (density | feature UMAP) and the dotplot
    # draw: the feature UMAP is the panel that reads the expression matrix.
    at.session_state["selected_markers"] = {"c1": ["ERBB2"]}
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]

    md = "\n".join(m.value for m in at.markdown)
    # The absent view states itself, in the feature UMAP and in the dotplot.
    assert md.count("per-cell expression not prepared for this study") == 2
    assert "scripts/prep_expr.py" in md
    # ... and the two views this study DOES have still draw. A guard that subtracted
    # the embedding as well would be just as wrong as one that let the page crash.
    assert "UMAP not prepared for this study" not in md
    assert "cell map not prepared for this study" not in md
    assert len(at.get("plotly_chart")) == 2
    # The evidence table survived the stage below it -- the thing the traceback took
    # with it. ERBB2 is demo_mini's own top marker, carried in by the copied inputs.
    assert "ERBB2" in md


# --------------------------------------------------------------------------- #
# 5. The Pathways stage owns two panels of its own, and they need the same guards
# --------------------------------------------------------------------------- #
def test_the_pathways_stage_states_every_absent_view_instead_of_raising(tmp_path):
    """``render_pathways_spatial`` reuses the marker renderers (guarded) but draws the
    leading-edge density and the leading-edge activity UMAP itself. Both read a view
    before the first guarded call, so on an unprepped study the Pathways page kept
    the original traceback after the Markers page had stopped taking it."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    _unprepped(tmp_path, ALL_FALSE)

    def _run(root: str) -> None:
        import pathlib
        import types

        from panoscope.agent import config as _cfg
        from panoscope.data.study import Study as _Study
        from panoscope.ui import enrichment_spatial as _es

        _cfg.set_study(_Study(id="unprepped", data_root=pathlib.Path(root)))
        # A stand-in for one enriched pathway: the stage reads only these two fields,
        # and a study with no viz has no enrichment result to pin a real one to.
        pathway = types.SimpleNamespace(
            gene_set="HALLMARK_SYNTHETIC", leading_edge=["GENE_A", "GENE_B"]
        )
        _es.render_pathways_spatial("c1", [pathway])

    at = AppTest.from_function(_run, default_timeout=120, args=(str(tmp_path),))
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]

    md = "\n".join(m.value for m in at.markdown)
    for label in ("cell map", "UMAP", "transcript density", "per-cell expression"):
        assert f"{label} not prepared for this study" in md, label
    # The density panel names the study's absent view, not the genes: "no precomputed
    # density for the leading-edge genes" would blame the program for the prep step.
    assert "no precomputed density for the leading-edge genes" not in md


# --------------------------------------------------------------------------- #
# 6. The panel names the path the loader will actually look for
# --------------------------------------------------------------------------- #
def test_every_named_producer_path_is_the_one_its_loader_resolves(tmp_path):
    """``_VIEW_SOURCE`` is a hand-written string per view, and nothing read it back.
    A moved artifact would leave the panel telling a biologist to look somewhere the
    pipeline no longer writes -- an honest absence with a wrong address. Pinned
    against the resolvers themselves, on a non-demo study so no legacy flat path can
    stand in for a tree path."""
    resolvers = {
        "cell_map": agent_data._cells_parquet,
        "umap": agent_data._umap_parquet,
        "density": agent_data._density_dir,
        "expr": da._marker_expr_parquet,
    }
    assert set(resolvers) == set(stage._VIEW_SOURCE)
    study = _unprepped(tmp_path, ALL_FALSE)
    with cfg.use_study(study):
        for view, resolve in resolvers.items():
            named = stage._VIEW_SOURCE[view].split(" · ")[0].rstrip("/")
            assert str(resolve().relative_to(study.dir)) == named, view


# --------------------------------------------------------------------------- #
# 7. The loader is the backstop, and it has to say where and who
# --------------------------------------------------------------------------- #
def test_the_expression_matrix_fails_with_the_study_tree_and_its_producer(tmp_path):
    """A manifest may veto a view, never grant one, so a study that declares ``expr``
    and has no file still reaches the loader -- and that is the message a biologist
    sees. It used to be a bare ``pd.read_csv`` failure: an errno, a filename, no
    producer, and for the demo a path under the legacy flat root instead of the
    study's own tree. Every other tidy table already failed with both."""
    study = _unprepped(tmp_path, {"expr": True})
    with cfg.use_study(study):
        assert stage._unavailable("expr") is None  # declared available: no veto
        with pytest.raises(FileNotFoundError) as excinfo:
            da.marker_expr_df()
    message = str(excinfo.value)
    assert str(study.dir / "viz" / "expr.parquet") in message
    assert "scripts/prep_expr.py" in message


def test_the_spatial_hover_says_what_the_rail_says() -> None:
    """The one label on the screen that did not follow an override.

    ``_cluster_celltype_map`` read ``cfg.CLUSTER_KEY`` directly while the rail,
    the marker header, the chat header and the Pathways header all go through
    ``display_cell_type``. So the rail said CAF and the tissue tooltip said
    Stromal, in the same viewport, for the same cells.
    """
    from panoscope.agent import config as cfg, memory
    from panoscope.ui import data_access as _da, spatial_stage as _ss

    assert _ss._cluster_celltype_map()["c2"] == "Stromal", "c2 must start as Stromal"

    memory.create_note(
        claim="c2 is CAF here", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
    )

    labels = _ss._cluster_celltype_map()
    assert labels["c2"] == _da.display_cell_type("c2") == "CAF", labels["c2"]
    assert labels["c1"] == "Tumor", "an unoverridden cluster must be untouched"
    assert labels[""] == "unassigned"


def test_the_feature_colourbar_is_the_genes_maximum_not_the_drawn_subsets() -> None:
    """A viewing control changes the picture, never a value.

    The UMAP is thinned -- all of the selected cluster plus a capped background
    sample -- and the colourbar maximum was read off that thinned frame. The
    brightest cell for a gene is usually not in it, so the only number on the
    panel CHANGED when the biologist clicked a different cluster, under a
    caption reading "this gene's own maximum". On the first 40 panel genes, 38
    drifted: ABCC11's maximum is 9.21 and the panel showed 7.42, 7.82 or 8.52.

    The number is CAPTURED from the renderer, not recomputed here. Computing
    the expectation in the test is how the first version of this passed with
    the fix reverted.
    """
    from panoscope.agent import config as cfg
    from panoscope.ui import data_access as _da
    from panoscope.ui import spatial_stage as _ss

    go = _ss._go()
    umap = _da.umap_df()
    drifted = checked = 0

    for gene in list(_da.panel_names())[:12]:
        expr = _da.marker_expr_col(gene)
        if not len(expr):
            continue
        true_max = round(float(expr["value"].max()), 6)

        shown: set[float] = set()
        captured: list[float] = []
        original = _ss._sqrt_colorbar
        _ss._sqrt_colorbar = lambda v, kind, _o=original: (
            captured.append(float(v)) or _o(v, kind)
        )
        try:
            for c in cfg.CLUSTER_ORDER:
                view = _ss._downsample_bg(umap, umap["cluster"] == c,
                                          _ss._MAX_BG_POINTS)
                captured.clear()
                _ss._umap_feature(go.Figure(), go, view, expr, c)
                if captured:
                    shown.add(round(captured[-1], 6))
        finally:
            _ss._sqrt_colorbar = original

        if not shown:
            continue
        checked += 1

        # What it USED to show: the maximum among the cells that survived thinning.
        thinned = set()
        for c in cfg.CLUSTER_ORDER:
            v = _ss._downsample_bg(umap, umap["cluster"] == c, _ss._MAX_BG_POINTS)
            m = v.merge(expr, on="cell_id", how="left")
            pos = m[m["value"] > 0]
            if len(pos):
                thinned.add(round(float(pos["value"].max()), 6))
        if len(thinned) > 1:
            drifted += 1

        assert shown == {true_max}, (
            f"{gene}: the colourbar printed {sorted(shown)}; the gene's own "
            f"maximum is {true_max}"
        )

    assert checked >= 5, "too few genes carried expression to prove anything"
    assert drifted, (
        "no gene's thinned maximum varied by cluster, so this run could not "
        "have detected the defect -- the fixture, not the code, is what passed"
    )

def test_the_pathways_dotplot_caption_counts_the_selection_not_the_cap() -> None:
    """Both halves of "N of M" came from the already-truncated list.

    ``render_pathways_spatial`` caps the dot plot at 14 genes and passed no
    ``n_selected``, so the caption computed BOTH numbers from the capped list
    and read "14 of 14 selected genes". On c4 the union of the selected
    programs' leading edges is 47 genes: thirty-three of them appeared nowhere
    on the stage and nothing said so. The "N of M" field exists precisely to
    name what the figure left out, and the markers path already passes it.

    This calls the real renderer and captures what it hands the dot plot,
    rather than grepping the source for the fix.
    """
    from panoscope.agent import config as cfg
    from panoscope.ui import data_access as _da
    from panoscope.ui import enrichment_spatial as _es

    seen: dict[str, tuple[int, int]] = {}

    def capture(genes, cluster, n_selected=None, **kw):
        seen[cluster] = (len(list(genes)), n_selected)

    original = _es._render_dotplot
    _es._render_dotplot = capture
    try:
        for c in cfg.CLUSTER_ORDER:
            ce = _da.enrichment_for(c)
            programs = list(ce.enriched) + list(ce.suggestive)
            if programs:
                _es.render_pathways_spatial(c, programs)
    finally:
        _es._render_dotplot = original

    assert seen, "the renderer never reached the dot plot"
    capped = {c: v for c, v in seen.items() if v[0] < v[1]}
    assert capped, (
        "no cluster's selection exceeded the cap, so this run could not have "
        f"detected the defect: {seen}"
    )
    for c, (drawn, selected) in seen.items():
        assert selected is not None, f"{c}: the caption was left to count the cap"
        assert selected >= drawn, f"{c}: drew {drawn} but claims {selected} selected"
