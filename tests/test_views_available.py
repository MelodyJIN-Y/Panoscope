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
from pathlib import Path

import pytest

from panoscope.agent import config as cfg
from panoscope.data.study import Study
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
