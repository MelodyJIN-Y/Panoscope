"""R4 with pixels: a plot with no conditionality record refuses to draw.

Three things are pinned here, in descending order of what they buy:

1. **The gate bites.** ``render_panel`` raises ``UndrawableFigure`` for a figure
   arriving without a ``Conditionality``. The type already refuses to construct
   one with a missing mandatory field, so both halves of R4 fail closed.
2. **No figure can route around the gate.** Every plot on the two spatial stages
   must go through ``render_panel`` — asserted against the source, because the
   failure mode is a *sixth* bare ``st.plotly_chart`` landing later and nobody
   noticing until a panel-only volcano is in a talk.
3. **The five real records say the things a reader cannot get from the pixels**:
   the panel denominator (derived from the file, never the constant 280), the
   display downsample, the bin µm and area normalisation, and what each figure
   is *not*.
"""

from __future__ import annotations

import pytest

from panoscope.types.uispec import Conditionality, UndrawableFigure
from panoscope.ui import conditionality as cnd


def _hb():
    """A two-bin density frame, shaped like a precomputed hexbin parquet."""
    import pandas as pd

    return pd.DataFrame(
        {"hx": [0.0, 50.0], "hy": [0.0, 0.0], "count": [3, 9], "density": [0.01, 0.084]}
    )


# --------------------------------------------------------------------------- #
# 1. The gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cond", [None, "", {}, 0, ("drawn", "denominator")])
def test_a_figure_with_no_conditionality_refuses_to_draw(cond):
    """The headline claim. Nothing that is not a Conditionality gets drawn."""
    with pytest.raises(UndrawableFigure):
        cnd.render_panel(object(), cond)


def test_a_lookalike_record_does_not_get_through():
    """Duck typing is not a conditionality: a stand-in carrying the four field
    names still refuses, because it was never validated by the landed type."""

    class _Fake:
        drawn = denominator = transform = not_this = "something"

        def caption(self):
            return ("a", "b")

    with pytest.raises(UndrawableFigure):
        cnd.render_panel(object(), _Fake())


@pytest.mark.parametrize("field", ["drawn", "denominator", "transform", "not_this"])
def test_a_record_missing_a_required_field_cannot_exist(field):
    """The other half of the gate, enforced one layer down at construction."""
    kw = dict(drawn="d", denominator="n", transform="t", not_this="x")
    kw[field] = "   "
    with pytest.raises(UndrawableFigure):
        Conditionality(**kw)


def test_no_plot_routes_around_the_gate():
    """Every figure on both spatial stages draws through ``render_panel``.

    A bare ``st.plotly_chart`` is exactly the regression this file exists to
    stop: it draws a panel-conditioned figure with nothing saying so.
    """
    from pathlib import Path

    root = Path(cnd.__file__).parent
    for name in ("spatial_stage.py", "enrichment_spatial.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "st.plotly_chart" not in src, f"{name} draws a figure outside the gate"
        assert "render_panel(" in src


# --------------------------------------------------------------------------- #
# 2. The five records
# --------------------------------------------------------------------------- #
def _records() -> dict[str, Conditionality]:
    """The five figures' real records, built the way the stages build them."""
    return {
        "cell_map": cnd.cell_map_cond(
            cluster="c1", n_total=158_379, n_selected=4_812,
            n_bg_drawn=45_000, n_bg_total=153_567,
        ),
        "cluster_umap": cnd.cluster_umap_cond(
            cluster="c1", n_total=166_363, n_selected=4_812,
            n_bg_drawn=45_000, n_bg_total=161_551,
        ),
        "feature_umap": cnd.feature_umap_cond(
            gene="ERBB2", n_expr_cells=40_946, n_total=166_363,
            n_bg_drawn=45_000, n_bg_total=161_551,
        ),
        "density": cnd.gene_density_cond(gene="ERBB2", hb=_hb(), bin_um=50),
        "set_density": cnd.set_density_cond(
            program="Emt", hb=_hb(), bin_um=50, n_genes=12, n_drawn=9,
        ),
        "leading_edge_umap": cnd.leading_edge_umap_cond(
            program="Emt", n_genes=12, n_drawn=9, n_expr_cells=40_946,
            n_total=166_363, n_bg_drawn=45_000, n_bg_total=161_551,
        ),
        "dotplot": cnd.dotplot_cond(
            genes_drawn=3, genes_selected=4, n_clusters=9,
            n_expr_cells=40_946, n_total=158_379,
        ),
    }


@pytest.mark.parametrize("key", sorted(_records()))
def test_every_real_record_is_drawable_and_captions(key):
    rec = _records()[key]
    assert isinstance(rec, Conditionality)
    line1, line2 = rec.caption()
    assert line1.strip() and line2.strip()


@pytest.mark.parametrize("key", sorted(_records()))
def test_every_real_record_carries_the_panel_denominator(key):
    """The denominator always left out. Derived from the panel file, so the
    assertion is on the shape, not on 280 — panel size is dynamic across studies.
    """
    from panoscope.ui import data_access as da

    n = len(da.panel_names())
    _, line2 = _records()[key].caption()
    assert f"a {n}-gene targeted panel" in line2 and "NOT whole-transcriptome" in line2


def test_panel_denominator_follows_the_file_not_a_constant(monkeypatch):
    """Point the panel reader at a 7-gene panel and the caption says 7."""
    from panoscope.ui import data_access as da

    monkeypatch.setattr(da, "panel_names", lambda: ["A", "B", "C", "D", "E", "F", "G"])
    assert "a 7-gene targeted panel" in cnd._panel_scope()
    assert "280" not in cnd.gene_density_cond(gene="X", hb=_hb(), bin_um=50).denominator


def test_the_downsample_is_stated_rather_than_deleted():
    """_MAX_BG_POINTS silently thinned the cell map and the UMAP. It is now a
    sentence under the panel (docs/interface-design.md §1.3)."""
    for key in ("cell_map", "cluster_umap", "feature_umap", "leading_edge_umap"):
        rendering = _records()[key].rendering
        assert "thinned to 45,000 of" in rendering and "drawn in full" in rendering


def test_nothing_is_claimed_when_nothing_was_thinned():
    """A frame under the cap must not claim a downsample that did not happen."""
    rec = cnd.cell_map_cond(
        cluster="c1", n_total=900, n_selected=100, n_bg_drawn=800, n_bg_total=800
    )
    assert rec.rendering == ""


def test_the_cell_map_says_centroids_not_boundaries():
    """Every reader assumes segmentation boundaries unless told otherwise."""
    assert "NOT segmentation boundaries" in _records()["cell_map"].not_this


def test_the_umap_says_it_is_a_layout_not_a_measurement():
    """The Row-1 caption that was deleted for a cleaner grid, as a record."""
    not_this = _records()["cluster_umap"].not_this
    assert "layout" in not_this and "distances are not" in not_this


def test_the_density_panel_states_the_bin_and_the_normalisation():
    rec = _records()["density"]
    assert "50 um square bins" in rec.transform and "area-normalised" in rec.transform
    assert "not a statistic" in rec.not_this
    assert "qv >=" in rec.exclusions  # read from the density index, not asserted


def test_the_density_scale_admits_panels_are_not_comparable():
    """cmax is per-panel, so two genes side by side both saturate. Say so."""
    assert "NOT comparable" in _records()["density"].scale
    assert "0.084" in _records()["density"].scale


def test_the_set_panels_state_how_many_driving_genes_were_dropped():
    """The sum silently skips a gene with no frame; the picture cannot show it."""
    assert "9 of 12" in _records()["set_density"].drawn
    assert "9 of 12" in _records()["leading_edge_umap"].drawn


def test_the_enrichment_panels_disclaim_the_enrichment_statistic():
    for key in ("set_density", "leading_edge_umap"):
        assert "not the enrichment score" in _records()[key].not_this


def test_the_dotplot_admits_it_derives():
    """The one figure on the stage that computes rather than renders. R5 says
    plots do not derive; this one does, and the caption says so out loud."""
    rec = _records()["dotplot"]
    assert "computed in the renderer" in rec.transform
    assert "no glm_coef" in rec.not_this


def test_the_expression_panels_state_the_subsample():
    """Per-cell expression covers a cluster-stratified subsample, not every cell.
    That denominator reached no pixel before."""
    for key in ("feature_umap", "leading_edge_umap", "dotplot"):
        assert "40,946 of 1" in _records()[key].denominator
        assert "subsample" in _records()[key].denominator


# --------------------------------------------------------------------------- #
# 3. The record reaches the pixel
# --------------------------------------------------------------------------- #
def test_the_caption_is_rendered_under_the_panel():
    """Headless proof that the record reaches the output, not just the code."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        import plotly.graph_objects as go

        from panoscope.ui import conditionality as _cnd

        _cnd.render_panel(
            go.Figure(),
            _cnd.cell_map_cond(
                cluster="c1", n_total=158_379, n_selected=4_812,
                n_bg_drawn=45_000, n_bg_total=153_567,
            ),
        )

    at = AppTest.from_function(_run)
    at.run(timeout=60)
    assert not at.exception, [str(e.value) for e in at.exception]
    md = "\n".join(m.value for m in at.markdown)
    assert "cell centroids at tissue coordinates" in md
    assert "NOT segmentation boundaries" in md
    assert "thinned to 45,000 of 153,567" in md
    assert "title=" in md  # the full record hover-expands


def test_the_set_density_count_comes_from_the_frames_that_were_summed(tmp_path):
    """The count in the caption is a DERIVED statistic, so it comes from what was drawn.

    ``_n_with_density`` counted against ``available_density_markers()`` — the density
    index's declaration of what was precomputed, and with no index at all the study's
    demo marker set. That is a claim about the run, not a reading of what was read.
    On a study declaring ``density: true`` with a PARTIAL precompute (exactly what a
    part-prepped second study looks like) the caption said "the SUM of 3 of 3 driving
    genes" over a figure summed from one frame: a number larger than the evidence
    behind the picture, on the panel whose whole job is to admit what the sum hides.

    Zero frames is the easy half — ``leading_edge_density`` returns None and the
    empty state fires instead of a caption — so the case that bites is one frame of
    three, with the figure really drawing.

    The tree is built under ``tmp_path`` and opened by an explicit ``data_root``, so
    no environment variable, no ``layout`` cache and nothing under ``data/studies/``
    is involved.
    """
    import json
    import shutil

    import pandas as pd

    from panoscope.agent import config as cfg
    from panoscope.data.study import Study
    from panoscope.ui import data_access as da
    from panoscope.ui import enrichment_spatial as es

    src = Study.open("synthetic_multisample_fixture")
    if not src.exists:
        pytest.skip("the fixture study is not on disk")

    dst = tmp_path / "studies" / "partprepped"
    shutil.copytree(src.dir, dst)
    manifest = json.loads((dst / "manifest.json").read_text(encoding="utf-8"))
    manifest["dataset_id"] = "partprepped"
    manifest["views_available"] = {**manifest.get("views_available", {}), "density": True}
    (dst / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    hexbin = dst / "viz" / "hexbin"
    hexbin.mkdir(parents=True, exist_ok=True)
    (hexbin / "_index.json").unlink(missing_ok=True)
    # one driving gene of three has a frame; the other two were never precomputed
    pd.DataFrame(
        {"hx": [25.0, 75.0], "hy": [25.0, 75.0], "count": [3, 1],
         "density": [0.0012, 0.0004]}
    ).to_parquet(hexbin / "SYNG01_50um.parquet")

    genes = ("SYNG01", "SYNG02", "SYNG03")
    with cfg.use_study(Study(id="partprepped", data_root=tmp_path)):
        # the declaration still names all three, which is what counted 3 of 3
        assert set(genes) <= set(da.available_density_markers())
        # ground truth: what the sum actually summed
        hb = da.leading_edge_density(genes, 50)
        assert hb is not None and not hb.empty  # the figure really draws
        on_disk = [g for g in genes if (hexbin / f"{g}_50um.parquet").exists()]
        assert len(on_disk) == 1

        n_drawn = es._n_with_density(genes, 50)
        assert n_drawn == len(on_disk)
        rec = cnd.set_density_cond(
            program="Emt", hb=hb, bin_um=50, n_genes=len(genes), n_drawn=n_drawn
        )
        assert "SUM of 1 of 3 driving genes" in rec.drawn

        # a frame exists per (gene, bin), so the count is read at the bin that was
        # drawn: at 100 µm nothing was precomputed and the two stay in step.
        assert da.leading_edge_density(genes, 100) is None
        assert es._n_with_density(genes, 100) == 0
