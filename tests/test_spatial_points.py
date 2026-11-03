"""The lazy Points reader, and the precomputed frames as its oracle.

Two halves, and the second is the one §5.1 asks for by name: *"validate the zarr
reader against the tidy adapter's output on the same study"*.

1. A SYNTHETIC store, built here, which runs everywhere. Small enough to reason
   about by hand, so a wrong bin centre or an off-by-one row offset is visible
   rather than inferred.
2. The REAL store against the 840 committed ``viz/hexbin/*.parquet`` frames. The
   store is 609 MB and cannot be committed (§12: fetch, never bundle), so this
   skips unless one is present -- which means it is a local gate, not a CI one.
   Point ``$PANOSCOPE_POINTS_STORE`` at a store to run it.

The synthetic store uses the dictionary encoding SpatialData requires, because a
test on a plain-string column would pass while the real store took a different
code path through pyarrow entirely.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from panoscope.data import spatial

REPO = Path(__file__).resolve().parents[1]


def _write_store(tmp: Path, rows: list[tuple[float, float, str]]) -> Path:
    """A minimal SpatialData-shaped Points store, sorted by feature."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = sorted(rows, key=lambda r: r[2])
    d = spatial.points_parquet(tmp)
    d.mkdir(parents=True, exist_ok=True)
    tbl = pa.table({
        "x": pa.array([r[0] for r in rows], pa.float64()),
        "y": pa.array([r[1] for r in rows], pa.float64()),
        "feature_name": pa.array([r[2] for r in rows]).dictionary_encode(),
    })
    pq.write_table(tbl, d / "part.0.parquet", row_group_size=2)
    return tmp


@pytest.fixture()
def tiny(tmp_path: Path) -> Path:
    return _write_store(tmp_path / "spatial.zarr", [
        (5.0, 5.0, "GENE_A"), (15.0, 5.0, "GENE_A"), (12.0, 7.0, "GENE_A"),
        (5.0, 25.0, "GENE_B"),
        (1.0, 1.0, "NegControlProbe_00042"),
    ])


# --------------------------------------------------------------------------- #
def test_the_index_finds_each_feature_exactly(tiny: Path) -> None:
    idx = spatial.build_index(tiny)
    assert idx.n_rows == 5
    assert idx.spans["GENE_A"] == (0, 3), idx.spans
    assert idx.spans["GENE_B"] == (3, 4), idx.spans
    assert (tiny / spatial.INDEX_NAME).is_file(), "the sidecar was not written"


def test_an_unsorted_store_is_refused_rather_than_answered_slowly(tmp_path) -> None:
    """The defect that returns the RIGHT number is the one nobody finds.

    An unsorted table indexes without error; every span then covers most of the
    file, the answers stay correct, and the only symptom is that it is slow. So
    this must raise.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = spatial.points_parquet(tmp_path / "spatial.zarr")
    d.mkdir(parents=True)
    pq.write_table(pa.table({
        "x": pa.array([1.0, 2.0, 3.0, 4.0]),
        "y": pa.array([1.0, 2.0, 3.0, 4.0]),
        "feature_name": pa.array(["A", "B", "A", "B"]).dictionary_encode(),
    }), d / "part.0.parquet", row_group_size=4)

    with pytest.raises(spatial.StoreUnavailable, match="not sorted"):
        spatial.build_index(tmp_path / "spatial.zarr")


def test_a_missing_store_names_itself_rather_than_returning_empty(tmp_path) -> None:
    with pytest.raises(spatial.StoreUnavailable, match="no Points parquet"):
        spatial.index(tmp_path / "absent.zarr")


def test_density_bins_to_centres_and_normalises_by_area(tiny: Path) -> None:
    out = spatial.density("GENE_A", 10.0, tiny)
    assert list(out.columns) == ["hx", "hy", "count", "density"]
    got = {(r.hx, r.hy): (r.count, r.density) for r in out.itertuples()}
    # (5,5) -> bin [0,10)x[0,10), centre (5,5), one point.
    # (15,5) and (12,7) -> bin [10,20)x[0,10), centre (15,5), two points.
    assert got == {(5.0, 5.0): (1, 0.01), (15.0, 5.0): (2, 0.02)}, got


def test_bin_size_is_a_viewing_control_so_any_size_is_accepted(tiny: Path) -> None:
    """CLAUDE.md: a viewing control changes the picture, never a value.

    The three sizes the precompute baked were a STORAGE limit, not a measurement
    one. Total count must be conserved across every bin size -- a re-bin that
    lost or invented a transcript would be a value changing under a view control.
    """
    for bin_um in (7.0, 10.0, 33.0, 50.0):
        out = spatial.density("GENE_A", bin_um, tiny)
        assert int(out["count"].sum()) == 3, f"bin_um={bin_um} lost a transcript"


def test_a_feature_never_recorded_gives_an_empty_frame_not_a_crash(tiny: Path) -> None:
    out = spatial.density("NOT_ON_THIS_PANEL", 50.0, tiny)
    assert out.empty and list(out.columns) == ["hx", "hy", "count", "density"]


def test_control_probes_are_reported_not_filtered_here(tiny: Path) -> None:
    """One home for the control rule, and it is ``PlatformAdapter.is_control``.

    Filtering here too would be a second place it can drift, which is exactly how
    a negative control once led ``key_markers``.
    """
    feats = spatial.available_features(tiny)
    assert "NegControlProbe_00042" in feats
    assert feats == tuple(sorted(feats))


# --------------------------------------------------------------------------- #
def _real_store() -> Path | None:
    env = os.getenv("PANOSCOPE_POINTS_STORE")
    if env and Path(env).is_dir():
        return Path(env)
    from panoscope.agent import config as cfg

    s = cfg.current_study().spatial
    return s if s.is_dir() else None


@pytest.mark.skipif(_real_store() is None,
                    reason="no SpatialData store (609 MB; fetched, never bundled)")
def test_the_lazy_path_reproduces_every_committed_frame() -> None:
    """The oracle. 840 frames baked by the one-time precompute, bin for bin.

    This is what lets the precompute be retired: not "the plots look similar" but
    every bin of every gene at every baked size being equal.
    """
    import pandas as pd

    store = _real_store()
    hexdirs = sorted(REPO.glob("data/studies/*/viz/hexbin"))
    assert hexdirs, "no committed hexbin frames to check against"

    checked = 0
    for p in sorted(hexdirs[0].glob("*.parquet")):
        m = re.fullmatch(r"(.+)_(\d+)um", p.stem)
        if not m:
            continue
        want = (pd.read_parquet(p).sort_values(["hx", "hy"], ignore_index=True)
                  [["hx", "hy", "count", "density"]])
        got = spatial.density(m.group(1), float(m.group(2)), store)
        pd.testing.assert_frame_equal(want, got[["hx", "hy", "count", "density"]],
                                      check_dtype=False, atol=1e-4,
                                      obj=f"{p.name}")
        checked += 1
    assert checked >= 800, f"only {checked} frames compared"


@pytest.mark.skipif(_real_store() is None, reason="no SpatialData store")
def test_hexbins_prefers_the_store_and_serves_an_unbaked_bin_size() -> None:
    """The seam, and the capability the precompute could never have.

    37 µm was never baked. If this returns a frame, the UI is reading the store.
    """
    from panoscope.agent import config as cfg
    from panoscope.data.study import Study
    from panoscope.ui import data_access as da

    store = _real_store()
    sid = store.parent.name if store.name == "spatial.zarr" else None
    if sid is None:
        pytest.skip("store is not inside a study tree")

    with cfg.use_study(Study.open(sid)):
        da.hexbins.clear()
        out = da.hexbins("ERBB2", 37)
    assert not out.empty and int(out["count"].sum()) > 0
