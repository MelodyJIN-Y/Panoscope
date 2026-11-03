"""Transcript Points, read lazily -- the reader ARCHITECTURE §5.1 asks for.

§5.1 replaces the hard-coded tidy layer with a SpatialData ``.zarr`` store, and the
row that matters for the interface is::

    density/*.parquet   ->  computed from Points            bin size in real um
    transcripts.csv.gz  ->  Points (lazy, no full read)

This module is the "computed from Points" half. It exists because the alternative
does not survive a second dataset: ``scripts/precompute_density.py`` streams 1.32 GB
of ``transcripts.csv.gz`` to bake **841 files** (280 genes x 3 bin sizes, 26 MB), and
those three bin sizes are then the only ones a biologist can ever choose. Two samples
double the file count; a 5,000-plex panel makes it 15,000.

MEASURED on the Xenium human breast sample1 store (34,493,510 transcripts at
qv>=20), best of three, with the committed precomputed frames as the oracle:

    gene      transcripts   this module   precomputed file
    CPA3           14,150          4 ms              7 ms
    LUM         1,087,247         66 ms              7 ms
    ERBB2       2,164,898        125 ms              7 ms

Slower per frame, and worth it: ANY bin size (37 um works), any gene on the panel,
no precompute step, one store instead of 841 files. 125 ms is the worst gene on the
panel, and the residual is reading and binning 2.16 M real points.

**Three things make it fast, each measured rather than assumed.**

1. *Sorted by feature.* Each gene is then one contiguous row span, so a query reads
   a handful of row groups instead of the file.
2. *A row-OFFSET index, not a predicate.* Filtering with ``pc.field(...) == gene``
   costs ~200 ms FLAT regardless of how small the gene is -- flat, because
   ``feature_name`` is ``dictionary``-encoded and dictionary columns carry no
   min/max row-group statistics for pyarrow to prune on. Re-encoding it as plain
   ``string`` makes the same filter 11-68x faster AND makes the store unreadable by
   ``spatialdata``, which requires the categorical. So the encoding stays exactly as
   SpatialData writes it and the index does the pruning instead: an exact
   ``(start_row, end_row)`` per gene, read the covering row groups, slice. There is
   no comparison at query time at all.
3. *200k-row row groups.* SpatialData writes ~1 M-row groups, which is four times
   more data per query than the answer needs.

**Reading needs no ``spatialdata``.** A Points element is parquet underneath
(``points/<name>/points.parquet/part.*.parquet``), so the query path uses pyarrow,
already a core dependency. ``spatialdata`` is needed to BUILD a store and for
interop -- the ``panoscope[spatial]`` extra, exactly as §5.1 places it -- and the
legacy tidy reader is retained beside this one, as §5.1 also requires.

**On the file name.** ``IMPLEMENTATION.md`` §1 names ``data/spatial.py``; its §10
migration map marks a DIFFERENT row mentioning that path as superseded, on the
grounds that the reader it meant "already shipped as ``platforms/xenium.py``". That
note is about splitting ``agent/data.py`` and about the TIDY reader. This is the
zarr reader §5.1 asks for, it is a reader, and ``panoscope/data/`` is where §1 says
readers live.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

#: Written beside the store by :func:`build_index`. One line per feature, derived,
#: and rebuilt whenever the parquet it describes has changed.
INDEX_NAME: str = "points_index.json"

#: Where SpatialData puts a Points element's parquet inside the ``.zarr``.
POINTS_RELPATH: str = "points/{name}/points.parquet"

DEFAULT_POINTS: str = "transcripts"


class StoreUnavailable(RuntimeError):
    """Raised when a study has no Points store, or it cannot be read.

    A distinct type rather than ``FileNotFoundError``, because the caller's choice
    differs: a missing store means *fall back to the tidy reader*; a malformed one
    means stop. Neither is "this gene has no transcripts", which is what a silently
    empty frame would say instead.
    """


@dataclass(frozen=True, slots=True)
class PointsIndex:
    """Where each feature's rows live, and how the file is chunked.

    ``spans`` is ``feature -> (start_row, end_row)``, half-open, over the whole
    table. It is only meaningful because the table is sorted by feature, which
    :func:`build_index` VERIFIES rather than trusts.
    """

    spans: dict[str, tuple[int, int]]
    row_group_size: int
    n_rows: int

    def covering_row_groups(self, gene: str) -> tuple[int, int, int]:
        """``(first_group, last_group, base_row)`` for one feature's span."""
        lo, hi = self.spans[gene]
        first, last = lo // self.row_group_size, (hi - 1) // self.row_group_size
        return first, last, first * self.row_group_size


def points_parquet(store_dir: Path | str, name: str = DEFAULT_POINTS) -> Path:
    """The parquet directory inside a ``.zarr`` holding one Points element."""
    return Path(store_dir) / POINTS_RELPATH.format(name=name)


def _parquet_file(store_dir: Path | str, name: str = DEFAULT_POINTS) -> Any:
    import pyarrow.parquet as pq

    d = points_parquet(store_dir, name)
    parts = sorted(d.glob("*.parquet"))
    if not parts:
        raise StoreUnavailable(
            f"no Points parquet under {d}; this study has no spatial store and the "
            "caller should read the tidy frames instead"
        )
    if len(parts) > 1:
        raise StoreUnavailable(
            f"{d} holds {len(parts)} parquet parts, and the row-offset index "
            "addresses one contiguous table. Write the element as a single part."
        )
    return pq.ParquetFile(parts[0])


def build_index(store_dir: Path | str, name: str = DEFAULT_POINTS) -> PointsIndex:
    """Scan the Points parquet once and record each feature's row span.

    One pass reading only ``feature_name``: about 3 s for 34.5 M rows. The result
    is written beside the store, so it costs that once per store rather than once
    per session.

    **The sortedness check is the load-bearing half.** A feature's rows are
    contiguous only if the table is sorted by feature. On an unsorted table every
    span would stretch from a gene's first row to its last, the query would read
    most of the file, and *nothing would look wrong* -- the answers would still be
    correct, just slow. A defect that returns the right number is the one nobody
    finds, so this raises instead.
    """
    import numpy as np
    import pyarrow as pa

    pf = _parquet_file(store_dir, name)
    rg_size = pf.metadata.row_group(0).num_rows if pf.num_row_groups else 0
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for rg in range(pf.num_row_groups):
        col = (pf.read_row_group(rg, columns=["feature_name"])
                 .column(0).cast(pa.string()).to_numpy(zero_copy_only=False))
        vals, starts, counts = np.unique(col, return_index=True, return_counts=True)
        for v, s, c in zip(vals, starts, counts):
            key = str(v)
            lo, hi = spans.get(key, (cursor + int(s), cursor + int(s)))
            spans[key] = (min(lo, cursor + int(s)),
                          max(hi, cursor + int(s) + int(c)))
        cursor += len(col)

    ordered = sorted(spans.items(), key=lambda kv: kv[1][0])
    for (g1, (_lo1, hi1)), (g2, (lo2, _hi2)) in zip(ordered, ordered[1:]):
        if hi1 > lo2:
            raise StoreUnavailable(
                f"the Points table is not sorted by feature_name: {g1!r} and "
                f"{g2!r} interleave. Sort before indexing -- an unsorted store "
                "indexes without error and then reads nearly the whole file."
            )

    idx = PointsIndex(spans={k: (int(a), int(b)) for k, (a, b) in spans.items()},
                      row_group_size=int(rg_size), n_rows=int(cursor))
    (Path(store_dir) / INDEX_NAME).write_text(
        json.dumps({"row_group_size": idx.row_group_size, "n_rows": idx.n_rows,
                    "spans": idx.spans}) + "\n", encoding="utf-8")
    return idx


@lru_cache(maxsize=8)
def _index_cached(store_dir: str, name: str, _mtime_ns: int) -> PointsIndex:
    p = Path(store_dir) / INDEX_NAME
    if p.is_file():
        raw = json.loads(p.read_text(encoding="utf-8"))
        return PointsIndex(
            spans={k: (int(v[0]), int(v[1])) for k, v in raw["spans"].items()},
            row_group_size=int(raw["row_group_size"]), n_rows=int(raw["n_rows"]))
    return build_index(store_dir, name)


def index(store_dir: Path | str, name: str = DEFAULT_POINTS) -> PointsIndex:
    """The row-offset index, read from disk or built once.

    Keyed on the parquet's mtime as well as its path. A rebuilt store under the
    same name must not be answered out of a cache describing the old bytes --
    the defect ``platforms/xenium._annotated_controls_cached`` was fixed for.
    """
    d = points_parquet(store_dir, name)
    parts = sorted(d.glob("*.parquet"))
    mt = parts[0].stat().st_mtime_ns if parts else 0
    return _index_cached(str(store_dir), name, mt)


def available_features(store_dir: Path | str,
                       name: str = DEFAULT_POINTS) -> tuple[str, ...]:
    """Every feature the store carries, sorted. Control probes INCLUDED.

    Not filtered here, deliberately. This module reports what the instrument
    recorded; deciding that a probe is not a subject of interpretation belongs to
    ``PlatformAdapter.is_control``, and doing it in two places is two places the
    rule can drift -- which is exactly how a negative control once led
    ``key_markers``.
    """
    return tuple(sorted(index(store_dir, name).spans))


def density(gene: str, bin_um: float, store_dir: Path | str,
            name: str = DEFAULT_POINTS) -> Any:
    """One feature's transcript density, binned at ``bin_um`` micrometres.

    Returns ``hx, hy, count, density`` with ``hx``/``hy`` the bin CENTRE in µm and
    ``density = count / bin_area`` -- the exact shape
    ``scripts/precompute_density.py`` wrote, so this is a drop-in for
    ``data_access.hexbins`` and the committed frames are its oracle.

    Area-normalised for the reason the precompute was: a coarser bin holds more
    transcripts, and a raw count would make it look hotter when nothing about the
    tissue changed. Bin size is a VIEWING control -- it changes the picture, never
    a value -- which is why any size is accepted. The three the precompute baked
    were a storage limit, not a measurement one.
    """
    import numpy as np
    import pandas as pd

    idx = index(store_dir, name)
    if gene not in idx.spans:
        # Empty frame, not an exception: a feature the instrument never recorded
        # has no density to draw, and the caller renders nothing. Whether that
        # ABSENCE is evidence is the lattice's question, asked elsewhere.
        return pd.DataFrame({"hx": [], "hy": [], "count": [], "density": []})

    pf = _parquet_file(store_dir, name)
    lo, hi = idx.spans[gene]
    first, last, base = idx.covering_row_groups(gene)
    tbl = pf.read_row_groups(list(range(first, last + 1)), columns=["x", "y"])
    x = tbl.column("x").to_numpy()[lo - base:hi - base]
    y = tbl.column("y").to_numpy()[lo - base:hi - base]

    hx = (x // bin_um).astype(np.int64) * bin_um + bin_um / 2
    hy = (y // bin_um).astype(np.int64) * bin_um + bin_um / 2
    out = pd.DataFrame({"hx": hx, "hy": hy}).value_counts().reset_index(name="count")
    out["density"] = (out["count"] / (bin_um * bin_um)).round(4)
    return out.sort_values(["hx", "hy"], ignore_index=True)
