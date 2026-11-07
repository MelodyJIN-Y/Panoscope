"""Transcript Points, read lazily -- the reader ARCHITECTURE §5.1 asks for.

The "density computed from Points" half of §5.1: any bin size, any panel gene, no
precompute step (``scripts/precompute_density.py`` bakes three bin sizes per gene).
Measured on the Xenium breast sample1 store: 4-125 ms per gene vs ~7 ms for a
precomputed file.

What makes it fast:

1. *Sorted by feature*, so each gene is one contiguous row span.
2. *A row-OFFSET index, not a predicate*: ``feature_name`` is dictionary-encoded (as
   ``spatialdata`` requires), which gives pyarrow no row-group statistics to prune
   on, so an exact ``(start_row, end_row)`` per gene does the pruning.
3. *200k-row row groups.*

**Reading needs no ``spatialdata``**: a Points element is parquet underneath, so the
query path uses pyarrow. ``spatialdata`` (the ``panoscope[spatial]`` extra) is needed
to BUILD a store.
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

    About 3 s for 34.5 M rows; written beside the store, so once per store.
    Raises if the table is not sorted by feature: the answers would still be
    right, just silently slow.
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
    # ``source_mtime_ns`` fingerprints the bytes this index describes. Without it
    # the sidecar was trusted forever: the in-process cache correctly missed when
    # the parquet changed, and the miss then re-read the same stale JSON off disk.
    # Measured on a rewritten store -- a gene removed from the store still listed,
    # a gene added still missing, and spans pointing at the wrong rows, which is
    # one gene's transcripts served under another gene's name.
    (Path(store_dir) / INDEX_NAME).write_text(
        json.dumps({"row_group_size": idx.row_group_size, "n_rows": idx.n_rows,
                    "source_mtime_ns": _source_mtime_ns(store_dir, name),
                    "spans": idx.spans}) + "\n", encoding="utf-8")
    return idx


def _source_mtime_ns(store_dir: Path | str, name: str) -> int:
    """mtime of the parquet an index describes, or 0 when there is none."""
    parts = sorted(points_parquet(store_dir, name).glob("*.parquet"))
    return parts[0].stat().st_mtime_ns if parts else 0


@lru_cache(maxsize=8)
def _index_cached(store_dir: str, name: str, _mtime_ns: int) -> PointsIndex:
    p = Path(store_dir) / INDEX_NAME
    if p.is_file():
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            fresh = int(raw.get("source_mtime_ns", -1)) == _mtime_ns
        except Exception:  # noqa: BLE001 - an unreadable sidecar is a rebuild
            fresh = False
        if fresh:
            return PointsIndex(
                spans={k: (int(v[0]), int(v[1])) for k, v in raw["spans"].items()},
                row_group_size=int(raw["row_group_size"]),
                n_rows=int(raw["n_rows"]))
        # Stale or unfingerprinted: rebuild. A wrong index would silently answer
        # with the wrong gene's rows.
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

    Filtering probes is ``PlatformAdapter.is_control``'s job, in one place.
    """
    return tuple(sorted(index(store_dir, name).spans))


def density(gene: str, bin_um: float, store_dir: Path | str,
            name: str = DEFAULT_POINTS) -> Any:
    """One feature's transcript density, binned at ``bin_um`` micrometres.

    Returns ``hx, hy, count, density`` with ``hx``/``hy`` the bin CENTRE in µm and
    ``density = count / bin_area`` -- the shape ``scripts/precompute_density.py``
    wrote, so the committed frames are its oracle. Bin size is a VIEWING control,
    so any size is accepted.
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
