"""Precompute transcript-density hexbins for the demo markers (ONE-TIME).

Streams the ~1.4 GB Xenium ``transcripts.csv.gz`` a SINGLE time (pandas
``chunksize`` — the file is never loaded whole), keeps only demo-marker rows that
pass quality control, then bins each gene's transcript locations at 25 / 50 / 100
µm and writes one area-normalized parquet per (gene, bin) under ``data/density/``.

Grounding discipline (CLAUDE.md):
  * The bin size is a *viewing* control: it changes the picture, never a value.
    Each (gene, bin) is a distinct precomputed frame; nothing is recomputed live.
  * The density colour scale is AREA-NORMALIZED (``density = count / bin_area``,
    transcripts per µm²) so coarser bins are not falsely hotter.
  * DEMO_MARKERS is read from ``agent/config.py`` (itself derived from source),
    never hand-typed here.

Outputs:
  data/density/{GENE}_{bin}um.parquet   cols: hx, hy, count, density
  data/density/_index.json              available (gene, bin) pairs + summary
  data/PREP_MANIFEST.json               (density fragment) per-file row counts

Run:
  .venv/bin/python scripts/precompute_density.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable so we can read the derived DEMO_MARKERS.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import config  # noqa: E402

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
BIN_SIZES_UM: tuple[int, ...] = (25, 50, 100)
QV_MIN: float = 20.0  # Xenium quality-value floor (10x default recommendation)
CHUNK_ROWS: int = 2_000_000  # rows per streaming chunk (~caps peak memory)

# Control / non-biological probe prefixes to drop (case-insensitive).
CONTROL_PREFIXES: tuple[str, ...] = (
    "negcontrolprobe",
    "negcontrolcodeword",
    "blank",
    "antisense",
)

# Columns we actually parse from the gzip (skip transcript_id, cell_id,
# overlaps_nucleus, z_location — not needed for a 2D density).
USECOLS: list[str] = ["feature_name", "x_location", "y_location", "qv"]

DENSITY_DIR: Path = config.DATA_DIR_PATH / "density"
INDEX_JSON: Path = DENSITY_DIR / "_index.json"
MANIFEST_JSON: Path = config.DATA_DIR_PATH / "PREP_MANIFEST.json"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _assert_inputs_exist() -> None:
    """Fail loudly (named errors) if a required raw input is missing."""
    if not config.TRANSCRIPTS_CSV_GZ.exists():
        raise FileNotFoundError(
            f"[precompute_density] transcripts file missing: "
            f"{config.TRANSCRIPTS_CSV_GZ} — check RAW_XEN path."
        )
    if not config.DEMO_MARKERS:
        raise ValueError(
            "[precompute_density] DEMO_MARKERS is empty; agent/config.py failed "
            "to derive markers from source. Refusing to bin nothing."
        )


def _is_control(feature_name: str) -> bool:
    """True if a feature name is a negative-control / blank / antisense probe."""
    low = feature_name.lower()
    return any(low.startswith(p) for p in CONTROL_PREFIXES)


def _report_progress(n_chunks: int, total_rows: int, kept_rows: int, t0: float) -> None:
    """Print a lightweight progress line each chunk."""
    elapsed = time.time() - t0
    rate = total_rows / elapsed if elapsed > 0 else 0.0
    print(
        f"  chunk {n_chunks}: scanned {total_rows:,} rows "
        f"({kept_rows:,} kept) | {elapsed:.0f}s | {rate/1e6:.2f}M rows/s",
        flush=True,
    )


def _stream_demo_transcripts(demo_set: frozenset[str]) -> dict[str, pd.DataFrame]:
    """Single streaming pass over the gzip; collect x/y per kept demo gene.

    Filters within each chunk (never materializes the whole file):
      feature_name in DEMO_MARKERS  AND  qv >= QV_MIN  AND  not a control probe.
    Returns {gene: DataFrame(x_location, y_location)}.
    """
    # Per-gene accumulators of small filtered frames, concatenated once at the end.
    parts: dict[str, list[pd.DataFrame]] = {g: [] for g in demo_set}

    total_rows = 0
    kept_rows = 0
    n_chunks = 0
    t0 = time.time()

    reader = pd.read_csv(
        config.TRANSCRIPTS_CSV_GZ,
        compression="gzip",
        usecols=USECOLS,
        chunksize=CHUNK_ROWS,
        dtype={
            "feature_name": "string",
            "x_location": "float32",
            "y_location": "float32",
            "qv": "float32",
        },
    )

    for chunk in reader:
        n_chunks += 1
        total_rows += len(chunk)

        # qv gate + demo-gene membership. Control probes are excluded because
        # they are not in demo_set; we also guard explicitly below.
        mask = (chunk["qv"] >= QV_MIN) & chunk["feature_name"].isin(demo_set)
        if not mask.any():
            _report_progress(n_chunks, total_rows, kept_rows, t0)
            continue

        sub = chunk.loc[mask, ["feature_name", "x_location", "y_location"]]

        # Defensive: never let a control probe through even if a name collides.
        keep = ~sub["feature_name"].map(_is_control)
        sub = sub.loc[keep]
        kept_rows += len(sub)

        for gene, grp in sub.groupby("feature_name", observed=True):
            parts[str(gene)].append(
                grp[["x_location", "y_location"]].reset_index(drop=True)
            )

        _report_progress(n_chunks, total_rows, kept_rows, t0)

    elapsed = time.time() - t0
    print(
        f"[precompute_density] stream done: {total_rows:,} rows in {n_chunks} "
        f"chunks, {kept_rows:,} kept, {elapsed:.1f}s",
        flush=True,
    )

    # Concatenate each gene's parts once (empty -> empty frame with right cols).
    result: dict[str, pd.DataFrame] = {}
    for gene, frames in parts.items():
        if frames:
            result[gene] = pd.concat(frames, ignore_index=True)
        else:
            result[gene] = pd.DataFrame(
                {
                    "x_location": pd.Series(dtype="float32"),
                    "y_location": pd.Series(dtype="float32"),
                }
            )
    return result


def _bin_density(xy: pd.DataFrame, bin_um: int) -> pd.DataFrame:
    """Square-bin transcript locations at ``bin_um`` and area-normalize.

    Bin identity = integer bin index on each axis; ``hx``/``hy`` are the bin
    CENTER coordinates in µm (stable, snapped to the global origin at 0,0 so a
    bin means the same physical square across genes). ``density`` = count /
    bin_area (transcripts per µm²), so coarser bins are not falsely hotter.
    Returns cols hx, hy, count, density (empty frame if no transcripts).
    """
    if xy.empty:
        return pd.DataFrame(
            {
                "hx": pd.Series(dtype="float64"),
                "hy": pd.Series(dtype="float64"),
                "count": pd.Series(dtype="int64"),
                "density": pd.Series(dtype="float64"),
            }
        )

    x = xy["x_location"].to_numpy(dtype="float64")
    y = xy["y_location"].to_numpy(dtype="float64")

    # Snap to a global origin at (0, 0) µm so bin edges are identical across
    # genes and reproducible run-to-run. Xenium coordinates are non-negative.
    ix = np.floor(x / bin_um).astype(np.int64)
    iy = np.floor(y / bin_um).astype(np.int64)

    # Count per (ix, iy) via a compact groupby.
    binned = pd.DataFrame({"ix": ix, "iy": iy})
    counts = (
        binned.groupby(["ix", "iy"], sort=True).size().reset_index(name="count")
    )

    bin_area = float(bin_um) * float(bin_um)  # µm²
    out = pd.DataFrame(
        {
            "hx": (counts["ix"].to_numpy() + 0.5) * bin_um,  # center x, µm
            "hy": (counts["iy"].to_numpy() + 0.5) * bin_um,  # center y, µm
            "count": counts["count"].to_numpy(dtype="int64"),
            "density": counts["count"].to_numpy(dtype="float64") / bin_area,
        }
    )
    return out


def _sha256(path: Path) -> str:
    """SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    _assert_inputs_exist()
    DENSITY_DIR.mkdir(parents=True, exist_ok=True)

    demo_markers = list(config.DEMO_MARKERS)
    demo_set = frozenset(demo_markers)
    print(
        f"[precompute_density] {len(demo_markers)} demo markers, bins "
        f"{BIN_SIZES_UM} µm, qv>= {QV_MIN}",
        flush=True,
    )
    print(f"[precompute_density] markers: {demo_markers}", flush=True)

    # ---- single streaming pass ----
    per_gene_xy = _stream_demo_transcripts(demo_set)

    # ---- bin + write per (gene, bin) ----
    index_pairs: list[dict] = []
    manifest_files: dict[str, dict] = {}
    per_gene_totals: dict[str, int] = {}
    genes_with_zero: list[str] = []

    for gene in demo_markers:  # preserve config order
        xy = per_gene_xy.get(gene, pd.DataFrame())
        kept = int(len(xy))
        per_gene_totals[gene] = kept
        if kept == 0:
            genes_with_zero.append(gene)

        for bin_um in BIN_SIZES_UM:
            frame = _bin_density(xy, bin_um)
            out_path = DENSITY_DIR / f"{gene}_{bin_um}um.parquet"
            frame.to_parquet(out_path, index=False)

            rel = out_path.relative_to(config.PROJECT_ROOT).as_posix()
            index_pairs.append(
                {
                    "gene": gene,
                    "bin_um": bin_um,
                    "file": rel,
                    "n_bins": int(len(frame)),
                    "n_transcripts": kept,
                }
            )
            manifest_files[rel] = {
                "gene": gene,
                "bin_um": bin_um,
                "n_bins": int(len(frame)),
                "n_transcripts": kept,
                "sha256": _sha256(out_path),
            }
            print(
                f"  wrote {out_path.name}: {len(frame):,} bins "
                f"({kept:,} transcripts)",
                flush=True,
            )

    # ---- index.json ----
    generated_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    index_doc = {
        "generated_at": generated_at,
        "dataset": config.DATASET_ID,
        "source": config.TRANSCRIPTS_CSV_GZ.relative_to(
            config.PROJECT_ROOT
        ).as_posix(),
        "qv_min": QV_MIN,
        "control_prefixes": list(CONTROL_PREFIXES),
        "bin_sizes_um": list(BIN_SIZES_UM),
        "normalization": "density = count / bin_area (transcripts per um^2)",
        "bin_scheme": "square bins snapped to origin (0,0); hx/hy are bin centers in um",
        "genes": demo_markers,
        "n_genes": len(demo_markers),
        "genes_with_zero_transcripts": genes_with_zero,
        "per_gene_transcript_counts": per_gene_totals,
        "pairs": index_pairs,
    }
    with open(INDEX_JSON, "w") as fh:
        json.dump(index_doc, fh, indent=2)
    print(f"[precompute_density] wrote {INDEX_JSON}", flush=True)

    # ---- PREP_MANIFEST.json (density fragment; merge-friendly) ----
    density_fragment = {
        "generated_at": generated_at,
        "source": index_doc["source"],
        "qv_min": QV_MIN,
        "n_files": len(manifest_files),
        "per_gene_transcript_counts": per_gene_totals,
        "files": manifest_files,
    }
    if MANIFEST_JSON.exists():
        try:
            with open(MANIFEST_JSON) as fh:
                manifest = json.load(fh)
        except (json.JSONDecodeError, OSError):
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest["density"] = density_fragment
    with open(MANIFEST_JSON, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(
        f"[precompute_density] wrote density fragment into {MANIFEST_JSON}",
        flush=True,
    )

    # ---- summary ----
    total_transcripts = sum(per_gene_totals.values())
    print("\n[precompute_density] SUMMARY", flush=True)
    print(
        f"  genes: {len(demo_markers)}  bins: {list(BIN_SIZES_UM)}  "
        f"files: {len(manifest_files)}",
        flush=True,
    )
    print(f"  total kept transcripts: {total_transcripts:,}", flush=True)
    if genes_with_zero:
        print(f"  WARNING zero-transcript genes: {genes_with_zero}", flush=True)


if __name__ == "__main__":
    main()
