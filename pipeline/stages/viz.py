"""Collect the precomputed viz frames into the dataset tree (migration stage).

Until the from-source reshape + density stages exist, this MOVES the already
precomputed viz frames out of the legacy flat ``data/`` layout into
``data/datasets/<id>/viz/`` so the tree is self-contained and the path-shim in
``agent.data`` / ``ui.data_access`` reads them from there:

    data/cells/cells.parquet         -> viz/cells.parquet
    data/embeddings/umap.parquet     -> viz/umap.parquet
    data/embeddings/marker_expr.parquet -> viz/expr.parquet
    data/density/{gene}_{bin}um.parquet + _index.json -> viz/hexbin/

Idempotent and non-destructive: a frame already in the tree is left alone; a
missing legacy frame is skipped (that view is simply absent, never faked). Moving
(not copying) keeps the repo size flat.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from agent import config as cfg

from pipeline import paths


def collect_viz(dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None) -> Path:
    """Move legacy precomputed viz frames into ``<id>/viz/``. Returns the viz dir."""
    vdir = paths.viz_dir(dataset_id, root)
    hexdir = vdir / "hexbin"
    vdir.mkdir(parents=True, exist_ok=True)
    hexdir.mkdir(exist_ok=True)

    # Move whichever variant a frame exists as (this demo ships cells/umap as
    # CSV, expr as parquet). The fat marker_expr.csv source is left in place (the
    # parquet is authoritative and read first).
    moves = [
        (cfg.DATA_DIR_PATH / "cells" / "cells.parquet", vdir / "cells.parquet"),
        (cfg.DATA_DIR_PATH / "cells" / "cells.csv", vdir / "cells.csv"),
        (cfg.DATA_DIR_PATH / "embeddings" / "umap.parquet", vdir / "umap.parquet"),
        (cfg.DATA_DIR_PATH / "embeddings" / "umap.csv", vdir / "umap.csv"),
        (cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.parquet", vdir / "expr.parquet"),
    ]
    for src, dst in moves:
        if src.exists() and not dst.exists():
            shutil.move(str(src), str(dst))

    legacy_density = cfg.DATA_DIR_PATH / "density"
    if legacy_density.exists():
        for f in sorted(legacy_density.glob("*")):
            if f.is_file():
                dst = hexdir / f.name
                if not dst.exists():
                    shutil.move(str(f), str(dst))

    return vdir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Collect precomputed viz frames into the tree.")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    args = ap.parse_args()
    vdir = collect_viz(args.dataset)
    n_hex = len(list((vdir / "hexbin").glob("*.parquet")))
    print(f"[viz] collected into {vdir} ({n_hex} hexbin frames)")
