"""Convert the R-written marker_expr.csv (cluster-stratified subsample, all 280
panel genes) to parquet — R here has no ``arrow``. The parquet is the committed
artifact (small, fresh-clone-friendly); the fat CSV is gitignored.

Run AFTER scripts/prep_data.R:
  .venv/bin/python scripts/prep_expr.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
_CSV = _ROOT / "data" / "embeddings" / "marker_expr.csv"
_PARQUET = _ROOT / "data" / "embeddings" / "marker_expr.parquet"


def main() -> None:
    if not _CSV.exists():
        raise FileNotFoundError(
            f"[prep_expr] {_CSV} missing — run scripts/prep_data.R first."
        )
    df = pd.read_csv(_CSV)
    if "cell_id" not in df.columns:
        raise ValueError("[prep_expr] marker_expr.csv has no 'cell_id' column")
    df.to_parquet(_PARQUET, index=False)
    print(
        f"[prep_expr] wrote {_PARQUET}: {len(df):,} rows x {df.shape[1]} cols "
        f"(csv {_CSV.stat().st_size/1e6:.1f}MB -> parquet "
        f"{_PARQUET.stat().st_size/1e6:.1f}MB)"
    )


if __name__ == "__main__":
    main()
