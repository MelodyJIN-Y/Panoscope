"""Pipeline entrypoint — one run per dataset.

    python -m pipeline.run --dataset xenium_hbreast_sample1

Slice 1 stages: validate inputs (Stage 0) -> copy raw inputs for provenance ->
persist verdicts + verdicts.csv (Stage 4) -> write the manifest. Deterministic
end to end (no network). Later slices add the viz precompute and live-cited notes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from agent import config as cfg
from agent import data

from pipeline import manifest as manifest_mod
from pipeline import paths
from pipeline.stages.validate import validate
from pipeline.stages.verdicts import run_verdicts

# Legacy raw-input sources copied into <id>/inputs/ for provenance. (In a later
# slice tissue/platform become per-dataset metadata inputs rather than constants.)
_RAW_SOURCES: dict[str, Path] = {
    "markers_top.csv": cfg.DATA_DIR_PATH / "jazzpanda" / "markers_top.csv",
    "panel.parquet": cfg.DATA_DIR_PATH / "panels" / "panel.parquet",
    "cluster_key.json": cfg.DATA_DIR_PATH / "cluster_key.json",
}
_TISSUE = "human breast"
_PLATFORM = "Xenium"


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _ok(loader: Callable[[], Any]) -> bool:
    try:
        loader()
        return True
    except Exception:  # noqa: BLE001 - a missing view is a False, not a crash
        return False


def _copy_inputs(dataset_id: str, root: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Copy the raw inputs into <id>/inputs/ and hash them for provenance."""
    idir = paths.inputs_dir(dataset_id, root)
    idir.mkdir(parents=True, exist_ok=True)
    prov: dict[str, dict[str, Any]] = {}
    for name, src in _RAW_SOURCES.items():
        if not src.exists():
            continue
        dst = idir / name
        shutil.copyfile(src, dst)
        prov[name] = {"file": name, "sha256": manifest_mod.sha256_file(dst)}
    return prov


def run(dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None) -> Path:
    """Run the pipeline for ``dataset_id``; return the dataset directory."""
    ddir = paths.dataset_dir(dataset_id, root)
    ddir.mkdir(parents=True, exist_ok=True)

    validate(dataset_id)
    inputs = _copy_inputs(dataset_id, root)
    verdicts = run_verdicts(dataset_id, root)

    # Hash every derived artifact for the manifest.
    artifacts: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        p = paths.cluster_json(dataset_id, v.cluster, root)
        artifacts[str(p.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(p)}
    csvp = paths.verdicts_csv(dataset_id, root)
    artifacts[str(csvp.relative_to(ddir))] = {
        "sha256": manifest_mod.sha256_file(csvp),
        "rows": len(verdicts),
    }

    try:
        panel_n = int(len(data.load_panel()))
    except Exception:  # noqa: BLE001
        panel_n = 0
    try:
        n_cells: Optional[int] = int(len(data.load_cells()))
    except Exception:  # noqa: BLE001
        n_cells = None

    views = {
        "cell_map": _ok(data.load_cells),
        "umap": _ok(data.load_umap),
        "density": (cfg.DATA_DIR_PATH / "density").exists(),
        "expr": (cfg.DATA_DIR_PATH / "embeddings" / "marker_expr.parquet").exists(),
    }

    man = manifest_mod.build_manifest(
        dataset_id=dataset_id,
        generated_at=_now_iso(),
        tissue=_TISSUE,
        platform=_PLATFORM,
        panel_gene_count=panel_n,
        n_cells=n_cells,
        clusters=list(cfg.CLUSTER_ORDER),
        views_available=views,
        inputs=inputs,
        artifacts=artifacts,
    )
    paths.manifest_json(dataset_id, root).write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return ddir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Panoscope per-dataset pipeline.")
    ap.add_argument("--dataset", default=cfg.DATASET_ID, help="dataset id")
    args = ap.parse_args()
    ddir = run(args.dataset)
    print(f"[pipeline] wrote {ddir}")


if __name__ == "__main__":
    main()
