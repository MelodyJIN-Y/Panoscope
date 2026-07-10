"""Pipeline entrypoint — one run per dataset.

    python -m pipeline.run --dataset xenium_hbreast_sample1           # deterministic
    python -m pipeline.run --dataset xenium_hbreast_sample1 --notes   # + live notes

Deterministic path (default, no network): validate inputs (Stage 0) -> copy raw
inputs for provenance -> collect viz frames into the tree -> persist verdicts +
verdicts.csv (Stage 4) -> write the Step-4 interp artifacts (holistic.json +
calibration.md) -> write the manifest. ``--notes`` additionally runs the one LIVE
stage (real-PubMed cell-type + per-marker biology notes) before the manifest, so
those files are hashed into it too.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from agent import config as cfg
from agent import data
from agent import holistic as agent_holistic

from pipeline import calibration as calibration_mod
from pipeline import manifest as manifest_mod
from pipeline import paths
from pipeline import serialize
from pipeline.stages.validate import validate
from pipeline.stages.verdicts import run_verdicts
from pipeline.stages.viz import collect_viz

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


def _read_tree_frame(vdir: Path, stem: str) -> Optional[pd.DataFrame]:
    """Read a viz frame straight from the tree (parquet, else csv), or None.

    Reads the collected tree files directly rather than through agent.data, whose
    paths are frozen at import to the pre-move legacy location.
    """
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        p = vdir / f"{stem}{suffix}"
        if p.exists():
            try:
                return reader(p)
            except Exception:  # noqa: BLE001 - a corrupt frame is a None, not a crash
                return None
    return None


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


def _write_deterministic_interp(dataset_id: str, verdicts, root: Optional[Path]) -> None:
    """Write the deterministic Step-4 interp artifacts (no network).

    ``holistic.json`` — the cross-cluster review (coherence notes + the one
    grounded refinement), serialized faithfully. ``calibration.md`` — the
    commit-vs-flag calibration table, a pure projection of ``verdicts``. Both are
    Tier A: they read only jazzPanda output / the cluster key, never the network.
    """
    idir = paths.interp_dir(dataset_id, root)
    idir.mkdir(parents=True, exist_ok=True)
    review = agent_holistic.holistic_review()
    paths.holistic_json(dataset_id, root).write_text(
        json.dumps(serialize.holistic_to_dict(review), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths.calibration_md(dataset_id, root).write_text(
        calibration_mod.calibration_markdown(list(verdicts)) + "\n", encoding="utf-8"
    )


def _run_notes(dataset_id: str, root: Optional[Path]) -> None:
    """Run the LIVE notes stage (cell-type + per-marker biology, real PubMed).

    Imported lazily so the deterministic pipeline never pulls in the agent loop /
    MCP session. Resumable and fail-soft inside the stage: a slow or failed
    lookup degrades to an honest deterministic clause, never a fabricated PMID.
    """
    from pipeline.stages import notes as notes_stage

    notes_stage.run_celltype_notes(dataset_id, root)
    notes_stage.run_gene_notes(dataset_id, root)


def run(
    dataset_id: str = cfg.DATASET_ID,
    root: Optional[Path] = None,
    notes: bool = False,
) -> Path:
    """Run the pipeline for ``dataset_id``; return the dataset directory.

    Deterministic by default (validate -> inputs -> viz -> verdicts -> holistic +
    calibration -> manifest). Pass ``notes=True`` to also run the LIVE notes stage
    (real-PubMed cell-type + per-marker biology notes) before the manifest, so
    the notes are hashed into it too.
    """
    ddir = paths.dataset_dir(dataset_id, root)
    ddir.mkdir(parents=True, exist_ok=True)

    validate(dataset_id)
    inputs = _copy_inputs(dataset_id, root)
    vdir = collect_viz(dataset_id, root)  # ensure viz frames live in the tree
    verdicts = run_verdicts(dataset_id, root)
    _write_deterministic_interp(dataset_id, verdicts, root)  # holistic + calibration
    if notes:
        _run_notes(dataset_id, root)  # LIVE: cell-type + per-marker biology notes

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
    # Interp artifacts: holistic + calibration always present (deterministic);
    # the two notes files are hashed when a --notes run produced them (or a prior
    # one left them in the tree).
    for name in ("holistic.json", "calibration.md", "celltype_notes.json", "gene_notes.json"):
        f = paths.interp_dir(dataset_id, root) / name
        if f.exists():
            artifacts[str(f.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(f)}
    # Small viz frames are hashed; the 840 hexbin frames are recorded by count.
    for name in ("cells.parquet", "cells.csv", "umap.parquet", "umap.csv", "expr.parquet"):
        f = vdir / name
        if f.exists():
            artifacts[str(f.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(f)}
    hexdir = vdir / "hexbin"
    if hexdir.exists():
        artifacts["viz/hexbin/"] = {"n_frames": len(list(hexdir.glob("*.parquet")))}

    try:
        panel_n = int(len(data.load_panel()))
    except Exception:  # noqa: BLE001
        panel_n = 0

    # views_available and n_cells are read from the tree viz/ contents we just
    # collected — NOT via data.load_cells(), whose paths agent.data froze at
    # import (to the legacy location, before collect_viz MOVED the frames), and
    # NOT via the legacy dir, which collect_viz leaves behind empty. Both were
    # confirmed first-run manifest bugs.
    cells_frame = _read_tree_frame(vdir, "cells")
    n_cells: Optional[int] = int(len(cells_frame)) if cells_frame is not None else None
    views = {
        "cell_map": (vdir / "cells.parquet").exists() or (vdir / "cells.csv").exists(),
        "umap": (vdir / "umap.parquet").exists() or (vdir / "umap.csv").exists(),
        "density": hexdir.exists() and bool(list(hexdir.glob("*.parquet"))),
        "expr": (vdir / "expr.parquet").exists(),
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
    ap.add_argument(
        "--notes",
        action="store_true",
        help="also run the LIVE notes stage (real-PubMed cell-type + per-marker biology)",
    )
    args = ap.parse_args()
    ddir = run(args.dataset, notes=args.notes)
    print(f"[pipeline] wrote {ddir}" + (" (+ live notes)" if args.notes else ""))


if __name__ == "__main__":
    main()
