"""Stage 0-prep: raw inputs -> tidy per-dataset inputs (the .Rds boundary).

Panoscope's Python pipeline consumes TIDY files under ``data/datasets/<id>/inputs/``;
it never reads a Seurat/jazzPanda ``.Rds`` at runtime. This stage is the documented
seam that produces those tidy files from the raw objects, by shelling out to the R
prep (``scripts/prep_data.R``). It is **read-if-present**: when the required tidy
inputs already exist (the bundled demo, or a dataset a collaborator already prepped),
it is a no-op and R is never invoked.

The tidy input contract (what the pipeline needs; produced by the R prep):

  inputs/markers_top.csv   REQUIRED  jazzPanda top_result: one row per gene, columns
                                     gene, top_cluster (c1..cN | NoSig), glm_coef,
                                     pearson, max_gg_corr, max_gc_corr[, cell_type]
  inputs/panel.parquet     REQUIRED  the analyzed panel: column `gene` (the absence
                                     primitive); optional ensembl_id, annotation
  inputs/enrichment.csv    optional  jazzPanda gene-set enrichment result (Pathways)
  viz/cells.(parquet|csv)  optional  cell x/y + cluster (Cell map)
  viz/umap.(parquet|csv)   optional  UMAP coords + cluster (UMAP view)

A dataset that ships these needs no R at all. A dataset with only ``.Rds`` objects
runs the R prep here (Rscript required on PATH); if R is unavailable the stage fails
with the contract above so the gap is explicit, never silent.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from agent import config as cfg
from pipeline import paths

_R_PREP_SCRIPT = cfg.PROJECT_ROOT / "scripts" / "prep_data.R"
_REQUIRED = ("markers_top.csv", "panel.parquet")


def _inputs_dir(dataset_id: str) -> Path:
    # Inputs always live in the canonical dataset tree; ``root`` is for outputs only.
    return paths.inputs_dir(dataset_id, None)


def _tidy_inputs_present(dataset_id: str) -> bool:
    idir = _inputs_dir(dataset_id)
    return all((idir / name).exists() for name in _REQUIRED)


def _contract_error(dataset_id: str, idir: Path) -> FileNotFoundError:
    return FileNotFoundError(
        f"[prep] dataset {dataset_id!r} is missing tidy inputs in {idir}.\n"
        f"Provide (or produce with scripts/prep_data.R):\n"
        f"  inputs/markers_top.csv  (gene, top_cluster, glm_coef, pearson, max_gg_corr, max_gc_corr)\n"
        f"  inputs/panel.parquet    (column 'gene')\n"
        f"  inputs/enrichment.csv   (optional; jazzPanda enrichment result)\n"
        f"  viz/cells.*, viz/umap.* (optional; spatial views)"
    )


def run_prep(dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None) -> None:
    """Ensure the dataset's tidy inputs exist. No-op if present; else run the R prep.

    Read-if-present: when ``inputs/markers_top.csv`` + ``inputs/panel.parquet`` are
    already there, returns immediately (R is never invoked). Otherwise runs the R
    prep to convert the raw ``.Rds`` objects into the tidy contract; raises a clear,
    actionable error if R is unavailable or the prep did not produce the inputs.
    """
    if _tidy_inputs_present(dataset_id):
        return

    idir = _inputs_dir(dataset_id)
    rscript = shutil.which("Rscript")
    if rscript is None or not _R_PREP_SCRIPT.exists():
        raise _contract_error(dataset_id, idir)

    idir.mkdir(parents=True, exist_ok=True)
    print(f"[prep] {dataset_id}: tidy inputs absent -> running {_R_PREP_SCRIPT.name}", flush=True)
    subprocess.run(
        [rscript, str(_R_PREP_SCRIPT), "--dataset", dataset_id],
        check=True,
        cwd=str(cfg.PROJECT_ROOT),
    )
    if not _tidy_inputs_present(dataset_id):
        raise _contract_error(dataset_id, idir)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prepare a dataset's tidy inputs (raw .Rds -> tidy).")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    args = ap.parse_args()
    run_prep(args.dataset)
    print(f"[prep] {args.dataset}: tidy inputs ready")
