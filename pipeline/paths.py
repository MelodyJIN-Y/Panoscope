"""On-disk layout for a per-dataset pipeline output tree.

    data/datasets/<id>/
    ├── manifest.json          # dataset index + provenance + artifact hashes
    ├── inputs/                # raw provenance, copied + hashed
    ├── viz/                   # per-cell viz tables (later slices)
    └── interp/
        ├── verdicts.csv       # canonical 11-column annotation CSV
        └── clusters/c{n}.json # full per-cluster verdict (+ evidence + opening)

All helpers take an optional ``root`` so tests can point the tree at a tmp dir;
default is ``data/datasets`` under the repo. Pure path math — no filesystem I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from agent import config as cfg

DEFAULT_DATASETS_ROOT: Path = cfg.DATA_DIR_PATH / "datasets"


def datasets_root(root: Optional[Path] = None) -> Path:
    return Path(root) if root is not None else DEFAULT_DATASETS_ROOT


def dataset_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return datasets_root(root) / dataset_id


def inputs_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return dataset_dir(dataset_id, root) / "inputs"


def viz_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return dataset_dir(dataset_id, root) / "viz"


def interp_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return dataset_dir(dataset_id, root) / "interp"


def clusters_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "clusters"


def cluster_json(dataset_id: str, cluster: str, root: Optional[Path] = None) -> Path:
    return clusters_dir(dataset_id, root) / f"{cluster}.json"


def verdicts_csv(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "verdicts.csv"


def holistic_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "holistic.json"


def calibration_md(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "calibration.md"


# --------------------------------------------------------------------------- #
# Enrichment workflow (second interpretation slice) — its own interp/ artifacts,
# parallel to the marker verdicts, never mixed in.
# --------------------------------------------------------------------------- #
def enrichment_dir(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "enrichment"


def enrichment_cluster_json(dataset_id: str, cluster: str, root: Optional[Path] = None) -> Path:
    return enrichment_dir(dataset_id, root) / f"{cluster}.json"


def enrichment_csv(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "enrichment.csv"


def pathway_themes_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "pathway_themes.json"


def pathway_notes_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "pathway_notes.json"


def manifest_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return dataset_dir(dataset_id, root) / "manifest.json"
