"""On-disk layout for a per-dataset pipeline output tree.

    data/studies/<id>/
    ├── manifest.json          # dataset index + provenance + artifact hashes
    ├── inputs/                # raw provenance, copied + hashed
    ├── viz/                   # per-cell viz tables (later slices)
    └── interp/
        ├── verdicts.csv       # canonical 11-column annotation CSV
        └── clusters/c{n}.json # full per-cluster verdict (+ evidence + opening)

All helpers take an optional ``root`` so tests can point the tree at a tmp dir;
default is ``data/studies`` under the repo. Pure path math — no filesystem I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from panoscope.agent import config as cfg

def datasets_root(root: Optional[Path] = None) -> Path:
    """``root`` if given, else the ACTIVE data root's ``studies/`` -- resolved now.

    **This used to be a module constant**, ``cfg.DATA_DIR_PATH / "studies"``
    evaluated at import. ``cfg.DATA_DIR_PATH`` is dynamic (``config.__getattr__``
    resolves it against ``PANOSCOPE_DATA_ROOT`` on every read) and this froze it to
    whatever the environment said the first time anything imported this module.

    Two things followed, and both were silent. A test that correctly redirects the
    root -- ``monkeypatch.setenv`` followed by ``layout.clear_cache()`` -- still had
    ``ui/chat_store`` write ``chat_threads.json`` into the REAL repository tree,
    gitignored, under a study id that exists nowhere; the suite did that on every
    run. And ``ui/report._dataset_undecidables`` read one store's records while the
    verdicts beside it followed ``cfg.current_study()``, so a document exported from
    a study opened outside ``data/studies/`` either lost its replicate caveats or,
    with a same-named store in the frozen root, attached THAT store's sentences to
    these rows by unit id.

    Resolving per call is the whole fix: every helper below routes through here.
    """
    return Path(root) if root is not None else cfg.DATA_DIR_PATH / "studies"


def __getattr__(name: str):
    """``DEFAULT_DATASETS_ROOT`` still resolves -- live, never frozen."""
    if name == "DEFAULT_DATASETS_ROOT":
        return datasets_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


# --------------------------------------------------------------------------- #
# Summary working-space edits (the biologist's hand-authored report prose). A
# runtime artifact the UI writes as the lab edits the Summary page — a sibling of
# the interp notes, not a build output. Gitignored (runtime user state).
# --------------------------------------------------------------------------- #
def summary_edits_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "summary_edits.json"


# --------------------------------------------------------------------------- #
# Sign-off / review state (the biologist's review checkmarks on the Summary
# board). Runtime user state, a sibling of ``summary_edits`` — records which
# calls the biologist has reviewed and accepted, never a computed value.
# --------------------------------------------------------------------------- #
def review_state_json(dataset_id: str, root: Optional[Path] = None) -> Path:
    return interp_dir(dataset_id, root) / "review_state.json"
