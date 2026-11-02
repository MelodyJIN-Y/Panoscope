"""Where things live on disk. Pure path math — no filesystem I/O beyond existence
probes, no module-level globals, nothing that raises.

This module exists because deriving the data root from ``__file__`` couples the
install layout to the data layout. That coupling broke once already when the package
moved under ``panoscope/``, and it is what makes a ``pip install`` of this tool unable
to find a study.

Resolution order for the data root, first match wins:

1. an explicit argument — the caller knows best
2. ``$PANOSCOPE_DATA_ROOT`` — the lever for an institutional or shared store
3. ``./data`` relative to the current working directory, if it exists — a source checkout
4. the repo root inferred from this file — the fallback that keeps a source checkout
   working from any cwd

Nothing here raises. A caller that needs the root to *exist* checks for itself and
reports, so a missing store is a named error at the call site rather than an import
crash three modules away.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

ENV_DATA_ROOT = "PANOSCOPE_DATA_ROOT"
ENV_STUDY = "PANOSCOPE_DATASET"  # historical name; a study id

_DATA_DIRNAME = "data"
_STUDIES_DIRNAME = "studies"  # the store is data/studies/<id>


def _package_repo_root() -> Path:
    """Repo root inferred from this file: panoscope/data/layout.py -> up three."""
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=8)
def _resolve_cached(explicit: Optional[str], env: Optional[str], cwd: str) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if env:
        return Path(env).expanduser().resolve()
    local = Path(cwd) / _DATA_DIRNAME
    if local.is_dir():
        return local.resolve()
    return (_package_repo_root() / _DATA_DIRNAME).resolve()


def resolve_data_root(explicit: Optional[Path] = None) -> Path:
    """The directory containing the study store. Never raises."""
    return _resolve_cached(
        str(explicit) if explicit is not None else None,
        os.getenv(ENV_DATA_ROOT),
        os.getcwd(),
    )


def clear_cache() -> None:
    """Forget memoized resolutions. Tests that change cwd or env must call this."""
    _resolve_cached.cache_clear()


def default_study_id() -> Optional[str]:
    """The study id from the environment, if one is set. No default is invented."""
    return os.getenv(ENV_STUDY) or None


# --------------------------------------------------------------------------- #
# Study tree layout
# --------------------------------------------------------------------------- #
def studies_root(data_root: Optional[Path] = None) -> Path:
    return resolve_data_root(data_root) / _STUDIES_DIRNAME


def study_dir(study_id: str, data_root: Optional[Path] = None) -> Path:
    return studies_root(data_root) / study_id


def inputs_dir(study_id: str, data_root: Optional[Path] = None) -> Path:
    return study_dir(study_id, data_root) / "inputs"


def interp_dir(study_id: str, data_root: Optional[Path] = None) -> Path:
    return study_dir(study_id, data_root) / "interp"


def viz_dir(study_id: str, data_root: Optional[Path] = None) -> Path:
    return study_dir(study_id, data_root) / "viz"


def manifest_json(study_id: str, data_root: Optional[Path] = None) -> Path:
    return study_dir(study_id, data_root) / "manifest.json"


def available_studies(data_root: Optional[Path] = None) -> tuple[str, ...]:
    """Study ids present on disk, sorted. Empty when the store is absent."""
    root = studies_root(data_root)
    if not root.is_dir():
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if (p / "manifest.json").is_file()))
