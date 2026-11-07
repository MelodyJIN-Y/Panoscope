"""Where things live on disk. Pure path math -- no filesystem I/O beyond existence
probes, no module-level globals, nothing that raises.

Data root resolution order, first match wins:

1. an explicit argument
2. the root this context opened (one Streamlit session's rerun; :func:`set_session_root`)
3. ``$PANOSCOPE_DATA_ROOT``
4. ``./data`` relative to the current working directory, if it exists
5. the repo root inferred from this file

A caller that needs the root to *exist* checks for itself and reports.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from typing import Optional

ENV_DATA_ROOT = "PANOSCOPE_DATA_ROOT"
ENV_STUDY = "PANOSCOPE_DATASET"  # historical name; a study id

_DATA_DIRNAME = "data"
_STUDIES_DIRNAME = "studies"  # the store is data/studies/<id>

_SESSION_ROOT: ContextVar[Optional[str]] = ContextVar("panoscope_data_root", default=None)


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
        str(explicit) if explicit is not None else _SESSION_ROOT.get(),
        os.getenv(ENV_DATA_ROOT),
        os.getcwd(),
    )


def set_session_root(root: Optional[Path]) -> None:
    """The data root for this context; ``None`` falls back to the environment."""
    _SESSION_ROOT.set(str(root) if root is not None else None)


def user_data_root() -> Path:
    """Where the app builds a biologist's own study: ``$PANOSCOPE_DATA_ROOT``, else ~/.panoscope/data."""
    env = os.getenv(ENV_DATA_ROOT)
    return Path(env or Path.home() / ".panoscope" / "data").expanduser().resolve()


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


def spatial_zarr(study_id: str, data_root: Optional[Path] = None) -> Path:
    """The study's SpatialData store (ARCHITECTURE §5.1).

    A SIBLING of inputs/ and not a child, because §5.2 keeps the store ("what was
    measured") separate from method results ("what someone computed from it"), and
    that boundary is what makes the confidence assessment an audit.
    """
    return study_dir(study_id, data_root) / "spatial.zarr"


def manifest_json(study_id: str, data_root: Optional[Path] = None) -> Path:
    return study_dir(study_id, data_root) / "manifest.json"


def available_studies(data_root: Optional[Path] = None) -> tuple[str, ...]:
    """Study ids present on disk, sorted. Empty when the store is absent."""
    root = studies_root(data_root)
    if not root.is_dir():
        return ()
    return tuple(sorted(p.name for p in root.iterdir() if (p / "manifest.json").is_file()))
