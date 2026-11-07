"""The ``Study`` object: what replaces the module-level dataset singleton.

Frozen and hashable, so it is the cache key: readers cached per study cannot serve
one study's data for another. Construction never touches the filesystem and never
raises; everything derived is computed on first access and cached per study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from panoscope.data import layout


class StudyNotFound(FileNotFoundError):
    """Raised by callers that require a study to exist. Never raised on construction."""


@dataclass(frozen=True, slots=True)
class Study:
    """One study on disk. Hashable by (id, data_root), so it is a safe cache key."""

    id: str
    data_root: Path

    # ---- construction ----------------------------------------------------- #
    @classmethod
    def open(cls, study_id: Optional[str] = None, data_root: Optional[Path] = None) -> "Study":
        """Resolve a study. Env is read at CALL time, never at import time."""
        sid = study_id or layout.default_study_id()
        if not sid:
            available = layout.available_studies(data_root)
            if len(available) == 1:
                sid = available[0]
            else:
                raise StudyNotFound(
                    "No study specified and none could be inferred. "
                    f"Set {layout.ENV_STUDY}, or pass study_id. "
                    f"Available: {', '.join(available) if available else '(none found)'}"
                )
        return cls(id=sid, data_root=layout.resolve_data_root(data_root))

    # ---- paths ------------------------------------------------------------ #
    @property
    def dir(self) -> Path:
        return layout.study_dir(self.id, self.data_root)

    @property
    def inputs(self) -> Path:
        return layout.inputs_dir(self.id, self.data_root)

    @property
    def interp(self) -> Path:
        return layout.interp_dir(self.id, self.data_root)

    @property
    def viz(self) -> Path:
        return layout.viz_dir(self.id, self.data_root)

    @property
    def spatial(self) -> Path:
        """The SpatialData store. May not exist -- a study can be tidy-only."""
        return layout.spatial_zarr(self.id, self.data_root)

    @property
    def exists(self) -> bool:
        return layout.manifest_json(self.id, self.data_root).is_file()

    def require(self) -> "Study":
        """Assert the study is on disk, with a message naming what is missing."""
        if not self.exists:
            raise StudyNotFound(
                f"Study {self.id!r} has no manifest at {layout.manifest_json(self.id, self.data_root)}. "
                f"Available: {', '.join(layout.available_studies(self.data_root)) or '(none)'}"
            )
        return self

    # ---- derived, cached per study ---------------------------------------- #
    @property
    def manifest(self) -> dict[str, Any]:
        return _manifest(self)

    @property
    def units(self) -> tuple[str, ...]:
        """Unit ids in natural order. Derived, never a constant."""
        return _units(self)

    @property
    def unit_labels(self) -> dict[str, dict[str, str]]:
        """unit id -> metadata. Resolution order is the one validate.py documents:
        the annotation written by the pipeline first, the legacy cluster key second,
        empty third. This is what kills the three-way cluster_key fallback."""
        return _unit_labels(self)

    @property
    def tissue(self) -> Optional[str]:
        return self.manifest.get("tissue") or None

    @property
    def platform(self) -> Optional[str]:
        return self.manifest.get("platform") or None

    @property
    def n_cells(self) -> Optional[int]:
        v = self.manifest.get("n_cells")
        return int(v) if isinstance(v, (int, float)) else None

    @property
    def views_available(self) -> dict[str, bool]:
        v = self.manifest.get("views_available")
        return dict(v) if isinstance(v, dict) else {}

    def label_for(self, unit: str) -> str:
        return str(self.unit_labels.get(unit, {}).get("cell_type", "Unknown"))


# --------------------------------------------------------------------------- #
# Cached derivations. Keyed on the Study, so caching is per study by construction.
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


@lru_cache(maxsize=32)
def _manifest(study: Study) -> dict[str, Any]:
    raw = _read_json(layout.manifest_json(study.id, study.data_root))
    return raw if isinstance(raw, dict) else {}


@lru_cache(maxsize=32)
def _unit_labels(study: Study) -> dict[str, dict[str, str]]:
    ann = _read_json(study.interp / "annotation.json")
    if isinstance(ann, dict):
        clusters = ann.get("clusters", ann)
        if isinstance(clusters, dict) and clusters:
            return {str(k): dict(v) for k, v in clusters.items() if isinstance(v, dict)}

    key = _read_json(study.inputs / "cluster_key.json")
    if isinstance(key, dict) and key:
        return {str(k): dict(v) for k, v in key.items() if isinstance(v, dict)}

    return {}


def _natural_key(unit: str) -> tuple[int, str]:
    digits = "".join(ch for ch in unit if ch.isdigit())
    return (int(digits) if digits else 1 << 30, unit)


@lru_cache(maxsize=32)
def _units(study: Study) -> tuple[str, ...]:
    """Prefer the manifest's declared list; fall back to the annotation's keys.

    Deliberately does NOT assert a count. A fixed unit count is exactly the kind of
    demo-shaped assumption that blocks a second study.
    """
    declared = study.manifest.get("clusters")
    if isinstance(declared, (list, tuple)) and declared:
        return tuple(str(c) for c in declared)
    labels = _unit_labels(study)
    return tuple(sorted(labels, key=_natural_key))


def clear_cache() -> None:
    """Forget every per-study derivation. Paired with layout.clear_cache()."""
    _manifest.cache_clear()
    _unit_labels.cache_clear()
    _units.cache_clear()
