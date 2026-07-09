"""The dataset manifest: one file that indexes a pipeline run.

``manifest.json`` records dataset metadata, input provenance (filename + sha256
of each copied raw input), and every derived artifact with its sha256 — so a run
is reproducible and a migration can prove the copied inputs are byte-identical to
the originals. Timestamps are metadata only and do not affect any grounded value.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

PIPELINE_VERSION = "0.1.0"


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file's bytes (hex)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    *,
    dataset_id: str,
    generated_at: str,
    tissue: str,
    platform: str,
    panel_gene_count: int,
    n_cells: int | None,
    clusters: list[str],
    views_available: dict[str, bool],
    inputs: dict[str, dict[str, Any]],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the manifest dict. ``generated_at`` is an ISO-8601 string.

    ``inputs`` maps a logical name -> {file, sha256}; ``artifacts`` maps a
    tree-relative path -> {sha256, ...}. Both are built by the caller.
    """
    return {
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "pipeline_version": PIPELINE_VERSION,
        "tissue": tissue,
        "platform": platform,
        "panel_gene_count": panel_gene_count,
        "n_cells": n_cells,
        "clusters": list(clusters),
        "views_available": dict(views_available),
        "inputs": dict(inputs),
        "artifacts": dict(artifacts),
    }
