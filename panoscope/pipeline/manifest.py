"""The dataset manifest: one file that indexes a pipeline run.

``manifest.json`` records dataset metadata, input provenance (filename + sha256
of each copied raw input), and every derived artifact with its sha256 — so a run
is reproducible and a migration can prove the copied inputs are byte-identical to
the originals. Timestamps are metadata only and do not affect any grounded value.

**A build may not eat a declaration.** The manifest holds two kinds of field. Most
are *derived*: the pipeline recomputes them from the files on every run and whatever
was there before is stale by definition. A few are *declared*: the biologist wrote
them, no file can produce them, and ``samples`` / ``clustering_mode`` are the two
§6.3 blockers among them. Rebuilding the whole dict from scratch — which is what this
used to do — made declaring a clustering mode impossible in practice: the first build
ate the declaration and the second raised ``UndeclaredClustering`` against the file it
had just emptied, so a jointly-clustered study could be deadlocked or silently
stamped ``independent``, and nothing in between. :data:`DERIVED_KEYS` is therefore the
closed list of what a build owns; every other key in the previous manifest is carried
forward untouched.
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


DERIVED_KEYS: frozenset[str] = frozenset({
    "dataset_id", "generated_at", "pipeline_version", "panel_gene_count",
    "n_cells", "clusters", "views_available", "inputs", "artifacts",
})
"""The keys a build recomputes from the files. Everything else was declared."""


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
    samples: list[str] | None = None,
    clustering_mode: str | None = None,
    declared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict. ``generated_at`` is an ISO-8601 string.

    ``inputs`` maps a logical name -> {file, sha256}; ``artifacts`` maps a
    tree-relative path -> {sha256, ...}. Both are built by the caller.

    ``samples`` and ``clustering_mode`` are the biologist's §6.3 declarations and
    are written only when they were declared: a build that invented either would be
    making a measurement claim on the biologist's behalf, and a single-element
    ``samples`` written from a fallback is exactly the false "one sample, therefore
    independent clustering" this pipeline must never record.

    ``declared`` is the manifest that was already there. Every key in it outside
    :data:`DERIVED_KEYS` survives the rebuild.

    Raises ``ValueError`` on an undeclared ``tissue`` or ``platform``. They block the
    study (§6.3), ``tissue`` is the corpus scoping key, and the alternative to failing
    here was the module constant that would have stamped the kidney study as breast.
    """
    if not str(tissue or "").strip():
        raise ValueError(
            f"study {dataset_id!r} declares no tissue, and tissue blocks the study "
            "(§6.3) — it is the key the corpus is scoped by, so a missing one is a "
            "literature search against nothing. Declare `tissue` in manifest.json."
        )
    if not str(platform or "").strip():
        raise ValueError(
            f"study {dataset_id!r} declares no platform, and platform blocks the "
            "study (§6.3): it is what selects the adapter that declares the "
            "capability frame. Declare `platform` in manifest.json."
        )

    built: dict[str, Any] = {
        "dataset_id": dataset_id,
        "generated_at": generated_at,
        "pipeline_version": PIPELINE_VERSION,
        "tissue": tissue,
        "platform": platform,
    }
    if samples is not None:
        built["samples"] = [str(s) for s in samples]
    if clustering_mode is not None:
        built["clustering_mode"] = str(clustering_mode)
    built.update(
        panel_gene_count=panel_gene_count,
        n_cells=n_cells,
        clusters=list(clusters),
        views_available=dict(views_available),
        inputs=dict(inputs),
        artifacts=dict(artifacts),
    )

    for key, value in (declared or {}).items():
        if key not in DERIVED_KEYS and key not in built:
            built[key] = value
    return built
