"""The dataset manifest: one file that indexes a pipeline run.

``manifest.json`` records dataset metadata, input provenance (filename + sha256) and
every derived artifact with its sha256. Timestamps are metadata only.

**A build may not eat a declaration.** :data:`DERIVED_KEYS` is the closed list of
what a build owns; every other key in the previous manifest (the biologist's
declarations, e.g. the §6.3 ``samples`` / ``clustering_mode``) is carried forward.
"""

from __future__ import annotations

from typing import Any

from panoscope.platforms.base import digest

PIPELINE_VERSION = "0.1.0"


#: Streaming sha256 of a file's bytes (hex); one implementation, the reader's.
sha256_file = digest


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
    tree-relative path -> {sha256, ...}. ``samples`` and ``clustering_mode`` are
    written only when declared, never invented. Every key of ``declared`` (the
    existing manifest) outside :data:`DERIVED_KEYS` survives the rebuild.

    Raises ``ValueError`` on an undeclared ``tissue`` or ``platform`` (§6.3 blockers).
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
