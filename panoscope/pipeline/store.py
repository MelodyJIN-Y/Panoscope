"""Read persisted per-dataset artifacts off the pipeline tree.

The reader for what the pipeline wrote. It is fail-soft by design: a missing or
malformed artifact returns ``None`` so callers fall back to the live engine and
the app never breaks on a partial tree. A returned verdict is byte-faithful to
the computed one (guaranteed by ``tests/test_pipeline.py``'s round-trip gate).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from panoscope.agent import config as cfg
from panoscope.agent.enrichment_themes import PathwayThemes
from panoscope.agent.holistic import HolisticReview
from panoscope.agent.types import ClusterEnrichment, ClusterVerdict

from panoscope.pipeline import paths
from panoscope.pipeline.serialize import (
    enrichment_from_dict,
    holistic_from_dict,
    pathway_themes_from_dict,
    verdict_from_dict,
)


def _read(p: Path, parse: Callable[[Any], Any], default: Any) -> Any:
    """``parse(json)`` of the file at ``p``, or ``default`` when it is absent or
    anything (read, JSON, or ``parse``) fails -- fail soft; the caller recomputes."""
    if not p.exists():
        return default
    try:
        return parse(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001 - fail soft; caller recomputes live
        return default


def _dict(data: Any) -> dict:
    return data if isinstance(data, dict) else {}


def load_verdict(
    cluster: str,
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[ClusterVerdict]:
    """Return the persisted verdict for ``cluster``, or None if absent/unreadable."""
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.cluster_json(dataset_id, cluster, root)
    return _read(p, verdict_from_dict, None)


def load_all_verdicts(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[list[ClusterVerdict]]:
    """Return all persisted verdicts in cluster order, or None if any is missing.

    All-or-nothing: a partial tree returns None so the caller recomputes the full
    set live rather than mixing persisted and computed calls.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    out: list[ClusterVerdict] = []
    for cluster in cfg.CLUSTER_ORDER:
        v = load_verdict(cluster, dataset_id, root)
        if v is None:
            return None
        out.append(v)
    return out


def load_holistic(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[HolisticReview]:
    """Return the persisted holistic review (Step 4), or None if absent/unreadable.

    The review is deterministic (Tier A), so the persisted object is byte-faithful
    to the freshly computed one (``tests/test_pipeline.py`` round-trip gate). The
    UI reads this tree-first and only falls back to computing it live when absent.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.holistic_json(dataset_id, root)
    return _read(p, holistic_from_dict, None)


def load_enrichment(
    cluster: str,
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[ClusterEnrichment]:
    """Return the persisted enrichment verdict for ``cluster``, or None if absent."""
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.enrichment_cluster_json(dataset_id, cluster, root)
    return _read(p, enrichment_from_dict, None)


def load_all_enrichments(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[list[ClusterEnrichment]]:
    """Return all persisted enrichment verdicts in cluster order, or None if any
    is missing (all-or-nothing, mirroring load_all_verdicts)."""
    dataset_id = dataset_id or cfg.DATASET_ID
    out: list[ClusterEnrichment] = []
    for cluster in cfg.CLUSTER_ORDER:
        e = load_enrichment(cluster, dataset_id, root)
        if e is None:
            return None
        out.append(e)
    return out


def load_pathway_themes(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> Optional[PathwayThemes]:
    """Return the persisted cross-cluster pathway themes, or None if absent."""
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.pathway_themes_json(dataset_id, root)
    return _read(p, pathway_themes_from_dict, None)


def load_celltype_notes(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict:
    """Return the per-cluster cell-type notes map, or {} if absent/unreadable.

    Shape: ``{cluster: {cell_type, summary, pmid|null, citation|null, verify}}``.
    Fail-soft: a missing file returns an empty dict so callers show a plain
    fallback rather than crashing.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.notes_json(dataset_id, "celltype_notes.json", root)
    return _read(p, _dict, {})


def load_gene_notes(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict:
    """Return the skill-grounded per-marker biology notes, or {} if absent.

    Shape: ``{cluster: {gene: {..evaluation.., summary, pmid, citation, ...}}}``.
    Fail-soft: a missing file returns {} so the caller can fall back to the legacy
    flat notes file during migration.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.notes_json(dataset_id, "gene_notes.json", root)
    return _read(p, _dict, {})


def load_summary_edits(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict:
    """Return the biologist's saved Summary-page edits ``{name: text}``, or {}.

    ``name`` is the working-space region key (a cluster id, ``"global"``, or
    ``"caveats"``). Fail-soft: a missing/malformed file returns {} so the UI just
    falls back to the freshly auto-drafted text.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.summary_edits_json(dataset_id, root)
    return _read(p, lambda d: _dict(_dict(d).get("edits")), {})


def save_summary_edits(
    edits: dict,
    dataset_id: str | None = None,
    root: Optional[Path] = None,
    saved_at: str = "",
) -> Path:
    """Persist the biologist's Summary-page edits (the ONE write in this module).

    ``edits`` is ``{name: text}`` for only the regions the biologist changed from
    the auto-draft (self-cleaning: an untouched region is absent, so it keeps
    tracking the freshest live draft). Written with the same stable formatting as
    the lab notes (indent=2, sorted keys, trailing newline). ``saved_at`` is passed
    in by the caller (an ISO string) so this stays free of clock calls.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.summary_edits_json(dataset_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_id, "saved_at": saved_at, "edits": edits}
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return p


def load_review_state(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict:
    """Return the biologist's sign-off state ``{cluster: {at, note_id}}``, or {}.

    Records which calls the biologist reviewed and accepted on the Summary board.
    ``at`` is an ISO timestamp; ``note_id`` is the id of the validation note a
    contested sign-off wrote (``None`` for a clean checkmark). Fail-soft: a
    missing/malformed file returns {} so the board just shows nothing signed off.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.review_state_json(dataset_id, root)
    return _read(p, lambda d: _dict(_dict(d).get("reviewed")), {})


def save_review_state(
    reviewed: dict,
    dataset_id: str | None = None,
    root: Optional[Path] = None,
    saved_at: str = "",
) -> Path:
    """Persist the sign-off state ``{cluster: {at, note_id}}``.

    Same stable formatting as the lab notes / summary edits (indent=2, sorted
    keys, trailing newline). ``saved_at`` is passed in by the caller (an ISO
    string) so this stays free of clock calls. A record is never a computed value —
    only which calls the biologist has reviewed.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.review_state_json(dataset_id, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset_id, "saved_at": saved_at, "reviewed": reviewed}
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, sort_keys=True)
        fh.write("\n")
    return p


def load_pathway_notes(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict:
    """Return the live-cited per-pathway biology notes, or {} if absent.

    Shape: ``{cluster: {gene_set: {..evidence.., summary, pmid, citation, ...}}}``.
    Fail-soft: a missing file returns {} (the Pathways table shows the grounded
    numbers with no biology prose until the notes stage has run).
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    p = paths.pathway_notes_json(dataset_id, root)
    return _read(p, _dict, {})
