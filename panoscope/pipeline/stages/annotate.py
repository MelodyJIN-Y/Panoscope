"""Stage: annotate -- derive the cell type per cluster with the marker-gene skill (LIVE).

For a dataset WITHOUT ``interp/annotation.json``, runs the marker-interpretation skill
per cluster on its top markers: cell type, lineage/category and short name. It never
writes canonical markers: those are reference data (§15), never model-written.

The result is a PROPOSAL; the biologist's call replaces it downstream. Resumable: an
existing annotation.json is read, not regenerated. Fail-soft per cluster: a failed
call writes NO record, so the cluster stays "not yet called" with an empty canonical
set -- never one filled from the measurement, since the cluster's own top markers
there would make the band the measurement agreeing with itself. ``_record`` takes no
markers, so that substitution is unrepresentable.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from panoscope.agent import annotation as agent_annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import loop as agent_loop
from panoscope.data import study as study_mod
from panoscope.pipeline import paths

_REQUIRED = ("cell_type", "cell_type_short", "category", "lineage")
_TOP_MARKERS = 8


def _cluster_markers(cluster: str) -> list[tuple[str, float, float]]:
    df = data.get_cluster_markers(cluster)  # glm_coef desc, NoSig excluded
    return [
        (str(r["gene"]), float(r["glm_coef"]), float(r["pearson"]))
        for _, r in df.head(_TOP_MARKERS).iterrows()
    ]


def _prompt(cluster: str, markers: list[tuple[str, float, float]]) -> str:
    listed = ", ".join(f"{g} (glm_coef {gc:.2f}, pearson {pe:.2f})" for g, gc, pe in markers)
    return (
        f"Cluster {cluster} top jazzPanda markers by glm_coef: {listed}. "
        f"Apply the jazzPanda-markers skill (Step 3) to assign this cluster's cell type "
        f"from its spatial marker signature. Reply with ONLY a compact JSON object and "
        f"nothing else (no prose, no code fence): "
        f'{{"cell_type": "...", "cell_type_short": "Prefix_Descriptor", "category": "...", '
        f'"lineage": "..."}}. Do NOT include any jazzPanda numbers or a PMID.'
    )


def _parse(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    # Present-but-empty is not a label. ``{"cell_type": null}`` passed the old
    # membership check and ``str(None)`` wrote the literal string "None" into the
    # label the biologist reads -- a fabricated label. An unusable label is no
    # label: the cluster stays not yet called.
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in _REQUIRED):
        return None
    return obj


def _record(cluster: str, obj: dict) -> dict:
    """Build the annotation record. The canonical slot stays empty whatever the model said.

    Takes no markers on purpose -- see the module docstring -- so no observed marker can
    be promoted into the canonical slot either.
    """
    return {
        "cluster": cluster,
        "cell_type": str(obj["cell_type"]),
        "cell_type_short": str(obj["cell_type_short"]),
        "category": str(obj["category"]),
        "lineage": str(obj["lineage"]),
        "canonical_markers": [],
        "offpanel_canonical": [],
    }


def _load_existing(out_path: Path) -> dict[str, dict]:
    """Read the annotation we are resuming, or stop.

    An existing-but-unreadable file raises rather than reading as "no annotation
    yet" and being overwritten. ``--force`` skips this to start over deliberately.
    """
    if not out_path.exists():
        return {}
    raw = json.loads(out_path.read_text(encoding="utf-8"))
    clusters = raw.get("clusters", raw) if isinstance(raw, dict) else None
    if not isinstance(clusters, dict):
        raise ValueError(
            f"[annotate] {out_path} is not a readable annotation (no 'clusters' object); "
            f"refusing to overwrite it -- fix or remove it, or re-run with --force"
        )
    return clusters


def _save(out_path: Path, dataset_id: str, clusters: dict[str, dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"dataset": dataset_id, "method": "jazzpanda-markers skill (Output 2)",
             "clusters": clusters},
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )


def run_annotate(
    dataset_id: str | None = None, root: Optional[Path] = None, force: bool = False
) -> dict[str, dict]:
    """Generate (resumable) the per-cluster skill annotation into interp/annotation.json.

    Reads an existing file and only fills clusters that are missing or "Unknown"
    (unless ``force``). Always writes the file, so a build does not re-ask for
    clusters left not yet called. Clears the annotation cache.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    out_path = paths.interp_dir(dataset_id, root) / "annotation.json"
    clusters: dict[str, dict] = {} if force else _load_existing(out_path)

    for cluster in cfg.CLUSTER_ORDER:
        have = clusters.get(cluster)
        if have and str(have.get("cell_type")) not in ("", "Unknown", "None"):
            continue
        markers = _cluster_markers(cluster)
        obj: Optional[dict] = None
        try:
            resp = agent_loop.chat(_prompt(cluster, markers), cluster=cluster)
            obj = _parse(resp.text)
        except Exception as exc:  # noqa: BLE001 - keep going; save what we have
            print(f"  ERROR {cluster}: {exc}", flush=True)
        else:
            if obj is None:
                # Unparseable is a failure too, not a result.
                print(f"  UNPARSEABLE {cluster}: {(resp.text or '')[:120]!r}", flush=True)
        if obj is None:
            clusters.pop(cluster, None)  # a stale "Unknown" is not a label either
            continue
        clusters[cluster] = _record(cluster, obj)
        print(f"  {cluster}: {clusters[cluster]['cell_type']}", flush=True)
        _save(out_path, dataset_id, clusters)
    _save(out_path, dataset_id, clusters)

    agent_annotation.load_annotation.cache_clear()
    study_mod.clear_cache()  # validate cached the label set before this wrote it
    uncalled = [c for c in cfg.CLUSTER_ORDER if c not in clusters]
    if uncalled:
        # Said once, plainly: silence is what let a fabricated canonical set look
        # like a successful annotation.
        print(
            f"  [annotate] {len(uncalled)}/{len(cfg.CLUSTER_ORDER)} clusters NOT YET "
            f"CALLED ({', '.join(uncalled)}): no proposal, the biologist calls them",
            flush=True,
        )
    return clusters


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Annotate clusters with the marker-gene skill (live).")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    ap.add_argument("--force", action="store_true", help="re-annotate every cluster")
    args = ap.parse_args()
    result = run_annotate(args.dataset, force=args.force)
    print(f"[annotate] {len(result)}/{len(cfg.CLUSTER_ORDER)} clusters proposed")
