"""Stage: annotate — derive the cell type per cluster with the marker-gene skill (LIVE).

For a dataset WITHOUT ``interp/annotation.json``, this runs the jazzpanda-markers
skill per cluster: given the cluster's top jazzPanda markers, the skill assigns the
cell type, lineage/category, a short name, and the canonical lineage markers -- the
skill's Output 2 (the per-cluster interpretation Panoscope shows). The off-panel
canonical set (the panel-absence rule's input) is then computed by checking those
canonical markers against the dataset's own panel, so it is grounded per dataset.

Resumable: an existing annotation.json is read, not regenerated, so the bundled demo
(which ships its annotation) never re-annotates. Fail-soft per cluster: a failed or
unparseable call yields an honest "Unknown" record rather than crashing the pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from agent import annotation as agent_annotation
from agent import config as cfg
from agent import data
from agent import loop as agent_loop
from pipeline import paths

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
        f'"lineage": "...", "canonical_markers": ["GENE1", "GENE2", "..."]}}. '
        f"canonical_markers = the established lineage markers for that cell type (gene "
        f"symbols only). Do NOT include any jazzPanda numbers or a PMID."
    )


def _parse(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or not all(k in obj for k in _REQUIRED):
        return None
    return obj


def _record(cluster: str, obj: Optional[dict], markers: list[tuple[str, float, float]]) -> dict:
    """Build the annotation record; compute off-panel canonical vs THIS dataset's panel."""
    if obj is None:
        top = [g for g, _, _ in markers[:5]]
        return {
            "cluster": cluster, "cell_type": "Unknown", "cell_type_short": "Unknown",
            "category": "Unknown", "lineage": "Unknown",
            "canonical_markers": top, "offpanel_canonical": [],
        }
    canon = [str(g) for g in (obj.get("canonical_markers") or [])]
    offpanel = [g for g in canon if not data.panel_contains(g)]  # grounded: never measured
    return {
        "cluster": cluster,
        "cell_type": str(obj["cell_type"]),
        "cell_type_short": str(obj["cell_type_short"]),
        "category": str(obj["category"]),
        "lineage": str(obj["lineage"]),
        "canonical_markers": canon,
        "offpanel_canonical": offpanel,
    }


def _load_existing(out_path: Path) -> dict[str, dict]:
    if not out_path.exists():
        return {}
    try:
        raw = json.loads(out_path.read_text(encoding="utf-8"))
        clusters = raw.get("clusters", raw)
        return clusters if isinstance(clusters, dict) else {}
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


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
    dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None, force: bool = False
) -> dict[str, dict]:
    """Generate (resumable) the per-cluster skill annotation into interp/annotation.json.

    Reads an existing file and only fills clusters that are missing or "Unknown"
    (unless ``force``). Clears the annotation cache so downstream stages see it.
    """
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
        clusters[cluster] = _record(cluster, obj, markers)
        print(f"  {cluster}: {clusters[cluster]['cell_type']}", flush=True)
        _save(out_path, dataset_id, clusters)

    agent_annotation.load_annotation.cache_clear()
    return clusters


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Annotate clusters with the marker-gene skill (live).")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    ap.add_argument("--force", action="store_true", help="re-annotate every cluster")
    args = ap.parse_args()
    result = run_annotate(args.dataset, force=args.force)
    named = sum(1 for v in result.values() if v.get("cell_type") not in ("Unknown", None))
    print(f"[annotate] {named}/{len(result)} clusters annotated")
