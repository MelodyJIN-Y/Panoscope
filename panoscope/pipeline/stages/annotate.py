"""Stage: annotate — derive the cell type per cluster with the marker-gene skill (LIVE).

For a dataset WITHOUT ``interp/annotation.json``, this runs the marker-interpretation
skill per cluster: given the cluster's top jazzPanda markers, the skill assigns the
cell type, lineage/category, a short name, and the canonical lineage markers -- the
skill's Output 2 (the per-cluster interpretation Panoscope shows). The off-panel
canonical set (the panel-absence rule's input) is then computed by checking those
canonical markers against the dataset's own panel, so it is grounded per dataset.

Resumable: an existing annotation.json is read, not regenerated, so the bundled demo
(which ships its annotation) never re-annotates. Fail-soft per cluster: a failed or
unparseable call yields an honest "Unknown" record rather than crashing the pipeline.

**Fail-soft means empty, never substituted.** The canonical-marker list this stage
writes is reference data: ``pipeline/records.py`` hands it straight to
``core/verdict.assess`` as the set deciding which marker may drive the band. So no
branch here may fill that slot from anything the measurement produced. The failure
path returns an EMPTY canonical set with an Unknown label, which the rubric already
reads correctly (``band:no_canonical_driver`` -> Low -> ``verify=True``). A cluster's
own top markers promoted into that slot would make every driver canonical by
construction, and the band would be the measurement agreeing with itself -- the one
thing CLAUDE.md says reference data may never do. It would fire hardest exactly where
the model struggles most: an unfamiliar panel against unfamiliar tissue. ``_record``
therefore takes no markers at all, so the fabrication is unrepresentable rather than
merely avoided.
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
from panoscope.pipeline import paths

_REQUIRED = ("cell_type", "cell_type_short", "category", "lineage")
_TOP_MARKERS = 8

# A gene symbol, loosely: a letter, then letters/digits/._-. Deliberately permissive
# about real symbols and strict about the things that are not symbols at all -- the
# prompt's own shape hints ("<gene symbol>", "..."), an empty string, a sentence.
_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")


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
        f'"lineage": "...", "canonical_markers": ["<gene symbol>", "<gene symbol>"]}}. '
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
    if not isinstance(obj, dict):
        return None
    # Present-but-empty is not a label. ``{"cell_type": null}`` passed the old
    # membership check and ``str(None)`` wrote the literal string "None" into the
    # label the biologist reads -- a fabricated label, and one the resume rule and
    # run.py::_annotation_complete both have to special-case by spelling. An
    # unusable label is no label: fall through to the honest Unknown record.
    if not all(isinstance(obj.get(k), str) and obj[k].strip() for k in _REQUIRED):
        return None
    return obj


def _canonical_markers(obj: dict) -> list[str]:
    """The gene symbols the model named as canonical. Reference data, so: strict.

    Anything that is not a list of symbol-shaped strings yields the EMPTY set rather
    than a salvaged one. ``{"canonical_markers": "ERBB2, EPCAM"}`` used to be iterated
    as a *string*, making each character a canonical marker: one-letter "genes", none
    of them on the panel, every one of them then reported to the biologist as a
    canonical marker this panel never measured. Dropping junk costs a band (empty ->
    ``band:no_canonical_driver`` -> verify); salvaging it invents markers.
    """
    raw = obj.get("canonical_markers")
    if not isinstance(raw, (list, tuple)):
        return []
    return [g.strip() for g in raw if isinstance(g, str) and _SYMBOL.match(g.strip())]


def _record(cluster: str, obj: Optional[dict]) -> dict:
    """Build the annotation record; compute off-panel canonical vs THIS dataset's panel.

    Takes no markers on purpose -- see the module docstring. With the measurement out
    of scope there is no expression in this function that could promote an observed
    marker into the canonical slot.
    """
    if obj is None:
        return {
            "cluster": cluster, "cell_type": "Unknown", "cell_type_short": "Unknown",
            "category": "Unknown", "lineage": "Unknown",
            "canonical_markers": [], "offpanel_canonical": [],
        }
    canon = _canonical_markers(obj)
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
    """Read the annotation we are resuming, or stop.

    An unreadable annotation.json used to come back as ``{}``, which is indistinguishable
    from "no annotation yet": the stage then re-annotated every cluster and ``_save``
    overwrote the damaged file with whatever the model said this time. A stored record
    is not something to silently replace, so an existing-but-unreadable file raises
    here. ``--force`` skips this function and is the way to deliberately start over.
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
    (unless ``force``). Clears the annotation cache so downstream stages see it.
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
                # Unparseable is a failure too, and it used to look like a result.
                print(f"  UNPARSEABLE {cluster}: {(resp.text or '')[:120]!r}", flush=True)
        clusters[cluster] = _record(cluster, obj)
        print(f"  {cluster}: {clusters[cluster]['cell_type']}", flush=True)
        _save(out_path, dataset_id, clusters)

    agent_annotation.load_annotation.cache_clear()
    unknown = [c for c, v in clusters.items() if str(v.get("cell_type")) == "Unknown"]
    if unknown:
        # Say it once, plainly: these units carry no canonical set, so their band is
        # Low with verify=True by rubric. Silence here is what let a fabricated
        # canonical set look like a successful annotation.
        print(
            f"  [annotate] {len(unknown)}/{len(clusters)} clusters UNANNOTATED "
            f"({', '.join(unknown)}): no canonical markers, band Low, verify set",
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
    named = sum(1 for v in result.values() if v.get("cell_type") not in ("Unknown", None))
    print(f"[annotate] {named}/{len(result)} clusters annotated")
