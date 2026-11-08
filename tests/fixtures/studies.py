"""``make_study``: write a minimal, valid study folder from slices of committed inputs.

Scenarios get realistic inputs without depending on the demo's committed OUTPUTS.

* ``source="demo"``: the real Xenium breast inputs (280-gene panel, jazzPanda
  ``markers_top.csv``, the cluster key, the lab reference list). Only the marker
  method has a real raw input committed, so only ``methods=("jazzpanda",)``.
* ``source="synthetic"``: ``synthetic_multisample_fixture`` (made-up genes, says so in
  its manifest), the only committed raw inputs for enrichment (``jazzpanda_cgst``) and
  DE (``edger``). Use it for enrichment-only and DE-only shapes.

Nothing here builds; the caller runs ``pipeline.run`` (or opens the app) with
``PANOSCOPE_DATA_ROOT`` at ``root`` -- the ``study_builder`` fixture does that.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

STUDIES = Path(__file__).resolve().parents[2] / "data" / "studies"
SOURCES = {"demo": "xenium_hbreast_sample1", "synthetic": "synthetic_multisample_fixture"}
METHOD_FILES = {"jazzpanda": "markers_top.csv", "jazzpanda_cgst": "enrichment.csv",
                "edger": "de_edger.csv"}

# Real, canonical kidney markers. On the demo's 280-gene breast panel, 2 of 13 are
# measured (PECAM1, VWF): the panel-absence rule at near-maximum severity.
KIDNEY_LAB_MARKERS = [
    ("Podocyte", "NPHS1"), ("Podocyte", "NPHS2"), ("Podocyte", "PODXL"),
    ("Proximal tubule", "LRP2"), ("Proximal tubule", "SLC34A1"), ("Proximal tubule", "CUBN"),
    ("Thick ascending limb", "UMOD"), ("Thick ascending limb", "SLC12A1"),
    ("Distal convoluted tubule", "SLC12A3"), ("Principal cell", "AQP2"),
    ("Intercalated cell", "ATP6V1B1"), ("Endothelial", "PECAM1"), ("Endothelial", "VWF"),
]


def make_study(root: Path, study_id: str = "study", *, source: str = "demo",
               clusters: list[str] | None = None, panel: list[str] | None = None,
               labels: bool | dict[str, str] = True, tissue: str = "human breast cancer",
               platform: str = "Xenium", samples: list[str] | None = None,
               clustering: str | None = None, methods: tuple[str, ...] = ("jazzpanda",),
               lab_markers: list[tuple[str, str]] | None = None) -> Path:
    """Write ``root/studies/<study_id>/`` and return that directory.

    ``clusters`` keeps those clusters' rows (and every NoSig gene); ``panel`` keeps
    those genes. ``labels``: True keeps the source labels, False removes every label
    (no ``cell_type`` column, no cluster key), a dict labels only the clusters it names.
    ``samples`` writes ``inputs/samples/<s>/`` copies of the marker table and declares
    them; ``clustering`` declares the mode (omit it to leave it undeclared).
    """
    src = STUDIES / SOURCES[source] / "inputs"
    if missing := [m for m in methods if not (src / METHOD_FILES[m]).exists()]:
        raise ValueError(f"no committed {source} input for {missing}")
    inputs = root / "studies" / study_id / "inputs"
    inputs.mkdir(parents=True)

    pan = pd.read_parquet(src / "panel.parquet")
    if panel is not None:
        pan = pan[pan["gene"].isin(panel)]
    pan.to_parquet(inputs / "panel.parquet", index=False)

    for m in methods:
        df = pd.read_csv(src / METHOD_FILES[m])
        col = "top_cluster" if m == "jazzpanda" else "cluster"
        if clusters is not None:
            df = df[df[col].isin([*clusters, "NoSig"])]
        if "gene" in df.columns:
            df = df[df["gene"].isin(pan["gene"])]
        if m == "jazzpanda":
            df = _label(df, labels)
        df.to_csv(inputs / METHOD_FILES[m], index=False)
        if m == "jazzpanda" and samples:
            for s in samples:
                (inputs / "samples" / s).mkdir(parents=True)
                df.to_csv(inputs / "samples" / s / METHOD_FILES[m], index=False)

    key = src / "cluster_key.json"
    if labels is True and key.exists():
        k = json.loads(key.read_text())
        keep = set(clusters or k)
        (inputs / "cluster_key.json").write_text(
            json.dumps({c: v for c, v in k.items() if c in keep}, indent=2))

    (inputs / "reference").mkdir()
    if lab_markers is None:
        shutil.copytree(src / "reference", inputs / "reference", dirs_exist_ok=True)
    else:
        pd.DataFrame(lab_markers, columns=["cell_type", "gene"]).assign(tissue=tissue).to_csv(
            inputs / "reference" / "lab_markers.csv", index=False)

    man = {"dataset_id": study_id, "tissue": tissue, "platform": platform}
    if samples:
        man["samples"] = list(samples)
    if clustering:
        man["clustering_mode"] = clustering
    (inputs.parent / "manifest.json").write_text(json.dumps(man, indent=2))
    return inputs.parent


def _label(df: pd.DataFrame, labels: bool | dict[str, str]) -> pd.DataFrame:
    if labels is True:
        return df
    df = df.drop(columns="cell_type", errors="ignore")
    if isinstance(labels, dict):
        df = df.assign(cell_type=df["top_cluster"].map(labels))
    return df


def make_kidney_standin(root: Path, study_id: str = "kidney_standin") -> Path:
    """The real 280-gene panel declared as kidney, labelless, with a lab list of real
    kidney markers almost all off-panel. Replace with the real kidney study when it lands."""
    return make_study(root, study_id, tissue="human kidney", labels=False,
                      lab_markers=KIDNEY_LAB_MARKERS)
