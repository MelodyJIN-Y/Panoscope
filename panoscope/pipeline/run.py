"""Pipeline entrypoint -- one run per dataset.

    panoscope build --study xenium_hbreast_sample1           # deterministic
    panoscope build --study xenium_hbreast_sample1 --notes   # + live notes

Deterministic path (no network): validate inputs -> copy raw inputs -> verdicts +
verdicts.csv -> interp artifacts (holistic.json + calibration.md) -> manifest.
``--notes`` also runs the one LIVE stage (real-PubMed notes) before the manifest.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from panoscope.agent import config as cfg
from panoscope.data.study import NOT_YET_CALLED, Study
from panoscope.agent import data
from panoscope.agent import enrichment as agent_enrichment
from panoscope.agent import enrichment_themes as agent_enrichment_themes
from panoscope.agent import holistic as agent_holistic

from panoscope.pipeline import calibration as calibration_mod
from panoscope.pipeline import manifest as manifest_mod
from panoscope.pipeline import paths
from panoscope.pipeline import serialize
from panoscope.pipeline.stages.enrichment import run_enrichment
from panoscope.pipeline.stages.prep import run_prep
from panoscope.pipeline.stages.validate import validate
from panoscope.pipeline.stages.verdicts import run_verdicts
from panoscope.reference import view as reference_view

# Legacy raw-input sources copied into <id>/inputs/ for provenance.
_RAW_SOURCES: dict[str, Path] = {
    "markers_top.csv": cfg.DATA_DIR_PATH / "jazzpanda" / "markers_top.csv",
    "panel.parquet": cfg.DATA_DIR_PATH / "panels" / "panel.parquet",
    "cluster_key.json": cfg.DATA_DIR_PATH / "cluster_key.json",
}
# `tissue` and `platform` are declarations, not derivations: read from the
# study's declarations, and build_manifest refuses an empty one.


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _declared(dataset_id: str, root: Optional[Path]) -> dict[str, Any]:
    """The biologist's declarations for this study, read before the build rewrites them.

    The output tree's manifest if there is one, else the canonical tree's. Returned
    whole so :func:`manifest.build_manifest` carries every declared key forward.
    """
    for mp in (paths.manifest_json(dataset_id, root), paths.manifest_json(dataset_id, None)):
        try:
            raw = json.loads(mp.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict) and raw:
            return raw
    return {}


def _has_enrichment_result() -> bool:
    """True when this study ships the enrichment table the Pathways slice reads.

    An existence check, not ``except FileNotFoundError``: the stage raises that same
    class for real failures, which must not be swallowed. Uses the stage's own
    resolver so the check and the stage read one path.
    """
    return agent_enrichment._jz_enrichment_csv().exists()


def _workflows() -> tuple[str, ...]:
    """The workflows whose inputs the active study ships, read off the method registry."""
    from panoscope import methods
    from panoscope.workflows.enrichment import WORKFLOW_ID as ENRICHMENT

    found = {m.workflow for m in
             methods.recognise_inputs(cfg.current_study().inputs).values() if m}
    if _has_enrichment_result():  # the demo's result lives outside inputs/
        found.add(ENRICHMENT)
    return tuple(sorted(found))


def _copy_inputs(dataset_id: str, root: Optional[Path]) -> dict[str, dict[str, Any]]:
    """Hash the dataset's own inputs for provenance.

    A dataset's ``inputs/`` files are the source of truth and are NEVER overwritten.
    Only the bundled demo bootstraps a missing input from the legacy global file
    (so a fresh clone works); any other dataset uses exactly what it shipped.
    """
    idir = paths.inputs_dir(dataset_id, root)
    idir.mkdir(parents=True, exist_ok=True)
    is_demo = dataset_id == cfg.BUNDLED_DEMO_ID
    prov: dict[str, dict[str, Any]] = {}
    for name, legacy_src in _RAW_SOURCES.items():
        dst = idir / name
        if is_demo and not dst.exists() and legacy_src.exists():
            shutil.copyfile(legacy_src, dst)  # bootstrap the bundled demo only
        if dst.exists():
            prov[name] = {"file": name, "sha256": manifest_mod.sha256_file(dst)}
    extra = idir / "enrichment.csv"  # per-dataset enrichment result, if provided
    if extra.exists():
        prov["enrichment.csv"] = {"file": "enrichment.csv", "sha256": manifest_mod.sha256_file(extra)}
    for p in sorted(idir.glob("de_*.csv")):  # every DE table the stage builds from
        prov[p.name] = {"file": p.name, "sha256": manifest_mod.sha256_file(p)}
    return prov


def _write_deterministic_interp(dataset_id: str, verdicts, root: Optional[Path]) -> None:
    """Write the deterministic Step-4 interp artifacts (no network).

    ``holistic.json`` — the cross-cluster review (coherence notes + the one
    grounded refinement), serialized faithfully. ``calibration.md`` — the
    commit-vs-flag calibration table, a pure projection of ``verdicts``. Both are
    Tier A: they read only jazzPanda output / the cluster key, never the network.
    """
    idir = paths.interp_dir(dataset_id, root)
    idir.mkdir(parents=True, exist_ok=True)
    review = agent_holistic.holistic_review()
    paths.holistic_json(dataset_id, root).write_text(
        json.dumps(serialize.holistic_to_dict(review), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths.calibration_md(dataset_id, root).write_text(
        calibration_mod.calibration_markdown(list(verdicts)) + "\n", encoding="utf-8"
    )


def _run_enrichment(dataset_id: str, root: Optional[Path]) -> None:
    """The enrichment slice (second workflow). A FileNotFoundError from INSIDE the
    stage is a real failure and reaches the caller, never swallowed."""
    run_enrichment(dataset_id, root)
    themes = agent_enrichment_themes.pathway_themes()
    paths.pathway_themes_json(dataset_id, root).write_text(
        json.dumps(serialize.pathway_themes_to_dict(themes), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _annotation_complete(dataset_id: str, root: Optional[Path]) -> bool:
    """True iff every cluster already has a named cell type, or the proposal already ran.

    A cluster the proposal left not yet called is the biologist's to call, so an
    existing ``annotation.json`` in either tree ends the asking.

    Checks the annotation the app/verdict actually read (``agent.annotation``,
    which resolves the dataset tree or the bundled fallback) rather than a specific
    output root, so the bundled demo skips the live annotate stage even when the
    pipeline writes into a fresh/temporary tree.
    """
    from panoscope.agent import annotation as agent_annotation

    if any((paths.interp_dir(dataset_id, r) / "annotation.json").exists()
           for r in (root, None)):
        return True
    ann = agent_annotation.load_annotation(dataset_id)
    return bool(ann) and all(
        str(ann.get(c, {}).get("cell_type")) not in ("", "Unknown", "None", NOT_YET_CALLED)
        for c in cfg.CLUSTER_ORDER
    )


def _ensure_annotation(dataset_id: str, root: Optional[Path]) -> None:
    """Ensure the per-cluster cell-type annotation exists (marker-gene skill, LIVE).

    Read-if-present: when ``interp/annotation.json`` is already there (the bundled
    demo ships one), this is a no-op and never imports the agent loop. Only a new
    dataset triggers the live annotate stage, whose labels are a proposal.
    """
    if _annotation_complete(dataset_id, root):
        return
    from panoscope.pipeline.stages import annotate as annotate_stage  # lazy (pulls the agent loop)

    annotate_stage.run_annotate(dataset_id, root)


def _run_notes(dataset_id: str, root: Optional[Path]) -> None:
    """Run the LIVE notes stages (cell-type + per-marker biology + per-pathway, PubMed).

    Imported lazily so the deterministic pipeline never pulls in the agent loop /
    MCP session. Resumable and fail-soft inside each stage: a slow or failed lookup
    degrades to an honest deterministic clause, never a fabricated PMID.
    """
    from panoscope.pipeline.stages import notes as notes_stage

    notes_stage.run_celltype_notes(dataset_id, root)
    notes_stage.run_gene_notes(dataset_id, root)
    if _has_enrichment_result():  # no enrichment result -> no pathway notes
        notes_stage.run_pathway_notes(dataset_id, root)


def run(
    dataset_id: str | None = None,
    root: Optional[Path] = None,
    notes: bool = False,
) -> Path:
    """Run the pipeline for ``dataset_id``; return the dataset directory.

    Deterministic by default; ``notes=True`` also runs the LIVE notes stage before
    the manifest.

    **The whole build is scoped to the study being built**: every agent-layer
    reader resolves against ``cfg.current_study()`` at call time, so without the
    scope a build would write the default study into X's directory. The scope uses
    the study's CANONICAL location; ``root`` only says where to write the tree.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    ddir = paths.dataset_dir(dataset_id, root)
    ddir.mkdir(parents=True, exist_ok=True)
    with cfg.use_study(Study.open(dataset_id)):
        return _run_scoped(dataset_id, ddir, root, notes)


def _run_de(dataset_id: str, root: Optional[Path], with_marker: bool) -> None:
    """The COMPARISON slice (third workflow), built only when a DE result exists.

    No DE table -> nothing built. A DE table no registered method can read raises.
    The marker evidence is passed in, not rebuilt: §9.7 check 4 joins DE against the
    spatial result for the SAME unit.
    """
    from panoscope.agent import config as _cfg
    from panoscope.pipeline import records
    from panoscope.pipeline.stages.de import run_de

    # ``cfg.current_study()`` for reading, exactly as run_verdicts does; ``root``
    # names where the records are written and nothing else.
    marker = {u: b.evidence
              for u, b in records.build_all(_cfg.current_study()).items()} if with_marker else {}
    run_de(marker, dataset_id, root)


def _run_scoped(dataset_id: str, ddir: Path, root: Optional[Path], notes: bool) -> Path:
    """The build itself, with ``dataset_id`` guaranteed to be the active study."""
    declared = _declared(dataset_id, root)  # read BEFORE the build rewrites the file

    from panoscope.workflows.enrichment import WORKFLOW_ID as ENRICHMENT
    from panoscope.workflows.marker import WORKFLOW_ID as MARKER

    run_prep(dataset_id, root)  # raw .Rds -> tidy inputs (no-op if already present)
    validate(dataset_id)
    workflows = _workflows()  # each workflow builds only when its inputs are present
    print(f"[pipeline] {dataset_id}: workflows available: {', '.join(workflows)}", flush=True)
    marker = MARKER in workflows
    inputs = _copy_inputs(dataset_id, root)
    vdir = paths.viz_dir(dataset_id, root)
    (vdir / "hexbin").mkdir(parents=True, exist_ok=True)
    verdicts: list = []
    if marker:
        _ensure_annotation(dataset_id, root)  # marker-skill cell types (no-op if present)
        verdicts = run_verdicts(dataset_id, root)
        _write_deterministic_interp(dataset_id, verdicts, root)  # holistic + calibration
    if ENRICHMENT in workflows:
        _run_enrichment(dataset_id, root)
    _run_de(dataset_id, root, marker)  # COMPARISON slice; an unrecognised de_*.csv raises
    if notes and marker:
        _run_notes(dataset_id, root)  # LIVE: cell-type + per-marker biology notes

    # Hash every derived artifact for the manifest.
    artifacts: dict[str, dict[str, Any]] = {}

    def _hash(p: Path) -> None:
        artifacts[str(p.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(p)}

    for v in verdicts:
        _hash(paths.cluster_json(dataset_id, v.cluster, root))
    # The v2 records (§10) are hashed like everything else: they are the artifact
    # the reproducibility claim is about, and an unhashed one is a file nobody can
    # say arrived unchanged.
    recdir = paths.interp_dir(dataset_id, root) / "records"
    for p in sorted(recdir.rglob("*.json")):
        _hash(p)
    csvp = paths.verdicts_csv(dataset_id, root)
    if marker:
        artifacts[str(csvp.relative_to(ddir))] = {
            "sha256": manifest_mod.sha256_file(csvp),
            "rows": len(verdicts),
        }
    # Interp artifacts: holistic + calibration always present (deterministic);
    # the two notes files are hashed when a --notes run produced them (or a prior
    # one left them in the tree).
    for name in (
        # The file the BAND's canonical set comes from (read-if-present), so it is
        # hashed as a provenance input.
        "annotation.json",
        "holistic.json",
        "calibration.md",
        "celltype_notes.json",
        "gene_notes.json",
        "enrichment.csv",
        "pathway_themes.json",
        "pathway_notes.json",
    ):
        f = paths.interp_dir(dataset_id, root) / name
        if f.exists():
            _hash(f)
    clusters = list(cfg.CLUSTER_ORDER)
    # Per-cluster enrichment verdicts (second workflow), when built.
    for c in clusters:
        f = paths.enrichment_cluster_json(dataset_id, c, root)
        if f.exists():
            _hash(f)
    # Small viz frames are hashed; the 840 hexbin frames are recorded by count.
    for name in ("cells.parquet", "cells.csv", "umap.parquet", "umap.csv", "expr.parquet"):
        f = vdir / name
        if f.exists():
            _hash(f)
    hexdir = vdir / "hexbin"
    if hexdir.exists():
        artifacts["viz/hexbin/"] = {"n_frames": len(list(hexdir.glob("*.parquet")))}

    try:
        panel_n = int(len(data.load_panel()))
    except Exception:  # noqa: BLE001
        panel_n = 0

    # n_cells comes from agent.data, scoped to the tree we just BUILT rather than
    # the canonical one ``run`` scoped: the viz frames live in the destination.
    with cfg.use_study(Study(id=dataset_id, data_root=ddir.parent.parent)):
        try:
            n_cells: Optional[int] = int(len(data.load_cells()))
        except (FileNotFoundError, ValueError):
            n_cells = None
    views = {
        "cell_map": (vdir / "cells.parquet").exists() or (vdir / "cells.csv").exists(),
        "umap": (vdir / "umap.parquet").exists() or (vdir / "umap.csv").exists(),
        "density": hexdir.exists() and bool(list(hexdir.glob("*.parquet"))),
        "expr": (vdir / "expr.parquet").exists(),
    }

    man = manifest_mod.build_manifest(
        dataset_id=dataset_id,
        generated_at=_now_iso(),
        tissue=str(declared.get("tissue") or ""),
        platform=str(declared.get("platform") or ""),
        samples=declared.get("samples"),
        clustering_mode=declared.get("clustering_mode"),
        declared=declared,
        panel_gene_count=panel_n,
        n_cells=n_cells,
        clusters=clusters,
        views_available=views,
        inputs=inputs,
        artifacts=artifacts,
        workflows=workflows,
        reference=dict(reference_view.for_study(
            Study(id=dataset_id, data_root=ddir.parent.parent)).releases),
    )
    paths.manifest_json(dataset_id, root).write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return ddir
