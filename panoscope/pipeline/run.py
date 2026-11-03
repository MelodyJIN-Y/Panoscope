"""Pipeline entrypoint — one run per dataset.

    python -m pipeline.run --dataset xenium_hbreast_sample1           # deterministic
    python -m pipeline.run --dataset xenium_hbreast_sample1 --notes   # + live notes

Deterministic path (default, no network): validate inputs (Stage 0) -> copy raw
inputs for provenance -> collect viz frames into the tree -> persist verdicts +
verdicts.csv (Stage 4) -> write the Step-4 interp artifacts (holistic.json +
calibration.md) -> write the manifest. ``--notes`` additionally runs the one LIVE
stage (real-PubMed cell-type + per-marker biology notes) before the manifest, so
those files are hashed into it too.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from panoscope.agent import config as cfg
from panoscope.data.study import Study
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
from panoscope.pipeline.stages.viz import collect_viz

# Legacy raw-input sources copied into <id>/inputs/ for provenance.
_RAW_SOURCES: dict[str, Path] = {
    "markers_top.csv": cfg.DATA_DIR_PATH / "jazzpanda" / "markers_top.csv",
    "panel.parquet": cfg.DATA_DIR_PATH / "panels" / "panel.parquet",
    "cluster_key.json": cfg.DATA_DIR_PATH / "cluster_key.json",
}
# `tissue` and `platform` used to be module constants here -- "human breast cancer"
# and "Xenium" -- passed straight into build_manifest. They are declarations, not
# derivations: no file in a store produces them, and a constant would have stamped
# the kidney study as breast. Tissue is the corpus scoping key, so that is not a
# cosmetic error, it is a literature search scoped to the wrong organ, reported as
# a correct one. They are read from the study's declarations instead, and
# build_manifest refuses an empty one.


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _declared(dataset_id: str, root: Optional[Path]) -> dict[str, Any]:
    """The biologist's declarations for this study, read before the build rewrites them.

    The manifest already in the output tree when there is one; otherwise the study's
    own manifest in the canonical tree, which is what a run into a tmp root is
    building from. Returned whole, so :func:`manifest.build_manifest` can carry every
    declared key forward -- the §6.3 pair ``samples`` / ``clustering_mode`` above all,
    which a from-scratch rebuild used to eat, leaving the next build to raise
    ``UndeclaredClustering`` against the file it had just emptied.
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

    An existence check and not ``except FileNotFoundError``. The absence of the
    input is a condition; the stage's v2 half (``read_tidy`` -> ``adapter.frames``
    -> ``record.write``) raises the *same* exception class for an unreadable tidy
    table or a record it cannot write, and catching the pair together let a build
    report success with ``interp/records/`` silently absent -- the artifact the
    reproducibility claim is about, missing, with nothing said.

    Reaches for the private resolver on purpose: it is the one function that knows
    where a study's enrichment result lives, the stage already calls it to make sure
    both engines read one file, and re-deriving the path here is how the check and
    the stage end up disagreeing.
    """
    return agent_enrichment._jz_enrichment_csv().exists()


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
    # Enrichment slice (second workflow) — built only when this dataset HAS an
    # enrichment result. Its absence just means the Pathways slice is not built; a
    # FileNotFoundError from INSIDE the stage is a real failure and now reaches the
    # caller, because the v2 record composition runs in there and a swallowed one
    # left records/ unbuilt while the build printed success.
    if _has_enrichment_result():
        run_enrichment(dataset_id, root)
        themes = agent_enrichment_themes.pathway_themes()
        paths.pathway_themes_json(dataset_id, root).write_text(
            json.dumps(serialize.pathway_themes_to_dict(themes), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _annotation_complete(dataset_id: str, root: Optional[Path]) -> bool:
    """True iff every cluster already has a named cell type (no Unknown).

    Checks the annotation the app/verdict actually read (``agent.annotation``,
    which resolves the dataset tree or the bundled fallback) rather than a specific
    output root, so the bundled demo skips the live annotate stage even when the
    pipeline writes into a fresh/temporary tree.
    """
    from panoscope.agent import annotation as agent_annotation

    ann = agent_annotation.load_annotation(dataset_id)
    return bool(ann) and all(
        str(ann.get(c, {}).get("cell_type")) not in ("", "Unknown", "None")
        for c in cfg.CLUSTER_ORDER
    )


def _ensure_annotation(dataset_id: str, root: Optional[Path]) -> None:
    """Ensure the per-cluster cell-type annotation exists (marker-gene skill, LIVE).

    Read-if-present: when a complete ``interp/annotation.json`` is already there (the
    bundled demo ships one), this is a no-op and never imports the agent loop. Only a
    new / incomplete dataset triggers the live annotate stage.
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

    Deterministic by default (validate -> inputs -> viz -> verdicts -> holistic +
    calibration -> manifest). Pass ``notes=True`` to also run the LIVE notes stage
    (real-PubMed cell-type + per-marker biology notes) before the manifest, so
    the notes are hashed into it too.

    **The whole build is scoped to the study being built**, which is what this
    wrapper exists for. Every agent-layer reader inside -- ``cfg.CLUSTER_ORDER``,
    ``cfg.DATASET_ID``, ``data.load_panel()``, each stage's own reads -- resolves
    against ``cfg.current_study()`` at CALL time. The scope used to sit around
    ``data.load_cells()`` alone, so ``panoscope build --study X`` built the
    *environment's default* study into X's directory and exited 0: nine demo
    clusters and a 280-gene panel written under a three-cluster, twelve-gene
    fixture, every number tracing honestly to method output -- the wrong
    specimen's. That is the confident floor satisfied per value and violated per
    study, which is its worst form, and nothing printed. Scoping here rather than
    in ``cli.py`` covers every caller: the CLI, ``python -m panoscope.pipeline.run``
    and the tests.

    The scope resolves the study at its CANONICAL location, not under ``root``.
    ``root`` means *write the tree here*, not *read it from here*: the inputs are
    copied out of the canonical tree by ``_copy_inputs`` and the destination is
    empty when the build starts, so pointing the readers at it would block every
    build on its own missing manifest.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    ddir = paths.dataset_dir(dataset_id, root)
    ddir.mkdir(parents=True, exist_ok=True)
    with cfg.use_study(Study.open(dataset_id)):
        return _run_scoped(dataset_id, ddir, root, notes)


def _run_de(dataset_id: str, root: Optional[Path]) -> None:
    """The COMPARISON slice (third workflow), built only when a DE result exists.

    Same shape as the enrichment slice: a study with no DE table simply does not get
    one, and ``run_de`` returns an empty dict rather than raising. What is NOT
    swallowed is a DE table no registered method can read -- that reaches the caller,
    because a build printing success over an unbuilt ``records/comparison/`` is the
    failure the enrichment stage already learned not to have.

    The marker evidence is passed in rather than rebuilt: ARCHITECTURE §9.7's check 4
    joins the DE result against the spatial one for the SAME unit, and deriving the
    spatial side twice would be two places it could differ.
    """
    from panoscope.agent import config as _cfg
    from panoscope.pipeline import records
    from panoscope.pipeline.stages.de import run_de

    # ``cfg.current_study()`` for reading, exactly as run_verdicts does; ``root``
    # names where the records are written and nothing else.
    marker = {u: b.evidence
              for u, b in records.build_all(_cfg.current_study()).items()}
    run_de(marker, dataset_id, root)


def _run_scoped(dataset_id: str, ddir: Path, root: Optional[Path], notes: bool) -> Path:
    """The build itself, with ``dataset_id`` guaranteed to be the active study."""
    declared = _declared(dataset_id, root)  # read BEFORE the build rewrites the file

    run_prep(dataset_id, root)  # raw .Rds -> tidy inputs (no-op if already present)
    validate(dataset_id)
    inputs = _copy_inputs(dataset_id, root)
    vdir = collect_viz(dataset_id, root)  # ensure viz frames live in the tree
    _ensure_annotation(dataset_id, root)  # marker-skill cell types (no-op if present)
    verdicts = run_verdicts(dataset_id, root)
    _write_deterministic_interp(dataset_id, verdicts, root)  # holistic + calibration
    _run_de(dataset_id, root)  # COMPARISON slice, when this dataset HAS a DE result
    if notes:
        _run_notes(dataset_id, root)  # LIVE: cell-type + per-marker biology notes

    # Hash every derived artifact for the manifest.
    artifacts: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        p = paths.cluster_json(dataset_id, v.cluster, root)
        artifacts[str(p.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(p)}
    # The v2 records (§10) are hashed like everything else: they are the artifact
    # the reproducibility claim is about, and an unhashed one is a file nobody can
    # say arrived unchanged.
    recdir = paths.interp_dir(dataset_id, root) / "records"
    for p in sorted(recdir.rglob("*.json")):
        artifacts[str(p.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(p)}
    csvp = paths.verdicts_csv(dataset_id, root)
    artifacts[str(csvp.relative_to(ddir))] = {
        "sha256": manifest_mod.sha256_file(csvp),
        "rows": len(verdicts),
    }
    # Interp artifacts: holistic + calibration always present (deterministic);
    # the two notes files are hashed when a --notes run produced them (or a prior
    # one left them in the tree).
    for name in (
        # The file the BAND comes from, and it was hashed nowhere. ``annotation.json``
        # is read-if-present: ``_ensure_annotation`` regenerates it only when absent,
        # so for the bundled demo -- and for every rebuild after the first -- it is a
        # provenance input that ``records.build_all`` turns into the ``canonical`` set
        # ``core_verdict.assess`` reads. Emptying one cluster's canonical_markers moves
        # c1 from Very High / "glm_coef direct" / verify=False to Low / "no canonical
        # driver" / verify=True, and nothing in the manifest recorded that the file had
        # changed. inputs/cluster_key.json, the FALLBACK for the same labels, was hashed;
        # the primary source was not.
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
            artifacts[str(f.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(f)}
    # Per-cluster enrichment verdicts (second workflow), when built.
    for c in cfg.CLUSTER_ORDER:
        f = paths.enrichment_cluster_json(dataset_id, c, root)
        if f.exists():
            artifacts[str(f.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(f)}
    # Small viz frames are hashed; the 840 hexbin frames are recorded by count.
    for name in ("cells.parquet", "cells.csv", "umap.parquet", "umap.csv", "expr.parquet"):
        f = vdir / name
        if f.exists():
            artifacts[str(f.relative_to(ddir))] = {"sha256": manifest_mod.sha256_file(f)}
    hexdir = vdir / "hexbin"
    if hexdir.exists():
        artifacts["viz/hexbin/"] = {"n_frames": len(list(hexdir.glob("*.parquet")))}

    try:
        panel_n = int(len(data.load_panel()))
    except Exception:  # noqa: BLE001
        panel_n = 0

    # n_cells comes from agent.data, scoped to the tree we just BUILT rather than
    # the canonical one ``run`` scoped: collect_viz has by now MOVED the frames into
    # it, so this is the only read in the build that must look at the destination.
    #
    # This used to go through a private _read_tree_frame() helper, because
    # agent.data froze its paths at import -- to the legacy location, before
    # collect_viz moved the frames -- which produced two confirmed first-run
    # manifest bugs. Now that the readers resolve against the current study at call
    # time, re-scoping for this one read is enough and the workaround is gone.
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
        clusters=list(cfg.CLUSTER_ORDER),
        views_available=views,
        inputs=inputs,
        artifacts=artifacts,
    )
    paths.manifest_json(dataset_id, root).write_text(
        json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return ddir


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Panoscope per-dataset pipeline.")
    ap.add_argument("--dataset", default=cfg.DATASET_ID, help="dataset id")
    ap.add_argument(
        "--notes",
        action="store_true",
        help="also run the LIVE notes stage (real-PubMed cell-type + per-marker biology)",
    )
    args = ap.parse_args()
    ddir = run(args.dataset, notes=args.notes)
    print(f"[pipeline] wrote {ddir}" + (" (+ live notes)" if args.notes else ""))


if __name__ == "__main__":
    main()
