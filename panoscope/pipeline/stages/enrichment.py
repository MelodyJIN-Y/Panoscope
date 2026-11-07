"""Enrichment stage -- compute + persist the per-cluster enrichment (deterministic).

Writes twice:

  * ``interp/enrichment/c{n}.json`` + ``interp/enrichment.csv`` -- the v1 view, from
    ``agent/enrichment.py``. No band: per-item census only.
  * ``interp/records/characterisation/c{n}.json`` -- the ``InterpretationRecord``,
    through ``platforms/tidy`` -> ``methods/jazzpanda_cgst`` ->
    ``workflows/enrichment`` -> ``core/record``, as ``pipeline/records.py`` does for
    the marker half.

Both engines count a set's panel hits against every gene frame (genes with no usable
spatial signal do not count). :func:`_report_divergence` checks they stay in step;
its leading-edge reading is expected to be non-zero while v2's ``leading_edge`` is
the raw selection.

Computes nothing new: both halves read the biologist's enrichment result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as agent_enrichment
from panoscope.agent.types import ClusterEnrichment
from panoscope.core import export

from panoscope.pipeline import paths
from panoscope.pipeline.serialize import enrichment_to_dict
from panoscope.types.outputs import TaskKind

#: The entity column of a gene-set result, and the column its statistic is IN.
#: ``UnitSpec`` declares the unit column; these are the two schema facts the caller
#: still supplies. The statistic column is the file's name for it ("test_statistic"),
#: which is NOT ``method.statistic_name`` ("cgst_z") -- the method's name for its own
#: number. ``ReadingRecord`` records what the file said, so the file's name is right
#: here. Named as constants so the gap stays visible, as in ``pipeline/records.py``.
ENTITY_COLUMN = "gene_set"
STATISTIC_COLUMN = "test_statistic"


def run_enrichment(
    dataset_id: str | None = None, root: Optional[Path] = None
) -> list[ClusterEnrichment]:
    """Compute all enrichment verdicts, write per-cluster JSON + enrichment.csv.

    Then build and write the v2 record for the same units, and report where the two
    engines disagree. The v2 half never touches the v1 files and never feeds them.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    enrichments = agent_enrichment.all_enrichments()

    edir = paths.enrichment_dir(dataset_id, root)
    edir.mkdir(parents=True, exist_ok=True)
    for e in enrichments:
        out = paths.enrichment_cluster_json(dataset_id, e.cluster, root)
        out.write_text(
            json.dumps(enrichment_to_dict(e), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    csv_path = paths.enrichment_csv(dataset_id, root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        export.to_csv(
            TaskKind.CHARACTERISATION,
            [(ce, p) for ce in enrichments for p in (*ce.enriched, *ce.suggestive)],
        ).value,
        encoding="utf-8",
    )

    write_records(enrichments, dataset_id, root)

    return enrichments


# --------------------------------------------------------------------------- #
# The v2 half. Nothing below writes into the v1 tree or is read by it.
# --------------------------------------------------------------------------- #
def write_records(
    enrichments: list[ClusterEnrichment],
    dataset_id: str | None = None,
    root: Optional[Path] = None,
) -> dict[str, Path]:
    """One ``InterpretationRecord`` per unit, under ``records/characterisation/``.

    Not wrapped in a ``try``: an enrichment table this method cannot read fails the
    build loudly rather than leaving the v2 half unbuilt.
    """
    import pandas as pd

    from panoscope import methods, platforms, workflows
    from panoscope.core import interpret as core_interpret
    from panoscope.core import record as core_record
    from panoscope.platforms.tidy import read_tidy
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.outputs import TaskKind
    from panoscope.types.subject import EntityKind
    from panoscope.workflows import enrichment as wf_enrichment

    dataset_id = dataset_id or cfg.DATASET_ID
    study = cfg.current_study()
    interp = paths.interp_dir(dataset_id, root)

    wf = workflows.get(wf_enrichment.WORKFLOW_ID)
    if wf.task_kind is TaskKind.ASSIGNMENT:
        # Not a style check. ``records/<task_kind>/<unit>.json`` is keyed on the task
        # kind, so an assignment-shaped workflow registered here would overwrite the
        # marker half's records with enrichment content, and nothing downstream
        # would notice.
        raise SystemExit(
            f"{wf.name!r} declares {wf.task_kind.value}; this stage would then write "
            f"under records/{wf.task_kind.value}/, which is the marker root's tree. "
            "Two interpretive questions cannot share one record path."
        )

    registered = methods.for_workflow(wf_enrichment.WORKFLOW_ID)
    if len(registered) != 1:
        raise SystemExit(
            f"{len(registered)} methods answer {wf_enrichment.WORKFLOW_ID!r} "
            f"({', '.join(m.name for m in registered) or 'none'}); the stage cannot "
            "pick one for you. The record names its method."
        )
    method = registered[0]

    # The SAME file the v1 engine read, resolved by the same function: two engines
    # being compared against each other must not be reading two files.
    path = agent_enrichment._jz_enrichment_csv()
    # The manifest says which technology produced this store; the registry
    # resolves the reader. Same reason as pipeline/records.py.
    adapter = platforms.for_study(study)
    table = pd.read_csv(path)
    spec = method.detect_unit(table)

    # ``scored_frame`` is NOT on the ``MethodModule`` protocol, and has to be called
    # anyway: without the GENE_SET capability frame every set in the file lands in
    # ``ReadingRecord.entities_outside_frame`` and the record asserts that no program
    # was ever measured. Named here so the protocol gap stays visible.
    world = MeasuredWorld(
        tuple(adapter.frames(study)) + (method.scored_frame(table),))

    reading = read_tidy(
        str(path),
        unit_column=spec.unit_column,
        entity_column=ENTITY_COLUMN,
        entity_kind=EntityKind.GENE_SET,
        statistic_column=STATISTIC_COLUMN,
        unit_kind=spec.unit_kind,
        world=world,
        clustering_mode=adapter.clustering_mode(study),
        not_units=frozenset(str(v) for v in table[spec.unit_column]) - set(spec.units),
        unit_detection_because=spec.because,
        # Store-relative, and ``relpath`` rather than ``relative_to``: the bundled
        # demo's enrichment result lives outside the study tree, and an absolute path
        # in a hashed record makes byte-stability a property of whose laptop produced
        # it.
        record_path=os.path.relpath(path, study.dir),
    ).value

    loaded = method.load(str(path), adapter, study)
    cell_types = {ce.cluster: ce.cell_type for ce in enrichments}

    written: dict[str, Path] = {}
    for unit in sorted(loaded.value):
        ev = loaded.value[unit]
        # Through ``core.interpret.interpret``, not ``wf.interpret`` directly: the
        # guard comparing the RETURNED shape against the declared task kind lives
        # there, and calling the workflow straight is what let it be skipped. A
        # CHARACTERISATION workflow that returned an ``Assignment`` would put a
        # unit-level band on a question with no label space -- well-formed, and
        # nothing downstream asks what question the object answers.
        finding = core_interpret.interpret(
            ev, wf,
            wf_enrichment.EnrichmentReference(cell_type=cell_types.get(unit, "")),
        )
        record = core_record.build(
            ev, reading,
            core_record.unit_decisions(ev, loaded),
            finding,
            task_kind=wf.task_kind,
        ).value
        written[unit] = core_record.write(record, interp).value

    _report_divergence(enrichments, loaded.value, interp)
    return written


def _panel_hits(ev) -> dict[str, int]:
    """``gene_set -> panel_hits``, READ OFF the v2 evidence, never recomputed here.

    The number is the ``Coverage`` the method attached at load, so the comparison
    below cannot drift from the one the written record carries.
    """
    from panoscope.workflows import enrichment as wf_enrichment

    out: dict[str, int] = {}
    for item in ev.items:
        cov = next((c for c in item.measured.coverage
                    if c.what == wf_enrichment.PANEL_COVERAGE), None)
        if cov is not None:
            out[item.subject.display()] = cov.k
    return out


def _leading_edge(ev) -> dict[str, int]:
    """``gene_set -> the v2 method's leading-edge count``, read off the evidence.

    Same discipline as :func:`_panel_hits`: the number is the ``Coverage`` the
    method attached at load, never recomputed here. ``"leading_edge"`` is the name
    ``methods/jazzpanda_cgst`` gives that coverage.
    """
    out: dict[str, int] = {}
    for item in ev.items:
        cov = next((c for c in item.measured.coverage
                    if c.what == "leading_edge"), None)
        if cov is not None:
            out[item.subject.display()] = cov.k
    return out


def _report_divergence(
    enrichments: list[ClusterEnrichment],
    evidence: dict,
    interp: Path,
) -> None:
    """Say out loud, every build, whether the two engines still agree.

    The expected ``panel_hits`` reading is zero; this is the check that it stays so.
    Always printed, so silence never means "nobody looked".
    """
    rows = differing = tier_edge = 0
    le_rows = le_genes = le_selected = 0
    disagreements: list[str] = []
    for ce in enrichments:
        ev = evidence.get(ce.cluster)
        if ev is None:
            continue
        hits = _panel_hits(ev)
        edges = _leading_edge(ev)
        for p in ce.all_tested:
            k = hits.get(p.gene_set)
            if k is None:
                continue
            rows += 1
            # How many genes the frames refused, counted the only place both numbers
            # exist side by side: v2 still reports the raw lasso selection, v1 now
            # reports the admitted subset, so the gap IS the rejection count.
            selected = edges.get(p.gene_set)
            if selected is not None and selected > p.n_leading_edge:
                le_rows += 1
                le_genes += selected - p.n_leading_edge
            if selected is not None:
                le_selected += selected
            if k == p.panel_hits:
                continue
            differing += 1
            disagreements.append(f"{ce.cluster}/{p.gene_set} v1={p.panel_hits} v2={k}")
            # The one threshold that moves a row between tiers, and so adds or
            # removes a line from the committed CSV.
            if (k >= cfg.MIN_PANEL_HITS) != (p.panel_hits >= cfg.MIN_PANEL_HITS):
                tier_edge += 1

    only_v1 = sorted({ce.cluster for ce in enrichments} - set(evidence))
    only_v2 = sorted(set(evidence) - {ce.cluster for ce in enrichments})

    print(
        f"[enrichment] capability frame, both engines: panel_hits differs on "
        f"{differing} of {rows} (unit, set) rows; {tier_edge} of those cross the "
        f"panel-hit floor"
        + (f" ({'; '.join(disagreements[:5])}"
           + (f"; +{len(disagreements) - 5} more" if len(disagreements) > 5 else "")
           + ")" if disagreements else "")
        + ". Zero is the expected reading: v1 stopped trusting the result's own "
        f"n_overlap. v2 records: {interp / 'records'}."
    )
    print(
        f"[enrichment] leading edge, capability frames: {le_genes} of {le_selected} "
        f"selected genes rejected across {le_rows} of {rows} (unit, set) rows; those "
        "genes are not counted, named or plotted as drivers. This reading is NOT "
        "expected to be zero -- it is how much of this panel's lasso selection the "
        "measurement could not usably see, which is a fact about the panel against "
        "this tissue. It is also the one place the count exists: a record rebuilt "
        "from disk carries only the admitted genes. v2's leading_edge coverage is "
        "the unfiltered selection (methods/jazzpanda_cgst), so the gap above is "
        "read off it rather than recomputed here."
    )
    if only_v1 or only_v2:
        print(
            f"[enrichment] the two engines do not agree on the unit set: v1-only "
            f"{only_v1 or '(none)'}, v2-only {only_v2 or '(none)'}. The comparison "
            "above covers only the units both of them found."
        )
