"""Enrichment stage — compute + persist the per-cluster enrichment (deterministic).

The enrichment mirror of ``stages/verdicts.py``, one step behind it. It writes
**twice**, and the two halves are not symmetric:

  * ``interp/enrichment/c{n}.json`` + ``interp/enrichment.csv`` — the v1 view, still
    produced by ``agent/enrichment.py`` itself. Byte-identical, because it is
    literally the same code: ``ClusterEnrichment.confidence`` is reproduced
    faithfully (``core/export.py`` and ``core/compat.py`` both record why that band
    on a CHARACTERISATION survives until the method refactor deletes it).
  * ``interp/records/characterisation/c{n}.json`` — the ``InterpretationRecord``,
    built through ``platforms/tidy`` -> ``methods/jazzpanda_cgst`` ->
    ``workflows/enrichment`` -> ``core/record``, which is exactly the composition
    ``pipeline/records.py`` runs for the marker half. Nothing reads these yet.

**Why the v1 half was not cut over, said here rather than left to be rediscovered.**
``verdicts.py`` could swap engines because ``core/compat.to_v1_assignment`` projects
v2 back to the committed bytes. There is no such projection for enrichment
(``core/compat.py`` says so outright), and there cannot be a faithful one: the two
engines disagree about the measurement. v1 reads ``n_overlap``, the method's own
count of a set's genes on the panel; ``jazzpanda_cgst._in_frame`` instead counts
``genes_tested`` against EVERY gene frame, so the 49 of 280 demo genes with no usable
spatial signal stop counting toward a set's footprint. On the committed demo that
moves ``panel_hits`` on half the rows and the band on three of nine units. Swapping
the engine would therefore move a committed CSV and the golden hash -- an
interpretation change, not a refactor -- so the divergence is **measured and printed**
instead, and :func:`run_enrichment` is the one place that evidence exists.

Computes nothing new. Both halves read the biologist's enrichment result; the stage
arranges and serialises.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from panoscope.agent import config as cfg
from panoscope.agent import enrichment as agent_enrichment
from panoscope.agent.types import ClusterEnrichment, PathwayEvidence

from panoscope.pipeline import paths
from panoscope.pipeline.serialize import enrichment_to_dict

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
    csv_path.write_text(agent_enrichment.to_csv(enrichments), encoding="utf-8")

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

    Deliberately not wrapped in a ``try``. A study whose enrichment table this method
    cannot read is a loud failure of ``panoscope build``, which is the point: the
    bugs this cutover work keeps turning up were all silent, and a swallowed
    exception here would leave the v2 half quietly unbuilt while the Pathways tab
    went on rendering from v1.
    """
    import pandas as pd

    from panoscope import methods, workflows
    from panoscope.core import record as core_record
    from panoscope.platforms.tidy import read_tidy
    from panoscope.platforms.xenium import XeniumAdapter
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
    adapter = XeniumAdapter()          # no platform registry exists yet; see report
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
        record_path=_relpath(path, study.dir),
    ).value.record

    loaded = method.load(str(path), adapter, study)
    cell_types = {ce.cluster: ce.cell_type for ce in enrichments}

    written: dict[str, Path] = {}
    for unit in sorted(loaded.value):
        ev = loaded.value[unit]
        finding = wf.interpret(
            ev,
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


def _relpath(path: Path, base: Path) -> str:
    import os

    return os.path.relpath(path, base)


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


def _band_under_v2(ce: ClusterEnrichment, hits: dict[str, int]) -> str | None:
    """The band this unit would carry under the v2 count, or ``None`` if unmoved.

    ``agent/enrichment._band_cluster`` is CALLED, not reproduced. It is the reference
    implementation of the enrichment ladder and this stage must not grow a second one
    -- a band on a characterisation computed in a pipeline stage is exactly what the
    no-band rule exists to catch. Nothing computed here is written anywhere; it
    answers one question, "would the committed band move", and only the reference can
    answer that about itself.
    """
    import dataclasses

    top: PathwayEvidence | None = ce.enriched[0] if ce.enriched else None
    if top is None:
        return None
    k = hits.get(top.gene_set)
    if k is None or k == top.panel_hits:
        return None
    moved = dataclasses.replace(
        top, panel_hits=k,
        panel_coverage=(k / top.set_size_full) if top.set_size_full else 0.0)
    band, _ = agent_enrichment._band_cluster(moved, ce.method)
    return band if band != ce.confidence else None


def _report_divergence(
    enrichments: list[ClusterEnrichment],
    evidence: dict,
    interp: Path,
) -> None:
    """Say out loud, every build, how far apart the two engines are.

    Always printed, including when nothing differs: silence has to mean "they agree"
    and never "nobody looked". This is the whole evidence base for the real cutover's
    go/no-go -- without it, the capability frame quietly disagreeing with
    ``n_overlap`` is one more silent divergence, which is the shape every bug this
    work has turned up has had.
    """
    rows = differing = tier_edge = 0
    moves: list[str] = []
    for ce in enrichments:
        ev = evidence.get(ce.cluster)
        if ev is None:
            continue
        hits = _panel_hits(ev)
        for p in ce.all_tested:
            k = hits.get(p.gene_set)
            if k is None:
                continue
            rows += 1
            if k == p.panel_hits:
                continue
            differing += 1
            # The one threshold that moves a row between tiers, and so adds or
            # removes a line from the committed CSV.
            if (k >= cfg.MIN_PANEL_HITS) != (p.panel_hits >= cfg.MIN_PANEL_HITS):
                tier_edge += 1
        band = _band_under_v2(ce, hits)
        if band is not None:
            moves.append(f"{ce.cluster} {ce.confidence}->{band}")

    only_v1 = sorted({ce.cluster for ce in enrichments} - set(evidence))
    only_v2 = sorted(set(evidence) - {ce.cluster for ce in enrichments})

    print(
        f"[enrichment] capability frame vs n_overlap: panel_hits differs on "
        f"{differing} of {rows} (unit, set) rows; {tier_edge} of those cross the "
        f"panel-hit floor; the v1 band would move on {len(moves)} of "
        f"{len(enrichments)} units"
        + (f" ({'; '.join(moves)})" if moves else "")
        + ". The v1 view on disk is unchanged -- this is the go/no-go evidence for "
        f"the cutover, not a cutover. v2 records: {interp / 'records'}."
    )
    if only_v1 or only_v2:
        print(
            f"[enrichment] the two engines do not agree on the unit set: v1-only "
            f"{only_v1 or '(none)'}, v2-only {only_v2 or '(none)'}. The comparison "
            "above covers only the units both of them found."
        )
