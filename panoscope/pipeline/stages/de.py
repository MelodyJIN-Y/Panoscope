"""The COMPARISON composition root: *does this result make sense?*

Composes ``workflows/de.py`` (its four §9.7 checks and ``core/reconcile.py``'s join),
as ``stages/verdicts.py`` does ASSIGNMENT and ``stages/enrichment.py``
CHARACTERISATION.

**The method is chosen by SCHEMA, not by filename**, through ``methods.recognise``.
Every ``de_*.csv`` and every table a DE method recognises is built; the schema names
the method, and that method's declared ``statistic_name`` is the column read.

**Check 4 takes the marker evidence** passed in, not rebuilt here.

Checks that need inputs no file declares (affected process, per-arm cell share,
per-replicate numbers) refuse and name what would settle them; this root supplies
no defaults to make them pass.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from panoscope.agent import config as cfg
from panoscope.pipeline import paths

#: Where a DE result lives, by convention. The glob names the KIND of result; which
#: METHOD produced it is read off the columns by the registry (see module docstring).
DE_GLOB: str = "de_*.csv"

ENTITY_COLUMN: str = "gene"


def de_result_paths(study: Any) -> list[Path]:
    """Every DE table this study ships, sorted; empty when it has none (not an error).

    A table a DE method recognises counts under any name, since ``run`` lists the
    workflow as available off the same recognition."""
    from panoscope import methods
    from panoscope.workflows.de import WORKFLOW_ID

    recognised = {p for p, m in methods.recognise_inputs(study.inputs).items()
                  if m is not None and m.workflow == WORKFLOW_ID}
    return sorted(recognised | set(Path(study.inputs).glob(DE_GLOB)))


def _method_for(path: Path, table: Any) -> Any:
    """The registered DE method that recognises this table. Never guesses."""
    from panoscope import methods
    from panoscope.workflows.de import WORKFLOW_ID

    method = methods.recognise(table)
    if method is None or method.workflow != WORKFLOW_ID:
        known = ", ".join(m.name for m in methods.for_workflow(WORKFLOW_ID))
        raise SystemExit(
            f"{path.name}: unrecognised -- no registered DE method ({known}) reads "
            f"its columns {list(table.columns)}"
        )
    return method


def _spatial_side(marker_ev: Any, world: Any) -> Any:
    """The ASSIGNMENT side of the cross-method join, from the marker build.

    Built in the composition root: a workflow may not reach for another workflow's
    result. ``Authority.MEASUREMENT`` -- reference data may never corroborate (§7).
    """
    from panoscope.core.reconcile import Side
    from panoscope.types.claim import Authority
    from panoscope.types.outputs import TaskKind
    from panoscope.workflows.marker import WORKFLOW_ID as MARKER_ID

    return Side(module=marker_ev.module, workflow=MARKER_ID,
                task_kind=TaskKind.ASSIGNMENT, unit=marker_ev.unit,
                authority=Authority.MEASUREMENT, items=marker_ev.items, world=world)


def _workflow() -> Any:
    """The registered COMPARISON workflow, resolved through the registry.

    Through ``workflows.get`` rather than imported as an object, so an unregistered
    workflow fails here with the registry's own message instead of this root quietly
    composing a question nobody declared.
    """
    from panoscope import workflows
    from panoscope.types.outputs import TaskKind
    from panoscope.workflows.de import WORKFLOW_ID

    wf = workflows.get(WORKFLOW_ID)
    if wf.task_kind is not TaskKind.COMPARISON:
        raise SystemExit(
            f"{wf.name!r} declares {wf.task_kind.value}, but this root writes "
            f"records under records/comparison/. Build a root for "
            f"{wf.task_kind.value} rather than borrowing this one."
        )
    return wf


def run_de(marker_evidence: dict[str, Any] | None = None,
           dataset_id: str | None = None,
           root: Optional[Path] = None) -> dict[str, dict[str, Path]]:
    """One ``InterpretationRecord`` per unit per DE table, keyed by the table's stem,
    under ``records/comparison/<stem>/``.

    ``marker_evidence`` maps unit id -> that unit's marker ``NormalisedEvidence``
    (check 4). A unit without it still gets a record, with check 4 *cannot be
    settled* rather than coherent. Not wrapped in a ``try``: an unreadable DE table
    fails the build loudly.
    """
    dataset_id = dataset_id or cfg.DATASET_ID
    # The ACTIVE study for READING and ``root`` only for WHERE THE RECORDS GO --
    # the same split ``stages/verdicts`` and ``stages/enrichment`` keep. They are
    # not the same directory during a build: ``run.py`` writes the manifest last,
    # so an output root has no manifest to read a platform out of yet.
    study = cfg.current_study()
    return {p.stem: _run_one(p, study, marker_evidence or {}, dataset_id, root)
            for p in de_result_paths(study)}


def _run_one(path: Path, study: Any, marker_evidence: dict[str, Any],
             dataset_id: str, root: Optional[Path]) -> dict[str, Path]:
    """One DE table's records. The statistic column is the matched method's own."""
    import pandas as pd

    from panoscope import platforms
    from panoscope.core import interpret as core_interpret
    from panoscope.core import record as core_record
    from panoscope.platforms.tidy import read_tidy
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.subject import EntityKind
    from panoscope.workflows import de as wf_de

    adapter = platforms.for_study(study)
    table = pd.read_csv(path)
    method = _method_for(path, table)
    spec = method.detect_unit(table)
    world = MeasuredWorld(adapter.frames(study))

    reading = read_tidy(
        str(path),
        unit_column=spec.unit_column,
        entity_column=ENTITY_COLUMN,
        entity_kind=EntityKind.GENE,
        statistic_column=method.statistic_name,
        unit_kind=spec.unit_kind,
        world=world,
        clustering_mode=adapter.clustering_mode(study),
        not_units=frozenset(str(v) for v in table[spec.unit_column]) - set(spec.units),
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(Path(path).relative_to(study.dir)),
    ).value

    loaded = method.load(str(path), adapter, study)
    wf = _workflow()
    interp = paths.interp_dir(dataset_id, root)

    written: dict[str, Path] = {}
    for unit in sorted(loaded.value):
        ev = loaded.value[unit]
        spatial = marker_evidence.get(unit)
        audit = wf_de.DEAudit(
            world=world,
            # process / canonical / abundance stay EMPTY. Nothing in this study
            # declares the affected process or this unit's per-arm cell share, and
            # a default for either would be a number nobody measured. The checks
            # refuse and name what would settle them.
            spatial=_spatial_side(spatial, world) if spatial is not None else None,
            tested_whole_frame=method.tests_whole_frame,
        )
        finding = core_interpret.interpret(ev, wf, audit)
        record = core_record.build(
            ev, reading,
            core_record.unit_decisions(ev, loaded),
            finding,
            task_kind=wf.task_kind,
        ).value
        written[unit] = core_record.write(record, interp, group=path.stem).value
    return written
