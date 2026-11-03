"""The COMPARISON composition root: *does this result make sense?*

The third of three, and the one that was missing. ``stages/verdicts.py`` composes
ASSIGNMENT and ``stages/enrichment.py`` composes CHARACTERISATION; ``workflows/de.py``
had no caller at all, so its four §9.7 checks and ``core/reconcile.py``'s join ran
only in tests.

**The method is chosen by SCHEMA, not by filename.** Every method registered into
``de-interpretation`` is asked whether it can read the table; ``detect_unit`` raises
``UnrecognisedResult`` on a shape it does not know, and that refusal is the answer.
A filename convention would have meant this root deciding that ``de_edger.csv`` is
an edgeR table -- the root knowing about a method, which is the thing the registry
exists to prevent. The convention names the KIND of result (``de_*.csv``); the
schema names the method. Two methods accepting one table is a collision and raises,
because silently taking the first would make the record's ``module`` field depend on
import order.

**Check 4 is why this stage takes the marker evidence.** Cross-method coherence
joins the DE result against the spatial one for the SAME unit, through
``core/reconcile.py``. Rebuilding the marker side here would be a second place the
spatial evidence is derived; it is passed in instead, exactly as
``stages/enrichment.py`` takes its ``enrichments``.

**Three of the four checks will refuse on a study like the fixture, and that is the
demonstration rather than a shortfall.** ``panel_scoping`` needs the affected
process and its canonical genes; ``composition_vs_expression`` needs this unit's
share of cells per arm; neither is in a DE table and neither is declared in any file
this repository has. ``replicate_confound`` refuses on every branch by design -- a
``topTags`` table reports one number per gene for the whole unit, so which replicate
drives it is not in the file whoever fitted it. Each refusal names what would settle
it. §9.7 calls that the clearest demonstration of the refusal behaviour, and a root
that supplied a default process or a default abundance to make the checks go green
would be manufacturing the answer.
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
STATISTIC_COLUMN: str = "logFC"


def de_result_path(study: Any) -> Optional[Path]:
    """This study's DE table, or ``None`` when it has none.

    ``None`` is not an error. Most studies have no DE result, and the COMPARISON
    slice simply is not built for them -- the same shape as the enrichment stage,
    which builds only when the dataset has an enrichment result. Sorted, so two
    candidates cannot resolve differently on two filesystems.
    """
    found = sorted(Path(study.inputs).glob(DE_GLOB))
    return found[0] if found else None


def _method_for(table: Any) -> Any:
    """The registered DE method whose schema this table matches. Never guesses.

    Asks every member of the registry and lets each one refuse. Zero acceptances
    names the schemas that were tried, because "this is not a DE table" is only
    useful next to what a DE table would have looked like. Two acceptances raise:
    the record stamps the method that produced it, and a tie broken by import order
    would make that stamp arbitrary.
    """
    from panoscope import methods
    from panoscope.types.contracts import UnrecognisedResult
    from panoscope.workflows.de import WORKFLOW_ID

    registered = methods.for_workflow(WORKFLOW_ID)
    if not registered:
        raise SystemExit(
            f"no method answers {WORKFLOW_ID!r}; a DE table is present but nothing "
            "is registered to read it"
        )

    accepted: list[Any] = []
    refusals: list[str] = []
    for m in registered:
        try:
            m.detect_unit(table)
        except UnrecognisedResult as exc:
            refusals.append(f"  {m.name}: {exc}")
            continue
        accepted.append(m)

    if len(accepted) > 1:
        raise SystemExit(
            f"{len(accepted)} methods accept this DE table "
            f"({', '.join(m.name for m in accepted)}); the record names the method "
            "that produced a result, and a tie broken by import order would make "
            "that name arbitrary. Narrow one method's schema."
        )
    if not accepted:
        raise SystemExit(
            "no registered method can read this DE table:\n" + "\n".join(refusals)
        )
    return accepted[0]


def _spatial_side(marker_ev: Any, world: Any) -> Any:
    """The ASSIGNMENT side of the cross-method join, from the marker build.

    Built here rather than in ``workflows/de.py`` because a workflow may not reach
    for another workflow's result: the join is a composition decision and this is
    the composition root. ``Authority.MEASUREMENT`` because the spatial items ARE a
    measurement -- a reference edge arriving on this side would be reference data
    corroborating a measurement, which §7 forbids outright.
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
           root: Optional[Path] = None) -> dict[str, Path]:
    """One ``InterpretationRecord`` per unit, under ``records/comparison/``.

    ``marker_evidence`` maps unit id -> that unit's ``NormalisedEvidence`` from the
    marker build, and supplies check 4. A unit with no marker evidence still gets a
    record: ``cross_method_coherence`` returns *cannot be settled* naming the missing
    spatial result, which is a different statement from checked-and-coherent and must
    not be collapsed into one.

    Deliberately not wrapped in a ``try``. A study whose DE table no registered
    method can read is a loud failure of ``panoscope build`` -- the same discipline
    ``stages/enrichment.write_records`` keeps, and for the same reason: a swallowed
    error leaves ``records/`` unbuilt while the build prints success.
    """
    import pandas as pd

    from panoscope import platforms
    from panoscope.core import interpret as core_interpret
    from panoscope.core import record as core_record
    from panoscope.platforms.tidy import read_tidy
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.subject import EntityKind
    from panoscope.workflows import de as wf_de

    dataset_id = dataset_id or cfg.DATASET_ID
    # The ACTIVE study for READING and ``root`` only for WHERE THE RECORDS GO --
    # the same split ``stages/verdicts`` and ``stages/enrichment`` keep. They are
    # not the same directory during a build: ``run.py`` writes the manifest last,
    # so an output root has no manifest to read a platform out of yet.
    study = cfg.current_study()
    path = de_result_path(study)
    if path is None:
        return {}

    adapter = platforms.for_study(study)
    table = pd.read_csv(path)
    method = _method_for(table)
    spec = method.detect_unit(table)
    world = MeasuredWorld(adapter.frames(study))

    reading = read_tidy(
        str(path),
        unit_column=spec.unit_column,
        entity_column=ENTITY_COLUMN,
        entity_kind=EntityKind.GENE,
        statistic_column=STATISTIC_COLUMN,
        unit_kind=spec.unit_kind,
        world=world,
        clustering_mode=adapter.clustering_mode(study),
        not_units=frozenset(str(v) for v in table[spec.unit_column]) - set(spec.units),
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(Path(path).relative_to(study.dir)),
    ).value.record

    loaded = method.load(str(path), adapter, study)
    marker_evidence = marker_evidence or {}
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
        )
        finding = core_interpret.interpret(ev, wf, audit)
        record = core_record.build(
            ev, reading,
            core_record.unit_decisions(ev, loaded),
            finding,
            task_kind=wf.task_kind,
        ).value
        written[unit] = core_record.write(record, interp).value
    return written
