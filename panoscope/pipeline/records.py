"""Study + method -> one ``InterpretationRecord`` per unit, and the v1 view of it.

The composition root ``IMPLEMENTATION.md`` §10 names: adapter -> ``read_tidy`` ->
``method.load`` -> ``core.measure`` -> ``core.verdict`` -> ``core.record``, with
``core.concordance`` between the last two when the store holds per-sample tables
under joint clustering (see :func:`_concordance`). It worked
before this module existed, but it lived in ``cli.py::_build_record`` where only
``panoscope explain`` could reach it, while ``pipeline/stages/verdicts.py`` ran
``agent/verdict.py``. Two engines, one of which wrote the files the UI reads: the
surface a biologist actually looks at was the one *not* under the §10 gate. Moving
the root here is the whole point of the cutover -- ``explain`` and ``build`` now
compose the same functions in the same order, and ``cli._build_record`` delegates to
:func:`build` rather than keeping a second copy.

**The v1 projection lives here, not in ``core/``.** :func:`to_v1_dict` is
``core/compat.to_v1_assignment`` plus the one field that adapter deliberately drops
(see :func:`_with_pctile`). ``core/`` may not grow a rule it decided not to have;
the pipeline, which owns the on-disk v1 *view*, is where the view is completed.

**Nothing here decides anything.** Every number comes up from ``core/`` and every
label from the study's own annotation. This module reads files, calls the registered
method, and arranges the results -- the one thing it must never do is compute a
statistic, because then ``build`` and ``explain`` could disagree about a value and
only one of them would be on disk.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from panoscope.data.study import Study

#: The interpretive question this root answers: *what IS this unit?* A registry key,
#: resolved through ``workflows.get`` rather than assumed -- the workflow says which
#: task kind the record carries, and therefore whether a band is even defined.
WORKFLOW_ID = "marker-interpretation"

#: The entity column of a marker table. ``UnitSpec`` declares the *unit* column and
#: its reasoning but says nothing about which column names the entity, so this is the
#: one schema fact the caller still supplies. It belongs on ``UnitSpec``, which is
#: frozen; named here so the gap stays visible.
ENTITY_COLUMN = "gene"


class UnitBuild(NamedTuple):
    """One unit's record and the pieces the record has no field for.

    ``InterpretationRecord`` carries the reading, the decisions, the claims and the
    undecidables -- no band, no ``verify``, no evidence items. A renderer or a v1
    projection cannot be written from the record alone, so the pieces it was
    assembled FROM travel beside it rather than being re-derived from it.

    ``canonical`` is upper-cased, which both consumers want: ``core.verdict.assess``
    and ``core.compat`` normalise case themselves, and ``cli``'s renderer compares
    against it directly.
    """

    record: Any          # InterpretationRecord
    evidence: Any        # NormalisedEvidence, statistic-descending
    verdict: Any         # core.verdict.Verdict
    canonical: frozenset[str]
    label: Any           # core.compat.UnitLabel


def _concordance(study: Study, adapter: Any, method: Any, clustering_mode: str,
                 units: tuple[str, ...]) -> tuple[dict[str, Any],
                                                  dict[str, tuple[str, ...]]]:
    """Replicate concordance per unit (ARCHITECTURE §9.10) -- or why there is none.

    Returns ``(traced_by_unit, undecidable_by_unit)``. The first carries the
    ``Traced[Concordance]`` for every unit the samples could actually be compared
    on, so :func:`build_all` can splice its decisions into that unit's record. The
    second carries the sentences that go into ``InterpretationRecord.undecidable``,
    and it is the half this function exists for.

    **Silence was the bug.** ``PlatformAdapter.sample_inputs`` has resolved
    ``inputs/samples/<sample>/markers_top.csv`` for a while and ``validate`` already
    refuses a multi-sample store that lacks them -- so a store could pass validation
    *because the per-sample tables are present* and then have nobody open them. The
    record's ``reading.path`` was the pooled table, which averages the samples
    together, and a unit whose profile differs between sections came out as one
    confident number with no caveat on it. That is a wrong answer rather than a
    refusal, which is the failure the four-state lattice exists to prevent, arriving
    one layer up. Both outcomes are therefore written down: either the record carries
    the comparison, or it carries why the comparison could not be made.

    **The sample count is checked before the clustering mode**, which is not the
    order §9.10 reads in. ``declared_clustering`` returns ``"independent"`` as a
    tautology for a single-sample store, so asking about the mode first would tell
    the demo's biologist that "the same unit id in two samples names two different
    units" about a study that has one sample. The honest first question is whether
    there is a second reading at all.

    **The lattice is not re-implemented here, and that is deliberate.** Whether a
    gene measured in sampleA and never measured in sampleB is a divergence is
    ``core.concordance``'s question, which it answers through ``reconcile._compare``
    -- a side that carries no number and whose absence is not evidence lands on
    NOT_COMPARABLE and can neither agree nor conflict. A second gate here would be a
    second place the four states could collapse into two, and on the kidney panel,
    where most canonical markers are off-panel, that collapse would make the whole
    check noise.

    **A unit a sample emitted no rows for drops that sample rather than joining it
    with an empty profile.** ``concordance`` adjudicates a missing row as
    not-comparable naming the sample, and a replicate with zero items would turn
    every subject of that unit into one -- a check seen to run and find nothing,
    which reads as agreement. The dropped sample is named in the record instead.

    ``method.load`` is re-run per sample rather than the pooled read being reused:
    the per-sample tables are different files with different rows, and the capability
    frames come from ``adapter.frames(study)``, which under joint clustering the
    replicates share by construction (``core/concordance.py``).
    """
    from panoscope.core.concordance import Replicate, concordance

    inputs = adapter.sample_inputs(study)
    declared = adapter.samples(study)
    readable = [s for s in declared if s in inputs]

    # Two reasons, not one, because they are different facts and only the second is
    # something a biologist can act on. A single-sample study has no replicate to be
    # concordant with; a multi-sample store whose per-sample tables are missing has
    # replicates that nothing here can read, and the fix is to put the files there.
    if len(declared) < 2:
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) does not apply: the study "
            f"holds one sample ({declared[0] if declared else 'unnamed'}), so there "
            "is no second reading for this unit's profile to be compared against"
        )
        return {}, {u: (why,) for u in units}

    if len(readable) < 2:
        missing = [s for s in declared if s not in inputs]
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) was not assessed: the study "
            f"declares {len(declared)} samples and {len(readable)} have a per-sample "
            f"marker table (none for {', '.join(missing)}). Only the pooled table is "
            "readable, so a divergence between samples was not looked for here and "
            "has not been ruled out"
        )
        return {}, {u: (why,) for u in units}

    if clustering_mode != "joint":
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) was not assessed: this study "
            f"declares {clustering_mode!r} clustering, so one unit id in "
            f"{', '.join(readable)} names units that were never joined; comparing "
            "their profiles would be a join on a key that means nothing"
        )
        return {}, {u: (why,) for u in units}

    per_sample = {s: method.load(str(inputs[s]), adapter, study).value
                  for s in readable}

    traced: dict[str, Any] = {}
    because: dict[str, tuple[str, ...]] = {}
    for unit in units:
        speaking = [s for s in readable if unit in per_sample[s]]
        silent = [s for s in readable if s not in speaking]
        if len(speaking) < 2:
            because[unit] = (
                f"replicate concordance (ARCHITECTURE 9.10) was not assessed for "
                f"this unit: {', '.join(silent)} reported no row under it, leaving "
                f"{len(speaking)} sample(s) to compare. Whether the method found "
                "nothing there or the unit is absent from that section is not in "
                "the table",
            )
            continue

        out = concordance(unit, tuple(Replicate(s, per_sample[s][unit].items)
                                      for s in speaking), joint=True)
        traced[unit] = out
        c = out.value
        notes: list[str] = []
        if c.divergent:
            notes.append(
                f"{', '.join(s.display() for s in c.diverging_subjects)} "
                f"read differently across {', '.join(c.samples)} "
                f"(reference {c.reference}): batch effect, section quality or real "
                "biology, and 9.10 does not claim which. Compare the diverging "
                "subject across sections before reading the pooled number -- "
                "agreement would not have raised confidence either"
            )
        # K of N, and only when it fell short. A check that compared nothing and a
        # check that compared everything both report "not divergent", and without
        # the denominator the first reads as the second -- the same
        # denominator-thrown-away failure ``Concordance.coverage`` was added for.
        if not c.coverage.ok:
            notes.append(
                f"replicate concordance compared {c.coverage.k} of "
                f"{c.coverage.n} subject(s) across {', '.join(c.samples)}: below the "
                f"floor of {c.coverage.floor}, so the samples not diverging is not a "
                "finding about this unit"
            )
        if notes:
            because[unit] = tuple(notes)
    return traced, because


def build_all(study: Study) -> dict[str, UnitBuild]:
    """Every unit the method found, built once off one read of its table.

    One read, not one per unit: the file's sha256 is hashed into every record, and
    ``method.load`` sorts each unit's rows by descending statistic because
    ``core/verdict._drivers`` takes the first eligible canonical marker *in caller
    order*. Re-reading per unit would be nine chances for those two facts to drift
    apart; doing it once is also what makes ``build`` cheap enough to call per unit.

    Heavy imports are local for the reason every CLI verb's are: ``panoscope
    doctor`` must still answer on an install missing them.
    """
    import dataclasses

    import pandas as pd

    from panoscope import methods, workflows
    from panoscope.core import compat, measure, record as core_record
    from panoscope.core import verdict as core_verdict
    from panoscope.platforms.tidy import read_tidy
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.outputs import TaskKind
    from panoscope.types.subject import EntityKind, gene, subject

    registered = methods.for_workflow(WORKFLOW_ID)
    if len(registered) != 1:
        raise SystemExit(
            f"{len(registered)} methods answer {WORKFLOW_ID!r} "
            f"({', '.join(m.name for m in registered) or 'none'}); the build cannot "
            "pick one for you. The record names its method."
        )
    method = registered[0]
    # The shape guard, at the one place that holds the declaration and the band.
    # This root is ASSIGNMENT-shaped by construction: it calls ``core.verdict.assess``
    # unconditionally and projects a band into interp/clusters/<unit>.json and into
    # verdicts.csv. ``core/interpret.py`` guards the same rule by comparing a
    # RETURNED object against the declaration; there is no returned object here --
    # the shape is this function's own code -- so the comparison is against the
    # declaration alone. Without it, registering a CHARACTERISATION workflow under
    # WORKFLOW_ID would write a unit-level band for a question with no label space,
    # and nothing downstream would notice: ``core.record.build`` takes task_kind as a
    # parameter and ``core.compat.to_v1_assignment`` is handed a Verdict and never
    # asks what question it answers. Loud, at the composition root, before any IO.
    wf = workflows.get(method.workflow)
    if wf.task_kind is not TaskKind.ASSIGNMENT:
        raise SystemExit(
            f"{wf.name!r} declares {wf.task_kind.value}, but this root writes a "
            f"unit-level band into interp/clusters and verdicts.csv. Without a "
            f"label space a band is undefined, not merely unhelpful -- build a "
            f"composition root for {wf.task_kind.value} rather than borrowing this "
            f"one."
        )
    task_kind = wf.task_kind
    adapter = XeniumAdapter()          # no platform registry exists yet; see report
    # Read once and used twice -- stamped into the ReadingRecord, and read by the
    # §9.10 gate below. Two calls could not disagree today, but the record and the
    # gate claiming different clustering modes is the one disagreement whose cost is
    # a divergence check that silently ran on a join nobody licensed.
    clustering_mode = adapter.clustering_mode(study)
    path = study.inputs / "markers_top.csv"
    if not path.is_file():
        raise SystemExit(f"Study {study.id!r} has no method output at {path}.")

    table = pd.read_csv(path)
    # The unit column comes off the SCHEMA, through the method, never from here.
    spec = method.detect_unit(table)
    world = MeasuredWorld(adapter.frames(study))
    reading = read_tidy(
        path,
        unit_column=spec.unit_column,
        entity_column=ENTITY_COLUMN,
        entity_kind=EntityKind.GENE,
        statistic_column=method.statistic_name,
        unit_kind=spec.unit_kind,
        world=world,
        clustering_mode=clustering_mode,
        # Whatever the method did not declare a unit is not a unit. Derived, so no
        # sentinel string ("NoSig") is restated outside the method that owns it.
        not_units=frozenset(str(v) for v in table[spec.unit_column]) - set(spec.units),
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(path.relative_to(study.dir)),
    ).value.record

    loaded = method.load(str(path), adapter, study)
    conc, conc_because = _concordance(study, adapter, method, clustering_mode,
                                      tuple(loaded.value))

    out: dict[str, UnitBuild] = {}
    for unit, ev in loaded.value.items():
        meta = study.unit_labels.get(unit, {})

        # Reference data, selecting which absences are worth reporting and nothing
        # else: the canonical markers this study's annotation lists for the label.
        # The lattice decides what each one IS -- a gene named here that the panel
        # does measure comes back tested-absent, not never-measured.
        reported = frozenset().union(*(f.members for f in ev.reported)) \
            if ev.reported else frozenset()
        absences = tuple(
            measure.measure(
                measure.Reading(subject=subject("gene", g=gene(str(g))),
                                statistic=None,
                                statistic_name=method.statistic_name),
                world, reported)
            for g in meta.get("offpanel_canonical", ())
        )
        ev = dataclasses.replace(ev, items=(
            tuple(i for i in ev.items if i.measured.carries_number)
            + tuple(i for i in ev.items if not i.measured.carries_number)
            + tuple(t.value for t in absences)
        ))

        canonical = frozenset(str(g).upper()
                              for g in meta.get("canonical_markers", ()))
        verdict = core_verdict.assess(ev.items, canonical)
        record = core_record.build(
            ev, reading,
            core_record.unit_decisions(ev, loaded, *absences),
            verdict,
            # Already one unit's, because ``concordance`` is called per unit -- so it
            # goes in whole rather than through ``unit_decisions``. Narrowing it
            # would re-run the anchor join over gene subjects that the POOLED table
            # may not carry, and drop exactly the divergence this wiring exists to
            # show.
            *((conc[unit],) if unit in conc else ()),
            task_kind=task_kind,
            undecidable=conc_because.get(unit, ()),
        ).value
        out[unit] = UnitBuild(
            record=record, evidence=ev, verdict=verdict.value, canonical=canonical,
            label=compat.UnitLabel(
                cell_type=str(meta.get("cell_type", "Unknown")),
                cell_type_short=str(meta.get("cell_type_short", "Unknown")),
                category=str(meta.get("category", "Unknown")),
                lineage=str(meta.get("lineage", "Unknown")),
            ),
        )
    return out


def build(study: Study, unit: str) -> UnitBuild:
    """One unit's build. Thin over :func:`build_all`, which owns the composition."""
    built = build_all(study)
    if unit not in built:
        raise SystemExit(
            f"No interpretation for unit {unit!r} in study {study.id!r}. "
            f"Units the method found: {', '.join(sorted(built)) or '(none)'}"
        )
    return built[unit]


# --------------------------------------------------------------------------- #
# The v1 compatibility view (§10). core/compat.py, plus the one field it drops.
# --------------------------------------------------------------------------- #
def _with_pctile(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put ``within_cluster_pctile`` back into the v1 marker rows.

    ``core/compat.V1_DROPPED`` omits it, and gives the reason: it is derived from
    the glm_coef RANK, v2 carries no rank anywhere, and re-deriving one inside the
    projection would mean re-implementing a rule the band does not use in the layer
    whose job is to make no decisions. That reasoning is about ``core/``. The v1
    *file* is a different question: ``pipeline/serialize._evidence_from_dict`` reads
    the key unconditionally, so a file written without it does not load at all, and
    ``stages/notes.py`` reads it off the loaded object. Until those two stop asking
    for it, the generated view has to carry it, and the pipeline -- which owns that
    view -- is where it is put back rather than in the adapter that decided not to.

    The value is ``agent/verdict.py``'s, unchanged: ``(n-1-rank)/(n-1)`` over the
    rows in statistic-descending order, 1.0 for a single row. Inserted after
    ``p_value`` so the key order is the committed files' and a rebuild is a no-op on
    disk rather than a reformat.
    """
    n = len(rows)
    out: list[dict[str, Any]] = []
    for rank, row in enumerate(rows):
        pctile = 1.0 if n <= 1 else (n - 1 - rank) / (n - 1)
        rebuilt: dict[str, Any] = {}
        for key, value in row.items():
            rebuilt[key] = value
            if key == "p_value":
                rebuilt["within_cluster_pctile"] = pctile
        out.append(rebuilt)
    return out


def to_v1_dict(built: UnitBuild) -> dict[str, Any]:
    """One unit's build as the v1 per-unit JSON dict (``ClusterVerdict`` shape).

    ``opening.driving_markers`` is re-pointed at the completed rows by gene rather
    than re-running :func:`_with_pctile` over it: the driving markers are *copies of*
    evidence rows, and a percentile computed over the short list would be a different
    number against the same gene in the same file.
    """
    from panoscope.core import compat

    d = compat.to_v1_assignment(built.evidence, built.verdict, built.label,
                                built.canonical).value
    rows = _with_pctile(list(d["evidence"]))
    by_gene = {r["gene"]: r for r in rows}
    d["evidence"] = rows
    d["opening"]["driving_markers"] = [by_gene[r["gene"]]
                                       for r in d["opening"]["driving_markers"]]
    return d
