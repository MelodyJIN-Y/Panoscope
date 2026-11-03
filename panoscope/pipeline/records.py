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
    #: The workflow's own output for this unit, through ``core.interpret.interpret``
    #: -- the ASSIGNMENT shape, guarded against the declared task kind. It carries
    #: the same band and ``verify`` as :attr:`verdict`, and the two are pinned to
    #: each other by ``tests/test_workflow_routed.py`` rather than left to agree by
    #: habit. They are not redundant: ``Assignment`` is the v2 output contract and
    #: the thing a surface renders, while ``Verdict`` carries the basis, the
    #: demotion ids and the driver that the v1 projection and ``cli explain`` branch
    #: on and for which ``Assignment`` has no field. Defaulted, so the positional
    #: constructor stays source-compatible with callers that build one directly.
    assignment: Any = None   # types.outputs.Assignment


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
    the per-sample tables are different files with different rows.

    **And each run gets that sample's own frames, through
    ``adapter.for_sample(s)``.** This function used to hand one
    ``adapter.frames(study)`` to every replicate, on the reading that joint
    clustering makes the replicates share one capability frame. Half of that is
    true: they share the panel, because one panel was run across the sections. The
    other half is false, and falsely in the direction that fabricates. Detectability
    is not declared anywhere -- it is read off a markers table as the genes it did
    not mark ``NoSig`` -- so the pooled frame says a section tested a gene whose own
    file reports no usable signal. ``reconcile._compare`` then reads that section's
    silence as a *measured* absence and returns CONFLICT, and the record told the
    biologist the samples "genuinely conflict" about a gene one of them never
    measured. The same leak ran the other way and was quieter: a gene silent in both
    sections but visible in the pooled read came out as both having tested it and
    neither reporting it, which is CONSISTENT and counts into ``coverage.k`` --
    inflating the one denominator that protects the kidney case.
    """
    from panoscope.core.concordance import Replicate, concordance

    inputs = adapter.sample_inputs(study)
    declared = adapter.samples(study)
    readable = [s for s in declared if s in inputs]
    # Declared but unreadable. Named in every comparison below, not only in the
    # refusal when fewer than two survive: with three samples declared and two
    # readable the check runs, reports "across sampleA, sampleB", and sampleC
    # leaves no trace anywhere -- a comparison that looks complete because the
    # thing missing from it is also missing from the sentence.
    unreadable = [s for s in declared if s not in inputs]

    # Two reasons, not one, because they are different facts and only the second is
    # something a biologist can act on. A single-sample study has no replicate to be
    # concordant with; a multi-sample store whose per-sample tables are missing has
    # replicates that nothing here can read, and the fix is to put the files there.
    if len(declared) < 2:
        # The empty case says "no samples at all", not "one sample (unnamed)". It is
        # unreachable through ``XeniumAdapter.samples``, which raises
        # ``UndeclaredSamples`` or returns ``[store.id]`` -- but the old sentence
        # asserted a sample count of one for a store that declared none, which is a
        # fact nobody supplied, and an adapter added later would inherit it silently.
        held = f"one sample ({declared[0]})" if declared else "no samples at all"
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) does not apply: the study "
            f"holds {held}, so there is no second reading for this unit's profile "
            "to be compared against"
        )
        return {}, {u: (why,) for u in units}, set()

    if len(readable) < 2:
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) was not assessed: the study "
            f"declares {len(declared)} samples and {len(readable)} have a per-sample "
            f"marker table (none for {', '.join(unreadable)}). Only the pooled "
            "table is readable, so a divergence between samples was not looked for "
            "here and "
            "has not been ruled out"
        )
        return {}, {u: (why,) for u in units}, set()

    if clustering_mode != "joint":
        why = (
            f"replicate concordance (ARCHITECTURE 9.10) was not assessed: this study "
            f"declares {clustering_mode!r} clustering, so one unit id in "
            f"{', '.join(readable)} names units that were never joined; comparing "
            "their profiles would be a join on a key that means nothing"
        )
        return {}, {u: (why,) for u in units}, set()

    per_sample = {s: method.load(str(inputs[s]), adapter.for_sample(s), study).value
                  for s in readable}

    traced: dict[str, Any] = {}
    because: dict[str, tuple[str, ...]] = {}
    uncomparable: set[str] = set()
    for unit in units:
        speaking = [s for s in readable if unit in per_sample[s]]
        silent = [s for s in readable if s not in speaking]
        if len(speaking) < 2:
            because[unit] = (
                f"replicate concordance (ARCHITECTURE 9.10) could not be assessed "
                f"for this unit: {', '.join(silent)} reported no row under it, "
                f"leaving {len(speaking)} sample(s) to compare, so NONE of this "
                "unit's profile has a second reading behind it. Whether the method "
                "found nothing there or the unit is absent from that section is "
                "not in the table. The band is demoted one rung and flagged for "
                "re-check",
            )
            # Not ``continue`` alone. This unit is the THINNEST possible replicate
            # coverage -- zero subjects comparable -- and leaving it out of ``conc``
            # made ``assess`` receive ``replicate=None``, which means *the question
            # was never askable*. That is true of the single-sample demo and false
            # here: the study is multi-sample and joint, two tables are readable,
            # the question WAS asked, and the answer is "nothing could be compared".
            #
            # Reporting it as not-askable inverted the gradient one last time: a
            # unit comparable on 1 of 12 subjects was demoted and flagged, while a
            # unit comparable on 0 of 12 came back at full band with no demotion and
            # shipped. In this project's own vocabulary that is UNDERPOWERED
            # reported as NEVER_MEASURED -- and it went quiet exactly as the
            # evidence thinned, on exactly the units an off-target panel is worst at.
            uncomparable.add(unit)
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
        # K of N, unconditionally. How many subjects were comparable out of how many
        # is a FACT about the comparison, not an exception to it: a check that
        # compared one subject of forty and a check that compared all forty both
        # report "not divergent", and only the denominator separates them. Writing
        # it only when ``ok`` was false wrote it almost never -- ``floor`` is 1, so
        # the line fired at k == 0 alone, and every point of the gradient from
        # 1-of-200 to 200-of-200 printed no concordance line at all, byte-identical
        # to a study where the check ran clean on everything. That inverted the
        # loudness gradient: the single-sample demo says "there is no second
        # reading" on all nine units while a multi-sample study at 8% coverage said
        # nothing. The study that cannot support the check shouted; the study that
        # silently under-supported it was quiet.
        #
        #
        # The floor is no longer 1, so the second clause now fires across the thin
        # half of the gradient instead of at k == 0 alone -- and it says the band
        # moved, because a caveat that does not name its consequence is a caveat
        # a reader skips. Thin must LOOK thin in the same neutral field.
        notes.append(
            f"replicate concordance compared {c.coverage.k} of {c.coverage.n} "
            f"subject(s) across {', '.join(c.samples)}"
            + ("" if c.coverage.ok else
               f": fewer than the floor of {c.coverage.floor}, so most of this "
               "unit's profile has no second reading behind it and the samples "
               "not diverging is not a finding about this unit. The band is "
               "demoted one rung and flagged for re-check")
        )
        # What was NOT compared, named -- in the same sentence block, because a
        # dropped sample is invisible exactly where the comparison looks complete.
        # "across sampleA, sampleB" reads as the whole study unless the study's
        # third section says otherwise. Two different drops, kept apart: a sample
        # with no per-sample table at all (fixable by putting the file there) and a
        # sample whose table holds no row under this unit (a fact about the method's
        # output, and the record must not guess which of the two it is).
        if unreadable:
            notes.append(
                f"not compared: {', '.join(unreadable)} -- declared by the study "
                "but with no per-sample marker table, so "
                f"{'it was' if len(unreadable) == 1 else 'they were'} left out of "
                "every unit's comparison"
            )
        if silent:
            notes.append(
                f"not compared for this unit: {', '.join(silent)} -- readable, but "
                "reported no row under it. Whether the method found nothing there "
                "or the unit is absent from that section is not in the table"
            )
        if notes:
            because[unit] = tuple(notes)
    return traced, because, uncomparable


def _replicate_coverage(unit: str, n_subjects: int, conc: dict[str, Any],
                        uncomparable: set[str]) -> Any:
    """The §9.10 K of N this unit's band must see, or ``None`` if it was never askable.

    Three outcomes, and the whole point is that they are three rather than two:

    * the comparison ran -- its own ``Coverage`` travels, K of N as measured;
    * it was ASKABLE and compared nothing -- ``K=0``, which is a real coverage and
      demotes, because a multi-sample joint-clustered study whose second section
      reported no row under this unit has a genuinely unsupported profile here;
    * it was never askable -- ``None``, and the band is untouched. One sample, no
      per-sample tables, or independent clustering: there is no second reading to
      be thin, and demoting for the absence of a question would punish the
      single-sample demo on all nine units.

    ``N`` is the count of subjects the METHOD reported for this unit, which is the
    same denominator :func:`concordance` uses, so one unit cannot read "0 of 17"
    down one path and "9 of 12" down the other. ``K=0`` with ``N>0`` is what makes
    :func:`core.verdict.assess` fire -- it guards on ``replicate.n``, so a zero
    denominator would silently mean "not assessed" and reopen the hole this
    function exists to close.
    """
    if unit in conc:
        return conc[unit].value.coverage
    if unit not in uncomparable:
        return None
    from panoscope.core.verdict import (
        REPLICATE_COVERAGE_FLOOR, REPLICATE_COVERAGE_WHAT, _coverage_floor,
    )
    from panoscope.types.measured import Coverage

    return Coverage(REPLICATE_COVERAGE_WHAT, 0, n_subjects,
                    _coverage_floor(n_subjects, REPLICATE_COVERAGE_FLOOR))


def _claim_store(study: Study) -> Any:
    """The curated claim store under this study's data root, or an empty one.

    ``claims.store.SEED_RELPATH`` is relative to the data root and not to the
    repository, so a study tree copied elsewhere resolves its own seed or none --
    which is the honest outcome and not an error. An absent seed yields an empty
    store, and the record then carries no ``claims_used`` at all rather than a claim
    borrowed from whichever checkout happened to be on disk.

    ``BadSeed`` is caught for the same reason and nothing else is: a malformed seed
    must not take the build down, because the band does not depend on it. Every
    other failure is left to raise.
    """
    from panoscope.claims import store as claims_store
    from panoscope.data import layout

    try:
        return claims_store.load_canonical_markers(
            layout.resolve_data_root() / claims_store.SEED_RELPATH)
    except claims_store.BadSeed:
        return claims_store.ClaimStore()


def _claims_used(store: Any, label: str,
                 world: Any, conditions: Any, context: str) -> Any:
    """ARCHITECTURE §14's *evaluability + transfer*, for one unit.

    Returns ``Traced[tuple[(claim_id, verdict), ...]]``. The decisions travel with
    it so the reasoning lands in the record rather than only the conclusion, and the
    refusals are kept: ``InterpretationRecord.claims_used`` carries considered-and-
    withheld claims on purpose, because a record listing only the applied ones reads
    as though the evidence all pointed one way.

    **Nothing here can move a band, and that is the point.** A curated claim is
    reference data: it may select and order evidence and carry a citation, and it
    may never supply a value or count as agreement. So this runs AFTER ``assess``
    and feeds only the record. The claim layer's job is to say which knowledge was
    brought to bear and what the measurement did with it -- on the kidney panel,
    where almost no canonical marker is on-panel, that answer is mostly
    *never measured here*, and reporting that is the panel-absence rule doing its
    work rather than a gap.

    **Narrowed by the unit's LABEL and by nothing else.** Filtering also on the
    subjects the method reported would have been the obvious next line and it would
    have been wrong in the one direction that matters: a curated marker for this
    label that the panel never carried produces no evidence item, so it would have
    been dropped silently -- and on an off-target panel, where almost no canonical
    marker is on-panel, that is most of them. The frame can answer without an
    evidence item. ``evaluability`` asks the world directly and returns
    NEVER_MEASURED, which is the productive half of the lattice and the kidney
    study's whole shape. Dropping those rows would turn the panel-absence rule into
    a shorter list.

    Narrowed BEFORE ``about`` rather than filtered after, because ``about`` emits a
    decision per out-of-scope claim per subject; asking it only about the claims
    that name this unit's label keeps the scope gate -- which is ``about``'s, not
    reimplemented here -- without writing the same refusal once per gene.
    """
    from panoscope.claims.store import ClaimStore, assess
    from panoscope.types.trace import Trace

    t = Trace()
    relevant = ClaimStore(tuple(c for c in store if c.object == label))
    used: list[tuple[str, str]] = []
    # Sorted: dict and set iteration order is not a fact about the study, and this
    # tuple is serialised into a record whose bytes are hashed into the manifest.
    for key in sorted({c.subject.key for c in relevant}):
        kept = t.absorb(relevant.about(key, context))
        # Scope-refused claims are carried too, with the verdict ``out_of_scope``.
        # ``about`` filters them before the result is formed and notes each one, so
        # without this the record would hold a decision anchored on a claim it does
        # not list -- a sentence with nothing to click, which §13.5 forbids. They
        # are also refusals, and the record carries refusals on purpose.
        kept_ids = {c.id for c in kept}
        for claim in kept:
            a = t.absorb(assess(claim, world, conditions, context))
            if a.usable:
                verdict = "applied"
            elif not a.evaluable.usable:
                # The frame refused it. Name the state, not a generic "no": the
                # difference between never-measured and undetectable is the whole
                # lattice, and a reader must be able to act on it.
                verdict = f"withheld:{a.evaluable.state.value}"
            else:
                verdict = f"withheld:{a.transfer.action}"
            used.append((claim.id, verdict))
        used.extend((c.id, "out_of_scope") for c in
                    sorted((c for c in relevant
                            if c.subject.key == key and c.id not in kept_ids),
                           key=lambda c: c.id))
    return t.done(tuple(used))


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

    from panoscope import methods, platforms, workflows
    from panoscope.core import compat, interpret as core_interpret, measure
    from panoscope.core import record as core_record
    from panoscope.core import verdict as core_verdict
    from panoscope.platforms.tidy import read_tidy
    from panoscope.types.measured import MeasuredWorld
    from panoscope.types.outputs import TaskKind
    from panoscope.types.subject import EntityKind, gene, subject
    from panoscope.workflows import marker as wf_marker

    registered = methods.for_workflow(WORKFLOW_ID)
    if len(registered) != 1:
        raise SystemExit(
            f"{len(registered)} methods answer {WORKFLOW_ID!r} "
            f"({', '.join(m.name for m in registered) or 'none'}); the build cannot "
            "pick one for you. The record names its method."
        )
    method = registered[0]
    # The shape guard, at the one place that holds the declaration and the band.
    # This root projects a band into interp/clusters/<unit>.json and into
    # verdicts.csv, so a workflow that declares no label space cannot be allowed to
    # reach it. ``core/interpret.py`` guards the other half of the same rule by
    # comparing the RETURNED object against the declaration, and now runs on this
    # path too (see the ``interpret`` call below) -- but it cannot run before the
    # IO, and a CHARACTERISATION workflow registered here should fail before the
    # study is read rather than nine units in. Declaration first, object second;
    # neither subsumes the other. Without it, registering a CHARACTERISATION workflow under
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
    # The study's manifest says which technology produced it, and the registry
    # resolves the reader. Hardcoding ``XeniumAdapter()`` here made ARCHITECTURE
    # §4's "the core does not change when a member is added" false for platforms:
    # a MERSCOPE store would have been read through the Xenium adapter, which does
    # not raise -- it answers, about a panel that was never run.
    adapter = platforms.for_study(study)
    # Read once and used twice -- stamped into the ReadingRecord, and read by the
    # §9.10 gate below. Two calls could not disagree today, but the record and the
    # gate claiming different clustering modes is the one disagreement whose cost is
    # a divergence check that silently ran on a join nobody licensed.
    clustering_mode = adapter.clustering_mode(study)
    path = study.inputs / "markers_top.csv"
    if not path.is_file():
        raise SystemExit(f"Study {study.id!r} has no method output at {path}.")

    table = pd.read_csv(path)
    # Same filter as ``method.load`` applies to each per-sample table, for the same
    # reason: a control probe is not a subject of interpretation, and a reading that
    # carries one hands it to every consumer as a NEVER_MEASURED gene.
    controls = frozenset(
        str(g).strip().upper() for g in table[ENTITY_COLUMN]
        if adapter.is_control(study, g)
    )
    table = table[~table[ENTITY_COLUMN].astype(str).str.strip().str.upper()
                  .isin(controls)].reset_index(drop=True)
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
        # ``read_tidy`` re-reads the FILE, so filtering ``table`` above is not
        # enough on its own: without this the control rows come back in, enter the
        # absence-enumeration universe, and are counted and named in
        # ``entities_outside_frame`` as genes the capability frames refused.
        not_subjects=controls,
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(path.relative_to(study.dir)),
    ).value.record

    loaded = method.load(str(path), adapter, study)
    conc, conc_because, conc_uncomparable = _concordance(
        study, adapter, method, clustering_mode, tuple(loaded.value))

    # The claim layer, read once for the whole build. ``conditions`` is the study's
    # OWN standing conditions -- what a claim's formation conditions are compared
    # against -- and the context is keyed on the dataset so a claim scoped to
    # another one cannot be seen here. Scope is fail-closed and the gate is
    # ``ClaimStore.about``'s, not this module's.
    claims = _claim_store(study)
    conditions = adapter.standing_conditions(study)
    claim_context = f"dataset:{study.id}"

    out: dict[str, UnitBuild] = {}
    for unit, ev in loaded.value.items():
        meta = study.unit_labels.get(unit, {})
        # Taken BEFORE the off-panel canonical absences are appended below. Those
        # are reference-selected -- the annotation's canonical list, not anything
        # the method reported -- so counting them would make the replicate
        # denominator disagree with the one ``concordance`` uses for the same unit
        # (the union of subjects across the speaking replicates' items), and the
        # same unit would read "0 of 17" or "9 of 12" depending which path it took.
        n_method_subjects = len({i.subject.key for i in ev.items})

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
        # The §9.10 K of N goes INTO the band, not beside it. Splicing the
        # concordance decisions into the record after assessing the pooled items
        # is what let "comparable on 1 of 12 subjects" and "comparable on 9 of 12"
        # come out byte-identical at Very High with verify false. ``None`` where
        # the comparison was not askable at all, which ``conc_because`` explains
        # in the same record.
        replicate = _replicate_coverage(unit, n_method_subjects, conc,
                                        conc_uncomparable)
        # ...and it travels ON the unit from here, because the workflow computes
        # the band and must not be handed evidence whose band-relevant context was
        # left behind at the root. ``Measured.coverage`` is where every K-of-N in
        # the system already lives and ``workflows/de.py`` already reads one off it.
        # Appended rather than assigned: the unit may already carry a coverage of
        # its own, and ``marker._replicate_coverage`` matches on ``what`` rather
        # than taking the first.
        if replicate is not None:
            ev = dataclasses.replace(ev, unit_measured=dataclasses.replace(
                ev.unit_measured,
                coverage=ev.unit_measured.coverage + (replicate,)))
        verdict = core_verdict.assess(ev.items, canonical, replicate)
        # §14's *evaluability + transfer*, run per unit because the frames and the
        # label are the unit's. ``claims`` and ``conditions`` are read once above.
        claims_t = _claims_used(
            claims, str(meta.get("cell_type", "Unknown")),
            world, conditions, claim_context)
        record = core_record.build(
            ev, reading,
            core_record.unit_decisions(ev, loaded, *absences),
            verdict,
            claims_t,
            # Already one unit's, because ``concordance`` is called per unit -- so it
            # goes in whole rather than through ``unit_decisions``. Narrowing it
            # would re-run the anchor join over gene subjects that the POOLED table
            # may not carry, and drop exactly the divergence this wiring exists to
            # show.
            *((conc[unit],) if unit in conc else ()),
            task_kind=task_kind,
            claims_used=claims_t.value,
            undecidable=conc_because.get(unit, ()),
        ).value
        # The workflow answers its own question. ``core.interpret.interpret``
        # rather than ``wf.interpret`` directly: the guard that compares the
        # RETURNED shape against the declared task kind lives there, and this is
        # the path that writes a band into verdicts.csv, so it is the path that
        # most needs it. The task-kind check above is the same rule asked of the
        # declaration before any IO; this one asks it of the object.
        #
        # ``MarkerReference`` is named here, which is the one coupling left: the
        # root has to know its workflow's reference TYPE to build one, exactly as
        # ``stages/enrichment.py`` names ``EnrichmentReference``. The registry
        # still chooses the workflow; what it cannot yet do is hand over a
        # reference builder, so a second ASSIGNMENT workflow with a different
        # reference carrier would need this line.
        assignment = core_interpret.interpret(ev, wf, wf_marker.MarkerReference(
            label=str(meta.get("cell_type", "Unknown")),
            # Sorted, not the frozenset's own order: ``interpret`` re-freezes it
            # so the band cannot depend on it, but an unsorted tuple in a record
            # that gets hashed would make byte-stability depend on PYTHONHASHSEED.
            canonical=tuple(sorted(canonical)),
            world=world,
        )).value
        out[unit] = UnitBuild(
            record=record, evidence=ev, verdict=verdict.value, canonical=canonical,
            assignment=assignment,
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
