"""Study + method -> one ``InterpretationRecord`` per unit, and the v1 view of it.

The composition root ``IMPLEMENTATION.md`` §10 names: adapter -> ``read_tidy`` ->
``method.load`` -> ``core.measure`` -> ``core.verdict`` -> ``core.record``, with
``core.concordance`` between the last two when the store holds per-sample tables
under joint clustering (see :func:`_concordance`). ``panoscope explain`` and
``panoscope build`` both compose it through :func:`build`.

**The v1 projection lives here, not in ``core/``.** :func:`to_v1_dict` is
``core/compat.to_v1_assignment`` plus the one field that adapter drops
(see :func:`_with_pctile`).

**Nothing here decides anything.** Every number comes up from ``core/`` and every
label from the study's own annotation; this module never computes a statistic.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from panoscope.data.study import NOT_YET_CALLED, Study

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

    ``InterpretationRecord`` carries no band, no ``verify`` and no evidence items, so
    the pieces it was assembled FROM travel beside it. ``canonical`` is upper-cased.
    """

    record: Any          # InterpretationRecord
    evidence: Any        # NormalisedEvidence, statistic-descending
    verdict: Any         # core.verdict.Verdict
    canonical: frozenset[str]
    label: Any           # core.compat.UnitLabel
    #: The workflow's own output (``core.interpret.interpret``, ASSIGNMENT shape).
    #: Same band and ``verify`` as :attr:`verdict`, pinned by
    #: ``tests/test_workflow_routed.py``; ``Verdict`` keeps the basis, demotion ids
    #: and driver that ``Assignment`` has no field for.
    assignment: Any = None   # types.outputs.Assignment


def _concordance(study: Study, adapter: Any, method: Any, clustering_mode: str,
                 units: tuple[str, ...]) -> tuple[dict[str, Any],
                                                  dict[str, tuple[str, ...]]]:
    """Replicate concordance per unit (ARCHITECTURE §9.10) -- or why there is none.

    Returns ``(traced_by_unit, undecidable_by_unit)``: the ``Traced[Concordance]``
    for every unit the samples could be compared on, and the sentences for
    ``InterpretationRecord.undecidable``. Either the record carries the comparison
    or it carries why the comparison could not be made -- never silence.

    * The sample count is checked before the clustering mode: ``declared_clustering``
      is ``"independent"`` by tautology for a single-sample store.
    * The four-state lattice is ``core.concordance``'s, not re-implemented here.
    * A unit a sample emitted no rows for drops that sample (named in the record)
      rather than joining with an empty profile, which would read as agreement.
    * ``method.load`` runs per sample with that sample's own frames
      (``adapter.for_sample(s)``): detectability is per-file, and a pooled frame
      would read a section's no-signal gene as a measured absence.
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
            # Not ``continue`` alone: the question WAS asked here (multi-sample,
            # joint) and compared nothing, which is K=0 and must demote, not
            # ``replicate=None`` (never askable).
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
        # K of N, unconditionally: only the denominator separates a check that
        # compared one subject of forty from one that compared all forty. Below
        # the floor, the line also says the band moved.
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

    * the comparison ran -- its own ``Coverage``, K of N as measured;
    * askable but compared nothing -- ``K=0``, which demotes;
    * never askable (one sample, no per-sample tables, or independent clustering)
      -- ``None``, band untouched.

    ``N`` is the count of subjects the METHOD reported for this unit, the same
    denominator :func:`concordance` uses; ``assess`` guards on ``replicate.n``.
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

    ``claims.store.SEED_RELPATH`` is relative to the data root, so a study tree
    copied elsewhere resolves its own seed or none. ``BadSeed`` is caught (the band
    does not depend on it); every other failure raises.

    Returns ``(store, why_absent)``: *no curated knowledge base here* must stay
    distinguishable from *nothing curated about this label*.
    """
    from panoscope.claims import store as claims_store
    from panoscope.data import layout

    path = layout.resolve_data_root() / claims_store.SEED_RELPATH
    try:
        return claims_store.load_canonical_markers(path), ""
    except claims_store.BadSeed as exc:
        return claims_store.ClaimStore(), str(exc)


def _claims_used(store: Any, label: str, world: Any, conditions: Any,
                 context: str, no_store: str, anchor: str) -> Any:
    """ARCHITECTURE §14's *evaluability + transfer*, for one unit.

    Returns ``Traced[tuple[(claim_id, verdict), ...]]``, refusals kept: a record
    listing only the applied claims reads as though the evidence all pointed one way.

    **Nothing here can move a band.** A curated claim is reference data, so this
    runs AFTER ``assess`` and feeds only the record.

    Narrowed by the unit's LABEL only, not by the subjects the method reported: a
    curated marker the panel never carried has no evidence item but must still
    come back NEVER_MEASURED. Narrowed before ``about`` so its scope gate runs
    without one refusal per gene.
    """
    from panoscope.claims.store import ClaimStore, assess
    from panoscope.types.trace import Trace, ref

    t = Trace()
    if no_store:
        # Said out loud, once per unit, rather than left to an empty tuple. "No
        # curated knowledge base was found" and "the curated knowledge base has
        # nothing about this cell type" are different facts, and only the first is
        # something the biologist can act on.
        t.note("claims:no_store", "not consulted", (no_store,), (ref(anchor),))
        return t.done(())
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

    One read keeps the file's sha256 and ``method.load``'s statistic-descending row
    order identical across units. Heavy imports are local so ``panoscope doctor``
    still answers on an install missing them.
    """
    import dataclasses

    import pandas as pd

    from panoscope import methods, platforms, workflows
    from panoscope.core import compat, interpret as core_interpret, measure
    from panoscope.core import record as core_record
    from panoscope.core import verdict as core_verdict
    from panoscope.platforms.tidy import read_tidy
    from panoscope.reference import view as reference_view
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
    # The shape guard on the declaration, before any IO: this root writes a
    # unit-level band, so a workflow without a label space must not reach it.
    # ``core/interpret.py`` guards the returned object (the ``interpret`` call
    # below); neither subsumes the other.
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
        # entity universe, and are counted and named in
        # ``entities_outside_frame`` as genes the capability frames refused.
        not_subjects=controls,
        unit_detection_because=spec.because,
        # Store-relative: an absolute path in a hashed record makes byte-stability a
        # property of whose laptop produced it.
        record_path=str(path.relative_to(study.dir)),
    ).value

    loaded = method.load(str(path), adapter, study)
    conc, conc_because, conc_uncomparable = _concordance(
        study, adapter, method, clustering_mode, tuple(loaded.value))

    # The claim layer, read once for the whole build. ``conditions`` is the study's
    # OWN standing conditions -- what a claim's formation conditions are compared
    # against -- and the context is keyed on the dataset so a claim scoped to
    # another one cannot be seen here. Scope is fail-closed and the gate is
    # ``ClaimStore.about``'s, not this module's.
    claims, claims_absent = _claim_store(study)
    conditions = adapter.standing_conditions(study)
    claim_context = f"dataset:{study.id}"
    ref_view = reference_view.for_study(study)

    out: dict[str, UnitBuild] = {}
    for unit, ev in loaded.value.items():
        meta = study.unit_labels.get(unit, {})
        # The canonical list comes from the study's reference view, never from the
        # annotation: a label no source marks in this tissue has none (§7.3).
        label = str(meta.get("cell_type", NOT_YET_CALLED))
        canon_list = ref_view.canonical(label)
        # Taken BEFORE the off-panel canonical absences are appended below. Those
        # are reference-selected -- the annotation's canonical list, not anything
        # the method reported -- so counting them would make the replicate
        # denominator disagree with the one ``concordance`` uses for the same unit
        # (the union of subjects across the speaking replicates' items), and the
        # same unit would read "0 of 17" or "9 of 12" depending which path it took.
        n_method_subjects = len({i.subject.key for i in ev.items})

        # Reference data, selecting which absences are worth reporting and nothing
        # else: the canonical markers the reference view lists for the label.
        # The lattice decides what each one IS -- a gene named here that the panel
        # does measure comes back tested-absent, not never-measured.
        reported = frozenset().union(*(f.members for f in ev.reported)) \
            if ev.reported else frozenset()
        # DERIVED from the canonical list: on-panel canonical markers are
        # already in ``ev.items``, so the de-dup leaves the off-panel ones.
        # Identity is stripped and case-folded, as ``Frame.contains`` does, so a
        # spelling difference cannot add an OBSERVED gene a second time.
        def _norm(name: object) -> str:
            return str(name).strip().casefold()

        seen = {_norm(i.subject.one("g").id) for i in ev.items}
        wanted: list[str] = []
        for g in ref_view.markers(label):
            if _norm(g) not in seen:
                seen.add(_norm(g))
                wanted.append(str(g))
        absences = tuple(
            measure.measure(
                measure.Reading(subject=subject("gene", g=gene(g)),
                                statistic=None,
                                statistic_name=method.statistic_name),
                world, reported)
            for g in wanted
        )
        ev = dataclasses.replace(ev, items=(
            tuple(i for i in ev.items if i.measured.carries_number)
            + tuple(i for i in ev.items if not i.measured.carries_number)
            + tuple(t.value for t in absences)
        ))

        canonical = frozenset(str(g).upper() for g in canon_list)
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
        called = str(meta.get("cell_type", "")) not in ("", NOT_YET_CALLED)
        verdict = core_verdict.assess(ev.items, canonical, replicate, called)
        # §14's *evaluability + transfer*, run per unit because the frames and the
        # label are the unit's. ``claims`` and ``conditions`` are read once above.
        claims_t = _claims_used(
            claims, str(meta.get("cell_type", NOT_YET_CALLED)),
            world, conditions, claim_context, claims_absent,
            ev.unit_subject.key)
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
        # ``core.interpret.interpret`` rather than ``wf.interpret``: it guards the
        # RETURNED shape against the declared task kind. ``MarkerReference`` is
        # the one coupling left: the registry cannot yet hand over a reference
        # builder.
        assignment = core_interpret.interpret(ev, wf, wf_marker.MarkerReference(
            label=str(meta.get("cell_type", NOT_YET_CALLED)),
            # Sorted, not the frozenset's own order: ``interpret`` re-freezes it
            # so the band cannot depend on it, but an unsorted tuple in a record
            # that gets hashed would make byte-stability depend on PYTHONHASHSEED.
            canonical=tuple(sorted(canonical)),
            world=world,
            called=called,
        )).value
        out[unit] = UnitBuild(
            record=record, evidence=ev, verdict=verdict.value, canonical=canonical,
            assignment=assignment,
            label=compat.UnitLabel(
                cell_type=str(meta.get("cell_type", NOT_YET_CALLED)),
                cell_type_short=str(meta.get("cell_type_short", NOT_YET_CALLED)),
                category=str(meta.get("category", NOT_YET_CALLED)),
                lineage=str(meta.get("lineage", NOT_YET_CALLED)),
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

    ``core/compat.V1_DROPPED`` omits it, but ``pipeline/serialize._evidence_from_dict``
    reads the key unconditionally and ``stages/notes.py`` reads it off the loaded
    object, so the v1 file must carry it.

    ``(n-1-rank)/(n-1)`` over the rows in statistic-descending order, 1.0 for a
    single row, inserted after ``p_value`` so a rebuild is a no-op on disk.
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
