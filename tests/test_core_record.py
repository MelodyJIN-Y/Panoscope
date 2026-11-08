"""Invariants for the interpretation record: what it carries, and what it may not do.

Four of these are the reasons the record exists rather than properties it happens to
have, and each one fails for a different reason:

- ``test_the_record_is_deterministic`` -- rebuild from the same hashed inputs and get
  the same bytes. Without it §10's migration gate is comparing two guesses.
- ``test_a_record_build_does_not_touch_the_committed_per_unit_jsons`` -- §10 states
  this outright. Hashed before and after, because "we only call read functions" is
  the kind of claim that stays true until someone adds a cache that writes.
- ``test_the_generated_layer_is_separable_without_naming_it`` -- a consumer gates
  ``prose`` by reading ``GENERATED_FIELDS`` off the type. The test asserts the
  mechanism, not the field, so it still bites when a second generated field lands.
- ``test_the_band_on_screen_is_the_committed_band`` -- the handover. ``explain``
  stopped reading the committed JSON; if the record disagrees with it, the migration
  moved a number and nobody would see it.

The record is built through ``pipeline.records.build`` -- the function
``panoscope explain`` calls -- rather than through a fixture of this module's own.
A second rebuild path is a second thing to drift, and the one that matters is the
one a user actually runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from panoscope import cli
from panoscope.pipeline import records
from panoscope.core import record as core_record
from panoscope.data.study import Study
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.evidence import EvidenceItem, NormalisedEvidence
from panoscope.types.measured import Frame, Measured, MeasuredState
from panoscope.types.outputs import TaskKind
from panoscope.types.record import GENERATED_FIELDS, InterpretationRecord, ReadingRecord
from panoscope.types.subject import EntityKind, cell_type, gene, subject
from panoscope.types.trace import Decision, Traced

UNIT = "c9"


# --------------------------------------------------------------------------- #
# Synthetic pieces: no study, no IO, so a failure here is about assembly only
# --------------------------------------------------------------------------- #
def _reading(path: str = "inputs/t.csv") -> ReadingRecord:
    return ReadingRecord(
        path=path, sha256="0" * 64, statistic_column="stat", unit_column="unit",
        unit_kind="cluster", unit_detection_because=("declared",),
        rows_in=3, rows_dropped=1, drop_reasons=(("unit is empty", 1),),
        entities_outside_frame=("OFFPANEL",), entities_not_in_reference=(),
        frames=(Frame(EntityKind.GENE, "panel:t", frozenset({"AAA"})).ref(),),
        clustering_mode="independent",
    )


def _evidence(unit: str = "u1") -> NormalisedEvidence:
    item = EvidenceItem(
        subject=subject("gene", g=gene("AAA")), statistic=1.0, statistic_name="stat",
        measured=Measured(MeasuredState.OBSERVED, "panel:t", ("reported",)),
    )
    return NormalisedEvidence(
        unit=unit, unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type("X")),
        unit_measured=Measured(MeasuredState.OBSERVED, "annotation:t", ()),
        module="m", items=(item,),
        reported=(Frame(EntityKind.GENE, "reported:u1", frozenset({"AAA"})),),
        conditions=Conditions(
            assay=Condition("t", Provenance.READ), modality=UNKNOWN,
            tissue=Condition("t", Provenance.STATED), method=UNKNOWN,
            segmentation=UNKNOWN, fixation=UNKNOWN,
            frames=(Frame(EntityKind.GENE, "panel:t", frozenset({"AAA"})).ref(),),
        ),
    )


def _digest(rec: InterpretationRecord, generated: bool = True) -> str:
    payload = core_record.to_dict(rec, generated=generated).value
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def test_build_splices_decisions_in_call_order_and_invents_none() -> None:
    """A record assembler that could record would be narrating, not transcribing."""
    first = Traced(value=None, decisions=(Decision("measure:observed", "observed"),))
    second = Traced(value=None, decisions=(Decision("band:from_coef", "High"),))
    rec = core_record.build(_evidence(), _reading(), first, second,
                            task_kind=TaskKind.ASSIGNMENT).value
    assert [d.rule for d in rec.decisions] == ["measure:observed", "band:from_coef"]
    assert rec.unit == "u1" and rec.task_kind is TaskKind.ASSIGNMENT
    assert rec.prose is None, "a complete record has no generated layer by default"


def test_spec_version_is_a_declared_constant_not_a_parameter() -> None:
    """Derived from a package version or a file, it would move the golden record on
    an install that changed nothing -- and two writers could stamp two versions."""
    rec = core_record.build(_evidence(), _reading(),
                            task_kind=TaskKind.ASSIGNMENT).value
    assert rec.spec_version == core_record.SPEC_VERSION
    assert "spec_version" not in core_record.build.__annotations__


def _other_unit(ev: NormalisedEvidence):
    """A second unit of the same SHAPE as ``ev``'s -- i.e. a sibling record."""
    sh = ev.unit_subject.shape
    role, entity = ev.unit_subject.parts[0]
    return subject(sh, **{role: dataclasses.replace(entity, id=entity.id + "_other")})


def test_unit_decisions_cuts_a_multi_unit_trace_into_this_unit_s_segment() -> None:
    """A method loads a whole file in one trace; a record is one unit.

    The anchor join alone cannot do this and used alone it LEAKED: an item subject
    is keyed on the entity, not on the unit it was read under, so ``AAA`` read under
    the sibling unit anchors on exactly the key this unit's record joins on. The
    decision below named ``sibling_aaa`` is that leak in its smallest form -- it is
    indistinguishable from this unit's own by ref, and only the segment it sits in
    tells them apart.
    """
    ev = _evidence()
    other = _other_unit(ev)
    assert other.key != ev.unit_subject.key

    mine_opens = Decision("measure:tested_absent", "tested_absent",
                          refs=(ev.unit_subject.key,))
    mine_aaa = Decision("measure:observed", "observed",
                        refs=(ev.items[0].subject.key + "#stat=1.0000",))
    mine_refless = Decision("band:no_canonical_driver", "Low")
    mine_foreign_anchor = Decision("measure:observed", "observed",
                                   refs=(subject("gene", g=gene("ZZZ")).key,))
    theirs_opens = Decision("measure:tested_absent", "tested_absent",
                            refs=(other.key,))
    sibling_aaa = Decision("measure:observed", "observed",
                           refs=(ev.items[0].subject.key + "#stat=9.0000",))
    sibling_refless = Decision("band:no_canonical_driver", "Low")

    kept = core_record.unit_decisions(ev, Traced(value=None, decisions=(
        mine_opens, mine_aaa, mine_refless, mine_foreign_anchor,
        theirs_opens, sibling_aaa, sibling_refless,
    )))

    assert list(kept.value) == [mine_opens, mine_aaa, mine_refless], (
        "the segment is everything from this unit's anchor up to the next unit's: "
        "a ref-less decision inside it survives because band:no_canonical_driver is "
        "the one decision that explains a Low band and has nothing to anchor on, "
        "and the sibling's rows are dropped however their refs happen to read"
    )
    assert kept.decisions == kept.value, "build splices .decisions, not .value"


def test_a_trace_holding_no_decision_of_this_unit_s_raises() -> None:
    """Keeping nothing is the silent reading: an empty source trace renders as
    "nothing was decided here", which is a sentence about the study rather than
    about the call site that joined the wrong two things."""
    ev = _evidence()
    theirs = Traced(value=None, decisions=(
        Decision("measure:tested_absent", "tested_absent",
                 refs=(_other_unit(ev).key,)),
        Decision("measure:observed", "observed",
                 refs=(ev.items[0].subject.key,)),
    ))
    with pytest.raises(core_record.UnattributableTrace):
        core_record.unit_decisions(ev, theirs)


def test_a_decision_before_the_first_unit_anchor_raises() -> None:
    """``core.measure.measure_unit`` resolves the unit BEFORE any item, so a
    multi-unit trace opens on a unit anchor. A decision ahead of the first one
    belongs to no unit; attributing it to the one that happens to follow is a guess,
    and dropping it silently is how a decision leaves the record unnoticed."""
    ev = _evidence()
    preamble = Traced(value=None, decisions=(
        Decision("measure:observed", "observed", refs=(ev.items[0].subject.key,)),
        Decision("measure:tested_absent", "tested_absent",
                 refs=(ev.unit_subject.key,)),
    ))
    with pytest.raises(core_record.UnattributableTrace):
        core_record.unit_decisions(ev, preamble)


def test_a_per_unit_trace_carrying_no_unit_anchor_is_left_alone() -> None:
    """A verdict's trace names genes, never the unit. It must not be segmented into
    nothing: ``core.verdict`` is where the band's reasoning comes from."""
    ev = _evidence()
    band = Traced(value=None, decisions=(
        Decision("band:from_coef", "High",
                 refs=(ev.items[0].subject.key + "#stat=1.0000",)),
        Decision("band:fragile_cap", "Medium-High"),
    ))
    kept = core_record.unit_decisions(ev, band)
    assert kept.value == band.decisions


# --------------------------------------------------------------------------- #
# On the studies themselves: the segments must PARTITION the method's trace
# --------------------------------------------------------------------------- #
def _studies_on_disk() -> list[str]:
    from panoscope.data import layout

    return sorted(layout.available_studies())


@pytest.mark.parametrize("study_id", _studies_on_disk())
def test_every_load_decision_lands_in_exactly_one_unit_s_record(study_id: str) -> None:
    """Nine units, one trace: the per-unit shares must sum to the whole.

    This is the measured form of the bug. A ``MethodModule.load`` traces one read of
    the file, and before the segmentation every unit's record kept every
    gene-anchored decision in it -- 2,522 of the demo's 2,529 on each of nine
    records, where the unit's own share is 281. Summing is what makes that visible:
    an over-count says decisions were copied between records, an under-count says
    some decision left the record entirely, and only equality says each one is
    filed once.

    Every study on disk and every unit in it, never one. ``_drivers`` takes the
    first eligible marker in caller order, so a one-unit version of a check like
    this has passed in this repo while five of nine units were wrong.
    """
    from panoscope import methods
    from panoscope.pipeline import records as pipeline_records
    from panoscope.platforms.xenium import XeniumAdapter

    study = Study.open(study_id)
    path = study.inputs / "markers_top.csv"
    if not path.is_file():
        pytest.skip(f"{study_id} has no marker table")
    method = methods.for_workflow(pipeline_records.WORKFLOW_ID)[0]
    loaded = method.load(str(path), XeniumAdapter(), study)
    assert len(loaded.value) > 1, "a one-unit study cannot show a cross-unit leak"

    shares = {unit: core_record.unit_decisions(ev, loaded).value
              for unit, ev in loaded.value.items()}
    assert sum(len(s) for s in shares.values()) == len(loaded.decisions), (
        f"{study_id}: the per-unit shares "
        f"{ {u: len(s) for u, s in shares.items()} } do not sum to the "
        f"{len(loaded.decisions)} decisions the method traced"
    )
    for unit, share in shares.items():
        anchors = {loaded.value[unit].unit_subject.key}
        foreign = {u.unit_subject.key for u in loaded.value.values()} - anchors
        assert not [d for d in share for r in d.refs
                    if r.split("#", 1)[0] in foreign], (
            f"{study_id}/{unit}: its share anchors on another unit's subject")


# --------------------------------------------------------------------------- #
# The generated layer
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-17-01")
def test_the_generated_layer_is_separable_without_naming_it(monkeypatch) -> None:
    """The gate reads ``GENERATED_FIELDS`` off the type.

    A gate that hardcodes ``"prose"`` silently stops covering the second generated
    field the day one is added, so this test never writes the string either: it
    drops whatever the type names and asserts the rest is untouched.
    """
    assert GENERATED_FIELDS, "the type must name its generated fields"
    base = core_record.build(_evidence(), _reading(),
                             task_kind=TaskKind.ASSIGNMENT).value
    written = dataclasses.replace(
        base, **{f: "a model wrote this" for f in GENERATED_FIELDS})

    full = core_record.to_dict(written).value
    gated = core_record.to_dict(written, generated=False).value

    assert set(full) - set(gated) == set(GENERATED_FIELDS)
    # The deterministic half is identical with and without the model's half.
    assert (core_record.to_dict(base, generated=False).value
            == core_record.to_dict(written, generated=False).value)
    assert _digest(base, generated=False) == _digest(written, generated=False)
    assert _digest(base) != _digest(written), "the full view must show the prose"

    # The mechanism, not the field: point GENERATED_FIELDS at a different existing
    # field and the gate must follow it. An implementation that popped the literal
    # "prose" passes every assertion above and fails this one, which is the whole
    # reason the constant exists.
    monkeypatch.setattr(core_record, "GENERATED_FIELDS", ("undecidable",))
    moved = core_record.to_dict(written, generated=False).value
    assert "undecidable" not in moved and "prose" in moved


# --------------------------------------------------------------------------- #
# The real study: determinism, the committed files, and the band
# --------------------------------------------------------------------------- #
def _committed(demo: Study, unit: str = UNIT) -> dict:
    return json.loads((demo.interp / "clusters" / f"{unit}.json").read_text("utf-8"))


def _clusters_digest(demo: Study) -> dict[str, str]:
    """sha256 of every committed per-unit JSON, by name. Bytes, not parsed values."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((demo.interp / "clusters").glob("*.json"))}


@pytest.mark.req("ARCH-17-01", "IMPL-9-02")
def test_the_record_is_deterministic(demo: Study) -> None:
    """Same hashed inputs, same record. §10's gate compares bytes; without this it
    would be comparing two guesses."""
    one = records.build(demo, UNIT).record
    two = records.build(demo, UNIT).record
    assert one == two
    assert _digest(one) == _digest(two)
    assert (json.dumps(core_record.to_dict(one).value, indent=2)
            == json.dumps(core_record.to_dict(two).value, indent=2))


@pytest.mark.req("ARCH-5.2-01")
def test_a_record_build_does_not_touch_the_committed_per_unit_jsons(
        demo: Study, tmp_path: Path) -> None:
    """§10, verbatim: "Never mutate the committed per-unit JSONs."

    Hashed before and after a build AND a write, because the claim being made is
    about bytes on disk, not about which functions were called.
    """
    before = _clusters_digest(demo)
    assert before, "the demo study must have committed per-unit JSONs to protect"

    rec = records.build(demo, UNIT).record
    written = core_record.write(rec, tmp_path).value

    assert _clusters_digest(demo) == before
    assert written.read_text("utf-8")
    assert "clusters" not in written.parts, (
        "a record must never land in the v1 tree, which is a generated view"
    )


@pytest.mark.req("IMPL-10-02")
def test_the_record_lands_under_interp_records_task_kind_unit(
        demo: Study, tmp_path: Path) -> None:
    rec = records.build(demo, UNIT).record
    assert (core_record.path_for(tmp_path, rec).value
            == tmp_path / "records" / "assignment" / f"{UNIT}.json")
    written = core_record.write(rec, tmp_path).value
    assert json.loads(written.read_text("utf-8"))["unit"] == UNIT


def test_the_band_on_screen_is_the_committed_band(demo: Study) -> None:
    """The handover, on every unit. ``explain`` no longer reads
    ``interp/clusters/<unit>.json``, so nothing but this would notice the migration
    moving a number.

    Every unit, not one: the driver is the first ELIGIBLE canonical marker in
    glm_coef order, so a unit whose only canonical markers are all eligible (or all
    not) reproduces the committed band under any ordering at all. c9 is one of
    those, and a one-unit version of this test passes while five others are wrong.
    """
    wrong = []
    for unit in demo.units:
        built = records.build(demo, unit)
        committed = _committed(demo, unit)
        got = (cli._band_from(built.record), built.verdict.band,
               built.verdict.verify, built.verdict.basis, built.verdict.fragile)
        want = (committed["confidence"], committed["confidence"],
                committed["verify"], committed["band_basis"], committed["small_n"])
        if got != want:
            wrong.append(f"{unit}: record {got} != committed {want}")
    assert not wrong, "the record disagrees with the committed band:\n  " + \
        "\n  ".join(wrong)


def test_the_record_carries_the_read_and_the_measurement_status(demo: Study) -> None:
    """The two halves a v1 record could not state: what the read cost, and what the
    apparatus could not have seen."""
    built = records.build(demo, UNIT)
    reading = built.record.reading

    assert reading.path == "inputs/markers_top.csv", "store-relative, never absolute"
    assert reading.sha256 == hashlib.sha256(
        (demo.inputs / "markers_top.csv").read_bytes()).hexdigest()
    assert reading.rows_dropped == len(reading.entities_outside_frame) > 0, (
        "the demo's unassigned genes are exactly the ones no frame admits"
    )
    # Never-measured items are rows that say so, never blanks.
    never = [i for i in built.evidence.items
             if i.measured.state is MeasuredState.NEVER_MEASURED]
    assert all(i.statistic is None for i in never)
    # The v1 view holds the PANEL-frame refusals and nothing else: a gene refused by
    # the detectability frame is on the panel, and v1's only shape for it is an
    # OffPanelNote whose message says it is not (core/compat.V1_UNREPRESENTABLE).
    # So the two counts deliberately differ, and the panel half must still match.
    off_panel = {e.id for i in never for e in i.subject.entities()
                 if i.measured.frame.startswith("panel:")}
    assert off_panel == {n["gene"] for n in _committed(demo)["offpanel_notes"]}
    assert [i for i in never if not i.measured.frame.startswith("panel:")], (
        "the demo's undetectable panel genes must reach the record as rows that name "
        "the detectability frame, not be dropped for lacking a v1 shape"
    )


def test_every_decision_in_the_record_resolves_to_this_unit(demo: Study) -> None:
    """Click-back: "a sentence with nothing to click should not have been rendered".

    A claim the record USED is a legitimate anchor, and ``claims/`` has anchored its
    decisions on ``claim:<id>`` since it was written (``evaluability.py:135``,
    ``transfer.py:158``, ``store.about``). So the anchor set is the unit's subjects
    plus the claims the record itself lists in ``claims_used`` -- which keeps the
    invariant exact rather than merely widening it: a decision anchored on a claim
    the record does not carry still fails, and that is precisely the unclickable
    sentence. It is also why the scope-refused claims are in ``claims_used``.
    """
    built = records.build(demo, UNIT)
    anchors = (
        {built.evidence.unit_subject.key}
        | {i.subject.key for i in built.evidence.items}
        | {f"claim:{cid}" for cid, _verdict in built.record.claims_used}
    )
    for d in built.record.decisions:
        for r in d.refs:
            assert r.split("#", 1)[0] in anchors, f"{d.rule} anchors outside the unit"
    assert built.record.decisions, (
        "a record with no decisions is a number nobody can argue with"
    )


def test_explain_renders_from_the_record_not_from_the_committed_json(
        demo: Study, capsys, monkeypatch) -> None:
    """The point of the exercise: the v1 file becomes unreadable and the human view
    still draws.

    The refusal is targeted at ``interp/clusters/`` rather than at the whole study,
    because hiding ``interp/`` also hides ``annotation.json`` -- the study's curated
    marker lists -- and a band computed with no canonical markers would "pass" this
    test by being wrong in a different way.

    ``--json`` still reads that file; it emits the committed compatibility view by
    definition, which is a different artifact from the record.
    """
    band = _committed(demo)["confidence"]
    real = Path.read_text

    def refuse(self, *a, **k):
        if "clusters" in self.parts:
            raise AssertionError(f"explain read the committed v1 file: {self}")
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", refuse)
    assert cli.main(["explain", UNIT, "--study", demo.id]) == 0
    out = capsys.readouterr().out
    assert band in out
    assert "never measured (" in out and "absence is NOT evidence" in out


# --------------------------------------------------------------------------- #
# ARCHITECTURE §9.8's named modes reach the record
# --------------------------------------------------------------------------- #
@pytest.mark.req("ARCH-9.8-01")
def test_the_record_names_the_failure_modes_its_own_decisions_raised(
        demo: Study) -> None:
    """§9.8 asks for the modes to be "named, computed, more actionable than a low
    band". Computed they already were, in six core modules; what was missing was
    the naming, and ``InterpretationRecord`` had no field to put it in.

    The assertion is CONSISTENCY, not a fixed list: every mode the record names
    must be one whose detector ids actually appear among that record's own
    decisions or its evidence's flags. A mode asserted without its detector would
    be exactly the second, drifting copy the index was written to avoid.
    """
    from panoscope.core.failure_modes import MODES

    built = records.build(demo, UNIT)
    rec = built.record
    fired = {d.rule for d in rec.decisions} | {
        f for i in built.evidence.items for f in i.flags}

    for mode_id in rec.failure_modes:
        assert mode_id in MODES, f"record names an unregistered mode {mode_id!r}"
        assert any(d in fired for d in MODES[mode_id].detected_by), (
            f"{mode_id} is named but none of its detectors "
            f"{MODES[mode_id].detected_by} fired for this unit"
        )
    # ...and the converse: a mode whose detector DID fire must be named, or the
    # record is quietly dropping one.
    for mode_id, mode in MODES.items():
        if any(d in fired for d in mode.detected_by):
            assert mode_id in rec.failure_modes, (
                f"{mode_id} fired ({mode.detected_by}) but the record does not "
                "name it"
            )


@pytest.mark.req("ARCH-9.8-01")
def test_failure_modes_discriminate_between_units(demo: Study) -> None:
    """A field that said the same thing everywhere would pass the test above and
    be worth nothing. The demo's thinnest unit must differ from a rich one."""
    from panoscope.pipeline import records

    built = records.build_all(demo)
    by_unit = {u: b.record.failure_modes for u, b in built.items()}
    assert len(set(by_unit.values())) > 1, (
        f"every unit raised the same modes ({by_unit}); the field is not reading "
        "anything unit-specific"
    )
    # c9 is the demo's smallest cluster (167 cells) and its call rests on few
    # markers, so the thin-profile mode is the one that must be there.
    assert "unit:thin_profile" in by_unit["c9"], by_unit["c9"]
