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

The record is built through ``cli._build_record`` rather than through a fixture of
this module's own. A second rebuild path is a second thing to drift, and the one
that matters is the one a user actually runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from panoscope import cli
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


def test_unit_decisions_narrows_a_multi_unit_trace_by_subject_key() -> None:
    """A method loads a whole file in one trace; a record is one unit. Narrowed on
    the ``Subject.key`` the interface clicks back on, never on position."""
    ev = _evidence()
    mine = Decision("measure:observed", "observed",
                    refs=(ev.items[0].subject.key + "#stat=1.0000",))
    theirs = Decision("measure:observed", "observed",
                      refs=(subject("gene", g=gene("ZZZ")).key,))
    unit_level = Decision("measure:tested_absent", "tested_absent",
                          refs=(ev.unit_subject.key,))
    refless = Decision("band:no_canonical_driver", "Low")

    kept = core_record.unit_decisions(
        ev, Traced(value=None, decisions=(mine, theirs, unit_level, refless)))
    assert list(kept.value) == [mine, unit_level, refless], (
        "a ref-less decision must survive: band:no_canonical_driver is the one "
        "decision that explains a Low band and it has nothing to anchor on"
    )
    assert kept.decisions == kept.value, "build splices .decisions, not .value"


# --------------------------------------------------------------------------- #
# The generated layer
# --------------------------------------------------------------------------- #
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
@pytest.fixture
def demo(demo_id: str, datasets_root: Path) -> Study:
    return Study.open(demo_id, data_root=datasets_root.parent).require()


def _committed(demo: Study, unit: str = UNIT) -> dict:
    return json.loads((demo.interp / "clusters" / f"{unit}.json").read_text("utf-8"))


def _clusters_digest(demo: Study) -> dict[str, str]:
    """sha256 of every committed per-unit JSON, by name. Bytes, not parsed values."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((demo.interp / "clusters").glob("*.json"))}


def test_the_record_is_deterministic(demo: Study) -> None:
    """Same hashed inputs, same record. §10's gate compares bytes; without this it
    would be comparing two guesses."""
    one = cli._build_record(demo, UNIT).record
    two = cli._build_record(demo, UNIT).record
    assert one == two
    assert _digest(one) == _digest(two)
    assert (json.dumps(core_record.to_dict(one).value, indent=2)
            == json.dumps(core_record.to_dict(two).value, indent=2))


def test_a_record_build_does_not_touch_the_committed_per_unit_jsons(
        demo: Study, tmp_path: Path) -> None:
    """§10, verbatim: "Never mutate the committed per-unit JSONs."

    Hashed before and after a build AND a write, because the claim being made is
    about bytes on disk, not about which functions were called.
    """
    before = _clusters_digest(demo)
    assert before, "the demo study must have committed per-unit JSONs to protect"

    rec = cli._build_record(demo, UNIT).record
    written = core_record.write(rec, tmp_path).value

    assert _clusters_digest(demo) == before
    assert written.read_text("utf-8")
    assert "clusters" not in written.parts, (
        "a record must never land in the v1 tree, which is a generated view"
    )


def test_the_record_lands_under_interp_records_task_kind_unit(
        demo: Study, tmp_path: Path) -> None:
    rec = cli._build_record(demo, UNIT).record
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
        built = cli._build_record(demo, unit)
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
    built = cli._build_record(demo, UNIT)
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
    """Click-back: "a sentence with nothing to click should not have been rendered"."""
    built = cli._build_record(demo, UNIT)
    anchors = {built.evidence.unit_subject.key} | {
        i.subject.key for i in built.evidence.items}
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
