"""Invariants for the platform layer: the adapter, the tidy reader, and the one
end-to-end read of the real demo store.

Three groups, and the third is the point of the other two.
``test_the_demo_store_reads_into_normalised_evidence`` takes the committed Xenium
study through ``XeniumAdapter`` and ``read_tidy`` into ``core.measure.measure_unit``
and asserts the four states come out on real data. Everything above it pins one thing
that demonstration depends on, so that when it breaks the failure names a cause
instead of just saying the numbers moved.

The numbers it asserts are read from the files in the same test that asserts them,
never typed in as constants -- except where a constant IS the claim, as in
``test_the_panel_frame_is_derived_from_the_file_not_the_manifest``, which deliberately
puts a wrong count in a manifest and checks that nothing believed it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from panoscope.core.measure import measure_unit
from panoscope.data.study import Study
from panoscope.platforms.base import declared_clustering, digest
from panoscope.platforms.tidy import read_tidy
from panoscope.platforms.xenium import NO_SIGNAL, XeniumAdapter
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import (
    PlatformAdapter, UndeclaredClustering, UnrecognisedResult,
)
from panoscope.types.measured import Frame, MeasuredState, MeasuredWorld
from panoscope.types.subject import EntityKind, cell_type, subject
from panoscope.types.trace import Traced

ADAPTER = XeniumAdapter()
MARKER_COLUMNS = ("gene", "top_cluster", "glm_coef", "pearson")


# --------------------------------------------------------------------------- #
# Synthetic stores. Written to tmp_path, never to the repo's data root.
# --------------------------------------------------------------------------- #
def _store(root: Path, sid: str, *, manifest: dict,
           panel: list[str] | None = None,
           markers: list[tuple] | None = None,
           cells: pd.DataFrame | None = None) -> Study:
    d = root / "studies" / sid
    (d / "inputs").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"dataset_id": sid, **manifest}))
    if panel is not None:
        pd.DataFrame({"gene": panel}).to_parquet(d / "inputs" / "panel.parquet")
    if markers is not None:
        pd.DataFrame(markers, columns=list(MARKER_COLUMNS)).to_csv(
            d / "inputs" / "markers_top.csv", index=False)
    if cells is not None:
        (d / "viz").mkdir(exist_ok=True)
        cells.to_csv(d / "viz" / "cells.csv", index=False)
    return Study.open(sid, data_root=root)


def _conditions(world: MeasuredWorld) -> Conditions:
    return Conditions(
        assay=Condition("xenium", Provenance.READ), modality=UNKNOWN,
        tissue=Condition("t", Provenance.STATED), method=UNKNOWN,
        segmentation=UNKNOWN, fixation=UNKNOWN, frames=world.refs(),
    )


@pytest.fixture
def demo(demo_id: str, datasets_root: Path) -> Study:
    return Study.open(demo_id, data_root=datasets_root.parent).require()


# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #
def test_the_adapter_satisfies_the_protocol_structurally() -> None:
    """Conformance by having the members, not by inheriting a base class.

    ``runtime_checkable`` only checks presence, which is exactly the half worth
    pinning here: a renamed method is a registry entry that imports fine and fails
    at the first call, three layers from the rename.
    """
    assert isinstance(ADAPTER, PlatformAdapter)


# --------------------------------------------------------------------------- #
# Frames: derived, never asserted
# --------------------------------------------------------------------------- #
def test_the_panel_frame_is_derived_from_the_file_not_the_manifest(tmp_path: Path) -> None:
    """A count in a manifest is a claim about a file; the file is the measured set.

    The manifest here says 280 and the panel holds three genes. A reader that
    believed the manifest would declare 277 genes that are not there -- the exact
    shape of the bug a hardcoded 280 produces on the second study.
    """
    store = _store(tmp_path, "s", manifest={"platform": "Xenium", "tissue": "t",
                                            "panel_gene_count": 280,
                                            "clusters": ["u1"]},
                   panel=["AAA", "BBB", "CCC"],
                   markers=[("AAA", "u1", 1.0, 0.5)])
    panel = next(f for f in ADAPTER.frames(store) if f.name.startswith("panel:"))
    assert panel.members == frozenset({"AAA", "BBB", "CCC"})


def test_negative_controls_are_not_part_of_the_measured_set(tmp_path: Path) -> None:
    """Xenium ships controls in the same table as the panel.

    A control probe left in the frame inflates the panel and makes a control's
    absence read as a biological absence.
    """
    store = _store(tmp_path, "s", manifest={"platform": "Xenium", "tissue": "t",
                                            "clusters": ["u1"]},
                   panel=["AAA", "NegControlProbe_00042", "BLANK_0003",
                          "UnassignedCodeword_0101"],
                   markers=[("AAA", "u1", 1.0, 0.5)])
    panel = next(f for f in ADAPTER.frames(store) if f.name.startswith("panel:"))
    assert panel.members == frozenset({"AAA"})


def test_detectability_is_a_second_gene_frame(demo: Study) -> None:
    """Not a footnote: a second ``GENE`` frame, which is what makes an undetectable
    gene ``NEVER_MEASURED`` rather than a tested absence that argues against a rival.
    """
    gene_frames = [f for f in ADAPTER.frames(demo) if f.kind is EntityKind.GENE]
    assert len(gene_frames) == 2
    panel = next(f for f in gene_frames if f.name.startswith("panel:"))
    detect = next(f for f in gene_frames if f.name.startswith("detect:"))

    markers = pd.read_csv(demo.inputs / "markers_top.csv")
    nosig = set(markers.loc[markers["top_cluster"] == NO_SIGNAL, "gene"])
    assert detect.members < panel.members, "detectable genes must be a panel subset"
    assert panel.members - detect.members == nosig
    assert len(nosig) == 49 and len(panel.members) == 280  # the 17.5% in CLAUDE.md


def test_every_entity_kind_the_study_declares_gets_a_frame(demo: Study) -> None:
    """Gene, cell type and unit. A kind with no frame resolves to ``NEVER_MEASURED``
    with "no frame declared", which is a correct verdict for the wrong reason."""
    kinds = {f.kind for f in ADAPTER.frames(demo)}
    assert kinds == {EntityKind.GENE, EntityKind.CELL_TYPE, EntityKind.UNIT}


# --------------------------------------------------------------------------- #
# §6.3's fourth blocker
# --------------------------------------------------------------------------- #
def test_a_multi_sample_store_with_no_declared_mode_is_blocked(tmp_path: Path) -> None:
    """The one conditional blocker, and the thing nothing in the repo enforced.

    Cluster 3 in sample A is the same population as cluster 3 in sample B under joint
    clustering and a different one under independent -- and the labels match either
    way, so a guess here is invisible downstream.
    """
    store = _store(tmp_path, "multi",
                   manifest={"platform": "Xenium", "tissue": "t",
                             "clusters": ["u1"], "samples": ["s1", "s2"]},
                   panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)])
    assert ADAPTER.samples(store) == ["s1", "s2"]
    with pytest.raises(UndeclaredClustering, match="does not declare"):
        ADAPTER.clustering_mode(store)


def test_a_multi_sample_store_that_declares_its_mode_is_not_blocked(tmp_path: Path) -> None:
    store = _store(tmp_path, "multi",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"],
                             "samples": ["s1", "s2"], "clustering_mode": "joint"},
                   panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)])
    assert ADAPTER.clustering_mode(store) == "joint"


def test_the_single_sample_demo_is_not_blocked(demo: Study) -> None:
    """With one sample the three modes are the same statement, so this is a tautology
    rather than a default -- and blocking it would stop a legitimate study dead for a
    fact that cannot differ."""
    assert len(ADAPTER.samples(demo)) == 1
    assert ADAPTER.clustering_mode(demo) == "independent"


def test_an_unrecognised_mode_is_not_a_declaration() -> None:
    """``"batch-corrected"`` is a word someone hoped would be understood. Accepting it
    lets an undeclared study past the blocker by typo."""
    with pytest.raises(UndeclaredClustering, match="not one of"):
        declared_clustering(["s1", "s2"], "batch-corrected", where="test")
    with pytest.raises(UndeclaredClustering):
        declared_clustering(["s1"], "sort-of-joint", where="test")


def test_validate_reports_the_blocker_rather_than_raising(tmp_path: Path) -> None:
    """Intake renders a checklist, and a traceback is not a checklist: it shows the
    first problem and hides the rest."""
    store = _store(tmp_path, "multi",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"],
                             "samples": ["s1", "s2"]},
                   panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)])
    assert any("does not declare how they were clustered"
               in p for p in ADAPTER.validate(store))


def test_validate_names_what_is_missing(tmp_path: Path) -> None:
    store = _store(tmp_path, "nopanel",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"]},
                   markers=[("AAA", "u1", 1.0, 0.5)])
    assert any("no measured set" in p for p in ADAPTER.validate(store))

    absent = Study.open("nope", data_root=tmp_path)
    assert ADAPTER.validate(absent) == [
        f"no manifest at {absent.dir / 'manifest.json'}"
    ]


def test_the_demo_store_validates_clean(demo: Study) -> None:
    assert ADAPTER.validate(demo) == []


# --------------------------------------------------------------------------- #
# Views and conditions
# --------------------------------------------------------------------------- #
def test_a_view_with_no_element_behind_it_is_not_offered(tmp_path: Path) -> None:
    """"Never fake": a manifest may veto a view it has the files for, but it may never
    grant one it does not. A claim in a manifest is not a file."""
    store = _store(tmp_path, "s",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"],
                             "views_available": {"cell_map": True, "umap": True,
                                                 "density": True, "expr": True}},
                   panel=["AAA"], markers=[("AAA", "u1", 1.0, 0.5)],
                   cells=pd.DataFrame({"cell_id": [1], "cluster": ["u1"],
                                       "x": [0.0], "y": [0.0]}))
    assert ADAPTER.available_views(store) == frozenset({"cell_map"})


def test_standing_conditions_never_write_inferred(demo: Study) -> None:
    """The adapter reads, or declares unknown. ``INFERRED`` means a model proposed it
    and the biologist confirmed, and neither has happened at this layer."""
    conds = ADAPTER.standing_conditions(demo)
    provenances = {getattr(conds, f).provenance
                   for f in ("assay", "modality", "tissue", "method",
                             "segmentation", "fixation")}
    assert Provenance.INFERRED not in provenances
    assert conds.assay.value == "Xenium"
    assert conds.assay.provenance is Provenance.READ
    # Modality is NOT among these: it follows from platform=Xenium, which is read
    # from the manifest, so it is READ -- the same kind of vendor fact as
    # coordinate_system()'s micron scale. The rest are genuinely unstated, and
    # FFPE "sounding right" is not a read.
    assert conds.modality.provenance is Provenance.READ
    assert set(conds.declared_unknown) == {"method", "segmentation", "fixation"}


def test_the_conditions_carry_every_frame(demo: Study) -> None:
    conds = ADAPTER.standing_conditions(demo)
    assert {r.name for r in conds.frames} == {f.name for f in ADAPTER.frames(demo)}


# --------------------------------------------------------------------------- #
# The tidy reader
# --------------------------------------------------------------------------- #
def _read(demo: Study, **over) -> Traced:
    kwargs = dict(
        unit_column="top_cluster", entity_column="gene", entity_kind=EntityKind.GENE,
        statistic_column="glm_coef", unit_kind="cluster",
        world=MeasuredWorld(ADAPTER.frames(demo)),
        clustering_mode=ADAPTER.clustering_mode(demo),
        corroborating_columns=("pearson",), not_units=frozenset({NO_SIGNAL}),
        unit_detection_because=("top_cluster holds cluster ids declared by the store",),
        enumerates_absence=True, record_path="inputs/markers_top.csv",
    )
    kwargs.update(over)
    return read_tidy(demo.inputs / "markers_top.csv", **kwargs)


def test_the_read_is_traced_even_though_reading_notes_nothing(demo: Study) -> None:
    """Empty by construction: ``RULES`` has no read step, so the reader cannot note a
    decision without an unreviewed contract change, and the ``ReadingRecord`` is the
    read's audit trail. Pinned so the emptiness stays a known gap, not a silent one.
    """
    traced = _read(demo)
    assert isinstance(traced, Traced)
    assert traced.decisions == ()


def test_what_was_dropped_is_counted_and_named(demo: Study) -> None:
    """``rows_in - rows_dropped`` is the only place a biologist sees that part of the
    table never reached the interpretation."""
    rec = _read(demo).value.record
    assert rec.rows_in == len(pd.read_csv(demo.inputs / "markers_top.csv"))
    assert rec.rows_dropped == 49
    assert rec.drop_reasons == (("top_cluster is 'NoSig', declared not a unit", 49),)


def test_the_record_hashes_the_bytes_that_were_read(demo: Study) -> None:
    """Over the file's bytes, not the parsed frame: a dtype-inference change in pandas
    would move a digest taken over the frame and report an input change that never
    happened. Checked against the hash the pipeline committed."""
    rec = _read(demo).value.record
    assert rec.sha256 == digest(demo.inputs / "markers_top.csv")
    manifest = json.loads((demo.dir / "manifest.json").read_text())
    assert rec.sha256 == manifest["inputs"]["markers_top.csv"]["sha256"]


def test_the_record_carries_no_absolute_path(demo: Study) -> None:
    """The record is hashed into the golden artifact, so an absolute path makes
    byte-stability a property of whose laptop produced it."""
    assert _read(demo).value.record.path == "inputs/markers_top.csv"


def test_out_of_frame_entities_are_named(demo: Study) -> None:
    """The panel-absence rule's output: what the apparatus could not have seen is
    reported, not dropped."""
    rec = _read(demo).value.record
    markers = pd.read_csv(demo.inputs / "markers_top.csv")
    nosig = sorted(markers.loc[markers["top_cluster"] == NO_SIGNAL, "gene"])
    assert list(rec.entities_outside_frame) == nosig


def test_the_same_file_reads_byte_identically_twice(demo: Study) -> None:
    """The reproducibility claim, at this layer."""
    assert _read(demo).value.record == _read(demo).value.record


def test_a_missing_column_refuses_rather_than_guessing(demo: Study) -> None:
    """A wrong unit column reinterprets every row beneath it."""
    with pytest.raises(UnrecognisedResult, match="no column"):
        _read(demo, unit_column="cluster_id")


def test_each_malformed_row_is_dropped_under_its_own_reason(tmp_path: Path) -> None:
    """One bucket per cause. A single "bad row" count cannot tell a truncated file
    from a mis-parsed one."""
    p = tmp_path / "t.csv"
    p.write_text(
        "gene,unit,stat\n"
        "AAA,u1,1.0\n"
        "AAA,u1,2.0\n"      # duplicate unit x entity
        ",u1,3.0\n"         # no entity
        "BBB,,4.0\n"        # no unit
        "CCC,u1,n/a\n"      # statistic is not a number
    )
    rec = read_tidy(p, unit_column="unit", entity_column="gene",
                    entity_kind=EntityKind.GENE, statistic_column="stat",
                    unit_kind="cluster", world=MeasuredWorld(),
                    clustering_mode="independent").value.record
    assert rec.rows_in == 5 and rec.rows_dropped == 4
    assert rec.drop_reasons == (
        ("duplicate unit x gene row", 1),
        ("gene is empty", 1),
        ("stat is not a number", 1),
        ("unit is empty", 1),
    )


def test_absence_is_enumerated_only_when_the_caller_declares_the_table_complete(
        tmp_path: Path) -> None:
    """A table truncated to its top rows has no tested-absences in it, and inventing
    them is an overclaim -- so the cautious reading is the default."""
    p = tmp_path / "t.csv"
    p.write_text("gene,unit,stat\nAAA,u1,1.0\nBBB,u2,2.0\n")
    world = MeasuredWorld((Frame(EntityKind.GENE, "panel:t", frozenset({"AAA", "BBB"})),))
    common = dict(unit_column="unit", entity_column="gene",
                  entity_kind=EntityKind.GENE, statistic_column="stat",
                  unit_kind="cluster", world=world, clustering_mode="independent")

    terse = read_tidy(p, **common).value.for_unit("u1")
    assert [r.subject.one("g").id for r in terse.readings] == ["AAA"]

    full = read_tidy(p, enumerates_absence=True, **common).value.for_unit("u1")
    assert [r.subject.one("g").id for r in full.readings] == ["AAA", "BBB"]
    assert full.readings[1].statistic is None


def test_a_statistic_for_an_out_of_frame_entity_is_carried_in_and_withheld_later(
        tmp_path: Path) -> None:
    """The reader does not pre-filter. Deleting the number here would delete the
    difference between "the method reported nothing" and "the method reported
    something the apparatus cannot support", which is the thing being audited."""
    p = tmp_path / "t.csv"
    p.write_text("gene,unit,stat\nAAA,u1,1.0\nOFFPANEL,u1,9.0\n")
    world = MeasuredWorld((Frame(EntityKind.GENE, "panel:t", frozenset({"AAA"})),))
    unit = read_tidy(p, unit_column="unit", entity_column="gene",
                     entity_kind=EntityKind.GENE, statistic_column="stat",
                     unit_kind="cluster", world=world,
                     clustering_mode="independent").value.for_unit("u1")
    assert [r.statistic for r in unit.readings] == [1.0, 9.0]

    ev = measure_unit("u1", "cluster", subject("cell_type", ct=cell_type("X")),
                      "m", unit.readings, world, unit.reported, _conditions(world))
    off = next(i for i in ev.value.items if i.subject.one("g").id == "OFFPANEL")
    assert off.measured.state is MeasuredState.NEVER_MEASURED
    assert off.statistic is None
    assert any(d.rule == "measure:never_measured" and "stat=9.0000" in " ".join(d.because)
               for d in ev.decisions), "the withheld number must be recorded, not dropped"


# --------------------------------------------------------------------------- #
# The demonstration: the real store, end to end
# --------------------------------------------------------------------------- #
def test_the_demo_store_reads_into_normalised_evidence(demo: Study) -> None:
    """The whole point. The committed Xenium study, through the adapter and the tidy
    reader, into ``measure_unit`` -- with the panel frame at its true size, the
    detectability frame populated, and three of the four states resolved on real data
    rather than on a fixture.

    ``UNDERPOWERED`` is absent because this read declares no coverage floor; it is
    exercised in ``test_core_measure``, and asserting it here would mean inventing a
    floor the study does not have.
    """
    world = MeasuredWorld(ADAPTER.frames(demo))
    tidy = _read(demo).value
    assert [u.unit for u in tidy.units] == list(demo.units)

    label = ADAPTER.unit_labels(demo)["c1"]
    unit = tidy.for_unit("c1")
    traced = measure_unit(
        "c1", "cluster", subject("cell_type", ct=cell_type(label)), "jazzpanda",
        unit.readings, world,
        unit.reported + (Frame(EntityKind.CELL_TYPE, "reported:cluster:c1",
                               frozenset({label})),),
        ADAPTER.standing_conditions(demo),
    )
    ev = traced.value

    panel = next(f for f in world.frames if f.name.startswith("panel:"))
    detect = next(f for f in world.frames if f.name.startswith("detect:"))
    assert len(panel.members) == 280 and len(detect.members) == 231

    states = Counter(i.measured.state for i in ev.items)
    assert sum(states.values()) == len(panel.members)
    assert states[MeasuredState.OBSERVED] == 84
    assert states[MeasuredState.TESTED_ABSENT] == 147
    assert states[MeasuredState.NEVER_MEASURED] == 49

    # The productive half: measured, tested, not reported here -- absence IS evidence.
    absent = next(i for i in ev.items
                  if i.measured.state is MeasuredState.TESTED_ABSENT)
    assert absent.statistic is None and absent.measured.absence_is_evidence

    # The other half: on the panel but no usable signal -- absence argues NOTHING,
    # and the frame that refused it is named.
    never = next(i for i in ev.items
                 if i.measured.state is MeasuredState.NEVER_MEASURED)
    assert never.statistic is None and not never.measured.absence_is_evidence
    assert never.measured.frame.startswith("detect:")
    assert never.subject.one("g").id in panel.members  # on the panel, still unmeasured

    observed = next(i for i in ev.items if i.measured.state is MeasuredState.OBSERVED)
    assert observed.statistic is not None
    assert [k for k, _ in observed.corroborating] == ["pearson"]

    # The unit's own verdict precedes every item's, because it explains them.
    assert traced.decisions[0].refs == (ev.unit_subject.key,)
    assert ev.unit_measured.state is MeasuredState.OBSERVED
    assert ev.conditions.frames == tuple(f.ref() for f in world.frames)
