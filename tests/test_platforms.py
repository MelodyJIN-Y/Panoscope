"""Invariants for the platform layer: the adapter, the tidy reader, and the one
end-to-end read of the real demo store.

Three groups, and the third is the point of the other two.
``test_the_demo_store_reads_into_normalised_evidence`` takes the committed Xenium
study through ``XeniumAdapter`` and jazzPanda's ``load`` into ``core.measure.measure_unit``
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

from panoscope.core.measure import Reading, measure_unit
from panoscope.data.study import Study
from panoscope.methods.jazzpanda import MODULE as JAZZPANDA
from panoscope.platforms.base import declared_clustering, digest
from panoscope.platforms.tidy import read_tidy
from panoscope.platforms.xenium import (
    NO_SIGNAL,
    XeniumAdapter,
    is_no_signal,
)
from panoscope.types.conditions import DISEASE_STATE, UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.contracts import (
    PlatformAdapter, UndeclaredClustering, UnrecognisedResult,
)
from panoscope.types.measured import (
    Frame, MeasuredState, MeasuredWorld, resolve,
)
from panoscope.types.subject import Entity, EntityKind, cell_type, subject
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


def _gene(store: Study, gene: str, reported: frozenset[str] = frozenset()):
    """One gene through the adapter's own frames -- the production path, not a fixture."""
    return resolve(subject("gene", g=Entity(EntityKind.GENE, gene)),
                   MeasuredWorld(ADAPTER.frames(store)), reported)


def test_a_study_that_detected_nothing_declares_an_empty_frame_and_measures_nothing(
    tmp_path: Path,
) -> None:
    """A declared-and-empty detectability frame is not an undeclarable one.

    Both gene helpers already separated "nothing declares this" (``None``) from "this
    is declared and empty" (``frozenset()``), and ``frames`` tested both for
    truthiness, which collapses them. The frame was dropped, the gene kind was left
    with ``panel:`` alone, every frame of the kind then admitted every panel gene --
    and AAA, whose own row in ``markers_top.csv`` says ``NoSig``, came back
    ``TESTED_ABSENT`` with ``absence_is_evidence``, which is what ``band:driver_nosig``
    reads as "measured and did not mark this unit". A fabricated measurement status
    produced by the frame built to prevent it, with no flag on it.

    The store below is the shape that makes the empty set reachable from a table that
    is not empty: the one row carrying a unit is a negative control, and the controls
    filter in ``_detectable_genes`` removes it. A store with any detectable gene at
    all exercises neither branch -- that is the easy half, and it is covered by
    ``test_detectability_is_a_second_gene_frame`` on the demo.
    """
    store = _store(tmp_path, "nosig",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"]},
                   panel=["AAA", "BBB"],
                   markers=[("AAA", NO_SIGNAL, 0.0, 0.0), ("BBB", NO_SIGNAL, 0.0, 0.0),
                            ("NegControlProbe_00042", "u1", 9.0, 0.9)])

    # The invariant first, so a regression fails on the fabrication itself rather
    # than on the frame lookup that happens to sit in front of it.
    absent = _gene(store, "AAA")
    assert absent.absence_is_evidence is False
    assert absent.state is MeasuredState.NEVER_MEASURED
    assert absent.frame.startswith("detect:"), "the frame that refused it is named"

    detect = next(f for f in ADAPTER.frames(store) if f.name.startswith("detect:"))
    assert detect.members == frozenset()
    # Nothing can promote it either: a gene the method never detected cannot carry a
    # number even if something claims to have reported one for it.
    assert _gene(store, "AAA", frozenset({"AAA"})).carries_number is False


def test_the_missing_and_the_empty_detectability_frame_are_different_blockers(
    tmp_path: Path,
) -> None:
    """One is a missing file, the other is a study in which the method found nothing.

    They need separate lines because the fix is different and because the message for
    the first is *false* about the second: ``markers_top.csv`` is right there, full of
    rows. And the empty case is the one Stage 0 would now miss, since ``frames`` emits
    the frame rather than dropping it: a check asking only "is there a detect frame?"
    passes on the worse of the two.

    §6.3 and the frame do different jobs, which is why this fix is both. The blocker
    is what reaches the biologist -- a study where nothing produced usable signal is
    uninterpretable, not negative. The frame is what the callers that never run
    Stage 0 read (``agent.discriminate``, ``methods.*``, ``pipeline.records``), and on
    those paths the empty frame is the only thing standing between a ``NoSig`` gene
    and an absence that argues.
    """
    manifest = {"platform": "Xenium", "tissue": "t", "clusters": ["u1"]}
    missing = _store(tmp_path, "nofile", manifest=manifest, panel=["AAA"])
    assert not any(f.name.startswith("detect:") for f in ADAPTER.frames(missing))
    assert any("no method result" in p for p in ADAPTER.validate(missing))

    empty = _store(tmp_path, "nosig", manifest=manifest, panel=["AAA"],
                   markers=[("AAA", NO_SIGNAL, 0.0, 0.0)])
    problems = ADAPTER.validate(empty)
    assert any("the detectability frame is empty" in p for p in problems)
    assert not any("no method result" in p for p in problems)


def test_a_panel_of_nothing_but_controls_is_an_empty_measured_set(tmp_path: Path) -> None:
    """The same conflation on the other gene frame, with the same consequence.

    ``_panel_genes`` returned ``frozenset()`` both when the file was absent and when
    it listed nothing but negative controls, so the second could not be told from the
    first and the ``panel:`` frame was dropped for it. Dropping it does NOT yield
    ``NEVER_MEASURED`` here, because the gene kind declares a second frame: AAA is
    detectable, so ``detect:`` admitted it alone and its silence argued -- a tested
    absence measured against a measured set with nothing in it.

    The annotation and cluster frames are not this shape and are deliberately left
    alone: each is the only frame of its kind, so omitting an empty one already
    resolves ``NEVER_MEASURED`` with a reason that is not false, and neither is read
    through a sentinel that could tell "declared and empty" from "not declared".
    """
    store = _store(tmp_path, "ctl",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["u1"]},
                   panel=["NegControlProbe_00042", "BLANK_0003"],
                   markers=[("AAA", "u1", 1.0, 0.5)])

    panel = next(f for f in ADAPTER.frames(store) if f.name.startswith("panel:"))
    assert panel.members == frozenset()

    m = _gene(store, "AAA", frozenset({"AAA"}))
    assert m.state is MeasuredState.NEVER_MEASURED and m.frame.startswith("panel:")
    assert _gene(store, "AAA").absence_is_evidence is False
    assert any("the measured set is empty" in p for p in ADAPTER.validate(store))


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


@pytest.mark.req("ARCH-5.3-01")
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
    assert set(conds.declared_unknown) == {"method", "segmentation", "fixation",
                                           *DISEASE_STATE}


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
        not_units=frozenset({NO_SIGNAL}),
        unit_detection_because=("top_cluster holds cluster ids declared by the store",),
        record_path="inputs/markers_top.csv",
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
    rec = _read(demo).value
    assert rec.rows_in == len(pd.read_csv(demo.inputs / "markers_top.csv"))
    assert rec.rows_dropped == 49
    assert rec.drop_reasons == (("top_cluster is 'NoSig', declared not a unit", 49),)


def test_the_record_hashes_the_bytes_that_were_read(demo: Study) -> None:
    """Over the file's bytes, not the parsed frame: a dtype-inference change in pandas
    would move a digest taken over the frame and report an input change that never
    happened. Checked against the hash the pipeline committed."""
    rec = _read(demo).value
    assert rec.sha256 == digest(demo.inputs / "markers_top.csv")
    manifest = json.loads((demo.dir / "manifest.json").read_text())
    assert rec.sha256 == manifest["inputs"]["markers_top.csv"]["sha256"]


def test_the_record_carries_no_absolute_path(demo: Study) -> None:
    """The record is hashed into the golden artifact, so an absolute path makes
    byte-stability a property of whose laptop produced it."""
    assert _read(demo).value.path == "inputs/markers_top.csv"


def test_out_of_frame_entities_are_named(demo: Study) -> None:
    """The panel-absence rule's output: what the apparatus could not have seen is
    reported, not dropped."""
    rec = _read(demo).value
    markers = pd.read_csv(demo.inputs / "markers_top.csv")
    nosig = sorted(markers.loc[markers["top_cluster"] == NO_SIGNAL, "gene"])
    assert list(rec.entities_outside_frame) == nosig


def test_the_same_file_reads_byte_identically_twice(demo: Study) -> None:
    """The reproducibility claim, at this layer."""
    assert _read(demo).value == _read(demo).value


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
                    clustering_mode="independent").value
    assert rec.rows_in == 5 and rec.rows_dropped == 4
    assert rec.drop_reasons == (
        ("duplicate unit x gene row", 1),
        ("gene is empty", 1),
        ("stat is not a number", 1),
        ("unit is empty", 1),
    )


def test_a_statistic_for_an_out_of_frame_entity_is_carried_in_and_withheld_later(
        ) -> None:
    """Readings are not pre-filtered. Deleting the number before ``measure_unit``
    would delete the difference between "the method reported nothing" and "the
    method reported something the apparatus cannot support", which is the thing
    being audited."""
    world = MeasuredWorld((Frame(EntityKind.GENE, "panel:t", frozenset({"AAA"})),))
    readings = tuple(
        Reading(subject=subject("gene", g=Entity(EntityKind.GENE, g)), statistic=v,
                statistic_name="stat")
        for g, v in (("AAA", 1.0), ("OFFPANEL", 9.0))
    )
    reported = (Frame(EntityKind.GENE, "reported:cluster:u1",
                      frozenset({"AAA", "OFFPANEL"})),)

    ev = measure_unit("u1", "cluster", subject("cell_type", ct=cell_type("X")),
                      "m", readings, world, reported, _conditions(world))
    off = next(i for i in ev.value.items if i.subject.one("g").id == "OFFPANEL")
    assert off.measured.state is MeasuredState.NEVER_MEASURED
    assert off.statistic is None
    assert any(d.rule == "measure:never_measured" and "stat=9.0000" in " ".join(d.because)
               for d in ev.decisions), "the withheld number must be recorded, not dropped"


# --------------------------------------------------------------------------- #
# The demonstration: the real store, end to end
# --------------------------------------------------------------------------- #
def test_the_demo_store_reads_into_normalised_evidence(demo: Study) -> None:
    """The whole point. The committed Xenium study, through the adapter and the
    method's own ``load`` (the product path) into ``measure_unit`` -- with the panel
    frame at its true size, the detectability frame populated, and three of the four
    states resolved on real data rather than on a fixture.

    ``UNDERPOWERED`` is absent because this read declares no coverage floor; it is
    exercised in ``test_core_measure``, and asserting it here would mean inventing a
    floor the study does not have.
    """
    world = MeasuredWorld(ADAPTER.frames(demo))
    traced = JAZZPANDA.load(str(demo.inputs / "markers_top.csv"), ADAPTER, demo)
    assert sorted(traced.value) == sorted(demo.units)
    ev = traced.value["c1"]

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
    assert sorted(k for k, _ in observed.corroborating) == sorted(
        ("pearson", "max_gc_corr", "max_gg_corr"))

    # The unit's own verdict precedes every item's, because it explains them.
    assert traced.decisions[0].refs == (ev.unit_subject.key,)
    assert ev.conditions.frames == tuple(f.ref() for f in world.frames)


# --------------------------------------------------------------------------- #
# The no-signal sentinel: one predicate, or the panel-absence rule inverts
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spelling", ["NoSig", "nosig", "NOSIG", "noSig", " NoSig ", "", "   "],
)
def test_the_no_signal_sentinel_is_read_in_any_case(spelling: str) -> None:
    """``markers_top.csv`` is a CONSUMED input; its spelling is the upstream run's.

    Four places tested this sentinel with ``!= "NoSig"`` independently and
    case-sensitively -- this adapter's detectability frame,
    ``agent/config._derive_cluster_order``, ``agent/grounding_check.is_detectable``
    and ``agent/verdict._detectable_genes``. A table spelling it ``nosig`` therefore
    put every undetectable gene INTO ``detect:<id>``, which makes its absence argue:
    the panel-absence rule inverted, and inverted hardest on an off-target panel
    where the undetectable set is most of the assay rather than 49 of 280.

    Blank counts too, and conservatively: an empty unit column is not a declaration
    that the gene was detected.
    """
    assert is_no_signal(spelling) is True


@pytest.mark.parametrize("unit", ["c1", "u1", "cluster_3", "C1", "NoSignal", "Nosigma"])
def test_a_real_unit_label_is_not_the_sentinel(unit: str) -> None:
    """The other direction: a predicate that ate a unit label would delete a cluster."""
    assert is_no_signal(unit) is False


def test_a_misspelled_sentinel_stays_out_of_the_detectability_frame(
    tmp_path: Path,
) -> None:
    """The frame, on a table whose sentinel is spelled three ways at once.

    Only ``AAA`` was assigned to a unit, so only ``AAA`` produced usable spatial
    signal. Before the shared predicate, ``BBB`` and ``CCC`` joined ``detect:`` and
    became eligible for ``TESTED_ABSENT`` with ``absence_is_evidence`` -- a
    fabricated measurement status on a gene its own file says yielded nothing.
    """
    store = _store(tmp_path, "s", manifest={"platform": "Xenium", "tissue": "t",
                                            "clusters": ["u1"]},
                   panel=["AAA", "BBB", "CCC", "DDD"],
                   markers=[("AAA", "u1", 1.0, 0.5),
                            ("BBB", "nosig", 0.0, 0.0),
                            ("CCC", "NOSIG", 0.0, 0.0),
                            ("DDD", "", 0.0, 0.0)])
    frames = ADAPTER.frames(store)
    detect = next(f for f in frames if f.name.startswith("detect:"))
    clusters = next(f for f in frames if f.name.startswith("clusters:"))
    assert detect.members == frozenset({"AAA"})
    # And the blank never becomes a unit, which is how it used to reach the cluster
    # label set and disarm Stage 0's unknown-label guard from the other side.
    assert clusters.members == frozenset({"u1"})

    world = MeasuredWorld(frames)
    for gene in ("BBB", "CCC", "DDD"):
        m = resolve(subject("gene", g=Entity(EntityKind.GENE, gene)), world, frozenset())
        assert m.state is MeasuredState.NEVER_MEASURED
        assert m.absence_is_evidence is False, (
            f"{gene} yielded no usable signal; its absence cannot argue"
        )


def test_the_missing_sample_blocker_counts_what_can_still_be_compared(
    tmp_path: Path,
) -> None:
    """A §6.3 line is the sentence a biologist acts on, so it must not assert.

    It said "only the pooled table is readable, so replicate concordance has
    nothing to run on" for every shape of this problem -- false as soon as two of
    the declared sections DO have a table, and false in the direction that stops a
    biologist reading a study that can still be compared. On a multi-sample kidney
    store, one section failing QC is the ordinary case, not the edge.
    """
    def _with_samples(sid, declared, on_disk):
        store = _store(tmp_path / sid, sid,
                       manifest={"platform": "Xenium", "tissue": "t",
                                 "clusters": ["u1"], "samples": list(declared),
                                 "clustering_mode": "joint"},
                       panel=["AAA", "BBB"],
                       markers=[("AAA", "u1", 1.0, 0.5), ("BBB", "NoSig", 0.0, 0.0)])
        for s in on_disk:
            d = store.inputs / "samples" / s
            d.mkdir(parents=True)
            pd.DataFrame([("AAA", "u1", 1.0, 0.5)],
                         columns=list(MARKER_COLUMNS)).to_csv(
                d / "markers_top.csv", index=False)
        return store

    # Two of three readable: the study is still comparable, out of two.
    some = ADAPTER.validate(_with_samples("some", ("s1", "s2", "s3"), ("s1", "s2")))
    line = next(p for p in some if "no per-sample marker table" in p)
    assert "s3" in line
    assert "2 of 3 sections can still be compared" in line, line
    assert "nothing to run on" not in line, (
        "the blocker asserted that no comparison is possible; two sections are "
        f"readable: {line}"
    )

    # One of three readable: now it genuinely has nothing to compare, and says so.
    one = ADAPTER.validate(_with_samples("one", ("s1", "s2", "s3"), ("s1",)))
    line = next(p for p in one if "no per-sample marker table" in p)
    assert "fewer than two sections are readable" in line, line


def test_a_control_probe_never_enters_the_absence_universe(tmp_path: Path) -> None:
    """A control is EXCLUDED from interpretation, not refused by a frame.

    ``read_tidy`` builds the entity universe from every entity the
    method named, before any unit check, so a control row entered it and every
    frame was then asked about it. Both gene frames refuse a control -- correctly --
    which put it in ``entities_outside_frame`` and into the count ``panoscope
    explain`` prints as *"N entity(s) the method named that no frame admits"*. That
    number reads as how much of this panel the measurement could not speak to, and
    six probes inflated it by six on the kidney rehearsal.

    Filtering the caller's dataframe is not enough, because ``read_tidy`` re-reads
    the file; ``not_subjects`` is the parameter that reaches the read itself. The
    rows are dropped with a reason, so the record still says six rows left and why.
    """
    csv = tmp_path / "m.csv"
    csv.write_text(
        "gene,top_cluster,glm_coef,pearson\n"
        "AAA,u1,9.0,0.9\n"
        "BBB,u1,1.0,0.5\n"
        "BlankCodeword_0001,u1,99.0,0.99\n"
        "antisenseXYZ,u1,98.0,0.99\n"
    )
    world = MeasuredWorld((
        Frame(EntityKind.GENE, "panel:t", frozenset({"AAA", "BBB"})),
        Frame(EntityKind.UNIT, "clusters:t", frozenset({"u1"})),
    ))
    kw = dict(unit_column="top_cluster", entity_column="gene",
              entity_kind=EntityKind.GENE, statistic_column="glm_coef",
              unit_kind="cluster", world=world, clustering_mode="joint")

    loose = read_tidy(csv, **kw).value
    assert set(loose.entities_outside_frame) == {"BlankCodeword_0001", "antisenseXYZ"}

    tight = read_tidy(
        csv, not_subjects=frozenset({"BLANKCODEWORD_0001", "ANTISENSEXYZ"}), **kw
    ).value
    assert tight.entities_outside_frame == (), tight.entities_outside_frame
    assert any("control probe" in reason for reason, _n in tight.drop_reasons), (
        tight.drop_reasons
    )
    assert dict(tight.drop_reasons)[
        "gene is a control probe, not a subject of interpretation"] == 2
    assert tight.rows_in == 4, "the record still says four rows were read"


# --------------------------------------------------------------------------- #
# The store is written by someone else: name what cannot be read
# --------------------------------------------------------------------------- #
def test_a_corrupt_parquet_names_the_file(tmp_path, monkeypatch):
    """pyarrow says ``'<Buffer>'`` and the biologist can act on nothing.

    pandas hands pyarrow a buffer, so a truncated or non-parquet file reports
    "Could not open Parquet input source '<Buffer>'" with no path in it. A
    study whose panel was damaged in transfer produced that traceback at the
    intake screen.
    """
    import shutil

    from panoscope.agent import config as cfg
    from panoscope.data import layout
    from panoscope.data.study import Study
    from panoscope.platforms.xenium import UnreadableStoreFile, XeniumAdapter

    root = tmp_path / "data"
    shutil.copytree(layout.resolve_data_root(), root)
    (root / "studies" / cfg.DATASET_ID / "inputs" / "panel.parquet").write_text(
        "not a parquet file")
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()

    with pytest.raises(UnreadableStoreFile) as exc:
        XeniumAdapter().validate(Study.open(cfg.DATASET_ID))
    assert "panel.parquet" in str(exc.value), str(exc.value)


def test_an_unfilled_sample_column_is_not_a_sample_named_nan(tmp_path, monkeypatch):
    """A null reads back as the float NaN, and ``str(nan)`` is the non-empty
    string "nan" -- which passed the emptiness filter. A vendor export whose
    sample column is present but unfilled was accepted as a study with one
    sample literally named "nan", and every per-sample view described it."""
    import shutil

    import pandas as pd

    from panoscope.agent import config as cfg
    from panoscope.data import layout
    from panoscope.data.study import Study
    from panoscope.platforms.xenium import XeniumAdapter

    root = tmp_path / "data"
    shutil.copytree(layout.resolve_data_root(), root)
    cells = root / "studies" / cfg.DATASET_ID / "viz" / "cells.csv"
    df = pd.read_csv(cells)
    df["sample"] = None
    df.to_csv(cells, index=False)
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()

    samples = XeniumAdapter().samples(Study.open(cfg.DATASET_ID))
    assert "nan" not in samples, samples
    assert samples == [cfg.DATASET_ID], samples


def test_a_multi_sample_de_only_study_is_not_blocked_by_missing_marker_tables(
    tmp_path: Path,
) -> None:
    """Per-sample marker tables are a marker-workflow input; a DE-only study has none."""
    store = _store(tmp_path, "deonly",
                   manifest={"platform": "Xenium", "tissue": "t", "clusters": ["c1"],
                             "samples": ["s1", "s2"], "clustering_mode": "joint"},
                   panel=["AAA", "BBB"])
    pd.DataFrame([("AAA", "c1", 2.5, 1e-6)],
                 columns=["gene", "cluster", "logFC", "FDR"]).to_csv(
        store.inputs / "de_edger.csv", index=False)
    assert not any("per-sample marker table" in p for p in ADAPTER.validate(store))
