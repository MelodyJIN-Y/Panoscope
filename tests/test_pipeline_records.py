"""The cutover's claim, in one assertion: the v1 files the UI reads are now built
by ``core/`` and have not moved a byte.

``pipeline/stages/verdicts.py`` used to call ``agent/verdict.py``. It now composes
``pipeline/records.build_all`` -> ``core.verdict`` -> ``core.compat``, so the files
under ``interp/clusters/`` are a *generated view* (``IMPLEMENTATION.md`` §10) rather
than a second engine's output. ``tests/test_core_compat.py`` proves the projection
reproduces those records field by field from hashed inputs; this proves the
**pipeline** reproduces them *as bytes*, through the one field the compat layer
deliberately drops and this layer puts back.

**Every unit, never one.** ``core/verdict._drivers`` takes the first eligible
canonical marker in caller order, so a unit whose canonical markers are all eligible
(or none are) reproduces the committed band under any ordering at all. A one-unit
version of this test passed while five of the demo's nine were wrong -- that is a
bug this repo actually shipped, and it is why the loop below is over every unit of
every study on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from panoscope.data.study import Study
from panoscope.pipeline import records


def _studies_with_committed_records() -> list[str]:
    """Every study on disk that has v1 per-unit files to be checked against."""
    from panoscope.data import layout

    return [sid for sid in layout.available_studies()
            if any((Study.open(sid).interp / "clusters").glob("*.json"))]


STUDIES = _studies_with_committed_records()


@pytest.fixture(scope="module", params=STUDIES or ["(none on disk)"])
def study(request) -> Study:
    if not STUDIES:
        pytest.skip("no study on disk carries committed per-unit records")
    return Study.open(request.param)


def test_the_v1_view_is_byte_identical_to_the_committed_file(study: Study) -> None:
    """The gate. Same serialisation the stage uses, compared to the bytes on disk.

    Bytes rather than parsed values, because key ORDER is part of what "untouched"
    means here: a rebuild that reorders ``within_cluster_pctile`` produces a diff in
    every committed file and a new sha256 in the manifest, while every value in it
    is still correct. That is a change nobody asked for, so it fails here.
    """
    built = records.build_all(study)
    wrong: list[str] = []
    for path in sorted((study.interp / "clusters").glob("*.json")):
        unit = path.stem
        assert unit in built, f"{study.id}/{unit}: the method produced no evidence"
        produced = json.dumps(records.to_v1_dict(built[unit]), indent=2,
                              ensure_ascii=False) + "\n"
        if produced != path.read_text(encoding="utf-8"):
            wrong.append(unit)
    assert not wrong, (
        f"{study.id}: the generated v1 view no longer reproduces "
        f"{', '.join(wrong)}. Either interpretation behaviour changed (a bug, not "
        "progress -- tests/test_core_compat.py says what moved), or the projection "
        "in pipeline/records.py lost a field."
    )


def test_the_generated_view_reloads_through_the_store(study: Study) -> None:
    """``pipeline/serialize`` must still read it. The one field ``core/compat.py``
    drops is read unconditionally there, so a view written without it does not load
    at all -- and ``store.load_verdict`` fails soft, which turns the breakage into a
    silent fallback to the live engine rather than an error anyone sees."""
    from panoscope.pipeline.serialize import verdict_from_dict

    built = records.build_all(study)
    for unit, b in built.items():
        v = verdict_from_dict(records.to_v1_dict(b))
        assert v.cluster == unit
        assert all(0.0 <= e.within_cluster_pctile <= 1.0 for e in v.evidence)


def test_the_record_lands_beside_the_view_and_never_in_it(study: Study,
                                                          tmp_path: Path) -> None:
    """§10: the v2 record is its own artifact under ``records/<task_kind>/``; the v1
    tree is a generated view and is not a path ``core/record.write`` can produce."""
    from panoscope.pipeline.stages.verdicts import write_record

    built = records.build_all(study)
    unit = sorted(built)[0]
    written = write_record(built[unit], tmp_path)

    assert written == tmp_path / "records" / "assignment" / f"{unit}.json"
    assert "clusters" not in written.parts
    assert json.loads(written.read_text("utf-8"))["unit"] == unit


# --------------------------------------------------------------------------- #
# The productive half: the lattice's third state, on real input
#
# ``methods/jazzpanda.py`` used to emit readings only for the rows whose
# ``top_cluster`` was the unit being built, so the production pipeline produced zero
# ``TESTED_ABSENT`` items and the three consumers that key on it -- ``core/rival.py``'s
# against-the-rival bucket, enrichment's NOT_SUPPORTED tier, ``core/reconcile.py``'s
# genuine-conflict case -- were correct, tested and unreachable from real data.
#
# jazzPanda's top_result enumerates the whole analysed panel (one row per gene, with
# the ``NoSig`` sentinel for the genes it assigned nowhere), so a gene whose row says
# it peaked in c4 WAS tested in c1 and did not come up there. The method now declares
# that with ``enumerates_absence`` and emits those rows as numberless readings.
#
# Every unit, never one -- for this file's own reason, and because the sort order of
# the unnumbered rows decides the driver and therefore the band.
# --------------------------------------------------------------------------- #
def test_every_unit_carries_the_whole_panel_table(study: Study) -> None:
    """One reading per gene the method's table names, for every unit alike.

    Not "some absences": the domain is the file's whole gene universe including the
    rows jazzPanda assigned nowhere, because a gene it looked at and placed elsewhere
    is the evidence, and a gene it could not place anywhere is the capability frame's
    business. Enumerating a subset would make which absences are visible depend on
    which unit you opened.
    """
    import pandas as pd

    from panoscope.pipeline.records import ENTITY_COLUMN

    table = pd.read_csv(study.inputs / "markers_top.csv")
    universe = {str(g) for g in table[ENTITY_COLUMN]}
    built = records.build_all(study)
    for unit, b in sorted(built.items()):
        genes = {e.id for i in b.evidence.items for e in i.subject.entities()}
        assert universe <= genes, f"{study.id}/{unit} drops {len(universe - genes)} genes"


def test_the_productive_half_is_non_empty_on_every_unit(study: Study) -> None:
    """``TESTED_ABSENT`` -- the state where an absence argues -- actually occurs.

    The assertion that would have caught the original bug. A census that reads
    ``{observed, never_measured}`` means the lattice has three live states' worth of
    machinery and two live states, and nothing downstream that keys on the third can
    ever fire.
    """
    from panoscope.types.measured import MeasuredState

    built = records.build_all(study)
    empty = [u for u, b in sorted(built.items())
             if not any(i.measured.state is MeasuredState.TESTED_ABSENT
                        for i in b.evidence.items)]
    assert not empty, (
        f"{study.id}: no measured absence reaches {', '.join(empty)}. Either the "
        "method stopped enumerating absence, or every gene it tested was reported."
    )


def test_an_item_with_no_number_never_precedes_one_that_has_one(study: Study) -> None:
    """The landmine. ``core/verdict._drivers`` takes the first eligible canonical
    marker IN CALLER ORDER, so an unnumbered row that sorted ahead of a numbered one
    would change which marker carries the call -- and the band read off it -- while
    every individual number stayed correct. A one-unit check passes while five are
    wrong; that bug has shipped here before, which is why this is over every unit."""
    built = records.build_all(study)
    for unit, b in sorted(built.items()):
        flags = [i.measured.carries_number for i in b.evidence.items]
        assert flags == sorted(flags, reverse=True), (
            f"{study.id}/{unit}: a numberless item sorts ahead of a numbered one at "
            f"index {flags.index(False)}; the driver is decided by this order."
        )


def test_the_v1_view_omits_them_deliberately_and_countably(study: Study) -> None:
    """The §10 answer to "what does the lossy view do with a state it cannot hold".

    v1 has two buckets and no third, so these items are omitted -- and the omission
    is a counted statement (``compat.v1_omissions``) with a written reason
    (``compat.V1_UNREPRESENTABLE``), announced by ``stages/verdicts.py`` when the view
    is generated, never a silent skip. What must never happen is the other option:
    an omitted item leaking into ``offpanel_notes``, whose message asserts in English
    that the gene is off-panel and was never measured.
    """
    from panoscope.core import compat
    from panoscope.types.measured import MeasuredState

    built = records.build_all(study)
    for unit, b in sorted(built.items()):
        v1 = records.to_v1_dict(b)
        noted = {n["gene"] for n in v1["offpanel_notes"]}
        measured = {e.id for i in b.evidence.items
                    for e in i.subject.entities()
                    if i.measured.state is MeasuredState.TESTED_ABSENT}
        assert not (noted & measured), (
            f"{study.id}/{unit}: {sorted(noted & measured)} were measured and are "
            "filed as off-panel notes, which say they never were"
        )
        counted = sum(n for _, n in compat.v1_omissions(b.evidence).value)
        assert counted == len(b.evidence.items) - len(v1["evidence"]) - len(noted)


def test_a_rival_is_argued_against_by_the_units_own_tested_absent_markers(
    study: Study,
) -> None:
    """What this unlocks, on real input.

    ``core/rival.py``'s ``against_rival`` bucket IS ``TESTED_ABSENT``: a rival is
    argued against precisely when the markers that would separate it were measured
    and peaked elsewhere. With no tested-absent items in the record, every such
    marker had to be synthesised by ``_unreported`` at question time, so the unit's
    own record said nothing about it and there was nothing to click back to. They are
    now items the method emitted, which is what this asserts -- the bucket being
    non-empty is necessary but was already true.

    The reference lists only SELECT which genes are asked about. Every state here
    comes from the measurement, adjudicated by ``core.measure``.
    """
    from panoscope.core.rival import rival_test
    from panoscope.platforms.xenium import XeniumAdapter
    from panoscope.types.measured import MeasuredWorld

    built = records.build_all(study)
    world = MeasuredWorld(XeniumAdapter().frames(study))
    units = sorted(built)
    argued = 0
    for call_id in units:
        rival_id = units[(units.index(call_id) + 1) % len(units)]
        call, rival = study.unit_labels[call_id], study.unit_labels[rival_id]
        r = rival_test(
            built[call_id].evidence, str(call.get("cell_type", call_id)),
            str(rival.get("cell_type", rival_id)),
            tuple(call.get("canonical_markers", ())),
            tuple(rival.get("canonical_markers", ())), world,
        ).value
        own = {i.subject.key for i in built[call_id].evidence.items}
        assert all(i.subject.key in own for i in r.against_rival), (
            f"{study.id}/{call_id}: a marker arguing against {r.alternative} is not "
            "in this unit's own record; it was invented at question time"
        )
        argued += len(r.against_rival)
    assert argued, (
        f"{study.id}: no rival is argued against anywhere -- the productive half of "
        "the lattice reaches core/rival.py not at all."
    )
