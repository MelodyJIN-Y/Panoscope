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
