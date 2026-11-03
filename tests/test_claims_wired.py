"""The claim layer runs inside a study build, and cannot move a number while it does.

ARCHITECTURE §14's RUN line puts *evaluability + transfer* between the interpretation
and the rival tests. It was not there: ``claims/store.py``, ``claims/evaluability.py``
and ``claims/transfer.py`` were written, tested against ``tmp_path`` fixtures, and
called by nothing. ``InterpretationRecord.claims_used`` -- a field whose docstring
explains why refusals belong in it -- was ``[]`` on every committed record.

Two things are pinned here and they pull in opposite directions, which is the point.

**That the layer runs.** Every unit's record now names the curated claims about its
own label, and says what the measurement could do with each one.

**That it changes nothing else.** A curated claim is reference data: it may select
and order evidence and carry a citation, and it may never supply a value or count as
agreement. So the band, ``verify`` and the whole v1 view must be identical with the
claim store emptied. If that ever fails, reference data has reached a number, which
is the failure CLAUDE.md puts first.

The last test is a guard the repo did not have. The committed records had drifted
out of date -- ``c1.json`` held 2522 decisions where a rebuild produces 296, because
it predated the fix that cuts a multi-unit trace down to one unit's segment -- and
nothing noticed, because nothing compared a committed record against what the code
actually writes.
"""

from __future__ import annotations

import json

import pytest

from panoscope.claims.store import ClaimStore
from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.pipeline import records

DEMO = "xenium_hbreast_sample1"


def _studies() -> list[str]:
    return sorted(layout.available_studies())


@pytest.fixture(scope="module")
def demo_built() -> dict:
    return records.build_all(Study.open(DEMO))


def test_the_curated_store_loads_and_is_not_empty() -> None:
    """If the seed stops loading, every assertion below passes vacuously."""
    store, absent = records._claim_store(Study.open(DEMO))
    assert not absent, f"the committed seed did not load: {absent}"
    assert len(store) > 0, (
        "no curated claims loaded; the tests below would pass by finding nothing"
    )
    assert all(c.source for c in store), "a claim with no source is uncited"


def test_every_claim_in_a_record_resolves_to_a_real_claim(demo_built: dict) -> None:
    """Click-back from the record to the claim, with no dangling ids.

    A record naming a claim id that is not in the store is a sentence with nothing
    to click, which §13.5 forbids -- and it is the shape a stale or hand-edited
    record would take.
    """
    known = {c.id for c in records._claim_store(Study.open(DEMO))[0]}
    seen = 0
    for unit, b in demo_built.items():
        for cid, verdict in b.record.claims_used:
            assert cid in known, f"{unit}: record cites unknown claim {cid!r}"
            assert verdict, f"{unit}: {cid} carries an empty verdict"
            seen += 1
    assert seen, "no unit used any claim; the layer did not run"


def test_an_off_panel_curated_marker_is_reported_not_dropped(
        demo_built: dict) -> None:
    """The panel-absence rule, through the claim layer. The kidney study's shape.

    A curated marker for this unit's label that the panel never carried produces no
    evidence item at all. Narrowing the claim lookup to the subjects the method
    reported would therefore have dropped it silently -- and on an off-target panel
    that is most of them. It must come back as a row that says *never measured*.
    """
    withheld = [
        (unit, cid, v)
        for unit, b in demo_built.items()
        for cid, v in b.record.claims_used
        if v.startswith("withheld:")
    ]
    assert withheld, (
        "no curated claim came back withheld on this panel; either every canonical "
        "marker is on panel, or the lookup is narrowed to what the method reported "
        "and the off-panel half is being dropped"
    )
    assert any(v == "withheld:never_measured" for _u, _c, v in withheld)


@pytest.mark.parametrize("study_id", _studies())
def test_the_claim_layer_cannot_move_a_number(
        study_id: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reference data may select and order evidence. It may never supply a value.

    Built twice -- once normally, once with the claim store emptied -- and every
    band, every ``verify`` and the whole v1 projection must be identical. This is
    the assertion that makes the wiring safe to keep: if it fails, a curated claim
    has reached a number.
    """
    study = Study.open(study_id)
    # Claims are narrowed by the unit's LABEL, so "the store is non-empty" is the
    # wrong precondition: the curated seed is breast-specific and has nothing to
    # say about a synthetic cell type, and zero claims there is the honest answer
    # rather than a dropped lookup. What makes the second arm meaningful is a claim
    # whose object is one of THIS study's labels.
    labels = {str(m.get("cell_type", "")) for m in study.unit_labels.values()}
    had_claims = any(c.object in labels
                     for c in records._claim_store(study)[0])

    with_claims = records.build_all(study)
    v1_with = {u: records.to_v1_dict(b) for u, b in with_claims.items()}

    monkeypatch.setattr(records, "_claim_store",
                        lambda _study: (ClaimStore(), ""))
    without = records.build_all(study)
    v1_without = {u: records.to_v1_dict(b) for u, b in without.items()}

    assert set(with_claims) == set(without)
    for unit in with_claims:
        assert with_claims[unit].verdict.band == without[unit].verdict.band
        assert with_claims[unit].verdict.verify is without[unit].verdict.verify
        assert with_claims[unit].assignment.band == without[unit].assignment.band
    assert v1_with == v1_without, (
        "the v1 view moved when the claim store was emptied; reference data "
        "reached a value"
    )
    # ...and the emptied arm really is empty, so the comparison had two arms.
    assert not any(b.record.claims_used for b in without.values())
    if had_claims:
        assert any(b.record.claims_used for b in with_claims.values()), (
            f"{study_id} loaded claims but used none; the narrowing dropped "
            "everything and the comparison above is vacuous"
        )


def test_no_store_and_nothing_to_say_are_different_records(
        tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """The collapse this project exists to prevent, in the claim layer.

    ``claims_used == ()`` is produced by two different facts: the curated store had
    nothing about this label, and there was no curated store at all. Only the second
    is something the biologist can fix, and before ``claims:no_store`` the two
    records were byte-identical -- a silence standing in for a negative result,
    which is the failure the four-state lattice is built around.
    """
    import shutil

    from panoscope.data import layout

    # A data root holding the studies and NO claims/ seed.
    root = tmp_path / "no_seed"  # type: ignore[operator]
    shutil.copytree(layout.studies_root(), root / "studies")
    monkeypatch.setenv("PANOSCOPE_DATA_ROOT", str(root))
    layout.clear_cache()
    try:
        study = Study.open("synthetic_multisample_fixture")
        store, absent = records._claim_store(study)
        assert len(store) == 0 and absent, "this arm must have no seed to read"
        rec = records.build_all(study)["c1"].record
        rules = [d.rule for d in rec.decisions if d.rule.startswith("claims")]
        assert rec.claims_used == ()
        assert "claims:no_store" in rules, (
            "a build with no claim store must say so; an empty claims_used alone "
            "is indistinguishable from a store that had nothing to say"
        )
    finally:
        monkeypatch.delenv("PANOSCOPE_DATA_ROOT", raising=False)
        layout.clear_cache()

    # The other arm: the real root, whose seed loads but holds nothing for this
    # study's synthetic labels. Same empty tuple, and NO no_store decision.
    rec = records.build_all(Study.open("synthetic_multisample_fixture"))["c1"].record
    assert rec.claims_used == ()
    assert "claims:no_store" not in [d.rule for d in rec.decisions]


@pytest.mark.parametrize("study_id", _studies())
def test_the_committed_records_are_what_the_code_writes_today(
        study_id: str) -> None:
    """The staleness guard the repo did not have.

    ``interp/records/`` is committed and its sha256 is carried in the manifest, so a
    record that drifts from the code is a published artifact asserting something the
    pipeline no longer produces. This caught 2522-decision records that predated the
    per-unit trace fix. Rebuild the study (``panoscope build --study <id>``) when it
    fails; the failure is not a reason to loosen the comparison.
    """
    from panoscope.core import record as core_record

    study = Study.open(study_id)
    committed = sorted((study.interp / "records" / "assignment").glob("*.json"))
    if not committed:
        pytest.skip(f"{study_id} carries no committed assignment records")

    built = records.build_all(study)
    stale: list[str] = []
    for path in committed:
        unit = path.stem
        assert unit in built, f"{study_id}: committed record for unknown unit {unit}"
        live = json.dumps(core_record.to_dict(built[unit].record).value,
                          indent=2) + "\n"
        if path.read_text(encoding="utf-8") != live:
            stale.append(unit)
    assert not stale, (
        f"{study_id}: committed records are stale for {stale}. Rebuild with "
        f"`panoscope build --study {study_id}`."
    )
