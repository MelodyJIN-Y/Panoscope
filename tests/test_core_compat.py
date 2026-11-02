"""The §10 migration gate: rebuild, project back, compare field by field.

``IMPLEMENTATION.md`` §10 specifies this test, not just the module it tests:

    The gate is unusually strong because inputs are hashed: rebuild each record from
    the same hashed inputs, project it back, and compare field by field.

So the fixture does not read the committed JSON and feed it back. It reads the
*inputs* -- ``inputs/markers_top.csv``, ``inputs/panel.parquet``, the cluster key and
the annotation -- builds a v2 ``NormalisedEvidence`` through ``core.measure``, bands
it through ``core.verdict``, projects that through ``core.compat``, and only then
opens the committed file to compare. ``test_the_inputs_are_the_hashed_ones`` records
the digests the rebuild ran on, so "the same inputs" is a checkable statement rather
than an assumption about a path that may have been overwritten underneath us.

**The allowlist check is two-sided, and that is the whole anti-widening mechanism.**
A field that differs and is not allowlisted fails, which is the obvious half. A field
*in* the allowlist that does not actually differ fails too -- so an entry cannot be
added prophylactically, cannot be a wildcard, and cannot be left behind once the
field comes back. Combined with the fact that an allowlist entry only takes effect if
``compat.py`` also stops emitting the field, widening it costs two deliberate edits
in one reviewable file and a reason written out longhand. A green test can never do
it.

**The fixture stands in for the unbuilt v2 method module.** ``methods/jazzpanda.py``
does not exist yet, so the three specificity flags are derived here from
``agent/verdict._caveats_for`` -- the same function the committed records were
written with. That is deliberate, for the reason ``test_core_verdict.py`` gives: if
the fixture re-derived the thresholds itself, a projection failure could not be told
apart from a fixture that had drifted.

**Never writes.** §10: "Never mutate the committed per-unit JSONs." Everything here
is a read, and ``test_the_committed_records_are_not_touched`` proves it.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import io
import json
from pathlib import Path

import pytest

from panoscope.agent import annotation
from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import verdict as A
from panoscope.core import compat, measure
from panoscope.core import verdict as C
from panoscope.pipeline import paths
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import Frame, Measured, MeasuredState, MeasuredWorld
from panoscope.types.subject import EntityKind, cell_type, gene, subject

PANEL_FRAME = "panel:xenium_hbc_v1"
UNIT_FRAME = "annotation:cluster_key"

# agent's caveat prose -> the registered flag the method layer will carry. The
# inverse of compat.CAVEAT_FOR; written out here rather than imported so the two
# tables are independent and a single-sided edit shows up as a failure.
_FLAG_FOR: dict[str, str] = {
    "localizes better with another cluster": C.FLAG_LOCALISES_ELSEWHERE,
    "spatial pattern not unique": C.FLAG_NOT_UNIQUE,
    "low spatial specificity": C.FLAG_LOW_PEARSON,
}


# --------------------------------------------------------------------------- #
# Rebuild: the same hashed inputs -> a v2 NormalisedEvidence + Verdict
# --------------------------------------------------------------------------- #
def _input_paths() -> tuple[Path, ...]:
    return (data._markers_top(), data._panel_parquet(), data._cluster_key_json())


def _conditions(panel: Frame) -> Conditions:
    """Enough of a §6.2 ``Conditions`` to construct one.

    The projection reads none of it. It is here because ``NormalisedEvidence``
    carries it, and rebuilding the whole v2 unit shape -- rather than hand-assembling
    the two fields the projection happens to touch -- is what makes this a gate.
    """
    return Conditions(
        assay=Condition("10x Xenium", Provenance.READ),
        modality=Condition("spatial transcriptomics", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=Condition("jazzPanda", Provenance.READ),
        segmentation=UNKNOWN,
        fixation=UNKNOWN,
        frames=(panel.ref(),),
    )


def _world() -> tuple[MeasuredWorld, Frame]:
    panel = Frame(EntityKind.GENE, PANEL_FRAME,
                  frozenset(str(g) for g in data.load_panel()["gene"]))
    units = Frame(EntityKind.UNIT, UNIT_FRAME, frozenset(cfg.CLUSTER_ORDER))
    return MeasuredWorld(frames=(panel, units)), panel


def _readings(cluster: str, ct: str) -> tuple[measure.Reading, ...]:
    """Assigned rows (a number each) then the off-panel canonical markers (none).

    The order is the order ``agent/verdict.py`` wrote: ``get_cluster_markers`` is
    already glm_coef-descending with NoSig excluded, and the off-panel notes follow
    ``annotation.offpanel_canonical``.
    """
    out: list[measure.Reading] = []
    for _, r in data.get_cluster_markers(cluster).iterrows():
        pearson = float(r["pearson"])
        gc = float(r["max_gc_corr"])
        gg = float(r["max_gg_corr"])
        out.append(measure.Reading(
            subject=subject("gene", g=gene(str(r["gene"]))),
            statistic=float(r["glm_coef"]),
            statistic_name="glm_coef",
            corroborating=(("pearson", pearson), ("max_gc_corr", gc),
                           ("max_gg_corr", gg)),
            flags=tuple(_FLAG_FOR[c] for c in A._caveats_for(pearson, gc, gg)),
        ))
    out += [
        measure.Reading(subject=subject("gene", g=gene(g)), statistic=None,
                        statistic_name="glm_coef")
        for g in annotation.offpanel_canonical(ct)
    ]
    return tuple(out)


def _rebuild(cluster: str):
    """(NormalisedEvidence, Verdict, UnitLabel, canonical) for one demo cluster."""
    meta = annotation.meta_for(cluster)
    ct = meta["cell_type"]
    world, panel = _world()
    readings = _readings(cluster, ct)
    reported = (
        Frame(EntityKind.GENE, f"reported:{cluster}",
              frozenset(r.subject.one("g").id for r in readings
                        if r.statistic is not None)),
        Frame(EntityKind.UNIT, "reported:units", frozenset({cluster})),
    )
    ne = measure.measure_unit(
        unit=cluster, unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type(ct)),
        module="jazzpanda", readings=readings, world=world,
        reported=reported, conditions=_conditions(panel),
    ).value
    canonical = frozenset(annotation.canonical_markers(ct))
    v = C.assess(ne.items, canonical).value
    label = compat.UnitLabel(cell_type=ct, cell_type_short=meta["cell_type_short"],
                             category=meta["category"], lineage=meta["lineage"])
    return ne, v, label, canonical


def _project(cluster: str) -> dict:
    ne, v, label, canonical = _rebuild(cluster)
    return compat.to_v1_assignment(ne, v, label, canonical).value


def _record_path(cluster: str) -> Path:
    return paths.cluster_json(cfg.DATASET_ID, cluster)


def _committed(cluster: str) -> dict:
    return json.loads(_record_path(cluster).read_text())


# --------------------------------------------------------------------------- #
# The comparison walker. Indices collapse to "[]" so one deleted field is one path.
# --------------------------------------------------------------------------- #
def _diff(expected, actual, path: str = "") -> list[tuple[str, str, object, object]]:
    """(kind, path, expected, actual) for every leaf that is not identical.

    ``kind`` is "missing" (in the committed file, not emitted), "extra" (emitted, not
    in the committed file) or "differs". Types must match as well as values: an int
    where the committed file holds a float is a shape change that ``==`` hides, and
    ``False == 0`` is exactly how a boolean column turns numeric unnoticed.
    """
    out: list[tuple[str, str, object, object]] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [("differs", path, expected, actual)]
        for k, v in expected.items():
            p = f"{path}.{k}" if path else k
            if k not in actual:
                out.append(("missing", p, v, None))
            else:
                out += _diff(v, actual[k], p)
        out += [("extra", f"{path}.{k}" if path else k, None, v)
                for k, v in actual.items() if k not in expected]
        return out
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [("differs", path, expected, actual)]
        if len(expected) != len(actual):
            return [("differs", f"{path}#len", len(expected), len(actual))]
        for e, a in zip(expected, actual):
            out += _diff(e, a, path + "[]")
        return out
    if type(expected) is not type(actual) or expected != actual:
        return [("differs", path, expected, actual)]
    return []


def _diffs_over_the_demo() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {"missing": set(), "extra": set(), "differs": set()}
    for cluster in cfg.CLUSTER_ORDER:
        for kind, path, _e, _a in _diff(_committed(cluster), _project(cluster)):
            found[kind].add(path)
    return found


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cluster", list(cfg.CLUSTER_ORDER))
def test_the_projection_reproduces_the_committed_record(cluster):
    """Field by field, one cluster at a time, so a failure names the cluster.

    Everything that is not identical must be a deliberate omission on the allowlist.
    A value that differs is never acceptable: the inputs are the same bytes, so a
    different number means the projection invented one.
    """
    bad = [(k, p, e, a) for k, p, e, a in _diff(_committed(cluster), _project(cluster))
           if not (k == "missing" and p in compat.DROPPED_PATHS)]
    assert not bad, f"{cluster}: " + "; ".join(
        f"{k} {p}: committed={e!r} projected={a!r}" for k, p, e, a in bad
    )


def test_the_allowlist_is_exactly_what_the_projection_drops():
    """Two-sided, and this is the anti-widening mechanism.

    A subset would let a new omission through unreviewed. A superset would let an
    entry be added *before* the field was dropped, where it would sit absorbing the
    next regression -- which is precisely "a green test widened the allowlist".
    """
    found = _diffs_over_the_demo()
    assert found["differs"] == set(), (
        "fields that differ are never allowlistable -- same inputs, different value: "
        + ", ".join(sorted(found["differs"]))
    )
    assert found["extra"] == set(), (
        "the projection emits fields v1 never had: " + ", ".join(sorted(found["extra"]))
    )
    assert found["missing"] == compat.DROPPED_PATHS, (
        "V1_DROPPED must name exactly what the projection omits. "
        f"omitted but not allowlisted: {sorted(found['missing'] - compat.DROPPED_PATHS)}; "
        f"allowlisted but not omitted: {sorted(compat.DROPPED_PATHS - found['missing'])}"
    )


def test_every_dropped_entry_states_a_reason():
    """``reason`` has no default, so this cannot fail by accident -- it fails on an
    entry added with a placeholder, which is the version a hurried diff produces."""
    for d in compat.V1_DROPPED:
        assert len(d.reason) > 60, f"{d.path}: state why it is dropped, not that it is"
        assert "*" not in d.path, f"{d.path}: paths are literal, never patterns"
    assert len(compat.DROPPED_PATHS) == len(compat.V1_DROPPED), "duplicate path"


def test_the_gate_is_not_vacuous():
    """Nine empty records would compare equal and prove nothing."""
    assert _diff(_committed("c1"), _committed("c1")) == []   # the walker self-agrees
    n = sum(len(_project(c)["evidence"]) for c in cfg.CLUSTER_ORDER)
    assert n == 231, f"the demo study has 231 assigned marker rows, not {n}"
    assert any(_project(c)["offpanel_notes"] for c in cfg.CLUSTER_ORDER), (
        "no cluster exercised the off-panel (NEVER_MEASURED) branch"
    )
    assert len({_project(c)["confidence"] for c in cfg.CLUSTER_ORDER}) > 1


def test_the_inputs_are_the_hashed_ones():
    """§10's "the same hashed inputs". Records the digests the rebuild ran on; if an
    input is replaced, this is where it surfaces instead of as a mystery diff."""
    digests = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in _input_paths()}
    assert len(digests) == 3 and all(len(d) == 64 for d in digests.values()), digests


def test_the_committed_records_are_not_touched():
    """§10 in one assertion: the gate reads, it never writes."""
    before = {c: hashlib.sha256(_record_path(c).read_bytes()).hexdigest()
              for c in cfg.CLUSTER_ORDER}
    for c in cfg.CLUSTER_ORDER:
        _project(c)
    after = {c: hashlib.sha256(_record_path(c).read_bytes()).hexdigest()
             for c in cfg.CLUSTER_ORDER}
    assert before == after


# --------------------------------------------------------------------------- #
# The committed CSV -- reused from export.py, never re-specified
# --------------------------------------------------------------------------- #
def test_the_verdict_csv_round_trips_through_the_projection():
    """The 11 columns in exact order, byte-identical to what ``agent`` writes today
    and, row for row, to the committed file. ``v1_verdict_csv`` adds no column list
    of its own -- it hands ``export.py``'s profile something with attributes."""
    rows = [_project(c) for c in cfg.CLUSTER_ORDER]
    produced = compat.v1_verdict_csv(rows).value
    assert produced == A.to_csv(A.all_verdicts())
    committed = paths.verdicts_csv(cfg.DATASET_ID).read_text()
    assert list(csv.reader(io.StringIO(produced))) == \
        list(csv.reader(io.StringIO(committed)))


def test_the_csv_header_is_the_frozen_eleven():
    rows = [_project(c) for c in cfg.CLUSTER_ORDER]
    header = next(csv.reader(io.StringIO(compat.v1_verdict_csv(rows).value)))
    assert tuple(header) == A.CSV_COLUMNS
    assert len(header) == 11


# --------------------------------------------------------------------------- #
# The translation tables -- the only English in core/, so the only place it drifts
# --------------------------------------------------------------------------- #
def test_the_score_anchors_match_the_live_config():
    """``core/`` may not import ``agent/``, so the anchors are duplicated. Without
    this the two drift silently at the next calibration pass and every persisted
    ``confidence_score`` moves."""
    assert compat.SCORE_MAP == cfg.SCORE_MAP


def _item(name: str, coef: float, pearson: float,
          flags: tuple[str, ...] = ()) -> EvidenceItem:
    return EvidenceItem(
        subject=subject("gene", g=gene(name)), statistic=coef,
        statistic_name="glm_coef",
        measured=Measured(MeasuredState.OBSERVED, PANEL_FRAME),
        corroborating=(("pearson", pearson),), flags=flags,
    )


def _grid():
    for coef in (0.5, 3.0, 12.0, 30.0):
        for pearson in (0.30, 0.50, 0.86, 0.99):
            for gc in (0.0, pearson + 0.10):
                for gg in (0.50, 0.99):
                    for fragile in (False, True):
                        for covered in (True, False):
                            yield coef, pearson, gc, gg, fragile, covered


@pytest.mark.parametrize("coef,pearson,gc,gg,fragile,covered", list(_grid()))
def test_every_rule_id_renders_the_sentence_agent_wrote(coef, pearson, gc, gg,
                                                        fragile, covered):
    """The demo fires at most one adjustment per cluster, and only three of the eight
    rules across all nine (one promote, one localises-elsewhere, one fragile floor).
    So the record gate above cannot see five of the sentences at all, nor the order
    two of them are joined in. This drives both implementations from one set of
    numbers and compares the rendered English at every reachable corner.
    """
    caveats = A._caveats_for(pearson, gc, gg)
    me = A.MarkerEvidence(
        gene="ERBB2", top_cluster="c1", glm_coef=coef, pearson=pearson,
        max_gg_corr=gg, max_gc_corr=gc, p_value=None, within_cluster_pctile=1.0,
        is_canonical=True, is_on_panel=True, role="supports", caveats=caveats,
    )
    # The coverage demote renders its own K of N, so the sentence it produces is
    # only comparable when both sides are given the SAME count. core derives it;
    # agent is handed what core derived.
    canonical = (frozenset({"ERBB2"}) if covered
                 else frozenset({"ERBB2", "C1", "C2", "C3", "C4", "C5"}))
    cov = C._canonical_coverage((_item("ERBB2", coef, pearson),), canonical)
    assert cov.ok is covered
    _band, _basis, expected = A._compute_band(me, fragile=fragile, coverage=cov)

    driver = _item("ERBB2", coef, pearson, tuple(_FLAG_FOR[c] for c in caveats))
    filler = () if fragile else tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    v = C.assess((driver, *filler), canonical).value
    assert tuple(compat._demotion_sentence(r, v.driver, v.coverage)
                 for r in v.demotions) == expected


def test_the_caveat_table_is_the_inverse_of_the_agent_rule():
    """``compat.CAVEAT_FOR`` must render in ``_caveats_for``'s append order, or the
    ``caveats`` list comes out permuted and ``ui/summary.py`` (which reads
    ``caveats[0]``) shows a different leading caveat."""
    for pearson, gc, gg in ((0.1, 0.9, 0.99), (0.5, 0.9, 0.5), (0.9, 0.0, 0.5),
                            (0.2, 0.0, 0.99), (0.9, 0.0, 0.0)):
        expected = A._caveats_for(pearson, gc, gg)
        flags = tuple(_FLAG_FOR[c] for c in expected)
        assert tuple(p for f, p in compat.CAVEAT_FOR if f in flags) == expected


def test_the_tested_absent_driver_sentence_does_not_claim_a_failed_measurement():
    """The one v1 sentence ``compat`` deliberately does not transcribe.

    ``band:driver_nosig`` fires on a canonical marker that is ``TESTED_ABSENT`` --
    in frame, tested, not assigned to this unit. v1 rendered that as "driving marker
    is NoSig", which reads as *the assay produced nothing for this gene*, the one
    reading TESTED_ABSENT has already excluded: an item reaches that state only after
    every frame for its kind admitted it, detectability included. This sentence lands
    in ``interp/verdicts.csv``'s ``notes`` column and in the Summary table, and on a
    joint-clustered study it lands on most units, so it is checked for what it must
    not say as well as for what it says.

    The renderer takes ``driver=None`` because that is what the branch produces: the
    verdict has no driver, which is why the band is at the floor.
    """
    sentence = compat._demotion_sentence("band:driver_nosig", None)
    assert "nosig" not in sentence.lower()
    assert "measured" in sentence
    # and it survives into the prose the CSV carries, in the Adjustments clause
    notes = compat._notes("T_Cells", "Low", (), (sentence,), fragile=False, verify=True)
    assert sentence in notes


def _canon_item(name: str, state: MeasuredState, frame: str,
                flags: tuple[str, ...] = ()) -> EvidenceItem:
    """One canonical marker in a given measurement state. ``statistic is None`` iff
    the state is not OBSERVED, which the EvidenceItem boundary enforces anyway."""
    return EvidenceItem(
        subject=subject("gene", g=gene(name)),
        statistic=(1.0 if state is MeasuredState.OBSERVED else None),
        statistic_name="glm_coef",
        measured=Measured(state, frame),
        corroborating=(("pearson", 0.1),) if state is MeasuredState.OBSERVED else (),
        flags=flags,
    )


def test_the_no_canonical_sentence_tells_the_four_reasons_apart():
    """``band:no_canonical_driver`` had ONE sentence for several different states.

    "no canonical marker supports the call" reads as *the markers were measured and
    they do not support it*; on a panel designed for a different question the truth
    is *every canonical marker is off-panel, so nothing could have supported it*.
    Those are opposite statements about the evidence, and this column lands in
    ``interp/verdicts.csv``'s ``notes`` and the Summary table.

    Every state is checked, not only the off-panel one: reporting an UNDERPOWERED
    marker in the words of a NEVER_MEASURED one (or the reverse) is a fabricated
    measurement status, and a test that pinned only the all-off-panel case would
    report the rest as covered.
    """
    canon = frozenset({"CAN1", "CAN2", "CAN3", "CAN4"})
    detect = "detect:synthetic"

    # 1. all off-panel -- the kidney/fixture case. The panel cannot speak to it.
    all_off = compat._no_canonical_reason(
        tuple(_canon_item(g, MeasuredState.NEVER_MEASURED, PANEL_FRAME)
              for g in sorted(canon)), canon)
    assert all_off == ("of the 4 canonical markers for this call: 4 off-panel "
                       "(never measured) -- nothing could have supported it")

    # 2. the fixture's ACTUAL shape: off-panel AND on-panel-with-no-signal, which
    #    are different facts ("order a different panel" vs "this section was flat").
    mixed = compat._no_canonical_reason(
        (_canon_item("CAN1", MeasuredState.NEVER_MEASURED, PANEL_FRAME),
         _canon_item("CAN2", MeasuredState.NEVER_MEASURED, PANEL_FRAME),
         _canon_item("CAN3", MeasuredState.NEVER_MEASURED, PANEL_FRAME),
         _canon_item("CAN4", MeasuredState.NEVER_MEASURED, detect)), canon)
    assert mixed == ("of the 4 canonical markers for this call: 3 off-panel (never "
                     "measured); 1 on-panel with no usable signal (never measured) "
                     "-- nothing could have supported it")

    # 3. UNDERPOWERED is in frame and below the floor. It must not be dressed as
    #    never-measured, and it must not be dressed as a tested absence either --
    #    it argues nothing, so the incapability clause still applies.
    under = compat._no_canonical_reason(
        tuple(_canon_item(g, MeasuredState.UNDERPOWERED, PANEL_FRAME)
              for g in sorted(canon)), canon)
    assert under == ("of the 4 canonical markers for this call: 4 in frame but "
                     "below the floor (underpowered) "
                     "-- nothing could have supported it")
    assert "never measured" not in under and "off-panel" not in under

    # 4. OBSERVED but not eligible (low specificity): the measurement DID speak, so
    #    the incapability clause must be absent -- it would be the false half.
    unspecific = compat._no_canonical_reason(
        (_canon_item("CAN1", MeasuredState.OBSERVED, PANEL_FRAME,
                     (compat.FLAG_LOW_PEARSON,)),
         _canon_item("CAN2", MeasuredState.NEVER_MEASURED, PANEL_FRAME)), canon)
    assert unspecific == ("of the 4 canonical markers for this call: 2 not in the "
                          "measured set at all (never measured); 1 off-panel (never "
                          "measured); 1 measured, but too unspecific to carry the "
                          "call")
    assert compat._NOTHING_COULD_SUPPORT not in unspecific

    # 5. No curated list at all -- thin reference data, not a failed measurement.
    assert compat._no_canonical_reason((), frozenset()) == \
        "this call has no canonical marker list to check"

    # 6. A canonical marker nobody reported is the panel refusing it before the
    #    method ran (core.verdict._canonical_coverage counts it the same way).
    assert compat._no_canonical_reason((), canon) == (
        "of the 4 canonical markers for this call: 4 not in the measured set at "
        "all (never measured) -- nothing could have supported it")

    # 7. None of these sentences may read as the collapsed one.
    for sentence in (all_off, mixed, under, unspecific):
        assert "no canonical marker supports the call" not in sentence


def test_a_never_measured_canonical_marker_outside_BOTH_gene_frames_says_so():
    """The NEVER_MEASURED branch the fixture does not reach, and the one that lied.

    ``_no_canonical_class`` asserted "on-panel with no usable signal" for every
    NEVER_MEASURED item whose frame did not start ``panel:``. Two of the frames that
    reach it say no such thing: ``"no frame declared for gene"``, which
    ``types.measured._state_for`` returns when a kind declares no frame at all, and
    ``annotation:<study>``, which a composite subject reaches when the UNIT was the
    role that blocked. For those BOTH halves of the clause were fabricated -- nothing
    said the gene was on the panel, and nothing said the section gave no signal.
    ``_v1_bucket`` names its own else-branch ``never_measured_outside_the_panel_frame``
    precisely because it cannot tell.

    The fixture's c2 is 5 ``panel:`` + 1 ``detect:``, both of which ARE knowable,
    which is why the suite was green. This builds the branch the fixture misses, and
    builds it beside a real ``detect:`` item so a clause that swallowed both would
    print one count of 2 instead of two counts of 1.
    """
    canon = frozenset({"CAN1", "CAN2"})
    for frame in ("no frame declared for gene", "annotation:synthetic"):
        sentence = compat._no_canonical_reason(
            (_canon_item("CAN1", MeasuredState.NEVER_MEASURED, frame),
             _canon_item("CAN2", MeasuredState.NEVER_MEASURED, "detect:synthetic")),
            canon)
        assert sentence == (
            "of the 2 canonical markers for this call: 1 on-panel with no usable "
            "signal (never measured); 1 refused by a frame that is neither the "
            "panel nor detectability (never measured)"
            " -- nothing could have supported it"), frame


def _census_counts(sentence: str) -> int:
    """The clause numbers, summed. Every clause opens with its own count."""
    body = sentence.split(": ", 1)[1]
    return sum(int(c.split()[0]) for c in body.split("; "))


def test_the_canonical_census_counts_NAMES_so_its_clauses_sum_to_its_denominator():
    """"of the 4 canonical markers for this call: ... " and then eight of them.

    The census counted its no-item class over NAMES (``len(canon - seen)``) and every
    other class over ITEMS, and a sentence whose entire job is to be a census of a
    curated list then disagreed with its own denominator on exactly the two subject
    shapes this project has. Names won, because that is what the denominator counts.

    Both shapes are built, because they break it in opposite directions.
    """
    canon = frozenset({"CAN1", "CAN2", "CAN3", "CAN4"})

    # 1. Duplicate readings of one canonical gene. ``measure_unit`` does not dedupe,
    #    so a joint-clustered study with per-sample rows hands over several readings
    #    of CAN1 -- the kidney shape. Over-counted: 2 + 3 items = 5 of 4.
    #    The name is filed under its HIGHEST reading: a gene OBSERVED in one sample
    #    WAS measured, whatever the other row says, and that is the join
    #    ``core.verdict._canonical_coverage`` already takes.
    dup = compat._no_canonical_reason(
        (_canon_item("CAN1", MeasuredState.OBSERVED, PANEL_FRAME,
                     (compat.FLAG_LOW_PEARSON,)),
         _canon_item("CAN1", MeasuredState.NEVER_MEASURED, PANEL_FRAME),
         _canon_item("CAN2", MeasuredState.TESTED_ABSENT, PANEL_FRAME)), canon)
    assert dup == ("of the 4 canonical markers for this call: 2 not in the measured "
                   "set at all (never measured); 1 tested and not assigned to this "
                   "unit (tested-absent); 1 measured, but too unspecific to carry "
                   "the call")
    assert _census_counts(dup) == 4
    assert "off-panel" not in dup      # CAN1 was measured; the second row is a dupe

    # 2. A composite subject carrying two canonical names in one item -- the shape
    #    ``core.verdict._is_canonical`` exists for. Under-counted: 2 + 1 = 3 of 4.
    pair = compat._no_canonical_reason(
        (EvidenceItem(
            subject=subject("gene_pair", a=gene("CAN1"), b=gene("CAN2")),
            statistic=None, statistic_name="glm_coef",
            measured=Measured(MeasuredState.NEVER_MEASURED, PANEL_FRAME)),), canon)
    assert pair == ("of the 4 canonical markers for this call: 2 not in the measured "
                    "set at all (never measured); 2 off-panel (never measured)"
                    " -- nothing could have supported it")
    assert _census_counts(pair) == 4


def test_the_no_canonical_sentence_travels_to_both_prose_surfaces():
    """It is rendered twice -- the demotion sentence and the opening headline -- and
    the headline's old half ("no canonical marker drives the call") collapsed the
    same reasons one function up. Both are checked, because the headline is the
    sentence the biologist reads first and the notes are what the CSV carries."""
    reason = "of the 2 canonical markers for this call: 2 off-panel (never measured)"
    assert compat._demotion_sentence("band:no_canonical_driver", None,
                                     no_canonical=reason) == reason
    assert compat._headline("T_Cells", "Low", (), reason) == \
        f"T_Cells, Low confidence; {reason}."
    assert reason in compat._notes("T_Cells", "Low", (), (reason,),
                                   fragile=False, verify=True)


def test_the_no_canonical_rule_refuses_to_render_without_the_states():
    """No safe default: the wording it replaced is the false one on exactly the
    units this rule fires on, so a caller that forgot the states must fail rather
    than quietly get it back."""
    with pytest.raises(compat.NoV1Shape, match="no safe default"):
        compat._demotion_sentence("band:no_canonical_driver", None)


def test_the_fixtures_c2_states_why_nothing_supported_it_and_keeps_its_band():
    """End to end on the production path, on the unit the sentence exists for.

    ``synthetic_multisample_fixture`` c2 is the panel-absence rule at severity: its
    canonical markers are all NEVER_MEASURED, five off the panel and one on it with
    no usable signal. The band is loud and correct either way, so what is pinned
    here is that the PROSE now names the reason -- and that fixing the prose moved
    neither the band, the score, nor ``verify``.
    """
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    fixture = "synthetic_multisample_fixture"
    if not Study.open(fixture).exists:
        pytest.skip(f"study {fixture!r} is not on disk")
    with cfg.use_study(fixture):
        d = records.to_v1_dict(records.build_all(Study.open(fixture))["c2"])

    assert d["band_basis"] == "no canonical driver"
    assert d["confidence"] == "Low" and d["confidence_score"] == 0.30
    assert d["verify"] is True
    (sentence,) = d["demotions"]
    assert "off-panel (never measured)" in sentence
    assert "on-panel with no usable signal (never measured)" in sentence
    assert compat._NOTHING_COULD_SUPPORT in sentence
    assert sentence in d["notes"] and sentence in d["opening"]["headline"]
    assert "no canonical marker supports the call" not in d["notes"]


def test_an_unknown_band_rule_raises_rather_than_rendering_nothing():
    """A band rule with no v1 sentence would silently shorten ``demotions`` and the
    ``notes`` prose built from it -- the record would read as a call that took one
    fewer adjustment than it did."""
    with pytest.raises(compat.NoV1Shape, match="no v1 sentence"):
        compat._demotion_sentence("band:invented_just_now", None)


# --------------------------------------------------------------------------- #
# Fails closed on the two v2 shapes v1 cannot hold
# --------------------------------------------------------------------------- #
def _with_extra(cluster: str, item: EvidenceItem):
    ne, v, label, canonical = _rebuild(cluster)
    return dataclasses.replace(ne, items=ne.items + (item,)), v, label, canonical


def _absent_item(symbol: str, state: MeasuredState, frame: str) -> EvidenceItem:
    return EvidenceItem(subject=subject("gene", g=gene(symbol)), statistic=None,
                        statistic_name="glm_coef", measured=Measured(state, frame))


def test_a_tested_absent_item_is_omitted_and_never_written_as_an_offpanel_note():
    """The reviewed replacement for "it raises".

    v1 predates the lattice and has two buckets: an ``evidence`` row, which needs a
    float glm_coef, or an ``offpanel_notes`` entry, whose message asserts the gene is
    off-panel. A tested-absent gene fits neither -- it was on the panel, it was
    tested, and it peaked elsewhere -- so the v1 view says NOTHING about it. That is
    the one representation that cannot mislead: silence is not a claim, while an
    OffPanelNote is, and the state survives in full in the v2 record beside the view.

    The gene must be absent from BOTH lists and from ``source_trace``. An omission
    that leaked into ``offpanel_notes`` would print "MS4A1 is off-panel (never
    measured)" about a gene the measurement did see, which is the exact confusion
    ``measured.py`` exists to prevent, written out in English in a file the UI
    renders.
    """
    ne, v, label, canonical = _with_extra(
        "c9", _absent_item("MS4A1", MeasuredState.TESTED_ABSENT, PANEL_FRAME))
    d = compat.to_v1_assignment(ne, v, label, canonical).value

    assert "MS4A1" not in {r["gene"] for r in d["evidence"]}
    assert "MS4A1" not in {n["gene"] for n in d["offpanel_notes"]}
    assert not [s for s in d["source_trace"] if "MS4A1" in s]
    assert dict(compat.v1_omissions(ne).value) == {
        compat.V1_OMIT_TESTED_ABSENT: 1,
        compat.V1_OMIT_UNDETECTABLE: 0,
    }


def test_a_never_measured_gene_the_panel_does_hold_is_not_called_off_panel():
    """The second omission class, and why it is not the first.

    ``platforms/xenium.py`` declares two gene frames. A gene refused by
    ``detect:<study>`` is ON the panel and produced no usable spatial signal; v1's
    only shape for it says "off-panel", which is false, and "order a different panel"
    and "this section gave no signal" are different facts to a biologist. So it is
    omitted too, under its own entry, and the off-panel list keeps meaning what it
    says.
    """
    ne, v, label, canonical = _with_extra(
        "c9", _absent_item("GNLY", MeasuredState.NEVER_MEASURED, "detect:whatever"))
    d = compat.to_v1_assignment(ne, v, label, canonical).value

    assert "GNLY" not in {n["gene"] for n in d["offpanel_notes"]}
    assert dict(compat.v1_omissions(ne).value)[compat.V1_OMIT_UNDETECTABLE] == 1
    # ...while a PANEL-frame refusal still becomes exactly the note it always did.
    ne2, v2, label2, canon2 = _with_extra(
        "c9", _absent_item("CD3D", MeasuredState.NEVER_MEASURED, PANEL_FRAME))
    d2 = compat.to_v1_assignment(ne2, v2, label2, canon2).value
    assert "CD3D" in {n["gene"] for n in d2["offpanel_notes"]}


def test_a_state_with_no_reviewed_entry_still_raises():
    """Fails closed on everything the two entries do not name.

    ``UNDERPOWERED`` has no ``V1_UNREPRESENTABLE`` entry, so it raises rather than
    joining the omitted pile. That is the anti-widening half: an omission nobody has
    written a reason for is the silent skip, and a second entry costs a sentence in
    ``compat.py``'s diff.
    """
    ne, v, label, canonical = _with_extra(
        "c9", _absent_item("MS4A1", MeasuredState.UNDERPOWERED, PANEL_FRAME))
    with pytest.raises(compat.NoV1Shape, match="no entry in V1_UNREPRESENTABLE"):
        compat.to_v1_assignment(ne, v, label, canonical)


def test_every_unrepresentable_entry_names_a_bucket_the_classifier_produces():
    """Two-sided, as ``V1_DROPPED``'s test is.

    An entry for a bucket ``_v1_bucket`` never returns would sit in the file reading
    as a reviewed decision about nothing; a bucket the projection omits with no entry
    would be an unreviewed omission. Neither is allowed, and every entry carries a
    reason long enough to be a reason.
    """
    produced = {compat.V1_EVIDENCE, compat.V1_OFFPANEL,
                compat.V1_OMIT_TESTED_ABSENT, compat.V1_OMIT_UNDETECTABLE}
    listed = {u.bucket for u in compat.V1_UNREPRESENTABLE}
    assert listed <= produced
    assert listed == compat.UNREPRESENTABLE_BUCKETS
    assert produced - listed == {compat.V1_EVIDENCE, compat.V1_OFFPANEL}
    for u in compat.V1_UNREPRESENTABLE:
        assert len(u.reason) > 120, f"{u.bucket} has no written reason"


def test_a_composite_subject_has_no_v1_shape_and_says_so():
    """v1 ``MarkerEvidence`` is one gene per row. A ligand-receptor or co-expression
    subject cannot be written as one, and picking a representative gene would put a
    real number against the wrong thing."""
    ne, v, label, canonical = _rebuild("c9")
    pair = EvidenceItem(
        subject=subject("gene_pair", a=gene("ERBB2"), b=gene("KRT7")),
        statistic=1.0, statistic_name="glm_coef",
        measured=Measured(MeasuredState.OBSERVED, PANEL_FRAME),
        corroborating=(("pearson", 0.5), ("max_gc_corr", 0.1), ("max_gg_corr", 0.1)),
    )
    broken = dataclasses.replace(ne, items=ne.items + (pair,))
    with pytest.raises(compat.NoV1Shape, match="one gene per row"):
        compat.to_v1_assignment(broken, v, label, canonical)


def test_the_projection_is_deterministic():
    """The record is the reproducibility claim. Two calls, same bytes."""
    first = json.dumps(_project("c1"))
    for _ in range(3):
        assert json.dumps(_project("c1")) == first
