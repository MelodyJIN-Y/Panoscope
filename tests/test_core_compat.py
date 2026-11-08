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

**The three specificity flags** are derived with ``methods/jazzpanda._flags_for``,
the one function that raises them in the product path, so a projection failure
cannot be told apart from a fixture whose thresholds drifted.

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
from panoscope.core import compat, measure
from panoscope.core import verdict as C
from panoscope.methods.jazzpanda import _flags_for
from panoscope.pipeline import paths
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.evidence import EvidenceItem
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld,
)
from panoscope.types.subject import EntityKind, cell_type, gene, subject

PANEL_FRAME = "panel:xenium_hbc_v1"
UNIT_FRAME = "annotation:cluster_key"

# The v1 caveat prose for each registered flag. The inverse of compat.CAVEAT_FOR;
# written out here rather than imported so the two tables are independent and a
# single-sided edit shows up as a failure.
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
            flags=_flags_for(pearson, gc, gg),
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
    man = json.loads(paths.manifest_json(cfg.DATASET_ID).read_text(encoding="utf-8"))
    assert digests == {k: v["sha256"] for k, v in man["inputs"].items()}, digests


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
    """The 11 columns in exact order, byte-identical to the committed file.
    ``v1_verdict_csv`` adds no column list of its own -- it hands ``export.py``'s
    profile something with attributes."""
    rows = [_project(c) for c in cfg.CLUSTER_ORDER]
    produced = compat.v1_verdict_csv(rows).value
    committed = paths.verdicts_csv(cfg.DATASET_ID).read_bytes().decode("utf-8")
    assert produced == committed


def test_the_csv_header_is_the_frozen_eleven():
    rows = [_project(c) for c in cfg.CLUSTER_ORDER]
    header = next(csv.reader(io.StringIO(compat.v1_verdict_csv(rows).value)))
    assert tuple(header) == (
        "cluster", "cell_type", "cell_type_short", "confidence", "confidence_score",
        "key_markers", "notes", "category", "lineage", "exclude", "verify",
    )


# --------------------------------------------------------------------------- #
# The translation tables -- the only English in core/, so the only place it drifts
# --------------------------------------------------------------------------- #
def _item(name: str, coef: float, pearson: float,
          flags: tuple[str, ...] = ()) -> EvidenceItem:
    return EvidenceItem(
        subject=subject("gene", g=gene(name)), statistic=coef,
        statistic_name="glm_coef",
        measured=Measured(MeasuredState.OBSERVED, PANEL_FRAME),
        corroborating=(("pearson", pearson),), flags=flags,
    )


def test_every_v1_rule_id_renders_its_v1_sentence():
    """The demo fires at most one adjustment per cluster, and only three of the eight
    rules across all nine (one promote, one localises-elsewhere, one fragile floor).
    So the record gate above cannot see the other sentences at all, nor the order
    they are joined in. Expected strings are the ones the retired
    ``agent/verdict.py`` wrote, verbatim.
    """
    thin = frozenset({"ERBB2", "C1", "C2", "C3", "C4", "C5"})
    filler = tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    cases = (
        ((_item("ERBB2", 3.0, 0.86, _flags_for(0.86, 0.96, 0.99)), *filler), thin, (
            "high pearson on driver (0.86) -> +1 band",
            "driver localizes better with another cluster -> -1 band",
            "driver spatial pattern not unique -> -1 band",
            "canonical coverage 1/6 below floor 2 -> -1 band",
        )),
        ((_item("ERBB2", 30.0, 0.5),), frozenset({"ERBB2"}), (
            "fragile cluster (<=2 markers): capped at Medium-High",
        )),
        ((_item("ERBB2", 0.5, 0.5, _flags_for(0.5, 0.6, 0.5)),), frozenset({"ERBB2"}), (
            "driver localizes better with another cluster -> -1 band",
            "fragile cluster with canonical driver: floored at Medium",
        )),
    )
    for items, canonical, expected in cases:
        v = C.assess(items, canonical).value
        assert tuple(compat._demotion_sentence(r, v.driver, v.coverage)
                     for r in v.demotions) == expected


def test_the_caveat_table_is_in_the_flag_emitters_order():
    """``compat.CAVEAT_FOR`` must render in ``_flags_for``'s append order, or the
    ``caveats`` list comes out permuted and ``ui/summary.py`` (which reads
    ``caveats[0]``) shows a different leading caveat."""
    prose = {f: p for p, f in _FLAG_FOR.items()}
    for pearson, gc, gg in ((0.1, 0.9, 0.99), (0.5, 0.9, 0.5), (0.9, 0.0, 0.5),
                            (0.2, 0.0, 0.99), (0.9, 0.0, 0.0)):
        flags = _flags_for(pearson, gc, gg)
        assert tuple(p for f, p in compat.CAVEAT_FOR if f in flags) == \
            tuple(prose[f] for f in flags)


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


# --------------------------------------------------------------------------- #
# The input the demo cannot supply: an out-of-frame entity CARRYING A NUMBER
#
# Everything above is parametrised over ``cfg.CLUSTER_ORDER`` of the demo, whose
# method table holds nothing the panel frame refuses. A method table naming a gene
# the panel does not hold is the ordinary case (a control probe, a table carried
# over from another panel): ``core`` refuses the row through ``core.measure`` (the
# panel frame does not admit it, so it comes back NEVER_MEASURED with no statistic,
# and ``EvidenceItem`` makes a numbered row for an unmeasured subject
# unconstructible).
#
# Built on disk, under ``tmp_path``, and run through the production path --
# ``records.build_all`` -- rather than through the hand-assembled ``_rebuild`` above: ``_rebuild`` builds its readings from
# ``data.get_cluster_markers`` and stamps every one of them ``OBSERVED``, so it
# cannot see a frame refusal by construction and would report this as covered.
# --------------------------------------------------------------------------- #
#: A gene no panel file in the repository holds, with a coefficient far above
#: every real row so a missing guard shows up as ``key_markers[0]`` rather than as
#: a tail entry nothing reads.
OUT_OF_FRAME_GENE: str = "OFFPANELGENE1"
_OUT_OF_FRAME_ROW: str = (
    f"{OUT_OF_FRAME_GENE},c3,99.0,0.99,0.5,0.5,Macrophages\n"
)
#: The on-panel row dropped in the fragile variant: with it, the unit has three
#: assigned markers and an unguarded engine stays out of the fragile rule; without
#: it, the unit has two real markers and the out-of-frame row is the only thing
#: standing between ``core``'s fragile cap and an unguarded band.
_DROPPED_ROW_PREFIX: str = "CD68,c3,"


def _study_with_an_out_of_frame_number(tmp_path, monkeypatch, *, drop: bool) -> str:
    """``demo_mini`` on disk under ``tmp_path``, plus one numbered off-panel row.

    Hermetic in the way this repository has twice failed to be:
    ``layout.resolve_data_root`` memoises, so ``monkeypatch.setenv`` redirects
    nothing until ``layout.clear_cache()`` runs AFTER it. The committed studies are
    copied, never edited, and nothing is written outside ``tmp_path``.
    """
    import shutil

    from panoscope.data import layout

    from tests.conftest import DATASETS_ROOT, MINI_ID

    study_id = "offpanel_number_fragile" if drop else "offpanel_number"
    dst = tmp_path / "studies" / study_id
    shutil.copytree(DATASETS_ROOT / MINI_ID, dst)
    table = dst / "inputs" / "markers_top.csv"
    text = table.read_text(encoding="utf-8")
    if drop:
        kept = [ln for ln in text.splitlines(keepends=True)
                if not ln.startswith(_DROPPED_ROW_PREFIX)]
        assert len(kept) == len(text.splitlines()) - 1, "the drop must actually land"
        text = "".join(kept)
    table.write_text(text + _OUT_OF_FRAME_ROW, encoding="utf-8")

    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(tmp_path))
    layout.clear_cache()          # AFTER the setenv, or the env redirects nothing
    return study_id


@pytest.mark.parametrize("drop", [False, True])
def test_the_engine_refuses_a_number_attached_to_an_out_of_frame_gene(
    tmp_path, monkeypatch, drop
):
    """The refusal the demo cannot reach, on both halves of the failure.

    ``drop=False`` is the half an evidence-list check alone would catch: the
    out-of-frame row is quoted as a marker row and leads ``key_markers``, so the
    record says ``glm_coef 99.0`` about a gene the panel never held, beside an
    ``is_on_panel`` of False in the same row.

    ``drop=True`` is the half that reaches the BAND, and it is why this is
    parametrised rather than written once. The unit is left with two on-panel
    markers, so ``core`` counts two assigned items and takes the fragile cap, while
    an engine that counts the out-of-frame row counts three and does not -- the
    out-of-frame gene then moves the one number the biologist acts on without ever
    appearing as a driver. A test pinned on ``drop=False`` alone would report this
    as covered.
    """
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    study_id = _study_with_an_out_of_frame_number(tmp_path, monkeypatch, drop=drop)
    store = Study.open(study_id)
    with cfg.use_study(store):
        built = {u: records.to_v1_dict(b)
                 for u, b in records.build_all(store).items()}

    assert OUT_OF_FRAME_GENE in (store.inputs / "markers_top.csv").read_text(), (
        "the row must be in the method's own table, or this proves nothing"
    )
    assert built, "no unit was built"
    for unit, core_d in sorted(built.items()):
        genes = [r["gene"] for r in core_d["evidence"]]
        assert OUT_OF_FRAME_GENE not in genes, (
            f"{unit}: a statistic is quoted for {OUT_OF_FRAME_GENE}, which the "
            f"panel frame refuses: {genes}"
        )
        assert OUT_OF_FRAME_GENE not in core_d["key_markers"], f"{unit}: key_markers"
        assert all(r["is_on_panel"] for r in core_d["evidence"]), (
            f"{unit}: a row carries a number and says it was never measured"
        )
        assert not [s for s in core_d["source_trace"]
                    if s.startswith(f"jz:{OUT_OF_FRAME_GENE}:")], (
            f"{unit}: a number for the out-of-frame gene reaches the source trace"
        )
    # The band half: with two real markers left, c3 is fragile.
    assert built["c3"]["small_n"] is drop


def test_the_out_of_frame_study_actually_reaches_the_rule_it_was_built_for(
    tmp_path, monkeypatch
):
    """Guards the construction above against being vacuous.

    The row must land on the unit it names, and the dropped variant must leave
    exactly two assigned markers there -- the fragile threshold. If a future
    ``demo_mini`` has four markers on c3, or the gene name becomes a panel member,
    this says so instead of passing on a study that no longer poses the question.
    """
    from panoscope.data.study import Study
    from panoscope.pipeline import records

    study_id = _study_with_an_out_of_frame_number(tmp_path, monkeypatch, drop=True)
    store = Study.open(study_id)
    with cfg.use_study(store):
        assert not data.panel_contains(OUT_OF_FRAME_GENE), (
            f"{OUT_OF_FRAME_GENE} is on this panel; it cannot pose the question"
        )
        rows = data.get_cluster_markers("c3")
        assert OUT_OF_FRAME_GENE in set(rows["gene"].astype(str)), (
            "the method table hands the row to c3, or the guard is never asked"
        )
        assert len(rows) == 3, "two on-panel rows plus the out-of-frame one"
        built = records.build_all(store)
    assert built["c3"].verdict.fragile, "core must take the fragile branch here"


# --------------------------------------------------------------------------- #
# The §9.10 replicate read in the v1 view (concordance-travels)
#
# ``interp/verdicts.csv`` mentioned replicate concordance on no unit of any study,
# so a divergent unit and a clean one exported the same eleven cells. The read goes
# into ``notes`` -- the one v1 field that can hold a sentence -- and not into a
# twelfth column, because the eleven are frozen (§10) and because a divergence is a
# flag: §9.10 does not claim whether it is batch effect, section quality or
# biology, and a cell beside ``confidence_score`` is read as a measurement.
# --------------------------------------------------------------------------- #
_REP_WHAT = "subjects_comparable"


def _rep(k: int, n: int = 12, floor: int = 6) -> Coverage:
    return Coverage(_REP_WHAT, k, n, floor)


def test_a_replicate_read_that_was_never_askable_leaves_the_v1_row_alone():
    """``replicate_coverage is None`` is the state every demo unit is in.

    The nine committed demo files are §10's frozen reference, so a sentence here
    would move the bytes the equivalence gate reads as the definition of
    "unchanged" -- and the reason is not lost: it is in the v2 record, in
    ``panoscope explain``, on the markers page and now in the exported document.
    ``V1_NO_REPLICATE_SHAPE`` is where that omission is written down.
    """
    before = {"cluster": "c1", "notes": "Confidence High."}
    after = compat.with_v1_replicate_read(
        before, ("replicate concordance (ARCHITECTURE 9.10) does not apply: the "
                 "study holds one sample (sample1)",), None).value

    assert after == before and after is not before      # a copy, never a mutation


def test_the_v1_k_of_n_separates_one_of_twelve_from_eleven_of_twelve():
    """The denominator travels, or the caveat is not a caveat.

    A unit comparable on 1 of 12 subjects and one comparable on 11 of 12 both
    report "not divergent". If the v1 row renders them identically it has thrown
    away the only thing that separates a profile with a second reading behind it
    from one without -- the same failure ``Concordance.coverage`` was added one
    level down to prevent.
    """
    thin = compat.with_v1_replicate_read(
        {"notes": "n."}, ("replicate concordance compared 1 of 12 subject(s) "
                          "across sampleA, sampleB",), _rep(1)).value["notes"]
    thick = compat.with_v1_replicate_read(
        {"notes": "n."}, ("replicate concordance compared 11 of 12 subject(s) "
                          "across sampleA, sampleB",), _rep(11)).value["notes"]

    assert thin != thick
    assert "compared 1 of 12" in thin and "compared 1 of 12" not in thick
    assert "compared 11 of 12" in thick


def test_the_three_outcomes_stay_three_in_the_v1_notes():
    """Divergent, not-comparable-and-cannot-say-why, and never-askable.

    The first two are carried in the record's own words; the third is omitted and
    recorded (:data:`compat.V1_NO_REPLICATE_SHAPE`). What must not happen is any
    two of them reading alike.
    """
    diverged = compat.with_v1_replicate_read(
        {"notes": "n."}, ("SYNG03 read differently across sampleA, sampleB",),
        _rep(9)).value["notes"]
    no_why = compat.with_v1_replicate_read(
        {"notes": "n."}, ("replicate concordance (ARCHITECTURE 9.10) could not be "
                          "assessed for this unit: sampleB reported no row under it",),
        _rep(0)).value["notes"]
    never = compat.with_v1_replicate_read({"notes": "n."}, ("anything",), None
                                          ).value["notes"]

    assert len({diverged, no_why, never}) == 3
    assert "read differently" in diverged
    assert "could not be assessed for this unit" in no_why
    assert never == "n."


def test_the_v1_row_carries_the_divergence_on_the_production_path(monkeypatch,
                                                                   tmp_path):
    """End to end through ``pipeline/stages/verdicts.py``, into a tmp tree.

    The unit tests above pin the composition; this pins that the stage calls it,
    that the sentence survives into ``interp/verdicts.csv``'s ``notes`` cell, and
    -- the half that matters most -- that carrying it moved neither the band, the
    score nor ``verify`` on any unit.

    **Both redirections are needed and they are different mechanisms.** The READ
    side resolves through ``layout.resolve_data_root``, an ``lru_cache`` keyed on
    the environment as it stood at the first call, so the env must be set and the
    cache cleared AFTER it. The WRITE side does not read the environment at all:
    ``pipeline/paths.DEFAULT_DATASETS_ROOT`` is ``cfg.DATA_DIR_PATH / "studies"``
    evaluated at import time, which is the frozen-root case ``conftest``'s tree
    guard warns about -- so every write goes through ``run_verdicts``' explicit
    ``root``. Setting only the env here wrote a full build into the committed
    fixture study and ``git status`` showed nothing, because the tree is
    gitignored under ``interp/``.
    """
    import csv
    import io
    import shutil

    from panoscope.data import layout
    from panoscope.data.study import Study
    from panoscope.pipeline import paths
    from panoscope.pipeline.stages.verdicts import run_verdicts

    fixture = "synthetic_multisample_fixture"
    source = Study.open(fixture)
    if not source.exists:
        pytest.skip(f"study {fixture!r} is not on disk")

    root = tmp_path / "data"
    studies = root / "studies"
    shutil.copytree(source.dir, studies / fixture)
    monkeypatch.setenv(layout.ENV_DATA_ROOT, str(root))
    layout.clear_cache()
    try:
        with cfg.use_study(Study.open(fixture, data_root=root)):
            run_verdicts(fixture, studies)
            rows = {r["cluster"]: r for r in csv.DictReader(io.StringIO(
                paths.verdicts_csv(fixture, studies).read_text(encoding="utf-8")))}
    finally:
        layout.clear_cache()

    # c1 and c2 diverge on SYNG03; c3 does not. All three were compared on 9 of 12.
    assert "SYNG03 read differently across sampleA, sampleB" in rows["c1"]["notes"]
    assert "read differently" not in rows["c3"]["notes"]
    for row in rows.values():
        assert "replicate concordance compared 9 of 12 subject(s)" in row["notes"]

    # The band, the score and verify are untouched: a replicate read is a flag, and
    # agreement never raises confidence either (§9.10).
    assert [rows[c]["confidence"] for c in ("c1", "c2", "c3")] == ["High", "Low", "Low"]
    assert [rows[c]["confidence_score"] for c in ("c1", "c2", "c3")] == \
        ["0.85", "0.3", "0.3"]
    assert {row["verify"] for row in rows.values()} == {"TRUE"}
