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
                        yield coef, pearson, gc, gg, fragile


@pytest.mark.parametrize("coef,pearson,gc,gg,fragile", list(_grid()))
def test_every_rule_id_renders_the_sentence_agent_wrote(coef, pearson, gc, gg, fragile):
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
    _band, _basis, expected = A._compute_band(me, fragile=fragile)

    driver = _item("ERBB2", coef, pearson, tuple(_FLAG_FOR[c] for c in caveats))
    filler = () if fragile else tuple(_item(f"G{i}", 1.0, 0.5) for i in range(3))
    v = C.assess((driver, *filler), frozenset({"ERBB2"})).value
    assert tuple(compat._demotion_sentence(r, v.driver) for r in v.demotions) == expected


def test_the_caveat_table_is_the_inverse_of_the_agent_rule():
    """``compat.CAVEAT_FOR`` must render in ``_caveats_for``'s append order, or the
    ``caveats`` list comes out permuted and ``ui/summary.py`` (which reads
    ``caveats[0]``) shows a different leading caveat."""
    for pearson, gc, gg in ((0.1, 0.9, 0.99), (0.5, 0.9, 0.5), (0.9, 0.0, 0.5),
                            (0.2, 0.0, 0.99), (0.9, 0.0, 0.0)):
        expected = A._caveats_for(pearson, gc, gg)
        flags = tuple(_FLAG_FOR[c] for c in expected)
        assert tuple(p for f, p in compat.CAVEAT_FOR if f in flags) == expected


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
