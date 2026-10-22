"""Invariants for the evidence contract.

The deliverable is these tests, not the dataclasses: "anything called an invariant
has a test that fails loudly". Each test pins a property something downstream is
entitled to assume, and names what breaks if it stops holding.

The headline one is the IFF between ``statistic`` and ``Measured``. It is the
type-level form of the project's worst failure -- a number quoted for a thing the
measurement never looked at -- and it is checked in both directions.
"""

from __future__ import annotations

import dataclasses

import pytest

from panoscope.agent import verdict as V
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.evidence import (
    FLAGS, Direction, EvidenceItem, NormalisedEvidence, UngroundedStatistic,
    UnregisteredFlag,
)
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld, resolve,
)
from panoscope.types.subject import EntityKind, cell_type, gene, subject
from panoscope.ui import evidence_table as et

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM", "ACTA2"}))
ANNOT = Frame(EntityKind.CELL_TYPE, "annotation:test", frozenset({"Tumor"}))
WORLD = MeasuredWorld((PANEL, ANNOT))
REPORTED = frozenset({"ERBB2", "LUM"})

ERBB2 = subject("gene", g=gene("ERBB2"))        # in frame, reported     -> OBSERVED
ACTA2 = subject("gene", g=gene("ACTA2"))        # in frame, not reported -> TESTED_ABSENT
COL1A1 = subject("gene", g=gene("COL1A1"))      # off panel              -> NEVER_MEASURED

OBSERVED = resolve(ERBB2, WORLD, REPORTED)
ABSENT = resolve(ACTA2, WORLD, REPORTED)
NEVER = resolve(COL1A1, WORLD, REPORTED)
UNDER = Measured(MeasuredState.UNDERPOWERED, PANEL.name,
                 coverage=(Coverage("cells_in_unit", 2, 200, 30),))
CONDITIONS = Conditions(
    assay=Condition("xenium", Provenance.READ),
    modality=Condition("spatial_transcriptomics", Provenance.READ),
    tissue=Condition("UBERON:0000310", Provenance.STATED),
    method=Condition("jazzpanda", Provenance.STATED),
    segmentation=UNKNOWN, fixation=UNKNOWN,
    frames=(PANEL.ref(), ANNOT.ref()),
)


def _item(subj, stat, measured, **kw) -> EvidenceItem:
    return EvidenceItem(subject=subj, statistic=stat, statistic_name="glm_coef",
                        measured=measured, **kw)


# --------------------------------------------------------------------------- #
# The IFF: a statistic exists iff the subject was observed
# --------------------------------------------------------------------------- #
def test_an_observed_subject_carries_its_number():
    """The permitted shape, so the rest of the file is about what it excludes."""
    ev = _item(ERBB2, 4.2, OBSERVED)
    assert ev.statistic == 4.2
    assert ev.measured.state is MeasuredState.OBSERVED


@pytest.mark.parametrize(
    "measured",
    [NEVER, ABSENT, UNDER],
    ids=["never_measured", "tested_absent", "underpowered"],
)
def test_a_number_for_an_unmeasured_subject_is_unrepresentable(measured):
    """The project's worst failure, made structurally impossible.

    Every marker, number and citation must trace to method output. An item that
    pairs a statistic with a state where the measurement could not have produced
    one is a fabricated number wearing a provenance field, and nothing downstream
    can detect it once it exists. So it must not be constructible.

    ``UNDERPOWERED`` is in this list deliberately: IMPLEMENTATION.md §2.2 lists
    only ``{TESTED_ABSENT, NEVER_MEASURED}``, which is a spec bug. "In frame, below
    the floor" is exactly the case where a number looks legitimate -- the gene IS
    on the panel -- and is not supported.
    """
    with pytest.raises(UngroundedStatistic):
        _item(COL1A1, 4.2, measured)


def test_an_observed_subject_without_its_number_is_unrepresentable():
    """The other direction, which is a real bug and not merely symmetry.

    A reader that drops the statistic turns a measured result into one that reads
    as untestable. The biologist cannot tell that refusal from a true one, so the
    apparatus under-reports in silence -- and refusal rate is reported next to
    accuracy precisely because a gate that fails closed can look safe while
    refusing.
    """
    with pytest.raises(UngroundedStatistic):
        _item(ERBB2, None, OBSERVED)


def test_the_iff_tracks_carries_number_for_every_state():
    """The drift guard: the rule has ONE source, ``Measured.carries_number``.

    If a fifth state is added, or ``carries_number`` is widened, this test moves
    with it automatically. A hand-maintained second list of states in evidence.py
    is how the spec bug above got written down in the first place -- the copy
    nobody updates is always the one that governs.
    """
    for state in MeasuredState:
        m = Measured(state, PANEL.name)
        permitted = m.carries_number
        try:
            _item(ERBB2, 1.0, m)
            with_number = True
        except UngroundedStatistic:
            with_number = False
        assert with_number is permitted, f"{state.value} with a number"
        try:
            _item(ERBB2, None, m)
            without_number = True
        except UngroundedStatistic:
            without_number = False
        assert without_number is (not permitted), f"{state.value} without a number"


def test_the_error_names_the_subject_and_the_state():
    """A grounding violation must be diagnosable from the traceback alone.

    A bare ValueError says nothing about WHICH item broke the floor, and these
    raise from inside a reader loop over hundreds of rows.
    """
    with pytest.raises(UngroundedStatistic, match="COL1A1"):
        _item(COL1A1, 4.2, NEVER)
    with pytest.raises(UngroundedStatistic, match="never_measured"):
        _item(COL1A1, 4.2, NEVER)


def test_an_unmeasured_item_still_carries_its_reason():
    """Never-measured is a reported state, not a dropped row.

    "Measurement status is visible, not implied": the UI has to say WHICH entity
    was off-panel, so the item must exist without a number rather than not exist.
    """
    ev = _item(COL1A1, None, NEVER)
    assert ev.measured.absence_is_evidence is False
    assert ev.measured.frame == PANEL.name


# --------------------------------------------------------------------------- #
# Flags: registered ids, never prose
# --------------------------------------------------------------------------- #
def test_an_unregistered_flag_raises():
    """Fail loudly. An unregistered id means a caveat no copy table can render and
    no test can match -- which is how prose crept into this layer last time."""
    with pytest.raises(UnregisteredFlag):
        _item(ERBB2, 1.0, OBSERVED, flags=("specificity:invented_just_now",))


def test_prose_is_not_a_flag():
    """The exact bug string, pinned. ``ui/evidence_table.py`` decided whether to
    draw a badge by matching this phrase, so a copy edit in the analysis layer
    silently stopped the badge rendering while its test stayed green."""
    with pytest.raises(UnregisteredFlag):
        _item(ERBB2, 1.0, OBSERVED, flags=("localizes better with another cluster",))


def test_flag_registry_keys_match_their_ids():
    """A mismatched key is a flag that can never be looked up by the id it reports."""
    for fid, flag in FLAGS.items():
        assert flag.id == fid, f"FLAGS key {fid!r} != Flag.id {flag.id!r}"


def test_every_caveat_the_analysis_layer_emits_has_registered_copy():
    """The registry is a transcription of today's behaviour, not a wish list.

    These three values make all three branches of ``_caveats_for`` fire at once.
    If someone adds a fourth caveat, or edits the English of an existing one, it
    fails here instead of silently dropping a badge in the UI -- which is exactly
    the failure the id/copy split exists to prevent.
    """
    emitted = set(V._caveats_for(pearson=-1.0, max_gc_corr=1.0, max_gg_corr=1.0))
    assert len(emitted) == 3, "expected all three specificity caveats to fire"
    assert emitted == set(et.FLAG_COPY.values())
    assert set(et.FLAG_COPY) == set(FLAGS), "a flag with no copy cannot be rendered"


def test_the_ui_badge_reads_its_english_from_the_copy_table():
    """The wiring, end to end: the badge no longer owns a literal of its own.

    ``_caveat_badge`` still matches prose because ``verdict.py`` still emits prose;
    what changed is that the English has ONE home. A copy edit now moves the badge
    and the table together.
    """
    assert et._SPECIFICITY_CAVEAT == et.FLAG_COPY[et._SPECIFICITY_FLAG]
    assert et._SPECIFICITY_FLAG in FLAGS


# --------------------------------------------------------------------------- #
# Determinism and contract hygiene
# --------------------------------------------------------------------------- #
def test_corroborating_is_sorted_at_construction():
    """Byte-determinism, which the golden-output test is the repo's claim on.

    The spec sketches a dict. A dict is unhashable on a frozen dataclass AND its
    insertion order flows through ``asdict`` into the serialised record, so two
    readers that added the same corroborators in different order would produce two
    different golden hashes for identical evidence.
    """
    a = _item(ERBB2, 1.0, OBSERVED, corroborating=(("pearson", 0.7), ("moran_i", 0.3)))
    b = _item(ERBB2, 1.0, OBSERVED, corroborating=(("moran_i", 0.3), ("pearson", 0.7)))
    assert a.corroborating == (("moran_i", 0.3), ("pearson", 0.7))
    assert a == b
    assert dataclasses.asdict(a) == dataclasses.asdict(b)


def test_evidence_contracts_are_frozen_and_hashable():
    """Frozen is what makes these safe as cache keys and dict members."""
    ev = _item(ERBB2, 1.0, OBSERVED, direction=Direction.UP,
               flags=("specificity:low",), corroborating=(("pearson", 0.1),))
    hash(ev)
    with pytest.raises((AttributeError, TypeError)):
        ev.statistic = 9.9


def test_normalised_evidence_holds_the_unit_level_status():
    """"Too few replicates" is a property of the unit, not of any marker in it.

    Pushed down onto the items it is duplicated once per row, and two rows can then
    disagree about the same design.
    """
    ne = NormalisedEvidence(
        unit="c1", unit_kind="cluster",
        unit_subject=subject("cell_type", ct=cell_type("Tumor")),
        unit_measured=UNDER, module="jazzpanda",
        items=(_item(ERBB2, 4.2, OBSERVED), _item(COL1A1, None, NEVER)),
        reported=(PANEL, ANNOT),
        conditions=CONDITIONS,
    )
    assert ne.unit_measured.state is MeasuredState.UNDERPOWERED
    assert [i.statistic for i in ne.items] == [4.2, None]


def test_no_mutable_fields_in_evidence_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. The same scan
    as ``test_types_measured.py::test_no_mutable_fields_in_contracts``, extended to
    this module -- notably ``corroborating``, which the spec sketches as a dict."""
    from panoscope.types import evidence as E

    bad = []
    for name in dir(E):
        obj = getattr(E, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            t = str(f.type)
            if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)
