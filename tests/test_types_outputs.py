"""Invariants for the output contract.

As in ``test_types_measured.py`` and ``test_types_evidence.py``, the tests are the
deliverable: each one pins a property something downstream is entitled to assume and
names what breaks when it stops holding.

Two of them are the point of the module:

* the tier/measured correspondence -- the reason the lattice has four states and the
  tier enum has four members rather than the three ``IMPLEMENTATION.md`` §2.4 lists;
* the ``Finding`` field-name guard -- which fails when a band is *declared*, not when
  one is finally populated.
"""

from __future__ import annotations

import dataclasses
import itertools

import pytest

from panoscope.types.evidence import Direction, EvidenceItem
from panoscope.types.measured import (
    Coverage, Frame, Measured, MeasuredState, MeasuredWorld, resolve,
)
from panoscope.types.outputs import (
    Assignment, AssignmentIsNotAFinding, Finding, FindingItem, MisfiledEvidence,
    MisTieredItem, TaskKind, Tier, permitted_tiers,
)
from panoscope.types.subject import EntityKind, cell_type, gene, subject

PANEL = Frame(EntityKind.GENE, "panel:test",
              frozenset({"ERBB2", "LUM", "ACTA2", "CXCL12"}))
ANNOT = Frame(EntityKind.CELL_TYPE, "annotation:test", frozenset({"Tumor", "Stromal"}))
WORLD = MeasuredWorld((PANEL, ANNOT))
REPORTED = frozenset({"ERBB2", "LUM", "CXCL12", "Tumor", "Stromal"})

ERBB2 = subject("gene", g=gene("ERBB2"))    # in frame, reported     -> OBSERVED
ACTA2 = subject("gene", g=gene("ACTA2"))    # in frame, not reported -> TESTED_ABSENT
COL1A1 = subject("gene", g=gene("COL1A1"))  # off panel              -> NEVER_MEASURED
# A composite: everything on-panel except the receptor, which is what blocks it.
LR = subject("interaction", ligand=gene("CXCL12"), receptor=gene("CXCR4"),
             source=cell_type("Tumor"), target=cell_type("Stromal"))

OBSERVED = resolve(ERBB2, WORLD, REPORTED)
ABSENT = resolve(ACTA2, WORLD, REPORTED)
NEVER = resolve(COL1A1, WORLD, REPORTED)
BLOCKED = resolve(LR, WORLD, REPORTED)
UNDER = Measured(MeasuredState.UNDERPOWERED, PANEL.name,
                 coverage=(Coverage("cells_in_unit", 2, 200, 30),))


def _ev(subj, stat, measured, **kw) -> EvidenceItem:
    return EvidenceItem(subject=subj, statistic=stat, statistic_name="glm_coef",
                        measured=measured, **kw)


SUPPORTING = _ev(ERBB2, 4.2, OBSERVED, direction=Direction.UP)
TESTED_ABSENT_EV = _ev(ACTA2, None, ABSENT)
NEVER_EV = _ev(COL1A1, None, NEVER)
BLOCKED_EV = _ev(LR, None, BLOCKED)


# --------------------------------------------------------------------------- #
# The tier/measured correspondence: why there are four tiers, not three
# --------------------------------------------------------------------------- #
def test_a_tested_absence_is_not_supported_rather_than_untestable():
    """The productive half of the panel-absence rule, pinned.

    ACTA2 is on the panel, was tested, and was not reported. That is an answer: it
    argues AGAINST any rival whose case rests on ACTA2. Filed as ``UNTESTABLE`` it
    reads to the biologist as "we could not look", the discriminator quietly stops
    discriminating, and nothing goes red -- which is the exact failure the four-state
    lattice exists to prevent, reproduced one layer up.
    """
    assert ABSENT.absence_is_evidence is True
    assert permitted_tiers(ABSENT) == frozenset({Tier.NOT_SUPPORTED})
    item = FindingItem(evidence=TESTED_ABSENT_EV, tier=Tier.NOT_SUPPORTED, verify=False)
    assert item.tier is Tier.NOT_SUPPORTED
    with pytest.raises(MisTieredItem):
        FindingItem(evidence=TESTED_ABSENT_EV, tier=Tier.UNTESTABLE, verify=False)


@pytest.mark.parametrize(
    "measured", [NEVER, UNDER, BLOCKED],
    ids=["never_measured", "underpowered", "composite_blocked"],
)
def test_an_unaskable_question_is_untestable_and_never_argues_against(measured):
    """The other half: out-of-frame absence is NOT evidence against anything.

    ``NOT_SUPPORTED`` would turn "we never looked" into an argument -- the most
    expensive mistake available on a panel designed for a different question, where
    kidney-resident types have almost no canonical markers on-panel and every one of
    them would read as actively refuted.
    """
    assert measured.absence_is_evidence is False
    assert permitted_tiers(measured) == frozenset({Tier.UNTESTABLE})
    with pytest.raises(MisTieredItem):
        FindingItem(evidence=_ev(COL1A1, None, measured),
                    tier=Tier.NOT_SUPPORTED, verify=False)


@pytest.mark.req("CLAUDE-MS-03")
def test_the_tier_tracks_the_measured_lattice_for_every_state_and_tier():
    """The drift guard: the correspondence has ONE source, ``permitted_tiers``.

    Every (state, tier) pair is constructed. An item is buildable exactly when the
    tier is permitted by the measurement underneath it -- so a future edit that
    collapses ``NOT_SUPPORTED`` into ``UNTESTABLE``, or lets ``SUPPORTED`` sit over
    an unmeasured subject, fails here with the pair named rather than shipping a
    tier that claims more than the apparatus could see.

    Note what this pins about the shape of the system: the tier is a function of the
    measurement everywhere except the ``SUPPORTED``/``SUGGESTIVE`` split, which is a
    question about the size of a number that exists. No other layer may add tiers,
    only subtract.
    """
    seen: set[Tier] = set()
    for state, tier in itertools.product(MeasuredState, Tier):
        m = Measured(state, PANEL.name)
        ev = _ev(ERBB2, 1.0 if m.carries_number else None, m)
        allowed = tier in permitted_tiers(m)
        seen |= permitted_tiers(m)
        try:
            FindingItem(evidence=ev, tier=tier, verify=False)
            built = True
        except MisTieredItem:
            built = False
        assert built is allowed, f"{state.value} + {tier.value}"
    assert seen == set(Tier), (
        "a tier no measurement state can produce is dead vocabulary: "
        f"{sorted(t.value for t in set(Tier) - seen)}"
    )


@pytest.mark.req("CLAUDE-MS-04")
def test_permitted_tiers_is_derived_from_the_measured_properties():
    """Not a second list of states. ``evidence.py`` learned this the hard way.

    §2.2 of the spec re-listed ``{TESTED_ABSENT, NEVER_MEASURED}`` by hand and got
    it wrong by omitting ``UNDERPOWERED``. Here the answer is read off
    ``carries_number``/``absence_is_evidence``, so a fifth state, or a widening of
    either property, moves the tier rule with it automatically.
    """
    for state in MeasuredState:
        m = Measured(state, PANEL.name)
        if m.carries_number:
            assert permitted_tiers(m) == frozenset({Tier.SUPPORTED, Tier.SUGGESTIVE})
        elif m.absence_is_evidence:
            assert permitted_tiers(m) == frozenset({Tier.NOT_SUPPORTED})
        else:
            assert permitted_tiers(m) == frozenset({Tier.UNTESTABLE})


def test_the_mistier_error_names_the_subject_the_tier_and_the_state():
    """These raise from inside a loop over hundreds of items; a bare ValueError
    does not say which one broke, and the traceback is all a reader gets."""
    with pytest.raises(MisTieredItem, match="COL1A1"):
        FindingItem(evidence=NEVER_EV, tier=Tier.SUPPORTED, verify=False)
    with pytest.raises(MisTieredItem, match="never_measured"):
        FindingItem(evidence=NEVER_EV, tier=Tier.SUPPORTED, verify=False)
    with pytest.raises(MisTieredItem, match="untestable"):
        FindingItem(evidence=NEVER_EV, tier=Tier.SUPPORTED, verify=False)


# --------------------------------------------------------------------------- #
# FindingItem wraps; it does not copy
# --------------------------------------------------------------------------- #
def test_finding_item_delegates_rather_than_copying_the_number():
    """One home per number.

    ``subject`` and ``direction`` are properties over ``self.evidence``. If anyone
    promotes ``statistic`` to a field here, the statistic/measured IFF has to be
    enforced in two places -- and a rule enforced in two places is a rule enforced
    in one. The field scan is the guard; the delegation is the behaviour it protects.
    """
    item = FindingItem(evidence=SUPPORTING, tier=Tier.SUPPORTED, verify=False)
    assert item.subject is SUPPORTING.subject
    assert item.direction is Direction.UP
    assert item.evidence.statistic == 4.2
    names = {f.name for f in dataclasses.fields(FindingItem)}
    copied = names & {"statistic", "statistic_name", "subject", "direction",
                      "measured", "corroborating", "flags"}
    assert not copied, f"FindingItem copies evidence fields: {sorted(copied)}"


def test_verify_is_stored_not_computed():
    """``verify`` comes from the deterministic rubric and is never the model's
    choice. It is a plain field so nothing in this layer can derive it; a property
    computed here would be a second rubric that no calibration covers."""
    assert "verify" in {f.name for f in dataclasses.fields(FindingItem)}
    assert FindingItem(evidence=NEVER_EV, tier=Tier.UNTESTABLE, verify=True).verify is True


# --------------------------------------------------------------------------- #
# Finding: no band, ever -- guarded on the FIELD, not on the value
# --------------------------------------------------------------------------- #
def test_finding_has_no_band():
    """The no-band rule as a field-name guard, which is the whole point.

    CHARACTERISATION and COMPARISON have no label space, so a unit-level band is
    *undefined* -- there is no alternative for the remaining probability mass to sit
    on. A behavioural test ("nothing populates a band") only fires once someone has
    built the plumbing and filled it in; this fires on the line that declares the
    field, when deleting it costs nothing.

    ``score`` and ``confidence_score`` are in the list because that is what the
    field gets called the second time someone adds it.
    """
    banned = {"band", "confidence", "confidence_score", "score"}
    declared = {f.name for f in dataclasses.fields(Finding)}
    assert not (declared & banned), (
        "Finding declares a band-like field "
        f"{sorted(declared & banned)}: characterisation and comparison have no "
        "label space, so a unit-level band is undefined, not merely unhelpful"
    )


def test_assignment_does_have_a_band():
    """The other side of the asymmetry, which is the entire boundary between the
    two types. ASSIGNMENT has a label space, so probability mass has somewhere to
    sit and the band is meaningful. If this stops holding, the no-band guard above
    has been applied to the one place it does not belong."""
    a = Assignment(unit="c1", unit_kind="cluster", label="Tumor", label_id="CL:0001064",
                   band="High", verify=False, drivers=(SUPPORTING,), unmeasured=(),
                   rationale="ERBB2 glm_coef 4.2")
    assert a.band == "High"


def test_a_finding_cannot_be_an_assignment():
    """Routing an assignment through the no-band shape does not remove the band from
    the question, only from the answer. The unit would read as interpreted while the
    one number the biologist acts on had been silently dropped."""
    with pytest.raises(AssignmentIsNotAFinding):
        Finding(unit="c1", unit_kind="cluster", task_kind=TaskKind.ASSIGNMENT,
                items=(), summary="")


@pytest.mark.parametrize("kind", [TaskKind.CHARACTERISATION, TaskKind.COMPARISON])
def test_the_other_two_task_kinds_are_findings(kind):
    """Both no-label-space kinds must be constructible, or the guard above has
    over-reached and left a task kind with nowhere to live."""
    assert Finding(unit="c1", unit_kind="cluster", task_kind=kind,
                   items=(), summary="").task_kind is kind


def test_untestable_count_excludes_tested_absences():
    """The header for a characterisation is per-item counts, so the counts carry the
    meaning a band would have carried. Counting ``NOT_SUPPORTED`` as untestable here
    re-collapses the distinction one layer further out: "3 untestable" would mix
    "we could not look" with "we looked and it argues against", and the biologist
    reading the header cannot tell which."""
    f = Finding(
        unit="c1", unit_kind="cluster", task_kind=TaskKind.CHARACTERISATION,
        items=(
            FindingItem(evidence=SUPPORTING, tier=Tier.SUPPORTED, verify=False),
            FindingItem(evidence=TESTED_ABSENT_EV, tier=Tier.NOT_SUPPORTED, verify=False),
            FindingItem(evidence=NEVER_EV, tier=Tier.UNTESTABLE, verify=True),
        ),
        summary="1 supported, 1 argues against, 1 untestable",
    )
    assert f.untestable_count == 1


# --------------------------------------------------------------------------- #
# Assignment: the unmeasured caveat has to name the blocker
# --------------------------------------------------------------------------- #
def test_unmeasured_names_which_constituent_blocked_the_call():
    """"CXCR4 was never measured", not "something was unmeasured".

    The spec's bare-name tuple throws away the frame and the entity that excluded
    it, so the interface can only print a count. Carrying the ``EvidenceItem`` keeps
    ``Measured.blocked_by`` reachable, which is what makes the caveat actionable --
    the biologist can see the interaction failed on the receptor and that the ligand
    was fine.
    """
    a = Assignment(unit="c1", unit_kind="cluster", label="Tumor", label_id=None,
                   band="Medium", verify=True, drivers=(SUPPORTING,),
                   unmeasured=(BLOCKED_EV,), rationale="receptor off panel")
    blocked = a.unmeasured[0].measured.blocked_by
    assert [e.id for e in blocked] == ["CXCR4"]
    assert a.unmeasured[0].measured.frame == PANEL.name
    assert a.unmeasured[0].measured.absence_is_evidence is False


def test_drivers_and_unmeasured_cannot_be_swapped():
    """The two tuples partition the evidence by whether a number exists.

    A driver with no statistic renders as a blank cell in the drivers table -- the
    blank row the interface rule forbids, because a never-measured item must be
    visible as never-measured and not as an empty value. An "unmeasured" item that
    carries a number is a caveat about something that was, in fact, measured.
    """
    with pytest.raises(MisfiledEvidence, match="COL1A1"):
        Assignment(unit="c1", unit_kind="cluster", label="Tumor", label_id=None,
                   band="Low", verify=True, drivers=(NEVER_EV,), unmeasured=(),
                   rationale="")
    with pytest.raises(MisfiledEvidence, match="ERBB2"):
        Assignment(unit="c1", unit_kind="cluster", label="Tumor", label_id=None,
                   band="Low", verify=True, drivers=(), unmeasured=(SUPPORTING,),
                   rationale="")


# --------------------------------------------------------------------------- #
# Contract hygiene
# --------------------------------------------------------------------------- #
def test_output_contracts_are_frozen_and_hashable():
    """Frozen is what makes these safe as cache keys and dict members, and it is
    what stops a renderer editing a tier on its way to the screen."""
    item = FindingItem(evidence=SUPPORTING, tier=Tier.SUPPORTED, verify=False,
                       because=("glm_coef 4.2",))
    for obj in (
        item,
        Finding(unit="c1", unit_kind="cluster", task_kind=TaskKind.COMPARISON,
                items=(item,), summary="s"),
        Assignment(unit="c1", unit_kind="cluster", label="Tumor", label_id=None,
                   band="High", verify=False, drivers=(SUPPORTING,), unmeasured=(),
                   rationale="r"),
    ):
        hash(obj)
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, next(iter(type(obj).__slots__)), "mutated")


def test_no_mutable_fields_in_output_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes, so the golden
    hash would move on the order a reader happened to fill a field. The same scan as
    ``test_types_measured.py::test_no_mutable_fields_in_contracts``, extended to
    this module."""
    from panoscope.types import outputs as O

    bad = []
    for name in dir(O):
        obj = getattr(O, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            t = str(f.type)
            if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)
