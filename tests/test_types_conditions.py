"""Invariants for the study-conditions contract.

Same deliverable rule as ``test_types_measured.py``: the dataclass is not the
contract, these are. Each test pins something a downstream applicability check is
entitled to assume, and says what breaks when it does not hold. The theme is narrow
and load-bearing -- ``unknown`` is first-class *and* it is not allowed everywhere
(ARCHITECTURE §6.3), and the difference between those two halves must fail loudly.
"""

from __future__ import annotations

import dataclasses

import pytest

from panoscope.types.conditions import (
    BLOCKS_THE_STUDY, DECLARED_UNKNOWN_OK, UNKNOWN, Blocked, Condition, Conditions,
    Provenance,
)
from panoscope.types.measured import Frame
from panoscope.types.subject import EntityKind

PANEL = Frame(EntityKind.GENE, "panel:test", frozenset({"ERBB2", "LUM"})).ref()


def conditions(**over) -> Conditions:
    """A complete, valid study. Tests vary one field; everything else stays honest."""
    base = dict(
        assay=Condition("Xenium", Provenance.READ),
        modality=Condition("spatial_transcriptomics", Provenance.READ),
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=Condition("jazzpanda", Provenance.READ),
        segmentation=Condition("nucleus_expansion_15um", Provenance.STATED),
        fixation=Condition("FFPE", Provenance.STATED),
        frames=(PANEL,),
        disease_state=Condition("invasive ductal carcinoma", Provenance.STATED),
        disease_subtype=Condition("HER2+", Provenance.STATED),
        disease_treatment=Condition("naive", Provenance.STATED),
        disease_stage=Condition("II", Provenance.STATED),
    )
    return Conditions(**{**base, **over})


# --------------------------------------------------------------------------- #
# Condition: a value never travels without its provenance
# --------------------------------------------------------------------------- #
def test_a_value_exists_iff_it_is_known():
    """The two incoherent states are unrepresentable, not merely discouraged.

    A value under UNKNOWN is a guess wearing a disclaimer: the check drops the field
    but the guess still reaches the serialised record and the UI. A None under STATED
    is the mirror -- it reads as *known* to every applicability check, which then
    compares against nothing and passes. Both are how an inference becomes a fact.
    """
    with pytest.raises(ValueError, match="iff it is known"):
        Condition("breast", Provenance.UNKNOWN)
    with pytest.raises(ValueError, match="iff it is known"):
        Condition(None, Provenance.STATED)


def test_the_four_provenances_are_distinguishable():
    """`inferred` must not be storable as `stated`: §6.2's entire purpose is that a
    model's proposal stays visibly a model's proposal even after confirmation."""
    assert len({p.value for p in Provenance}) == 4
    assert Condition("breast", Provenance.INFERRED).known is True
    assert UNKNOWN.known is False


# --------------------------------------------------------------------------- #
# The §6.3 line: what blocks, and what propagates
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", BLOCKS_THE_STUDY)
def test_an_unknown_blocking_field_raises_and_names_itself(name):
    """These cannot be carried as caveats, because nothing downstream can compensate.

    An unknown tissue scopes the literature corpus to no organ in particular, and the
    result still renders. The error names the field because §6.4's checklist is built
    from the names -- "study blocked" with no field is not something a biologist can
    act on.
    """
    with pytest.raises(Blocked, match=name):
        conditions(**{name: UNKNOWN})


def test_no_frame_blocks_the_study():
    """Structural: the measured set IS the panel file.

    With no FrameRef there is no capability frame, so ``resolve()`` returns
    NEVER_MEASURED for everything and the panel-absence rule stops discriminating --
    the study reads as maximally cautious while actually being uninformed.
    """
    with pytest.raises(Blocked, match="measured set"):
        conditions(frames=())


@pytest.mark.parametrize("name", DECLARED_UNKNOWN_OK)
def test_a_declared_unknown_flows_through_untouched(name):
    """The other half, and the reason the line exists at all.

    A biologist reading a collaborator's section may genuinely not know the
    segmentation version. Blocking on it teaches intake to extract an answer, which
    is the guessing this package exists to stop. So it builds, and the unknown is
    still there afterwards -- not defaulted, not dropped.
    """
    c = conditions(**{name: UNKNOWN})
    assert getattr(c, name) is UNKNOWN
    assert name in c.declared_unknown


def test_declared_unknown_names_every_missing_field_in_a_stable_order():
    """A check reading an unknown field returns *cannot be settled* and names what is
    missing (§6.3). Names, not a boolean: a flag tells the biologist nothing he can
    fix. Order is declaration order so the rendered caveat is byte-stable."""
    assert conditions().declared_unknown == ()
    both = conditions(segmentation=UNKNOWN, fixation=UNKNOWN, method=UNKNOWN)
    assert both.declared_unknown == ("method", "segmentation", "fixation")


def test_every_condition_field_is_classified_exactly_once():
    """A seventh condition field must not be able to arrive unclassified.

    Without this, a new field defaults to whichever branch the author happened not to
    touch -- silently non-blocking, which is how a blocking condition becomes
    optional with no review. ``frames`` is excluded deliberately: it carries no
    provenance and blocks by being empty instead.
    """
    declared = {f.name for f in dataclasses.fields(Conditions)} - {"frames"}
    blocking, allowed = set(BLOCKS_THE_STUDY), set(DECLARED_UNKNOWN_OK)
    assert not blocking & allowed, "a field cannot both block and be optional"
    # Pinned literally against §6.3, not read back from the constant: the
    # parametrised test above iterates BLOCKS_THE_STUDY, so demoting a field would
    # make that test quietly stop testing it rather than fail.
    assert blocking == {"assay", "tissue"}
    assert declared == blocking | allowed, (
        "unclassified condition field(s): " + ", ".join(sorted(declared - blocking - allowed))
    )


# --------------------------------------------------------------------------- #
# Contract hygiene -- the same two rules the measured contracts are held to
# --------------------------------------------------------------------------- #
def test_contracts_are_frozen_and_hashable():
    """Conditions is a cache key: the applicability of a claim is a function of it."""
    for obj in (UNKNOWN, Condition("Xenium", Provenance.READ), conditions()):
        hash(obj)
        with pytest.raises((AttributeError, TypeError)):
            setattr(obj, next(iter(type(obj).__slots__)), "mutated")


@pytest.mark.req("IMPL-0-01")
def test_no_mutable_fields_in_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. Same scan
    ``test_types_measured`` runs over subject/measured, extended to conditions."""
    from panoscope.types import conditions as C

    bad = []
    for name in dir(C):
        obj = getattr(C, name)
        if not (isinstance(obj, type) and dataclasses.is_dataclass(obj)):
            continue
        for f in dataclasses.fields(obj):
            t = str(f.type)
            if any(x in t for x in ("dict[", "list[", "set[")) and "frozenset" not in t:
                bad.append(f"{obj.__name__}.{f.name}: {f.type}")
    assert not bad, "mutable/unordered fields in frozen contracts: " + "; ".join(bad)
