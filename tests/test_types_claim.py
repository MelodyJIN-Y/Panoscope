"""Invariants for the claim contract.

As in ``test_types_measured.py`` and ``test_types_outputs.py``, the tests are the
deliverable: each pins a property something downstream is entitled to assume, and
names what breaks when it stops holding.

Three are the point of the module:

* authority ORDERS and never merges -- and orders the right way round, which a
  ``StrEnum`` would have got backwards for three of the four classes;
* a LAB claim with no formation conditions is unstorable, so the lab's judgement
  can never be applied to a study it was not formed in;
* the scope gate is fail-closed, including for inputs it cannot parse.
"""

from __future__ import annotations

import dataclasses
import sys

import pytest

from panoscope.types.claim import (
    PREDICATES, SOURCE_SCHEMES, Authority, BadScope, Claim, UnformedLabClaim,
    UnknownSource, UnregisteredPredicate, by_authority,
)
from panoscope.types.conditions import UNKNOWN, Condition, Conditions, Provenance
from panoscope.types.measured import FrameRef
from panoscope.types.subject import EntityKind, gene, subject

ERBB2 = subject("gene", g=gene("ERBB2"))
PANEL_REF = FrameRef(EntityKind.GENE, "panel:xenium_hbc_v1", "9f3a" * 16, 280)


def _conditions(frames: tuple[FrameRef, ...] = (PANEL_REF,)) -> Conditions:
    return Conditions(
        assay=Condition("xenium", Provenance.READ),
        modality=UNKNOWN,
        tissue=Condition("UBERON:0000310", Provenance.STATED),
        method=UNKNOWN,
        segmentation=UNKNOWN,
        fixation=UNKNOWN,
        frames=frames,
    )


def _claim(**kw) -> Claim:
    base = dict(
        id="c1", subject=ERBB2, predicate="marks", object="Tumor",
        authority=Authority.CURATED, source="panel:xenium_hbc_v1",
    )
    return Claim(**{**base, **kw})


# --------------------------------------------------------------------------- #
# Authority: ordered, never merged
# --------------------------------------------------------------------------- #
def test_authority_orders_measurement_above_everything():
    """ARCHITECTURE §10.2, and the reason this is an IntEnum.

    Sorted as strings these come out curated, lab, literature, measurement -- which
    puts literature above lab and measurement last. The ordering is the only thing
    the enum is for, so it has to be the numbers.
    """
    assert (Authority.MEASUREMENT > Authority.LAB > Authority.CURATED
            > Authority.LITERATURE)
    assert sorted(Authority) == [Authority.LITERATURE, Authority.CURATED,
                                 Authority.LAB, Authority.MEASUREMENT]


def test_by_authority_groups_and_is_stable():
    """Highest first; ties keep the store's order.

    Stability is the guard against resolve-by-recency creeping back in as a display
    detail: nothing here reorders two claims of equal authority.
    """
    lit_a = _claim(id="a", authority=Authority.LITERATURE, source="pmid:12345")
    lit_b = _claim(id="b", authority=Authority.LITERATURE, source="pmid:67890")
    meas = _claim(id="m", authority=Authority.MEASUREMENT,
                  source="file:sha256:" + "0" * 64)
    out = by_authority((lit_a, lit_b, meas))
    assert [c.id for c in out] == ["m", "a", "b"]


def test_nothing_merges_two_claims():
    """There is deliberately no combiner. Conflicts are grouped and displayed.

    Pinned as an API absence rather than a behaviour: the moment a ``merge`` or
    ``dedupe`` helper exists, something calls it and the authority label blurs.
    """
    import panoscope.types.claim as mod

    assert not [n for n in dir(mod)
                if any(w in n.lower() for w in ("merge", "dedupe", "consolidat"))]


# --------------------------------------------------------------------------- #
# The LAB invariant (IMPLEMENTATION §2.7)
# --------------------------------------------------------------------------- #
def test_lab_claim_without_conditions_is_unstorable():
    """The invariant §2.7 states outright.

    A lab claim is formed while looking at one measurement of one tissue. Stored
    without the frames it was formed under it cannot be transfer-checked, so it
    would apply everywhere -- and the caveat about one FFPE block would silently
    qualify a fresh-frozen study.
    """
    with pytest.raises(UnformedLabClaim):
        _claim(authority=Authority.LAB, source="note:7a1c", conditions=None)


def test_lab_claim_with_frames_is_storable():
    c = _claim(authority=Authority.LAB, source="note:7a1c",
               conditions=_conditions(), author="jin.m")
    assert c.conditions is not None and c.conditions.frames == (PANEL_REF,)


def test_only_lab_requires_conditions():
    """Literature is the reason ``conditions`` is optional at all.

    A claim read off a PMID has no study intake behind it: its assay and tissue are
    genuinely unknown, and ``Conditions`` refuses to be constructed with an unknown
    assay. Requiring one here, as §2.7 does, would make imported literature
    unstorable.
    """
    lit = _claim(authority=Authority.LITERATURE, source="pmid:31753849")
    assert lit.conditions is None


# --------------------------------------------------------------------------- #
# Registered predicate, resolvable source
# --------------------------------------------------------------------------- #
def test_predicate_must_be_registered():
    with pytest.raises(UnregisteredPredicate):
        _claim(predicate="is_probably_a")


def test_every_registered_predicate_has_a_description():
    """A registry entry nothing can explain is a retrieval mask nobody can test."""
    assert PREDICATES
    assert all(p.id and p.what for p in PREDICATES.values())
    assert all(k == p.id for k, p in PREDICATES.items())


@pytest.mark.parametrize("uri", [
    "pmid:31753849", "doi:10.1038/s41586-021-03634-9", "note:7a1c",
    "panel:xenium_hbc_v1", "file:sha256:" + "ab" * 32,
])
def test_known_source_schemes_are_accepted(uri):
    assert _claim(source=uri).source == uri


@pytest.mark.parametrize("uri", [
    "Smith et al. 2019",          # the failure mode: a citation nothing can resolve
    "pmid:",                      # a scheme with nothing behind it
    "arxiv:2301.00001",           # plausible, unknown, and silently unresolvable
    "",
])
def test_unresolvable_source_is_refused(uri):
    """A fabricated citation is the worst failure this project can produce, and an
    unresolvable one is indistinguishable from it once it is in the report."""
    with pytest.raises(UnknownSource):
        _claim(source=uri)


def test_source_schemes_cover_every_writer():
    """pmid/doi for literature, panel+file for measurement, note for the lab. A
    writer with no scheme has nowhere to put its source but free text."""
    assert SOURCE_SCHEMES == {"pmid", "doi", "note", "panel", "file"}


# --------------------------------------------------------------------------- #
# Scope: fail-closed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scope,context,expected", [
    ("lab", "dataset:d1", True),         # a lab-wide claim holds downward
    ("lab", "lab", True),
    ("study", "sample:s1", True),
    ("study", "lab", False),             # ... but never upward
    ("dataset:d1", "dataset:d1", True),
    ("dataset:d1", "dataset:d2", False),
    ("sample:s1", "sample:s1", True),
    ("sample:s1", "sample:s2", False),   # the note about one FFPE block
    ("sample:s1", "study", False),
    ("dataset:d1", "sample:s1", False),  # membership is not derivable from strings
])
def test_scope_gate(scope, context, expected):
    assert _claim(scope=scope).applies_in(context) is expected


def test_malformed_context_is_false_not_an_error():
    """Contexts arrive from the session. A typo must narrow what is shown, never
    widen it, and never raise in the middle of a retrieval."""
    assert _claim(scope="lab").applies_in("datset:d1") is False
    assert _claim(scope="lab").applies_in("") is False
    assert _claim(scope="lab").applies_in("sample:") is False


@pytest.mark.parametrize("bad", ["dataset", "dataset:", "sample:s1:extra",
                                 "everywhere", ""])
def test_unparseable_scope_fails_at_the_write(bad):
    """Fail-closed makes a bad scope invisible rather than loud -- the claim would
    be filtered out of every retrieval, silently. So the failure is moved forward."""
    with pytest.raises(BadScope):
        _claim(scope=bad)


# --------------------------------------------------------------------------- #
# Lifecycle coherence
# --------------------------------------------------------------------------- #
def test_supersession_points_both_ways_or_neither():
    """A claim marked superseded with nothing to follow is a dead end; a pointer on
    an active claim reads as active to retrieval and replaced to the reader."""
    with pytest.raises(ValueError):
        _claim(valid="superseded")
    with pytest.raises(ValueError):
        _claim(valid="active", superseded_by="c2")
    assert _claim(valid="superseded", superseded_by="c2").superseded_by == "c2"


def test_status_is_orthogonal_to_authority():
    """§10 says keep ``status``: authority is who may be believed, status is how
    firmly they said it. A tentative measurement is a real thing."""
    c = _claim(authority=Authority.MEASUREMENT, status="tentative",
               source="file:sha256:" + "0" * 64)
    assert (c.authority, c.status) == (Authority.MEASUREMENT, "tentative")
    with pytest.raises(ValueError):
        _claim(status="maybe")
    with pytest.raises(ValueError):
        _claim(valid="deleted")


def test_claim_needs_an_id_to_click_back_to():
    with pytest.raises(ValueError):
        _claim(id="")


# --------------------------------------------------------------------------- #
# Structural
# --------------------------------------------------------------------------- #
def test_claims_are_hashable_and_frozen():
    """They are hashed into the record and grouped in sets at retrieval."""
    c = _claim()
    assert {c, _claim()} == {c}
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.authority = Authority.MEASUREMENT  # type: ignore[misc]


def test_no_clock_or_io_in_the_claim_module():
    """``created`` is a string someone else minted. A default of ``now()`` would put
    a clock into a type that is hashed into the golden record."""
    path = sys.modules["panoscope.types.claim"].__file__
    assert path is not None
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    for forbidden in ("import time", "datetime", "import random", "open("):
        assert forbidden not in text, f"{forbidden!r} in claim.py"
    assert _claim().created == ""


def test_no_mutable_fields_in_contracts():
    """dict/list/set fields break byte-determinism: they are unhashable and their
    iteration order flows through asdict() into the serialised bytes. The record
    determinism invariant cannot hold if any contract carries one."""
    from panoscope.types import claim as C

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
