"""What a method module declares about its own result, checked against the result.

A method module is a machine-readable declaration of what a method's paper says it
measures. The declarations that are CONSTANTS -- a bar, a floor, a schema -- are
statements about the method and are true on every study. The ones that describe the
RESULT are not, and a constant standing in for one of those states false provenance
on every study where it happens to be wrong, silently and in a committed artifact.
"""

from __future__ import annotations

from panoscope.methods import jazzpanda_cgst as CGST
from panoscope.types.measured import Frame
from panoscope.types.subject import EntityKind


def _table(*gene_sets: str) -> list[dict[str, object]]:
    """The minimum a row needs for ``scored_frame``: its set's name."""
    return [{"gene_set": g} for g in gene_sets]


def test_the_scored_collection_is_derived_from_the_result_not_asserted():
    """``COLLECTION = "MSigDB_Hallmark"`` was hardcoded onto every study.

    True on the demo and false on the synthetic fixture, whose four gene sets are
    invented -- so the GENE_SET frame named a real, restrictively-licensed collection
    as the provenance of rows that came from nowhere, in the one study that exists
    because nothing licensed may be bundled. ``agent/enrichment.py`` had the same
    constant and it was fixed there last round; this is the twin that was disclosed
    and left.

    The fixture's shape is the discriminating input: on the demo the derived answer
    and the old constant are the SAME string, so a test built on hallmark sets alone
    would pass against the defect.
    """
    assert not hasattr(CGST, "COLLECTION")

    demo = CGST.MODULE.scored_frame(_table("HALLMARK_EMT", "HALLMARK_MYC_TARGETS_V1"))
    assert demo.name == "collection:MSigDB_Hallmark"

    fixture = CGST.MODULE.scored_frame(
        _table("SYNSET_ALPHA", "SYNSET_BETA", "SYNSET_GAMMA", "SYNSET_DELTA"))
    assert fixture.name == f"collection:{CGST.UNKNOWN_COLLECTION}"
    assert CGST.UNKNOWN_COLLECTION == "unknown"
    assert "MSigDB" not in fixture.name


def test_a_frame_that_cannot_name_one_collection_names_none():
    """One frame covers the whole table, so it can carry one name honestly.

    A mixed result gets ``unknown`` rather than the half that matched: the frame is
    what tells a reader which universe a TESTED_ABSENT set was asked in, and half an
    answer there reads as a whole one. An empty table has nothing to derive from and
    lands in the same place.
    """
    mixed = CGST.MODULE.scored_frame(_table("HALLMARK_EMT", "SYNSET_ALPHA"))
    assert mixed.name == "collection:unknown"
    assert CGST.MODULE.scored_frame(_table()).name == "collection:unknown"


def test_the_frame_still_admits_every_set_the_result_scored():
    """The derivation changes the frame's NAME; its membership is the result's own
    set of scored sets, which is what separates TESTED_ABSENT from NEVER_MEASURED."""
    frame = CGST.MODULE.scored_frame(_table("HALLMARK_EMT", " SYNSET_ALPHA "))
    assert isinstance(frame, Frame) and frame.kind is EntityKind.GENE_SET
    assert frame.members == frozenset({"HALLMARK_EMT", "SYNSET_ALPHA"})
