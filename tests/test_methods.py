"""What a method module declares about its own result, checked against the result.

A method module is a machine-readable declaration of what a method's paper says it
measures. The declarations that are CONSTANTS -- a bar, a floor, a schema -- are
statements about the method and are true on every study. The ones that describe the
RESULT are not, and a constant standing in for one of those states false provenance
on every study where it happens to be wrong, silently and in a committed artifact.
"""

from __future__ import annotations

import pytest

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


# --------------------------------------------------------------------------- #
# The trust boundary: the method's table is produced by someone else
# --------------------------------------------------------------------------- #
def _row(**over):
    base = {"gene": "ERBB2", "top_cluster": "c1", "glm_coef": 4.0, "pearson": 0.9,
            "max_gg_corr": 0.9, "max_gc_corr": 0.9, "cell_type": "Tumor"}
    return {**base, **over}


def test_a_padded_gene_symbol_is_stripped_not_carried():
    """A padded symbol became the entity id: it matched no capability frame, was
    not canonical and did not click back, while still carrying its number.
    Stripping is normalisation, not invention."""
    from panoscope.methods.jazzpanda import _normalised

    out = _normalised((_row(gene=" KRT7 "), ))
    assert out[0]["gene"] == "KRT7"


def test_a_statistic_that_is_not_a_number_is_refused():
    """``glm_coef = NaN`` gave driver=('ERBB2', nan), state OBSERVED, band
    Medium, basis "glm_coef direct", verify FALSE -- the band ladder running on
    a non-number, and ``statistic is None iff NOT OBSERVED`` violated by a value
    that is not a value."""
    from panoscope.methods.jazzpanda import MalformedMethodOutput, _normalised

    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(MalformedMethodOutput) as exc:
            _normalised((_row(glm_coef=bad), ))
        assert "ERBB2" in str(exc.value) and "glm_coef" in str(exc.value)

    # ...and every other statistic column is held to the same rule.
    with pytest.raises(MalformedMethodOutput):
        _normalised((_row(pearson=float("nan")), ))


def test_one_measurement_may_not_carry_two_numbers():
    """c1 ABCC11 appeared twice at 99.9 and 4.176, and both reached the evidence:
    the table showed one gene with two contradictory values and nothing
    downstream could choose between them."""
    from panoscope.methods.jazzpanda import MalformedMethodOutput, _normalised

    with pytest.raises(MalformedMethodOutput) as exc:
        _normalised((_row(glm_coef=4.0), _row(glm_coef=99.9)))
    assert "more than once" in str(exc.value)
    assert "glm_coef" in str(exc.value)

    # An exactly duplicated row says the same thing twice; collapse it.
    assert len(_normalised((_row(), _row()))) == 1
    # Different units are different measurements.
    assert len(_normalised((_row(top_cluster="c1"), _row(top_cluster="c2")))) == 2


def test_the_real_study_is_unchanged_by_the_guard():
    """The guard must cost the committed study nothing."""
    from panoscope.agent import config as cfg
    from panoscope.pipeline import records

    built = records.build_all(cfg.current_study())["c1"]
    assert built.verdict.band == "Very High"
    assert built.verdict.driver.subject.one("g").id == "ERBB2"
    assert len(built.evidence.items) == 281


def test_the_guard_is_wired_into_the_reader_not_just_available():
    """``_rows`` is what ``load`` calls, so the guard has to sit on that path.

    Testing ``_normalised`` alone proved it works and not that anything uses
    it: reverting the single line that calls it left every one of these tests
    green.
    """
    import pandas as pd

    from panoscope.methods.jazzpanda import MalformedMethodOutput, _rows

    assert _rows(pd.DataFrame([_row(gene=" KRT7 ")]))[0]["gene"] == "KRT7"

    with pytest.raises(MalformedMethodOutput):
        _rows(pd.DataFrame([_row(glm_coef=float("nan"))]))

    with pytest.raises(MalformedMethodOutput):
        _rows(pd.DataFrame([_row(glm_coef=4.0), _row(glm_coef=99.9)]))
