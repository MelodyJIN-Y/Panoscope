"""Evidence table: the jazzPanda-derived specificity badge.

The badge must be driven off the record's own evidence (Tier A), not the LLM
biology note — so it is grounded and shows even when a gene has no note. It fires
only for ``max_gc_corr > pearson`` ("localizes better with another cluster"), never
for another caveat. These tests pin that behavior.

The badge now reads the **registered flag id** (``types.evidence.FLAGS``) off the
``EvidenceItem`` and looks its English up in ``evidence_table.FLAG_COPY``, rather
than matching a literal English phrase. That is the direction the id->copy table was
built for: a copy edit in the analysis layer used to stop the badge rendering while
this file stayed green, because both sides spelled the sentence out.
"""

from __future__ import annotations

from types import SimpleNamespace

from panoscope.ui import evidence_table as et

_FLAG = "specificity:localises_elsewhere"
_COPY = et.FLAG_COPY[_FLAG]


def _item(flags=(), gene="XIST"):
    return SimpleNamespace(flags=tuple(flags),
                           subject=SimpleNamespace(display=lambda: gene))


def test_the_chip_is_present_for_a_flagged_item_and_absent_otherwise():
    assert _COPY in et._chips(_item((_FLAG,)), frozenset())
    assert et._chips(_item(), frozenset()) == ()
    # A different registered flag must NOT produce the specificity copy.
    assert _COPY not in et._chips(_item(("specificity:low",)), frozenset())


def test_every_registered_flag_has_copy_so_no_id_leaks_to_the_screen():
    """``_chips`` falls back to the raw id, which would put ``specificity:low`` on a
    biologist's screen. This table is the one home for that English."""
    from panoscope.types.evidence import FLAGS

    assert set(FLAGS) == set(et.FLAG_COPY)


def test_the_chip_comes_off_the_record_not_off_the_biology_note():
    """Grounded: the chip is read off the item's flags, so it does not depend on a
    gene having a precomputed biology note."""
    built = et.unit_build("c1")
    rows = et.rows_for("c1", built)
    flagged = [r for r in rows if _COPY in r.values["flags"]]
    assert flagged, "expected at least one specificity-flagged marker in c1"
    items = {i.subject.display(): i for i in built.evidence.items}
    for row in flagged:
        assert _FLAG in items[row.values["gene"]].flags


def test_canonical_membership_is_a_chip_and_selects_nothing_else():
    """Reference data may select which rows matter; it may never supply a value."""
    assert et._chips(_item(gene="ERBB2"), frozenset({"ERBB2"})) == ("canonical",)
    assert et._chips(_item(gene="ERBB2"), frozenset()) == ()
