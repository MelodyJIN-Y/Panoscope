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


# --------------------------------------------------------------------------- #
# The replicate read (§9.10) on the page, not only in the terminal
#
# ``grep -rn "undecidable" panoscope/ui/`` returned nothing: the one flag the
# concordance wiring exists to produce -- "sampleA and sampleB read this gene
# differently" -- reached ``panoscope explain`` and no pixel. These tests are about
# what the markers surface renders, which is why the last one drives the real
# renderer rather than the pure functions.
#
# Nothing below hard-codes WHICH subject of the fixture diverges. The per-sample
# frames are changing underneath this, and a test that pins the gene would pass by
# agreeing with today's answer instead of with the record.
# --------------------------------------------------------------------------- #
import pytest

from panoscope.data.study import Study
from panoscope.pipeline import records
from panoscope.ui import surface

FIXTURE = "synthetic_multisample_fixture"


@pytest.fixture(scope="module")
def fixture_builds():
    """The two-sample fixture, built through the composition root. Reads only."""
    return records.build_all(Study.open(FIXTURE))


def _rules(record, rule):
    return [d for d in record.decisions if d.rule == rule]


def _anchors(unit, built):
    """Every anchor THIS unit's page carries: one per row, plus the unit itself."""
    return {i.subject.key for i in built.evidence.items} | {et._unit_key(unit)}


def test_every_divergence_in_the_record_reaches_the_surface(fixture_builds):
    """The defect itself: a divergent decision with no line is a flag nobody sees."""
    for unit, built in fixture_builds.items():
        lines = et.concordance_lines(unit, built)
        claimed = {ln.claim for ln in lines if ln.cls == "diverg"}
        assert claimed == {d.outcome for d in _rules(built.record, "concordance:divergent")}, unit
        html = et.concordance_html(lines, _anchors(unit, built))
        for ln in lines:
            if ln.cls == "diverg":
                assert ln.text in html and ln.claim in html


def test_not_comparable_does_not_render_as_a_disagreement(fixture_builds):
    """The hard half. "the samples could not be compared" and "they genuinely
    disagree" are different facts, and the renderer is the last place the four
    states can collapse into two. Both must be present AND must not read the same."""
    blocked, diverged = [], []
    for unit, built in fixture_builds.items():
        lines = et.concordance_lines(unit, built)
        blocked += [ln for ln in lines if ln.cls == "blocked"]
        diverged += [ln for ln in lines if ln.cls == "diverg"]
    assert blocked and diverged, "the fixture carries both outcomes; both must render"

    for ln in blocked:
        assert "could not be compared" in ln.text
        assert "read differently" not in ln.text
        # The state that blocked it, named -- not implied by an empty line.
        assert "cannot be compared" in ln.why
    for ln in diverged:
        assert "read differently" in ln.text
        assert "could not be compared" not in ln.text
    assert {ln.glyph for ln in blocked}.isdisjoint({ln.glyph for ln in diverged})


def test_no_replicate_line_borrows_a_measurement_glyph(fixture_builds):
    """The fabricated measurement status, pinned at the source and on the page.

    ``&#8856;`` is the registered NEVER_MEASURED glyph and the rows directly above
    the replicate read use it with that meaning. A concordance outcome is not a
    measurement state: ``concordance:not_comparable`` covers a frame that refused
    the subject, an underpowered sample AND a sample that reported no row at all,
    and the record does not say which. Wearing the never-measured glyph it asserted
    the first on an on-panel gene that was run -- *off panel, nothing to see*, which
    is the kidney study's background hypothesis for almost every marker and so the
    one reading a biologist would not stop to question.
    """
    lattice = {g for g, _ in surface._GLYPH.values()}
    assert "&#8856;" in lattice, "the glyph this test is about must still be the real one"
    assert {g for _, g, _ in et._CONC_COPY.values()}.isdisjoint(lattice)

    for unit, built in fixture_builds.items():
        for ln in et.concordance_lines(unit, built):
            assert ln.glyph not in lattice, ln


def test_the_hedge_is_in_the_sentence_not_only_in_the_hover(fixture_builds):
    """"Measurement status is visible, not implied" is not satisfied by a tooltip.

    What the record cannot tell apart must be said in the text a biologist reads
    without hovering, so the rendered line names all three causes the one rule id
    covers rather than picking the most reassuring of them.
    """
    seen = 0
    for unit, built in fixture_builds.items():
        lines = [ln for ln in et.concordance_lines(unit, built)
                 if ln.cls == "blocked"]
        html = et.concordance_html(lines, _anchors(unit, built))
        for ln in lines:
            seen += 1
            for cause in ("never measured it", "underpowered",
                          "reported no row at all"):
                assert cause in ln.text, (cause, ln)
            assert "does not say which" in ln.text
            # In the visible text, not only in title=.
            assert f">{ln.glyph} {ln.text}</div>" in html
    assert seen, "the fixture must carry a not-comparable subject for this to prove anything"


def test_agreement_is_summarised_with_its_denominator_and_never_as_a_lift(
        fixture_builds):
    """§9.10: agreement never raises confidence. It is one line carrying the count,
    because a check that compared one subject and a check that compared forty both
    render as *no divergence* without it."""
    built = fixture_builds["c3"]                      # the unit the samples agree on
    n = len(_rules(built.record, "concordance:consistent"))
    assert n, "c3 is the agreement control; it must carry consistent decisions"

    lines = et.concordance_lines("c3", built)
    same = [ln for ln in lines if ln.cls == "same"]
    assert len(same) == 1 and f"{n} subject(s)" in same[0].text
    assert "NEVER raises confidence" in same[0].text
    assert not [ln for ln in lines if ln.cls == "diverg"]


def test_a_single_sample_study_says_why_there_is_no_replicate_read():
    """The demo. A never-measured thing is not a blank row here either: the page
    says the comparison does not apply rather than rendering nothing."""
    built = et.unit_build("c1")
    lines = et.concordance_lines("c1", built)
    assert [ln.cls for ln in lines] == ["undecidable"]
    assert "one sample" in lines[0].text
    assert "does not apply" in lines[0].text
    html = et.concordance_html(lines, _anchors("c1", built))
    assert "one sample" in html and lines[0].claim in html


def test_every_line_resolves_to_a_claim_and_an_unanchored_one_refuses_to_render(
        fixture_builds):
    """Click-back. A sentence with nothing to click should not have been rendered."""
    for unit, built in fixture_builds.items():
        # THIS unit's anchors, not every unit's: a c1 line that resolved only to a
        # row on c2's page is a click-back that lands nowhere on the page it is on.
        keys = _anchors(unit, built)
        for ln in et.concordance_lines(unit, built):
            assert ln.claim.strip()
            # A subject line resolves to its own row; a unit line to the unit.
            assert ln.claim in keys, ln

    with pytest.raises(surface.NothingToClick):
        et.concordance_html([et.ConcLine("diverg", "!", "something diverged", "")],
                            ())


# --------------------------------------------------------------------------- #
# Click-back, the half the guard was missing
#
# ``NothingToClick`` only ever checked the EMPTY STRING, which fires when a caller
# forgot an anchor -- a bug that does not survive its first run. What shipped was a
# line anchored on a subject NO ROW ON THE PAGE carries: a gene only a per-sample
# table reported, whose ``data-claim`` none of the page's anchors resolves, with the
# join key printed where the gene name belongs.
#
# The fixture's divergent subjects all have rows, so nothing below reads the
# fixture for this: these build the one shape that separates the halves.
# --------------------------------------------------------------------------- #
_OFF_PAGE = "gene|g=gene:SYNG13"


def _built_with(rule, outcome, *, rows=("SYNG01",)):
    """A minimal built record carrying one concordance decision.

    Synthetic on purpose. Which subjects come back divergent is changing under this
    file, and a test that waited for the fixture to grow a subject with no row would
    pin whatever the fixture happens to do today.
    """
    from panoscope.types.subject import Entity, EntityKind, subject as subj

    def _it(sym):
        s = subj("gene", g=Entity(EntityKind.GENE, sym))
        return SimpleNamespace(subject=s)

    return SimpleNamespace(
        record=SimpleNamespace(
            decisions=[SimpleNamespace(rule=rule, outcome=outcome,
                                       because=("why",))],
            undecidable=(),
        ),
        evidence=SimpleNamespace(items=[_it(s) for s in rows]),
    )


def test_a_subject_with_no_row_is_named_and_resolves_to_something_on_the_page():
    """The hard half of click-back AND of the leaking key, in one line.

    ``gene|g=gene:SYNG13`` is the string reconciliation joins on, not a gene name,
    and it was rendered as prose. The line now carries the NAME, says in the visible
    text that the evidence above has no row for it, and anchors on the unit -- the
    one thing on this page it can honestly resolve to.
    """
    built = _built_with("concordance:divergent", _OFF_PAGE)
    (ln,) = et.concordance_lines("c1", built)

    assert "SYNG13" in ln.text
    assert "gene|g=" not in ln.text and ":gene:" not in ln.text
    assert "no row for it in the evidence above" in ln.text
    assert ln.claim == et._unit_key("c1")
    # And it renders, because its anchor now resolves to something on the page.
    assert "SYNG13" in et.concordance_html([ln], _anchors("c1", built))


def test_a_subject_with_a_row_still_resolves_to_its_own_row():
    """The easy half, kept: retargeting every line to the unit would be a second
    bug wearing the fix's clothes -- a click-back that no longer lands on the row
    the sentence is about."""
    built = _built_with("concordance:divergent", "gene|g=gene:SYNG01")
    (ln,) = et.concordance_lines("c1", built)
    assert ln.claim == "gene|g=gene:SYNG01"
    assert "no row for it" not in ln.text


def test_an_anchor_that_resolves_to_nothing_refuses_to_render():
    """The widened guard itself: a non-empty key naming nothing on the page is the
    same defect as no key at all, with a key in it to make it look answered."""
    ln = et.ConcLine("diverg", "!", "SYNG13 read differently", _OFF_PAGE)
    assert et.concordance_html([ln], {_OFF_PAGE})          # resolvable: renders
    with pytest.raises(surface.NothingToClick) as e:
        et.concordance_html([ln], {"gene|g=gene:SYNG01"})  # not on the page
    assert _OFF_PAGE in str(e.value)


def test_every_registered_concordance_rule_has_copy_or_is_summarised():
    """``concordance_lines`` drops any ``concordance:*`` id it has no copy for, in
    silence. The outcomes are not a closed set -- a new one must fail here, loudly,
    rather than render as a replicate read with a line missing from it."""
    from panoscope.types.trace import RULES

    ids = {r for r in RULES if r.startswith("concordance:")}
    assert ids - set(et._CONC_COPY) == {"concordance:consistent"}, \
        "a concordance outcome with no copy would not reach the page at all"


def test_the_block_reaches_the_rendered_page_for_a_two_sample_study():
    """The production path: the same thing, through ``render_evidence_table``.

    The pure functions above would all pass with the call to them deleted from the
    renderer, which is exactly the state this stream found the codebase in."""
    try:
        from streamlit.testing.v1 import AppTest
    except Exception:  # noqa: BLE001 - Streamlit testing API absent
        pytest.skip("streamlit.testing.v1 unavailable")

    def _run() -> None:
        from panoscope.agent import config as _cfg
        from panoscope.data.study import Study as _Study
        from panoscope.ui import evidence_table as _et

        # Set inside the script: the active study is a ContextVar and the AppTest
        # script runs in its own thread.
        _cfg.set_study(_Study.open("synthetic_multisample_fixture"))
        _et.render_evidence_table("c1")

    at = AppTest.from_function(_run, default_timeout=120)
    at.run()
    if at.exception and "unexpected keyword argument 'key'" in str(at.exception[0].value):
        pytest.skip("Streamlit too old for st.container(key=...)")
    assert not at.exception, [str(e.value) for e in at.exception]

    md = "\n".join(m.value for m in at.markdown)
    expected = et.concordance_lines("c1", records.build_all(Study.open(FIXTURE))["c1"])
    assert [ln for ln in expected if ln.cls == "diverg"], \
        "the fixture must still carry a divergence for this to prove anything"
    for ln in expected:
        assert ln.claim in md, ln
    assert "read differently" in md and "replicate read" in md
