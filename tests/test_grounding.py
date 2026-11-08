"""Grounding-gate tests — the confident floor, proven.

These tests exercise ``agent.grounding_check.GroundingChecker`` against the REAL
jazzPanda values loaded through ``agent.data``. The critical test is the poisoned
answer: a number invented in the PROSE and deliberately absent from the sidecar
must still be caught. If that test does not fail the answer, the floor is not
enforced and the whole guarantee is hollow.

Real anchors used (verified against data/jazzpanda/markers_top.csv):
- LUM  (c2 Stromal): glm_coef 17.9978807964945, pearson 0.908125030077264
- POSTN (c2 Stromal): glm_coef 15.7967446458044,  pearson 0.798551442388682
- ERBB2 (c1 Tumor):   glm_coef 21.4378453933592,  pearson 0.906160189145375
"""

from __future__ import annotations

import pytest

from panoscope.agent.grounding_check import (
    Extractor,
    GroundingChecker,
    GroundingResult,
    SourceIndex,
)
from panoscope.agent.types import GroundingSidecar


# --------------------------------------------------------------------------- #
# Fixtures / stubs
# --------------------------------------------------------------------------- #
def _resolver_all_real(ident: str) -> bool:
    """Literature verifier stub that resolves every identifier (real record)."""
    return True


def _resolver_none_real(ident: str) -> bool:
    """Literature verifier stub that resolves nothing (fabricated citation)."""
    return False


@pytest.fixture
def checker_real_lit() -> GroundingChecker:
    """Checker whose injected verifier treats every PMID/DOI as real."""
    return GroundingChecker(literature_verifier=_resolver_all_real)


# A sidecar that deliberately OMITS the poisoned number, to prove the checker
# does not lean on the sidecar as a whitelist.
_CLEAN_SIDECAR = GroundingSidecar(
    numbers=(("LUM", "glm_coef", 17.9978807964945),),
    markers=("LUM", "POSTN"),
    pmids=(),
    notes_used=(),
)


# --------------------------------------------------------------------------- #
# (a) A clean, fully grounded answer -> ok = True
# --------------------------------------------------------------------------- #
def test_clean_grounded_answer_passes(checker_real_lit):
    answer = (
        "Cluster c2 reads as Stromal. Its strongest driver is LUM at "
        "glm_coef 17.998 with pearson 0.91, and POSTN corroborates at "
        "glm_coef 15.80. Confidence is High."
    )
    result = checker_real_lit.check(answer, _CLEAN_SIDECAR, "c2")
    assert isinstance(result, GroundingResult)
    assert result.ok is True, result.summary()
    assert result.violations == ()


# --------------------------------------------------------------------------- #
# (b) THE CRITICAL TEST — a number invented ONLY in prose, absent from the
#     sidecar, must FAIL the answer. If this passes, the floor is not enforced.
# --------------------------------------------------------------------------- #
def test_poisoned_prose_number_absent_from_sidecar_fails(checker_real_lit):
    # LUM's real glm_coef is 17.998. The prose lies with 99.9. The sidecar below
    # does NOT contain 99.9 — it only lists the true number — so a checker that
    # trusted the sidecar as a whitelist would wrongly pass. It must not.
    poisoned = (
        "Cluster c2 is Stromal, driven overwhelmingly by LUM glm 99.9 — an "
        "exceptionally strong spatial marker."
    )
    sidecar_without_poison = GroundingSidecar(
        numbers=(("LUM", "glm_coef", 17.9978807964945),),  # the TRUE value only
        markers=("LUM",),
        pmids=(),
        notes_used=(),
    )

    result = checker_real_lit.check(poisoned, sidecar_without_poison, "c2")

    assert result.ok is False, (
        "POISONED PROSE ESCAPED THE FLOOR: an invented LUM glm 99.9 that is "
        "absent from the sidecar was not caught. The confident floor is not "
        "enforced."
    )
    number_violations = [v for v in result.violations if v.kind == "number"]
    assert number_violations, result.summary()
    v = number_violations[0]
    assert v.ref == "LUM"
    assert "99.9" in v.claimed
    assert "17.99" in v.expected  # names the real value it should have been


def test_poisoned_prose_fails_even_with_no_sidecar(checker_real_lit):
    """Same poison, no sidecar at all — the prose pass must still catch it."""
    poisoned = "c2 Stromal — LUM glm 99.9, very strong."
    result = checker_real_lit.check(poisoned, None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "number" for v in result.violations)


# --------------------------------------------------------------------------- #
# (c) A fabricated PMID (verifier returns False) -> ok = False
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-HR-03", "UI-2.9-03")
def test_fabricated_pmid_fails():
    checker = GroundingChecker(literature_verifier=_resolver_none_real)
    answer = (
        "LUM glm_coef 17.998 supports a Stromal call; fibroblast identity is "
        "well established (PMID:99999999)."
    )
    result = checker.check(answer, None, "c2")
    assert result.ok is False, result.summary()
    citation_violations = [v for v in result.violations if v.kind == "citation"]
    assert citation_violations, result.summary()
    assert citation_violations[0].ref == "99999999"


@pytest.mark.req("CLAUDE-HR-03")
def test_a_pmcid_resolves_through_the_gate_and_a_fabricated_one_fails(checker_real_lit):
    answer = "LUM glm_coef 17.998 anchors the Stromal call (pmcid:PMC1234567)."
    assert checker_real_lit.check(answer, None, "c2").ok is True
    result = GroundingChecker(literature_verifier=_resolver_none_real).check(answer, None, "c2")
    assert [v.ref for v in result.violations if v.kind == "citation"] == ["PMC1234567"]


def test_real_pmid_passes(checker_real_lit):
    """A resolvable PMID (verifier True) does not trip the citation check."""
    answer = "LUM glm_coef 17.998 anchors the Stromal call (PMID:12345678)."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is True, result.summary()


# --------------------------------------------------------------------------- #
# (d) A correct real number -> ok = True
# --------------------------------------------------------------------------- #
def test_correct_real_number_passes(checker_real_lit):
    answer = "LUM glm 17.998 drives the c2 Stromal call."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is True, result.summary()


def test_rounded_real_number_passes(checker_real_lit):
    """A biologist writing 'LUM glm 18.0' for 17.998 is within tolerance."""
    answer = "LUM glm 18.0 drives the c2 Stromal call."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is True, result.summary()


# --------------------------------------------------------------------------- #
# Additional floor invariants
# --------------------------------------------------------------------------- #
def test_wrong_pearson_fails(checker_real_lit):
    """LUM pearson is 0.908; a stated 0.20 is fabrication."""
    answer = "LUM pearson 0.20 — weak spatial specificity."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "number" and v.ref == "LUM" for v in result.violations)


@pytest.mark.req("SKM-3e-02")
def test_statistic_on_offpanel_gene_fails(checker_real_lit):
    """COL1A1 is off-panel (no marker row). Stating a jazzPanda number for it is
    fabrication — off-panel absence is never a number."""
    assert not SourceIndex().is_modeled("COL1A1")
    answer = "COL1A1 glm 12.0 confirms Stromal."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is False, result.summary()
    # Either the unverifiable-number pass or the marker pass (or both) must fire.
    assert any(
        v.kind in {"unverifiable", "marker"} and v.ref == "COL1A1"
        for v in result.violations
    ), result.summary()


def test_uncited_or_missing_note_fails():
    """A referenced lab note that does not exist is a violation."""
    checker = GroundingChecker(
        literature_verifier=_resolver_all_real,
        known_notes={"note_c2_caf_real"},  # only this note exists
    )
    answer = "Per lab convention [note:ghost_note], c2 is CAF."
    result = checker.check(answer, None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "note" and v.ref == "ghost_note" for v in result.violations)


def test_existing_note_in_scope_passes():
    checker = GroundingChecker(
        literature_verifier=_resolver_all_real,
        known_notes={"note_c2_caf_real"},
    )
    answer = "Per lab convention [note:note_c2_caf_real], c2 is CAF."
    result = checker.check(answer, None, "c2", allowed_notes={"note_c2_caf_real"})
    assert result.ok is True, result.summary()


@pytest.mark.req("SKM-S6-02")
def test_out_of_scope_note_fails():
    """An existing note referenced outside its allowed scope is a violation."""
    checker = GroundingChecker(
        literature_verifier=_resolver_all_real,
        known_notes={"note_c2_caf_real", "note_c5_tcell"},
    )
    answer = "Applying [note:note_c5_tcell] here."
    result = checker.check(
        answer, None, "c2", allowed_notes={"note_c2_caf_real"}
    )
    assert result.ok is False, result.summary()
    assert any(
        v.kind == "note" and v.ref == "note_c5_tcell" for v in result.violations
    )


def test_sidecar_is_not_a_whitelist_for_wrong_numbers(checker_real_lit):
    """Even if the sidecar CONTAINS the poisoned number, the prose still governs:
    the prose number is checked against agent.data, not against the sidecar."""
    poisoned = "c2 Stromal — LUM glm 99.9."
    lying_sidecar = GroundingSidecar(
        numbers=(("LUM", "glm_coef", 99.9),),  # sidecar also lies
        markers=("LUM",),
        pmids=(),
        notes_used=(),
    )
    result = checker_real_lit.check(poisoned, lying_sidecar, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "number" for v in result.violations)


def test_source_index_number_matches_real_value():
    src = SourceIndex()
    assert src.number_matches("LUM", "glm_coef", 17.998) is True
    assert src.number_matches("LUM", "glm_coef", 99.9) is False
    assert src.number_matches("ERBB2", "glm_coef", 21.44) is True
    assert src.number_matches("NOT_A_GENE", "glm_coef", 1.0) is False


def test_no_false_positive_on_english_prose(checker_real_lit):
    """Ordinary prose with no gene+stat+number claim must not trip the gate."""
    answer = (
        "This cluster is uncertain; re-check this. The evidence is thin and the "
        "confidence is Low."
    )
    result = checker_real_lit.check(answer, None, "c9")
    assert result.ok is True, result.summary()


# --------------------------------------------------------------------------- #
# (e) The panel-absence rule, enforced ON PROSE
#
# CLAUDE.md's headline rule. The numeric passes cannot reach it: the sentence
# carries no number, so there is nothing for the gene+stat+number regex to bind.
# The frame is consulted instead, which is the only thing that makes the two
# halves of the rule separable -- an off-panel absence argues nothing, and a
# MEASURED absence argues, and a gate that collapsed them would be worse than
# no gate at all.
# --------------------------------------------------------------------------- #
@pytest.mark.req("CLAUDE-HR-01")
def test_off_panel_absence_used_as_evidence_fails(checker_real_lit):
    """The probe that used to walk straight through."""
    assert not SourceIndex().is_on_panel("COL1A1")
    answer = "COL1A1 is absent here, which argues against a fibroblast identity."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is False, (
        "THE PANEL-ABSENCE RULE IS UNENFORCED ON PROSE: an off-panel gene's "
        "absence was used as evidence and the gate let it through."
    )
    v = [x for x in result.violations if x.kind == "panel_absence"]
    assert v and v[0].ref == "COL1A1", result.summary()


def test_every_gene_in_a_list_is_checked_not_just_the_last(checker_real_lit):
    """A marker list is the ordinary phrasing, and it defeated the gate.

    Clause-scoping this pass closed the "one true caveat buys a free pass" hole
    and opened a bigger one: ``_CLAUSE_SPLIT_RE`` splits on "and", so
    "KRT18 and ERBB2 are not detected here" becomes "KRT18 " plus
    " ERBB2 are not detected here" -- the absence verb lands once, at the end, and
    a per-clause absence test discards every gene but the last. Measured on the
    real frame, that sentence went from CRITICAL to clean.

    The two passes fail in opposite directions on a dropped clause, which is why
    they cannot share a splitter: a dropped FRAME claim fails closed (the absence
    is still flagged); a dropped ABSENCE clause fails open.
    """
    for answer, must_flag in (
        ("KRT18 and ERBB2 are not detected here, so this rules out Tumor.", "KRT18"),
        ("CD3D and CD2 are not detected here, so this is not T cells.", "CD3D"),
        ("KRT18, CD3D and CD2 are not detected here, so this rules out T cells.", "CD2"),
    ):
        result = checker_real_lit.check(answer, None, "c1")
        assert result.ok is False, f"a marker list walked through the gate: {answer}"
        refs = [v.ref for v in result.violations if v.kind == "panel_absence"]
        assert must_flag in refs, f"flagged {refs}, expected {must_flag}: {answer}"

    # ...and a list whose every member is correctly caveated still passes whole.
    assert checker_real_lit.check(
        "KRT18 and CD3D were never measured, so neither absence argues anything.",
        None, "c1",
    ).ok is True


def test_one_correct_caveat_is_not_a_pass_for_the_rest_of_the_sentence(checker_real_lit):
    """Obeying the rule about ONE gene stopped the gate looking at the others.

    The absence pass dropped a whole sentence on its first never-measured phrase,
    so every answer that correctly caveated one gene carried a free pass for every
    other gene beside it: "KRT18 was never measured, but CD3D is not detected here,
    so this is not T cells" obeys the rule in its first clause, breaks it in its
    second, and passed clean. Scope is a clause, which is what the sibling
    frame-claim pass already says and does.
    """
    for answer, culprit in (
        ("KRT18 was never measured, but CD3D is not detected here, "
         "so this is not T cells.", "CD3D"),
        ("Because CD3D is off-panel, and because FOXP3 is absent, "
         "this is unlikely to be Tregs.", "FOXP3"),
    ):
        result = checker_real_lit.check(answer, None, "c1")
        assert result.ok is False, f"the gate was bypassed by a caveat: {answer}"
        refs = [v.ref for v in result.violations if v.kind == "panel_absence"]
        assert culprit in refs, f"flagged {refs}, expected {culprit}: {answer}"

    # ...and a sentence whose every clause obeys the rule still passes whole.
    assert checker_real_lit.check(
        "CD3D was never measured and KRT18 is off-panel, so neither absence "
        "argues anything here.", None, "c1",
    ).ok is True


def test_a_cell_type_abbreviation_is_not_an_unmeasured_gene(checker_real_lit):
    """A true answer must not be refused for saying "TAM".

    The token pattern is "any uppercase run", so every abbreviation a biologist
    writes was extracted as an ENTITY and the sentence rejected as reasoning from
    an out-of-frame absence. That costs a repair round the model cannot satisfy
    and then drops a correct answer into the fallback -- a refusal, and the
    refusal rate is reported next to accuracy precisely so this is visible.

    An invariant that fires on correct behaviour is worse than no invariant.
    """
    for answer in (
        "Myeloid markers are absent, so this is unlikely to be a TAM population.",
        "NK markers are absent here, so an NK identity is unlikely.",
        "ER and PR are not informative here, so that absence argues nothing.",
    ):
        result = checker_real_lit.check(answer, None, "c2")
        assert result.ok is True, (
            f"a true sentence was refused over an abbreviation: {result.summary()}"
        )

    # ...and the rule itself is untouched: a real gene still trips it.
    assert checker_real_lit.check(
        "COL1A1 is absent here, which argues against a fibroblast identity.",
        None, "c2",
    ).ok is False


@pytest.mark.req("ARCH-9.5-04")
def test_an_undetectable_genes_absence_is_gated_too(checker_real_lit):
    """ON the panel and still never measured -- the half the gate used to exempt.

    The pass asked ``is_on_panel`` alone, so all 49 genes that produced no usable
    spatial signal walked through: they ARE on the panel, and the detectability
    frame is the one that refused them. 17.5% of this study's panel sat outside
    the headline rule, including genes a biologist actually asks about in a
    breast tumour.

    ADIPOQ is the anchor because it is unambiguously both: on the 280-gene panel,
    and in the 49-row no-signal set.
    """
    src = SourceIndex()
    assert src.is_on_panel("ADIPOQ"), "anchor must be ON the panel"
    assert not src.is_detectable("ADIPOQ"), "anchor must have no usable signal"

    answer = "ADIPOQ is absent here, which argues against an adipocyte component."
    result = checker_real_lit.check(answer, None, "c1")
    assert result.ok is False, (
        "an on-panel gene that produced NO SIGNAL was reasoned from as if its "
        "absence were evidence, and the gate accepted it"
    )
    v = [x for x in result.violations if x.kind == "panel_absence"]
    assert v and v[0].ref == "ADIPOQ", result.summary()
    assert "no usable signal" in v[0].detail, (
        f"the violation must name the frame that refused it: {v[0].detail}"
    )


def test_no_statistic_is_served_for_a_gene_that_produced_no_signal():
    """The gate and the evidence layer must not disagree about the same gene.

    ``markers_top.csv`` carries a row for every panel gene, so a no-signal gene
    still has ``glm_coef`` 0.0 in the file. Serving that made a number for an
    unmeasured gene groundable here while ``EvidenceItem`` raises
    ``UngroundedStatistic`` for the same gene -- the representability hole the
    four-state lattice exists to close.
    """
    src = SourceIndex()
    assert src.is_on_panel("ADIPOQ") and not src.is_detectable("ADIPOQ")
    assert src.stat_value("ADIPOQ", "glm_coef") is None, (
        "a statistic was served for a gene that produced no usable signal"
    )
    # ...and a real, observed gene still resolves, so this is not a blanket None.
    assert src.stat_value("ERBB2", "glm_coef") is not None


@pytest.mark.req("CLAUDE-HR-01")
def test_off_panel_absence_passes_when_the_sentence_says_it_was_never_measured(
    checker_real_lit,
):
    """The rule's own escape hatch, and the reason the pass is not a word filter.

    By the project's own rule the ONLY legitimate sentence about an off-panel
    entity's absence is one that says it was never measured. Each of these is
    that sentence, and each must render.
    """
    for answer in (
        "COL1A1 is not on the panel, so its absence argues nothing here.",
        "COL1A1 was never measured in this study; nothing follows from its "
        "absence, so the fibroblast rival stays undecidable.",
        "The rival needs COL1A1, which is off-panel -- that is not evidence "
        "against it.",
    ):
        result = checker_real_lit.check(answer, None, "c2")
        assert result.ok is True, f"{answer!r} -> {result.summary()}"


def test_the_caveat_may_sit_in_a_different_clause_from_the_genes(
    checker_real_lit,
):
    """The app's own prose, and the shape clause-level excusal could not see.

    The deterministic fallback writes the caveat ONCE at the end of a sentence
    whose subject is a list of genes. Gene-level excusal looks in the clause
    the gene sits in, finds no caveat there, and refuses -- eight of the app's
    own answers across c2, c7, c8 and c9 were refused by their own gate.

    The discriminator is not where the caveat is, it is whether the sentence
    concludes from the absence or declines to. The last case concludes, so it
    must still be refused.
    """
    for answer in (
        "COL1A1, COL1A2, DCN, VIM, FAP are canonical Stromal markers that are "
        "off-panel, they were never measured in this panel, so their absence "
        "is not evidence against the call.",
        "CLDN5 and CDH5 were never measured on this panel; their absence "
        "argues nothing.",
    ):
        assert checker_real_lit.check(answer, None, "c2").ok is True, answer

    drawn = (
        "COL1A1 was never measured, but CD3D is not detected here, so this "
        "is not T cells."
    )
    assert checker_real_lit.check(drawn, None, "c2").ok is False


def test_measured_absence_used_as_evidence_passes(checker_real_lit):
    """The PRODUCTIVE half. LUM is on the panel and was measured, so a sentence
    that reasons from where it did and did not peak is exactly what the tool is
    for. A gate that refused this would have collapsed the four states into
    two in the one direction that costs the biologist evidence."""
    assert SourceIndex().is_on_panel("LUM")
    answer = (
        "LUM was tested and not reported in this unit, which argues against "
        "a Stromal call."
    )
    result = checker_real_lit.check(answer, None, "c5")
    assert result.ok is True, result.summary()


def test_absence_with_no_conclusion_drawn_from_it_passes(checker_real_lit):
    """Reporting an off-panel gene's absence is not the violation; REASONING
    from it is. The rule is about the inference, not about the observation."""
    answer = "COL1A1 is absent from this panel."
    result = checker_real_lit.check(answer, None, "c2")
    assert result.ok is True, result.summary()


def test_panel_absence_pass_declines_without_a_frame(monkeypatch, checker_real_lit):
    """With no panel loaded every token is 'off-panel', and a gate that refused
    every absence sentence on a frame it does not have is the fail-closed-looks-
    safe failure CLAUDE.md warns about. It declines instead."""
    from panoscope.agent import grounding_check as gc

    monkeypatch.setattr(gc.SourceIndex, "panel_known", lambda self: False)
    answer = "COL1A1 is absent here, which argues against a fibroblast identity."
    assert checker_real_lit.check(answer, None, "c2").ok is True


# --------------------------------------------------------------------------- #
# (e2) Fabricated absence -- the SAME rule, run backwards
#
# evals/sycophancy measured it on real model output: 9 of 27 answers declared an
# ON-PANEL gene never measured, one of them the unit's own driver, and all nine
# cleared the floor. Claiming a measured gene was unmeasured invents a limitation
# the assay does not have, and it reads as caution while doing it. Three cases,
# one violation -- the other two are TRUE and must keep rendering.
# --------------------------------------------------------------------------- #
def test_fabricated_absence_on_a_measured_gene_fails(checker_real_lit):
    """The probe that used to walk straight through, in the other direction.

    CPA3 is on the panel, produced usable signal, and is c9's own driver.
    """
    src = SourceIndex()
    assert src.is_on_panel("CPA3") and src.is_detectable("CPA3")
    answer = "CPA3 was never measured on this panel, so it cannot settle the call."
    result = checker_real_lit.check(answer, None, "c9")
    assert result.ok is False, (
        "A MEASURED GENE WAS DECLARED UNMEASURED AND THE GATE PASSED IT: this "
        "manufactures a limitation the assay does not have."
    )
    v = [x for x in result.violations if x.kind == "fabricated_absence"]
    assert v and v[0].ref == "CPA3", result.summary()


def test_tested_and_not_reported_still_passes(checker_real_lit):
    """The PRODUCTIVE half, again. "Tested and not reported here" is TESTED_ABSENT
    and it is true of CPA3 in another unit; it is not a frame claim at all."""
    answer = (
        "CPA3 was tested and not reported in this unit, which argues against "
        "a Mast_Cells call."
    )
    assert checker_real_lit.check(answer, None, "c5").ok is True


@pytest.mark.req("ARCH-9.5-04")
def test_failed_detectability_is_a_real_frame_and_passes(checker_real_lit):
    """The second gene frame. 49 of the demo's 280 panel genes carry NoSig and
    genuinely WERE unmeasurable here, so a sentence saying so is true. A gate
    that read "on the panel" as "measured" would refuse it."""
    src = SourceIndex()
    assert src.is_on_panel("ADIPOQ") and not src.is_detectable("ADIPOQ")
    for answer in (
        # the frame claim governs ADIPOQ directly -- this is the clause the
        # detectability guard has to answer, and it is TRUE
        "ADIPOQ was never measured in this study.",
        "ADIPOQ is on the panel but produced no usable spatial signal, so it "
        "was never measured here.",
    ):
        assert checker_real_lit.check(answer, None, "c2").ok is True, answer


def test_fabricated_absence_of_a_whole_category_fails(checker_real_lit):
    """The shape the model actually wrote. The clause names no gene, so the
    claim is about a CATEGORY, and this study's own annotation holds Mast_Cells
    markers that are on the panel and detectable."""
    answer = (
        "The measured genes support dendritic identity, but mast cell markers "
        "were not included in the 280-gene panel."
    )
    result = checker_real_lit.check(answer, None, "c8")
    v = [x for x in result.violations if x.kind == "fabricated_absence"]
    assert v and v[0].ref == "Mast_Cells", result.summary()
    assert "CPA3" in v[0].expected, v[0].expected


def test_a_truly_off_panel_category_claim_passes(checker_real_lit):
    """Hepatocytes have no canonical marker set in this study, so "hepatocyte
    markers were not measured" may well be true and the gate does not guess."""
    answer = (
        "Hepatocyte markers were not measured on this panel, so that rival "
        "cannot be evaluated here."
    )
    assert checker_real_lit.check(answer, None, "c1").ok is True


def test_a_frame_claim_does_not_reach_across_a_conjunction(checker_real_lit):
    """Clause scope, and the reason for it. This sentence names four on-panel
    genes the frame claim does not govern; a sentence-scoped pass would refuse a
    true answer on all four."""
    answer = (
        "The spatial GLM supports MYLK, ACTA2, DST and SFRP1 here, but "
        "hepatocyte markers were not measured on this 280-gene panel."
    )
    assert checker_real_lit.check(answer, None, "c4").ok is True


def test_naming_the_genuinely_off_panel_genes_answers_the_category(checker_real_lit):
    """What discriminate() actually ships. The clause names its genes, both
    really off-panel, so the gene reading governs and the category is not
    second-guessed -- otherwise the gate would refuse its own output."""
    src = SourceIndex()
    assert not src.is_on_panel("MS4A2") and not src.is_on_panel("TPSB2")
    answer = (
        "MS4A2, TPSB2 -- canonical Mast_Cells markers -- are off-panel and "
        "were never measured, so they cannot settle it."
    )
    assert checker_real_lit.check(answer, None, "c9").ok is True


def test_naming_an_on_panel_gene_with_no_usable_signal_is_not_a_panel_claim(
        checker_real_lit):
    """The other half of the sentence above, and the one this dataset cannot reach.

    ``discriminate.offpanel_absent`` holds two kinds of never-measured marker since
    the cutover: off the panel, and on the panel with no usable spatial signal (49 of
    280 demo genes). Calling the second off-panel would be a fabricated fact about the
    panel, so the prose branches -- and the gate must accept the new clause, which on
    the demo study nothing but this test exercises: no canonical marker of any of the
    9 annotated types is NoSig here. The guard keeps the case from rotting into a
    fiction if the committed table changes.
    """
    from panoscope.agent import data

    src = SourceIndex()
    for symbol in ("GNLY", "PRF1"):
        assert src.is_on_panel(symbol), f"{symbol} left the panel; pick another case"
        assert str(data.get_marker(symbol)["top_cluster"]) == "NoSig", (
            f"{symbol} now carries spatial signal; pick another case"
        )
    answer = (
        "GNLY, PRF1 -- canonical NK_Cells markers -- are on the panel but "
        "produced no usable spatial signal, so they cannot settle it."
    )
    assert checker_real_lit.check(answer, None, "c3").ok is True


def test_fabricated_absence_pass_declines_without_a_frame(monkeypatch,
                                                          checker_real_lit):
    """Same fail-open-on-a-missing-frame contract as the forward pass."""
    from panoscope.agent import grounding_check as gc

    monkeypatch.setattr(gc.SourceIndex, "panel_known", lambda self: False)
    answer = "CPA3 was never measured on this panel."
    assert checker_real_lit.check(answer, None, "c9").ok is True


# --------------------------------------------------------------------------- #
# (f) The note-scope pass, wired to the REAL store
#
# ``allowed_notes`` used to be a parameter only the tests passed, so "a claim
# used outside its scope" was unenforced on everything the agent actually said.
# These tests call ``check`` exactly the way ``loop._finalize`` does -- text,
# sidecar, cluster -- and pass NO note parameters at all.
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_note_store(tmp_path, monkeypatch):
    """Two cluster-scoped notes in an isolated store, as the memory tools write them."""
    from panoscope.agent import memory
    from panoscope.agent import tools as agent_tools

    base = str(tmp_path / "context")
    prev = agent_tools.memory_base_dir()
    agent_tools.set_memory_base_dir(base)
    try:
        here = memory.create_note(
            claim="c2 is CAF, not generic stromal", scope="cluster", cluster="c2",
            basis="convention", base_dir=base, literature_search=lambda _q: [],
        )
        elsewhere = memory.create_note(
            claim="c5 is a T-cell niche", scope="cluster", cluster="c5",
            basis="convention", base_dir=base, literature_search=lambda _q: [],
        )
        yield here, elsewhere
    finally:
        agent_tools.set_memory_base_dir(prev)


def test_note_scope_fires_in_production_with_no_parameter(real_note_store):
    """THE WIRING TEST. A note scoped to c5, cited while answering about c2, with
    ``allowed_notes`` never passed -- the production call shape."""
    _here, elsewhere = real_note_store
    checker = GroundingChecker(literature_verifier=_resolver_all_real)
    result = checker.check(f"Applying [note:{elsewhere.id}] here.", None, "c2")
    assert result.ok is False, (
        "THE NOTE-SCOPE PASS IS STILL INERT: a c5-scoped note was cited in a c2 "
        "answer through the production call path and nothing fired."
    )
    assert any(
        v.kind == "note" and v.ref == elsewhere.id for v in result.violations
    ), result.summary()


def test_in_scope_note_still_passes_in_production(real_note_store):
    """The other half: the gate must not refuse the note that does apply here."""
    here, _elsewhere = real_note_store
    checker = GroundingChecker(literature_verifier=_resolver_all_real)
    result = checker.check(f"Per lab convention [note:{here.id}], c2 is CAF.",
                           None, "c2")
    assert result.ok is True, result.summary()


def test_ghost_note_fails_against_the_real_store(real_note_store):
    """Existence is read from the same store as scope, so an invented id fails
    without anyone injecting a known-notes set."""
    checker = GroundingChecker(literature_verifier=_resolver_all_real)
    result = checker.check("Per [note:ghost_note], c2 is CAF.", None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "note" and v.ref == "ghost_note"
               for v in result.violations), result.summary()


def test_injected_known_notes_still_own_the_note_universe():
    """A caller that injects ``known_notes`` owns both halves; the gate must not
    mix an injected existence set with a filesystem-derived scope set."""
    checker = GroundingChecker(
        literature_verifier=_resolver_all_real,
        known_notes={"note_c2_caf_real"},
    )
    result = checker.check("Per [note:note_c2_caf_real], c2 is CAF.", None, "c2")
    assert result.ok is True, result.summary()


# --------------------------------------------------------------------------- #
# (g) Claim ids -- resolved through the store's own USE gate
# --------------------------------------------------------------------------- #
def _claim_store():
    from panoscope.claims.store import ClaimStore
    from panoscope.types.claim import Authority, Claim
    from panoscope.types.subject import Entity, EntityKind, subject

    base = dict(
        subject=subject("gene", g=Entity(EntityKind.GENE, "LUM")),
        predicate="marks", object="Stromal", authority=Authority.CURATED,
        source="pmid:12345678",
    )
    return ClaimStore((
        Claim(id="curated:marks:LUM:Stromal", scope="dataset:here", **base),
        Claim(id="curated:marks:LUM:Other", scope="dataset:elsewhere", **base),
        Claim(id="curated:marks:LUM:Gone", scope="dataset:here",
              valid="retracted", **base),
    ))


def _claim_checker():
    return GroundingChecker(
        literature_verifier=_resolver_all_real,
        claims=_claim_store(),
        claim_context="dataset:here",
    )


def test_claim_id_resolves(checker_real_lit):
    answer = "LUM is a canonical Stromal marker [claim:curated:marks:LUM:Stromal]."
    assert _claim_checker().check(answer, None, "c2").ok is True


def test_fabricated_claim_id_fails():
    result = _claim_checker().check("As recorded [claim:no_such_claim].", None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "claim" and v.ref == "no_such_claim"
               for v in result.violations), result.summary()


def test_out_of_scope_claim_id_fails():
    """The store's own fail-closed scope gate, reached from the prose gate."""
    result = _claim_checker().check(
        "As recorded [claim:curated:marks:LUM:Other].", None, "c2"
    )
    assert result.ok is False, result.summary()
    v = [x for x in result.violations if x.kind == "claim"]
    assert v and "scope" in v[0].detail, result.summary()


def test_retracted_claim_id_fails():
    result = _claim_checker().check(
        "As recorded [claim:curated:marks:LUM:Gone].", None, "c2"
    )
    assert result.ok is False, result.summary()
    v = [x for x in result.violations if x.kind == "claim"]
    assert v and "retracted" in v[0].detail, result.summary()


def test_claim_citation_with_no_store_fails(checker_real_lit):
    """Fail closed: a citation the gate cannot resolve is not a citation."""
    result = checker_real_lit.check("As recorded [claim:anything].", None, "c2")
    assert result.ok is False, result.summary()
    assert any(v.kind == "claim" for v in result.violations), result.summary()


# --------------------------------------------------------------------------- #
# (h) Reference-graph edges
# --------------------------------------------------------------------------- #
def _edge_graph():
    from panoscope.reference.graph import Edge, Graph
    from panoscope.types.subject import Entity, EntityKind

    return Graph(edges=(Edge(
        subject=Entity(EntityKind.GENE, "LUM"), predicate="marks",
        object=Entity(EntityKind.CELL_TYPE, "Stromal"),
        source="panglao@2020", citation="pmid:12345678", licence="CC-BY",
    ),))


def test_edge_reference_resolves():
    checker = GroundingChecker(literature_verifier=_resolver_all_real,
                               graph=_edge_graph())
    answer = "Reference data lists [edge:gene:LUM marks cell_type:Stromal]."
    assert checker.check(answer, None, "c2").ok is True, answer


def test_fabricated_edge_reference_fails():
    checker = GroundingChecker(literature_verifier=_resolver_all_real,
                               graph=_edge_graph())
    result = checker.check(
        "Reference data lists [edge:gene:LUM marks cell_type:Tumor].", None, "c2"
    )
    assert result.ok is False, result.summary()
    assert any(v.kind == "edge" for v in result.violations), result.summary()


# --------------------------------------------------------------------------- #
# (i) ``record.prose`` -- gated as the generated layer, by NAME READ OFF THE TYPE
# --------------------------------------------------------------------------- #
def _record(prose=None):
    from panoscope.types.outputs import TaskKind
    from panoscope.types.record import InterpretationRecord, ReadingRecord
    from panoscope.types.trace import Decision

    return InterpretationRecord(
        unit="c2", task_kind=TaskKind.ASSIGNMENT,
        reading=ReadingRecord(
            path="results/markers.csv", sha256="0" * 64,
            statistic_column="glm_coef", unit_column="cluster_id",
            unit_kind="cluster", unit_detection_because=("named column",),
            rows_in=10, rows_dropped=0, drop_reasons=(),
            entities_outside_frame=("COL1A1",), entities_not_in_reference=(),
            frames=(), clustering_mode="independent",
        ),
        decisions=(Decision("band:from_coef", "High"),),
        claims_used=(), undecidable=(), spec_version="0.1", prose=prose,
    )


def test_record_with_no_prose_passes(checker_real_lit):
    """``prose=None`` is a complete, valid record -- the one the demo falls back
    to -- and there is nothing generated in it to gate."""
    assert checker_real_lit.check_record(_record()).ok is True


@pytest.mark.req("ARCH-17-01", "IMPL-9-02")
def test_record_prose_is_gated(checker_real_lit):
    result = checker_real_lit.check_record(
        _record("c2 is Stromal, driven by LUM glm 99.9.")
    )
    assert result.ok is False, result.summary()
    assert any(v.kind == "number" and v.ref == "LUM" for v in result.violations)
    assert all(v.detail.startswith("prose: ") for v in result.violations), (
        "a violation must name the generated field it came from"
    )


def test_record_prose_panel_absence_is_gated(checker_real_lit):
    """The headline rule reaches the generated layer too."""
    result = checker_real_lit.check_record(
        _record("COL1A1 is absent, which argues against a fibroblast call.")
    )
    assert result.ok is False, result.summary()
    assert any(v.kind == "panel_absence" for v in result.violations)


def test_record_gate_reads_generated_fields_off_the_type(monkeypatch,
                                                         checker_real_lit):
    """THE POINT of :data:`GENERATED_FIELDS`: a gate that hardcoded ``"prose"``
    silently stops covering the second generated field the day one is added.
    Add one here and the gate must cover it with no edit to the gate."""
    from panoscope.agent import grounding_check as gc

    monkeypatch.setattr(gc, "GENERATED_FIELDS", ("prose", "caption"))

    class _TwoGenerated:
        unit = "c2"
        prose = "LUM glm 17.998 drives the c2 Stromal call."      # clean
        caption = "LUM glm 99.9."                                  # poisoned

    result = checker_real_lit.check_record(_TwoGenerated())
    assert result.ok is False, result.summary()
    assert any(v.detail.startswith("caption: ") for v in result.violations), (
        "the gate is still hardcoded to 'prose' and did not read the type"
    )


def test_record_deterministic_fields_are_not_gated(checker_real_lit):
    """Only the generated layer is checked. ``entities_outside_frame`` holds
    COL1A1 -- an off-panel gene name sitting in a DETERMINISTIC field -- and the
    gate must not read it as something a model said."""
    assert checker_real_lit.check_record(_record()).ok is True


# --------------------------------------------------------------------------- #
# Every statistic is grounded, not just the first one per gene
# --------------------------------------------------------------------------- #
def test_a_second_statistic_on_the_same_gene_is_checked(checker_real_lit):
    """The app's OWN opening format, and the number in it that nothing checked.

    The extractor matched gene -> stat -> number in a single pattern, and
    ``finditer`` consumes the gene, so in "LUM (glm_coef 18.00, pearson 0.91)"
    only the glm_coef was ever extracted. Every statistic after the first, for
    every gene, in every answer, was compared against nothing -- a fabricated
    ``pearson 0.99`` passed, and so did ``pearson 42.0``, which is outside the
    coefficient's possible range.
    """
    real = SourceIndex().stat_value("LUM", "pearson")
    assert real is not None and abs(real - 0.91) < 0.01, real

    good = "c2 reads as Stromal, driven by LUM (glm_coef 18.00, pearson 0.91)."
    assert checker_real_lit.check(good, None, "c2").ok is True

    for bad in (
        "c2 reads as Stromal, driven by LUM (glm_coef 18.00, pearson 0.99).",
        "c2 reads as Stromal, driven by LUM (glm_coef 18.00, pearson 42.0).",
    ):
        result = checker_real_lit.check(bad, None, "c2")
        assert result.ok is False, f"{bad!r} passed the floor"
        assert any(v.kind == "number" for v in result.violations), result.summary()


def test_a_statistic_stated_in_ordinary_english_is_checked(checker_real_lit):
    """The register the skill contract tells the model to write in.

    The gap between gene and stat allowed only punctuation and of/at/is/was, so
    "ERBB2 has a glm coefficient of 99.9" extracted NOTHING: a fabricated number
    and, in the second case, a fabricated gene with it.
    """
    for bad in (
        "c1 is Tumor: ERBB2 has a glm coefficient of 99.9 here.",
        "c1 is Tumor: ERBB2 shows a pearson correlation of 0.99 here.",
        "c1 is Tumor: NOTAGENE7 shows a pearson correlation of 0.99 here.",
    ):
        assert checker_real_lit.check(bad, None, "c1").ok is False, f"{bad!r} passed"

    # ...and the true statement in the same register still renders.
    real = SourceIndex().stat_value("ERBB2", "glm_coef")
    assert real is not None
    ok = f"c1 is Tumor: ERBB2 has a glm coefficient of {real:.2f} here."
    assert checker_real_lit.check(ok, None, "c1").ok is True


def test_a_stat_binds_to_its_own_gene_not_the_previous_one(checker_real_lit):
    """Binding by position must not reach across a gene into the wrong subject."""
    from panoscope.agent.grounding_check import Extractor, _canonical_stat

    ex = Extractor.__new__(Extractor)
    got = [(m["gene"], _canonical_stat(m["stat"]), m["value"])
           for m in ex._stat_matches("LUM glm_coef 18.00 and POSTN pearson 0.80")]
    assert got == [("LUM", "glm_coef", "18.00"), ("POSTN", "pearson", "0.80")], got


def test_the_floor_does_not_refuse_the_apps_own_answers(checker_real_lit):
    """The other direction, and the one that matters more.

    "An invariant that fires on correct behaviour is worse than none." Widening
    the extractor is exactly the change that risks it, so the refusal rate over
    every deterministic answer the product can produce is asserted at zero
    rather than assumed.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import fallback as fb

    refused, total = [], 0
    for unit in cfg.CLUSTER_ORDER:
        for q in ("", "what supports this call?", "what was never measured?",
                  "could this be something else?", "what about the missing marker?",
                  "what would settle it?", "how confident are you?"):
            resp = fb.fallback_opening(unit) if not q else fb.fallback_answer(q, unit)
            if resp is None:
                continue
            total += 1
            result = checker_real_lit.check(resp.text, resp.grounding, unit)
            if not result.ok:
                refused.append(f"{unit} {q!r}: {result.summary()}")

    assert total >= 40, f"only {total} answers exercised"
    assert not refused, (
        f"the floor refuses {len(refused)}/{total} of the app's own answers:\n  "
        + "\n  ".join(refused))


def test_the_panel_absence_rule_survives_a_full_stop(checker_real_lit):
    """The headline rule, bypassed by a sentence break.

    The absence and the inference had to sit in ONE sentence, so "COL1A1 is not
    detected here, so this is not fibroblasts" was refused while "COL1A1 is not
    detected here. So this is not fibroblasts." passed clean -- as did putting
    the inference in the next paragraph. COL1A1 is off-panel for this study, so
    every one of these reasons from an absence the measurement could not have
    produced.
    """
    for bad in (
        "COL1A1 is not detected here, so this is not fibroblasts.",
        "COL1A1 is not detected here. So this is not fibroblasts.",
        "There is no COL1A1 signal in c2. That argues against a fibroblast call.",
        "COL1A1 is absent from c2.\n\nThat rules out fibroblasts.",
    ):
        assert checker_real_lit.check(bad, None, "c2").ok is False, f"{bad!r} passed"

    # Reach is ONE sentence, and an absence nobody reasons from is not a claim.
    assert checker_real_lit.check("COL1A1 is absent from this panel.", None, "c2").ok
    assert checker_real_lit.check(
        "COL1A1 was never measured here, so its absence argues nothing.", None, "c2"
    ).ok
