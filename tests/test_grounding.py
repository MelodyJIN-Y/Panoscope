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

from agent.grounding_check import (
    Extractor,
    GroundingChecker,
    GroundingResult,
    SourceIndex,
)
from agent.types import GroundingSidecar


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
