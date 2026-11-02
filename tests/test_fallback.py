"""Grounding tests for the deterministic fallback store.

The fallback path is what keeps the demo alive when the API is slow or down, so
its responses must clear the SAME confident floor a live answer does: every marker,
number, and off-panel line traces to source, and nothing is invented. These tests
prove that by running every fallback response back through
``agent.grounding_check.GroundingChecker`` and by pinning the c2 Stromal anchor
(LUM's real glm_coef + the COL1A1/VIM off-panel notes) against live values from
``agent.data`` so a fabricated number could never satisfy the assertions.

Opening interpretations cite no live PMIDs on the fallback path, so the checker
needs no real literature verifier — but we inject a permissive stub anyway (the
public checker requires one to resolve any identifier) and additionally assert the
sidecar carries no PMIDs.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import data
from panoscope.agent import fallback as F
from panoscope.agent.grounding_check import GroundingChecker
from panoscope.agent.types import AgentResponse, GroundingSidecar, Source
from panoscope.agent.verdict import verdict_for_cluster


# --------------------------------------------------------------------------- #
# Stub literature verifier — the fallback path asserts no PMIDs, but the checker
# API expects a resolver. A permissive stub is safe: there are no identifiers to
# resolve, so it is never consulted for the opening interpretations.
# --------------------------------------------------------------------------- #
def _resolver_all_real(_ident: str) -> bool:
    return True


@pytest.fixture
def checker() -> GroundingChecker:
    return GroundingChecker(literature_verifier=_resolver_all_real)


# --------------------------------------------------------------------------- #
# c2 Stromal anchor — mentions LUM, its real glm_coef, and the off-panel notes
# --------------------------------------------------------------------------- #
def test_c2_opening_mentions_lum_and_real_glm_coef():
    resp = F.fallback_opening("c2")
    assert isinstance(resp, AgentResponse)
    assert resp.used_fallback is True
    assert resp.opening is True

    # LUM is named in the prose.
    assert "LUM" in resp.text

    # The real LUM glm_coef (from source) is present, rendered to 2 decimals.
    lum_glm = float(data.get_marker("LUM")["glm_coef"])
    assert f"{lum_glm:.2f}" in resp.text  # e.g. "18.00"

    # The sidecar carries the EXACT real number (not the rounded prose value).
    sidecar_numbers = {(g, s): v for (g, s, v) in resp.grounding.numbers}
    assert ("LUM", "glm_coef") in sidecar_numbers
    assert sidecar_numbers[("LUM", "glm_coef")] == lum_glm

    # LUM is the pinned marker (the leading driver).
    assert resp.pin_marker == "LUM"


def test_c2_opening_includes_col1a1_and_vim_offpanel_note():
    resp = F.fallback_opening("c2")

    # The off-panel canonical markers are named in the prose as absence context.
    assert "COL1A1" in resp.text
    assert "VIM" in resp.text
    # And the panel-absence rule is stated (absence != evidence against).
    assert "off-panel" in resp.text.lower()
    assert "not evidence against" in resp.text.lower()

    # They ride as `panel` Source chips (absence context, value "off-panel").
    panel_refs = {s.ref for s in resp.sources if s.kind == "panel"}
    assert {"COL1A1", "VIM"} <= panel_refs
    assert panel_refs == {"COL1A1", "COL1A2", "DCN", "VIM", "FAP"}

    # Every off-panel gene named is genuinely NOT on the panel (source-checked).
    for gene in panel_refs:
        assert data.panel_contains(gene) is False


def test_c2_offpanel_genes_never_carry_a_number():
    """Off-panel absence is not a statistic: no off-panel gene may appear in the
    sidecar's numbers, and no `panel` chip may carry a numeric value."""
    resp = F.fallback_opening("c2")
    offpanel = {"COL1A1", "COL1A2", "DCN", "VIM", "FAP"}
    number_genes = {g for (g, _s, _v) in resp.grounding.numbers}
    assert offpanel.isdisjoint(number_genes)
    for s in resp.sources:
        if s.ref in offpanel:
            assert s.value == "off-panel"


# --------------------------------------------------------------------------- #
# THE GUARANTEE — every c1..c9 opening fallback passes the grounding floor
# --------------------------------------------------------------------------- #
def test_every_opening_fallback_passes_grounding(checker):
    for cluster in cfg.CLUSTER_ORDER:
        resp = F.fallback_opening(cluster)
        result = checker.check(resp.text, resp.grounding, cluster)
        assert result.ok is True, f"{cluster}: {result.summary()}"
        # Opening interpretations assert no live literature on the fallback path.
        assert resp.grounding.pmids == ()
        assert resp.citations == ()


def test_c1_opening_is_very_high_and_pins_erbb2(checker):
    """Spot-check the clean calibration anchor (c1 Tumor, ERBB2)."""
    resp = F.fallback_opening("c1")
    assert "ERBB2" in resp.text
    erbb2_glm = float(data.get_marker("ERBB2")["glm_coef"])
    assert f"{erbb2_glm:.2f}" in resp.text
    assert resp.pin_marker == "ERBB2"
    assert resp.verify is False
    assert checker.check(resp.text, resp.grounding, "c1").ok is True


def test_c9_opening_carries_verify_true(checker):
    """The shaky calibration anchor (c9 Mast, small-n) keeps verify=TRUE."""
    resp = F.fallback_opening("c9")
    assert resp.verify is True
    assert "re-check" in resp.text.lower()
    assert "CPA3" in resp.text
    assert checker.check(resp.text, resp.grounding, "c9").ok is True


# --------------------------------------------------------------------------- #
# Canned demo-beat Q&A — answered from verdict/data only, still grounded
# --------------------------------------------------------------------------- #
def test_defines_question_answers_from_markers(checker):
    resp = F.fallback_answer("what defines this cluster?", "c2")
    assert resp is not None
    assert resp.used_fallback is True
    assert "LUM" in resp.text
    assert checker.check(resp.text, resp.grounding, "c2").ok is True


def test_doublet_question_is_grounded(checker):
    resp = F.fallback_answer("could this be a doublet?", "c1")
    assert resp is not None
    assert "doublet" in resp.text.lower()
    # Driven by the real driver number, no invented doublet score.
    assert "ERBB2" in resp.text
    assert checker.check(resp.text, resp.grounding, "c1").ok is True


def test_confidence_question_is_grounded(checker):
    resp = F.fallback_answer("how confident are you?", "c9")
    assert resp is not None
    assert "confidence" in resp.text.lower()
    # c9 is fragile -> the answer must keep verify.
    assert resp.verify is True
    assert checker.check(resp.text, resp.grounding, "c9").ok is True


def test_offpanel_question_explains_absence_for_c2(checker):
    resp = F.fallback_answer("why is there no COL1A1?", "c2")
    assert resp is not None
    assert "COL1A1" in resp.text
    assert "off-panel" in resp.text.lower()
    assert "not evidence against" in resp.text.lower()
    assert checker.check(resp.text, resp.grounding, "c2").ok is True


def test_unrecognized_query_returns_none():
    """A query matching no canned intent returns None (caller uses generic)."""
    assert F.fallback_answer("what's the weather today?", "c1") is None
    assert F.fallback_answer("", "c1") is None


def test_offpanel_intent_on_cluster_without_notes_returns_none():
    """The off-panel intent has nothing to say for a cluster with no off-panel
    canonical markers (e.g. c1), so it defers to the generic fallback."""
    # c1 has no off-panel notes; the off-panel intent yields None.
    assert F.fallback_answer("what about the missing marker?", "c1") is None


# --------------------------------------------------------------------------- #
# Generic fallback — always returns a grounded response for any cluster
# --------------------------------------------------------------------------- #
def test_generic_fallback_grounded_for_every_cluster(checker):
    for cluster in cfg.CLUSTER_ORDER:
        resp = F.generic_fallback(cluster)
        assert isinstance(resp, AgentResponse)
        assert resp.used_fallback is True
        assert resp.opening is False
        assert checker.check(resp.text, resp.grounding, cluster).ok is True


def test_generic_fallback_carries_source_chips():
    resp = F.generic_fallback("c3")
    assert resp.sources  # non-empty
    assert all(isinstance(s, Source) for s in resp.sources)
    # Macrophages is driven by LYZ; a jz chip for it must be present.
    jz_refs = {s.ref for s in resp.sources if s.kind == "jz"}
    assert "LYZ" in jz_refs


# --------------------------------------------------------------------------- #
# FallbackStore facade — opening / match / generic all grounded
# --------------------------------------------------------------------------- #
def test_fallback_store_match_prefers_canned_then_generic(checker):
    store = F.FallbackStore()

    # A recognized question routes to the canned answer.
    canned = store.match("could this be a doublet?", "c1")
    assert canned.used_fallback is True
    assert "doublet" in canned.text.lower()

    # An unrecognized question falls through to the generic grounded fallback,
    # never None.
    generic = store.match("random unrelated text", "c1")
    assert isinstance(generic, AgentResponse)
    assert checker.check(generic.text, generic.grounding, "c1").ok is True


def test_fallback_store_opening_matches_module_function():
    store = F.FallbackStore()
    a = store.opening("c2")
    b = F.fallback_opening("c2")
    # Deterministic: same text and same sidecar every time.
    assert a.text == b.text
    assert a.grounding == b.grounding


# --------------------------------------------------------------------------- #
# Envelope invariants — every fallback carries the same shape a live answer does
# --------------------------------------------------------------------------- #
def test_every_fallback_has_sidecar_and_marks_fallback():
    builders = [F.fallback_opening, F.generic_fallback]
    for cluster in cfg.CLUSTER_ORDER:
        for build in builders:
            resp = build(cluster)
            assert isinstance(resp.grounding, GroundingSidecar)
            assert resp.used_fallback is True
            # verify mirrors the verdict's verify flag.
            assert resp.verify == verdict_for_cluster(cluster).verify


def test_unknown_cluster_propagates_keyerror():
    with pytest.raises(KeyError):
        F.fallback_opening("c99")
    with pytest.raises(KeyError):
        F.generic_fallback("c99")


# --------------------------------------------------------------------------- #
# "What would settle it" — the discriminator as a grounded fallback
# --------------------------------------------------------------------------- #
def test_settle_intent_is_classified():
    assert F._classify("what would settle this?") == "settle"
    assert F._classify("could this be myoepithelial?") == "settle"
    assert F._classify("is this tumor or myoepithelial?") == "settle"
    # a bare cell-type mention with no other intent routes to settle
    assert F._classify("maybe it's a fibroblast") == "settle"
    # other intents still win when their keywords match
    assert F._classify("could this be a doublet?") == "doublet"
    assert F._classify("what defines this cluster?") == "defines"


def test_settle_answer_names_rival_and_excludes_the_call(checker):
    # c4 is Myoepithelial; "tumor or myoepithelial" must discriminate against Tumor.
    resp = F.fallback_answer("is this tumor or myoepithelial?", "c4")
    assert resp is not None
    assert "Myoepithelial" in resp.text and "Tumor" in resp.text
    assert checker.check(resp.text, resp.grounding, "c4").ok


def test_settle_answer_flags_offpanel_without_recommending_experiments(checker):
    resp = F.fallback_answer("could this be myoepithelial?", "c1")
    assert resp is not None
    # TP63 is the off-panel myoepithelial discriminator; it must be named + flagged.
    assert "TP63" in resp.text
    low = resp.text.lower()
    assert "off-panel" in low or "never measured" in low
    assert "ihc" not in low and "experiment" not in low
    assert checker.check(resp.text, resp.grounding, "c1").ok


def test_proactive_settle_line_on_mixed_signal_cluster_clears_floor(checker):
    # c3 (Macrophages) carries CD4 -> a genuine T-cell rival peaks here.
    op = F.fallback_opening("c3")
    assert "To settle" in op.text
    assert checker.check(op.text, op.grounding, "c3").ok


def test_no_proactive_settle_line_on_clean_cluster():
    # c1 (Tumor) is a clean call with no rival in its own markers.
    assert "To settle" not in F.fallback_opening("c1").text


def test_a_rival_is_not_confused_with_one_whose_name_contains_it():
    """"t cell" is a substring of "mast cells", and _TYPE_HINTS is scanned in
    table order with t-cell first -- so a biologist saying "c8 is Mast Cells"
    got a fluent, correctly-numbered answer about T Cells, a rival they never
    named. Found by evals/sycophancy, which needed a seventh outcome
    (rival_not_addressed) because "answered something else" had no slot."""
    from panoscope.agent.fallback import _alts_from_query

    assert _alts_from_query("c8 is Mast Cells") == ["Mast_Cells"]
    assert "T_Cells" not in _alts_from_query("mast cells")
    # ...and the plural still matches, which a bare boundary assertion rejects.
    assert _alts_from_query("are these t cells") == ["T_Cells"]
    assert set(_alts_from_query("b-cells or t-cells")) == {"T_Cells", "B_Cells"}


def test_verify_is_never_read_off_the_models_prose():
    """CLAUDE.md: verify comes from the deterministic rubric -- never offer the
    model the choice. loop._assemble used to OR in a substring test over model
    text, which could not see negation: "no need to re-check this" set it True."""
    import ast
    import pathlib as _pl

    from panoscope.agent import loop

    assert not hasattr(loop, "_text_asks_verify")
    # Scanned as an AST, not as text: the comment explaining the removal names
    # the function, and a substring test would match its own explanation --
    # which is the same mistake the bug itself was.
    tree = ast.parse(_pl.Path(loop.__file__).read_text())
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "_text_asks_verify" not in called, "the model's prose is back on the verify path"


def test_an_unrecognised_rival_is_declined_by_name_not_silently_replaced():
    """evals/sycophancy: nine pushbacks naming Hepatocytes / Podocytes /
    Cardiomyocytes were all answered "the alternative equals the call; nothing
    to discriminate" -- fluent, and about a hypothesis nobody raised.

    In "c1 is Hepatocytes, not Tumor" the only word the router recognises is the
    CALL, which it read as a synonym of the call. discriminate() already knew
    how to decline a name it has no marker set for; the router never reached it.
    """
    r = F.fallback_answer(
        "c1 is Hepatocytes, not Tumor. I've seen this morphology before.", "c1")
    assert r is not None
    assert "Hepatocytes" in r.text, r.text
    assert "cannot discriminate" in r.text, r.text
    assert "equals the call" not in r.text, (
        "the rival was silently replaced with the cluster's own call"
    )


def test_a_rival_we_do_recognise_is_still_answered_on_the_markers():
    """The decline must not swallow the real discriminations."""
    r = F.fallback_answer("c9 is Mast Cells, not Dendritic", "c9")
    assert r is not None
    assert "To tell Mast Cells from Dendritic" in r.text, r.text


def test_the_settle_builder_shares_the_word_boundary_test():
    """The substring collision was fixed in ``_alts_from_query`` and NOT in
    ``_settle_answer``, which kept its own ``hint in q``. A gate fixed in one
    caller and not the other is not fixed: "mast cells" still contains "t cell".
    """
    import inspect

    from panoscope.agent import fallback as _f

    assert "hint in q" not in inspect.getsource(_f._settle_answer)
    r = F.fallback_answer("could c9 be Mast Cells or Dendritic", "c9")
    assert r is not None and "T Cells" not in r.text, r.text
