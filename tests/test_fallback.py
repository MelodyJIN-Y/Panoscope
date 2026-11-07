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
    canonical markers, so it defers to the generic fallback.

    Anchored on **c6** and read off the record rather than hardcoded. It used to
    name c1, on the premise that c1 had no off-panel notes -- which was true only
    because ``offpanel_canonical`` was empty for eight of nine units in the
    committed annotation. Deriving it gave c1 its real note (canonical KRT18 is
    off-panel), and this test went red for the right reason. c6 B_Cells is the one
    unit whose canonical set really is entirely on-panel.
    """
    from panoscope.agent import verdict as V

    empty = [v.cluster for v in V.all_verdicts() if not v.offpanel_notes]
    assert "c6" in empty, (
        f"c6 was the unit with no off-panel canonical markers; now {empty}. "
        "Re-anchor on one that is, rather than relaxing the assertion."
    )
    assert F.fallback_answer("what about the missing marker?", "c6") is None


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
# match — canned answer first, else the generic grounded fallback
# --------------------------------------------------------------------------- #
def test_match_prefers_canned_then_generic(checker):
    # A recognized question routes to the canned answer.
    canned = F.match("could this be a doublet?", "c1")
    assert canned.used_fallback is True
    assert "doublet" in canned.text.lower()

    # An unrecognized question falls through to the generic grounded fallback,
    # never None.
    generic = F.match("random unrelated text", "c1")
    assert isinstance(generic, AgentResponse)
    assert checker.check(generic.text, generic.grounding, "c1").ok is True


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


def test_a_rival_named_the_natural_way_is_actually_read():
    """"X, not Y" and "could this be X rather than Y" are how the question is asked.

    Two shapes defeated the rival pattern and both ended in the same wrong answer
    -- "the alternative equals the call; nothing to discriminate" -- which asserts
    the biologist asked about the call they were disputing.

    * a three-letter rival: the pattern needed four characters, so the demo's own
      signature pushback, c2 Stromal -> CAF, never matched;
    * a word between the modal and "be": "could THIS be Hepatocytes rather than
      Tumor" needed ``could be`` adjacent.
    """
    assert F._contrasted_rival("c2 is a CAF, not Stromal") == "CAF"
    assert F._contrasted_rival("could this be Hepatocytes rather than Tumor?") \
        == "Hepatocytes"
    assert F._contrasted_rival("might it be Podocytes instead of Tumor") == "Podocytes"
    # ...and ordinary prose still names no rival.
    assert F._contrasted_rival("this could be important") is None
    assert F._contrasted_rival("it is not the one") is None


def test_an_unrecognised_rival_is_declined_by_name_not_mistaken_for_the_call():
    """The honest answer exists in ``discriminate``; it just was not reachable.

    ``discriminate`` already declines an unknown rival by name. It is reached only
    when an alternative is supplied, and when the pattern missed, the synonym
    branch supplied the CALL instead -- so a question about Hepatocytes was
    answered "the alternative equals the call (Tumor); nothing to discriminate".
    """
    r = F.fallback_answer("could this be Hepatocytes rather than Tumor?", "c1")
    assert r is not None, "the settle intent did not fire"
    assert "equals the call" not in r.text, (
        f"a rival the panel cannot test was read as the call: {r.text[:200]}"
    )
    assert "Hepatocytes" in r.text, f"the rival is not named: {r.text[:200]}"


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


def test_verify_is_the_same_whatever_the_model_wrote():
    """The BEHAVIOURAL half of the rule above, which the name scan cannot cover.

    The scan asserts one removed function is not called back. It stays green under
    the obvious regression: re-add the prose test under any other name and OR it in
    (``verify = _prose_wants_recheck(text) or _cluster_verify_default(cluster)``).
    That mutation was run against the whole suite and 2190 tests passed -- so the
    proposition "``_assemble`` uses the rubric and NOTHING else" had no coverage at
    all, on a rule CLAUDE.md states twice.

    So: one unit, two prose strings a substring test would split on, and the flag
    must not move. The second is the string that broke the original implementation
    -- a substring test cannot see negation, so "no need to re-check this" set
    verify TRUE.
    """
    from panoscope.agent import loop

    agent = loop.PanoscopeAgent.__new__(loop.PanoscopeAgent)  # no client, no network
    unit = "c1"
    rubric = loop._cluster_verify_default(unit)

    says_no = agent._assemble("The call is solid; no need to re-check this.",
                              loop._Ledger(), unit)
    says_yes = agent._assemble("Uncertain on this one. Please re-check this.",
                               loop._Ledger(), unit)

    assert says_no.verify == says_yes.verify, (
        "verify moved with the model's prose; the rubric is no longer its only input"
    )
    assert says_no.verify == rubric, (
        f"verify={says_no.verify} but the rubric says {rubric} for {unit}"
    )


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


# --------------------------------------------------------------------------- #
# The biologist's override reaches the chat pane
# --------------------------------------------------------------------------- #
def test_the_chat_pane_speaks_the_biologists_call_not_the_replaced_one() -> None:
    """The composition happened in ``ui/data_access``; the prose is built below it.

    So the board, the marker header and the exported overview all said CAF while
    five of c2's seven chat answers went on saying Stromal -- the opening
    interpretation, the rivals answer, the panel-absence answer, the
    what-would-settle-it answer and the confidence answer. That is the silent
    overrule the memory rule forbids, committed by the layering.

    Both directions are asserted: the call must be adopted, and it must not be
    adopted SILENTLY -- an answer that simply swapped the label would hide the
    same disagreement from the other side.
    """
    from panoscope.agent import annotation, config as cfg
    from panoscope.agent import memory

    before = F.fallback_opening("c2").text
    assert "Stromal" in before, "this test needs c2 to start out called Stromal"
    assert annotation.canonical_markers("CAF") == (), (
        "this test needs a call the annotation does NOT name; pick another"
    )

    memory.create_note(
        claim="c2 is CAF here", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", subject_lineage="Stromal",
        subject_category="Mesenchymal", note_type="celltype_override",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
    )

    opening = F.fallback_opening("c2").text
    assert opening.startswith("c2 reads as CAF"), opening
    assert "replacing the computed Stromal" in opening, (
        f"the call was adopted in silence: {opening}"
    )

    answers = {
        q: F.fallback_answer(q, "c2")
        for q in ("could this be something else?", "what about the missing marker?",
                  "what would settle it?", "how confident are you?")
    }
    for q, r in answers.items():
        assert r is not None, q
        # Canonicity is an ANNOTATION claim and the annotation does not name CAF,
        # so naming these genes canonical CAF markers would be the fabrication
        # the panel-absence rule exists to prevent.
        assert "canonical CAF" not in r.text, f"{q}: {r.text}"
        if "Stromal" in r.text:
            assert "the computed call" in r.text, (
                f"{q} argues the replaced call without saying so: {r.text}"
            )

    assert "c2 CAF: Very High confidence" in answers["how confident are you?"].text


def test_a_cluster_with_no_override_reads_exactly_as_before() -> None:
    """The composition costs an unedited cluster nothing. c5's canonical T_Cells
    phrasing is the one the annotation CAN source, and it has to survive the
    branch added for the one it cannot."""
    text = F.fallback_opening("c5").text
    assert "c5 reads as T_Cells, Medium-High confidence" in text, text
    assert "your call" not in text
    assert ("CD3D is a canonical T Cells marker that is off-panel "
            "(never measured here)") in text, text
