"""The second-opinion skeptic: grounded, calibrated, and never fabricating.

The skeptic must (a) let a clean call withstand challenge, (b) flag a fragile call
for re-check, and (c) quote only numbers that trace to jazzPanda output.
"""
from __future__ import annotations

import pytest

from panoscope.agent import skeptic
from panoscope.agent.grounding_check import GroundingChecker


def test_clean_call_withstands_challenge():
    # c2 Stromal is Very High on LUM/POSTN — a clean call with no competing signal.
    report = skeptic.second_opinion("c2")
    assert report.survives is True
    assert report.call == "Stromal"
    assert "withstands challenge" in report.verdict_line
    # no challenge may mark the call as weakened
    assert all(c.effect != "weakens" for c in report.challenges)


def test_fragile_call_is_flagged_for_recheck():
    # c9 Mast_Cells rests on a single marker (CPA3) and is flagged verify — fragile.
    report = skeptic.second_opinion("c9")
    assert report.survives is False
    assert any(c.effect == "weakens" for c in report.challenges)
    assert "re-check" in report.verdict_line


def test_every_number_in_the_report_is_grounded():
    # The skeptic's prose must clear the SAME grounding gate the agent loop uses:
    # every gene+stat+number it quotes has to resolve against jazzPanda output.
    checker = GroundingChecker(literature_verifier=lambda _ident: True)
    for cluster in ("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"):
        report = skeptic.second_opinion(cluster)
        prose = skeptic.skeptic_summary(report)
        result = checker.check(prose)
        assert result.ok, f"{cluster} skeptic prose failed grounding: {result.summary()}"


def test_every_bucket_is_reported_not_just_the_first():
    """The three competing-hypothesis branches are independent, not an elif chain.

    c3/c4/c7 are the only clusters that derive a rival, and on all three a rival
    marker peaks here — so the old chain stopped at the first branch and the
    biologist never heard that the rival's other panel markers localise elsewhere
    (which argues FOR the call) or that one was never measured at all.
    """
    expected = {
        "c3": ("CD4", "CD3E", "CD3D"),   # peaks here / localises to c5 / off-panel
        "c4": ("KIT", "CPA3", "TPSB2"),
        "c7": ("IL3RA", "LILRA4", "IRF7"),
    }
    for cluster, (here, elsewhere, never) in expected.items():
        report = skeptic.second_opinion(cluster)
        assert {c.kind for c in report.challenges} >= {"competing_signal", "checked"}, cluster
        assert {c.effect for c in report.challenges} >= {"weakens", "supports", "neutral"}, cluster
        prose = skeptic.skeptic_summary(report)
        for gene in (here, elsewhere, never):
            assert gene in prose, f"{cluster}: {gene} dropped from the second opinion"
        # the added branches carry no "weakens", so the verdict itself is untouched
        assert report.survives is False
        assert "re-check" in report.verdict_line


def test_unknown_cluster_raises():
    with pytest.raises(KeyError):
        skeptic.second_opinion("c99")


# --------------------------------------------------------------------------- #
# The LIVE second-opinion agent (loop.pressure_test): tissue-aware prompt, and a
# deterministic grounded fallback with no model (forced offline, no network).
# --------------------------------------------------------------------------- #
def test_pressure_test_offline_falls_back_to_grounded_skeptic():
    from panoscope.agent import loop

    ag = loop.PanoscopeAgent()
    ag._get_client = lambda: None  # force offline: no live model
    resp = ag.pressure_test("c9")
    assert resp.used_fallback is True
    assert resp.verify is True  # c9 is fragile -> re-check
    # the fallback prose clears the SAME grounding gate as any answer
    checker = GroundingChecker(literature_verifier=lambda _ident: True)
    assert checker.check(resp.text).ok, resp.text
    # a clean call withstands (verify False)
    assert ag.pressure_test("c2").verify is False


def test_skeptic_system_prompt_is_tissue_aware_and_guardrailed():
    from panoscope.agent import loop

    sp = loop.build_system_prompt("c9", skill="skeptic")
    assert "SECOND-OPINION SKEPTIC" in sp          # the adversarial contract
    assert "ADVERSARIAL SCAFFOLD" in sp            # the deterministic grounded seed
    assert "DATASET CONTEXT" in sp and "human breast cancer" in sp  # tissue-aware
    # the "no nonsense" guardrails must be present
    assert "off-panel" in sp and "cannot weigh" in sp
    assert "do NOT cry wolf" in sp.lower() or "do not cry wolf" in sp.lower()


def test_dataset_context_is_a_preference_not_a_filter():
    from panoscope.agent import loop

    loop._dataset_context.cache_clear()
    ctx = loop._dataset_context()
    assert "human breast cancer" in ctx
    assert "NOT a citation filter" in ctx
    # open-ceiling boundary: it can never move a number
    assert "never change a jazzpanda number" in ctx.lower()
