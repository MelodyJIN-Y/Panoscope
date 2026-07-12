"""The second-opinion skeptic: grounded, calibrated, and never fabricating.

The skeptic must (a) let a clean call withstand challenge, (b) flag a fragile call
for re-check, and (c) quote only numbers that trace to jazzPanda output.
"""
from __future__ import annotations

import pytest

from agent import skeptic
from agent.grounding_check import GroundingChecker


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
    for cluster in ("c1", "c2", "c9"):
        report = skeptic.second_opinion(cluster)
        prose = skeptic.skeptic_summary(report)
        result = checker.check(prose)
        assert result.ok, f"{cluster} skeptic prose failed grounding: {result.summary()}"


def test_unknown_cluster_raises():
    with pytest.raises(KeyError):
        skeptic.second_opinion("c99")
