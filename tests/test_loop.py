"""Tests for agent/loop.py — the Anthropic tool-use loop the biologist talks to.

The three guarantees this file proves (per the task brief):

1. **Grounded opening.** ``opening_interpretation("c2")`` returns a grounded
   :class:`AgentResponse` that names LUM with its real jazzPanda numbers and the
   off-panel absence context, and PASSES the grounding floor. The opening is
   posted BEFORE any question, built from the deterministic verdict engine.
2. **Fallback path.** With the Anthropic call forced to fail (monkeypatch),
   ``chat()`` still returns a grounded fallback :class:`AgentResponse` — no
   exception reaches the caller, and the answer clears the floor.
3. **Live loop (optional).** If ``ANTHROPIC_API_KEY`` is present, one REAL
   ``chat("what defines cluster 2?")`` call whose answer passes the grounding
   floor. Skipped cleanly when no key / no network.

Grounding discipline in the tests: LUM's real glm_coef is read from
``agent.data`` at collect time so no assertion hardcodes a value that could drift
from source. The live test uses the SAME grounding checker the loop uses, wired
to the live MCP connector, so a fabricated PMID in a real answer would fail it.
"""

from __future__ import annotations

import os

import pytest

from agent import config as cfg  # noqa: F401 - imported for parity / future use
from agent import data as _data
from agent import loop as agent_loop
from agent.grounding_check import GroundingChecker
from agent.loop import PanoscopeAgent
from agent.types import AgentResponse, GroundingSidecar, Source

# --------------------------------------------------------------------------- #
# LUM's real jazzPanda numbers for c2 (read straight from agent.data so the
# assertion never hardcodes a value that could drift from source).
# --------------------------------------------------------------------------- #
_LUM_ROW = _data.get_marker("LUM")
LUM_GLM_COEF = float(_LUM_ROW["glm_coef"])


# --------------------------------------------------------------------------- #
# A permissive literature verifier for the offline grounding checks. The
# fallback/opening paths on which we assert the floor cite either NO PMIDs (pure
# verdict opening) or real ones; a permissive resolver is only ever consulted if
# a PMID is present, and the deterministic paths under test emit none, so it is
# safe. The LIVE test builds its own connector-backed verifier instead.
# --------------------------------------------------------------------------- #
def _resolver_all_real(_ident: str) -> bool:
    return True


@pytest.fixture
def offline_checker() -> GroundingChecker:
    return GroundingChecker(literature_verifier=_resolver_all_real)


@pytest.fixture
def offline_agent() -> PanoscopeAgent:
    """An agent whose grounding gate uses the permissive offline verifier.

    We pass ``api_key=None`` so no live model is constructed; the opening's
    optional live-enrichment literature call then returns ok=False and the
    opening is the pure, deterministic verdict opening (no PMID).
    """
    return PanoscopeAgent(
        cluster="c2",
        api_key=None,
        literature_verifier=_resolver_all_real,
    )


# --------------------------------------------------------------------------- #
# 1. Grounded opening interpretation for c2 (LUM + real numbers + off-panel note)
# --------------------------------------------------------------------------- #
def test_opening_interpretation_c2_is_grounded(offline_agent, offline_checker):
    """opening_interpretation('c2') -> grounded AgentResponse, passes the floor.

    With api_key=None the opening is the pure, deterministic verdict opening. It
    must name LUM with its real glm_coef, surface the off-panel absence context,
    and clear the gate.
    """
    resp = offline_agent.opening_interpretation("c2")

    assert isinstance(resp, AgentResponse)
    assert resp.opening is True

    # LUM is named with its REAL glm_coef (rendered to 2 decimals, within the
    # checker's tolerance of the true value from source).
    assert "LUM" in resp.text
    assert f"{LUM_GLM_COEF:.2f}" in resp.text

    # The sidecar carries the exact real number, not the rounded prose value.
    sidecar_numbers = {(g, s): v for (g, s, v) in resp.grounding.numbers}
    assert ("LUM", "glm_coef") in sidecar_numbers
    assert sidecar_numbers[("LUM", "glm_coef")] == LUM_GLM_COEF

    # The off-panel canonical markers (c2 Stromal spine) are surfaced as absence
    # context, and the panel-absence rule is stated in plain words.
    assert "COL1A1" in resp.text
    assert "off-panel" in resp.text.lower()
    assert "not evidence against" in resp.text.lower()

    # LUM is the pinned marker (the leading driver).
    assert resp.pin_marker == "LUM"

    # THE FLOOR: the opening passes the grounding check.
    result = offline_checker.check(resp.text, resp.grounding, "c2")
    assert result.ok is True, result.summary()


def test_opening_interpretation_requires_a_cluster():
    """No cluster set and none passed -> KeyError (explicit, never a silent guess)."""
    agent = PanoscopeAgent(api_key=None, literature_verifier=_resolver_all_real)
    with pytest.raises(KeyError):
        agent.opening_interpretation()


def test_opening_interpretation_unknown_cluster_raises():
    agent = PanoscopeAgent(api_key=None, literature_verifier=_resolver_all_real)
    with pytest.raises(KeyError):
        agent.opening_interpretation("c99")


# --------------------------------------------------------------------------- #
# 2. FALLBACK PATH — Anthropic call forced to fail; chat() still grounded
# --------------------------------------------------------------------------- #
def test_chat_falls_back_when_anthropic_fails(monkeypatch, offline_checker):
    """chat() returns a grounded fallback (no exception) when the model call fails.

    We give the agent a real-looking client (so it enters the loop) but force the
    inner loop primitive to RAISE; the loop must swallow it and return a
    deterministic grounded fallback that clears the floor.
    """
    agent = PanoscopeAgent(
        cluster="c2",
        api_key="sk-ant-fake-key-for-test",
        literature_verifier=_resolver_all_real,
    )

    # Force a client to exist (so chat() takes the live path).
    monkeypatch.setattr(agent, "_get_client", lambda: object())

    # A hard raise inside the loop must not escape chat(): it degrades to fallback.
    def _boom(*_a, **_k):
        raise RuntimeError("simulated Anthropic outage")

    monkeypatch.setattr(agent, "_run_loop", _boom)

    resp = agent.chat("what defines cluster 2?")

    assert isinstance(resp, AgentResponse)
    assert resp.used_fallback is True
    # The fallback is grounded on jazzPanda (c2 -> LUM) and clears the floor.
    assert "LUM" in resp.text
    assert offline_checker.check(resp.text, resp.grounding, "c2").ok is True


def test_chat_falls_back_when_no_client(monkeypatch, offline_checker):
    """No usable client -> deterministic grounded fallback, never a crash.

    Note: passing ``api_key=None`` is NOT enough to disable the client — the
    Anthropic SDK (and this agent's ``__init__``) still pick up
    ``ANTHROPIC_API_KEY`` from the environment/.env. To exercise the no-client
    branch deterministically we force ``_get_client`` to return None (the exact
    state on a machine with no key and no SDK).
    """
    agent = PanoscopeAgent(
        cluster="c1", api_key=None, literature_verifier=_resolver_all_real
    )
    monkeypatch.setattr(agent, "_get_client", lambda: None)
    resp = agent.chat("could this be a doublet?")
    assert isinstance(resp, AgentResponse)
    assert resp.used_fallback is True
    # c1 Tumor is driven by ERBB2; the doublet fallback names it and grounds it.
    assert "ERBB2" in resp.text
    assert offline_checker.check(resp.text, resp.grounding, "c1").ok is True


def test_chat_safe_create_none_triggers_fallback(monkeypatch, offline_checker):
    """When the guarded model call returns None mid-loop, chat() falls back cleanly.

    This exercises the real ``_run_loop`` (not stubbed): the guarded model call
    returns None on the first round, so the loop returns a grounded fallback.
    """
    agent = PanoscopeAgent(
        cluster="c9",
        api_key="sk-ant-fake",
        literature_verifier=_resolver_all_real,
    )
    monkeypatch.setattr(agent, "_get_client", lambda: object())
    monkeypatch.setattr(agent, "_safe_create", lambda *a, **k: None)

    resp = agent.chat("how confident are you about cluster 9?")
    assert isinstance(resp, AgentResponse)
    assert resp.used_fallback is True
    # c9 Mast is fragile/small-n -> verify stays TRUE on the fallback.
    assert resp.verify is True
    assert offline_checker.check(resp.text, resp.grounding, "c9").ok is True


def test_module_level_chat_never_raises(monkeypatch):
    """The module-level chat() convenience also degrades to a grounded fallback.

    We install a default agent whose client is forced off (see the note in
    :func:`test_chat_falls_back_when_no_client` — ``api_key=None`` alone does not
    disable the SDK's env-var key), then confirm the module-level ``chat`` returns
    a grounded fallback.
    """
    stub = PanoscopeAgent(api_key=None, literature_verifier=_resolver_all_real)
    monkeypatch.setattr(stub, "_get_client", lambda: None)
    agent_loop._DEFAULT_AGENT = stub
    try:
        resp = agent_loop.chat("what defines this cluster?", cluster="c2")
        assert isinstance(resp, AgentResponse)
        assert resp.used_fallback is True
        assert "LUM" in resp.text
    finally:
        # reset so we don't leak the stubbed default agent to other tests
        agent_loop._DEFAULT_AGENT = None


# --------------------------------------------------------------------------- #
# System prompt assembly — SKILL.md + contract + cluster context all present
# --------------------------------------------------------------------------- #
def test_system_prompt_loads_skill_and_contract():
    sys = agent_loop.build_system_prompt("c2")
    # SKILL.md content (the panel-absence rule headline) is loaded.
    assert "panel-absence" in sys.lower()
    # The confident-floor contract is present.
    assert "CONFIDENT-FLOOR CONTRACT" in sys
    assert "NEVER fabricate" in sys
    # The active cluster is named.
    assert "ACTIVE CLUSTER: c2" in sys
    # The cluster key is present (c2 -> Stromal).
    assert "Stromal" in sys


# --------------------------------------------------------------------------- #
# Envelope invariants on the opening response
# --------------------------------------------------------------------------- #
def test_opening_response_shape(offline_agent):
    resp = offline_agent.opening_interpretation("c2")
    assert isinstance(resp.grounding, GroundingSidecar)
    assert all(isinstance(s, Source) for s in resp.sources)
    # verify mirrors the verdict for c2 (Very High -> verify False).
    from agent.verdict import verdict_for_cluster

    assert resp.verify == verdict_for_cluster("c2").verify


# --------------------------------------------------------------------------- #
# 3. LIVE loop — one REAL chat call, answer must clear the floor (opt-in)
# --------------------------------------------------------------------------- #
def _has_anthropic_key() -> bool:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:  # pragma: no cover
        pass
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.live
@pytest.mark.skipif(not _has_anthropic_key(), reason="no ANTHROPIC_API_KEY / offline")
def test_live_chat_passes_grounding_floor():
    """One real chat('what defines cluster 2?') whose answer clears the floor.

    Uses the loop's OWN grounding gate (live MCP-backed literature verifier), so
    a fabricated PMID or number in the real answer would fail it. If the model or
    network is unavailable, the loop returns a grounded fallback instead — either
    way the answer must pass the floor and mention LUM (c2's driving marker).
    """
    agent = PanoscopeAgent(cluster="c2")  # real key from .env, live verifier
    resp = agent.chat("what defines cluster 2?")

    assert isinstance(resp, AgentResponse)
    # The answer clears the confident floor using the agent's own checker.
    checker = GroundingChecker(literature_verifier=agent._verifier)
    result = checker.check(resp.text, resp.grounding, "c2")
    assert result.ok is True, f"live answer failed the floor: {result.summary()}\n\n{resp.text}"

    # c2's headline driver is LUM; a grounded answer about c2 should mention it
    # (true for both a real tool-grounded answer and the deterministic fallback).
    assert "LUM" in resp.text.upper()
