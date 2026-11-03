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

from panoscope.agent import config as cfg
from panoscope.agent import data as _data
from panoscope.agent import loop as agent_loop
from panoscope.agent.grounding_check import GroundingChecker
from panoscope.agent.loop import PanoscopeAgent
from panoscope.agent.types import AgentResponse, GroundingSidecar, Source

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
    from panoscope.agent.verdict import verdict_for_cluster

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


def _preflight_or_skip() -> None:
    """Prove the connector really answers BEFORE the live assertions run.

    The old gate was ``ANTHROPIC_API_KEY`` being *present*. With a present-but-
    invalid key the loop swallows the 401 and returns the deterministic fallback,
    which satisfies every assertion below — so the test reported PASS while the
    model was never reached. That is strictly worse than the SKIP-shaped bug this
    repo already fixed in ``test_mcp_client.py``.

    So: one real 1-token call, and classify the outcome.

      * SDK missing / no key                 -> skip (nothing to test)
      * connection refused, DNS, timeout     -> skip (genuinely offline)
      * 429 / 5xx                            -> skip (service unavailable right now)
      * 401 / 403 / any other 4xx            -> FAIL (a broken key is a broken test,
                                                not an excuse to pass)
    """
    try:
        import anthropic
    except ImportError:  # pragma: no cover - SDK is an optional extra
        pytest.skip("anthropic SDK not installed")

    if not _has_anthropic_key():  # also loads .env, which the agent does lazily
        pytest.skip("no ANTHROPIC_API_KEY / offline")
    key = os.environ["ANTHROPIC_API_KEY"]

    try:
        anthropic.Anthropic(api_key=key).messages.create(
            model=cfg.PRIMARY_MODEL,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except anthropic.APIConnectionError as exc:  # incl. APITimeoutError
        pytest.skip(f"Anthropic connector unreachable (offline?): {exc!r}")
    except anthropic.APIStatusError as exc:
        status = getattr(exc, "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            pytest.skip(f"Anthropic unavailable right now (HTTP {status}): {exc!r}")
        pytest.fail(
            f"the live path is NOT usable (HTTP {status}): {exc!r}\n"
            f"This test must not pass on the deterministic fallback — fix the "
            f"credential or deselect the `live` marker."
        )


@pytest.mark.live
def test_live_chat_passes_grounding_floor():
    """One real chat('what defines cluster 2?') whose answer clears the floor.

    Uses the loop's OWN grounding gate (live MCP-backed literature verifier), so
    a fabricated PMID or number in the real answer would fail it.

    The fallback is tested OFFLINE above (``test_chat_falls_back_*``). Here a
    fallback is a FAILURE, not an acceptable outcome: the whole point of this test
    is that the agent's own output is grounding-checked, and a fallback answer
    proves nothing about the model. ``used_fallback`` is the discriminator —
    ``agent/fallback.py`` sets it True on every path it owns, and only the live
    ``_assemble`` leaves it False.
    """
    _preflight_or_skip()

    agent = PanoscopeAgent(cluster="c2")  # real key from .env, live verifier
    resp = agent.chat("what defines cluster 2?")

    assert isinstance(resp, AgentResponse)

    # (1) It is a REAL model answer, not the deterministic fallback wearing its
    # clothes. Without this the test passes with an invalid key.
    assert resp.used_fallback is False, (
        "chat() returned the deterministic fallback on the LIVE path — the model "
        "was never reached (or its answer failed the floor and degraded). This "
        "test asserts the AGENT'S OWN output is grounded; the fallback is covered "
        "offline by test_chat_falls_back_when_anthropic_fails.\n\n" + resp.text
    )

    # (2) The answer was grounded through the tool loop. A fallback carries no
    # tools_used; an answer that quotes c2's numbers without marker_lookup could
    # not have cleared the floor in the first place.
    assert resp.tools_used, (
        "the live answer ran no tools — nothing grounded it. tools_used is empty, "
        "which the deterministic fallback also is; a live grounded answer about c2 "
        "must have pulled its numbers from marker_lookup."
    )

    # (3) The answer clears the confident floor using the agent's own checker.
    checker = GroundingChecker(literature_verifier=agent._verifier)
    result = checker.check(resp.text, resp.grounding, "c2")
    assert result.ok is True, f"live answer failed the floor: {result.summary()}\n\n{resp.text}"

    # (4) c2's headline driver is LUM; a grounded answer about c2 names it.
    assert "LUM" in resp.text.upper()

    # Proof in the log that the live path really ran.
    print(f"\n[live] tools_used={resp.tools_used} len(text)={len(resp.text)}")


# --------------------------------------------------------------------------- #
# Prompt context is keyed on the STUDY, not on the empty tuple.
#
# ``_global_interpretation_lines`` and ``_dataset_context`` take no arguments, so
# an lru_cache keys every study to one slot: the first study opened in the process
# freezes into every later study's system prompt. Under Streamlit that is a
# long-lived process serving both studies.
# --------------------------------------------------------------------------- #
def _units(study_id: str) -> tuple:
    with cfg.use_study(study_id):
        return tuple(cfg.CLUSTER_ORDER)


def test_global_interpretation_lines_follow_the_active_study():
    with cfg.use_study("xenium_hbreast_sample1"):
        breast = agent_loop._global_interpretation_lines()
    with cfg.use_study("demo_mini"):
        mini = agent_loop._global_interpretation_lines()
    with cfg.use_study("xenium_hbreast_sample1"):
        breast_again = agent_loop._global_interpretation_lines()

    assert breast != mini, (
        "the second study's cluster lines are the first study's — the prompt "
        "cache is not keyed on the study"
    )
    assert breast_again == breast, "switching back did not restore the first study"
    assert len(mini.splitlines()) == len(_units("demo_mini"))
    assert len(breast.splitlines()) == len(_units("xenium_hbreast_sample1"))


def test_dataset_context_is_keyed_per_study_too():
    """The other study-dependent cache in the prompt path.

    Both demo studies are breast/Xenium, so this one has no behavioural
    distinguisher on the data we ship -- the check is structural instead: a
    ``@lru_cache`` on a zero-arg reader exposes ``cache_info``; ``_per_study``
    does not. It fails the moment the decorator is put back.
    """
    assert not hasattr(agent_loop._dataset_context, "cache_info"), (
        "_dataset_context is back on a plain lru_cache -- it would freeze the "
        "first study's tissue into every later study's prompt"
    )
    agent_loop._dataset_context.cache_clear()
    with cfg.use_study("demo_mini"):
        assert isinstance(agent_loop._dataset_context(), str)


# --------------------------------------------------------------------------- #
# The in-scope lab-notes block renders THREE lookup states, not two.
#
# It used to branch on the stored ``thin`` flag, which cannot tell "we searched
# and the literature is thin" (a finding) from "no lookup completed" (which says
# nothing at all). The rest of the codebase derives that from ``looked_up_at``
# via claims.notes.lookup_state; the prompt now does too.
# --------------------------------------------------------------------------- #
def _note_block_for(tmp_path, literature_search):
    from panoscope.agent import memory
    from panoscope.agent import tools as agent_tools

    base = tmp_path / "context"
    draft = memory.draft_note(
        claim="LUM marks CAFs in this cluster",
        scope="cluster",
        basis="own_validation",
        cluster="c2",
        subject_markers=["LUM"],
        literature_search=literature_search,
    )
    memory.save_draft(draft, base_dir=base)
    prev = agent_tools._MEMORY_BASE_DIR
    agent_tools.set_memory_base_dir(str(base))
    try:
        return agent_loop._inscope_notes_block("c2")
    finally:
        agent_tools.set_memory_base_dir(prev)


def test_notes_block_says_searched_when_the_search_came_back_empty(tmp_path):
    block = _note_block_for(tmp_path, lambda q: [])
    assert "searched, literature thin" in block, block
    assert "no lookup completed" not in block


def test_notes_block_says_no_lookup_when_none_ran(tmp_path):
    block = _note_block_for(tmp_path, None)
    assert "no lookup completed" in block, block
    assert "searched, literature thin" not in block


def test_default_skill_contract_resolves():
    """Bug-shape #2: the SKILL.md path broke in the package move and the bare
    except hid it, so the interpretation contract was absent from every prompt."""
    text = agent_loop._skill_text()
    assert text.strip(), f"no SKILL.md under {agent_loop._SKILLS_DIR}"


def test_a_pathways_answer_carries_the_enrichment_rubric_not_the_marker_one() -> None:
    """Two workflows, two rubrics, and ``_assemble`` knew about only one.

    ``verify`` is deterministic and must come from the rubric of the workflow the
    answer belongs to. Every chat turn read the marker/ASSIGNMENT rubric, so a
    Pathways answer carried the marker call's re-check flag. The two disagree on
    6 of the demo's 9 clusters, every one of them marker-False /
    enrichment-True: the tab that should have said "re-check this" said nothing.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import loop as _loop

    disagree = [c for c in cfg.CLUSTER_ORDER
                if _loop._verify_default(c, "marker-interpretation")
                != _loop._verify_default(c, "enrichment-interpretation")]
    assert disagree, (
        "this test is vacuous unless the two rubrics actually differ somewhere"
    )
    for c in disagree:
        assert _loop._verify_default(c, "enrichment-interpretation") == \
            _loop._enrichment_verify_default(c)
        assert _loop._verify_default(c, "marker-interpretation") == \
            _loop._cluster_verify_default(c)


def test_the_prompt_leads_with_the_biologists_call_not_the_replaced_one() -> None:
    """The block is headed "authoritative — every cluster's call", and it stated
    the computed one.

    After an override the model was handed "c2: Stromal" twice as fact, with
    "c2 is CAF here" further down as a note, and left to resolve the
    contradiction itself -- while the board, the marker header, the chat and the
    exported document all said CAF. Whatever it picked, the grounding gate then
    graded.

    Both directions are asserted: the call must lead, and the computed one must
    stay visible, because a silent swap hides the disagreement the memory layer
    exists to keep.
    """
    from panoscope.agent import config as cfg
    from panoscope.agent import loop as _loop
    from panoscope.agent import memory

    before = _loop.build_system_prompt("c2", skill="marker-interpretation")
    assert "c2: Stromal — " in before, "c2 must start out called Stromal"

    memory.create_note(
        claim="c2 is CAF here", scope="cluster", basis="own_validation",
        cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
    )

    after = _loop.build_system_prompt("c2", skill="marker-interpretation")
    assert "c2: CAF (the biologist's call; computed Stromal)" in after, after[:0] or (
        "the global interpretation still states the replaced call"
    )
    assert "# ACTIVE CLUSTER: c2 — CAF (the biologist's call; computed Stromal)" in after
    assert "c2: Stromal — " not in after

    # Only the LABEL moves: the band and the drivers are still the computed ones.
    assert "Very High confidence (drivers: LUM, POSTN, CCDC80)" in after
    # ...and an unoverridden cluster is untouched.
    assert "c1: Tumor — " in after
