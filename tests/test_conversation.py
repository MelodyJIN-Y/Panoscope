"""The chat pane's opening turn, and what happens to it when the call changes.

One rule, from CLAUDE.md's memory section: *never a silent overrule*. The
opening interpretation is posted once per cluster and the transcript is then
history -- which is right, up to the moment the biologist overrides the call
and the pane goes on saying the old one beside a header saying the new one.

The transcript is not rewritten here. A fresh opening is appended, so both the
call that was made and the call that replaced it stay on the page.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import memory
from panoscope.ui import state


@pytest.fixture
def session(monkeypatch):
    """A plain dict standing in for ``st.session_state``."""
    ss: dict = {}
    monkeypatch.setattr(state, "_ss", lambda: ss)
    return ss


def test_the_posted_opening_remembers_which_call_it_was_written_for(session):
    state.mark_opening_posted("c2", "Stromal")
    assert state.opening_was_posted("c2") is True
    assert state.opening_posted_call("c2") == "Stromal"


def test_a_restored_transcript_reports_an_unknown_call_not_a_wrong_one(session):
    """``_hydrate_threads`` writes ``True``: ``chat_store`` persists role, text,
    source kinds and verify, so the call is simply not recoverable. Returning
    None says so; returning the CURRENT call would assert that history agreed
    with the present, which is the claim this whole area keeps getting wrong."""
    session[state.K_OPENING_POSTED] = {"c2": True}
    assert state.opening_was_posted("c2") is True
    assert state.opening_posted_call("c2") is None


def test_the_opening_is_reposted_when_the_biologist_changes_the_call(session):
    """The end-to-end behaviour, at the level that does not need a browser."""
    from panoscope.ui import conversation

    posted: list[str] = []
    monkey_call = {"v": "Stromal"}
    import panoscope.ui.data_access as dax

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dax, "display_cell_type", lambda c: monkey_call["v"])
        mp.setattr(conversation, "_safe_opening",
                   lambda c: type("R", (), {"text": f"{c} reads as {monkey_call['v']}"})())
        mp.setattr(state, "append_message",
                   lambda c, m: posted.append(m["text"]))

        conversation._ensure_opening("c2")
        assert posted == ["c2 reads as Stromal"]

        # Same call, same session: posted once, not on every rerun.
        conversation._ensure_opening("c2")
        assert posted == ["c2 reads as Stromal"]

        # The biologist overrides. The old turn stays; a new one is appended.
        monkey_call["v"] = "CAF"
        conversation._ensure_opening("c2")
        assert posted == ["c2 reads as Stromal", "c2 reads as CAF"], posted

        # ...and then it settles again.
        conversation._ensure_opening("c2")
        assert len(posted) == 2


def test_a_real_override_moves_the_pane_off_the_replaced_call(session):
    """Through the real note writer and the real composition, no stubs."""
    from panoscope.ui import conversation, data_access as dax

    posted: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(state, "append_message", lambda c, m: posted.append(m["text"]))
        conversation._ensure_opening("c2")
        assert posted and posted[0].startswith("c2 reads as Stromal"), posted[0]

        memory.create_note(
            claim="c2 is CAF here", scope="cluster", basis="own_validation",
            cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
            dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
        )
        assert dax.display_cell_type("c2") == "CAF"

        conversation._ensure_opening("c2")
        assert len(posted) == 2, "the pane kept the replaced call"
        assert posted[1].startswith("c2 reads as CAF"), posted[1]
        assert "replacing the computed Stromal" in posted[1]


def test_the_pathways_pane_reposts_too(session):
    """The sibling pane, which the first fix did not touch.

    ``ui/conversation`` and ``ui/enrichment_conversation`` each keep their own
    opening guard, and only the marker one learned to notice a changed call. The
    Pathways bubble went on reading "c2 Stromal: 3 supported ..." under a header
    saying CAF -- the same defect, one file over, which is how it survived the
    commit that fixed it.
    """
    from panoscope.ui import data_access as dax
    from panoscope.ui import enrichment_conversation as ec

    posted: list[str] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(state, "append_message", lambda k, m: posted.append(m["text"]))

        ec._ensure_opening("c2")
        assert posted and posted[0].startswith("c2 Stromal:"), posted[:1]

        ec._ensure_opening("c2")
        assert len(posted) == 1, "posted once per call, not once per rerun"

        memory.create_note(
            claim="c2 is CAF here", scope="cluster", basis="own_validation",
            cluster="c2", subject_cell_type="CAF", note_type="celltype_override",
            dataset=cfg.DATASET_ID, literature_search=lambda _q: (),
        )
        assert dax.display_cell_type("c2") == "CAF"

        ec._ensure_opening("c2")
        assert len(posted) == 2, "the Pathways pane kept the replaced call"
        assert posted[1].startswith("c2 CAF:"), posted[1][:60]

        ec._ensure_opening("c2")
        assert len(posted) == 2, "and then it settles"


def test_the_two_panes_keep_separate_threads(session):
    """``pw::c2`` and ``c2``. A shared key would make one pane's opening suppress
    the other's, which is the failure mode the prefix exists to prevent."""
    from panoscope.ui import enrichment_conversation as ec

    assert ec._thread_key("c2") == "pw::c2" != "c2"
    state.mark_opening_posted("c2", "Stromal")
    assert state.opening_was_posted("c2") is True
    assert state.opening_was_posted(ec._thread_key("c2")) is False
