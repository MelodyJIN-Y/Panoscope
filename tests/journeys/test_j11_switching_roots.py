"""J11: the demo to learn the UI, then my own study, then back (S10 in the walks)."""
from __future__ import annotations

import pytest

from tests.journeys.flow import DEMO_INPUTS, accept, call_it, folder_with, go, onboard, upload
from tests.ui.screen import plain, raw_markdown, run


def _decision(at, cluster: str) -> str:
    at = go(at, "markers", cluster)
    block = next(plain(m.value) for m in at.markdown
                 if "pano-sf-block" in m.value and "decision" in m.value)
    return block.split("·")[-1].strip()


@pytest.mark.req("PLAN-P1.4-01")
def test_switching_between_the_demo_and_my_study_never_leaks_a_call(
        isolated, tmp_path, no_progress_sleep):
    """Given a biologist who explores the demo and accepts c9,
    When they build their own folder, Then their own c9 is open (no bleed);
    When they call their c1 and go back to the demo, Then the demo's c1 is open, its c9
    still accepted, and the demo Summary never shows their label;
    When they paste their folder again and rebuild, Then their c1 call survived."""
    at = run(upload().button(key="ob_analyze").click())
    demo = at.session_state["_study"]
    at = accept(at, "c9")
    assert _decision(at, "c9") == "accepted"

    folder = folder_with(tmp_path, "mine", {"markers_top.csv": DEMO_INPUTS / "markers_top.csv",
                                            "panel.parquet": DEMO_INPUTS / "panel.parquet"})
    at = onboard(folder, at=at, tissue="human breast cancer")
    mine = at.session_state["_study"]
    assert mine != demo and mine[1].startswith("mine-")
    assert _decision(at, "c9") == "open"
    at = call_it(at, "c1", free="Luminal HR+ tumour")
    assert _decision(at, "c1") == "overridden"

    at = run(go(at, "upload").button(key="ob_analyze").click())
    assert at.session_state["_study"] == demo
    assert (_decision(at, "c1"), _decision(at, "c9")) == ("open", "accepted")
    assert "Luminal HR+" not in raw_markdown(go(at, "summary"))

    at = onboard(folder, at=at, tissue="human breast cancer")
    assert at.session_state["_study"] == mine
    assert _decision(at, "c1") == "overridden"


@pytest.mark.req("PLAN-P1.4-01", "ARCH-11.1-01")
def test_a_reloaded_chat_thread_keeps_its_fallback_label_and_renders():
    """A thread saved with a fallback turn reloads with that label (and the attributes
    the bubble reads); before, a reload crashed the chat pane on the missing field."""
    from types import SimpleNamespace as NS

    from panoscope.ui import chat_store

    resp = NS(text="c1 is called Tumor.", sources=(), verify=True, used_fallback=True,
              opening=False)
    back = chat_store._msg_from_dict(chat_store._msg_to_dict(
        {"role": "agent", "text": resp.text, "resp": resp}))["resp"]
    assert (back.used_fallback, back.verify, back.opening) == (True, True, False)
