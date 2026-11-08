"""A scripted Anthropic client, so the REAL agent loop runs offline.

``install(monkeypatch)`` puts a :class:`FakeClient` behind ``panoscope.agent.loop``:
the loop's own ``_get_client`` / ``_run_loop`` / ``_run_tools`` / ``_finalize`` /
``_repair`` all run, and only ``messages.create`` is replaced. Script replies in the
order the loop will ask for them::

    client.script(reply(tool_use("get_markers", {"cluster": "c1"})), "c1 reads as ...")

A plain string is a one-text-block reply; an exception instance is raised from that
call (a failed model call). When the script runs out, ``create`` raises, which the
loop treats as a failed call -> deterministic fallback. ``client.calls`` holds every
request's keyword arguments. Citations resolve only if listed in ``client.real_ids``.
"""
from __future__ import annotations

from types import SimpleNamespace as NS


def text(s: str) -> NS:
    return NS(type="text", text=s)


def tool_use(name: str, input: dict, id: str | None = None) -> NS:
    return NS(type="tool_use", id=id or f"toolu_{name}", name=name, input=input)


def reply(*blocks: NS) -> NS:
    tool = any(b.type == "tool_use" for b in blocks)
    return NS(content=list(blocks), stop_reason="tool_use" if tool else "end_turn")


class FakeClient:
    def __init__(self) -> None:
        self.replies: list = []
        self.calls: list[dict] = []
        self.real_ids: set[str] = set()
        self.messages = self  # client.messages.create(...)

    def script(self, *replies) -> "FakeClient":
        self.replies += [reply(text(r)) if isinstance(r, str) else r for r in replies]
        return self

    def create(self, **kw):
        self.calls.append(kw)
        if not self.replies:
            raise RuntimeError("fake model: script exhausted")
        r = self.replies.pop(0)
        if isinstance(r, BaseException):
            raise r
        return r


def install(monkeypatch) -> FakeClient:
    from panoscope.agent import loop

    client = FakeClient()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-fake-model")
    monkeypatch.setattr(loop, "_ANTHROPIC_IMPORTED", True)
    monkeypatch.setattr(loop, "anthropic", NS(Anthropic=lambda **_: client))
    monkeypatch.setattr(loop, "_default_literature_verifier",
                        lambda ident: str(ident).strip() in client.real_ids)
    monkeypatch.setattr(loop, "_DEFAULT_AGENT", None)  # rebuilt against the fake
    forget_agent()
    return client


def forget_agent() -> None:
    """Drop the app's ``st.cache_resource`` agent, which outlives tests (it is no
    ``lru_cache``), so the next app run rebuilds it against this test's client."""
    from panoscope.ui import data_access

    data_access.get_agent.clear()
