"""Journeys (plan Stage 6): one whole biologist journey per test, Given/When/Then.

Every journey is marked ``journey`` and checks that the committed ``data/studies``
tree is byte-unchanged after it ran (the session guard names the path; this one
names the journey)."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import diff_tree, snapshot_tree

_HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    for item in items:
        if _HERE in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.journey)


@pytest.fixture(autouse=True)
def _repo_tree_untouched():
    before = snapshot_tree()
    yield
    assert diff_tree(before, snapshot_tree()) == []


@pytest.fixture
def no_progress_sleep(monkeypatch):
    """'Explore the demo' plays a 2.85 s progress animation; it reads, runs nothing."""
    from panoscope.ui import onboarding

    monkeypatch.setattr(onboarding, "_run_progress", lambda: None)


@pytest.fixture
def offline_user(tmp_path, monkeypatch) -> dict:
    """The ``isolated`` harness without its demo tree, for builder studies: own HOME,
    no model key, Europe PMC ids answered from the returned dict."""
    from panoscope.agent import loop
    from panoscope.corpus import exchange

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PANOSCOPE_DATASET", raising=False)
    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    ids: dict = {}
    monkeypatch.setattr(exchange, "resolve_id", lambda pid, retrieved: ids.get(pid))
    return ids
