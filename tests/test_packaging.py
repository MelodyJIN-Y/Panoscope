"""IMPLEMENTATION §8: requires-python >= 3.11, every dependency upper-bounded, and a
lockfile shipped; the model provider stays a direct, declared client (llm-access)."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
DEPS = list(PROJECT["dependencies"]) + [
    d for group in PROJECT["optional-dependencies"].values() for d in group]


def _name(req: str) -> str:
    return re.split(r"[<>=!~\[; ]", req, maxsplit=1)[0].strip().lower()


@pytest.mark.req("IMPL-8-01")
def test_python_floor_is_three_eleven():
    assert PROJECT["requires-python"].replace(" ", "") == ">=3.11"


@pytest.mark.req("IMPL-8-01")
@pytest.mark.parametrize("req", DEPS)
def test_every_dependency_is_upper_bounded(req):
    assert re.search(r"<\s*\d", req), f"{req!r} has no upper bound"


@pytest.mark.req("IMPL-8-01")
def test_the_lockfile_pins_every_base_dependency():
    lock = (REPO / "requirements.lock").read_text().lower()
    pinned = {_name(line) for line in lock.splitlines() if "==" in line}
    missing = {_name(d) for d in PROJECT["dependencies"]} - pinned
    assert not missing, missing


@pytest.mark.req("IMPL-8-01", "CLAUDE-CMP-01")
def test_no_model_gateway_is_required():
    assert not {_name(d) for d in DEPS} & {"litellm", "openrouter"}
