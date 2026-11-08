"""Refused by decision (ARCHITECTURE §15, CLAUDE.md Compliance): each refusal is
asserted as an absence, so building one turns this file red.

* no novelty verdict ("is this finding new?"), no SVG interpretation workflow, no
  ensemble vote across methods, no ranking by citation count;
* no consumer-subscription auth: no OAuth, session cookie or claude.ai token path.
  ``doctor`` never printing a secret is ``test_cli.py``'s
  ``test_doctor_reports_presence_but_never_the_secret``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from panoscope import methods, workflows
from panoscope.types.outputs import TaskKind

PKG = Path(__file__).resolve().parents[1] / "panoscope"
SOURCES = sorted(PKG.rglob("*.py"))


def _identifiers(path: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
    return out


ALL_IDS = {i.lower() for p in SOURCES for i in _identifiers(p)}


@pytest.mark.req("ARCH-15-01")
def test_no_registered_question_is_novelty_svg_or_a_vote():
    assert set(TaskKind) == {TaskKind.ASSIGNMENT, TaskKind.CHARACTERISATION,
                             TaskKind.COMPARISON}
    refused = re.compile(r"(?i)novel|\bnew\b|svg|spatially variable|ensemble|vote")
    for w in workflows.WORKFLOWS.values():
        assert not refused.search(f"{w.name} {w.question}"), w.name
    for m in methods.METHODS.values():
        assert not refused.search(f"{m.name} {m.workflow}"), m.name


@pytest.mark.req("ARCH-15-01", "ARCH-8.1-04", "SKC-R6-01")
def test_no_code_path_names_a_refused_feature():
    refused = re.compile(r"novel|svg_|spatially_variable|ensemble|vote|citation_count|"
                         r"cited_by|times_cited|relative_citation|icite")
    assert not sorted(i for i in ALL_IDS if refused.search(i))


@pytest.mark.req("CLAUDE-CMP-01")
@pytest.mark.parametrize("path", SOURCES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_consumer_subscription_auth_path(path):
    """API keys only (docs/llm-access.md). A consumer login would be an OAuth flow, a
    session cookie or a claude.ai token; none may appear anywhere in the package."""
    text = path.read_text().lower()
    for word in ("oauth", "claude.ai", "sessionkey", "session_key", "cookie",
                 "claude_code_oauth", "setup-token"):
        assert word not in text, word
