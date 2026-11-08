"""Corpus A25: the "live citation" wording is gone everywhere a reader or the model
sees it, replaced by "looked up from a real source at build or top-up, never from
memory". A citation is no longer fetched while the biologist reads, so promising one
"live" describes a path that does not exist.

Scope: the README, the current docs and skills (``docs/archive`` is superseded and
excluded), and every string the package renders or sends to the model. Docstrings and
comments are developer notes and are left out.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LIVE = re.compile(r"(?i)live[- ]cit|cite[sd]? live|live citation")

DOCS = [REPO / "README.md", *sorted((REPO / "docs").glob("*.md")),
        *sorted((REPO / "panoscope" / "skills").rglob("SKILL.md"))]


def _strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)}
    return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and id(n) not in docstrings]


@pytest.mark.req("CORP-A25")
@pytest.mark.parametrize("path", DOCS, ids=lambda p: str(p.relative_to(REPO)))
def test_no_document_promises_a_live_citation(path):
    assert not LIVE.findall(path.read_text())


@pytest.mark.req("CORP-A25")
@pytest.mark.parametrize("path", sorted((REPO / "panoscope").rglob("*.py")),
                         ids=lambda p: str(p.relative_to(REPO)))
def test_no_rendered_or_prompted_string_promises_a_live_citation(path):
    assert not [s for s in _strings(path) if LIVE.search(s)]
