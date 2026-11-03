"""Local research profile — the biologist's field, to sharpen literature search.

A one-line description of the biologist's background/research interest, captured
at onboarding and stored LOCALLY in ``context/profile.json``. It is used only as
context for the agent's PubMed searches so citations are more precise and
tissue-appropriate.

Privacy + confident floor: the profile never leaves the machine except as query
context to the literature connector, and it can NEVER change a jazzPanda number, a
marker, a confidence band, or a cell-type call. It only influences WHICH real
paper is searched for and cited; the grounding gate still requires every cited
PMID to resolve to a real record.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_MAX_CHARS = 400


def _path() -> Path:
    """Where the profile lives. Resolved at CALL time, never from ``__file__``.

    This used to be ``Path(__file__).parent.parent / "context" / "profile.json"``,
    which put it INSIDE the installed package. Three things were wrong with that,
    and ``data/layout.py`` already documents all three: deriving a data path from
    ``__file__`` couples the install layout to the data layout; a pip-installed
    package directory is often read-only, so ``save()`` would fail exactly where
    it matters; and ``.gitignore``'s ``context/profile.json`` is anchored to the
    repo root, so the package copy was UNTRACKED rather than ignored -- a file
    this module's docstring calls local-and-private was one ``git add`` from
    being published.

    ``PANOSCOPE_CONTEXT_DIR`` first, matching ``agent/tools.py``, which already
    reads it for the memory notes that live beside this. Otherwise ``./context``,
    the path the docstring, the .gitignore rule and the biologist's existing
    profile all already agree on.
    """
    base = os.getenv("PANOSCOPE_CONTEXT_DIR")
    return (Path(base) if base else Path.cwd() / "context") / "profile.json"


def load() -> str:
    """Return the saved research interest (empty string if unset or unreadable)."""
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
        return str(data.get("research_interest", "")).strip()[:_MAX_CHARS]
    except (OSError, json.JSONDecodeError, ValueError):
        return ""


def save(text: str) -> None:
    """Persist the research interest locally (trimmed, capped). Empty clears it."""
    clean = (text or "").strip()[:_MAX_CHARS]
    _path().parent.mkdir(parents=True, exist_ok=True)
    _path().write_text(
        json.dumps({"research_interest": clean}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
