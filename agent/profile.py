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
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "context" / "profile.json"
_MAX_CHARS = 400


def load() -> str:
    """Return the saved research interest (empty string if unset or unreadable)."""
    try:
        data = json.loads(_PATH.read_text(encoding="utf-8"))
        return str(data.get("research_interest", "")).strip()[:_MAX_CHARS]
    except (OSError, json.JSONDecodeError, ValueError):
        return ""


def save(text: str) -> None:
    """Persist the research interest locally (trimmed, capped). Empty clears it."""
    clean = (text or "").strip()[:_MAX_CHARS]
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(
        json.dumps({"research_interest": clean}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
