"""Local research profile: the biologist's field, used only as PubMed search context.

Stored locally in ``context/profile.json``. It can NEVER change a number, marker,
band or call; it only steers which real paper is searched for (grounding still applies).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_MAX_CHARS = 400


def _path() -> Path:
    """``PANOSCOPE_CONTEXT_DIR`` or ``./context``, resolved per call.

    Never under the package: installs may be read-only and .gitignore is repo-anchored.
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
