"""Portable user memory: the biologist's decisions, distilled to carry across projects.

A confirmed lab decision (an override or note) can optionally be saved here in
addition to the dataset-local note. A lab note is scoped to one dataset and never
fires elsewhere; a user-memory entry is the opposite by design: a self-contained,
tissue-tagged distillation the biologist chooses to keep across projects. It is the
"personal research assistant" layer the lab owns and carries forward.

Stored LOCALLY in ``context/user_memory.jsonl`` (git-ignored, same privacy model as
:mod:`agent.profile`). Like the research profile, it is OPEN-CEILING context: it is
injected as prior lab knowledge to sharpen the agent's reasoning and literature
search, and it can NEVER change a jazzPanda number, a marker, a confidence band, or a
cell-type call. The grounding gate still governs every stated number.

Each entry is composed from the agent's own decision prose (the note claim it drafted
from the chat) plus structured provenance, so the memory is auto-written and
attributed, not free text typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

_PATH = Path(__file__).resolve().parent.parent / "context" / "user_memory.jsonl"

# How much prior knowledge to inject into the system prompt: recent entries only,
# hard-capped so a long history can never crowd out the skill or the contract.
_MAX_ENTRIES_IN_PROMPT = 12
_MAX_PROMPT_CHARS = 1800


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """Append one distilled decision to the local store. Best-effort, never raises."""
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        with _PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return entry


def load() -> list[dict[str, Any]]:
    """Return all saved entries, oldest first (empty if unset or unreadable)."""
    out: list[dict[str, Any]] = []
    try:
        raw = _PATH.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def distill(*, claim: str, cell_type: str, markers: Iterable[str], dataset_label: str) -> str:
    """Compose a portable one-line headline from the agent-written decision.

    Deterministic and grounded: it re-states the cell-type call and its driving
    markers with the dataset's tissue/platform tag, so the entry reads self-contained
    when recalled in another project. The full nuance stays in the note ``claim``,
    which the agent already wrote from the chat.
    """
    ct = (cell_type or "").replace("_", " ").strip() or "cell type"
    driver = ", ".join(g for g in list(markers)[:4] if g)
    head = f"{ct} in {dataset_label}" if dataset_label else ct
    return f"{head}, driven by {driver}" if driver else head


def as_prompt_context() -> str:
    """Render recent entries as a labelled, capped prior-knowledge block for the
    system prompt (empty string if none).

    Open-ceiling context only, and clearly marked as such, so the agent treats it as
    the lab's prior judgment to sharpen reasoning and literature search, never as a
    number source that can override jazzPanda.
    """
    entries = load()[-_MAX_ENTRIES_IN_PROMPT:]
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries:
        head = e.get("summary") or e.get("claim") or ""
        if not head:
            continue
        basis = e.get("basis", "")
        status = e.get("status", "")
        tag = f"[{status}, {basis}] " if (status or basis) else ""
        src = e.get("from", "")
        line = f"- {tag}{head}"
        if src:
            line += f" (from {src})"
        lines.append(line)
    block = "\n".join(lines)[:_MAX_PROMPT_CHARS]
    if not block:
        return ""
    return (
        "# PRIOR LAB KNOWLEDGE (the biologist's own saved decisions, portable across "
        "datasets)\n"
        "Treat these as the lab's prior judgment to sharpen your reasoning and which "
        "real paper you cite. They must NEVER invent, change, or override any jazzPanda "
        "number, marker, confidence band, or cell-type call.\n" + block
    )
