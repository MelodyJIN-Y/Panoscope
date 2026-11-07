"""Portable user memory: the biologist's decisions, distilled to carry across projects.

Stored locally in ``context/user_memory.jsonl``. Open-ceiling prompt context only: it
can NEVER change a number, marker, band or call.

Not a ``Claim``: a LAB claim needs the ``Conditions`` it was formed under (IMPLEMENTATION
§10), and a portable entry by design has none.
"""
from __future__ import annotations

import json
from typing import Any, Iterable


def _path():
    """``memory._default_base() / user_memory.jsonl``, resolved per call (honours the env var)."""
    from panoscope.agent.memory import _default_base

    return _default_base() / "user_memory.jsonl"

# How much prior knowledge to inject into the system prompt: recent entries only,
# hard-capped so a long history can never crowd out the skill or the contract.
_MAX_ENTRIES_IN_PROMPT = 12
_MAX_PROMPT_CHARS = 1800


def record(entry: dict[str, Any]) -> dict[str, Any]:
    """Append one distilled decision to the local store. Best-effort, never raises."""
    try:
        path = _path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass
    return entry


def load() -> list[dict[str, Any]]:
    """Return all saved entries, oldest first (empty if unset or unreadable)."""
    out: list[dict[str, Any]] = []
    try:
        raw = _path().read_text(encoding="utf-8")
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
    """Deterministic one-line headline: the call, its driving markers and the dataset tag."""
    ct = (cell_type or "").replace("_", " ").strip() or "cell type"
    driver = ", ".join(g for g in list(markers)[:4] if g)
    head = f"{ct} in {dataset_label}" if dataset_label else ct
    return f"{head}, driven by {driver}" if driver else head


def as_prompt_context() -> str:
    """Recent entries as a capped, labelled prior-knowledge block ("" if none).

    Marked open-ceiling: never a number source.
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
