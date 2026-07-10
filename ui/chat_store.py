"""Persist chat threads to disk so a browser refresh doesn't lose the conversation.

The DURABLE lab knowledge is the notes under ``context/`` — the chat is a working
transcript. We keep a lightweight per-dataset copy (gitignored) so a reload restores
what was said. Only the render-relevant fields are stored (role, text, the source
KINDS, and the verify flag); a minimal ``resp`` is rehydrated for the bubble. Pending
unconfirmed note drafts are session-only and intentionally NOT persisted.

Fail-soft throughout: a missing/malformed file loads as no threads, and a write
failure is swallowed — losing the transcript must never break the app.
"""
from __future__ import annotations

import json
from types import SimpleNamespace


def _path():
    from agent import config as cfg
    from pipeline import paths

    return paths.interp_dir(cfg.DATASET_ID) / "chat_threads.json"


def _msg_to_dict(msg: dict) -> dict:
    resp = msg.get("resp")
    return {
        "role": msg.get("role", "agent"),
        "text": msg.get("text", ""),
        "src_kinds": [s.kind for s in (getattr(resp, "sources", ()) or ())],
        "verify": bool(getattr(resp, "verify", False)),
    }


def _msg_from_dict(d: dict) -> dict:
    """Rebuild a render-ready message. Agent turns get a minimal resp carrying the
    source kinds + verify flag so the 'Sources' line and re-check note re-render (the
    clickable PMIDs come from the text). User/system turns keep resp=None."""
    from agent.types import Source

    kinds = d.get("src_kinds") or []
    verify = bool(d.get("verify", False))
    resp = None
    if kinds or verify:
        resp = SimpleNamespace(
            sources=tuple(Source(kind=k, ref="", value=None) for k in kinds),
            verify=verify,
            note_draft=None,
            pin_marker=None,
        )
    return {"role": d.get("role", "agent"), "text": d.get("text", ""), "resp": resp}


def load_all() -> dict:
    """Return ``{thread_key: [msg, ...]}`` restored from disk (empty on absence/error)."""
    p = _path()
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {
            str(k): [_msg_from_dict(m) for m in v if isinstance(m, dict)]
            for k, v in raw.items()
            if isinstance(v, list)
        }
    except Exception:  # noqa: BLE001 - a bad transcript loads as no threads
        return {}


def save_all(threads: dict) -> None:
    """Persist all non-empty threads (best-effort; a write failure is swallowed)."""
    p = _path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        raw = {k: [_msg_to_dict(m) for m in v] for k, v in threads.items() if v}
        with p.open("w", encoding="utf-8") as fh:
            json.dump(raw, fh, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - losing the transcript must not break the app
        pass


__all__ = ["load_all", "save_all"]
