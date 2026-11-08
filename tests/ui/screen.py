"""What the biologist sees, read off an AppTest: the shared helpers for tests/ui.

``screen_text`` is visible text only. Tags and ``<style>`` blocks are stripped, so a
``title=`` tooltip never counts as shown: a word that lives only in a hover is not
on the screen. ``figure_values`` decodes each Plotly figure's own JSON (``proto.spec``),
binary-packed arrays included, so a test compares what the chart was given to draw.
"""
from __future__ import annotations

import base64
import html
import json
import re
from pathlib import Path

APP = str(Path(__file__).resolve().parents[2] / "panoscope" / "app.py")

_TAG = re.compile(r"<[^>]*>")
_STYLE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.S | re.I)
_TEXT_KINDS = ("markdown", "caption", "header", "subheader", "title", "info", "warning",
               "success", "error", "code")
_WIDGET_KINDS = ("button", "toggle", "checkbox", "selectbox", "radio", "multiselect",
                 "text_input", "text_area", "button_group", "download_button")
#: One evidence-table row: its anchor and the cells inside it.
_ROW = re.compile(r'<span class="pano-sf-anchor" data-claim="([^"]*)">'
                  r'<div class="pano-sf-row"[^>]*>(.*?)</div></div></span>', re.S)
_CELL = re.compile(r'<div class="pano-sf-cell[^"]*">(.*?)</div>', re.S)


def plain(raw: str) -> str:
    """HTML or markdown as a reader sees it: no styles, tags, code ticks or bold stars,
    entities decoded, one space."""
    text = html.unescape(_TAG.sub(" ", _STYLE.sub(" ", raw))).replace("`", "").replace("**", "")
    return re.sub(r"\s+", " ", text).strip()


def raw_markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def screen_text(at) -> str:
    """Every visible string: text elements plus widget labels and options, one per line."""
    out: list[str] = []
    for kind in _TEXT_KINDS:
        out += [plain(str(getattr(e, "value", "") or "")) for e in at.get(kind)]
    for kind in _WIDGET_KINDS:
        for w in at.get(kind):
            out.append(str(getattr(w, "label", "") or ""))
            out += [str(o) for o in (getattr(w, "options", None) or ())]
    return "\n".join(s for s in out if s)


def widget_keys(at) -> set[str]:
    return {w.key for kind in _WIDGET_KINDS for w in at.get(kind) if getattr(w, "key", None)}


def _decode(v):
    """Plotly's binary array encoding ({dtype, bdata[, shape]}) back to plain lists."""
    if isinstance(v, dict):
        if "bdata" in v and "dtype" in v:
            import numpy as np

            arr = np.frombuffer(base64.b64decode(v["bdata"]), dtype=v["dtype"])
            if v.get("shape"):
                arr = arr.reshape([int(n) for n in str(v["shape"]).split(",")])
            return arr.tolist()
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v


def figure_values(at) -> dict[str, list[dict]]:
    """``{chart key: [trace, ...]}``, each trace's data decoded from the figure JSON."""
    return {c.key: _decode(json.loads(c.proto.spec)["data"]) for c in at.get("plotly_chart")}


def evidence_rows(at) -> dict[str, dict]:
    """The evidence table's rows by subject key: visible ``cells``, ``text`` and ``html``."""
    rows: dict[str, dict] = {}
    for key, inner in _ROW.findall(raw_markdown(at)):
        cells = [plain(c) for c in _CELL.findall(inner)]
        rows[key] = {"cells": cells, "text": plain(inner), "html": inner}
    return rows


def open_app(page: str, *, cluster: str | None = None, markers: str | None = None,
             study: tuple[str, str] | None = None, **session):
    """The real app on ``page`` (data root from the environment), run once."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["active_page"] = page
    if study is not None:
        at.session_state["_study"] = study
    for k, v in session.items():
        at.session_state[k] = v
    if cluster:
        at.query_params["cluster"] = cluster
    if markers:
        at.query_params["m"] = markers
    return run(at)


def run(x):
    """``x.run()`` (an AppTest or a widget just set) that fails on a rendered exception."""
    at = x.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at
