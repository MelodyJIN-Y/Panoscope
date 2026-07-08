"""Panoscope UI package.

The Streamlit surface for the annotation-confidence layer: a 3-pane shell
(cluster rail | evidence + spatial stage | conversation) built on the pure
``agent`` modules. This package owns only presentation and session state — it
never computes a marker, a number, or a confidence value. Every value it shows
traces to ``agent.data`` (jazzPanda output + panel), ``agent.verdict``, or a
stored lab note read through ``agent.memory``.

Foundation modules (import-safe with no server running):
- ``ui.state``       — session_state schema + typed accessors + ``init_state()``
- ``ui.theme``       — ``inject_css()`` design tokens (teal accent, chip colors)
- ``ui.data_access`` — cached wrappers over ``agent.data`` / ``agent.verdict``
- ``ui.format``      — pure formatters (confidence/role chips, cluster colors)

Submodules are imported lazily by callers (not re-exported here) so that
importing ``ui`` does not pull in Streamlit at package-import time.
"""

from __future__ import annotations

__all__: list[str] = ["state", "theme", "data_access", "format"]
