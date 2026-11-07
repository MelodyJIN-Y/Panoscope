"""Panoscope UI package.

The Streamlit surface: presentation and session state only. It never computes a
marker, a number or a confidence value; every value traces to ``agent.data``,
``pipeline``, or a stored note. Submodules are not re-exported, so importing
``ui`` does not pull in Streamlit.
"""

from __future__ import annotations

__all__: list[str] = ["state", "theme", "data_access", "format"]
