"""Renderers that consume a declared ``UISpec``.

``ui/surface.py`` is the HTML one; this package holds the text one. Nothing here may
import from ``panoscope.ui``: the point is that the declaration alone is enough to
render from.
"""

from panoscope.render.text import render as render_text

__all__ = ["render_text"]
