"""Renderers that consume a declared ``UISpec``.

``ui/surface.py`` is the first one (Streamlit/HTML). This package holds the second
one, in text, which is what makes "the review surface is declared by the workflow"
checkable: with a single renderer, *declared* and *hand-built in one place* produce
identical evidence.

Nothing here may import from ``panoscope.ui``. A renderer that wrapped the HTML one
would prove the HTML renderer is reusable; what is in question is whether the
declaration is sufficient to render from.
"""

from panoscope.render.text import render as render_text

__all__ = ["render_text"]
