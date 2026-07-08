"""Citation paper drawer — the right-hand slide-over that shows one paper.

When the biologist clicks a ``📄 Author Year`` citation in the conversation or
the opening interpretation, that pane calls ``ui.state.open_paper(pmid)`` and the
drawer renders the paper's journal / PMID / title / authors / abstract plus a
link to ``pubmed.ncbi.nlm.nih.gov/{pmid}`` (see the ``.paper`` panel in
``dashboard_wireframe_panels.html``).

Grounding discipline (Panoscope's confident floor): **every field shown here is
metadata the agent layer already fetched live** — a ``Citation`` object produced
by ``agent.loop`` through the PubMed / bioRxiv MCP connector. This drawer never
writes a PMID, a title, or an abstract from memory. It only displays a
``Citation`` some caller registered. If a PMID has no registered citation, or the
registered citation is not a resolved real record, the drawer says so plainly and
does NOT synthesise a PubMed link that might resolve to the wrong paper.

Decoupling: the panes that already hold ``Citation`` objects (chat, opening
interpretation) call ``register_citations(...)`` before they render. The drawer
reads them back by PMID from a session-scoped registry it owns. This keeps the
drawer self-contained — it never reaches into another pane's chat thread — while
guaranteeing the metadata is the agent's, not the drawer's.

Streamlit is imported lazily inside every function so importing ``ui.paper_drawer``
never needs a running server.
"""

from __future__ import annotations

import html
from typing import Any, Iterable, Optional

from agent.types import Citation

from ui import data_access as ui_data  # shared cached-read layer (infra)
from ui import format as ui_format  # shared pure formatters (infra)
from ui import state as ui_state

# ``ui_data``/``ui_format`` are imported per the pane's shared-infra contract.
# The drawer displays already-fetched citation metadata verbatim, so it does not
# format numbers or read a data frame here; the references keep the import graph
# uniform across panes and are available if a future field needs them.
_ = (ui_data, ui_format)

# --------------------------------------------------------------------------- #
# Session-scoped citation registry (owned by this module).
# Callers register the ``Citation`` objects they already have; the drawer reads
# them back by PMID. Keyed by PMID string so a re-render of the same citation is
# idempotent. This module owns this key exclusively; it is not in ui.state's
# schema because only the paper drawer's caller/reader contract uses it.
# --------------------------------------------------------------------------- #
_K_CITE_REGISTRY = "_ui_citation_registry"

_PUBMED_BASE = "https://pubmed.ncbi.nlm.nih.gov/"


def _ss() -> Any:
    """Return ``st.session_state`` (Streamlit imported lazily; import-safe)."""
    import streamlit as st

    return st.session_state


def _registry() -> dict[str, Citation]:
    """Return the live PMID->Citation registry dict (created on first use)."""
    ss = _ss()
    reg = ss.get(_K_CITE_REGISTRY)
    if reg is None:
        reg = {}
        ss[_K_CITE_REGISTRY] = reg
    return reg


def register_citations(citations: Iterable[Citation]) -> None:
    """Register already-fetched ``Citation`` objects so the drawer can show them.

    Called by whichever pane renders clickable citations (chat, opening
    interpretation). Each citation is stored under its PMID. Metadata is never
    altered here — the drawer displays exactly what the agent layer fetched. A
    citation with an empty PMID is skipped (nothing to key it on / link to).
    """
    reg = _registry()
    for cite in citations:
        pmid = (getattr(cite, "pmid", "") or "").strip()
        if pmid:
            reg[pmid] = cite


def get_registered_citation(pmid: Optional[str]) -> Optional[Citation]:
    """Return the registered ``Citation`` for a PMID, or None if none is known.

    Returning None (rather than fabricating) is deliberate: the drawer must not
    invent metadata for a PMID the agent layer never resolved.
    """
    if not pmid:
        return None
    return _registry().get(str(pmid).strip())


# --------------------------------------------------------------------------- #
# Rendering helpers (pure string builders; no Streamlit)
# --------------------------------------------------------------------------- #
def _esc(value: Any) -> str:
    """HTML-escape a value for safe injection into the drawer markup."""
    return html.escape(str(value), quote=True)


def _pubmed_url(cite: Citation) -> str:
    """Canonical PubMed URL for a citation.

    Prefers the URL the agent layer already stored; otherwise builds the standard
    ``pubmed.ncbi.nlm.nih.gov/{pmid}/`` form from the (agent-provided) PMID. The
    PMID itself is never invented — it is a field on the fetched ``Citation``.
    """
    url = (getattr(cite, "url", "") or "").strip()
    if url:
        return url
    return f"{_PUBMED_BASE}{_esc(cite.pmid)}/"


def _header_line(cite: Citation) -> str:
    """Journal · PMID eyebrow line (mono, accent), skipping empty fields."""
    bits: list[str] = []
    if cite.journal:
        bits.append(_esc(cite.journal))
    if cite.pmid:
        bits.append(f"PMID {_esc(cite.pmid)}")
    return " · ".join(bits) if bits else "citation"


def _author_line(cite: Citation) -> str:
    """Authors · year byline, tolerant of a missing year (0 / unset)."""
    authors = _esc(cite.authors) if cite.authors else "Authors unavailable"
    year = getattr(cite, "year", 0)
    if isinstance(year, int) and year > 0:
        return f"{authors} · {year}"
    return authors


def _drawer_html(cite: Citation) -> str:
    """Build the drawer's inner HTML for a resolved citation.

    Renders journal/PMID, title, authors/year, abstract, and a PubMed link. When
    the citation is flagged not-real (``is_real == False``), no live PubMed link
    is offered and an honest note replaces it — a fabricated link is worse than
    none. When the abstract is empty, an honest placeholder is shown rather than
    an empty box.
    """
    title = _esc(cite.title) if cite.title else "Title unavailable"
    abstract = _esc(cite.abstract) if cite.abstract else (
        "No abstract text was returned for this record."
    )

    is_real = bool(getattr(cite, "is_real", True))
    if is_real and cite.pmid:
        link = (
            f'<a class="lnk" href="{_esc(_pubmed_url(cite))}" '
            f'target="_blank" rel="noopener">view on PubMed →</a>'
        )
    else:
        link = (
            '<div class="mocknote">This reference is not a resolved live record; '
            "no PubMed link is shown. The agent does not link a PMID it could not "
            "verify.</div>"
        )

    fetched = (getattr(cite, "fetched_at", "") or "").strip()
    stamp = (
        f'<div class="mocknote">Fetched live via the PubMed / bioRxiv connector'
        f'{" · " + _esc(fetched) if fetched else ""}. '
        "The agent never writes a PMID from memory.</div>"
    )

    return (
        f'<div class="paper-drawer">'
        f'<div class="j">{_header_line(cite)}</div>'
        f"<h4>{title}</h4>"
        f'<div class="au">{_author_line(cite)}</div>'
        f'<div class="ab">{abstract}</div>'
        f"{link}"
        f"{stamp}"
        f"</div>"
    )


def _empty_html(pmid: Optional[str]) -> str:
    """Honest empty state when the active PMID has no registered citation."""
    ref = f" (PMID {_esc(pmid)})" if pmid else ""
    return (
        '<div class="paper-drawer">'
        '<div class="j">citation</div>'
        f"<h4>No paper metadata available{ref}</h4>"
        '<div class="ab">This citation has not been resolved through the '
        "literature connector yet, so there is nothing to show. The drawer only "
        "displays records the agent fetched live — it does not reconstruct a "
        "paper from a bare PMID.</div>"
        "</div>"
    )


# Drawer-local styling. Class names (.j / .au / .ab / .lnk / .mocknote) mirror the
# wireframe's `.paper` panel and reuse theme tokens injected by ui.theme.
_DRAWER_CSS = """
<style>
.paper-drawer .j {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .08em; color: var(--accent); margin-bottom: 8px;
}
.paper-drawer h4 { margin: 0 0 8px; font-size: 16px; line-height: 1.35; color: var(--ink); }
.paper-drawer .au { font-size: 12px; color: var(--muted); margin-bottom: 12px; }
.paper-drawer .ab {
  font-size: 13px; color: var(--ink); line-height: 1.55;
  border-top: 1px solid var(--hair2); padding-top: 12px;
}
.paper-drawer .lnk {
  display: inline-block; margin-top: 14px; font-family: var(--mono);
  font-size: 12px; color: var(--accent); text-decoration: none;
  border: 1px solid var(--hair); border-radius: 7px; padding: 7px 12px;
}
.paper-drawer .mocknote {
  margin-top: 16px; font-family: var(--mono); font-size: 10px; color: var(--faint);
}
</style>
"""


def render_paper_drawer() -> None:
    """Render the citation paper drawer for the active PMID, if it is open.

    Reads exactly two pieces of state via ``ui.state``:
      * ``is_paper_open()`` — whether the drawer should show at all, and
      * ``get_active_pmid()`` — which paper to show.

    Looks the PMID up in this module's citation registry (populated by callers via
    ``register_citations``). Renders the resolved ``Citation``'s journal / PMID /
    title / authors / abstract and a PubMed link. If the drawer is closed, renders
    nothing; if the PMID has no registered citation, renders an honest empty state
    rather than inventing metadata. A close control writes ``close_paper()``.
    """
    import streamlit as st

    if not ui_state.is_paper_open():
        return

    pmid = ui_state.get_active_pmid()
    cite = get_registered_citation(pmid)

    with st.container():
        st.markdown(_DRAWER_CSS, unsafe_allow_html=True)

        header_col, close_col = st.columns([6, 1])
        with header_col:
            st.markdown(
                '<div class="pano-eyebrow">Citation</div>',
                unsafe_allow_html=True,
            )
        with close_col:
            # Key is stable so repeated reruns reuse the same widget.
            if st.button("✕", key="paper_drawer_close", help="Close paper"):
                ui_state.close_paper()
                st.rerun()

        body = _drawer_html(cite) if cite is not None else _empty_html(pmid)
        st.markdown(body, unsafe_allow_html=True)


__all__ = [
    "render_paper_drawer",
    "register_citations",
    "get_registered_citation",
]
