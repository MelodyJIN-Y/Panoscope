"""Holistic review panel — "all 9 clusters together" (SKILL Step 4).

Renders the cross-cluster coherence pass and the ONE grounded refinement it
surfaces (c8 Dendritic -> Plasmacytoid DC (pDC)). Everything shown here comes
straight off :func:`agent.holistic.holistic_review`:

* the ``coherence_notes`` are grounded observations whose every number was
  computed at runtime from ``agent.data`` (cell counts) and the authoritative
  ``CLUSTER_KEY`` — this panel never recomputes or invents one;
* the refinement's ``evidence_markers`` were READ from c8's jazzPanda markers,
  and the pDC reading is labelled a *literature direction* (a proposal the
  biologist decides on), never a changed jazzPanda value.

The one live piece is the citation: for the refinement's ``lit_query`` we fetch
ONE real PMID through the PubMed connector (via ``agent.tools.literature_search``)
and, only if it resolves to a real record, show it as a clickable button that
opens the citation drawer. On any connector failure / timeout / zero hits we
render "literature: connector unavailable — re-check" and NEVER fabricate a PMID
(Panoscope's confident floor; a fabricated citation is the worst failure).

Streamlit is imported lazily inside every function so ``import ui.holistic``
works with no server running (the module touches no ``st.*`` at import time).
"""

from __future__ import annotations

import html
from typing import Optional

from agent.holistic import HolisticReview, Refinement, holistic_review
from agent.types import Citation

from ui import paper_drawer
from ui import state

# --------------------------------------------------------------------------- #
# Copy — the panel's headings. Kept as constants so prose never drifts.
# --------------------------------------------------------------------------- #
_PANEL_TITLE = "Holistic review — all 9 clusters together"
_PANEL_SUB = (
    "After annotating each cluster on its own, re-read the whole set for "
    "coherence. Numbers below are computed from the data; the one refinement is "
    "a proposal you decide on."
)
_COHERENCE_HEAD = "Coherence check"
_REFINE_HEAD = "One refinement to consider"
_CONNECTOR_UNAVAILABLE = "literature: connector unavailable — re-check"

# How many real PMIDs to pull for the refinement's lit_query. One clickable
# citation is enough to ground the direction; we keep the fetch small.
_LIT_MAX_RESULTS = 3

# --------------------------------------------------------------------------- #
# Panel-local styling. Class names (.hol-*) reuse the design tokens injected by
# ``ui.theme.inject_css`` (--hair, --accent, --muted, --absent, --mono, etc.),
# and lean on ``.pcite`` / ``.tension`` / ``.gene`` which ``ui.theme`` already
# defines. Re-injecting the same block across reruns is harmless.
# --------------------------------------------------------------------------- #
_HOL_CSS = """
.hol-sub { font-size:12px; color:var(--muted); line-height:1.45; margin-bottom:12px; }
.hol-head { font-family:var(--mono); font-size:10px; text-transform:uppercase;
            letter-spacing:.1em; color:var(--faint); font-weight:500; margin:14px 0 8px; }
.hol-note { font-size:13px; color:var(--ink); line-height:1.5;
            padding:9px 11px; border:1px solid var(--hair); border-left:3px solid var(--accent);
            border-radius:8px; background:var(--paper); margin-bottom:8px; }
.hol-card { border:1px solid var(--hair); border-radius:10px; padding:12px 13px;
            background:var(--paper); }
.hol-call { display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; margin-bottom:6px; }
.hol-call .cid { font-family:var(--mono); font-size:11px; color:var(--faint); }
.hol-call .arr { font-family:var(--mono); font-size:13px; color:var(--accent); }
.hol-call .from { font-size:14px; color:var(--muted); }
.hol-call .to   { font-size:15px; font-weight:600; color:var(--ink); }
.hol-tag { font-family:var(--mono); font-size:9px; text-transform:uppercase;
           letter-spacing:.08em; color:var(--absent); background:var(--absent-bg);
           padding:2px 7px; border-radius:5px; }
.hol-ev { font-family:var(--mono); font-size:11px; color:var(--muted);
          margin:6px 0; display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.hol-ev .lbl { color:var(--faint); }
.hol-rat { font-size:12px; color:var(--muted); line-height:1.45; margin:6px 0 2px; }
.hol-lit { font-family:var(--mono); font-size:11px; margin-top:8px; }
.hol-lit.thin { color:var(--absent); }
.hol-lit .q { color:var(--faint); }
"""


def _inject_local_css() -> None:
    """Inject the panel styles once per rerun (guarded; import-safe)."""
    import streamlit as st

    st.markdown(f"<style>{_HOL_CSS}</style>", unsafe_allow_html=True)


def _esc(value: object) -> str:
    """HTML-escape a value for safe injection into the panel markup."""
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Live citation for the refinement's literature direction.
# Real-or-absent: return a resolved Citation only. On any connector failure /
# timeout / zero hits, return None so the caller renders the honest fallback.
# --------------------------------------------------------------------------- #
def _first_real_citation(lit_query: str) -> Optional[Citation]:
    """Fetch ONE real PMID for ``lit_query`` via the live PubMed connector.

    Goes through ``agent.tools.literature_search`` (the same real-or-absent path
    the agent uses): ``ok=False`` when the connector is down, an empty result set
    when the literature is thin. In every non-resolving case this returns ``None``
    — it NEVER fabricates a PMID. On success it returns a real, is_real Citation
    built from the connector's own record.
    """
    q = (lit_query or "").strip()
    if not q:
        return None

    from agent import tools

    try:
        env = tools.literature_search(q, max_results=_LIT_MAX_RESULTS)
    except Exception:  # noqa: BLE001 - a lookup error is a fallback, never a crash
        return None

    if not env.get("ok"):
        return None
    results = (env.get("data") or {}).get("results") or []
    for rec in results:
        pmid = str(rec.get("pmid", "")).strip()
        if not pmid.isdigit():
            continue
        return Citation(
            pmid=pmid,
            title=str(rec.get("title", "")).strip(),
            authors=str(rec.get("authors", "")).strip(),
            year=int(rec.get("year", 0) or 0),
            journal=str(rec.get("journal", "")).strip(),
            abstract="",  # abstract is fetched on demand by the paper drawer
            url=str(rec.get("url", "")).strip(),
            stance="context",
            is_real=True,
        )
    return None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _render_coherence(review: HolisticReview) -> None:
    """Render each grounded coherence note as its own card (verbatim, no recompute)."""
    import streamlit as st

    st.markdown(f'<div class="hol-head">{_COHERENCE_HEAD}</div>', unsafe_allow_html=True)
    for note in review.coherence_notes:
        st.markdown(f'<div class="hol-note">{_esc(note)}</div>', unsafe_allow_html=True)


def _cite_label(cite: Citation) -> str:
    """Short citation button label, e.g. ``[doc] Reizis 2019`` or ``[doc] PMID:123``."""
    first = (cite.authors or "").split(",")[0].split(";")[0].strip()
    year = f" {cite.year}" if cite.year else ""
    if first:
        return f"\U0001f4c4 {first}{year}"
    return f"\U0001f4c4 PMID:{cite.pmid}"


def _render_refinement(ref: Refinement) -> None:
    """Render the c8 -> pDC refinement card with a live citation (or honest fallback).

    The call, evidence markers, and rationale come straight off the ``Refinement``
    (grounded). The citation is fetched live: only if it resolves to a real record
    is a clickable PMID shown; otherwise the honest "connector unavailable" line.
    """
    import streamlit as st

    st.markdown(f'<div class="hol-head">{_REFINE_HEAD}</div>', unsafe_allow_html=True)

    markers = " ".join(
        f'<span class="gene">{_esc(m)}</span>' for m in ref.evidence_markers
    )
    st.markdown(
        f"""<div class="hol-card">
  <div class="hol-call">
    <span class="cid">{_esc(ref.cluster)}</span>
    <span class="from">{_esc(ref.from_call)}</span>
    <span class="arr">&rarr;</span>
    <span class="to">{_esc(ref.to_call)}</span>
    <span class="hol-tag">subtype — you decide</span>
  </div>
  <div class="hol-ev"><span class="lbl">driving markers:</span> {markers}</div>
  <div class="hol-rat">{_esc(ref.rationale)}. This is a subtype sharpening within
    the same immune/dendritic lineage — the jazzPanda numbers do not change.</div>
</div>""",
        unsafe_allow_html=True,
    )

    # Live literature direction — real PMID or an honest fallback, never fabricated.
    cite = _first_real_citation(ref.lit_query)
    if cite is not None and cite.pmid:
        paper_drawer.register_citations([cite])
        st.markdown(
            '<div class="hol-lit"><span class="q">literature direction — '
            f"live: {_esc(ref.lit_query)}</span></div>",
            unsafe_allow_html=True,
        )
        if st.button(
            _cite_label(cite),
            key=f"hol_cite_{cite.pmid}",
            help="Open the paper in the citation drawer",
        ):
            state.open_paper(cite.pmid)
            st.rerun()
    else:
        st.markdown(
            f'<div class="hol-lit thin">{_CONNECTOR_UNAVAILABLE} '
            f'<span class="q">(query: {_esc(ref.lit_query)})</span></div>',
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Public: the panel
# --------------------------------------------------------------------------- #
def render_holistic_review() -> None:
    """Render the holistic cross-cluster review panel.

    Calls :func:`agent.holistic.holistic_review` and renders, in order: the
    grounded coherence notes (all numbers computed from source), then the single
    c8 Dendritic -> Plasmacytoid DC (pDC) refinement as a card with its evidence
    markers, rationale, and a LIVE citation for its literature direction. The
    citation is shown as a clickable PMID only if it resolves to a real record;
    on any connector failure it degrades to an honest "connector unavailable —
    re-check" line and never fabricates a PMID.
    """
    import streamlit as st

    _inject_local_css()
    review = holistic_review()

    with st.container():
        st.markdown(
            f'<div class="pano-eyebrow">{_esc(_PANEL_TITLE)}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="hol-sub">{_esc(_PANEL_SUB)}</div>', unsafe_allow_html=True)
        _render_coherence(review)
        for ref in review.refinements:
            _render_refinement(ref)


__all__ = ["render_holistic_review"]
