"""Cross-cluster review panel (SKILL Step 4) — the whole set, read together.

Renders the coherence pass over all nine calls and the ONE grounded refinement it
surfaces (c8 Dendritic -> Plasmacytoid DC (pDC)). Everything shown comes straight
off :func:`ui.data_access.holistic` (the review the pipeline persisted, or a live
recompute):

* the ``coherence_notes`` are grounded observations whose every number was
  computed from ``agent.data`` (cell counts) and the authoritative ``CLUSTER_KEY``;
* the refinement's ``evidence_markers`` were READ from c8's jazzPanda markers, and
  the pDC reading is labelled a *literature direction* the biologist decides on,
  never a changed jazzPanda value.

The one live piece is the citation: for the refinement's ``lit_query`` we fetch
ONE real PMID through the PubMed connector and, only if it resolves, show it as a
clickable inline link (same style as the evidence table). On any connector
failure / timeout / zero hits we render an honest "literature thin" line and NEVER
fabricate a PMID (Panoscope's confident floor).

Streamlit is imported lazily inside every render function so ``import ui.holistic``
works with no server running.
"""

from __future__ import annotations

import html
from functools import lru_cache
from typing import Optional

from agent.holistic import HolisticReview, Refinement
from agent.types import Citation

from ui import data_access

# --------------------------------------------------------------------------- #
# Copy — kept as constants so prose never drifts.
# --------------------------------------------------------------------------- #
_TITLE = "Cross-cluster review"
_SUB = (
    "After annotating each cluster on its own, the whole set is re-read for "
    "coherence. Every number below is computed from the data; the one refinement "
    "is a proposal you decide on."
)
_COH_TITLE = "Coherence check"
_REFINE_TITLE = "Suggested refinement"

# Labels for the three grounded coherence notes, in the order
# ``agent.holistic._coherence_notes`` emits them. Applied positionally only when
# the count matches, so a change in the review never mislabels a note.
_COH_LABELS = ("Compartments", "Proportions", "Redundancy")

# One clickable citation is enough to ground the direction; keep the fetch small.
_LIT_MAX_RESULTS = 3

# --------------------------------------------------------------------------- #
# Panel styling. Reuses the design tokens from ``ui.theme`` (--accent, --hair,
# --muted, --ink, --paper, --mono, --absent, .gene). Modern surface cards with an
# all-around hairline (no left-accent bars), a clear type hierarchy, and generous
# rhythm. Re-injecting across reruns is harmless.
# --------------------------------------------------------------------------- #
_HOL_CSS = """
.pano-hol { margin-top: 30px; }
.pano-hol-h { font-family: var(--sans); font-size: 18px; font-weight: 600;
              letter-spacing: -.01em; color: var(--ink); margin: 0 0 5px; }
.pano-hol-sub { font-size: 13px; color: var(--muted); line-height: 1.55;
                max-width: 74ch; margin: 0 0 18px; }

/* A surface card: all-around hairline, soft radius, white paper on the grey page. */
.pano-card { background: var(--paper); border: 1px solid var(--hair);
             border-radius: 14px; padding: 4px 20px 8px; margin-bottom: 16px; }
.pano-card-head { display: flex; align-items: center; gap: 10px;
                  padding: 14px 0 12px; border-bottom: 1px solid var(--hair2); }
.pano-card-title { font-family: var(--sans); font-size: 13px; font-weight: 600;
                   color: var(--ink); letter-spacing: -.01em; }
.pano-card-title .kicker { font-family: var(--mono); font-size: 10px; font-weight: 500;
                   text-transform: uppercase; letter-spacing: .1em; color: var(--faint); }
.pano-pass { margin-left: auto; font-family: var(--mono); font-size: 10px; font-weight: 600;
             letter-spacing: .04em; color: var(--accent); background: var(--accent-soft);
             padding: 3px 10px; border-radius: 999px; display: inline-flex; align-items: center; gap: 6px; }
.pano-pass::before { content: '\\2713'; font-size: 10px; }

/* Coherence items: a check glyph, a bold lead-in label, then the grounded note. */
.pano-coh-item { display: grid; grid-template-columns: 20px 1fr; gap: 12px;
                 padding: 14px 0; border-bottom: 1px solid var(--hair2); }
.pano-coh-item:last-child { border-bottom: 0; }
.pano-coh-check { color: var(--accent); font-size: 13px; font-weight: 700; line-height: 1.5; }
.pano-coh-body { font-size: 13px; color: var(--muted); line-height: 1.6; }
.pano-coh-body .lbl { font-weight: 600; color: var(--ink); margin-right: 4px; }

/* Refinement card: the proposal reads as an editorial call, markers as chips. */
.pano-ref-call { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
                 padding: 16px 0 10px; }
.pano-ref-call .cid { font-family: var(--mono); font-size: 11px; color: var(--faint); }
.pano-ref-call .from { font-size: 14px; color: var(--muted); }
.pano-ref-call .arr { font-family: var(--mono); font-size: 15px; color: var(--accent); }
.pano-ref-call .to { font-size: 19px; font-weight: 700; letter-spacing: -.02em; color: var(--ink); }
.pano-ref-tag { font-family: var(--mono); font-size: 9px; text-transform: uppercase;
                letter-spacing: .08em; color: var(--accent); background: var(--accent-soft);
                padding: 3px 9px; border-radius: 6px; }
.pano-ref-markers { display: flex; align-items: center; gap: 7px; flex-wrap: wrap;
                    padding: 4px 0 12px; }
.pano-ref-markers .k { font-family: var(--mono); font-size: 10px; color: var(--faint);
                       text-transform: uppercase; letter-spacing: .06em; margin-right: 3px; }
.pano-ref-chip { font-family: var(--mono); font-size: 12px; font-weight: 600; color: var(--ink);
                 background: var(--hair2); border-radius: 6px; padding: 3px 9px; }
.pano-ref-rat { font-size: 13px; color: var(--muted); line-height: 1.6; max-width: 78ch;
                padding-bottom: 14px; }
.pano-ref-lit { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
                border-top: 1px solid var(--hair2); padding: 13px 0 15px; }
.pano-ref-lit .q { font-family: var(--mono); font-size: 10px; text-transform: uppercase;
                   letter-spacing: .06em; color: var(--faint); }
.pano-ref-cite { font-family: var(--mono); font-size: 12px; color: var(--accent);
                 text-decoration: none; display: inline-flex; align-items: center; gap: 5px; }
.pano-ref-cite:hover { text-decoration: underline; }
.pano-ref-thin { font-family: var(--mono); font-size: 11px; color: var(--absent); }
"""


def _inject_local_css() -> None:
    import streamlit as st

    st.markdown(f"<style>{_HOL_CSS}</style>", unsafe_allow_html=True)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


# --------------------------------------------------------------------------- #
# Live citation for the refinement's literature direction.
# Real-or-absent: return a resolved Citation only. On any connector failure /
# timeout / zero hits, return None so the caller renders the honest fallback.
# Cached per query so the Summary page never re-hits the connector on a rerun.
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def _first_real_citation(lit_query: str) -> Optional[Citation]:
    """Fetch ONE real PMID for ``lit_query`` via the live PubMed connector.

    Goes through ``agent.tools.literature_search`` (the same real-or-absent path
    the agent uses). Returns ``None`` on connector failure or thin literature — it
    NEVER fabricates a PMID. On success returns a real Citation from the record.
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
            abstract="",
            url=str(rec.get("url", "")).strip(),
            stance="context",
            is_real=True,
        )
    return None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _coherence_card_html(review: HolisticReview) -> str:
    """The coherence card: a Passed pill + labelled grounded checks (verbatim)."""
    notes = review.coherence_notes
    labelled = len(notes) == len(_COH_LABELS)
    pass_pill = '<span class="pano-pass">Passed</span>' if review.set_is_coherent else ""

    items = []
    for i, note in enumerate(notes):
        lbl = f'<span class="lbl">{_esc(_COH_LABELS[i])}.</span> ' if labelled else ""
        items.append(
            '<div class="pano-coh-item">'
            '<div class="pano-coh-check">✓</div>'
            f'<div class="pano-coh-body">{lbl}{_esc(note)}</div>'
            "</div>"
        )
    return (
        '<div class="pano-card">'
        '<div class="pano-card-head">'
        f'<span class="pano-card-title">{_esc(_COH_TITLE)}</span>{pass_pill}'
        "</div>"
        + "".join(items)
        + "</div>"
    )


def _cite_link_html(cite: Citation) -> str:
    """An inline clickable PubMed link (same style as the evidence-table cite)."""
    first = (cite.authors or "").split(",")[0].split(";")[0].strip()
    year = f" {cite.year}" if cite.year else ""
    label = f"{first}{year}".strip() or f"PMID {cite.pmid}"
    href = f"https://pubmed.ncbi.nlm.nih.gov/{_esc(cite.pmid)}/"
    return (
        f'<a class="pano-ref-cite" href="{href}" target="_blank" rel="noopener">'
        f"\U0001f4c4 {_esc(label)} ↗</a>"
    )


def _refinement_card_html(ref: Refinement, cite: Optional[Citation]) -> str:
    """The refinement card: the call, marker chips, rationale, and a live cite."""
    chips = "".join(f'<span class="pano-ref-chip">{_esc(m)}</span>' for m in ref.evidence_markers)
    lit = (
        _cite_link_html(cite)
        if (cite is not None and cite.pmid)
        else f'<span class="pano-ref-thin">literature thin — re-check ({_esc(ref.lit_query)})</span>'
    )
    return (
        '<div class="pano-card">'
        '<div class="pano-card-head">'
        f'<span class="pano-card-title">{_esc(_REFINE_TITLE)}</span>'
        "</div>"
        '<div class="pano-ref-call">'
        f'<span class="cid">{_esc(ref.cluster)}</span>'
        f'<span class="from">{_esc(ref.from_call)}</span>'
        '<span class="arr">→</span>'
        f'<span class="to">{_esc(ref.to_call)}</span>'
        '<span class="pano-ref-tag">subtype · you decide</span>'
        "</div>"
        f'<div class="pano-ref-markers"><span class="k">driving markers</span>{chips}</div>'
        f'<div class="pano-ref-rat">{_esc(ref.rationale)}. This is a subtype sharpening '
        "within the same immune/dendritic lineage; the jazzPanda numbers do not change.</div>"
        f'<div class="pano-ref-lit"><span class="q">literature direction, live</span>{lit}</div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# Public: the panel
# --------------------------------------------------------------------------- #
def render_holistic_review() -> None:
    """Render the cross-cluster review: heading, coherence card, refinement card.

    Sources the review from :func:`ui.data_access.holistic` (tree-first). The
    coherence card shows the grounded checks with a Passed pill; the refinement
    card shows the c8 -> pDC proposal with a LIVE citation rendered as an inline
    PubMed link, degrading to an honest "literature thin" line and never a
    fabricated PMID.
    """
    import streamlit as st

    _inject_local_css()
    review = data_access.holistic()

    parts = ['<div class="pano-hol">']
    parts.append(f'<div class="pano-hol-h">{_esc(_TITLE)}</div>')
    parts.append(f'<div class="pano-hol-sub">{_esc(_SUB)}</div>')
    parts.append(_coherence_card_html(review))
    for ref in review.refinements:
        cite = _first_real_citation(ref.lit_query)
        parts.append(_refinement_card_html(ref, cite))
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


__all__ = ["render_holistic_review"]
