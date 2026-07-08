"""Evidence table — the panel that renders the panel-absence rule visible.

``render_evidence_table(cluster)`` draws one cluster's marker evidence: each row
carries the gene, its jazzPanda numbers (glm_coef + pearson), a specificity
strip across all nine clusters, and the ROLE column
(``supports`` / ``expected, absent`` / ``off-panel``). Every value on screen
comes straight from the cached ``ClusterVerdict`` — this module computes nothing.

Two grounding-relevant guarantees live here:

1. **Click-to-pin never recomputes.** Each row has a pin control that calls
   ``ui.state.toggle_pin`` — it sets ``pinned_marker`` and returns. It is a
   viewing control (it drives the linked spatial views), never a value. The
   verdict is read from ``ui.data_access.verdict_for`` (cached, computed once).

2. **The panel-absence rule is on screen.** A cluster's canonical markers that
   were never measured (off the panel) render as explicit
   ``expected, absent (off-panel — not measured)`` rows, sourced from the
   verdict's ``offpanel_notes``. Their absence is shown as uninformative, never
   as evidence against the call.

Streamlit is imported lazily inside ``render_evidence_table`` so this module
imports cleanly with no server running.
"""

from __future__ import annotations

import html
from typing import Optional

from agent.config import CLUSTER_ORDER
from agent.types import ClusterVerdict, MarkerEvidence, OffPanelNote

from ui import data_access, format as fmt, state

# --------------------------------------------------------------------------- #
# How many rows to show. A cluster can carry dozens of assigned markers (c1 has
# 84); showing all of them buries the role story. We always keep every canonical
# marker (the drivers the call rests on), then top up with the strongest
# non-canonical supporters by glm_coef until we reach the cap. Nothing is
# recomputed — this only *selects* rows the verdict already produced.
# --------------------------------------------------------------------------- #
MAX_SUPPORT_ROWS: int = 12

# Specificity-strip teal ramp (ported verbatim from the wireframe's ``teal(v)``).
# v is a 0..1 spatial-specificity value (pearson); higher -> deeper teal.
_STRIP_EMPTY = "#DCE3E5"  # "not an assigned marker in this cluster" (faint)


def _teal(value: float) -> str:
    """Step teal ramp for the specificity strip (wireframe parity)."""
    v = value
    if v < 0.22:
        return "#DCE3E5"
    if v < 0.40:
        return "#B7CDD0"
    if v < 0.60:
        return "#7FB0B6"
    if v < 0.80:
        return "#3C8F99"
    return "#0F5B65"


# --------------------------------------------------------------------------- #
# Row selection (pure; selects verdict rows, never recomputes a value)
# --------------------------------------------------------------------------- #
def _rows_to_show(verdict: ClusterVerdict) -> list[MarkerEvidence]:
    """Canonical markers (always) + strongest non-canonical supporters, capped.

    Evidence is already glm_coef-descending from the verdict. We preserve that
    order but guarantee every canonical marker is present (drivers must never be
    hidden by the cap), then fill the remaining budget with non-canonical rows.
    """
    canonical = [e for e in verdict.evidence if e.is_canonical]
    non_canonical = [e for e in verdict.evidence if not e.is_canonical]
    budget = max(0, MAX_SUPPORT_ROWS - len(canonical))
    kept = {id(e) for e in canonical} | {id(e) for e in non_canonical[:budget]}
    # Re-emit in the verdict's original glm_coef-descending order.
    return [e for e in verdict.evidence if id(e) in kept]


# --------------------------------------------------------------------------- #
# HTML fragments (pure string builders; escape all gene-derived text)
# --------------------------------------------------------------------------- #
def _specificity_strip(ev: MarkerEvidence, selected_cluster: str) -> str:
    """Nine-cell strip: the marker's own cluster lit by its pearson, rest faint.

    A jazzPanda marker is assigned to exactly one ``top_cluster``; that is the
    only cluster where it carries a real spatial signal. So the strip lights that
    one cell (teal keyed to the marker's pearson) and leaves the other eight
    faint — an honest "specific to one cluster" picture, not an invented profile.
    The selected cluster's cell gets a ring so the biologist sees where they are.
    """
    cells: list[str] = []
    gene = html.escape(str(ev.gene))
    for c in CLUSTER_ORDER:
        is_home = c == ev.top_cluster
        color = _teal(ev.pearson) if is_home else _STRIP_EMPTY
        ring = " pano-sc-sel" if c == selected_cluster else ""
        if is_home:
            title = f"{gene} · {c}: pearson {ev.pearson:.2f}"
        else:
            title = f"{gene} · {c}: not an assigned marker"
        cells.append(
            f'<span class="pano-sc{ring}" '
            f'style="background:{color}" title="{html.escape(title)}"></span>'
        )
    return f'<span class="pano-strip">{"".join(cells)}</span>'


def _marker_row_html(
    ev: MarkerEvidence,
    *,
    selected_cluster: str,
    is_pinned: bool,
) -> str:
    """One measured-marker row: gene, glm_coef, pearson, strip, role chip."""
    gene = html.escape(str(ev.gene))
    role_css, role_txt = fmt.role_chip(ev.role)
    strip = _specificity_strip(ev, selected_cluster)
    glm = fmt.num_fmt(ev.glm_coef, 2)
    pear = fmt.num_fmt(ev.pearson, 2)
    dim = "" if ev.role == "supports" else " dim"
    pin = '<span class="pin">📌</span>' if is_pinned else ""
    canon = (
        '<span class="pano-canon" title="canonical marker for this cell type">canonical</span>'
        if ev.is_canonical
        else ""
    )
    pinned_cls = " pinned" if is_pinned else ""
    return (
        f'<div class="pano-evrow{pinned_cls}">'
        f'<div class="pano-ev-gene"><span class="gene">{gene}</span>{pin}{canon}</div>'
        f'<div class="pano-ev-num"><span class="num{dim}">{glm}</span>'
        f'<span class="num dim pano-ev-sub">pearson {pear}</span></div>'
        f'<div class="pano-ev-strip">{strip}</div>'
        f'<div class="pano-ev-role"><span class="role {role_css}">{html.escape(role_txt)}</span></div>'
        f"</div>"
    )


def _offpanel_row_html(note: OffPanelNote, selected_cluster: str) -> str:
    """One off-panel canonical row — the panel-absence rule, made visible.

    No numbers, no pin: the gene was never measured, so there is no spatial
    signal to show and nothing to drive. The strip is drawn all-faint (measured
    nowhere) and the role chip reads ``off-panel``. The caption states plainly
    that its absence is not evidence against the call.
    """
    gene = html.escape(str(note.gene))
    role_css, _ = fmt.role_chip("off_panel")
    faint_cells = "".join(
        f'<span class="pano-sc{" pano-sc-sel" if c == selected_cluster else ""}" '
        f'style="background:{_STRIP_EMPTY}" '
        f'title="{gene} · {c}: never measured (off-panel)"></span>'
        for c in CLUSTER_ORDER
    )
    strip = f'<span class="pano-strip">{faint_cells}</span>'
    return (
        '<div class="pano-evrow pano-evrow-off">'
        f'<div class="pano-ev-gene"><span class="gene dimgene">{gene}</span>'
        '<span class="pano-canon pano-canon-off" '
        'title="canonical marker for this cell type, but not on the panel">canonical</span></div>'
        '<div class="pano-ev-num"><span class="num dim">—</span>'
        '<span class="num dim pano-ev-sub">not measured</span></div>'
        f'<div class="pano-ev-strip">{strip}</div>'
        f'<div class="pano-ev-role"><span class="role {role_css}">⊘ expected, absent</span></div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# One-time CSS for the evidence-table grid (chips/tokens come from ui.theme)
# --------------------------------------------------------------------------- #
_EVIDENCE_CSS = """
<style>
.pano-ev-head, .pano-evrow {
  display: grid;
  grid-template-columns: minmax(120px, 1.4fr) minmax(96px, 1fr) minmax(150px, 1.3fr) minmax(140px, 1fr);
  gap: 12px; align-items: center;
}
.pano-ev-head {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--faint); font-weight: 500;
  padding: 0 4px 8px;
}
.pano-ev-head .r { text-align: left; }
.pano-evrow {
  border-top: 1px solid var(--hair2); padding: 9px 4px; min-height: 44px;
}
.pano-evrow.pinned { background: var(--accent-soft); border-radius: 7px; }
.pano-evrow-off { background: var(--offpanel-bg); border-radius: 7px; }
.pano-ev-gene { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.pano-ev-gene .gene.dimgene { color: var(--faint); }
.pano-ev-num { display: flex; flex-direction: column; line-height: 1.35; }
.pano-ev-sub { font-size: 10px; }
.pano-canon {
  font-family: var(--mono); font-size: 9px; color: var(--accent);
  background: var(--accent-soft); padding: 1px 6px; border-radius: 4px;
}
.pano-canon-off { color: var(--offpanel); background: var(--offpanel-bg); }
.pano-strip { display: inline-flex; gap: 3px; }
.pano-sc { width: 15px; height: 15px; border-radius: 3px; display: inline-block; }
.pano-sc-sel { box-shadow: 0 0 0 2px var(--ink); }
.pano-panelrule {
  margin: 14px 0 8px; font-family: var(--mono); font-size: 10px;
  text-transform: uppercase; letter-spacing: .08em; color: var(--faint);
}
.pano-abscap {
  font-family: var(--sans); font-size: 12px; color: var(--muted);
  margin: 4px 4px 0; line-height: 1.5;
}
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint);
  padding: 16px 4px;
}
</style>
"""


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
def render_evidence_table(cluster: str) -> None:
    """Render the marker-evidence table for ``cluster`` into the current column.

    Reads the cached verdict (``ui.data_access.verdict_for`` — computed once) and
    the pinned marker (``ui.state.get_pinned_marker``). Draws a header, one row
    per shown marker with a pin control that only sets ``pinned_marker``, then the
    off-panel canonical rows that render the panel-absence rule. Recomputes
    nothing; pinning drives the spatial views without touching a value.
    """
    import streamlit as st

    verdict: ClusterVerdict = data_access.verdict_for(cluster)
    pinned: Optional[str] = state.get_pinned_marker()
    selected = state.get_selected_cluster()

    st.markdown(_EVIDENCE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="pano-eyebrow">Marker evidence · '
        f"{html.escape(verdict.cell_type.replace('_', ' '))} · "
        f"{len(verdict.evidence)} assigned markers</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pano-ev-head">'
        '<span>Gene</span><span class="r">jazzPanda signal</span>'
        "<span>Specificity across clusters</span><span>Role</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    rows = _rows_to_show(verdict)
    if not rows:
        st.markdown(
            '<div class="pano-ev-empty">No assigned markers for this cluster.</div>',
            unsafe_allow_html=True,
        )

    # Each row = a narrow pin-button column + a wide HTML-content column. The
    # pin button is the click-to-PIN control; it sets pinned_marker and returns.
    for ev in rows:
        is_pinned = pinned is not None and ev.gene.upper() == pinned.upper()
        pin_col, body_col = st.columns([0.06, 0.94], vertical_alignment="center")
        with pin_col:
            _pin_button(st, ev.gene, is_pinned)
        with body_col:
            st.markdown(
                _marker_row_html(
                    ev, selected_cluster=selected, is_pinned=is_pinned
                ),
                unsafe_allow_html=True,
            )

    _render_offpanel_section(st, verdict, selected)


def _pin_button(st, gene: str, is_pinned: bool) -> None:
    """Draw the pin toggle for one marker. Sets ``pinned_marker`` ONLY, no recompute.

    Uses a stable per-gene key so Streamlit tracks each row's button across
    reruns. On click it calls ``state.toggle_pin`` (a single session-state write)
    and lets Streamlit rerun to re-read the already-cached verdict — the value is
    never recomputed by a pin.
    """
    label = "📌" if is_pinned else "📍"
    help_txt = (
        "Unpin — stop driving the spatial views"
        if is_pinned
        else "Pin to drive the spatial views"
    )
    if st.button(
        label,
        key=f"pin_{gene}",
        help=help_txt,
        use_container_width=True,
    ):
        state.toggle_pin(gene)
        st.rerun()


def _render_offpanel_section(st, verdict: ClusterVerdict, selected: str) -> None:
    """Render the off-panel canonical rows (the panel-absence rule on screen).

    Nothing here down-weights the call: these genes were never measured, so their
    absence is uninformative. If the cell type has no off-panel canonical markers
    we say so plainly rather than implying the panel covered everything.
    """
    st.markdown(
        '<div class="pano-panelrule">Panel-absence check · canonical markers not on the panel</div>',
        unsafe_allow_html=True,
    )
    notes = verdict.offpanel_notes
    if not notes:
        st.markdown(
            '<div class="pano-abscap">Every canonical marker for '
            f"{html.escape(verdict.cell_type.replace('_', ' '))} that jazzPanda could "
            "model is on the panel — no off-panel absences to flag here.</div>",
            unsafe_allow_html=True,
        )
        return

    off_html = "".join(_offpanel_row_html(n, selected) for n in notes)
    st.markdown(off_html, unsafe_allow_html=True)
    st.markdown(
        '<div class="pano-abscap">These canonical markers were <b>never measured</b> '
        "on this panel. Their absence is not evidence against the call — a missing "
        "off-panel gene tells us nothing about the cell type.</div>",
        unsafe_allow_html=True,
    )


__all__ = ["render_evidence_table"]
