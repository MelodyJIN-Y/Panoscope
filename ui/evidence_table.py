"""Evidence table — the marker rows a cluster's call rests on, stripped to essentials.

``render_evidence_table(cluster)`` draws one cluster's marker evidence: each row
carries the gene (with a ``canonical`` tag when it is a canonical marker for the
cell type) and its jazzPanda numbers (glm_coef + pearson). Every value on screen
comes straight from the cached ``ClusterVerdict`` — this module computes nothing.

The one grounding-relevant guarantee that lives here:

**Click-to-select never recomputes.** Each row has a borderless teal select-dot
that calls ``ui.state.toggle_marker`` — it toggles the gene in the ordered
multi-select set and returns. It is a viewing control (it drives the linked
spatial small-multiples), never a value. The verdict is read from
``ui.data_access.verdict_for`` (cached, computed once).

The panel-absence rule is no longer rendered here: it now lives in the agent's
opening interpretation (``agent/fallback.py``). Numbers still trace to jazzPanda.

Streamlit is imported lazily inside ``render_evidence_table`` so this module
imports cleanly with no server running.
"""

from __future__ import annotations

import html

from agent.types import ClusterVerdict, MarkerEvidence

from ui import data_access, format as fmt, state

# --------------------------------------------------------------------------- #
# How many rows to show. A cluster can carry dozens of assigned markers (c1 has
# 84); showing all of them buries the drivers. We always keep every canonical
# marker (the drivers the call rests on), then top up with the strongest
# non-canonical supporters by glm_coef until we reach the cap. Nothing is
# recomputed — this only *selects* rows the verdict already produced.
# --------------------------------------------------------------------------- #
MAX_SUPPORT_ROWS: int = 8


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
# HTML fragment (pure string builder; escape all gene-derived text)
# --------------------------------------------------------------------------- #
def _marker_row_html(ev: MarkerEvidence, *, is_selected: bool) -> str:
    """One marker row's content: gene (+ canonical tag) and its glm_coef/pearson.

    The select-dot is a separate Streamlit column (rendered by ``_select_dot``);
    this builds only the two content columns. Selected rows get a soft tint.
    """
    gene = html.escape(str(ev.gene))
    glm = fmt.num_fmt(ev.glm_coef, 2)
    pear = fmt.num_fmt(ev.pearson, 2)
    canon = (
        '<span class="pano-canon" title="canonical marker for this cell type">canonical</span>'
        if ev.is_canonical
        else ""
    )
    selected_cls = " selected" if is_selected else ""
    return (
        f'<div class="pano-evrow{selected_cls}">'
        f'<div class="pano-ev-gene"><span class="gene">{gene}</span>{canon}</div>'
        f'<div class="pano-ev-num"><span class="num">{glm}</span>'
        f'<span class="num dim pano-ev-sub">pearson {pear}</span></div>'
        f"</div>"
    )


# --------------------------------------------------------------------------- #
# One-time CSS for the evidence-table grid (chips/tokens come from ui.theme;
# the select-dot chrome comes from ui.theme's `.pano-select-dot`)
# --------------------------------------------------------------------------- #
_EVIDENCE_CSS = """
<style>
.pano-ev-head, .pano-evrow {
  display: grid;
  grid-template-columns: minmax(140px, 1.6fr) minmax(110px, 1fr);
  gap: 12px; align-items: center;
}
.pano-ev-head {
  font-family: var(--mono); font-size: 10px; text-transform: uppercase;
  letter-spacing: .05em; color: var(--faint); font-weight: 500;
  padding: 0 4px 8px;
}
.pano-ev-head .r { text-align: left; }
.pano-evrow {
  border-top: 1px solid var(--hair2); padding: 9px 4px; min-height: 40px;
}
.pano-evrow.selected { background: var(--accent-soft); border-radius: 7px; }
.pano-ev-gene { display: flex; align-items: center; gap: 7px; flex-wrap: wrap; }
.pano-ev-num { display: flex; flex-direction: column; line-height: 1.35; }
.pano-ev-sub { font-size: 10px; }
.pano-canon {
  font-family: var(--mono); font-size: 9px; color: var(--accent);
  background: var(--accent-soft); padding: 1px 6px; border-radius: 4px;
}
.pano-ev-empty {
  font-family: var(--mono); font-size: 12px; color: var(--faint);
  padding: 16px 4px;
}
/* Tighten the select-dot column (keyed container st-key-evrows_*) so the dot
   sits right next to the gene name and lines up with it. */
div[class*="st-key-evrows"] [data-testid="stHorizontalBlock"] { gap: 4px !important; align-items: center; }
div[class*="st-key-evrows"] [data-testid="stColumn"] { padding: 0 !important; }
div[class*="st-key-evrows"] [data-testid="stColumn"]:first-child {
  display: flex; justify-content: flex-end; align-items: center;
}
</style>
"""


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
def render_evidence_table(cluster: str) -> None:
    """Render the marker-evidence table for ``cluster`` into the current column.

    Reads the cached verdict (``ui.data_access.verdict_for`` — computed once) and
    the multi-select set (``ui.state.get_selected_markers``). Draws a header, one
    row per shown marker with a select-dot that only toggles membership in the
    multi-select set. Recomputes nothing; selecting drives the spatial
    small-multiples without touching a value.
    """
    import streamlit as st

    verdict: ClusterVerdict = data_access.verdict_for(cluster)

    st.markdown(_EVIDENCE_CSS, unsafe_allow_html=True)
    st.markdown(
        '<p class="pano-eyebrow">Marker evidence · '
        f"{html.escape(verdict.cell_type.replace('_', ' '))} · "
        f"{len(verdict.evidence)} assigned markers</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="pano-ev-head">'
        '<span>Gene</span><span class="r">glm_coef / pearson</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    rows = _rows_to_show(verdict)
    if not rows:
        st.markdown(
            '<div class="pano-ev-empty">No assigned markers for this cluster.</div>',
            unsafe_allow_html=True,
        )

    _render_marker_rows(st, rows, key="evrows_shown")

    # The long tail (non-driver markers beyond the default view) is tucked behind
    # an expander so the drivers and the tissue stay above the fold. Shown and
    # hidden rows are disjoint, so per-gene select-dot keys stay unique.
    shown_ids = {id(e) for e in rows}
    hidden = [e for e in verdict.evidence if id(e) not in shown_ids]
    if hidden:
        with st.expander(
            f"Show all {len(verdict.evidence)} assigned markers", expanded=False
        ):
            _render_marker_rows(st, hidden, key="evrows_hidden")


def _render_marker_rows(st, rows, *, key: str) -> None:
    """Render each marker: a narrow select-dot column + a wide HTML content column,
    inside a keyed container so the theme can tighten the dot↔gene gap and keep the
    dot aligned next to the gene name.

    The select-dot toggles multi-select membership and returns (no recompute).
    """
    with st.container(key=key):
        for ev in rows:
            is_selected = state.is_marker_selected(ev.gene)
            dot_col, body_col = st.columns([0.05, 0.95], vertical_alignment="center")
            with dot_col:
                _select_dot(st, ev.gene, is_selected)
            with body_col:
                st.markdown(
                    _marker_row_html(ev, is_selected=is_selected),
                    unsafe_allow_html=True,
                )


def _select_dot(st, gene: str, is_selected: bool) -> None:
    """Draw the borderless teal select-dot for one marker (● selected / ○ add).

    The dot is a chromeless ``st.button`` whose label is the glyph; the theme's
    ``.pano-select-dot`` wrapper strips all button chrome and colours the glyph.
    On click it calls ``state.toggle_marker`` (a single session-state write) and
    lets Streamlit rerun to re-read the already-cached verdict — the value is
    never recomputed by a selection.
    """
    glyph = "●" if is_selected else "○"
    help_txt = (
        "Deselect — remove from the spatial comparison"
        if is_selected
        else "Select to add to the spatial comparison"
    )
    # A real keyed container genuinely wraps the button (a bare markdown <span>
    # auto-closes and never contains it), so the theme's `st-key-seldot_*` rules
    # strip all chrome; the on/off suffix drives the ● teal / ○ muted colour.
    with st.container(key=f"seldot_{gene}"):
        # STABLE container key (does NOT encode selection); selection state is
        # carried by type= (primary=teal ● / secondary=muted ○), like the rail.
        # Plain in-line toggle: the evidence table renders BEFORE the spatial stage
        # in the same run, so toggling here is visible to the grid immediately.
        if st.button(
            glyph,
            key=f"select_{gene}",
            help=help_txt,
            use_container_width=True,
            type="primary" if is_selected else "secondary",
        ):
            state.toggle_marker(gene)
            st.rerun()


__all__ = ["render_evidence_table"]
