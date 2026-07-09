"""Verdict header — the call the biologist reads first, above the evidence.

``render_verdict(cluster)`` draws the top of the center pane exactly like the
wireframe's ``.center`` header block: an id line (``CLUSTER 3 · 4,812 cells``),
the cell-type call as an ``<h1>`` next to a confidence chip (plus a verify badge
when the call is flagged), a one-line grounded rationale, and the Confirm /
Override decision buttons.

Grounding stance for this pane:

* Every value shown is read, never computed here. The call, confidence band, and
  rationale come straight off the cached :class:`ClusterVerdict`
  (``ui.data_access.verdict_for``); the cell count comes off the cached cluster
  cell frame. This module runs no statistic and invents no number.
* The confidence chip / verify badge classes are the exact strings
  ``ui.format`` returns, so styling from ``ui.theme`` drops straight on.
* **Override** does not decide anything. It opens the capture-at-override flow in
  the conversation pane (``ui.state.open_capture``); the biologist makes the call
  and states scope + basis there. The tool informs and preserves — it never makes
  the call.
* **Confirm** records the biologist's acceptance as an attributed system message
  in the chat thread (never a silent "got it"); the verdict itself is untouched.

Streamlit is imported lazily inside :func:`render_verdict` so importing this
module never needs a running server.
"""

from __future__ import annotations

from typing import Optional

from agent.types import ClusterVerdict

from ui import format as fmt
from ui import state
from ui.data_access import cluster_cells_df, verdict_for

# --------------------------------------------------------------------------- #
# Small helpers (pure; no Streamlit)
# --------------------------------------------------------------------------- #
def _cell_count(cluster: str) -> Optional[int]:
    """Number of cells assigned to ``cluster`` (from the cached cell frame).

    Returns None if the cell frame is unavailable so the id line degrades to just
    the cluster label rather than crashing the header. This is a *read*, never a
    computed value — it is the row count of the precomputed cluster cell table.
    """
    try:
        df = cluster_cells_df(cluster)
    except Exception:
        return None
    if df is None:
        return None
    try:
        return int(len(df))
    except Exception:
        return None


def _idline(cluster: str) -> str:
    """The mono id line: ``CLUSTER 3 · 4,812 cells`` (count omitted if unknown)."""
    label = f"CLUSTER {fmt.short_cluster_id(cluster)}"
    n = _cell_count(cluster)
    if n is None:
        return label
    return f"{label} · {n:,} cells"


def _esc(text: str) -> str:
    """Minimal HTML escape for the grounded strings we drop into markup."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _header_html(cluster: str, verdict: ClusterVerdict) -> str:
    """Build the id line + verdict line + rationale block as one HTML string.

    Mirrors the wireframe ``.idline`` / ``.verdict`` / ``.rat`` structure and uses
    the ``.pano-*`` classes from ``ui.theme``. Every value is read off the verdict
    (call, confidence, verify, notes); nothing is computed here.
    """
    cf_class, cf_text = fmt.confidence_chip(verdict.confidence)
    verify_html = ""
    if verdict.verify:
        verify_html = f'<span class="pano-verify">{_esc(fmt.verify_badge(True))}</span>'

    # Terse rationale: name the drivers only. The full numbers live in the
    # evidence table (right below) and the conversation's opening interpretation,
    # so restating glm_coef/pearson here is redundant clutter.
    drivers = ", ".join(verdict.key_markers) if verdict.key_markers else verdict.cell_type
    rationale = f"Driven by {drivers}."
    if verdict.verify:
        rationale += " Flagged to re-check."

    return (
        f'<div class="pano-idline">{_esc(_idline(cluster))}</div>'
        f'<div class="pano-verdict">'
        f"<h1>{_esc(verdict.cell_type)}</h1>"
        f'<span class="cf {cf_class}">{_esc(cf_text)}</span>'
        f"{verify_html}"
        f"</div>"
        f'<div class="pano-rat">{_esc(rationale)}</div>'
    )


# --------------------------------------------------------------------------- #
# Actions (wire buttons to shared UI state; no verdict recompute)
# --------------------------------------------------------------------------- #
def _on_override() -> None:
    """Open the capture-at-override flow in the conversation pane.

    Opens the note-capture panel and defaults its scope to ``cluster`` (the tap
    where the biologist's knowledge diverges from the default). The biologist
    still makes the call and states basis/status there — this only routes them.
    """
    state.set_scope(state.DEFAULT_SCOPE)
    state.open_capture()


def _on_confirm(verdict: ClusterVerdict) -> None:
    """Record the biologist's acceptance of the call as an attributed message.

    Appends a short system message to the chat thread so the acceptance is
    visible and dated in-thread — never a silent overrule, never a bare "got it".
    The verdict object is not mutated.
    """
    state.close_capture()
    state.append_message(
        {
            "role": "sys",
            "text": (
                f"Confirmed: {verdict.cell_type} "
                f"({verdict.confidence} confidence) for "
                f"cluster {fmt.short_cluster_id(verdict.cluster)}, "
                f"kept as called."
            ),
            "kind": "confirm",
            "cluster": verdict.cluster,
        }
    )


# --------------------------------------------------------------------------- #
# Public render
# --------------------------------------------------------------------------- #
def render_verdict(cluster: str) -> None:
    """Render the verdict header for ``cluster`` at the top of the center pane.

    Reads the cached ``ClusterVerdict`` (``verdict_for``) — the call, confidence
    band, verify flag, and grounded one-line rationale — and the cached cluster
    cell count, then lays out:

    * a mono id line with the cluster and its cell count,
    * the cell-type call beside a confidence chip (and a verify badge if flagged),
    * the grounded rationale that cites the driving markers' numbers,
    * Confirm / Override buttons.

    **Override** opens the capture-at-override flow in the conversation
    (``ui.state.open_capture``); **Confirm** posts an attributed acceptance to the
    chat thread. Neither recomputes a verdict — the verdict is read once, cached.
    """
    import streamlit as st

    verdict = verdict_for(cluster)

    st.markdown(_header_html(cluster, verdict), unsafe_allow_html=True)

    confirm_col, override_col, _spacer = st.columns([1, 1, 3])
    with confirm_col:
        st.button(
            "Confirm annotation",
            key=f"verdict_confirm_{cluster}",
            type="primary",
            use_container_width=True,
            on_click=_on_confirm,
            args=(verdict,),
        )
    with override_col:
        st.button(
            "Override…",
            key=f"verdict_override_{cluster}",
            use_container_width=True,
            on_click=_on_override,
        )


__all__ = ["render_verdict"]
