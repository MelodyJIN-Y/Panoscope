"""The tab strip, derived from the workflow registry.

``app.py`` asserted three tabs. A registry the navigation does not read gives you a
declared surface nobody can reach, so the strip is built from
:data:`panoscope.workflows.WORKFLOWS` instead: one tab per registered question,
titled by that question's own ``UISpec.title`` and routed on its own
``UISpec.workflow_id`` -- which is already what ``?page=`` spells, so the existing
URL contract needs no alias table.

**Order is declared, not listed.** Tabs sort by ``TaskKind`` -- assignment, then
characterisation, then comparison: *what is this* before *what is running* before
*does this make sense*, which is the order the evidence is read in. ``TaskKind`` is
frozen at three (CLAUDE.md), so this is a total order no new workflow can need a
line added to. Ties inside a kind break on the surface id, because dict insertion
order is import order and import order is a fact about nothing.

**A registered workflow with no data for this study still gets a tab**, marked in
the strip, opening on a page that states the absence. Hiding it is the
navigation-level version of collapsing ``NEVER_MEASURED`` into "absent": the
biologist could not tell *this study has no enrichment result* from *Panoscope
cannot interpret enrichment*, and the second is a far bigger claim to make silently.
This is recurring, not a DE special case -- the kidney study arrives with a panel
that answers one of these questions well and another barely at all.

It is deliberately **not** ``uispec.degraded_spec``. That one is for an analysis
type *nobody declared*, and it degrades the surface down to provenance because the
declaration is missing. Here the declaration is present and correct; what is missing
is a result to draw on it. Degrading the surface would throw away a good declaration
because of an absent file, and would make the two cases indistinguishable again.

This module is pure -- no module-level Streamlit import -- so the strip can be built
and asserted with no server running. ``app.py`` owns the wiring (which renderer,
which availability check); this module owns the model.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping

from panoscope import workflows
from panoscope.types.outputs import TaskKind

#: The integrated review surface. Not a workflow -- it reads every workflow's
#: output and is where sign-off happens -- so it is named here rather than derived.
SUMMARY_PAGE = "summary"

#: The front door, before any study is open. Also not workflows.
ONBOARDING_PAGES: tuple[str, ...] = ("welcome", "upload")

#: Session key for the active page, mirrored to the ``?page=`` query param.
PAGE_KEY = "active_page"

#: Brand assets (the Panoscope logo mark + the with-text logo).
ASSETS = Path(__file__).resolve().parent.parent / "assets"


def set_page(page: str) -> None:
    """Make ``page`` active: session state, and the URL so a refresh keeps the tab."""
    import streamlit as st

    st.session_state[PAGE_KEY] = page
    st.query_params["page"] = page


@lru_cache(maxsize=2)
def logo_data_uri(name: str) -> str:
    """Base64 data URI for a brand PNG so it embeds inline in header HTML
    (Streamlit's strict CSP blocks external asset requests). '' if missing."""
    path = ASSETS / name
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

_TASK_ORDER: dict[TaskKind, int] = {
    TaskKind.ASSIGNMENT: 0,
    TaskKind.CHARACTERISATION: 1,
    TaskKind.COMPARISON: 2,
}

#: ``{surface_id: (render, has_data)}``. Supplied by ``app.py``; a surface id with
#: no entry has no page wired, which from the biologist's seat is the same fact as
#: no data -- there is nothing to draw -- and reads the same on screen.
Wiring = Mapping[str, tuple[Callable[[], None], Callable[[], bool]]]


@dataclass(frozen=True, slots=True)
class Tab:
    """One entry in the strip: where it routes, what it is called, what draws it."""

    page: str                               # the ?page= name == UISpec.workflow_id
    title: str                              # UISpec.title
    question: str                           # Workflow.question, for the tooltip
    render: Callable[[], None] | None = None
    has_data: Callable[[], bool] | None = None

    @property
    def key(self) -> str:
        """The widget key. Derived from the page so it cannot drift from the route."""
        return f"nav_{self.page}"

    def available(self) -> bool:
        """Does this study have something to draw here?

        Deliberately a call and not a field: building the strip must not read a
        parquet. The onboarding screens resolve the page before any study is open,
        and an eagerly-resolved field would load every workflow's result to decide
        which tab to underline. A missing or unreadable result is *no data*, never
        a traceback in the nav bar.
        """
        if self.has_data is None:
            return False
        try:
            return bool(self.has_data())
        except Exception:  # noqa: BLE001 - an unreadable result is an unavailable tab
            return False


def tabs(wired: Wiring, summary: Callable[[], None]) -> tuple[Tab, ...]:
    """The strip: Summary, then one tab per registered workflow in task-kind order.

    Summary leads because it is the surface the biologist lands on and signs off
    from; the per-workflow panes are the evidence it stands on.
    """
    ordered = sorted(
        workflows.WORKFLOWS.values(),
        key=lambda w: (_TASK_ORDER.get(w.task_kind, len(_TASK_ORDER)),
                       w.uispec().workflow_id),
    )
    strip = [Tab(page=SUMMARY_PAGE, title="Summary", question="",
                 render=summary, has_data=lambda: True)]
    for wf in ordered:
        spec = wf.uispec()
        entry = wired.get(spec.workflow_id)
        strip.append(Tab(
            page=spec.workflow_id,
            title=spec.title,
            question=wf.question,
            render=entry[0] if entry is not None else None,
            has_data=entry[1] if entry is not None else None,
        ))
    return tuple(strip)


def pages(strip: tuple[Tab, ...]) -> tuple[str, ...]:
    """Every ``?page=`` name this build answers to, onboarding included."""
    return ONBOARDING_PAGES + tuple(t.page for t in strip)


def find(strip: tuple[Tab, ...], page: str) -> Tab | None:
    return next((t for t in strip if t.page == page), None)


def render_unavailable(tab: Tab) -> None:
    """The page for a registered workflow with no result in this study.

    It names the question the workflow would have answered, because that is what is
    missing -- not the tab, and not the capability.
    """
    import html

    import streamlit as st

    question = f"<div>{html.escape(tab.question)}</div>" if tab.question else ""
    st.markdown(
        '<div style="font-family:var(--mono);font-size:12px;color:var(--faint);'
        'padding:16px 4px;margin-top:24px;line-height:1.7">'
        f"<div>No {html.escape(tab.title.lower())} result for this study.</div>"
        f"{question}"
        "<div>The question is registered; nothing in this study answers it yet. "
        "Add a result for it to build this slice.</div>"
        "</div>",
        unsafe_allow_html=True,
    )
