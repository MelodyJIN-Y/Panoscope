"""The tab strip, derived from the workflow registry.

One tab per workflow in :data:`panoscope.workflows.WORKFLOWS`, titled by its
``UISpec.title`` and routed on its ``UISpec.workflow_id`` (the ``?page=`` value).
Tabs sort by ``TaskKind``, ties by surface id.

A registered workflow with no data for this study still gets a tab, opening on a
page that states the absence: "no result here" must stay distinguishable from
"cannot interpret this". That is not ``uispec.degraded_spec``, which is for an
undeclared analysis type.

No module-level Streamlit import; ``app.py`` owns the wiring.
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

#: ``{surface_id: (render, has_data)}``, supplied by ``app.py``. No entry reads as no data.
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
        """Does this study have something to draw here? A call, not a field, so
        building the strip reads no data; an unreadable result is *no data*."""
        if self.has_data is None:
            return False
        try:
            return bool(self.has_data())
        except Exception:  # noqa: BLE001 - an unreadable result is an unavailable tab
            return False


def tabs(wired: Wiring, summary: Callable[[], None]) -> tuple[Tab, ...]:
    """The strip: Summary, then one tab per registered workflow in task-kind order."""
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
    """The page for a registered workflow with no result in this study, naming the
    question it would have answered."""
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
