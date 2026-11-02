"""The composition root refuses a workflow whose declared task kind has no band.

``pipeline/records.build_all`` is ASSIGNMENT-shaped by construction: it calls
``core.verdict.assess`` unconditionally and projects a band into
``interp/clusters/<unit>.json`` and ``interp/verdicts.csv``. Nothing downstream
would catch a mismatch -- ``core.record.build`` takes ``task_kind`` as a parameter
and ``core.compat.to_v1_assignment`` is handed a ``Verdict`` and never asks what
question it answers -- so registering a CHARACTERISATION workflow under
``WORKFLOW_ID`` would write a unit-level band for a question with no label space.

``core/interpret.py`` guards the same rule by comparing a RETURNED object against
the declaration. There is no returned object at this root (the shape is the
function's own code), so the comparison here is against the declaration alone.

The stand-in is ``dataclasses.replace`` on the registered workflow: every other
attribute stays real and exactly one thing -- the declaration -- moves. A
hand-rolled fake could pass this test while diverging from the protocol.
"""

from __future__ import annotations

import dataclasses

import pytest

from panoscope import workflows
from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.pipeline import records
from panoscope.types.outputs import TaskKind

NON_ASSIGNMENT = [k for k in TaskKind if k is not TaskKind.ASSIGNMENT]


@pytest.fixture
def study() -> Study:
    sid = next(iter(layout.available_studies()), None)
    if sid is None:
        pytest.skip("no study on disk")
    return Study.open(sid)


@pytest.mark.parametrize("kind", NON_ASSIGNMENT, ids=lambda k: k.value)
def test_a_bandless_task_kind_is_refused_at_the_root(
    study: Study, kind: TaskKind, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = workflows.get(records.WORKFLOW_ID)
    monkeypatch.setitem(workflows.WORKFLOWS, records.WORKFLOW_ID,
                        dataclasses.replace(real, task_kind=kind))

    with pytest.raises(SystemExit) as excinfo:
        records.build_all(study)

    msg = str(excinfo.value)
    assert kind.value in msg, msg
    assert "band" in msg, msg


def test_the_refusal_lands_before_the_root_reads_the_method_output(
    study: Study, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Loud *and* early: nothing is read before the declaration is checked.

    A guard placed after ``read_csv``/``method.load`` still refuses, but it refuses
    having already paid for -- and in a future root, possibly written -- work for a
    question it was about to reject. The cheapest failure is the one taken on the
    declaration alone.
    """
    import pandas as pd

    real = workflows.get(records.WORKFLOW_ID)
    monkeypatch.setitem(workflows.WORKFLOWS, records.WORKFLOW_ID,
                        dataclasses.replace(real,
                                            task_kind=TaskKind.CHARACTERISATION))

    def _no(*a: object, **k: object) -> None:
        raise AssertionError("the root read the method output before refusing")

    monkeypatch.setattr(pd, "read_csv", _no)

    with pytest.raises(SystemExit):
        records.build_all(study)
