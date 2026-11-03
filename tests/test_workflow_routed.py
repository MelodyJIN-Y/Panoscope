"""The ASSIGNMENT root answers its question through the workflow that owns it.

``pipeline/records.build_all`` used to consult the registry for the method and the
task kind and then call ``core.verdict.assess`` itself. The band on screen was
therefore computed by the composition root, not by ``workflows/marker.py``, and the
consequences were not cosmetic:

* ``MarkerInterpretation.interpret`` had no production caller at all, so the
  ``Assignment`` -- the v2 output contract, the shape every ASSIGNMENT surface is
  built to render -- was exercised only by tests;
* ``core/interpret.py``'s ``ShapeContradictsTaskKind`` guard compares a RETURNED
  object against the declared task kind, and with nothing returned it never ran on
  the one path that writes a band into ``interp/verdicts.csv``;
* registering a second ASSIGNMENT method meant editing the root, which is the
  opposite of what ARCHITECTURE §4 claims a registry buys.

These tests hold that wiring closed from the production end. They are deliberately
NOT a second set of band expectations: ``tests/test_core_verdict.py`` drives the
ladder and ``tests/test_concordance_wired.py`` drives §9.10, so a third copy here
would be the duplicate that drifts. What is asserted is only what the routing adds
-- that the workflow ran, that what it returned agrees with the ``Verdict`` the v1
projection still reads, and that the §9.10 coverage survived the trip.

**The agreement test is the load-bearing one and it is not a tautology.** The root
calls ``assess`` for the ``Verdict`` and the workflow calls it again for the
``Assignment``. Same pure function, so they agree only while they are handed the
same arguments -- and the third argument, the replicate coverage, is exactly the one
the workflow did not used to receive. Before ``Measured.coverage`` carried it, these
two calls would have disagreed on every thin multi-sample unit, silently, in the
direction that reports a demoted call at full confidence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.pipeline import records
from panoscope.types.outputs import Assignment
from panoscope.workflows import marker as wf_marker
from tests.conftest import DEMO_ID, MINI_ID
from tests.test_concordance_wired import _one_unit_built_two_ways


def _studies() -> list[str]:
    return sorted(layout.available_studies())


@pytest.mark.parametrize("study_id", _studies())
def test_every_unit_carries_the_workflows_own_assignment(study_id: str) -> None:
    """The workflow ran for every unit of every study, and returned the right shape.

    ``isinstance`` rather than a truthiness check: ``core.interpret.interpret``
    raises on a shape that contradicts the declaration, so a ``None`` here would
    mean the root skipped the call rather than that the guard let something through.
    """
    built = records.build_all(Study.open(study_id))
    assert built, f"{study_id} produced no units"
    for unit, b in built.items():
        assert isinstance(b.assignment, Assignment), (
            f"{study_id}:{unit} has no Assignment; the root did not route through "
            "the workflow"
        )
        assert b.assignment.unit == unit


@pytest.mark.parametrize("study_id", _studies())
def test_the_workflow_and_the_v1_verdict_cannot_disagree(study_id: str) -> None:
    """One band, two readers. The whole point of routing rather than duplicating.

    ``Assignment`` is what a v2 surface renders and ``Verdict`` is what the v1
    projection and ``cli explain`` branch on. They come from two separate ``assess``
    calls, and nothing but identical arguments keeps them equal -- which is why this
    is pinned rather than assumed.
    """
    for unit, b in records.build_all(Study.open(study_id)).items():
        assert b.assignment.band == b.verdict.band, (
            f"{study_id}:{unit} -- workflow says {b.assignment.band!r}, the v1 "
            f"projection says {b.verdict.band!r}"
        )
        assert b.assignment.verify is b.verdict.verify, (
            f"{study_id}:{unit} -- verify disagrees between the two readers"
        )


def test_the_replicate_coverage_reaches_the_workflow_and_moves_its_band(
        tmp_path: Path) -> None:
    """§9.10 survives the trip through the workflow, in both directions.

    This is the test that would have caught the naive wiring. ``MarkerReference``
    carries no coverage and ``Workflow.interpret`` takes only ``(ev, reference)``,
    so a root that simply called the workflow would have handed it two arguments
    where the band needs three -- and the thin build below would have come back at
    ``Very High`` with ``verify`` False, which is the exact defect
    ``tests/test_concordance_wired.py`` was written to close, reopened one layer up.

    The pair is the assertion. The rich build must still commit at full band, or a
    rule that demoted everything would pass the thin half and be worse than nothing.
    """
    rich = records.build_all(
        _one_unit_built_two_ways(tmp_path, "routed_rich", dropout=False))["c1"]
    thin = records.build_all(
        _one_unit_built_two_ways(tmp_path, "routed_thin", dropout=True))["c1"]

    # The band first, because this is the assertion with the consequence. With the
    # coverage withheld the thin build comes back Very High / verify False from the
    # workflow while the v1 verdict says High / verify True -- the surface showing
    # full confidence on a call the deterministic rubric demoted AND flagged.
    assert (thin.assignment.band, thin.assignment.verify) == ("High", True), (
        f"thin build: workflow says {thin.assignment.band!r}/"
        f"verify={thin.assignment.verify}, v1 says {thin.verdict.band!r}/"
        f"verify={thin.verdict.verify}; the §9.10 coverage did not reach the band"
    )
    # The rich half, or a rule that demoted everything would pass the thin case.
    assert (rich.assignment.band, rich.assignment.verify) == ("Very High", False)

    # And the coverage itself, read back through the workflow's own accessor rather
    # than through the Verdict, so this still fails if it never left the root.
    assert wf_marker._replicate_coverage(rich.evidence) is not None
    assert wf_marker._replicate_coverage(rich.evidence).k == 9
    assert wf_marker._replicate_coverage(thin.evidence).k == 3


@pytest.mark.parametrize("study_id", [DEMO_ID, MINI_ID])
def test_a_single_sample_study_tells_the_workflow_the_question_was_never_asked(
        study_id: str) -> None:
    """``None``, not ``Coverage(k=0)``. The distinction the lattice is built on.

    A single-sample study has no second reading for a profile to be thin against,
    so §9.10 was never askable and the band must be untouched. Reporting it as a
    coverage of zero would demote the demo on every unit for the absence of a
    question -- a silence read as a negative result, which is the failure the four
    states exist to prevent.
    """
    for unit, b in records.build_all(Study.open(study_id)).items():
        assert wf_marker._replicate_coverage(b.evidence) is None, (
            f"{study_id}:{unit} -- a single-sample study handed the workflow a "
            "replicate coverage; absence of a question is not a thin answer"
        )
        assert b.verdict.replicate_coverage is None


def test_the_shape_guard_now_runs_on_the_path_that_writes_the_band(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """``ShapeContradictsTaskKind`` protects ``verdicts.csv``, not just the tests.

    The declaration-side guard in ``build_all`` (``tests/test_pipeline_shape_guard``)
    fires before any IO and catches a workflow that *declares* the wrong kind. It
    cannot catch one that declares ASSIGNMENT and returns a ``Finding``, because
    that is a fact about the returned object. Routing through
    ``core.interpret.interpret`` is what closes that half, and this fails if a
    future refactor calls ``wf.interpret`` directly again.
    """
    from panoscope.core import interpret as core_interpret
    from panoscope.types.outputs import Finding, TaskKind
    from panoscope.types.trace import Traced

    def _wrong_shape(self, ev, reference):  # noqa: ANN001 - test double
        return Traced(Finding(unit=ev.unit, unit_kind=ev.unit_kind,
                              task_kind=TaskKind.CHARACTERISATION,
                              items=(), summary="not an assignment"))

    monkeypatch.setattr(type(wf_marker.WORKFLOW), "interpret", _wrong_shape)
    with pytest.raises(core_interpret.ShapeContradictsTaskKind):
        records.build_all(Study.open(MINI_ID))
