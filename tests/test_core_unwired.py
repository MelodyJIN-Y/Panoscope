"""The two ``core/`` modules that stay unwired on purpose -- and the facts that say so.

An unwired module is indistinguishable from an absent one, so each of these carries
its reason in its own docstring. A reason in prose outlives the fact it rests on;
these tests are the expiry date. Each asserts the *condition* that makes leaving the
module unwired the right answer, so the day the condition stops holding the
justification fails here and names the wiring that is now owed.

Neither of these is "this module is unused and that is fine". Both say: its input has
no producer yet, and wiring it today would put a check on the review surface that the
biologist would see run and find nothing. That is the failure the four-state lattice
exists to prevent, arriving one layer up -- a silence read as a negative result.

``core/concordance.py`` and ``core/failure_modes.py`` were deliberately NOT in here,
because their inputs exist today (``PlatformAdapter.sample_inputs`` resolves two
per-sample tables on the multi-sample fixture; ``detect`` raises a named mode on
eight of the demo's nine units) -- so for those two "unwired" was a gap, not a
justification.

``core/concordance.py`` is now wired: ``pipeline/records._concordance`` runs it from
the composition root, and ``tests/test_concordance_wired.py`` holds that gap closed
from the record end. ``core/failure_modes.py`` is still open, and it is a gap rather
than a justification -- ``InterpretationRecord`` has no field a named mode could be
written into, and ``types/record.py`` is where that would have to be added.
"""

from __future__ import annotations

import pytest

from panoscope import methods
from panoscope.data import layout
from panoscope.data.study import Study
from panoscope.platforms.xenium import XeniumAdapter
from panoscope.workflows import de as de_workflow


def _studies() -> list[str]:
    return sorted(layout.available_studies())


@pytest.mark.parametrize("study_id", _studies())
def test_no_registered_method_reports_a_shared_localisation(study_id: str) -> None:
    """``core/adjacency.py`` reads ``EvidenceItem.shared_with``. Nothing fills it.

    ``adjacency_check`` returns an empty ``Adjacency`` for an item with no shared
    units, so wiring it into the composition root today would add a per-item check
    that is structurally incapable of firing on any unit of any study on disk. A
    check that is seen to run and find nothing is read as a negative result; §9.9's
    whole point is that a shared localisation is a flag whose firing rate has not
    been measured, and a firing rate of zero-by-construction is not a measurement
    of it.

    Every study and every unit, because a one-unit check has passed in this repo
    while most of the demo was wrong.
    """
    study = Study.open(study_id)
    path = study.inputs / "markers_top.csv"
    if not path.is_file():
        pytest.skip(f"{study_id} has no marker table")
    method = methods.for_workflow("marker-interpretation")[0]
    loaded = method.load(str(path), XeniumAdapter(), study)

    populated = {
        unit: [i.subject.display() for i in ev.items if i.shared_with]
        for unit, ev in loaded.value.items()
    }
    assert not any(populated.values()), (
        f"{study_id}: a method now reports shared_with ({populated}). "
        "core/adjacency.py has real input and must be wired -- its reading is no "
        "longer empty by construction, and the justification in its docstring has "
        "expired"
    )


def test_nothing_emits_the_check_result_core_propagate_attaches_to() -> None:
    """``core/propagate.py`` returns a ``CheckResult``; only ``workflows/de.py``
    builds one, and ``de-interpretation`` has no registered method.

    So there is no production path a condition-unknown could propagate ALONG. The
    marker workflow is not a substitute: its four checks read the measurement, the
    panel frame and a curated marker list, and declaring a ``Condition`` dependency
    on one of them so this module would have a caller would invent the dependency --
    and on the demo, where ``segmentation`` and ``fixation`` are declared unknown,
    it would turn a settled check into CANNOT BE SETTLED on a fabricated ground.
    """
    registered = methods.for_workflow(de_workflow.WORKFLOW_ID)
    assert registered == (), (
        f"{[m.name for m in registered]} now answers "
        f"{de_workflow.WORKFLOW_ID!r}; its four checks run in production, so "
        "core/propagate.py has the caller it was waiting for and the four "
        "hand-rolled 'notice an input is absent, refuse, name what would settle it' "
        "blocks in workflows/de.py should go through it"
    )
