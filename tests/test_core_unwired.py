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
from the record end. The DE workflow is now wired too -- ``pipeline/stages/de.py``
runs its four checks against a declared-synthetic artifact -- which retired the
proxy the propagate test used to assert; see that test for the condition that
replaced it. ``core/failure_modes.py`` is still open, and it is a gap rather
than a justification -- ``InterpretationRecord`` has no field a named mode could be
written into, and ``types/record.py`` is where that would have to be added.
"""

from __future__ import annotations

from pathlib import Path

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


def test_no_check_declares_a_dependency_on_a_declared_condition() -> None:
    """``core/propagate.py`` stays unwired -- and the REASON changed, so the test did.

    It used to assert that ``de-interpretation`` had no registered method, reasoning
    that only ``workflows/de.py`` builds a ``CheckResult`` and nothing ran it. The DE
    workflow and two methods are registered now and the four checks run in
    production, so that proxy has expired. The underlying condition has not.

    ``requires`` answers one question: which declared ``Conditions`` fields does this
    check read, and which are UNKNOWN. A caller is honest only where the check
    genuinely depends on such a field. None of the four does, and it is worth being
    specific rather than hand-waving:

    * ``panel_scoping`` reads the gene frame and a curated list. ``Conditions.frames``
      is explicitly NOT a ``Condition`` and is not in ``CONDITION_FIELDS``; ``tissue``
      is a blocking field, which ``requires`` can only ever return OK for.
    * ``replicate_confound`` reads a sample count, and ``types/conditions.py`` says in
      terms that ``Conditions`` carries none -- "inventing a sample count on an object
      that has no other use for one".
    * ``composition_vs_expression`` is the plausible one, and it does not hold.
      ARCHITECTURE §9.7 calls it "answerable from cluster abundances", and the only
      two places in this repository pairing composition with ``segmentation`` are
      ``propagate``'s own docstring example and ``propagate``'s own test -- which is
      circular, not evidence. ``methods/seurat_de`` does name segmentation, as an
      assumption about the per-cell MEAN; a different claim, already declared where
      it belongs.
    * ``cross_method_coherence`` takes the other method's ``Side`` and delegates to
      ``core/reconcile``; it never reads ``Conditions.method``.

    So the guard is now the thing that is actually true: ``requires`` has no
    production caller. Wiring one means a check that honestly reads a declared
    condition, and this failing is the prompt to say WHICH and WHY -- not to delete
    the assertion.

    This is NOT "the rule is unimplemented". ``claims/transfer.py`` emits a
    CANNOT_BE_SETTLED delta per unknown condition field and runs in every build;
    what has no caller is the *check-side* half.
    """
    root = Path(__file__).resolve().parents[1] / "panoscope"
    callers = [
        f"{p.relative_to(root.parent)}:{n}"
        for p in sorted(root.rglob("*.py"))
        if p.name != "propagate.py"
        for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
        if "propagate.requires(" in line and not line.lstrip().startswith("#")
    ]
    assert not callers, (
        f"core/propagate.requires now has production caller(s): {callers}. That is "
        "allowed -- but only for a check that genuinely reads a declared Conditions "
        "field. Name the check and the field here and say why the dependency is "
        "real, rather than removing this assertion."
    )
