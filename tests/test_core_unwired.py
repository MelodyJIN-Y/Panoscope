"""The ``core/`` module that stays unwired on purpose -- and the fact that says so.

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
replaced it. ``core/failure_modes.py`` is wired too: ``types/record.py`` gained the field a
named mode goes in, and ``core/record.build`` indexes the modes off the same traces
the decisions come from. So what is left in this file is ADJACENCY alone.
"""

from __future__ import annotations

from pathlib import Path

import re

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


def test_the_only_condition_dependency_is_one_a_check_really_has() -> None:
    """``core/propagate.py`` is WIRED now, and this guards it from spreading.

    The rule CLAUDE.md states twice -- a check reading an unknown field returns
    *cannot be settled* and names what is missing -- had no implementation for
    checks. ``workflows/de.composition_vs_expression`` now declares its dependency
    on ``segmentation`` through ``requires``, which is the call ``propagate``'s own
    module docstring gives as the example. It is a real dependency: abundance is a
    count of CELLS per arm, and cells are what segmentation produces.

    What this pins is that it stays the only one. The danger with a mechanism like
    this is not that nobody calls it, it is that it becomes decoration -- a check
    declaring a condition it does not read turns a settled answer into CANNOT BE
    SETTLED on a fabricated ground, and on the demo, where segmentation and
    fixation are both unknown, that silences real results.

    So: adding a caller is allowed, and this failing is the prompt to say which
    check reads which field and why -- not to widen the list.
    """
    root = Path(__file__).resolve().parents[1] / "panoscope"
    callers: list[str] = []
    for src in sorted(root.rglob("*.py")):
        if src.name == "propagate.py":
            continue
        for line in src.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("#") or "import" in line:
                continue
            # both spellings: the dotted call and the imported symbol
            if re.search(r"(?<![\w.])(propagate\.)?requires\(", line):
                callers.append(f"{src.relative_to(root.parent)} {line.strip()}")

    # The file and the FIELD, never the line number: an edit above the call is not
    # a change to which condition the check depends on.
    assert len(callers) == 1, f"expected exactly one condition dependency: {callers}"
    where, call = callers[0].split(" ", 1)
    assert where == "panoscope/workflows/de.py", where
    assert '"segmentation"' in call, call
    assert "COMPOSITION" in call, call
