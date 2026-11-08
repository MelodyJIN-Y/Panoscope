"""One cluster, one call, on every surface that names it.

The biologist's override is applied at DISPLAY, over a computed verdict that is
never mutated. That composition lived in ``ui/data_access`` while half the prose
was generated underneath it in ``agent/``, so for a long time the board, the
marker header and the exported overview said CAF while the chat pane, the
cross-cluster check, the spatial hover, the rival chip and the model's own
system prompt went on saying Stromal.

These tests are deliberately written as ONE sweep rather than one test per
surface. The defect was never in a single reader; it was that nobody asked all
of them the same question at once.

Two properties, and the second is the one that keeps biting:

* every surface adopts the call, and
* it adopts it even when the surface was READ FIRST -- a per-study cache made
  ``loop._global_interpretation_lines`` unreachable by any override written
  after the session's first prompt, which is every override.
"""

from __future__ import annotations

import pytest

from panoscope.agent import config as cfg
from panoscope.agent import fallback as F
from panoscope.agent import loop, memory
from panoscope.ui import cluster_rail  # noqa: F401  (import parity with the app)
from panoscope.ui import data_access as da
from panoscope.ui import enrichment_conversation as ec
from panoscope.ui import evidence_table as et
from panoscope.ui import report as R
from panoscope.ui import spatial_stage as ss


@pytest.fixture(autouse=True)
def _proposal_revealed(monkeypatch) -> None:
    """These pin composition; the reveal gate is pinned in tests/test_review_surface.py."""
    from panoscope.ui import review

    monkeypatch.setattr(review, "revealed", lambda _c: True)


def _override(cluster: str, call: str, **kw) -> None:
    memory.create_note(
        claim=f"{cluster} is {call} here", scope="cluster", basis="own_validation",
        cluster=cluster, subject_cell_type=call, note_type="celltype_override",
        dataset=cfg.DATASET_ID, literature_search=lambda _q: (), **kw,
    )


def _global_line(cluster: str) -> str:
    return next(l.strip() for l in loop.build_system_prompt(cluster).splitlines()
                if l.strip().startswith(f"{cluster}:"))


def test_every_surface_follows_an_override_written_after_it_was_read():
    """Caches warmed FIRST, deliberately.

    Reading each surface before the note exists is what a real session does: the
    biologist opens the cluster, looks at it, and only then overrides the call.
    A reader cached per study answers the first question forever, so this order
    is the one that catches it and the reverse order is the one that passes
    whatever the code does.
    """
    probes = {
        "display_cell_type": lambda: da.display_cell_type("c2"),
        "display_verdict_for": lambda: da.display_verdict_for("c2").cell_type,
        "composed_verdicts": lambda: {v.cluster: v.cell_type
                                      for v in da.composed_verdicts()}["c2"],
        "marker header": lambda: et._header_values("c2")["label"],
        "spatial hover": lambda: ss._cluster_celltype_map()["c2"],
        "chat opening": lambda: F.fallback_opening("c2").text,
        "pathways opening": lambda: ec._opening_text("c2")[0],
        "model prompt": lambda: _global_line("c2"),
        "cross-cluster check": lambda: R.global_check_text(da.holistic(), None),
    }
    before = {k: f() for k, f in probes.items()}
    assert all("Stromal" in str(v) for v in before.values()), (
        "c2 must start out called Stromal on every surface, or this proves nothing"
    )

    _override("c2", "CAF", subject_lineage="Stromal", subject_category="Mesenchymal")

    stale = []
    for k, f in probes.items():
        after = str(f())
        if "CAF" not in after:
            stale.append(f"{k}: {after[:120]}")
    assert not stale, "surfaces a runtime override cannot reach:\n  " + "\n  ".join(stale)


def test_the_computed_call_stays_visible_where_the_surface_has_room_for_it():
    """Adopting the call is half the rule. CLAUDE.md: never a silent overrule.

    A surface that simply swapped the label would hide the disagreement the
    memory layer exists to keep. The prose surfaces say both; the chips and
    hovers, which have room for one word, say the biologist's.
    """
    _override("c2", "CAF", subject_lineage="Stromal", subject_category="Mesenchymal")

    assert "replacing the computed Stromal" in F.fallback_opening("c2").text
    assert "(the biologist's call; computed Stromal)" in _global_line("c2")
    # The one-word surfaces carry the call alone, and that is deliberate.
    assert da.display_cell_type("c2") == "CAF"
    assert ss._cluster_celltype_map()["c2"] == "CAF"


def test_a_rival_computed_for_the_old_call_is_withheld_not_relabelled():
    """c3 has a genuine rival, so it is the cluster that exercises this.

    ``discriminate`` reads the deterministic table and cannot answer about a call
    the annotation does not define. Relabelling its output would attribute a
    Macrophages marker set to Dendritic; printing it unchanged argued a call the
    biologist rejected. It is withheld, which ``ui/report._settle_for`` already
    did in the document.
    """
    assert "Macrophages" in (et._header_values("c3").get("runner_up") or "")

    _override("c3", "Dendritic")

    assert et._header_values("c3")["label"] == "Dendritic"
    assert et._header_values("c3").get("runner_up") == ""


def test_nothing_moves_for_a_cluster_nobody_overrode():
    """The composition costs an unedited cluster nothing."""
    _override("c2", "CAF")

    assert da.display_cell_type("c1") == "Tumor"
    assert ss._cluster_celltype_map()["c1"] == "Tumor"
    assert F.fallback_opening("c1").text.startswith("c1 reads as Tumor,")
    assert "your call" not in F.fallback_opening("c1").text
    assert _global_line("c1").startswith("c1: Tumor — ")
