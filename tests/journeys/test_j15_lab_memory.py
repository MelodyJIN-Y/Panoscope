"""J15: lab memory across two studies -- the reconciliation layer the lab owns."""
from __future__ import annotations

import json

import pytest

from tests.conftest import DEMO_ID, MINI_ID
from tests.ui.screen import open_app, screen_text


def _save(study_id: str, **kw):
    from panoscope.agent import config as cfg
    from panoscope.agent import memory, tools

    with cfg.use_study(study_id):
        draft = memory.draft_note(basis="own_validation", dataset=study_id, **kw)
        return memory.save_draft(draft, attributed_to="pi@lab", base_dir=tools.memory_base_dir())


def _fires(study_id: str, cluster: str) -> set[str]:
    from panoscope.agent import config as cfg
    from panoscope.agent import memory, tools

    with cfg.use_study(study_id):
        return {n.id for n in memory.apply_notes(cluster, dataset=study_id,
                                                 base_dir=tools.memory_base_dir())}


@pytest.fixture
def two_studies(isolated):
    """Study A (the demo) where the biologist saved a cluster-scoped call and a
    lab-scoped convention; study B (demo_mini) opened afterwards."""
    cluster_note = _save(DEMO_ID, claim="c5 is CAF (our IHC)", scope="cluster", cluster="c5",
                         subject_cell_type="CAF")
    lab_note = _save(DEMO_ID, claim="ERBB2 marks HER2+ tumour epithelium in our hands",
                     scope="lab", note_type="marker_convention", subject_markers=["ERBB2"])
    return cluster_note, lab_note


@pytest.mark.req("ARCH-10.4-01", "ARCH-10.6-01", "UI-2.9-02", "CLAUDE-MEM-01")
def test_a_lab_note_travels_to_the_next_study_and_a_cluster_note_never_does(
        isolated, two_studies):
    """Given study A where the biologist saved 'c5 is CAF' for cluster c5 only and a lab
    convention 'ERBB2 marks HER2+ tumour epithelium',
    When they open study B,
    Then the cluster note fires nowhere in B (not even on B's own c-ids), the lab note
    fires on every B cluster and is cited with the study it was formed on and its
    literature state visible; a note with an unrecognised scope fires nowhere; and no
    note file was rewritten by any of it."""
    from panoscope.agent import memory, tools

    cluster_note, lab_note = two_studies
    folder = memory.corrections_dir(tools.memory_base_dir())
    bad = dict(json.loads((folder / f"{cluster_note.id}.json").read_text()),
               id="badscope00001", scope="cluster ")
    (folder / "badscope00001.json").write_text(json.dumps(bad))
    before = {p.name: p.read_bytes() for p in folder.glob("*.json")}

    assert cluster_note.id in _fires(DEMO_ID, "c5") and cluster_note.id not in _fires(DEMO_ID, "c4")
    for c in ("c1", "c2", "c3"):
        fired = _fires(MINI_ID, c)
        assert lab_note.id in fired and cluster_note.id not in fired
        assert "badscope00001" not in fired
    line = memory.render_citation(lab_note)
    assert f"lab formed_on:{DEMO_ID}" in line
    assert "no literature lookup completed" in line          # 'nobody looked', said so

    at = open_app("markers", cluster="c1", study=(str(isolated.root.parent), MINI_ID))
    assert "in our hands" in screen_text(at)                    # shown beside ERBB2 in B
    assert {p.name: p.read_bytes() for p in folder.glob("*.json")} == before


@pytest.mark.req("ARCH-10.5-02", gap=True)
def test_a_measurement_that_contradicts_a_lab_note_flags_it_and_never_rewrites_it(
        isolated):
    """Given a lab note 'ERBB2 is not expressed by tumour epithelium in our hands',
    When study B's c1 (tumour) measures ERBB2 as its top marker,
    Then the note is shown flagged as contradicted by this study's measurement -- never
    retracted, never edited."""
    note = _save(DEMO_ID, claim="ERBB2 is not expressed by tumour epithelium in our hands",
                 scope="lab", note_type="marker_convention", subject_markers=["ERBB2"])
    at = open_app("markers", cluster="c1", study=(str(isolated.root.parent), MINI_ID))
    text = screen_text(at)
    assert note.claim in text
    assert "contradicted" in text.lower(), text[:400]
