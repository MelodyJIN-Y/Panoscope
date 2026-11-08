"""The whole biologist path through the app on a folder with no labels: path box ->
checklist -> build -> evidence-first review -> accept, override (scope + basis), defer,
call the rest -> export, refused while any cluster is OPEN and naming them."""
from __future__ import annotations

import json
import shutil
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import panoscope
from panoscope.data.study import Study

APP = str(Path(panoscope.__file__).parent / "app.py")
HER2 = "HER2+ Breast Tumor Epithelial"
CAF = "Cancer-Associated Fibroblast"


def _proposal(prompt: str, cluster: str) -> SimpleNamespace:
    """The skill's proposal for every cluster but c3, which stays not yet called."""
    if cluster == "c3":
        raise RuntimeError("no proposal for c3")
    return SimpleNamespace(text=json.dumps({"cell_type": HER2, "cell_type_short": "Tum_HER2",
                                            "category": "Tumor", "lineage": "Epithelial"}))


def _captions(at) -> str:
    return "\n".join(c.value for c in at.caption)


def _ok(at) -> None:
    assert not at.exception, [str(e.value) for e in at.exception]


def _page(at, page: str) -> None:
    at.session_state["active_page"] = page
    at.run()
    _ok(at)


def _docx_text(data: bytes) -> str:
    import docx

    doc = docx.Document(BytesIO(data))
    return "\n".join([p.text for p in doc.paragraphs]
                     + [c.text for t in doc.tables for r in t.rows for c in r.cells])


def test_a_labelless_folder_is_built_reviewed_called_and_exported(tmp_path, monkeypatch) -> None:
    from streamlit.testing.v1 import AppTest

    from panoscope.agent import loop as agent_loop
    from panoscope.data import layout
    from panoscope.pipeline.stages import annotate
    from panoscope.reference import sources as ref_sources
    from panoscope.ui import report

    folder = tmp_path / "mystudy"
    folder.mkdir()
    for name in ("markers_top.csv", "panel.parquet"):  # no labels, no manifest
        shutil.copyfile(Study.open("xenium_hbreast_sample1").inputs / name, folder / name)
    (folder / "reference").mkdir()            # the lab's own reference list, offline
    shutil.copyfile(Study.open("demo_mini").inputs / "reference" / "lab_markers.csv",
                    folder / "reference" / "lab_markers.csv")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(layout.ENV_DATA_ROOT, raising=False)
    layout.clear_cache()

    def no_model(*_a, **_k):
        raise RuntimeError("no model key")

    monkeypatch.setattr(agent_loop, "chat", no_model)
    monkeypatch.setattr(agent_loop, "has_model_key", lambda: False)
    # Only the annotate stage has a "model": its proposal is the skill's, canned here.
    monkeypatch.setattr(annotate, "agent_loop",
                        SimpleNamespace(has_model_key=lambda: True, chat=_proposal))
    exported: dict = {}
    real_docx = report.working_docx

    def keep_docx(*a, **k):
        exported["docx"] = real_docx(*a, **k)   # the very bytes the download button serves
        return exported["docx"]

    monkeypatch.setattr(report, "working_docx", keep_docx)
    monkeypatch.setattr(ref_sources, "uberon_candidates",
                        lambda text: [("UBERON:0000310", "breast")])

    # Path box -> checklist -> declare platform and tissue -> build.
    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["active_page"] = "upload"
    at.run()
    at.text_input(key="ob_folder").input(str(folder)).run()
    at.selectbox(key="ob_platform").select("xenium").run()
    at.text_input(key="ob_f_tissue").input("human breast cancer")
    at.button(key="ob_apply").click().run()
    at.button(key="ob_uberon_find").click().run()
    at.selectbox(key="ob_uberon_pick").select(("UBERON:0000310", "breast")).run()
    at.button(key="ob_uberon_use").click().run()
    at.button(key="ob_build").click().run()
    _ok(at)
    at.button(key="ob_corpus_skip").click().run()
    _ok(at)

    # Export is refused before any call, naming every OPEN cluster.
    _page(at, "summary")
    assert not at.download_button
    assert "still OPEN: c1, c2, c3, c4, c5, c6, c7, c8, c9" in _captions(at)

    _page(at, "markers")
    # c1: accept the proposal, one tap away behind the evidence.
    at.button(key="rail_c1").click().run()
    at.toggle(key="rv_reveal_c1").set_value(True).run()
    at.button(key="rv_accept_c1").click().run()
    _ok(at)
    # c2: the biologist's own call; scope this cluster, basis a paper.
    at.button(key="rail_c2").click().run()
    at.toggle(key="rv_calling_c2").set_value(True).run()
    at.selectbox(key="rv_pick_c2").select(CAF).run()
    at.button_group(key="rv_basis_c2").set_value("a paper").run()
    at.button(key="rv_save_c2").click().run()
    _ok(at)
    assert sum('class="pano-rv-assess"' in m.value for m in at.markdown) == 2
    # c3: no proposal to accept; deferred.
    at.button(key="rail_c3").click().run()
    at.toggle(key="rv_reveal_c3").set_value(True).run()
    assert at.button(key="rv_accept_c3").disabled
    at.button(key="rv_defer_c3").click().run()
    # Partial annotation: the rest are still OPEN, so export still refuses and names them.
    _page(at, "summary")
    assert not at.download_button
    assert "still OPEN: c4, c5, c6, c7, c8, c9" in _captions(at)

    _page(at, "markers")
    for c in ("c4", "c5", "c6", "c7", "c8", "c9"):
        at.button(key=f"rail_{c}").click().run()
        at.toggle(key=f"rv_reveal_{c}").set_value(True).run()
        at.button(key=f"rv_accept_{c}").click().run()
        _ok(at)

    _page(at, "summary")
    assert len(at.download_button) == 2
    text = _docx_text(exported["docx"])
    lines = text.split("Annotation decisions", 1)[1].split("Cross-cluster", 1)[0]
    assert f"c1: accepted · proposal {HER2} (Very High) · your call {HER2} (Very High)" in lines
    # The same band function on the biologist's label, beside the proposal's, plus the claim.
    assert (f"c2: overridden · proposal {HER2} (Low, verify) · your call {CAF} (Very High) · "
            f'override claim note:') in lines
    assert f'"c2 is {CAF} (the biologist\'s call)" — cluster c2, basis: a paper' in lines
    assert "c3: deferred · proposal not yet called (no band)" in lines
    # Heading and overview row: the biologist's label with ITS band; accepted unchanged.
    import docx

    doc = docx.Document(BytesIO(exported["docx"]))
    heads = [p.text for p in doc.paragraphs]
    assert f"c2 · {CAF} (Very High)" in heads and f"c1 · {HER2} (Very High)" in heads
    rows = {r.cells[0].text: (r.cells[1].text, r.cells[3].text) for r in doc.tables[0].rows}
    assert rows["c2"] == (CAF, "Very High") and rows["c1"] == (HER2, "Very High")
    assert all(f"{c}: accepted · proposal {HER2}" in lines for c in ("c4", "c5", "c6", "c7", "c8", "c9"))
