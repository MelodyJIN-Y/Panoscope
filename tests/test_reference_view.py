"""The per-study reference view: the canonical-list source. Every reference file is
synthetic and written to ``tmp_path``; nothing touches the network."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from panoscope.agent import annotation
from panoscope.data.study import Study
from panoscope.reference import view
from panoscope.reference.sources import PanglaoDB

KIDNEY = "UBERON:0002113"
OBO = """format-version: 1.2

[Term]
id: CL:0000653
name: podocyte
synonym: "glomerular visceral epithelial cell" EXACT []
synonym: "shared name" EXACT []

[Term]
id: CL:0000002
name: other cell
synonym: "shared name" EXACT []
"""


def _study(root: Path, sid: str, tissue: str, lab: str = "", **manifest) -> Study:
    d = root / "studies" / sid
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"dataset_id": sid, "tissue": tissue,
                                                 **manifest}))
    if lab:
        (d / view.LAB_FILE).parent.mkdir(parents=True)
        (d / view.LAB_FILE).write_text(lab)
    return Study(id=sid, data_root=root)


def _kidney_cache(ref: Path) -> None:
    """Synthetic PanglaoDB (one kidney row, one breast row) and a synthetic CL."""
    p = ref / "panglaodb" / PanglaoDB.release / PanglaoDB.filename
    p.parent.mkdir(parents=True)
    rows = ["species\tofficial gene symbol\tcell type\torgan\tcanonical marker",
            "Hs\tNPHS2\tpodocyte\tKidney\t1",
            "Hs\tUMOD\tother cell\tKidney\t1",
            "Hs\tKRT8\tLuminal epithelial cells\tMammary gland\t1"]
    p.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))
    cl = ref / "cl" / "2026-06-08" / "cl-basic.obo"
    cl.parent.mkdir(parents=True)
    cl.write_text(OBO)


def test_a_breast_only_label_has_no_canonical_set_in_a_kidney_view(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(ref))
    _kidney_cache(ref)
    v = view.for_study(_study(tmp_path, "kid", KIDNEY))
    assert v.canonical("podocyte") == ("NPHS2",)
    assert v.canonical("Luminal epithelial cells") == ()        # no cross-tissue fallback
    assert "not in reference data" in v.reads_as("Luminal epithelial cells")
    # labels match through a real exact synonym, never an ambiguous one
    assert v.canonical("glomerular visceral epithelial cell") == ("NPHS2",)
    assert v.canonical("shared name") == ()


def test_the_demo_view_reproduces_the_demo_canonical_lists(demo_id, datasets_root) -> None:
    """The lab list replaced the hardcoded breast dict without moving a set."""
    demo = Study.open(demo_id, data_root=datasets_root.parent)
    ann = json.loads((demo.interp / "annotation.json").read_text())["clusters"]
    for meta in ann.values():
        got = annotation.canonical_markers(meta["cell_type"], demo_id)
        assert set(got) == set(meta["canonical_markers"]), meta["cell_type"]


def test_an_expected_type_absent_from_the_view_is_not_a_candidate(tmp_path) -> None:
    lab = "cell_type,gene,tissue\nT cell,CD3E,t\n"
    s = _study(tmp_path, "s", "t", lab,
               background={"expected_cell_types": [{"value": "Zeta cell"}]})
    v = view.for_study(s)
    assert v.candidates() == ("T cell",)
    assert v.canonical("Zeta cell") == ()


def test_a_lab_list_edge_carries_licence_lab_and_its_canonical_flag(tmp_path) -> None:
    lab = "cell_type,gene,tissue,canonical\nT cell,CD3E,t,yes\nT cell,VIM,t,no\n"
    v = view.for_study(_study(tmp_path, "s", "t", lab))
    assert {e.licence for e in v.marks("T cell")} == {"lab"}
    assert v.releases[0][0] == "lab:t"
    assert v.markers("T cell") == ("CD3E", "VIM")
    assert v.canonical("T cell") == ("CD3E",)


def test_a_corrected_tissue_rebuilds_the_view(tmp_path) -> None:
    """Re-answering tissue at intake must not keep the first tissue's lists."""
    from panoscope.data import study as study_mod

    lab = "cell_type,gene,tissue\nT cell,CD3E,a\nT cell,MS4A1,b\n"
    s = _study(tmp_path, "s", "a", lab)
    assert view.for_study(s).canonical("T cell") == ("CD3E",)
    (s.dir / "manifest.json").write_text(json.dumps({"dataset_id": "s", "tissue": "b"}))
    study_mod.clear_cache()
    assert view.for_study(s).canonical("T cell") == ("MS4A1",)


def test_a_tissue_id_chosen_at_intake_keys_the_organ_sources_beside_prose(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(ref))
    _kidney_cache(ref)
    assert view.for_study(_study(tmp_path, "a", "kidney")).canonical("podocyte") == ()
    v = view.for_study(_study(tmp_path, "b", "kidney", tissue_id=KIDNEY))
    assert v.canonical("podocyte") == ("NPHS2",)
    assert v.scope == ("panglaodb:kidney: species Hs (undeclared; read as human)",)


def test_names_join_on_case_and_plural_never_fuzzily(tmp_path):
    lab = "cell_type,gene,tissue\nMacrophages,CD68,t\nT cells,CD3E,t\n"
    s = _study(tmp_path, "s", "t", lab)
    (s.dir / view.LAB_SYNONYMS).write_text("synonym,cell_type\nTAM,Macrophages\n")
    v = view.for_study(s)
    assert v.canonical("macrophage") == ("CD68",)
    assert v.canonical("TAM") == () and ("TAM", "Macrophages") in v.synonyms()  # names a rival only
    assert v.canonical("T cell") == ("CD3E",)
    assert v.canonical("Macrophage-like") == ()                 # never fuzzy
    assert ("lab-synonyms", v.releases[1][1]) in v.releases


def test_a_cl_id_joins_a_source_name_to_the_ontology_term(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(ref))
    _kidney_cache(ref)
    lab = "cell_type,gene,tissue\nPodocytes,PODXL,k\n"
    v = view.for_study(_study(tmp_path, "s", "k", lab))
    assert v.canonical("glomerular visceral epithelial cell") == ("PODXL",)  # plural -> CL id


def test_only_a_sources_own_canonical_markers_become_off_panel_absences(tmp_path, monkeypatch):
    ref = tmp_path / "ref"
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(ref))
    p = ref / "panglaodb" / PanglaoDB.release / PanglaoDB.filename
    p.parent.mkdir(parents=True)
    p.write_bytes(gzip.compress(b"species\tofficial gene symbol\tcell type\torgan\tcanonical marker\n"
                                b"Hs\tNPHS2\tpodocyte\tKidney\t1\nHs\tACTB\tpodocyte\tKidney\t0\n"))
    v = view.for_study(_study(tmp_path, "k", KIDNEY))
    assert v.markers("podocyte") == ("NPHS2",)


def test_disease_state_narrows_what_carries_it_and_the_rest_say_so(tmp_path):
    srcs, scope = view._sources({"tissue": "t", "tissue_id": KIDNEY, "disease_state": "healthy",
                                 "species": "zebrafish"}, tmp_path)
    assert not any(isinstance(s, PanglaoDB) for s in srcs)       # no source carries the species
    assert scope == ("species 'zebrafish': no organ-keyed source carries it, none read",)
    srcs, scope = view._sources({"tissue_id": KIDNEY, "disease_state": "healthy"}, tmp_path)
    assert scope == ("panglaodb:kidney: species Hs (undeclared; read as human); "
                     "carries no disease state, not narrowed",)
