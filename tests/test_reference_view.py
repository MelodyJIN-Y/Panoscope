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
    assert v.cl_id("podocyte") == "CL:0000653"
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
