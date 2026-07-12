"""The annotate stage: parse the skill's JSON, ground off-panel, generate offline.

The live agent call is mocked, so these stay deterministic and network-free; the
run writes only under a pytest tmp dir (never the bundled demo tree).
"""
from __future__ import annotations

from pipeline.stages import annotate


def test_parse_accepts_valid_and_rejects_bad():
    good = ('{"cell_type":"Tumor","cell_type_short":"Tum_Epi","category":"Epithelial",'
            '"lineage":"Epithelial","canonical_markers":["ERBB2","EPCAM"]}')
    obj = annotate._parse(good)
    assert obj and obj["cell_type"] == "Tumor"
    assert annotate._parse("no json here") is None
    assert annotate._parse('{"cell_type":"Tumor"}') is None  # missing required fields


def test_record_computes_offpanel_against_the_panel():
    obj = {"cell_type": "Tumor", "cell_type_short": "Tum", "category": "Epithelial",
           "lineage": "Epithelial", "canonical_markers": ["ERBB2", "ZZZ_NOT_ON_PANEL"]}
    rec = annotate._record("c1", obj, [("ERBB2", 21.0, 0.9)])
    assert rec["cell_type"] == "Tumor"
    assert "ZZZ_NOT_ON_PANEL" in rec["offpanel_canonical"]  # never measured on this panel
    assert "ERBB2" not in rec["offpanel_canonical"]         # on panel


def test_record_fail_soft_unknown_when_model_unusable():
    rec = annotate._record("c1", None, [("ERBB2", 21.0, 0.9), ("KRT7", 13.0, 0.9)])
    assert rec["cell_type"] == "Unknown"
    assert rec["canonical_markers"][:1] == ["ERBB2"]


def test_run_annotate_generates_offline(monkeypatch, tmp_path):
    class _Resp:
        text = ('{"cell_type":"Tumor","cell_type_short":"Tum","category":"Epithelial",'
                '"lineage":"Epithelial","canonical_markers":["ERBB2","EPCAM"]}')

    monkeypatch.setattr(annotate.agent_loop, "chat", lambda *a, **k: _Resp())
    result = annotate.run_annotate(root=tmp_path, force=True)
    assert result and all(v["cell_type"] == "Tumor" for v in result.values())
