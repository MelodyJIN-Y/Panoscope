"""The brief for the biologist's own LLM: precomputed vocabulary, no placeholders, a stable
hash the import checks, and failed lookups named. Stubbed lookups; no network."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from panoscope.corpus import brief as BR
from panoscope.corpus import build as B
from panoscope.corpus.exchange import SCHEMA
from panoscope.corpus.window import _PLACEHOLDER, CellTypeEntry
from panoscope.reference.sources import SourceUnavailable
from tests.test_corpus_read import _ctx

STUDY = SimpleNamespace(id="kid", manifest={"tissue_label": "kidney", "background": {
    "papers_of_interest": [{"value": "pmid:123"}]}})
HGNC = {"symbol": "CXCL13", "alias_symbol": ["BLC", "BCA-1"],
        "name": "C-X-C motif chemokine ligand 13"}


@pytest.fixture
def stub(monkeypatch):
    frame = (CellTypeEntry("CD8 T cells"), CellTypeEntry("B cells", "CL:0000236",
                                                         (("B lymphocyte", "cl"),)))
    monkeypatch.setattr(B, "context_for", lambda s: _ctx(study=s.id, tissue="human kidney",
                                                         cell_types=frame))
    monkeypatch.setattr(B, "scope", lambda s, types: ("CXCL13",))
    monkeypatch.setattr(BR.vocab, "hgnc", lambda g: HGNC)
    monkeypatch.setattr(BR.vocab, "mesh_terms", lambda g: ("CXCL13 Chemokine",))
    monkeypatch.setattr(BR, "organ_disease_terms", lambda label: ("Lupus Nephritis",
                                                                 "Renal Cell Carcinoma"))
    return monkeypatch


def test_the_brief_carries_the_precomputed_vocabulary_and_no_placeholder(stub, tmp_path):
    out = BR.write(STUDY, tmp_path / "b")
    text = (out / "BRIEF.md").read_text(encoding="utf-8")
    for term in ('"BCA-1"', '"BLC"†', '"CXCL13 Chemokine"', '"C-X-C motif chemokine ligand 13"',
                 '"kidney"', '"human kidney"', '"Lupus Nephritis"', '"Renal Cell Carcinoma"',
                 '"CD8+ T cells"', '"CD8+T cell"', '"B lymphocyte"', "pmid:123",
                 "window W0*", B.brief_hash(STUDY), "Do NOT add the disease terms"):
        assert term in text, term
    assert not _PLACEHOLDER.search(text)
    ctx = json.loads((out / "study_context.json").read_text(encoding="utf-8"))
    assert ctx["brief_hash"] == B.brief_hash(STUDY)
    assert ctx["tissue"]["blocks"] == [["kidney", "human kidney", "Lupus Nephritis",
                                        "Renal Cell Carcinoma"]]
    assert (out / "format.schema.json").read_bytes() == SCHEMA.read_bytes()


def test_the_brief_hash_is_stable_for_a_study_and_moves_with_its_scope(stub):
    first = B.brief_hash(STUDY)
    assert B.brief_hash(STUDY) == first == BR.context(STUDY)["brief_hash"]
    stub.setattr(B, "scope", lambda s, types: ("CXCL13", "CD8A"))
    assert B.brief_hash(STUDY) != first


def test_a_failed_lookup_is_named_never_an_empty_vocabulary(stub):
    def down(*_a):
        raise SourceUnavailable("hgnc: 503")

    stub.setattr(BR.vocab, "hgnc", down)
    stub.setattr(BR, "organ_disease_terms", down)
    c = BR.context(STUDY)
    assert c["genes"][0]["aliases_status"] == "FAILED: hgnc: 503"
    assert [a["text"] for a in c["genes"][0]["aliases"]] == ["CXCL13", "CXCL13 Chemokine"]
    assert c["organ_disease_terms"] == {"status": "FAILED: hgnc: 503", "source": "MeSH",
                                        "terms": []}
    stub.setattr(BR, "organ_disease_terms", lambda label: ())
    assert BR.context(STUDY)["organ_disease_terms"]["status"] == \
        'none: no MeSH heading "kidney Diseases"'


def test_organ_disease_terms_keep_two_levels_of_natural_order_entry_terms(monkeypatch):
    rows = {"1": ("C12.1", ["Kidney Diseases", "Kidney Disease", "Diseases, Kidney"]),
            "2": ("C12.1.5.7", ["Renal Cell Carcinoma", "Carcinoma (Kidney)"]),
            "3": ("C12.1.5.7.9", ["Too Deep"])}

    def get(url, params):
        if url.endswith("esearch.fcgi"):
            return {"esearchresult": {"idlist": ["1"] if "[MH]" in params["term"] else
                                      ["1", "2", "3"]}}
        ids = params["id"].split(",")
        return {"result": {"uids": ids, **{i: {"ds_idxlinks": [{"treenum": rows[i][0]}],
                                               "ds_meshterms": rows[i][1]} for i in ids}}}

    monkeypatch.setattr(BR.vocab, "_get", get)
    assert BR.organ_disease_terms("kidney") == ("Kidney Diseases", "Renal Cell Carcinoma")
