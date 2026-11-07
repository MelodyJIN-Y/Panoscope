"""The corpus window: input contract, vocabulary, rung, species, modality, and the
vocabulary fetch with the network stubbed. Synthetic genes only ("GENEX")."""

from __future__ import annotations

import dataclasses
import itertools
import json
import urllib.request
from types import SimpleNamespace as NS

import pytest

from panoscope.corpus import vocab as F
from panoscope.corpus import window as W
from panoscope.corpus.retrieve import Support, key_state
from panoscope.corpus.store import Corpus, CorpusUnavailable, QueryRun, QueryStatus
from panoscope.corpus.window import Facets, Rung, Species, rung
from panoscope.data.study import Study
from panoscope.reference.sources import SourceUnavailable

DEMO = "xenium_hbreast_sample1"
CTX = dict(study="s1", cell_types=(W.CellTypeEntry("Fibroblasts"),), tissue="breast",
           tissue_id="UBERON:0000310", platform="Xenium",
           modality_class="imaging_spatial_transcriptomics")


# --------------------------------------------------------------------------- #
# §2 input contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [
    {"tissue_id": ""}, {"modality_class": ""}, {"cell_types": ()},
    {"disease_state": "{DISEASE}"}, {"release_status": "confirmed"},
    {"release_status": "maybe"},
])
def test_the_study_context_fails_closed(bad):
    with pytest.raises(CorpusUnavailable):
        W.StudyContext(**{**CTX, **bad})


def test_an_undeclared_disease_state_is_the_w0_star_window():
    assert W.StudyContext(**CTX).w0_star
    assert not W.StudyContext(**CTX, disease_state="DCIS").w0_star


def test_the_frame_is_labels_plus_candidates_joined_by_key_with_synonyms():
    edge = NS(predicate="synonym", subject=NS(id="fibroblast"), object=NS(id="Fibroblasts"),
              source="cell-ontology@2026")
    view = NS(candidates=lambda: ("fibroblast", "T cell"),
              key=lambda n: {"fibroblasts": "CL:0000057", "fibroblast": "CL:0000057"}
              .get(n.casefold(), n.casefold()),
              graph=NS(edges=(edge,)))
    frame = W.cell_type_frame(view, ("Fibroblasts", "Mystery cells"))
    assert [c.label for c in frame] == ["Fibroblasts", "Mystery cells", "T cell"]
    assert frame[0].ontology_id == "CL:0000057"
    assert frame[0].synonyms == (("fibroblast", "cell-ontology"),)


def test_the_demo_context_reads_its_frame_and_refuses_without_a_tissue_id():
    demo = Study.open(DEMO)
    with pytest.raises(CorpusUnavailable, match="tissue_id"):
        W.context_for(demo)
    fake = NS(id=demo.id, dir=demo.dir, unit_labels=demo.unit_labels,
              platform=demo.platform,
              manifest={**demo.manifest, "tissue_id": "UBERON:0000310"})
    ctx = W.context_for(fake)
    labels = {v["cell_type"] for v in demo.unit_labels.values()}
    assert labels <= {c.label for c in ctx.cell_types}
    assert ctx.platform == "Xenium" and ctx.w0_star
    assert ctx.release_status == "unresolved"


# --------------------------------------------------------------------------- #
# §3 vocabulary
# --------------------------------------------------------------------------- #
def test_the_restriction_rule():
    for t in ("AR", "CD4", "cd105", "REST", "SARS"):
        assert W.restricted(t), t
    for t in ("POSTN", "periostin", "CXCL13"):
        assert not W.restricted(t), t


def test_gene_terms_keep_origin_and_never_invent_without_hgnc():
    doc = {"symbol": "GENEX", "prev_symbol": ["OLDX"], "alias_symbol": ["GX1", "genex"],
           "name": "gene x protein"}
    terms = W.gene_terms("GENEX", doc, ("Gene X",))
    assert [(t.text, t.origin) for t in terms] == [
        ("GENEX", "hgnc_symbol"), ("OLDX", "hgnc_prev"), ("GX1", "hgnc_alias"),
        ("gene x protein", "hgnc_name"), ("Gene X", "mesh")]
    assert [t.origin for t in W.gene_terms("GENEX", None)] == ["study"]


# --------------------------------------------------------------------------- #
# §4 ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("facets,declared,want", [
    (Facets(True, True, True, True), True, Rung.W0),
    (Facets(True, True, None, True), False, Rung.W0_STAR),
    (Facets(True, None, True, True), True, Rung.W1),
    (Facets(True, True, False, True), True, Rung.W2),
    (Facets(False, True, None, True), True, Rung.W3),
    (Facets(True, True, True, True, Species.NON_HUMAN), True, Rung.W4),
    (Facets(None, True, None, False), True, Rung.W5),
    (Facets(True, True, True, True, resolution="BULK"), True, Rung.W1),
    (Facets(True, True, True, True, biofluid=True), True, None),
    (Facets(None, None, None, True), True, None),
])
def test_each_rung(facets, declared, want):
    assert rung(facets, disease_declared=declared) is want


def _width(r: Rung | None) -> int:
    return 9 if r is None else r.width


def test_removing_a_verified_facet_never_narrows():
    """Exhaustive over the facet space: True -> None, or human -> UNSTATED."""
    tri = (True, False, None)
    for t, c, d, v, sp, res, decl in itertools.product(
            tri, tri, tri, tri, tuple(Species), ("UNKNOWN", "BULK"), (True, False)):
        f = Facets(t, c, d, v, sp, resolution=res)
        before = _width(rung(f, disease_declared=decl))
        weaker = [dataclasses.replace(f, **{n: None}) for n in
                  ("tissue", "cell_type", "disease", "in_vivo") if getattr(f, n) is True]
        if sp in (Species.HUMAN_STATED, Species.HUMAN_INFERRED):
            weaker.append(dataclasses.replace(f, species=Species.UNSTATED))
        for g in weaker:
            assert _width(rung(g, disease_declared=decl)) >= before, (f, g)


@pytest.mark.parametrize("span,methods,read,want", [
    ("GENEX in human HER2 tumours", "", "", Species.UNSTATED),
    ("GENEX in tumours from 40 patients", "", "", Species.HUMAN_STATED),
    ("GENEX in human breast tissue samples", "", "", Species.HUMAN_STATED),
    ("GENEX in tumours", "", "surgical resection cohort", Species.HUMAN_INFERRED),
    ("GENEX in tumours", "", "resection; orthotopic mouse model", Species.UNSTATED),
    ("GENEX in tumours", "murine 4T1 model", "", Species.NON_HUMAN),
])
def test_species_state(span, methods, read, want):
    assert W.species_state(span, methods, read) is want


@pytest.mark.parametrize("text,want", [
    ("profiled by Xenium", "SAME_PLATFORM"),
    ("MERFISH imaging", "SAME"),
    ("scRNA-seq of dissociated tumours", "ADJACENT_RNA"),
    ("immunohistochemistry on sections", "OTHER_ANALYTE"),
    ("Visium spots", "SPOT"),
    ("bulk RNA-seq", "BULK"),
    ("we observed", "UNKNOWN"),
])
def test_modality_match(text, want):
    assert W.modality_match(text, "Xenium") == want


# --------------------------------------------------------------------------- #
# Vocabulary fetch, network stubbed
# --------------------------------------------------------------------------- #
def _fake_get(fail_hgnc: bool = False):
    calls: list[str] = []

    def get(url, params, **_):
        calls.append(url)
        if url.startswith(F.HGNC):
            if fail_hgnc:
                raise SourceUnavailable("offline")
            return {"response": {"docs": [{"symbol": "GENEX", "alias_symbol": ["GX1"]}]}}
        if params.get("db") == "mesh" and url.endswith("esearch.fcgi"):
            return {"esearchresult": {"idlist": ["1", "2"]}}
        if url.endswith("esummary.fcgi"):
            # The live shape: the symbol is inside entry terms, never one on its own.
            return {"result": {"uids": ["1", "2", "3"],
                               "1": {"ds_meshterms": ["Gene X", "GENEX Chemokine"]},
                               "2": {"ds_meshterms": ["Unrelated"]},
                               "3": {"ds_meshterms": ["Genex protein, mouse"]}}}
        return {"esearchresult": {"count": str(len(params["term"]))}}
    return get, calls


def test_vocabulary_logs_a_census_row_per_alias_and_caches_hgnc(monkeypatch):
    get, calls = _fake_get()
    monkeypatch.setattr(F, "_get", get)
    terms, runs = F.vocabulary("GENEX", ("breast",))
    assert [(t.text, t.origin) for t in terms] == [
        ("GENEX", "hgnc_symbol"), ("GX1", "hgnc_alias"), ("Gene X", "mesh"),
        ("GENEX Chemokine", "mesh")]
    census = [r for r in runs if r.channel == "census"]
    assert census[0].query == '"GENEX"[tiab] AND ("breast"[tiab])'
    assert len(census) == 4 and all(r.total > 0 for r in census)
    n = sum(u.startswith(F.HGNC) for u in calls)
    F.vocabulary("GENEX", ("breast",))
    assert sum(u.startswith(F.HGNC) for u in calls) == n      # served from the cache


def test_a_failed_hgnc_lookup_keeps_the_gene_incomplete_never_thin(monkeypatch):
    get, _ = _fake_get(fail_hgnc=True)
    monkeypatch.setattr(F, "_get", get)
    _, runs = F.vocabulary("GENEX", ("breast",))
    assert [r.channel for r in runs if r.status is QueryStatus.FAILED] == ["vocab:hgnc"]
    searched = QueryRun(query='"GENEX"[tiab]', channel="C1", gene="GENEX",
                        cell_type="Fibroblasts")
    s = key_state(Corpus(queries=(*runs, searched)), "GENEX", "Fibroblasts")
    assert s.state is Support.NONE_FOUND and s.gaps == ("vocab:hgnc failed",)


def test_eutils_requests_name_the_tool_and_send_email_only_from_env(monkeypatch):
    seen: list[str] = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"esearchresult": {"count": "3"}}).encode()

    def fake_open(req, timeout=0):
        seen.append(req.full_url)
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_open)
    monkeypatch.delenv("NCBI_EMAIL", raising=False)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    assert F.count('"GENEX"[tiab]') == 3
    assert "tool=panoscope" in seen[-1] and "email=" not in seen[-1]
    monkeypatch.setenv("NCBI_EMAIL", "lab@example.org")
    F.count('"GENEX"[tiab]')
    assert "email=lab%40example.org" in seen[-1]
