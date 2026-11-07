"""Retrieval channels (harness v2 §5): query invariant, paging, narrowing, dedupe, gate,
full text, co-occurrence sentences and resume. Recorded synthetic responses only."""

from __future__ import annotations

import re
from datetime import date

import pytest

from panoscope.corpus import channels as C
from panoscope.corpus.retrieve import key_state
from panoscope.corpus.store import QueryStatus, load
from panoscope.corpus.window import CellTypeEntry, StudyContext, Term
from panoscope.reference.sources import SourceUnavailable

TODAY = date(2026, 11, 8)
POSTN = (Term("POSTN", "hgnc_symbol", False), Term("periostin", "hgnc_name", False),
         Term("OSF2", "hgnc_prev", False))


def _ctx(**kw):
    base = dict(study="s", tissue="breast", tissue_id="UBERON:0000310", platform="Xenium",
                modality_class="spatial", release_status="unpublished",
                cell_types=(CellTypeEntry("Fibroblasts", None, (("fibroblast", "lab"),)),
                            CellTypeEntry("Myoepithelial", None, (("basal cell", "CL"),))))
    return StudyContext(**{**base, **kw})


def _hit(pmid="", doi="", title="A paper", abstract="", oa="N", pmcid="",
         types=("Journal Article",), preprint_of="", source="MED"):
    h = {"id": pmid or doi, "source": source, "pmid": pmid or None, "doi": doi, "title": title,
         "abstractText": abstract, "isOpenAccess": oa, "pmcid": pmcid,
         "pubTypeList": {"pubType": list(types)}, "firstPublicationDate": "2026-01-01"}
    if preprint_of:
        h["commentCorrectionList"] = {"commentCorrection": [
            {"source": "MED", "id": preprint_of, "type": "Preprint of"}]}
    return h


JATS = ("<article><front/><body><sec><title>Results</title><p>POSTN was high in "
        "fibroblasts. Basal cells lacked periostin.</p><p>Tumour size varied.</p>"
        "<fig><caption><title>POSTN map</title><p>Shown in breast.</p></caption></fig>"
        "<table><tr><td>POSTN</td><td>2.1</td></tr></table>"
        "</sec></body></article>")


def _c3(q: str) -> bool:
    return q.startswith("(RESULTS:") and "FIRST_PDATE" not in q and "Fibroblast" in q


class World:
    """Recorded responses keyed by what the query asks for; counts every call."""

    def __init__(self, epmc=(), pubmed=(), meta=(), fail=()):
        self.epmc, self.pubmed, self.fail = list(epmc), list(pubmed), set(fail)
        self.meta = {h["pmid"]: h for h in meta}
        self.calls: list[str] = []

    def __call__(self, url, params, raw=False):
        q = params.get("query") or params.get("term") or url
        self.calls.append(q)
        if any(f in q for f in self.fail):
            raise SourceUnavailable(q)
        if raw:
            return JATS
        if url.endswith("esearch.fcgi"):
            for pred, n, ids in self.pubmed:
                if pred(q):
                    return {"esearchresult": {"count": str(n),
                                              "idlist": ids[:int(params["retmax"])]}}
            return {"esearchresult": {"count": "0", "idlist": []}}
        if "EXT_ID:" in q:
            res = [self.meta[i] for i in re.findall(r"EXT_ID:(\d+)", q) if i in self.meta]
            return {"hitCount": len(res), "resultList": {"result": res}}
        for pred, hits in self.epmc:
            if pred(q):
                start = 0 if params["cursorMark"] == "*" else int(params["cursorMark"])
                return {"hitCount": len(hits), "nextCursorMark": str(start + 100),
                        "resultList": {"result": hits[start:start + 100]}}
        return {"hitCount": 0, "nextCursorMark": "*", "resultList": {"result": []}}


def _run(monkeypatch, tmp_path, world, ctx=None, aliases=POSTN, **kw):
    monkeypatch.setattr(C, "_get", world)
    return C.retrieve(ctx or _ctx(), "POSTN", aliases, today=TODAY,
                      path=tmp_path / "corpus.sqlite", **kw)


# --------------------------------------------------------------------------- #
def test_a_frame_label_matches_its_marker_plus_and_singular_spellings_but_not_a_symbol():
    cells = C.matcher((), ("CD8 T cells", "Podocytes"))
    assert all(cells.search(t) for t in ("exhausted CD8+T cells", "a CD8+ T cell", "podocyte"))
    assert not cells.search("CD8 T") and not C.matcher((Term("PSORS", "x", True),)).search("PSOR")


def test_a_gene_only_or_gene_plus_generic_query_cannot_be_built():
    g = '("POSTN")'
    for bad in (dict(), dict(extra=(C._or(C.M_CELL),))):
        with pytest.raises(ValueError, match="anchor"):
            C.compose(g, **bad)
    for ok in (dict(tissue='("breast")'), dict(cells='("fibroblast")'),
               dict(cites="CITES:1_MED")):
        assert C.compose(g, **ok).startswith(("CITES", g))


def test_every_logged_search_carries_a_tissue_cell_or_cites_anchor(monkeypatch, tmp_path):
    c = _run(monkeypatch, tmp_path, World(), ctx=_ctx(release_status="confirmed",
                                                      release_pmid="38114474"))
    searches = [q for q in c.queries if q.channel[:2] in ("C0", "C1", "C2", "C3", "C4")]
    assert {q.channel for q in searches} == {"C0", "C3", "C4:C3", "C1", "C4:PPR"}
    for q in searches:
        assert ("CITES:38114474_MED" in q.query or '"breast"' in q.query
                or '"fibroblast"' in q.query or '"basal cell"' in q.query)


def test_restricted_aliases_are_proximity_only_in_c1(monkeypatch, tmp_path):
    c = _run(monkeypatch, tmp_path, World(),
             aliases=(*POSTN, Term("OSF", "hgnc_alias", True)))
    c1 = next(q.query for q in c.queries if q.channel == "C1")
    assert '"OSF Fibroblasts"[tiab:~6]' in c1 and '"OSF"[tiab]' not in c1
    assert all(q.status is QueryStatus.OK for q in c.queries if q.channel == "C3")
    # a gene with only restricted aliases still searches, and its EPMC lanes read degraded
    c = _run(monkeypatch, tmp_path / "b", World(), aliases=(Term("CD4", "study", True),))
    assert {q.status for q in c.queries if q.channel == "C3"} == {QueryStatus.DEGRADED}


def test_epmc_pages_to_exhaustion_newest_first(monkeypatch, tmp_path):
    hits = [_hit(pmid=str(1000 + i), abstract="POSTN") for i in range(150)]
    w = World(epmc=[(_c3, hits)])
    c = _run(monkeypatch, tmp_path, w)
    c3 = next(q for q in c.queries if q.channel == "C3" and q.cell_type == "Fibroblasts")
    assert (c3.total, c3.retrieved, c3.status) == (150, 150, QueryStatus.OK)
    assert len(set(c3.ids)) == 150


def test_over_the_cap_narrows_and_logs_each_step(monkeypatch, tmp_path):
    w = World(pubmed=[(lambda q: "Xenium" in q and "Fibroblast" in q, 3, ["1", "2", "3"]),
                      (lambda q: "Fibroblast" in q, 500, [])],
              meta=[_hit(pmid=p, abstract="POSTN") for p in "123"])
    w.fail = {"Xenium"}                       # the narrowed step fails; a re-run resumes it
    _run(monkeypatch, tmp_path, w)
    w.fail = set()
    c = _run(monkeypatch, tmp_path, w)
    c1 = [q for q in c.queries if q.channel == "C1" and q.cell_type == "Fibroblasts"]
    assert [q.status for q in c1] == [QueryStatus.NARROWED, QueryStatus.OK]
    assert not {"C2", "C4:C1"} & {q.channel for q in c.queries}
    assert not any(g.startswith("C1") for g in key_state(c, "POSTN", "Fibroblasts").gaps)


def test_still_over_after_narrowing_is_truncated_and_runs_c2_and_recency(monkeypatch, tmp_path):
    ids = [str(i) for i in range(1, 400)]
    w = World(pubmed=[(lambda q: "Fibroblast" in q and "~6" not in q, 399, ids)])
    c = _run(monkeypatch, tmp_path, w)
    c1 = [q for q in c.queries if q.channel == "C1" and q.cell_type == "Fibroblasts"]
    assert [q.status for q in c1] == [QueryStatus.NARROWED] * 3 + [QueryStatus.TRUNCATED]
    assert (c1[-1].total, c1[-1].retrieved) == (399, C.CAP)
    assert '"Fibroblasts"[ti]' in c1[2].query and '"2023/11/08"[edat]' in c1[3].query
    assert {"C2", "C4:C1"} <= {q.channel for q in c.queries if q.cell_type == "Fibroblasts"}
    assert "C1 truncated" in key_state(c, "POSTN", "Fibroblasts").gaps
    assert not {"C2", "C4:C1"} & {q.channel for q in c.queries
                                  if q.cell_type == "Myoepithelial"}


def test_dedupe_by_pmid_then_doi_and_preprints_collapse(monkeypatch, tmp_path):
    w = World(epmc=[(_c3, [_hit(pmid="1", doi="10.1/A", abstract="POSTN"),
                           _hit(doi="10.1/a", abstract="POSTN", source="PPR"),
                           _hit(doi="10.2/pre", abstract="POSTN", source="PPR",
                                preprint_of="2")])],
              meta=[_hit(pmid="2", abstract="POSTN in fibroblasts")])
    c = _run(monkeypatch, tmp_path, w)
    led = {d.id: d.disposition for d in c.ledger}
    assert led["doi:10.1/a"] == "DEDUP_OF:pmid:1"
    assert led["doi:10.2/pre"] == "DEDUP_OF:pmid:2"
    assert "pmid:1" not in led and "pmid:2" not in led       # open: awaiting a read


def test_the_gate(monkeypatch, tmp_path):
    w = World(epmc=[(_c3, [_hit(pmid="10", abstract="no alias here")])],
              pubmed=[(lambda q: "Fibroblast" in q and "Xenium" not in q, 4,
                       ["11", "12", "13", "14"])],
              meta=[_hit(pmid="11", abstract="POSTNL1 only"),
                    _hit(pmid="12", abstract="POSTN rises", types=("Review",)),
                    _hit(pmid="13", abstract="POSTN", types=("Retracted Publication",)),
                    _hit(pmid="14", abstract="Periostin in stroma."),
                    _hit(pmid="15", abstract="nothing on the gene"),
                    _hit(pmid="16", abstract="POSTN in stroma")])
    c = _run(monkeypatch, tmp_path, w, papers_of_interest=("15", "16"))
    led = {d.id: d.disposition for d in c.ledger}
    assert "pmid:10" not in led                         # a C3 body hit passes without alias
    assert led["pmid:11"] == "PREFILTER_NO_ALIAS"
    assert led["pmid:12"] == "REVIEW_CHASE_ONLY"
    assert led["pmid:13"] == "RETRACTED"
    assert "pmid:14" not in led                         # name alias, case-folded
    assert led["pmid:15"] == "PREFILTER_NO_ALIAS"       # C5 is screened, never auto-accepted
    assert "pmid:16" not in led
    # another gene whose alias the paper names reopens it; one that does not, never closes
    # a paper an earlier gene's gate holds open
    monkeypatch.setattr(C, "_get", World(meta=w.meta.values()))
    c = C.retrieve(_ctx(), "GENE2", (Term("gene", "study", False),), today=TODAY,
                   path=tmp_path / "corpus.sqlite", papers_of_interest=("15",))
    led = {d.id for d in c.ledger}
    assert "pmid:15" not in led and "pmid:16" not in led


def test_full_text_for_gated_oa_and_the_cooccurrence_index(monkeypatch, tmp_path):
    w = World(epmc=[(_c3, [_hit(pmid="20", abstract="POSTN in breast.", oa="Y", pmcid="PMC20"),
                           _hit(pmid="21", abstract="POSTN.", oa="N", pmcid="PMC21")])])
    c = _run(monkeypatch, tmp_path, w)
    rec = {r.uri: r for r in c.records}
    ft = rec["pmid:20"].full_text
    assert ft.startswith("## Results\nPOSTN was high in fibroblasts.")
    assert "\nPOSTN map\n" in ft                         # a caption title is not a heading
    assert ft.endswith("\nPOSTN | 2.1")                  # table rows kept
    assert not rec["pmid:21"].full_text and sum("fullTextXML" in x for x in w.calls) == 1
    frame = [t for e in _ctx().cell_types for t in C.cell_terms(e)]
    got = C.cooccurrence(rec["pmid:20"], POSTN, frame)
    assert [s.text for s in got] == ["POSTN was high in fibroblasts.",
                                     "Basal cells lacked periostin."]   # a rival type too
    assert all(rec["pmid:20"].text(s.section)[s.start:s.end] == s.text for s in got)


def test_a_failed_full_text_is_logged_and_the_abstract_stays(monkeypatch, tmp_path):
    w = World(epmc=[(_c3, [_hit(pmid="20", abstract="POSTN.", oa="Y", pmcid="PMC20")])],
              fail={"PMC20"})
    c = _run(monkeypatch, tmp_path, w)
    assert [q.status for q in c.queries if q.channel == "fulltext"] == [QueryStatus.FAILED]
    assert {r.uri: r for r in c.records}["pmid:20"].abstract == "POSTN."


def test_resume_reruns_nothing_done_and_retries_what_failed(monkeypatch, tmp_path):
    hits = [_hit(pmid="30", abstract="POSTN", oa="Y", pmcid="PMC30")]
    first = World(epmc=[(_c3, hits)], fail={"Myoepithelial"})
    _run(monkeypatch, tmp_path, first)
    failed = [q for q in load(tmp_path / "corpus.sqlite").queries
              if q.status is QueryStatus.FAILED]
    assert failed and all(q.cell_type == "Myoepithelial" for q in failed)
    again = World(epmc=first.epmc)
    c = _run(monkeypatch, tmp_path, again)
    assert again.calls and all("Myoepithelial" in q for q in again.calls)
    assert not [q for q in c.queries if q.status is QueryStatus.FAILED]
    assert len({(q.request_hash, q.cell_type) for q in c.queries}) == len(c.queries)


def test_an_unresolved_release_paper_degrades_c0(monkeypatch, tmp_path):
    c = _run(monkeypatch, tmp_path, World(), ctx=_ctx(release_status="unresolved"))
    assert "C0 degraded" in key_state(c, "POSTN", "Fibroblasts").gaps
    assert not any("CITES:" in q.query for q in c.queries if q.status is QueryStatus.OK)


def test_a_europe_pmc_search_is_posted_so_a_long_vocabulary_is_not_refused(monkeypatch):
    import urllib.request

    sent = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"hitCount": 0, "resultList": {"result": []}}'

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout: sent.append(req) or Resp())
    long = "(" + " OR ".join(f'RESULTS:"alias {i}"' for i in range(300)) + ")"
    assert C._search(long, pubmed=False, cap=1, probe=True) == (0, [])
    assert "?" not in sent[0].full_url and b"alias+299" in sent[0].data


@pytest.mark.live
def test_live_epmc_cites_query_smoke():
    total, hits = C._search('CITES:38114474_MED AND ("POSTN")', pubmed=False, cap=2, probe=True)
    assert total >= 1 and hits and all("title" in h for h in hits)
