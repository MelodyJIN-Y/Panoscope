"""The Cell Ontology, CellMarker 2.0 and lab-curated sources, the tissue map, and the
``reference fetch`` verb. Every file is synthetic and written to ``tmp_path``; only the
``live`` tests touch the network."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from panoscope import cli
from panoscope.reference import sources as S
from panoscope.reference.graph import NotClaimable
from panoscope.reference.sources import (
    CellMarker, CellOntology, LabCurated, PanglaoDB, SourceUnavailable,
    UnrecognisedSource,
)

OBO = """format-version: 1.2
data-version: cl/releases/2026-06-08/cl-basic.owl

[Term]
id: CL:0000001
name: stromal cell

[Term]
id: CL:0000002
name: fibroblast
synonym: "fibrocyte" EXACT []
synonym: "FB" RELATED []
is_a: CL:0000001 ! stromal cell
is_a: UBERON:0000001 ! not a cell type

[Term]
id: CL:0000003
name: old cell
is_obsolete: true

[Term]
id: CP:0000001
name: not a CL term

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture
def cache(tmp_path, monkeypatch):
    root = tmp_path / "refcache"
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(root))
    return root


@pytest.mark.req("CORP-B4", "PLAN-P2.1-01")
def test_cell_ontology_reads_live_cl_terms_with_is_a_and_exact_synonyms(cache):
    p = CellOntology().path
    p.parent.mkdir(parents=True)
    p.write_text(OBO)
    nodes = {n.entity.id: n.aliases for n in CellOntology().nodes()}
    assert nodes == {"stromal cell": {"CL:0000001"}, "fibroblast": {"CL:0000002"}}
    edges = {(e.subject.id, e.predicate, e.object.id) for e in CellOntology().edges()}
    assert edges == {("fibroblast", "is_a", "stromal cell"),
                     ("fibrocyte", "synonym", "fibroblast")}
    with pytest.raises(NotClaimable):                    # structure, never a claim
        next(iter(CellOntology().edges())).to_claim("x")


def _xlsx(path, header, rows):
    """A minimal .xlsx: header via shared strings, body as inline strings."""
    def cell(col, i, v, shared):
        ref = f"{chr(65 + col)}{i}"
        if shared:
            return f'<c r="{ref}" t="s"><v>{col}</v></c>'
        return f'<c r="{ref}" t="inlineStr"><is><t>{v}</t></is></c>'
    ns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    body = [f'<row r="1">{"".join(cell(c, 1, h, True) for c, h in enumerate(header))}</row>']
    body += [f'<row r="{i}">{"".join(cell(c, i, v, False) for c, v in enumerate(r) if v)}</row>'
             for i, r in enumerate(rows, start=2)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", f'<sst {ns}>'
                   + "".join(f"<si><t>{h}</t></si>" for h in header) + "</sst>")
        z.writestr("xl/worksheets/sheet1.xml",
                   f'<worksheet {ns}><sheetData>{"".join(body)}</sheetData></worksheet>')


CM_HEADER = ("species", "tissue_class", "cancer_type", "cell_name", "Symbol", "PMID")


def test_cellmarker_scopes_to_human_and_tissue_and_cites_the_row_pmid(cache):
    _xlsx(CellMarker("Breast").path, CM_HEADER, [
        ("Human", "Breast", "Normal", "Fibroblast", "POSTN", "11111111"),
        ("Human", "Breast", "Normal", "Luminal cell", "KRT8", ""),
        ("Mouse", "Breast", "Normal", "Fibroblast", "Postn", "22222222"),
        ("Human", "Kidney", "Normal", "Podocyte", "NPHS2", "33333333"),
    ])
    edges = {(e.subject.id, e.object.id, e.citation) for e in CellMarker("Breast").edges()}
    assert edges == {("POSTN", "Fibroblast", "pmid:11111111"),
                     ("KRT8", "Luminal cell", CellMarker.citation)}
    assert not any(n == "panel" for e in CellMarker("Breast").edges()
                   for n, _ in e.qualifiers)


def test_cellmarker_refuses_a_file_without_its_columns(cache):
    _xlsx(CellMarker("Breast").path, ("species", "cell_name"), [("Human", "B cell")])
    with pytest.raises(UnrecognisedSource):
        list(CellMarker("Breast").edges())


@pytest.mark.parametrize("src", [CellOntology(), CellMarker("Breast"),
                                 LabCurated(Path("/nonexistent/lab.csv"), "Breast")])
@pytest.mark.parametrize("read", ["nodes", "edges"])
def test_an_unfetched_source_is_unavailable_not_empty(cache, src, read):
    with pytest.raises(SourceUnavailable):
        list(getattr(src, read)())


def test_marker_sources_are_not_redistributable():
    """The licence gate keys on this flag; flipping it lets the file into the repo."""
    assert CellMarker.redistributable is False and LabCurated.redistributable is False


def test_a_fetch_that_returns_the_wrong_file_is_refused_and_writes_nothing(
        cache, monkeypatch):
    """A 403 page saved as the release would parse as "this source has no rows"."""
    class Page(io.BytesIO):
        def __enter__(self):
            return self

    monkeypatch.setattr(S.urllib.request, "urlopen",
                        lambda req, timeout: Page(b"<html>403 Forbidden</html>"))
    with pytest.raises(SourceUnavailable):
        CellOntology().fetch()
    assert not CellOntology().path.exists()


def test_lab_list_scopes_to_tissue_and_cites_the_file_when_the_row_does_not(tmp_path):
    f = tmp_path / "lab.csv"
    f.write_text("cell_type,gene,tissue,note,citation\n"
                 "Fibroblasts,POSTN,breast,stroma marker,\n"
                 "Fibroblasts,FAP,breast,,pmid:12345678\n"
                 "Podocytes,NPHS2,kidney,,\n")
    src = LabCurated(f, "Breast")
    edges = {(e.subject.id, e.citation, e.qualifier("note")) for e in src.edges()}
    assert {(g, n) for g, _, n in edges} == {("POSTN", "stroma marker"), ("FAP", None)}
    assert dict((g, c) for g, c, _ in edges)["FAP"] == "pmid:12345678"
    assert dict((g, c) for g, c, _ in edges)["POSTN"].startswith("file:sha256:")
    before = src.release
    f.write_text(f.read_text() + "B cells,MS4A1,breast,,\n")
    assert src.release != before                         # an edited list is a new release


@pytest.mark.req("PLAN-P2.2-01")
def test_an_unmapped_tissue_has_no_key_and_never_borrows_another():
    assert S.organ_key("UBERON:0000310", "panglaodb") == "Mammary gland"
    assert S.organ_key("UBERON:0002113", "panglaodb") == "Kidney"
    assert S.organ_key("UBERON:0002113", "cellmarker") is None
    assert S.organ_key("UBERON:9999999", "panglaodb") is None


def test_reference_fetch_reports_a_failed_source_and_exits_nonzero(monkeypatch, capsys):
    class Down:
        release, licence = "r1", "x"

        def fetch(self):
            raise SourceUnavailable("offline")

    monkeypatch.setattr(S, "fetchable", lambda: (Down(),))
    assert cli.main(["reference", "fetch"]) == 1
    assert "not fetched -- offline" in capsys.readouterr().out


def test_doctor_reports_each_source_release(cache, capsys):
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert f"CellOntology  {CellOntology.release}" in out and "not fetched" in out


@pytest.mark.live
@pytest.mark.parametrize("src", [CellOntology(), CellMarker(""), PanglaoDB("")])
def test_live_fetch_lands_in_the_cache(cache, src):
    assert src.fetch().is_file()


def test_cellmarker_narrows_to_normal_rows_and_carries_the_cl_id(cache):
    _xlsx(CellMarker("Breast").path, (*CM_HEADER, "cellontology_id"), [
        ("Human", "Breast", "Normal", "Fibroblast", "POSTN", "11111111", "CL_0000057"),
        ("Human", "Breast", "Breast Cancer", "Fibroblast", "FAP", "22222222", ""),
    ])
    normal = list(CellMarker("Breast", normal=True).edges())
    assert [(e.subject.id, e.qualifier("cl_id")) for e in normal] == [("POSTN", "CL:0000057")]
    assert [e.subject.id for e in CellMarker("Breast", normal=False).edges()] == ["FAP"]
    assert len(list(CellMarker("Breast").edges())) == 2          # undeclared: not narrowed


def test_a_refused_cellmarker_fetch_says_to_download_it_in_a_browser(cache, monkeypatch):
    def down(req, timeout):
        raise OSError("403")

    monkeypatch.setattr(S.urllib.request, "urlopen", down)
    with pytest.raises(SourceUnavailable, match="in a browser, save Cell_marker_All.xlsx"):
        CellMarker("").fetch()


def test_uberon_candidates_keep_only_uberon_terms_and_offline_is_unavailable(monkeypatch):
    class Page(io.BytesIO):
        def __enter__(self):
            return self

    body = (b'{"response": {"docs": [{"obo_id": "UBERON:0000310", "label": "breast"},'
            b'{"obo_id": "NCIT:C12971", "label": "Breast"},'
            b'{"obo_id": "UBERON:0000310", "label": "breast"}]}}')
    seen = []
    monkeypatch.setattr(S.urllib.request, "urlopen",
                        lambda req, timeout: seen.append(req.full_url) or Page(body))
    assert S.uberon_candidates("breast") == [("UBERON:0000310", "breast")]
    S.uberon_candidates("UBERON:0000310")
    assert "queryFields=obo_id" in seen[-1] and "queryFields" not in seen[0]

    def down(req, timeout):
        raise OSError("offline")

    monkeypatch.setattr(S.urllib.request, "urlopen", down)
    with pytest.raises(SourceUnavailable):
        S.uberon_candidates("breast")


def test_a_cellmarker_cl_id_joins_its_name_to_the_ontology_term(cache):
    from panoscope.reference import view

    (p := CellOntology().path).parent.mkdir(parents=True)
    p.write_text(OBO)
    _xlsx(CellMarker("Breast").path, (*CM_HEADER, "cellontology_id"), [
        ("Human", "Breast", "Normal", "Fibro-like cell", "POSTN", "11111111", "CL_0000002")])
    g = S.build(srcs=(CellOntology(), CellMarker("Breast"))).value
    v = view.View(g, index=view._index(g))
    assert v.canonical("fibrocyte") == ("POSTN",)          # CL synonym -> CL id <- cl_id
