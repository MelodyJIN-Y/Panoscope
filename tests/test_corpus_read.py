"""Model reading (P1/P2), code-side span checks and computed facets, cards, and the
corpus as the notes and reconciliation source. A stub reader only; no network, no model."""

from __future__ import annotations

import json
import re

import pytest

from panoscope.corpus import cards
from panoscope.corpus import read as R
from panoscope.corpus.retrieve import key_state
from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, CorpusVersion, QueryRun, QueryStatus, SourceRecord, Statement,
    corpus_path, load, save,
)
from panoscope.corpus.window import CellTypeEntry, StudyContext, Term

POSTN = (Term("POSTN", "hgnc_symbol", False), Term("periostin", "hgnc_name", False))
ABSTRACT = ("POSTN marks fibroblasts in human breast cancer tissue. "
            "Basal cells lacked POSTN expression.")
FULL = ("## Methods\nPatients' breast tumour samples were profiled with Xenium.\n"
        "## Results\nPOSTN was a specific marker of fibroblasts in breast tumours.\n"
        "Tumour size varied.")


def _ctx(**kw):
    base = dict(study="s", tissue="breast", tissue_id="UBERON:0000310", platform="Xenium",
                modality_class="spatial", release_status="unpublished",
                cell_types=(CellTypeEntry("Fibroblasts", None, (("fibroblast", "lab"),)),
                            CellTypeEntry("Myoepithelial", None, (("basal cell", "CL"),))))
    return StudyContext(**{**base, **kw})


def _corpus(tmp_path, *, uri="pmid:1", full=FULL):
    rec = SourceRecord(uri=uri, title="POSTN in breast", abstract=ABSTRACT, full_text=full,
                       via="Europe PMC", year=2026)
    rows = [QueryRun("prefilter gate for POSTN", "gate", "POSTN", total=1, retrieved=1,
                     ids=(uri,)),
            QueryRun("hgnc POSTN", "vocab:hgnc", "POSTN", total=1, retrieved=1),
            QueryRun('"POSTN"[tiab] AND breast', "census", "POSTN", total=3)]
    rows += [QueryRun(f"q {c}", "C1", "POSTN", c, total=1, retrieved=1)
             for c in ("Fibroblasts", "Myoepithelial")]
    p = tmp_path / "corpus.sqlite"
    save(Corpus(records=(rec,), queries=tuple(rows)), p)
    return p


class Stub:
    """P1 answers ``setting``; P2 answers ``statements`` and accounts sentences."""

    def __init__(self, statements=(), setting=None, account=True, fail=False):
        self.statements, self.account, self.fail = list(statements), account, fail
        self.setting = setting if setting is not None else {"tissue": "human breast cancer tissue"}
        self.calls: list[str] = []

    def __call__(self, system, user):
        self.calls.append("P1" if system == R.P1 else "P2")
        if self.fail:
            raise RuntimeError("model down")
        if system == R.P1:
            return json.dumps(self.setting), 100, 10
        ids = re.search(r"SENTENCES TO ACCOUNT FOR: (.*)$", user).group(1).split(", ")
        acc = [{"sentence_id": i, "statements": "no_claim"} for i in ids if i] \
            if self.account else []
        return json.dumps({"statements": self.statements if self.calls.count("P2") == 1
                           else [], "accounted": acc}), 200, 20


MARK = {"passage": "A", "span": "POSTN marks fibroblasts in human breast cancer tissue.",
        "cell_type": "Fibroblasts", "relation": "MARKS"}
LACK = {"passage": "A", "span": "Basal cells lacked POSTN expression.",
        "cell_type": "Myoepithelial", "relation": "NOT_EXPRESSED_IN"}


def _read(p, stub, ctx=None):
    receipt = R.Receipt()
    corpus = R.read(ctx or _ctx(), "POSTN", POSTN, reader=stub, receipt=receipt, path=p)
    return corpus, receipt


def _ledger(corpus):
    return {d.id: d.disposition for d in corpus.ledger}


def test_verified_spans_become_statements_with_code_computed_rung_and_modality(tmp_path):
    corpus, receipt = _read(_corpus(tmp_path), Stub([MARK, LACK]))
    by = {s.cell_type: s for s in corpus.statements}
    mark, lack = by["Fibroblasts"], by["Myoepithelial"]
    assert ABSTRACT[mark.start:mark.end] == mark.span
    assert (mark.stance, mark.rung, mark.species) == ("supports", "W0*", "HUMAN_STATED")
    assert mark.modality_match == "SAME_PLATFORM"      # Xenium named in Methods
    assert lack.stance == "argues_against" and lack.rung == "W0*"
    assert _ledger(corpus)["pmid:1"] == "READ:2:0"
    assert receipt.calls == 2 and "claude" in receipt.text and receipt.rejected == 0
    assert load(tmp_path / "corpus.sqlite").statements == corpus.statements


def test_a_span_not_verbatim_or_not_naming_its_cell_type_is_rejected(tmp_path):
    bad = {**MARK, "span": "POSTN marks fibroblasts in human breast tissue."}
    elsewhere = {**MARK, "cell_type": "Myoepithelial"}     # span names fibroblasts only
    corpus, receipt = _read(_corpus(tmp_path), Stub([bad, elsewhere]))
    assert corpus.statements == () and receipt.rejected == 2


def test_a_quote_typed_with_plain_spaces_binds_to_a_source_with_thin_or_no_break_ones(tmp_path):
    from dataclasses import replace

    p = _corpus(tmp_path)
    src = ABSTRACT.replace("human breast", "human breast").replace("cancer tissue",
                                                                       "cancer\xa0tissue")
    save(replace(load(p), records=tuple(replace(r, abstract=src) for r in load(p).records)), p)
    corpus, receipt = _read(p, Stub([MARK], setting={"tissue": "human breast cancer tissue"}))
    (st,) = corpus.statements
    assert receipt.rejected == 0 and st.span == src[st.start:st.end] and " " in st.span
    assert [s.quote for s in corpus.settings if s.field == "tissue"] == ["human breast "
                                                                         "cancer\xa0tissue"]


def test_relation_gates_are_code_not_model():
    rx = R.matcher(POSTN)
    assert R.relation("POSTN signaling in fibroblasts", "MARKS", "", rx) == "STATE_ASSOCIATED"
    assert R.relation("POSTN in fibroblasts", "NOT_EXPRESSED_IN", "", rx) is None
    assert R.relation("POSTN in fibroblasts", "MARKS", "", rx) == "EXPRESSED_IN"
    assert R.relation("POSTN marker of fibroblasts", "MARKS", "", rx) == "MARKS"
    span = "POSTN was higher in fibroblasts than in basal cells"
    assert R.relation(span, "ENRICHED_VS", "luminal cells", rx) == "EXPRESSED_IN"
    assert R.relation(span, "ENRICHED_VS", "basal cells", rx) == "ENRICHED_VS"


def test_p1_keeps_only_verbatim_quotes_and_a_biofluid_specimen_closes_the_paper(tmp_path):
    stub = Stub(setting={"tissue": "human lung tissue", "specimen": "Serum was collected."})
    corpus, receipt = _read(_corpus(tmp_path, full=FULL + "\nSerum was collected."), stub)
    assert [s.field for s in corpus.settings] == ["specimen"] and receipt.rejected == 1
    assert stub.calls == ["P1"] and _ledger(corpus)["pmid:1"] == "NON_TISSUE"


def test_a_paper_with_tissue_as_well_as_a_biofluid_is_read_not_closed(tmp_path):
    stub = Stub([MARK], setting={"tissue": "human breast cancer tissue",
                                 "specimen": "Serum was collected."})
    corpus, _ = _read(_corpus(tmp_path, full=FULL + "\nSerum was collected."), stub)
    assert stub.calls[:2] == ["P1", "P2"] and _ledger(corpus)["pmid:1"] == "READ:1:0"
    assert [s.rung for s in corpus.statements] == ["W0*"]     # its spans keep their rung


def test_headline_counts_species_stated_papers_at_the_study_modality(tmp_path):
    base, _ = _read(_corpus(tmp_path), Stub([MARK]))
    st = base.statements[0]
    recs = tuple(SourceRecord(uri=f"pmid:{i}", title="t", abstract=ABSTRACT, via="Europe PMC",
                              year=2026) for i in (2, 3))
    other = Statement(**{**{f: getattr(st, f) for f in st.__slots__}, "uri": "pmid:2",
                         "modality_match": "BULK"})
    unstated = Statement(**{**{f: getattr(st, f) for f in st.__slots__}, "uri": "pmid:3",
                            "flags": ("SPECIES_UNSTATED",)})
    corpus = Corpus(records=(*base.records, *recs), queries=base.queries,
                    statements=(st, other, unstated))
    assert cards.card(corpus, "POSTN", "Fibroblasts").headline.startswith(
        "1 in-window paper(s) at the study modality (this assay), 2 in window in all")


def test_unaccounted_sentences_are_re_asked_once_then_count_as_unextracted(tmp_path):
    stub = Stub([MARK], account=False)
    corpus, _ = _read(_corpus(tmp_path), stub)
    assert stub.calls == ["P1", "P2", "P2"]
    row = next(q for q in corpus.queries if q.channel == "read")
    assert row.status is QueryStatus.DEGRADED and row.retrieved < row.total
    assert _ledger(corpus)["pmid:1"].endswith(f":{row.total - row.retrieved}")
    ks = key_state(corpus, "POSTN", "Myoepithelial")
    assert "incomplete search" in ks.reads_as and not re.search(r"\bthin\b", ks.reads_as)


def test_a_failed_model_call_is_a_gap_not_an_empty_read_and_resumes(tmp_path):
    p = _corpus(tmp_path)
    corpus, _ = _read(p, Stub(fail=True))
    assert key_state(corpus, "POSTN", "Myoepithelial").gaps
    corpus, _ = _read(p, Stub([MARK]))                  # resume retries the failed paper
    assert len(corpus.statements) == 1
    assert not key_state(corpus, "POSTN", "Myoepithelial").gaps


def test_an_exhausted_passage_budget_closes_the_rest_unread_and_stays_spent(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "BUDGET_PASSAGES", 0)
    p, stub = _corpus(tmp_path), Stub([MARK])
    corpus, _ = _read(p, stub)
    assert stub.calls == [] and _ledger(corpus)["pmid:1"] == "NOT_READ_BUDGET"
    assert "incomplete search" in key_state(corpus, "POSTN", "Myoepithelial").reads_as
    monkeypatch.setattr(R, "BUDGET_PASSAGES", 400)
    _read(p, stub)                                      # a resume does not spend it again
    assert stub.calls == []


def test_a_quoted_exclusion_closes_the_paper_without_p2(tmp_path):
    stub = Stub(setting={"exclude": {"reason": "other tissue", "quote": "Tumour size varied."}})
    corpus, _ = _read(_corpus(tmp_path), stub)
    assert stub.calls == ["P1"]
    assert _ledger(corpus)["pmid:1"].startswith("P1_EXCLUDED:")


def test_an_exclusion_with_an_invented_quote_does_not_close_the_paper(tmp_path):
    stub = Stub(setting={"exclude": {"reason": "mouse only", "quote": "All mice were bred."}})
    corpus, _ = _read(_corpus(tmp_path), stub)
    assert stub.calls == ["P1", "P2"] and _ledger(corpus)["pmid:1"].startswith("READ:")


def test_release_paper_renders_apart_and_is_never_counted(tmp_path):
    corpus, _ = _read(_corpus(tmp_path), Stub([MARK]),
                      ctx=_ctx(release_pmid="1", release_status="confirmed"))
    assert _ledger(corpus)["pmid:1"] == "SAME_DATASET_READ"
    c = cards.card(corpus, "POSTN", "Fibroblasts", "Xenium")
    assert c.headline.startswith("0 in-window paper(s)")
    assert "own papers (not counted)" in c.headline
    assert "This dataset's own papers:" in c.lines() and c.window == () == c.consensus
    assert cards.note(c, corpus)["pmid"] is None        # never the note's citation


def test_card_shows_consensus_and_dissent_and_labels_wider_rungs(tmp_path):
    corpus, _ = _read(_corpus(tmp_path), Stub([MARK, LACK]))
    wide = Statement(uri="pmid:1", gene="POSTN", cell_type="Myoepithelial", tissue="breast",
                     predicate="EXPRESSED_IN", stance="qualifies", span=LACK["span"],
                     start=ABSTRACT.index(LACK["span"]), end=len(ABSTRACT), section="abstract",
                     rung="W4")
    corpus = Corpus(records=corpus.records, queries=corpus.queries,
                    statements=(*corpus.statements, wide))
    lines = cards.card(corpus, "POSTN", "Myoepithelial").lines()
    assert "In window, dissent:" in lines and "Widened to W4 other species:" in lines
    assert cards.card(corpus, "POSTN", "Fibroblasts").headline.startswith("1 in-window")


def test_absence_reads_thin_only_when_complete_and_never_as_novelty(tmp_path):
    p = _corpus(tmp_path)
    corpus = load(p)
    texts = [cards.card(corpus, "POSTN", "Fibroblasts").headline,      # complete, nothing held
             cards.card(corpus, "ACTA2", "Fibroblasts").headline]      # never queried
    assert "incomplete search" in texts[0] and "cannot settle" in texts[1]     # no SATURATED stop
    assert not any(re.search(r"\bthin\b", t) for t in texts)
    failed, _ = _read(p, Stub(fail=True))
    texts.append(cards.card(failed, "POSTN", "Fibroblasts").headline)
    assert "incomplete search" in texts[2]
    assert not any(re.search(r"novel|not previously reported", t, re.I) for t in texts)


def _verify(ctx, abstract, span, *, full="", more=()):
    rec = SourceRecord(uri="pmid:1", title="t", abstract=abstract, full_text=full, via="X")
    return R.verified(ctx, Corpus(records=(rec, *more)), rec, "POSTN", R.matcher(POSTN), span,
                      "Fibroblasts", "MARKS", "", "abstract", 0, None, ("breast",), "stub")


def test_disease_matches_the_disease_blocks_mesh_entry_terms(monkeypatch):
    from panoscope.corpus import build as B

    monkeypatch.setattr(B, "_context", lambda s: _ctx(disease_state="breast cancer"))
    records = {"D": ["Breast Neoplasms", "Breast Cancer", "Breast Carcinoma"],     # the heading
               "P": ["Nuclear Receptor Coactivator 3", "Amplified in Breast Cancer-1 Protein"]}

    def get(url, params, **_):          # live shape: a bare phrase hits proteins, not the heading
        if "esearch" in url:
            return {"esearchresult": {"idlist": ["D"] if "[mh]" in params["term"] else ["P"]}}
        ids = params["id"].split(",")
        return {"result": {"uids": ids, **{u: {"ds_meshterms": records[u]} for u in ids}}}

    monkeypatch.setattr(B.vocab, "_get", get)
    ctx = B.context_for(None)
    span = "POSTN marks fibroblasts in human breast carcinoma samples."
    assert _verify(ctx, span, span).rung == "W0"
    assert _verify(_ctx(disease_state="breast cancer"), span, span).rung == "W2"
    protein = "POSTN marks fibroblasts with nuclear receptor coactivator 3 in human breast."
    assert _verify(ctx, protein, protein).rung != "W0"        # a protein is not the disease


def test_a_paper_whose_methods_name_this_dataset_is_same_dataset():
    span = "POSTN marks fibroblasts in human breast tissue samples."
    full = "## Methods\nWe reanalysed the public Xenium data (GSE243280).\n## Results\nx"
    release = SourceRecord(uri="pmid:9", title="release", via="X",
                           full_text="## Data availability\nDeposited under GSE243280.")
    assert "SAME_DATASET" in _verify(_ctx(release_pmid="9"), span, span, full=full,
                                     more=(release,)).flags
    assert "SAME_DATASET" in _verify(_ctx(dataset_keys=("GSE243280",)), span, span,
                                     full=full).flags
    assert "SAME_DATASET" not in _verify(_ctx(release_pmid="9"), span, span, full=full).flags


def test_a_cell_named_only_as_a_place_near_the_expression_does_not_bind():
    near = "POSTN was high in areas of fibroblast clusters in human breast tissue."
    st = _verify(_ctx(), near, near)
    assert st.cell_type == "" and "AMBIGUOUS_CELL" in st.flags and st.rung == "W1"
    span = "POSTN marks fibroblasts in human breast tissue samples."
    assert _verify(_ctx(), span, span).cell_type == "Fibroblasts"


def test_no_model_key_refuses_the_build(monkeypatch):
    from panoscope.agent import loop

    monkeypatch.setattr(loop, "has_model_key", lambda: False)
    with pytest.raises(CorpusUnavailable, match="no model key"):
        R.model_reader(R.Receipt())


# --------------------------------------------------------------------------- #
# The corpus as the notes, reconciliation and report source
# --------------------------------------------------------------------------- #
def _demo_corpus():
    """A corpus saved where the demo study's corpus lives (the temp reference dir)."""
    from panoscope.agent import config as cfg
    from panoscope.pipeline import v1_verdicts

    v = v1_verdicts.verdict_for(cfg.CLUSTER_ORDER[0])
    gene, ct = v.evidence[0].gene, v.cell_type
    a, b = f"{gene} marks {ct} in human breast tissue.", f"{gene} was absent from {ct} cells."
    rec = SourceRecord(uri="pmid:7", title="t", abstract=f"{a} {b}", via="Europe PMC",
                       year=2025)
    sts = (Statement("pmid:7", gene, ct, "breast", "MARKS", "supports", a, 0, len(a),
                     "abstract", rung="W0*"),
           Statement("pmid:7", gene, ct, "breast", "NOT_EXPRESSED_IN", "argues_against", b,
                     len(a) + 1, len(a) + 1 + len(b), "abstract", rung="W0*"))
    corpus = Corpus(records=(rec,), statements=sts,
                    queries=(QueryRun("q", "C1", gene, ct, total=1, retrieved=1),),
                    version=CorpusVersion(cfg.DATASET_ID, (), (), "stub", "h", built_at="B"))
    save(corpus, corpus_path(cfg.DATASET_ID))
    return gene, ct


def test_gene_notes_come_from_cards_when_a_corpus_exists(tmp_tree, monkeypatch):
    from panoscope.agent import config as cfg
    from panoscope.agent import loop as agent_loop
    from panoscope.pipeline.stages import notes as notes_stage

    gene, _ = _demo_corpus()
    monkeypatch.setattr(agent_loop, "chat", lambda *a, **k: pytest.fail("no model call"))
    got = notes_stage.run_gene_notes(cfg.DATASET_ID, root=tmp_tree)
    note = got[cfg.CLUSTER_ORDER[0]][gene]
    assert note["pmid"] == "7" and note["dissent"] == 1 and note["verify"] is False
    assert "Dissent:" in note["summary"] and "glm_coef" in note


def test_reconcile_derives_agree_and_dissent_from_statements_at_read_time():
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.agent.types import Note, ScopeRef, Tension

    gene, ct = _demo_corpus()
    note = Note(id="n", claim="x", scope="dataset", scope_ref=ScopeRef(cfg.DATASET_ID, None),
                basis="paper", status="firm", subject_cell_type=ct, subject_markers=(gene,),
                tension=Tension((), (), True, "", ""), author="", created_at="",
                trigger="override", supersedes=None)
    t = memory.reconcile(note, lambda q: pytest.fail("no live search with a corpus"))
    assert [c.stance for c in t.agree] == ["agree"]
    assert [c.stance for c in t.dissent] == ["dissent"]
    assert t.looked_up_at == "B"


def test_report_reads_the_corpus_version_keyed_by_study():
    from panoscope.agent import config as cfg
    from panoscope.ui import report

    _demo_corpus()
    status = report._provenance(cfg.DATASET_ID)["corpus_status"]
    assert status.startswith("B") and "2 statement(s)" in status and "version" in status


def test_celltype_notes_come_from_cards_when_a_corpus_exists(tmp_tree, monkeypatch):
    from panoscope.agent import config as cfg
    from panoscope.agent import loop as agent_loop
    from panoscope.pipeline.stages import notes as notes_stage

    _demo_corpus()
    monkeypatch.setattr(agent_loop, "chat", lambda *a, **k: pytest.fail("no model call"))
    got = notes_stage.run_celltype_notes(cfg.DATASET_ID, root=tmp_tree)
    first = got[cfg.CLUSTER_ORDER[0]]
    assert first["pmid"] == "7" and first["dissent"] == 1 and first["verify"] is False
    others = [n for n in got.values() if n["cell_type"] != first["cell_type"]]
    assert others and all("cannot settle" in n["summary"] for n in others)


def test_an_uncalled_cluster_gets_no_corpus_note(tmp_tree, monkeypatch):
    """A built corpus has no key for "not yet called"; a note would say it was not built."""
    from dataclasses import replace

    from panoscope.agent import config as cfg
    from panoscope.data.study import NOT_YET_CALLED
    from panoscope.pipeline import v1_verdicts
    from panoscope.pipeline.stages import notes as notes_stage

    _demo_corpus()
    first, last = cfg.CLUSTER_ORDER[0], cfg.CLUSTER_ORDER[-1]
    real = v1_verdicts.verdict_for
    monkeypatch.setattr(v1_verdicts, "verdict_for", lambda c: replace(
        real(c), cell_type=NOT_YET_CALLED) if c == last else real(c))
    for run in (notes_stage.run_celltype_notes, notes_stage.run_gene_notes):
        got = run(cfg.DATASET_ID, root=tmp_tree)
        assert first in got and last not in got


def test_reconcile_stamps_a_lookup_only_when_the_search_was_complete():
    from panoscope.agent import config as cfg
    from panoscope.agent import memory
    from panoscope.agent.types import Note, ScopeRef, Tension

    rows = (QueryRun("q", "C1", "ACTA2", "Fibroblasts", total=0),
            QueryRun("hgnc", "vocab:hgnc", "ACTA2", total=1, retrieved=1),
            QueryRun("census", "census", "ACTA2", total=0))
    note = Note(id="n", claim="x", scope="dataset", scope_ref=ScopeRef(cfg.DATASET_ID, None),
                basis="paper", status="firm", subject_cell_type="Fibroblasts",
                subject_markers=("ACTA2",), tension=Tension((), (), True, "", ""), author="",
                created_at="", trigger="override", supersedes=None)
    stamp = []
    for extra in ((), (QueryRun("q2", "C2", "ACTA2", "Fibroblasts",
                                status=QueryStatus.FAILED),)):
        save(Corpus(queries=(*rows, *extra), version=CorpusVersion(
            cfg.DATASET_ID, (), (), "stub", "h", built_at="B")), corpus_path(cfg.DATASET_ID))
        stamp.append(memory.reconcile(note).looked_up_at)
    assert stamp == ["B", ""]       # complete and empty is thin; a gap is no lookup at all
