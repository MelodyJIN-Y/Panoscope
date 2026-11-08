"""The corpus: three states, a live-only write path, and a licence gate that bites.

What these tests defend, in order of how badly it goes wrong when it breaks:

1. **No fabricated citation.** There is no construction path to a ``SourceRecord``
   that does not start at a connector response, and a hollow one is refused.
2. **Looked-and-found-nothing is not could-not-look.** Both produce an empty corpus;
   they must never produce the same answer. Commit ``5b7ab08`` is that collapse.
3. **Nothing is bundled.** No corpus data in the tree, and a non-redistributable
   source cannot resolve a path inside the checkout even when asked to.

**Offline, and that is asserted, not assumed.** ``test_no_socket_is_opened_...``
replaces ``socket.socket`` with a tripwire for the whole offline surface -- fetch
against a fake, save, load, resolve, search, path resolution. Nothing in this file
reaches a network.

**About the fixtures.** Two kinds, deliberately spelled differently:

* :data:`REAL_RECORDS` are **three genuine PubMed metadata records**, retrieved live
  through ``agent/mcp_client.py`` on 2026-11-02 with the query "POSTN
  cancer-associated fibroblast breast" (5 hits, top 3 fetched). Title, authors, year
  and journal exactly as the server returned them -- including ``pmid:39930931``'s
  empty author string, which is what that response actually carried and is left wrong
  rather than tidied. **Abstracts are stripped**: they are the publisher's, not the
  NLM's, and this file is committed.
* Everything named ``FAKE_*`` is **synthesised for this test and is not evidence of
  anything**. The ids in them are not PubMed ids and are not citable. They exist to
  drive branches that need a shape, never to stand in for a lookup.
"""

from __future__ import annotations

import dataclasses
import inspect
import socket
import typing
from pathlib import Path

import pytest

from panoscope.corpus import retrieve as R
from panoscope.corpus import store as S
from panoscope.corpus.retrieve import Hit, Retrieval, Support, search
from panoscope.corpus.store import (
    CITATION_SCHEMES, Corpus, CorpusUnavailable, HollowRecord, Miss, Resolution,
    Resolved, SourceRecord, corpus_path, fetch, load, resolve, save,
)
from panoscope.reference.sources import NotRedistributable, cache_root
from panoscope.types.claim import SOURCE_SCHEMES
from panoscope.types.trace import RULES, Traced

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# REAL records. Retrieved live 2026-11-02; metadata only, abstracts stripped.
# --------------------------------------------------------------------------- #
REAL_RECORDS: tuple[SourceRecord, ...] = (
    SourceRecord(
        uri="pmid:38631570",
        title=("Exploring the association of POSTN⁺ cancer-associated "
               "fibroblasts with triple-negative breast cancer"),
        authors="Lin S, Zhou M, Cheng L, Shuai Z, Zhao M, Jie R, Wan Q, Peng F, et al.",
        year=2024,
        journal="International journal of biological macromolecules",
        via="PubMedMCP",
        retrieved="2026-11-02",
    ),
    SourceRecord(
        uri="pmid:39930931",
        title=("CDK16⁺ Luminal Progenitor Cell-Like Tumor Cells Interacted with "
               "POSTN⁺ Cancer-Associated Fibroblasts"),
        authors="",          # the server returned none; not filled in from anywhere
        year=2025,
        journal="Small methods",
        via="PubMedMCP",
        retrieved="2026-11-02",
    ),
    SourceRecord(
        uri="pmid:38246584",
        title=("Single-Cell RNA Sequencing Reveals Dysregulated POSTN+WNT5A+ "
               "Fibroblast Subclusters in Prurigo Nodularis"),
        authors=("Patel JR, Joel MZ, Lee KK, Kambala A, Cornman H, Oladipo O, "
                 "Taylor M, Imo BU, et al."),
        year=2024,
        journal="The Journal of investigative dermatology",
        via="PubMedMCP",
        retrieved="2026-11-02",
    ),
)

REAL_CORPUS = Corpus(records=REAL_RECORDS, built="2026-11-02")


# --------------------------------------------------------------------------- #
# The fake connector. SYNTHETIC. Nothing it returns is a lookup.
# --------------------------------------------------------------------------- #
class FakeConnector:
    """A stand-in for ``agent.mcp_client.PubMedMCP``. Returns nothing real.

    It exists to drive the three outcomes of :func:`fetch` without a network, and it
    models the one contract that matters: ``last_error is None`` means the answer was
    complete, whatever its length.
    """

    def __init__(self, articles: list[dict] | None = None,
                 error: str | None = None) -> None:
        self.articles = articles or []
        self.last_error = error
        self.asked: list[str] = []

    def fetch_articles(self, pmids: list[str]) -> list[dict]:
        self.asked.extend(pmids)
        return list(self.articles)


# A SYNTHETIC article payload in the mcp_client Citation shape. "10000001" is not a
# PubMed id and this record is not a citation.
FAKE_ARTICLE = {
    "pmid": "10000001",
    "title": "A synthesised record used only to drive a branch in this test",
    "authors": "Nobody N",
    "year": 2020,
    "journal": "Not A Journal",
    "abstract": "Synthesised abstract text; licence-bearing field, not a real one.",
}


# --------------------------------------------------------------------------- #
# 1. Never fabricate
# --------------------------------------------------------------------------- #
def test_a_record_with_no_title_is_refused():
    """The half-read response. Stored, it renders as a citation and exports as one."""
    with pytest.raises(HollowRecord):
        SourceRecord(uri="pmid:38631570", title="")
    with pytest.raises(HollowRecord):
        SourceRecord(uri="pmid:38631570", title="   ")


def test_a_uri_that_names_no_paper_is_refused():
    """``note:`` and ``file:`` are citable Claim sources but are not papers; a source
    record keyed to one would resolve a citation to a local note."""
    for bad in ("note:n1", "file:sha256:abc", "pmid:", "38631570", "ftp:x"):
        with pytest.raises(HollowRecord):
            SourceRecord(uri=bad, title="Something")


def test_fetch_is_the_only_way_a_record_enters_the_corpus():
    """Static: the package constructs a ``SourceRecord`` in exactly two places -- the
    connector response path and the deserialiser. A convenience constructor is how a
    plausible stub gets minted by a caller in a hurry."""
    src = Path(S.__file__).read_text(encoding="utf-8")
    callers = [ln.strip() for ln in src.splitlines() if "SourceRecord(" in ln
               and "class SourceRecord" not in ln and not ln.strip().startswith("#")]
    assert len(callers) == 2, (
        "SourceRecord is constructed in more places than fetch() and load(): "
        + "; ".join(callers)
    )


def test_a_record_keyed_to_a_real_pmid_but_carrying_no_title_is_a_failed_lookup():
    """Not a miss. The paper plainly exists -- what failed is the parse, and calling
    that 'no such record' would publish a parse bug as a fact about the literature."""
    hollow = dict(FAKE_ARTICLE, title="")
    out = fetch(["pmid:10000001"], connector=FakeConnector([hollow]))
    assert out.value.records == ()
    assert out.value.misses == (), "a parse failure must not become thin literature"
    assert [d.rule for d in out.decisions] == [
        "corpus:lookup_failed", "corpus:fetched",
    ]


# --------------------------------------------------------------------------- #
# 2. Three states, and the two that look alike
# --------------------------------------------------------------------------- #
def test_resolved_carries_the_record():
    out = resolve(REAL_CORPUS, "pmid:38631570")
    assert out.value.state is Resolution.RESOLVED
    assert out.value.record is REAL_RECORDS[0]
    assert out.value.is_thin is False
    assert "International journal of biological macromolecules" in out.value.reads_as
    assert [d.rule for d in out.decisions] == ["corpus:resolved"]


def test_a_completed_empty_lookup_is_thin_literature():
    """The connector answered with no error and resolved nothing: the paper is not
    there. A ``Miss`` is written, and that is the one thing that may read as thin."""
    client = FakeConnector(articles=[], error=None)
    built = fetch(["pmid:10000001"], connector=client).value
    assert built.records == ()
    assert built.misses == (Miss(uri="pmid:10000001", looked_with="FakeConnector"),)

    out = resolve(built, "pmid:10000001")
    assert out.value.state is Resolution.NOT_FOUND
    assert out.value.is_thin is True
    assert "thin" in out.value.reads_as
    assert [d.rule for d in out.decisions] == ["corpus:not_found"]


def test_a_failed_lookup_writes_nothing_and_is_not_thin_literature():
    """THE test. Same empty corpus by every byte-level measure; different answer.

    ``5b7ab08``: ``ok=True, n_results=0`` reported a lookup that never completed as a
    finding about the world. Here the failure leaves no record AND no miss, so the
    URI falls through to NOT_LOOKED_UP and the sentence changes with it.
    """
    client = FakeConnector(articles=[], error="pubmed_fetch_articles timed out")
    built = fetch(["pmid:10000001"], connector=client).value
    assert built.records == () and built.misses == ()

    out = resolve(built, "pmid:10000001")
    assert out.value.state is Resolution.NOT_LOOKED_UP
    assert out.value.is_thin is False, "a timeout is not evidence the paper is absent"
    assert "says nothing" in out.value.reads_as
    assert [d.rule for d in out.decisions] == ["corpus:not_looked_up"]


def test_the_two_empty_corpora_answer_differently():
    """Proof the distinction is carried by the data and not by a live object.

    Both corpora hold zero records. The three states are three different sentences.
    """
    thin = fetch(["pmid:10000001"],
                 connector=FakeConnector([], error=None)).value
    unlooked = fetch(["pmid:10000001"],
                     connector=FakeConnector([], error="transport closed")).value
    assert len(thin.records) == len(unlooked.records) == 0

    states = {
        resolve(REAL_CORPUS, "pmid:38631570").value.state,
        resolve(thin, "pmid:10000001").value.state,
        resolve(unlooked, "pmid:10000001").value.state,
    }
    assert states == {Resolution.RESOLVED, Resolution.NOT_FOUND,
                      Resolution.NOT_LOOKED_UP}

    sentences = {
        resolve(REAL_CORPUS, "pmid:38631570").value.reads_as,
        resolve(thin, "pmid:10000001").value.reads_as,
        resolve(unlooked, "pmid:10000001").value.reads_as,
    }
    assert len(sentences) == 3, "two of the three states render identically"


def test_the_two_empty_corpora_still_disagree_after_a_round_trip(tmp_path):
    """Where a lazier design loses it: serialise, and the misses go with the file."""
    thin = fetch(["pmid:10000001"], connector=FakeConnector([], error=None)).value
    unlooked = fetch(["pmid:10000001"],
                     connector=FakeConnector([], error="boom")).value
    a = load(save(thin, tmp_path / "a" / "sources.json"))
    b = load(save(unlooked, tmp_path / "b" / "sources.json"))
    assert resolve(a, "pmid:10000001").value.state is Resolution.NOT_FOUND
    assert resolve(b, "pmid:10000001").value.state is Resolution.NOT_LOOKED_UP


def test_a_scheme_with_no_connector_is_a_failed_lookup_not_a_miss():
    """``doi:`` is citable and unresolvable today. "We cannot look this up" must not
    render as "this paper does not exist"."""
    out = fetch(["doi:10.1093/database/baz046"], connector=FakeConnector())
    assert out.value.records == () and out.value.misses == ()
    assert [d.rule for d in out.decisions] == ["corpus:lookup_failed"]
    assert resolve(out.value, "doi:10.1093/database/baz046").value.state is (
        Resolution.NOT_LOOKED_UP
    )


def test_a_resolved_state_without_a_record_is_unrepresentable():
    """The iff. Quoting a title for a source nobody resolved cannot be built."""
    with pytest.raises(ValueError):
        Resolved(uri="pmid:38631570", state=Resolution.RESOLVED)
    with pytest.raises(ValueError):
        Resolved(uri="pmid:38631570", state=Resolution.NOT_FOUND,
                 record=REAL_RECORDS[0])


def test_an_absent_corpus_raises_rather_than_reading_as_empty(tmp_path):
    """``5b7ab08`` one layer down: a caller handed ``Corpus()`` reports silence."""
    with pytest.raises(CorpusUnavailable):
        load(tmp_path / "never-built" / "sources.json")
    bad = tmp_path / "sources.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(CorpusUnavailable):
        load(bad)
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(CorpusUnavailable):
        load(bad)


# --------------------------------------------------------------------------- #
# 3. Nothing is bundled, and the gate is what stops it
# --------------------------------------------------------------------------- #
def test_no_corpus_data_is_bundled_in_the_repo():
    """The in-suite form of §12's CI grep, for the literature half.

    A corpus file anywhere in the checkout is either a licence violation waiting to
    be pushed or a fixture somebody will mistake for evidence.
    """
    skip = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache"}
    found = [p for p in REPO.rglob(S.STORE_FILENAME)
             if not skip & set(p.relative_to(REPO).parts)]
    assert not found, f"corpus data committed into the repo: {found}"
    assert not list(REPO.glob("data/corpus/**/*.json")), (
        "data/corpus/ holds a corpus; it is fetched, never bundled"
    )


def test_a_corpus_cannot_resolve_a_path_inside_the_repository(monkeypatch):
    """The licence gate, reused from ``reference.sources`` rather than re-written.

    This is the shortcut somebody takes on a laptop -- point the cache at the
    checkout so it is "easy to inspect" -- and then commits. It raises at path
    resolution, before any byte is written.
    """
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(REPO))
    with pytest.raises(NotRedistributable):
        corpus_path("xenium_hbreast_sample1")
    monkeypatch.setenv("PANOSCOPE_REFERENCE_DIR", str(REPO / "data" / "corpus"))
    with pytest.raises(NotRedistributable):
        corpus_path("xenium_hbreast_sample1")


def test_the_default_corpus_path_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("PANOSCOPE_REFERENCE_DIR", raising=False)
    p = corpus_path("xenium_hbreast_sample1")
    assert REPO not in p.parents and p != REPO
    assert p.parent.parent == cache_root() / "corpus"
    assert p.name == S.STORE_FILENAME


def test_the_committed_fixtures_carry_no_abstract():
    """Metadata is the NLM's record; the abstract is the publisher's. This file is
    committed, so it may hold the first and never the second."""
    for rec in REAL_RECORDS:
        assert rec.abstract == ""
        assert rec.redistributable is True
    assert REAL_CORPUS.licensed_records == ()


def test_a_record_with_an_abstract_is_marked_not_redistributable():
    held = fetch(["pmid:10000001"], connector=FakeConnector([FAKE_ARTICLE])).value
    assert len(held.records) == 1
    assert held.records[0].redistributable is False
    assert held.licensed_records == held.records


# --------------------------------------------------------------------------- #
# 4. Retrieval
# --------------------------------------------------------------------------- #
def test_retrieval_finds_support_and_states_its_ranking():
    out = search(REAL_CORPUS, ["POSTN", "fibroblast"])
    assert out.value.state is Support.FOUND
    assert out.value.searched == 3
    scores = [(h.record.uri, h.score) for h in out.value.hits]
    # The declared ceiling, visible: TITLE_WEIGHT per distinct title term, no
    # stemming. Only pmid:38246584's title says "Fibroblast"; the other two say
    # "fibroblasts" and score on POSTN alone. The ponytail comment in retrieve.py
    # names this, and the day alias/stem normalisation lands these become 4, 4, 4.
    assert scores == [("pmid:38246584", 4), ("pmid:38631570", 2),
                      ("pmid:39930931", 2)]
    assert out.value.hits[0].matched == ("fibroblast", "postn")
    assert out.value.hits[1].matched == ("postn",)
    assert "not confirmation" in out.value.reads_as
    assert [d.rule for d in out.decisions] == ["corpus:support_found"]


def test_ranking_is_deterministic_and_ties_break_on_uri():
    a = search(REAL_CORPUS, ["POSTN"]).value
    b = search(REAL_CORPUS, ["postn"]).value
    assert [h.record.uri for h in a.hits] == [h.record.uri for h in b.hits]
    assert [h.record.uri for h in a.hits] == sorted(h.record.uri for h in a.hits)


def test_a_title_term_outweighs_a_body_term():
    """The one weight in the ranking, and it has to be visible on the Hit."""
    out = search(REAL_CORPUS, ["investigative"]).value
    top = out.hits[0]
    assert top.record.uri == "pmid:38246584"
    assert top.in_title == () and top.elsewhere == ("investigative",)
    assert top.score == 1
    in_title = search(REAL_CORPUS, ["prurigo"]).value.hits[0]
    assert in_title.in_title == ("prurigo",)
    assert in_title.score == R.TITLE_WEIGHT


def test_searched_and_found_nothing_is_not_the_same_as_never_built():
    """The retrieval-layer form of the same split. Two empty answers, two sentences."""
    nothing = search(REAL_CORPUS, ["ZZZQQX9"]).value
    assert nothing.state is Support.NONE_FOUND
    assert nothing.searched == 3 and nothing.hits == ()
    assert "thin" in nothing.reads_as
    assert "3 record(s)" in nothing.reads_as

    unbuilt = search(Corpus(), ["ZZZQQX9"]).value
    assert unbuilt.state is Support.NOT_BUILT
    assert unbuilt.searched == 0
    assert "no corpus has been built" in unbuilt.reads_as
    assert nothing.reads_as != unbuilt.reads_as


def test_a_corpus_of_only_misses_was_built_and_says_so():
    """Every lookup came back empty. That corpus WAS built, and its silence is the
    productive kind -- NOT_BUILT would report an answered question as unasked."""
    only_misses = Corpus(
        misses=(Miss(uri="pmid:10000001", looked_with="FakeConnector"),)
    )
    out = search(only_misses, ["POSTN"])
    assert out.value.state is Support.NONE_FOUND
    assert [d.rule for d in out.decisions] == ["corpus:no_support"]


def test_the_relevance_floor_is_applied():
    low = search(REAL_CORPUS, ["investigative"], floor=1).value
    assert low.state is Support.FOUND
    high = search(REAL_CORPUS, ["investigative"], floor=2).value
    assert high.state is Support.NONE_FOUND
    assert high.floor == 2 and "floor 2" in high.reads_as


def test_an_empty_query_finds_nothing_rather_than_everything():
    out = search(REAL_CORPUS, []).value
    assert out.state is Support.NONE_FOUND
    assert out.hits == ()


def test_hits_is_non_empty_iff_support_was_found():
    with pytest.raises(ValueError):
        Retrieval(terms=("postn",), state=Support.FOUND)
    with pytest.raises(ValueError):
        Retrieval(terms=("postn",), state=Support.NONE_FOUND,
                  hits=(Hit(record=REAL_RECORDS[0], score=2),))
    with pytest.raises(ValueError):
        Retrieval(terms=("postn",), state=Support.NOT_BUILT, searched=3)


def test_retrieval_supplies_no_value_and_no_band():
    """A score orders evidence. It must not be mistakable for a statistic about the
    measurement, so nothing in the package names one or imports one."""
    for mod in (S, R):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        for forbidden in ("glm_coef", "pearson", "Band(", "verify ="):
            assert forbidden not in src, f"{mod.__name__} mentions {forbidden}"


# --------------------------------------------------------------------------- #
# 5. The house rules
# --------------------------------------------------------------------------- #
def test_every_corpus_rule_is_registered_under_its_own_step():
    ids = {r for r in RULES if r.startswith("corpus:")}
    assert ids == {
        "corpus:resolved", "corpus:not_found", "corpus:not_looked_up",
        "corpus:lookup_failed", "corpus:fetched", "corpus:support_found",
        "corpus:no_support", "corpus:not_built",
    }
    for rid in ids:
        assert RULES[rid].step == "corpus"


def test_the_deciding_functions_return_traced():
    for fn in (resolve, fetch, search):
        ret = typing.get_type_hints(fn).get("return")
        assert typing.get_origin(ret) is Traced, f"{fn.__name__} -> {ret!r}"


def test_no_corpus_function_accepts_a_trace():
    for mod in (S, R):
        for name, fn in vars(mod).items():
            if name.startswith("_") or not inspect.isfunction(fn):
                continue
            if fn.__module__ != mod.__name__:
                continue
            hints = typing.get_type_hints(fn)
            assert "trace" not in {k.lower() for k in hints}, f"{name} takes a trace"


def test_every_dataclass_is_frozen_slotted_and_holds_no_mutable_field():
    """The shape rule. A dict/list/set field is a shared mutable the record would
    hash differently on, and a frozen wrapper around one is a lie."""
    bad: list[str] = []
    seen = 0
    for mod in (S, R):
        for name, obj in vars(mod).items():
            if not (inspect.isclass(obj) and dataclasses.is_dataclass(obj)):
                continue
            if obj.__module__ != mod.__name__:
                continue
            seen += 1
            params = obj.__dataclass_params__
            if not params.frozen or not getattr(obj, "__slots__", None):
                bad.append(f"{name}: frozen={params.frozen} "
                           f"slots={bool(getattr(obj, '__slots__', None))}")
            for f in dataclasses.fields(obj):
                hint = typing.get_type_hints(obj).get(f.name, f.type)
                origin = typing.get_origin(hint) or hint
                if origin in (dict, list, set, frozenset):
                    bad.append(f"{name}.{f.name}: {hint}")
    assert seen >= 6, f"the dataclass scan is not biting ({seen} found)"
    assert not bad, "; ".join(bad)


def test_citation_schemes_have_not_drifted_from_the_claim_contract():
    """A scheme citable by a Claim but unknown to the corpus renders as a citation
    and resolves to nothing -- which is what a fabrication looks like downstream."""
    assert CITATION_SCHEMES <= SOURCE_SCHEMES
    assert CITATION_SCHEMES == frozenset({"pmid", "pmcid", "doi"})


def test_a_corpus_never_merges_two_records_for_one_uri():
    """``claim.py``'s rule. Two connectors that both hold a PMID contribute two
    records, and whoever reads them sees the pair."""
    twice = Corpus(records=(REAL_RECORDS[0],
                            dataclasses.replace(REAL_RECORDS[0], via="Other")))
    assert len(twice.find("pmid:38631570")) == 2


def test_the_store_is_clock_free():
    """``retrieved`` and ``built`` are strings the caller minted. A now() here would
    put a timestamp into bytes that are hashed into the golden record."""
    for mod in (S, R):
        # The module docstring is dropped first: it SAYS "a now() here would put a
        # timestamp into bytes that are hashed", and a scan that tripped on the
        # explanation of the rule would be a test of its own prose.
        code = Path(mod.__file__).read_text(encoding="utf-8").split('"""', 2)[-1]
        for forbidden in ("datetime", "time.time", "now()"):
            assert forbidden not in code, f"{mod.__name__} reaches for a clock"


def test_save_round_trips_exactly(tmp_path):
    built = Corpus(records=REAL_RECORDS,
                   misses=(Miss("pmid:10000001", "FakeConnector", "2026-11-02"),),
                   built="2026-11-02")
    p = save(built, tmp_path / "sources.json")
    assert load(p) == built
    # Saving again replaces the file wholesale and reads back the same corpus.
    assert load(save(built, p)) == built
    assert load(p).spec_version == S.SPEC_VERSION


def test_no_socket_is_opened_across_the_whole_offline_surface(monkeypatch, tmp_path):
    """Offline is asserted, not assumed. Every path here, with the network removed."""

    def tripwire(*a, **k):
        raise AssertionError("the corpus opened a socket")

    monkeypatch.setattr(socket, "socket", tripwire)
    monkeypatch.setattr(socket, "create_connection", tripwire)

    built = fetch(["pmid:10000001"], connector=FakeConnector([FAKE_ARTICLE])).value
    p = save(built, tmp_path / "sources.json")
    reloaded = load(p)
    resolve(reloaded, "pmid:10000001")
    resolve(reloaded, "pmid:10000002")
    search(reloaded, ["synthesised"])
    search(REAL_CORPUS, ["POSTN"])
    corpus_path("xenium_hbreast_sample1")
