"""Tests for the persistent PubMed MCP client (agent/mcp_client.py).

Two flavors:

* Pure/offline tests exercise the normalization helpers and the graceful-failure
  contract with **no** server spawn — they run everywhere, including CI with no
  network.
* One ``@pytest.mark.live`` test spawns the real ``@cyanheads/pubmed-mcp-server``
  via npx and asserts the confident-floor guarantee end to end: a real search
  returns real PMIDs, ``verify_pmid`` confirms a real one, and a bogus PMID is
  rejected. It skips cleanly (never hangs) if npx or the network is unavailable.

Run just the live check with:  pytest tests/test_mcp_client.py -m live -q
"""

from __future__ import annotations

import shutil
import warnings

import pytest

from panoscope.agent.mcp_client import (
    PubMedMCP,
    _article_to_citation,
    _format_authors,
    _journal_of,
    _summary_to_citation,
    _year_of,
    get_mcp_client,
    reset_mcp_client,
)

# The `live` marker is intentionally opt-in. The pytest config (now in
# pyproject.toml's [tool.pytest.ini_options]) does not use
# --strict-markers, so this only silences the benign "unknown mark" warning
# and keeps `-m live` / `-m "not live"` selection working.
warnings.filterwarnings("ignore", category=pytest.PytestUnknownMarkWarning)

# A well-known real PubMed record (Lumican multi-omics review), used to prove
# the real-PMID path resolves without depending on which PMIDs a search returns.
KNOWN_REAL_PMID = "37692065"
BOGUS_PMID = "99999999999"
LIVE_QUERY = "lumican LUM fibroblast marker"


# --------------------------------------------------------------------------- #
# Pure helpers — normalization of server payloads (no server, always run)
# --------------------------------------------------------------------------- #
def test_format_authors_compact_and_truncated():
    authors = [
        {"lastName": "Guo", "firstName": "Zehuai", "initials": "Z"},
        {"lastName": "Li", "firstName": "Zeyun", "initials": "Z"},
    ]
    assert _format_authors(authors) == "Guo Z, Li Z"

    many = [{"lastName": f"A{i}", "initials": "X"} for i in range(10)]
    out = _format_authors(many, limit=8)
    assert out.endswith(", et al.")
    assert out.count(",") == 8  # 8 names + the "et al." tail comma


def test_format_authors_handles_missing_initials_and_junk():
    assert _format_authors([{"lastName": "Solo", "firstName": "Han"}]) == "Solo H"
    assert _format_authors(None) == ""
    assert _format_authors([{}, "garbage", 3]) == ""


def test_year_and_journal_extraction_from_journalinfo():
    article = {
        "pmid": "37692065",
        "journalInfo": {
            "title": "Frontiers in molecular biosciences",
            "isoAbbreviation": "Front Mol Biosci",
            "publicationDate": {"year": "2023"},
        },
    }
    assert _year_of(article) == 2023
    assert _journal_of(article) == "Frontiers in molecular biosciences"


def test_year_of_returns_zero_when_unparseable():
    assert _year_of({}) == 0
    assert _year_of({"journalInfo": {"publicationDate": {"year": "n/a"}}}) == 0


def test_article_to_citation_shape():
    article = {
        "pmid": "37692065",
        "title": "A title.",
        "abstractText": "Some abstract.",
        "authors": [{"lastName": "Guo", "initials": "Z"}],
        "journalInfo": {"title": "Front Mol Biosci", "publicationDate": {"year": "2023"}},
    }
    cit = _article_to_citation(article)
    assert cit["pmid"] == "37692065"
    assert cit["title"] == "A title."
    assert cit["authors"] == "Guo Z"
    assert cit["year"] == 2023
    assert cit["journal"] == "Front Mol Biosci"
    assert cit["abstract"] == "Some abstract."
    assert cit["url"] == "https://pubmed.ncbi.nlm.nih.gov/37692065/"


def test_summary_to_citation_is_flat_and_keyed():
    cit = _summary_to_citation("12345", None)
    assert set(cit) == {"pmid", "title", "authors", "year", "journal"}
    assert cit["pmid"] == "12345"
    assert cit["year"] == 0


# --------------------------------------------------------------------------- #
# Graceful-failure contract — a client that never started must never raise
# --------------------------------------------------------------------------- #
def test_unstarted_client_returns_empty_never_raises():
    client = PubMedMCP()  # not started
    assert client.available is False
    assert client.search_articles("anything") == []
    assert client.fetch_articles(["37692065"]) == []
    assert client.verify_pmid("37692065") is False
    assert client.last_error is not None


def test_verify_rejects_non_numeric_without_touching_server():
    client = PubMedMCP()
    # Non-digit ids fail closed before any call is attempted.
    assert client.verify_pmid("not-a-pmid") is False
    assert client.verify_pmid("") is False


def test_search_empty_query_short_circuits():
    client = PubMedMCP()
    assert client.search_articles("   ") == []


# --------------------------------------------------------------------------- #
# Live PubMed path — real server, real PMIDs (opt-in, skips if unavailable)
# --------------------------------------------------------------------------- #
def _npx_available() -> bool:
    return shutil.which("npx") is not None


@pytest.mark.live
def test_live_pubmed_search_verify_and_bogus_rejection():
    """Confident-floor end-to-end: real search -> real PMID -> verify True;
    bogus PMID -> verify False. Skips cleanly if npx/network is unavailable.
    """
    if not _npx_available():
        pytest.skip("npx not on PATH; cannot spawn the PubMed MCP server")

    reset_mcp_client()
    client = get_mcp_client()
    try:
        if not client.available:
            pytest.skip(f"PubMed MCP server did not start: {client.last_error}")

        # 1) A real search returns >= 1 real PMID.
        hits = client.search_articles(LIVE_QUERY, max_results=3)
        if not hits:
            pytest.skip(f"live search returned nothing (network?): {client.last_error}")
        assert len(hits) >= 1
        pmids = [h["pmid"] for h in hits]
        assert all(p.isdigit() for p in pmids), pmids

        # 2) verify_pmid on a real returned PMID is True.
        assert client.verify_pmid(pmids[0]) is True

        # 3) A definitely-bogus PMID verifies False (fabrication is caught).
        assert client.verify_pmid(BOGUS_PMID) is False

        # 4) The singleton reflects the live session.
        assert client.available is True

        # Expose the real PMIDs so the test log proves the live path worked.
        print(f"\n[live] search PMIDs: {pmids}")
    finally:
        reset_mcp_client()


@pytest.mark.live
def test_live_fetch_returns_real_metadata_for_known_pmid():
    """Fetching a known real PMID returns populated Citation-like metadata."""
    if not _npx_available():
        pytest.skip("npx not on PATH; cannot spawn the PubMed MCP server")

    reset_mcp_client()
    client = get_mcp_client()
    try:
        if not client.available:
            pytest.skip(f"PubMed MCP server did not start: {client.last_error}")

        records = client.fetch_articles([KNOWN_REAL_PMID])
        if not records:
            pytest.skip(f"live fetch returned nothing (network?): {client.last_error}")

        rec = records[0]
        assert rec["pmid"] == KNOWN_REAL_PMID
        assert rec["title"]  # non-empty real title
        assert rec["url"] == f"https://pubmed.ncbi.nlm.nih.gov/{KNOWN_REAL_PMID}/"
        print(f"\n[live] fetched: {rec['title'][:80]!r} ({rec['year']})")
    finally:
        reset_mcp_client()


# --------------------------------------------------------------------------- #
# Markdown response shape — the bytes @cyanheads/pubmed-mcp-server 2.10.x sends
#
# Captured verbatim from a live call; no network is touched by these tests. The
# server returns NO structuredContent for either tool, so the text path is the
# only thing standing between a real lookup and three silently discarded PMIDs.
# --------------------------------------------------------------------------- #
SEARCH_TEXT = (
    "## PubMed Search Results\n"
    "**Query:** lumican LUM fibroblast marker\n"
    "**Returned:** 3 of 17 | **Offset:** 0\n"
    "**Search URL:** https://pubmed.ncbi.nlm.nih.gov/?term=lumican%20LUM%20fibroblast%20marker\n"
    "\n"
    "**PMIDs:** 41553378, 37692065, 39613390"
)
# The server sends a second text block; both must be tolerated.
SEARCH_TEXT_TAIL = (
    "\n\n**Effective Query:** lumican LUM fibroblast marker\n**Applied Filters:** none"
)
SEARCH_PMIDS = ["41553378", "37692065", "39613390"]

FETCH_TEXT = (
    "## PubMed Articles\n"
    "**Articles Returned:** 1\n"
    "\n"
    "### Multi-omics analysis reveals the prognostic and tumor micro-environmental "
    "value of lumican in multiple cancer types.\n"
    "\n"
    "**Authors (9):**\n"
    "- Zehuai Guo (Z) [aff 0]\n"
    "- Zeyun Li (Z) [aff 1]\n"
    "\n"
    "**Affiliations:**\n"
    "- [0] Department of Internal Medicine, Cancer Hospital of Shantou University (SUMC)\n"
    "\n"
    "**Journal:** Frontiers in molecular biosciences, (Front Mol Biosci), 2023, **10**, "
    "1158747, pii: 1158747, ISSN 2296-889X\n"
    "**Record Type:** journal-article\n"
    "**PMID:** 37692065\n"
    "**DOI:** 10.3389/fmolb.2023.1158747\n"
    "**PubMed:** https://pubmed.ncbi.nlm.nih.gov/37692065/\n"
    "**Article Dates:** Electronic 2023-08-24\n"
    "\n"
    "#### Abstract\n"
    "Background: Lumican (LUM), a proteoglycan of the extracellular matrix.\n"
    "\n"
    "**Keywords:** Lumican, TME"
)
FETCH_ZERO_TEXT = "## PubMed Articles\n**Articles Returned:** 0\n**Unavailable PMIDs:** 99999999999"


class _FakeBlock:
    """One ``content`` text block of an MCP tool result."""

    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResult:
    """A CallToolResult stand-in: text blocks plus optional structuredContent."""

    def __init__(self, *texts: str, structured=None) -> None:
        self.content = [_FakeBlock(t) for t in texts]
        self.structuredContent = structured
        self.isError = False


def _client_returning(result) -> PubMedMCP:
    """A client whose one tool call returns ``result`` (the server is never spawned)."""
    client = PubMedMCP()
    client._call_tool = lambda *_a, **_k: result  # type: ignore[assignment]
    client.last_error = None
    return client


def test_search_text_shape_yields_the_real_pmids():
    """The Markdown shape must surface the server's PMIDs, not discard them."""
    client = _client_returning(_FakeResult(SEARCH_TEXT, SEARCH_TEXT_TAIL))
    hits = client.search_articles("lumican LUM fibroblast marker", max_results=3)
    assert [h["pmid"] for h in hits] == SEARCH_PMIDS
    assert client.last_error is None
    # This shape carries no per-article metadata; we leave it empty rather than
    # invent any. literature_fetch is how a title is obtained.
    assert all(h["title"] == "" and h["year"] == 0 for h in hits)


def test_search_structured_shape_still_wins():
    """The old structured path is intact and takes precedence over the text."""
    structured = {"pmids": ["11111111"], "summaries": [{"title": "T", "journal": "J", "year": "2020"}]}
    client = _client_returning(_FakeResult(SEARCH_TEXT, structured=structured))
    hits = client.search_articles("q")
    assert [h["pmid"] for h in hits] == ["11111111"]
    assert hits[0]["title"] == "T"
    assert hits[0]["year"] == 2020


def test_search_text_count_mismatch_is_refused_not_guessed():
    """If the server's own count disagrees with the ids we read, surface nothing."""
    tampered = SEARCH_TEXT.replace("**PMIDs:** 41553378, 37692065, 39613390", "**PMIDs:** 41553378")
    client = _client_returning(_FakeResult(tampered))
    assert client.search_articles("q") == []
    assert client.last_error is not None  # a failed lookup, not thin literature


def test_search_unrecognized_text_is_an_error_not_an_empty_result():
    client = _client_returning(_FakeResult("Service temporarily unavailable (503)."))
    assert client.search_articles("q") == []
    assert client.last_error is not None


def test_search_text_never_reads_a_count_or_a_url_as_a_pmid():
    """No digits outside the server's own **PMIDs:** line may become a PMID."""
    zero = (
        "## PubMed Search Results\n"
        "**Query:** zzz\n"
        "**Returned:** 0 of 0 | **Offset:** 0\n"
        "**Search URL:** https://pubmed.ncbi.nlm.nih.gov/?term=12345678\n"
    )
    client = _client_returning(_FakeResult(zero))
    assert client.search_articles("zzz") == []
    # A search the server ran and answered with nothing is NOT a failure.
    assert client.last_error is None


def test_fetch_text_shape_yields_real_metadata():
    client = _client_returning(_FakeResult(FETCH_TEXT))
    recs = client.fetch_articles(["37692065"])
    assert len(recs) == 1
    rec = recs[0]
    assert rec["pmid"] == "37692065"
    assert rec["title"].startswith("Multi-omics analysis reveals")
    assert rec["authors"] == "Guo Z, Li Z"  # affiliation bullets are not people
    assert rec["year"] == 2023
    assert rec["journal"] == "Frontiers in molecular biosciences"
    assert rec["abstract"].startswith("Background: Lumican")
    assert "Keywords" not in rec["abstract"]
    assert rec["url"] == "https://pubmed.ncbi.nlm.nih.gov/37692065/"


def test_verify_pmid_resolves_through_the_text_shape():
    """The grounding gate's verifier works again (it fails every PMID otherwise)."""
    client = _client_returning(_FakeResult(FETCH_TEXT))
    assert client.verify_pmid("37692065") is True
    assert client.verify_pmid("41553378") is False  # not among the returned records


def test_fetch_zero_articles_is_a_real_answer_not_a_failure():
    client = _client_returning(_FakeResult(FETCH_ZERO_TEXT))
    assert client.fetch_articles(["99999999999"]) == []
    assert client.last_error is None


def test_fetch_unrecognized_text_is_an_error():
    client = _client_returning(_FakeResult("### A block with no PMID field\n"))
    assert client.fetch_articles(["37692065"]) == []
    assert client.last_error is not None
