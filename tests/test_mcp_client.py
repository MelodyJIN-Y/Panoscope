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

from agent.mcp_client import (
    PubMedMCP,
    _article_to_citation,
    _format_authors,
    _journal_of,
    _summary_to_citation,
    _year_of,
    get_mcp_client,
    is_available,
    reset_mcp_client,
)

# The `live` marker is intentionally opt-in. pytest.ini does not use
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
    assert client.health() is False
    assert client.last_error is not None


def test_verify_rejects_non_numeric_without_touching_server():
    client = PubMedMCP()
    # Non-digit ids fail closed before any call is attempted.
    assert client.verify_pmid("not-a-pmid") is False
    assert client.verify_pmid("") is False


def test_search_empty_query_short_circuits():
    client = PubMedMCP()
    assert client.search_articles("   ") == []


def test_module_is_available_false_before_any_start():
    reset_mcp_client()
    try:
        assert is_available() is False
    finally:
        reset_mcp_client()


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

        # 4) is_available() reflects the live session.
        assert is_available() is True

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
