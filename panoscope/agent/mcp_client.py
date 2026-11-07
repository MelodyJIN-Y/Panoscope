"""Persistent PubMed MCP stdio client (the live-literature spine of the app).

Panoscope's confident floor requires every citation to resolve to a *real*
PubMed record fetched through an actual connector. This module owns that
connector: it spawns the same stdio server ``.mcp.json`` declares
(``npx -y @cyanheads/pubmed-mcp-server@latest``), keeps one warm MCP session
alive on a dedicated background asyncio loop, and exposes plain synchronous
methods so the rest of the (sync, Streamlit) codebase never touches asyncio.

Design (see docs/archive/BLUEPRINT.md §5):

* One daemon thread runs a private asyncio event loop for the process lifetime.
* On that loop we open ``stdio_client(...) -> ClientSession``, call
  ``initialize()`` once, and keep the session open.
* Sync callers submit coroutines with ``run_coroutine_threadsafe(...).result(timeout)``
  so a hung server can never block a caller past ``DEFAULT_TIMEOUT_S``.
* **Graceful fallback (never raise):** if the server fails to start, or a call
  errors or times out, the sync methods return an empty result. ``search_articles``
  / ``fetch_articles`` return ``[]``; ``verify_pmid`` returns ``False``. Every
  method also records the failure so callers (via ``client.available``) can decide
  to fall back to a frozen citation cache. The demo never breaks on the network.
* **Two response shapes.** The server answers with either ``structuredContent``
  JSON or a Markdown report in its ``content`` text blocks (what
  @cyanheads/pubmed-mcp-server 2.10.x emits). Both are parsed; structured wins
  when present. The text parsers are deliberately literal: they read only the
  server's own labelled fields and refuse — returning nothing — on any shape
  they do not recognize, because a guessed PMID is worse than no PMID.
* **Empty is not ambiguous.** ``[]`` with ``last_error is None`` means the server
  genuinely returned zero records. ``[]`` with ``last_error`` set means the
  lookup failed (transport, timeout, or a shape we could not parse). Callers MUST
  keep the two apart: the first is thin literature, the second is a broken
  lookup, and reporting the second as the first is a claim about the world we
  never made.

The returned dicts are Citation-shaped and flat:
``{"pmid", "title", "authors", "year", "journal"}`` (``fetch_articles`` adds
``"abstract"`` and ``"url"``). Nothing here writes a PMID from memory — every
value comes from a live server response.
"""

from __future__ import annotations

import asyncio
import os
import re
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Optional

# python-dotenv: load NCBI_* + creds from .env so the spawned server authenticates.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    def load_dotenv(*_a: Any, **_k: Any) -> bool:  # type: ignore[misc]
        return False

# The python `mcp` SDK. Guarded so importing this module never hard-crashes the
# app on a machine without the SDK — ``available`` will simply stay False.
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    _MCP_IMPORTED = True
except Exception:  # pragma: no cover - environment without mcp SDK
    ClientSession = None  # type: ignore[assignment]
    StdioServerParameters = None  # type: ignore[assignment]
    stdio_client = None  # type: ignore[assignment]
    _MCP_IMPORTED = False


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
SERVER_COMMAND = "npx"
SERVER_ARGS = ("-y", "@cyanheads/pubmed-mcp-server@latest")

# Tool names on @cyanheads/pubmed-mcp-server (verified live this session).
TOOL_SEARCH = "pubmed_search_articles"
TOOL_FETCH = "pubmed_fetch_articles"

DEFAULT_TIMEOUT_S = 8.0
# The very first call also pays npx cold-start + server boot + initialize().
# Give startup a longer leash than steady-state calls so a cold demo does not
# spuriously time out, while still bounding it hard.
STARTUP_TIMEOUT_S = 45.0

PUBMED_URL_TMPL = "https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


# --------------------------------------------------------------------------- #
# Result-parsing helpers (pure) — normalize server payloads to flat dicts
# --------------------------------------------------------------------------- #
def _structured(result: Any) -> dict[str, Any]:
    """Return the tool's structuredContent as a dict, or {} if absent."""
    sc = getattr(result, "structuredContent", None)
    return sc if isinstance(sc, dict) else {}


def _text(result: Any) -> str:
    """Join every text block of a tool result (the server sends 2+) into one string."""
    out: list[str] = []
    for block in getattr(result, "content", None) or ():
        txt = getattr(block, "text", None)
        if isinstance(txt, str):
            out.append(txt)
    return "\n".join(out)


# The Markdown report the server emits. Every pattern is anchored on a label the
# server wrote itself — never a bare run of digits, which could be a year, a
# result count ("3 of 17"), a volume, an ISSN or a PMID inside a URL.
_SEARCH_HEADER = "## PubMed Search Results"
_FETCH_HEADER = "## PubMed Articles"
_RE_RETURNED = re.compile(r"^\*\*Returned:\*\*\s*(\d+)\s+of\s+\d+", re.M)
_RE_PMIDS_LINE = re.compile(r"^\*\*PMIDs:\*\*[ \t]*([\d,\s]+?)[ \t]*$", re.M)
_RE_N_ARTICLES = re.compile(r"^\*\*Articles Returned:\*\*\s*(\d+)", re.M)
_RE_PMID_FIELD = re.compile(r"^\*\*PMID:\*\*[ \t]*(\d+)[ \t]*$", re.M)
_RE_AUTHOR_LINE = re.compile(r"^-\s+(.+?)\s+\(([^)]*)\)(?:\s+\[aff[^\]]*\])?\s*$", re.M)
_RE_JOURNAL_LINE = re.compile(r"^\*\*Journal:\*\*[ \t]*(.+)$", re.M)
_RE_YEAR = re.compile(r"(?<!\d)(19|20)\d{2}(?!\d)")


def _search_pmids_from_text(text: str) -> Optional[list[str]]:
    """PMIDs out of a Markdown search report, or None if the shape is not ours.

    Trusts exactly two of the server's own lines: the ``**Returned:** n of m``
    count and the ``**PMIDs:**`` list. The count is a cross-check, not a source —
    if it disagrees with the ids we read, we parsed something we do not
    understand and return None (the caller then reports a failed lookup) rather
    than surface a half-read list. ``[]`` means the server reported zero hits.
    """
    if _SEARCH_HEADER not in text:
        return None
    m_count = _RE_RETURNED.search(text)
    if m_count is None:
        return None
    m_line = _RE_PMIDS_LINE.search(text)
    pmids = (
        [p for p in re.split(r"[,\s]+", m_line.group(1)) if p.isdigit()]
        if m_line is not None
        else []
    )
    if len(pmids) != int(m_count.group(1)):
        return None
    return pmids


def _authors_from_text_block(block: str) -> str:
    """Flatten the ``- Zehuai Guo (Z) [aff 0]`` bullets into a citation string.

    Only the ``**Authors (n):**`` section is read, so the affiliation bullets
    that follow it can never be mistaken for people.
    """
    _, sep, tail = block.partition("**Authors")
    if not sep:
        return ""
    section = tail.split("\n**", 1)[0]
    people: list[dict[str, Any]] = []
    for name, initials in _RE_AUTHOR_LINE.findall(section):
        words = name.split()
        if not words:
            continue
        # ponytail: last whitespace-token is the surname; "van der Berg" reads as
        # "Berg". Swap in a real name parser only if the display is wrong often.
        people.append({"lastName": words[-1], "initials": initials.strip()})
    return _format_authors(people)


def _article_from_text_block(block: str, pmid: str) -> dict[str, Any]:
    """One ``### <title>`` block of a Markdown fetch report -> a Citation dict.

    Every field is read off a label the server wrote. A field that is not there
    stays empty (or year 0) — this never fills a gap from anywhere but these
    bytes.
    """
    title = block.splitlines()[0].strip() if block else ""
    journal, year = "", 0
    m_journal = _RE_JOURNAL_LINE.search(block)
    if m_journal is not None:
        line = m_journal.group(1)
        journal = line.split(",")[0].strip()
        m_year = _RE_YEAR.search(line)
        if m_year is not None:
            year = int(m_year.group(0))
    abstract = ""
    _, sep, tail = block.partition("#### Abstract")
    if sep:
        # The abstract runs to the next bolded field (e.g. "**Keywords:**").
        abstract = tail.split("\n**", 1)[0].strip()
    return {
        "pmid": pmid,
        "title": title,
        "authors": _authors_from_text_block(block),
        "year": year,
        "journal": journal,
        "abstract": abstract,
        "url": PUBMED_URL_TMPL.format(pmid=pmid),
    }


def _articles_from_text(text: str) -> Optional[list[dict[str, Any]]]:
    """Articles out of a Markdown fetch report, or None if the shape is not ours.

    Same discipline as :func:`_search_pmids_from_text`: a block we cannot key to
    an explicit ``**PMID:**`` field, or a block count that disagrees with the
    server's own ``**Articles Returned:**``, voids the whole response. ``[]``
    means the server resolved none of the requested ids (the bogus-PMID case).
    """
    if _FETCH_HEADER not in text:
        return None
    m_count = _RE_N_ARTICLES.search(text)
    if m_count is None:
        return None
    out: list[dict[str, Any]] = []
    for block in text.split("\n### ")[1:]:
        m_pmid = _RE_PMID_FIELD.search(block)
        if m_pmid is None:
            return None
        out.append(_article_from_text_block(block, m_pmid.group(1)))
    if len(out) != int(m_count.group(1)):
        return None
    return out


def _format_authors(authors: Any, limit: int = 8) -> str:
    """Flatten the server's author objects into a compact citation string.

    Server shape: ``[{"lastName","firstName","initials",...}, ...]``.
    Produces e.g. ``"Guo Z, Li Z, Chen M, et al."``.
    """
    if not isinstance(authors, list):
        return ""
    names: list[str] = []
    for a in authors:
        if not isinstance(a, dict):
            continue
        last = str(a.get("lastName") or "").strip()
        initials = str(a.get("initials") or "").strip()
        if not initials:
            first = str(a.get("firstName") or "").strip()
            initials = first[:1]
        if last and initials:
            names.append(f"{last} {initials}")
        elif last:
            names.append(last)
    if not names:
        return ""
    if len(names) > limit:
        return ", ".join(names[:limit]) + ", et al."
    return ", ".join(names)


def _year_of(article: dict[str, Any]) -> int:
    """Pull a 4-digit publication year out of an article record, else 0."""
    ji = article.get("journalInfo")
    if isinstance(ji, dict):
        pd = ji.get("publicationDate")
        if isinstance(pd, dict):
            y = pd.get("year")
            try:
                return int(str(y)[:4])
            except (TypeError, ValueError):
                pass
    # Fallbacks for other possible shapes.
    for key in ("pubYear", "year"):
        v = article.get(key)
        try:
            return int(str(v)[:4])
        except (TypeError, ValueError):
            continue
    return 0


def _journal_of(article: dict[str, Any]) -> str:
    ji = article.get("journalInfo")
    if isinstance(ji, dict):
        for key in ("title", "isoAbbreviation"):
            v = ji.get(key)
            if v:
                return str(v)
    v = article.get("journal")
    return str(v) if v else ""


def _article_to_citation(article: dict[str, Any]) -> dict[str, Any]:
    """Normalize one fetched article record to a flat Citation-like dict."""
    pmid = str(article.get("pmid") or "").strip()
    return {
        "pmid": pmid,
        "title": str(article.get("title") or "").strip(),
        "authors": _format_authors(article.get("authors")),
        "year": _year_of(article),
        "journal": _journal_of(article),
        "abstract": str(article.get("abstractText") or article.get("abstract") or "").strip(),
        "url": PUBMED_URL_TMPL.format(pmid=pmid) if pmid else "",
    }


def _summary_to_citation(pmid: str, summary: Any) -> dict[str, Any]:
    """Build a Citation-like dict from a search summary (may be sparse)."""
    s = summary if isinstance(summary, dict) else {}
    year = 0
    for key in ("year", "pubYear", "pubdate", "epubdate"):
        v = s.get(key)
        if v:
            try:
                year = int(str(v)[:4])
                break
            except (TypeError, ValueError):
                continue
    authors = s.get("authors")
    authors_str = _format_authors(authors) if isinstance(authors, list) else str(s.get("authors") or "")
    return {
        "pmid": str(pmid).strip(),
        "title": str(s.get("title") or "").strip(),
        "authors": authors_str,
        "year": year,
        "journal": str(s.get("journal") or s.get("source") or s.get("fulljournalname") or "").strip(),
    }


# --------------------------------------------------------------------------- #
# The persistent client
# --------------------------------------------------------------------------- #
class PubMedMCP:
    """Warm, thread-safe PubMed MCP client.

    A single instance owns one background thread + asyncio loop + MCP session.
    All public methods are synchronous and *never raise* on server/network
    failure — they return an empty result and set ``self.last_error`` instead.
    """

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[Any] = None
        self._ready = threading.Event()
        self._started = False
        self._start_lock = threading.Lock()
        # AsyncExitStack manages the stdio_client + ClientSession context lifetimes.
        self._stack: Optional[Any] = None
        self.available: bool = False
        self.last_error: Optional[str] = None

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> bool:
        """Spawn the server + open the session. Idempotent. Returns availability.

        Blocks up to ``STARTUP_TIMEOUT_S`` for the session to initialize. On any
        failure returns False and leaves ``self.available == False`` — callers
        fall back rather than crash.
        """
        with self._start_lock:
            if self._started:
                return self.available
            self._started = True

            if not _MCP_IMPORTED:
                self.last_error = "mcp SDK not importable"
                self.available = False
                return False

            load_dotenv()  # populate NCBI_ADMIN_EMAIL / NCBI_API_KEY into os.environ
            missing = [k for k in ("NCBI_ADMIN_EMAIL", "NCBI_API_KEY") if not os.environ.get(k)]
            # The server can still boot without creds (lower rate limits), so a
            # missing key is recorded but not fatal — we still try to start.
            if missing:
                self.last_error = f"missing env (continuing): {', '.join(missing)}"

            self._thread = threading.Thread(
                target=self._run_loop, name="pubmed-mcp-loop", daemon=True
            )
            self._thread.start()

            # Wait for _bootstrap to signal ready (success or failure both set it).
            if not self._ready.wait(timeout=STARTUP_TIMEOUT_S):
                self.last_error = "server startup timed out"
                self.available = False
                return False
            return self.available

    def _run_loop(self) -> None:
        """Thread target: own a private event loop for this client's lifetime."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._bootstrap())
            if self.available:
                loop.run_forever()
        finally:
            try:
                loop.run_until_complete(self._teardown())
            except Exception:  # pragma: no cover - best-effort cleanup
                pass
            try:
                loop.close()
            except Exception:  # pragma: no cover
                pass

    async def _bootstrap(self) -> None:
        """On the loop thread: spawn server, open session, initialize once."""
        from contextlib import AsyncExitStack

        try:
            env = dict(os.environ)  # pass creds + PATH through to npx/node
            params = StdioServerParameters(
                command=SERVER_COMMAND,
                args=list(SERVER_ARGS),
                env=env,
            )
            self._stack = AsyncExitStack()
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._session = session
            self.available = True
            self.last_error = None
        except Exception as exc:  # noqa: BLE001 - graceful fallback, never raise up
            self.available = False
            self.last_error = f"startup failed: {exc!r}"
            if self._stack is not None:
                try:
                    await self._stack.aclose()
                except Exception:  # pragma: no cover
                    pass
                self._stack = None
        finally:
            # Unblock start() regardless of outcome.
            self._ready.set()

    async def _teardown(self) -> None:  # pragma: no cover - process-exit path
        if self._stack is not None:
            try:
                await self._stack.aclose()
            finally:
                self._stack = None
        self._session = None

    def close(self) -> None:
        """Stop the loop and tear down the session. Safe to call multiple times."""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self.available = False

    # -- core call path ---------------------------------------------------- #
    def _call_tool(self, name: str, args: dict[str, Any], timeout: float) -> Optional[Any]:
        """Run one MCP tool call on the background loop. Returns the raw result,
        or None on any failure/timeout (and records ``last_error``).
        """
        if not self.available or self._session is None or self._loop is None:
            self.last_error = self.last_error or "client not available"
            return None
        try:
            coro = self._session.call_tool(name, args)
            fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"dispatch failed: {exc!r}"
            return None
        try:
            result = fut.result(timeout=timeout)
        except FutureTimeoutError:
            fut.cancel()
            self.last_error = f"{name} timed out after {timeout}s"
            return None
        except Exception as exc:  # noqa: BLE001 - server-side / protocol error
            self.last_error = f"{name} failed: {exc!r}"
            return None
        if getattr(result, "isError", False):
            self.last_error = f"{name} returned isError"
            return None
        # A clean response clears any earlier failure, so ``last_error`` always
        # describes THIS call: an empty result plus a None error is a real zero.
        self.last_error = None
        return result

    # -- public API -------------------------------------------------------- #
    def search_articles(
        self, query: str, max_results: int = 5, *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> list[dict[str, Any]]:
        """Search PubMed. Returns a list of flat Citation-like dicts (possibly
        empty). Every returned PMID came from the live server, never memory.

        An empty list means zero hits ONLY when ``last_error`` is None; otherwise
        the lookup failed and the caller must say so rather than report thin
        literature. The Markdown shape carries PMIDs and no per-article
        metadata, so those records come back with an empty title/journal and
        year 0 — ``fetch_articles`` is how you get the rest. Nothing is filled in
        from anywhere but the server's own bytes.
        """
        if not query or not query.strip():
            self.last_error = "empty query"
            return []
        args = {"query": query.strip(), "maxResults": max(1, int(max_results))}
        result = self._call_tool(TOOL_SEARCH, args, timeout)
        if result is None:
            return []

        data = _structured(result)
        pmids = data.get("pmids")
        if isinstance(pmids, list):
            summaries = data.get("summaries")
            summaries = summaries if isinstance(summaries, list) else []
            out: list[dict[str, Any]] = []
            for i, pmid in enumerate(pmids):
                pmid = str(pmid).strip()
                if not pmid:
                    continue
                summary = summaries[i] if i < len(summaries) else None
                out.append(_summary_to_citation(pmid, summary))
            return out

        text_pmids = _search_pmids_from_text(_text(result))
        if text_pmids is None:
            self.last_error = f"{TOOL_SEARCH}: unrecognized response shape"
            return []
        return [_summary_to_citation(p, None) for p in text_pmids]

    def fetch_articles(
        self, pmids: list[str], *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> list[dict[str, Any]]:
        """Fetch full metadata for PMIDs. Returns flat Citation-like dicts
        (with abstract + url), possibly empty. Only PMIDs the server actually
        resolved are returned.

        As with :meth:`search_articles`, an empty list is "the server resolved
        none of these" only when ``last_error`` is None; set, it means the fetch
        failed and nothing can be concluded about the ids.
        """
        clean = [str(p).strip() for p in (pmids or []) if str(p).strip().isdigit()]
        if not clean:
            self.last_error = "no numeric PMIDs requested"
            return []
        args = {"pmids": clean, "includeMesh": False}
        result = self._call_tool(TOOL_FETCH, args, timeout)
        if result is None:
            return []
        data = _structured(result)
        articles = data.get("articles")
        if isinstance(articles, list):
            return [_article_to_citation(a) for a in articles if isinstance(a, dict)]
        text_articles = _articles_from_text(_text(result))
        if text_articles is None:
            self.last_error = f"{TOOL_FETCH}: unrecognized response shape"
            return []
        return text_articles

    def verify_pmid(self, pmid: str, *, timeout: float = DEFAULT_TIMEOUT_S) -> bool:
        """True iff ``pmid`` resolves to a real PubMed record via a live fetch.

        A bogus PMID comes back in ``unavailablePmids`` (empty ``articles``),
        so we simply check the requested id is among the returned records.
        On any server/network failure returns False (fail-closed: an unverified
        citation must not be treated as real).
        """
        pid = str(pmid).strip()
        if not pid.isdigit():
            return False
        articles = self.fetch_articles([pid], timeout=timeout)
        return any(a.get("pmid") == pid for a in articles)


# --------------------------------------------------------------------------- #
# Module-level singleton + convenience API
# --------------------------------------------------------------------------- #
_CLIENT: Optional[PubMedMCP] = None
_CLIENT_LOCK = threading.Lock()


def get_mcp_client() -> PubMedMCP:
    """Return the process-wide warm client, starting it on first use.

    Safe to call from ``@st.cache_resource``. Never raises: a failed start
    yields a client with ``available == False`` whose methods return empties.
    """
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is None:
            client = PubMedMCP()
            client.start()
            _CLIENT = client
        return _CLIENT


def reset_mcp_client() -> None:
    """Tear down the singleton (tests / explicit shutdown). Best-effort."""
    global _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            try:
                _CLIENT.close()
            except Exception:  # pragma: no cover
                pass
            _CLIENT = None
