"""Persistent PubMed MCP stdio client (the live-literature spine of the app).

Panoscope's confident floor requires every citation to resolve to a *real*
PubMed record fetched through an actual connector. This module owns that
connector: it spawns the same stdio server ``.mcp.json`` declares
(``npx -y @cyanheads/pubmed-mcp-server@latest``), keeps one warm MCP session
alive on a dedicated background asyncio loop, and exposes plain synchronous
methods so the rest of the (sync, Streamlit) codebase never touches asyncio.

Design (see BLUEPRINT.md §5):

* One daemon thread runs a private asyncio event loop for the process lifetime.
* On that loop we open ``stdio_client(...) -> ClientSession``, call
  ``initialize()`` once, and keep the session open.
* Sync callers submit coroutines with ``run_coroutine_threadsafe(...).result(timeout)``
  so a hung server can never block a caller past ``DEFAULT_TIMEOUT_S``.
* **Graceful fallback (never raise):** if the server fails to start, or a call
  errors or times out, the sync methods return an empty result. ``search_articles``
  / ``fetch_articles`` return ``[]``; ``verify_pmid`` returns ``False``. Every
  method also records the failure so callers (and ``is_available()``) can decide
  to fall back to a frozen citation cache. The demo never breaks on the network.

The returned dicts are Citation-shaped and flat:
``{"pmid", "title", "authors", "year", "journal"}`` (``fetch_articles`` adds
``"abstract"`` and ``"url"``). Nothing here writes a PMID from memory — every
value comes from a live server response.
"""

from __future__ import annotations

import asyncio
import os
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
# app on a machine without the SDK — is_available() will simply report False.
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

    def __init__(self, *, startup_timeout: float = STARTUP_TIMEOUT_S) -> None:
        self._startup_timeout = startup_timeout
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

        Blocks up to ``startup_timeout`` for the session to initialize. On any
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
            if not self._ready.wait(timeout=self._startup_timeout):
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
        return result

    # -- public API -------------------------------------------------------- #
    def search_articles(
        self, query: str, max_results: int = 5, *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> list[dict[str, Any]]:
        """Search PubMed. Returns a list of flat Citation-like dicts (possibly
        empty). Every returned PMID came from the live server, never memory.
        """
        if not query or not query.strip():
            return []
        args = {"query": query.strip(), "maxResults": max(1, int(max_results))}
        result = self._call_tool(TOOL_SEARCH, args, timeout)
        if result is None:
            return []

        data = _structured(result)
        pmids = data.get("pmids")
        if not isinstance(pmids, list):
            return []
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

    def fetch_articles(
        self, pmids: list[str], *, timeout: float = DEFAULT_TIMEOUT_S
    ) -> list[dict[str, Any]]:
        """Fetch full metadata for PMIDs. Returns flat Citation-like dicts
        (with abstract + url), possibly empty. Only PMIDs the server actually
        resolved are returned.
        """
        clean = [str(p).strip() for p in (pmids or []) if str(p).strip().isdigit()]
        if not clean:
            return []
        args = {"pmids": clean, "includeMesh": False}
        result = self._call_tool(TOOL_FETCH, args, timeout)
        if result is None:
            return []
        data = _structured(result)
        articles = data.get("articles")
        if not isinstance(articles, list):
            return []
        return [_article_to_citation(a) for a in articles if isinstance(a, dict)]

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

    def health(self) -> bool:
        """Cheap liveness check: is the session up and answering a 1-result search."""
        if not self.available:
            return False
        hits = self.search_articles("CD3D T cell", max_results=1)
        return len(hits) >= 1


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


def is_available() -> bool:
    """True iff the live PubMed MCP path is up (server started + session ready).

    Callers use this to decide between a live lookup and the frozen citation
    cache. Does not itself spawn the server if it has not been requested yet;
    call ``get_mcp_client()`` first to warm it.
    """
    client = _CLIENT
    return bool(client and client.available)


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
