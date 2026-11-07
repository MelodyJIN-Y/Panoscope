"""Vocabulary lookups from real sources: HGNC REST (CC0), MeSH and PubMed counts (E-utilities).

harness v2 §3. Identifies as ``tool=panoscope``; ``NCBI_API_KEY`` / ``NCBI_EMAIL`` are
read from the environment only. Rate-limited (3 req/s, 10 with a key) with backoff.
A failed lookup is logged as a FAILED query-log row, never as an empty answer.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Final

from panoscope.corpus.store import QueryRun, QueryStatus
from panoscope.corpus.window import Term, gene_terms
from panoscope.reference.sources import SourceUnavailable, cache_dir

EUTILS: Final = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
HGNC: Final = "https://rest.genenames.org/fetch/"
_HEADERS: Final = {"User-Agent": "panoscope (corpus vocabulary)", "Accept": "application/json"}
_lock = threading.Lock()
_last = [0.0]


def _get(url: str, params: dict[str, str], *, retries: int = 3) -> dict[str, Any]:
    """GET a JSON endpoint under the shared rate limit; SourceUnavailable after retries."""
    if url.startswith(EUTILS):
        params = {**params, "tool": "panoscope", "retmode": "json"}
        for env, key in (("NCBI_API_KEY", "api_key"), ("NCBI_EMAIL", "email")):
            if os.getenv(env):
                params[key] = os.environ[env]
    gap = 0.1 if os.getenv("NCBI_API_KEY") else 0.34
    full = url + ("?" + urllib.parse.urlencode(params) if params else "")
    err: Exception | None = None
    for attempt in range(retries):
        with _lock:
            time.sleep(max(0.0, _last[0] + gap - time.monotonic()))
            _last[0] = time.monotonic()
        try:
            req = urllib.request.Request(full, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            err = exc
            if exc.code not in (429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            err = exc
        time.sleep(2 ** attempt)
    raise SourceUnavailable(f"{url}: {err}")


def hgnc(symbol: str) -> dict[str, Any] | None:
    """The HGNC record for a symbol (approved, else previous), cached; None if HGNC has none."""
    path: Path = cache_dir("hgnc", "rest", redistributable=True) / f"{symbol}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")) or None
    doc = None
    for field in ("symbol", "prev_symbol"):
        docs = _get(f"{HGNC}{field}/{urllib.parse.quote(symbol)}", {}) \
            .get("response", {}).get("docs", [])
        if len(docs) == 1:
            doc = docs[0]
            break
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc or {}, sort_keys=True), encoding="utf-8")
    return doc


def mesh_terms(term: str) -> tuple[str, ...]:
    """MeSH entry terms of the descriptors whose entry terms include ``term``."""
    ids = _get(EUTILS + "esearch.fcgi", {"db": "mesh", "term": f'"{term}"'}) \
        .get("esearchresult", {}).get("idlist", [])
    if not ids:
        return ()
    res = _get(EUTILS + "esummary.fcgi", {"db": "mesh", "id": ",".join(ids[:5])}) \
        .get("result", {})
    out: list[str] = []
    for uid in res.get("uids", []):
        entry = [str(t) for t in res.get(uid, {}).get("ds_meshterms", [])]
        if term.casefold() in (t.casefold() for t in entry):
            out += entry
    return tuple(dict.fromkeys(out))


def count(query: str) -> int:
    """PubMed hit count for an exact query string."""
    return int(_get(EUTILS + "esearch.fcgi", {"db": "pubmed", "term": query,
                                             "rettype": "count"})["esearchresult"]["count"])


def vocabulary(symbol: str, tissue_terms: tuple[str, ...]
               ) -> tuple[tuple[Term, ...], tuple[QueryRun, ...]]:
    """A gene's aliases plus one census row per alias (alias[tiab] AND tissue block).

    A failed HGNC or MeSH lookup and a failed count are FAILED rows, which keep the
    gene's search incomplete (never thin) downstream.
    """
    runs: list[QueryRun] = []

    def failed(query: str, channel: str) -> None:
        runs.append(QueryRun(query=query, channel=channel, gene=symbol,
                             status=QueryStatus.FAILED))

    try:
        h = hgnc(symbol)
        runs.append(QueryRun(query=f"{HGNC}symbol/{symbol}", channel="vocab:hgnc",
                             gene=symbol, total=int(h is not None),
                             retrieved=int(h is not None)))
    except SourceUnavailable:
        h = None
        failed(f"{HGNC}symbol/{symbol}", "vocab:hgnc")
    mesh: tuple[str, ...] = ()
    try:
        mesh = mesh_terms(symbol)
        runs.append(QueryRun(query=f'"{symbol}"', channel="vocab:mesh", gene=symbol,
                             total=len(mesh), retrieved=len(mesh)))
    except SourceUnavailable:
        failed(f'"{symbol}"', "vocab:mesh")
    terms = gene_terms(symbol, h, mesh)
    block = " OR ".join(f'"{t}"[tiab]' for t in tissue_terms)
    for t in terms:
        q = f'"{t.text}"[tiab] AND ({block})'
        try:
            runs.append(QueryRun(query=q, channel="census", gene=symbol, total=count(q)))
        except SourceUnavailable:
            failed(q, "census")
    return terms, tuple(runs)
