"""The source corpus: what a ``pmid:`` URI actually resolves to.

The store ``types/claim.py`` points at (claims carry a URI, not an abstract), which is
what makes "citations must be real and looked up live" checkable.

**Three states, never two:**

* :attr:`Resolution.RESOLVED` -- a record is held, and it came off a live connector.
* :attr:`Resolution.NOT_FOUND` -- a lookup **ran to completion** and the source does
  not exist (a :class:`Miss`). Thin literature, and a finding.
* :attr:`Resolution.NOT_LOOKED_UP` -- never looked, or the lookup failed. Says nothing.

In :func:`fetch` a connector error, timeout or unparseable response writes **nothing**,
so the URI stays ``NOT_LOOKED_UP``; only a clean empty answer produces a ``Miss``.

**Nothing in here invents a record**: :class:`SourceRecord` refuses an empty title and
every record starts at a connector response.

**Licence:** abstracts are the publisher's, so :func:`corpus_path` goes through
``reference.sources.cache_dir`` with ``redistributable=False`` and cannot resolve
inside the repository (ARCHITECTURE §10.6, §12).

Clock-free: ``retrieved`` is an ISO string the caller minted.

**Storage** is one SQLite file per study (§8.3): source records, misses, statements
bound to verbatim spans, the query log, the disposition ledger and the corpus
version. Rebuilt wholesale, so rows are JSON and the file is replaced atomically.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Iterable, Protocol

from panoscope.reference.sources import cache_dir
from panoscope.types.claim import SOURCE_SCHEMES
from panoscope.types.trace import Trace, Traced

CITATION_SCHEMES: Final[frozenset[str]] = frozenset({"pmid", "doi"})
"""The schemes that name a *paper*. A subset of ``claim.SOURCE_SCHEMES``, and the
same two ``claims.store.thin_literature`` keys off -- ``test_corpus`` asserts they
have not drifted apart, because a scheme citable in one module and not the other is a
claim that renders as a citation and resolves to nothing."""

STORE_FILENAME: Final = "corpus.sqlite"
SPEC_VERSION: Final = "0.3"

RESOLVABLE: Final[frozenset[str]] = frozenset({"pmid"})
"""Schemes a connector exists for today. ``doi:`` is citable but not resolvable: the
PubMed client takes PMIDs. That gap is reported as a FAILED lookup, never as a miss --
"we have no connector for this" must not render as "this paper does not exist"."""


class CorpusUnavailable(RuntimeError):
    """Raised when the corpus could not be read -- never built, or unreadable.

    The sibling of ``reference.sources.SourceUnavailable``, and for its reason: an
    empty corpus and an unread corpus are the same bytes and completely different
    claims. Returning an empty :class:`Corpus` here would turn "nobody has built this
    study's corpus" into "the literature says nothing about this study".
    """


class HollowRecord(ValueError):
    """Raised for a record with no title, or with a URI that names no paper.

    A stub with a real-looking PMID and nothing else is the fabrication this project
    fears most, arriving by accident: it renders as a citation, it exports, and
    nothing downstream can tell it from a record somebody actually read.
    """


class Resolution(StrEnum):
    RESOLVED = "resolved"              # a record is held
    NOT_FOUND = "not_found"            # a lookup completed; no such source -> thin
    NOT_LOOKED_UP = "not_looked_up"    # never looked, or the lookup failed -> silent


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SourceRecord:
    """One source, as a connector returned it. Flat, because its origin is flat.

    Exactly ``agent/mcp_client.py``'s Citation shape, so no field is ever filled from
    memory. ``abstract`` is the one licence-bearing field (the publisher's);
    :attr:`redistributable` says so, and the path gate does not depend on it.
    """

    uri: str
    title: str
    authors: str = ""
    year: int = 0                 # 0 means the connector did not say; never guessed
    journal: str = ""
    abstract: str = ""
    via: str = ""                 # the connector that produced it (PubMed, Europe PMC)
    retrieved: str = ""           # fetched_at: ISO string the caller minted; no clock here
    pmcid: str = ""
    doi: str = ""
    publication_type: tuple[str, ...] = ()
    licence: str = ""             # as the connector stated it; "" = not stated
    full_text: str = ""           # licence-bearing like the abstract; store only
    open_access: bool = False     # the connector said the full text is OA
    epub: str = ""                # earliest publication date as the connector gave it
    preprint_of: str = ""         # the published version's uri, when the connector links one

    def __post_init__(self) -> None:
        scheme, sep, rest = self.uri.partition(":")
        if not sep or scheme not in CITATION_SCHEMES or not rest:
            raise HollowRecord(
                f"{self.uri!r} does not name a paper; a source record's URI must use "
                f"one of {sorted(CITATION_SCHEMES)}"
            )
        if not self.title.strip():
            raise HollowRecord(
                f"{self.uri}: a record with no title is a half-read response, not a "
                "source. Store nothing rather than a stub that renders as a citation."
            )

    @property
    def redistributable(self) -> bool:
        """False once an abstract or full text is attached. Metadata alone is the NLM's."""
        return not (self.abstract.strip() or self.full_text.strip())

    @property
    def is_real(self) -> bool:
        """A connector produced it; records are only ever built from a response."""
        return bool(self.via)

    @property
    def retracted(self) -> bool:
        return any("retract" in t.casefold() for t in self.publication_type)

    def text(self, section: str) -> str:
        """The stored text a span is checked against: title, abstract or full_text."""
        if section not in SECTIONS:
            raise UnboundSpan(f"{section!r} is not a section ({sorted(SECTIONS)})")
        return getattr(self, section)

    @property
    def citation(self) -> str:
        """The one-line rendering. Built only from fields the connector supplied."""
        bits = [b for b in (self.authors, self.title, self.journal) if b]
        if self.year:
            bits.append(str(self.year))
        return ". ".join(bits) + f" [{self.uri}]"


@dataclass(frozen=True, slots=True)
class Miss:
    """A lookup that ran and found nothing. The ONLY way to record thin literature.

    ``looked_with`` names the connector that answered, so "thin" can be read as what
    it is -- one connector's complete answer -- rather than as a property of the
    world. A second connector may hold it.
    """

    uri: str
    looked_with: str
    retrieved: str = ""


@dataclass(frozen=True, slots=True)
class Resolved:
    """What the corpus says about one URI, and what that is worth.

    ``record is not None`` **iff** ``state is RESOLVED``, enforced below. Same trick
    as ``Measured``'s "statistic is None iff the state is not OBSERVED":
    quoting a title for a source nobody resolved is made structurally
    unrepresentable rather than forbidden in prose.
    """

    uri: str
    state: Resolution
    record: SourceRecord | None = None
    looked_with: str = ""

    def __post_init__(self) -> None:
        if (self.state is Resolution.RESOLVED) is not (self.record is not None):
            raise ValueError(
                f"{self.uri}: record is not None iff state is RESOLVED "
                f"(state={self.state!r}, record={'set' if self.record else 'None'})"
            )

    @property
    def is_thin(self) -> bool:
        """True for NOT_FOUND only. A failed lookup is not thin literature."""
        return self.state is Resolution.NOT_FOUND

    @property
    def reads_as(self) -> str:
        """The sentence a reviewer sees. Never "there is no such paper"."""
        if self.state is Resolution.RESOLVED and self.record is not None:
            return f"{self.uri} resolves to {self.record.citation}"
        if self.state is Resolution.NOT_FOUND:
            return (f"{self.uri} was looked up ({self.looked_with}) and no record "
                    "came back -- the literature is thin here")
        return (f"{self.uri} has not been looked up -- nothing was recorded either "
                "way, so this says nothing about whether the source exists")


SECTIONS: Final[frozenset[str]] = frozenset({"title", "abstract", "full_text"})


class UnboundSpan(ValueError):
    """A statement whose span is not verbatim at its offsets in its record's text."""


@dataclass(frozen=True, slots=True)
class Statement:
    """One model-read statement about a gene, bound to a verbatim span of one record.

    ``rung``, modality and ``species`` are computed by code (``corpus.window``), never
    by the model; ``written_by`` is the generator (model + route) that read the span.
    """

    uri: str
    gene: str
    cell_type: str              # the frame label; "" when the span names none
    tissue: str
    predicate: str
    stance: str                 # supports | argues_against | qualifies | neutral
    span: str
    start: int
    end: int
    section: str                # title | abstract | full_text
    rung: str = ""              # window.Rung value; "" = beyond the ladder
    analyte: str = "UNKNOWN"
    resolution: str = "UNKNOWN"
    modality_match: str = "UNKNOWN"
    species: str = "UNSTATED"
    flags: tuple[str, ...] = ()
    written_by: str = ""


@dataclass(frozen=True, slots=True)
class Setting:
    """One P1 setting field of a paper (species, tissue, assay ...) as a verbatim quote."""

    uri: str
    field: str
    quote: str
    start: int
    end: int
    section: str


class QueryStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    DEGRADED = "degraded"
    TRUNCATED = "truncated"     # k of N retrieved; degraded for absence
    NARROWED = "narrowed"       # over the cap, re-run narrowed; the narrowed row decides


@dataclass(frozen=True, slots=True)
class QueryRun:
    """One query-log row: the exact string sent, and what came back of how much.

    ``channel`` "census" is count-only (nothing retrieved, ``total`` is the hit count).
    Any other channel reports OK only when everything was retrieved.
    """

    query: str
    channel: str
    gene: str
    cell_type: str = ""         # "" for gene-level rows (vocabulary, census)
    total: int = 0
    retrieved: int = 0
    status: QueryStatus = QueryStatus.OK
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", QueryStatus(self.status))
        if "{" in self.query or not self.query.strip():
            raise ValueError(f"not an exact query string: {self.query!r}")
        if (self.status is QueryStatus.OK and self.channel != "census"
                and self.retrieved < self.total):
            raise ValueError(f"{self.retrieved} of {self.total} retrieved is "
                             "TRUNCATED, not OK")

    @property
    def request_hash(self) -> str:
        return hashlib.sha256(f"{self.channel}\0{self.query}".encode()).hexdigest()[:16]


DISPOSITIONS: Final[tuple[str, ...]] = (
    "DEDUP_OF:", "PREFILTER_NO_ALIAS", "RETRACTED", "REVIEW_CHASE_ONLY", "P1_EXCLUDED:",
    "NON_TISSUE", "OUT_OF_LADDER", "NOT_SCREENED_TRUNCATED", "NOT_READ_BUDGET", "READ:",
    "SAME_DATASET_READ",
)


@dataclass(frozen=True, slots=True)
class Disposition:
    """The one terminal disposition of an identifier any query returned (§6 ledger)."""

    id: str
    disposition: str

    def __post_init__(self) -> None:
        d = self.disposition
        if not any(d == p or (p.endswith(":") and d.startswith(p) and d != p)
                   for p in DISPOSITIONS):
            raise ValueError(f"{self.id}: {d!r} is not a ledger disposition")


@dataclass(frozen=True, slots=True)
class CorpusVersion:
    """What one build was made from (§8.3). ``built_at`` and ``stale_after`` are outside the content hash."""

    study: str
    conditions: tuple[tuple[str, str], ...]     # Conditions snapshot, (field, value)
    reference: tuple[tuple[str, str], ...]      # (source, release)
    generator: str                              # model + route
    query_log_hash: str
    background_hash: str = ""
    cost: str = ""                              # the build's receipt, as written
    retrieval: str = "BM25+aliases"
    embedding: str = "none"
    spec_version: str = SPEC_VERSION
    built_at: str = ""
    stale_after: str = ""                       # ISO date the header starts flagging it

    @property
    def content_hash(self) -> str:
        d = {f.name: getattr(self, f.name) for f in fields(self)
             if f.name not in ("built_at", "stale_after")}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def log_hash(queries: Iterable[QueryRun]) -> str:
    """The query log's digest, in run order, for :attr:`CorpusVersion.query_log_hash`."""
    h = hashlib.sha256()
    for q in queries:
        h.update(f"{q.request_hash}:{q.status.value}:{q.total}:{q.retrieved}\n".encode())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The corpus
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Corpus:
    """The built corpus for one study. Records and misses, both first class.

    ``misses`` is what keeps "asked, found nothing" apart from "never asked" across a
    save and reload. Never merged: two records for one PMID stay two.
    """

    records: tuple[SourceRecord, ...] = ()
    misses: tuple[Miss, ...] = ()
    built: str = ""                   # the corpus version's build stamp (§8.3)
    spec_version: str = SPEC_VERSION
    statements: tuple[Statement, ...] = ()
    queries: tuple[QueryRun, ...] = ()
    ledger: tuple[Disposition, ...] = ()
    version: CorpusVersion | None = None
    settings: tuple[Setting, ...] = ()

    def __post_init__(self) -> None:
        for st in (*self.statements, *self.settings):
            held = self.find(st.uri)
            if not any(r.text(st.section)[st.start:st.end] == _span(st) for r in held):
                raise UnboundSpan(
                    f"{st.uri} {st.section}[{st.start}:{st.end}] is not {_span(st)!r}"
                    + ("" if held else " (no record held)"))
        ids = [d.id for d in self.ledger]
        if len(ids) != len(set(ids)):
            raise ValueError("an identifier holds two dispositions in the ledger")

    @property
    def unclosed(self) -> tuple[str, ...]:
        """Ids a query returned that hold no disposition; a build check fails on any."""
        done = {d.id for d in self.ledger}
        return tuple(dict.fromkeys(i for q in self.queries for i in q.ids if i not in done))

    def find(self, uri: str) -> tuple[SourceRecord, ...]:
        # ponytail: linear scan. A per-study corpus is hundreds of records, not
        # millions; index by uri if a study ever makes that false.
        return tuple(r for r in self.records if r.uri == uri)

    @property
    def licensed_records(self) -> tuple[SourceRecord, ...]:
        """The records carrying an abstract -- i.e. the ones that may not ship."""
        return tuple(r for r in self.records if not r.redistributable)


def _span(row: Statement | Setting) -> str:
    return row.span if isinstance(row, Statement) else row.quote


def resolve(corpus: Corpus, uri: str) -> Traced[Resolved]:
    """What the corpus says about ``uri``, with the decision that settled it.

    Pure, never the network. A record wins over a miss; no record and no miss is
    ``NOT_LOOKED_UP`` (fail-closed).
    """
    t = Trace()
    held = corpus.find(uri)
    if held:
        rec = held[0]
        t.note("corpus:resolved", Resolution.RESOLVED.value,
               because=(f"{len(held)} record(s) held; via {rec.via or 'unstated'}",),
               refs=(uri,))
        return t.done(Resolved(uri=uri, state=Resolution.RESOLVED, record=rec,
                               looked_with=rec.via))

    for m in corpus.misses:
        if m.uri == uri:
            t.note("corpus:not_found", Resolution.NOT_FOUND.value,
                   because=(f"{m.looked_with} answered and held no record",),
                   refs=(uri,))
            return t.done(Resolved(uri=uri, state=Resolution.NOT_FOUND,
                                   looked_with=m.looked_with))

    t.note("corpus:not_looked_up", Resolution.NOT_LOOKED_UP.value,
           because=(f"no record and no completed lookup in a corpus of "
                    f"{len(corpus.records)} record(s), {len(corpus.misses)} miss(es)",),
           refs=(uri,))
    return t.done(Resolved(uri=uri, state=Resolution.NOT_LOOKED_UP))


# --------------------------------------------------------------------------- #
# The live path. Every record in the corpus comes through here.
# --------------------------------------------------------------------------- #
class Connector(Protocol):
    """What :func:`fetch` needs of a literature client.

    Structural, so ``agent.mcp_client.PubMedMCP`` satisfies it unedited.
    ``last_error`` is the contract: ``[]`` with it ``None`` is a completed empty
    answer, ``[]`` with it set is a lookup that did not happen.
    """

    last_error: str | None

    def fetch_articles(self, pmids: list[str]) -> list[dict[str, Any]]: ...


def _pubmed_connector() -> Connector:
    """The process-wide warm PubMed client. Imported lazily so this package is
    importable without the ``[agent]`` extra and touches no network at import."""
    from panoscope.agent.mcp_client import get_mcp_client

    return get_mcp_client()


def fetch(uris: Iterable[str], *, connector: Connector | None = None,
          retrieved: str = "", built: str = "") -> Traced[Corpus]:
    """Resolve ``uris`` against a live connector and build the corpus they produce.

    One outcome per URI:

    * the connector returned a record -> a :class:`SourceRecord`;
    * the connector answered cleanly and returned nothing -> a :class:`Miss`;
    * the connector errored, timed out, returned a shape it could not parse, or
      there is no connector for the scheme -> **nothing is written**, and the trace
      carries ``corpus:lookup_failed`` (a ``Miss`` here would fake a finding).
    """
    t = Trace()
    wanted = tuple(dict.fromkeys(u.strip() for u in uris if u and u.strip()))

    pmids: list[str] = []
    for uri in wanted:
        scheme, _, rest = uri.partition(":")
        if scheme in RESOLVABLE and rest.isdigit():
            pmids.append(rest)
            continue
        t.note("corpus:lookup_failed", uri,
               because=(f"no connector resolves {scheme or 'an empty'}: URIs "
                        f"(resolvable today: {sorted(RESOLVABLE)})",
                        "recorded as unlooked-up, NOT as absent"),
               refs=(uri,))

    if not pmids:
        return t.done(Corpus(built=built))

    client = connector if connector is not None else _pubmed_connector()
    articles = client.fetch_articles(pmids)
    error = getattr(client, "last_error", None)
    via = type(client).__name__

    if error:
        # The whole batch is unsettled. Not one Miss: nothing was learned about any
        # of these ids, and the honest corpus is the empty one plus this decision.
        t.note("corpus:lookup_failed", f"{len(pmids)} pmid(s)",
               because=(f"{via}: {error}",
                        "no record and no miss written; these URIs stay unlooked-up"),
               refs=tuple(f"pmid:{p}" for p in pmids))
        return t.done(Corpus(built=built))

    returned = {str(a.get("pmid") or "").strip(): a for a in articles
                if str(a.get("pmid") or "").strip()}
    records: list[SourceRecord] = []
    misses: list[Miss] = []
    for pmid in pmids:
        art = returned.get(pmid)
        uri = f"pmid:{pmid}"
        if art is None:
            misses.append(Miss(uri=uri, looked_with=via, retrieved=retrieved))
            t.note("corpus:not_found", uri,
                   because=(f"{via} answered with no error and resolved no record",),
                   refs=(uri,))
            continue
        try:
            rec = SourceRecord(
                uri=uri,
                title=str(art.get("title") or "").strip(),
                authors=str(art.get("authors") or "").strip(),
                year=int(art.get("year") or 0),
                journal=str(art.get("journal") or "").strip(),
                abstract=str(art.get("abstract") or "").strip(),
                via=via,
                retrieved=retrieved,
                pmcid=str(art.get("pmcid") or "").strip(),
                doi=str(art.get("doi") or "").strip(),
                publication_type=tuple(str(x) for x in art.get("publication_types") or ()),
                licence=str(art.get("license") or "").strip(),
            )
        except (HollowRecord, TypeError, ValueError) as exc:
            # A response keyed to a real PMID but carrying no title. Storing it would
            # mint a citation out of a parse failure, so it is a failed lookup -- and
            # it is NOT a miss, because the paper plainly exists.
            t.note("corpus:lookup_failed", uri, because=(str(exc),), refs=(uri,))
            continue
        records.append(rec)
        t.note("corpus:resolved", uri,
               because=(f"{via} returned a record; "
                        f"abstract {'held' if rec.abstract else 'absent'}",),
               refs=(uri,))

    t.note("corpus:fetched", f"{len(records)} record(s), {len(misses)} miss(es)",
           because=(f"{len(wanted)} URI(s) requested via {via}",
                    f"{sum(1 for r in records if r.abstract)} carry an abstract"),
           refs=tuple(wanted))
    return t.done(Corpus(records=tuple(records), misses=tuple(misses), built=built))


# --------------------------------------------------------------------------- #
# Storage. Outside the repo, and the path is what enforces it.
# --------------------------------------------------------------------------- #
def corpus_path(study: str) -> Path:
    """Where one study's corpus lives -- and the licence gate, reused not rebuilt.

    ``redistributable=False`` is unconditional. Not "False when a record holds an
    abstract": a corpus is fetched incrementally and the flag would have to be
    re-evaluated on every write, which is a gate that holds until the one time it
    does not. One rule, decided at the path, before a byte is written.
    """
    return cache_dir("corpus", study, redistributable=False) / STORE_FILENAME


_TABLES: Final = ("records", "misses", "statements", "queries", "ledger", "settings")


def _tup(v: Any) -> Any:
    return tuple(_tup(x) for x in v) if isinstance(v, list) else v


def _rows(db: sqlite3.Connection, table: str) -> list[dict[str, Any]]:
    return [{k: _tup(v) for k, v in json.loads(d).items()}
            for (d,) in db.execute(f"SELECT data FROM {table} ORDER BY n")]


def save(corpus: Corpus, path: Path | str) -> Path:
    """Write the corpus as one SQLite file, replaced atomically. Misses are rows too."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.unlink(missing_ok=True)
    with closing(sqlite3.connect(tmp)) as db:
        db.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        version = json.dumps(asdict(corpus.version), sort_keys=True) if corpus.version else ""
        db.executemany("INSERT INTO meta VALUES (?, ?)", [
            ("spec_version", corpus.spec_version), ("built", corpus.built),
            ("version", version)])
        for t in _TABLES:
            # ponytail: rows are JSON; add typed columns and indexes when a query needs them.
            db.execute(f"CREATE TABLE {t} (n INTEGER PRIMARY KEY, data TEXT NOT NULL)")
            db.executemany(f"INSERT INTO {t} (data) VALUES (?)",
                           [(json.dumps(asdict(r), sort_keys=True),)
                            for r in getattr(corpus, t)])
        db.commit()
    os.replace(tmp, p)
    return p


def load(path: Path | str) -> Corpus:
    """Read a corpus back. Raises :class:`CorpusUnavailable` when there is none.

    Not "return an empty corpus": a caller that catches this records the corpus as
    unbuilt; a caller handed ``Corpus()`` would report the literature as silent.
    """
    p = Path(path)
    if not p.is_file():
        raise CorpusUnavailable(
            f"{p} has no corpus. It is fetched, never bundled -- build it with "
            "`panoscope corpus build`, or point PANOSCOPE_REFERENCE_DIR at a cache "
            "that has one. Until then the corpus is unread, not empty."
        )
    try:
        with closing(sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            meta = dict(db.execute("SELECT k, v FROM meta"))
            rows = {t: _rows(db, t) for t in _TABLES}
    except sqlite3.Error as exc:
        raise CorpusUnavailable(
            f"{p} is not a corpus store ({exc}). Refusing rather than guessing -- a "
            "half-read corpus reads as thin literature."
        ) from exc
    v = json.loads(meta["version"]) if meta.get("version") else None
    return Corpus(
        records=tuple(SourceRecord(**d) for d in rows["records"]),
        misses=tuple(Miss(**d) for d in rows["misses"]),
        built=meta.get("built", ""),
        spec_version=meta.get("spec_version", SPEC_VERSION),
        statements=tuple(Statement(**d) for d in rows["statements"]),
        queries=tuple(QueryRun(**d) for d in rows["queries"]),
        ledger=tuple(Disposition(**d) for d in rows["ledger"]),
        version=CorpusVersion(**{k: _tup(x) for k, x in v.items()}) if v else None,
        settings=tuple(Setting(**d) for d in rows["settings"]),
    )


def read_version(path: Path | str) -> CorpusVersion | None:
    """Only the version row, for headers (no records read); CorpusUnavailable when unbuilt."""
    p = Path(path)
    if not p.is_file():
        raise CorpusUnavailable(f"{p} has no corpus")
    try:
        with closing(sqlite3.connect(p.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            v = dict(db.execute("SELECT k, v FROM meta")).get("version")
    except sqlite3.Error as exc:
        raise CorpusUnavailable(f"{p} is not a corpus store ({exc})") from exc
    return CorpusVersion(**{k: _tup(x) for k, x in json.loads(v).items()}) if v else None


# A scheme citable by a Claim but unknown here would render as a citation and resolve
# to nothing. Asserted at import as well as in a test, because the drift is silent and
# the cost of catching it late is a citation that looks fabricated.
assert CITATION_SCHEMES <= SOURCE_SCHEMES, (
    "corpus.CITATION_SCHEMES has drifted from claim.SOURCE_SCHEMES"
)
