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
"""

from __future__ import annotations

import json
from dataclasses import dataclass
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

STORE_FILENAME: Final = "sources.json"
SPEC_VERSION: Final = "0.1"

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
    via: str = ""                 # which connector produced it
    retrieved: str = ""           # ISO string the caller minted; no clock here

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
        """False once an abstract is attached. Metadata-only records are the NLM's."""
        return not self.abstract.strip()

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

    def find(self, uri: str) -> tuple[SourceRecord, ...]:
        # ponytail: linear scan. A per-study corpus is hundreds of records, not
        # millions; index by uri if a study ever makes that false.
        return tuple(r for r in self.records if r.uri == uri)

    @property
    def licensed_records(self) -> tuple[SourceRecord, ...]:
        """The records carrying an abstract -- i.e. the ones that may not ship."""
        return tuple(r for r in self.records if not r.redistributable)


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


def _record_to_dict(r: SourceRecord) -> dict[str, Any]:
    return {"uri": r.uri, "title": r.title, "authors": r.authors, "year": r.year,
            "journal": r.journal, "abstract": r.abstract, "via": r.via,
            "retrieved": r.retrieved}


def save(corpus: Corpus, path: Path | str) -> Path:
    """Write the corpus as JSON. Misses are written too -- see :class:`Corpus`.

    ponytail: one JSON file. ARCHITECTURE §8.3 wants the study's SQLite store; this
    is hundreds of records rebuilt wholesale, so a query engine buys nothing yet.
    Move it when spans land and the corpus is paged rather than loaded.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec_version": corpus.spec_version,
        "built": corpus.built,
        "records": [_record_to_dict(r) for r in corpus.records],
        "misses": [{"uri": m.uri, "looked_with": m.looked_with,
                    "retrieved": m.retrieved} for m in corpus.misses],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
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
            "corpus.store.fetch(...) and save it, or point PANOSCOPE_REFERENCE_DIR "
            "at a cache that has one. Until then the corpus is unread, not empty."
        )
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusUnavailable(f"{p} could not be read: {exc}") from exc
    if not isinstance(data, dict) or "records" not in data:
        raise CorpusUnavailable(
            f"{p} is not a corpus file (no 'records' key). Refusing rather than "
            "guessing -- a half-read corpus reads as thin literature."
        )
    return Corpus(
        records=tuple(SourceRecord(
            uri=str(d["uri"]), title=str(d.get("title", "")),
            authors=str(d.get("authors", "")), year=int(d.get("year") or 0),
            journal=str(d.get("journal", "")), abstract=str(d.get("abstract", "")),
            via=str(d.get("via", "")), retrieved=str(d.get("retrieved", "")),
        ) for d in data.get("records", ())),
        misses=tuple(Miss(
            uri=str(d["uri"]), looked_with=str(d.get("looked_with", "")),
            retrieved=str(d.get("retrieved", "")),
        ) for d in data.get("misses", ())),
        built=str(data.get("built", "")),
        spec_version=str(data.get("spec_version", SPEC_VERSION)),
    )


# A scheme citable by a Claim but unknown here would render as a citation and resolve
# to nothing. Asserted at import as well as in a test, because the drift is silent and
# the cost of catching it late is a citation that looks fabricated.
assert CITATION_SCHEMES <= SOURCE_SCHEMES, (
    "corpus.CITATION_SCHEMES has drifted from claim.SOURCE_SCHEMES"
)
