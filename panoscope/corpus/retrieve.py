"""Retrieval over the corpus: a stated ranking, a floor, and three outcomes.

ARCHITECTURE §10.4 sketches *"hybrid score (BM25 + dense, weighted RRF) -> relevance
floor"*. This is the floor and a lexical score only: a linear scan suits a per-study
corpus of hundreds of records, and **gene symbols break embedding search**, so
lexical matching is the anchor. BM25 + RRF is the upgrade path once corpus size or
measured ranking quality justifies it.

**The ranking is stated, not inferred.** A title term is worth :data:`TITLE_WEIGHT`,
any other term 1, each query term counts once, ties break on URI -- all visible on
the :class:`Hit`.

**Three outcomes:**

* :attr:`Support.FOUND` -- records cleared the floor.
* :attr:`Support.NONE_FOUND` -- the corpus was searched and nothing cleared it: thin
  literature, *n records searched, none matched*.
* :attr:`Support.NOT_BUILT` -- there was nothing to search. Not a finding at all.

**A hit is topical, never corroboration** (§8.2): :attr:`Retrieval.reads_as` says so.
A score here orders evidence and supplies no value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Iterable

from panoscope.corpus.store import Corpus, QueryStatus, SourceRecord, Statement
from panoscope.types.trace import Trace, Traced

TITLE_WEIGHT: Final = 2
"""A term in the title counts double. The one weight in the ranking, named rather
than inlined so the record can quote it and a tuning change is one edit."""

DEFAULT_FLOOR: Final = 1
"""§10.4's relevance floor: below it the answer is *no support found*. One matched
term is a deliberately low bar -- the floor exists to keep a zero-match record out of
a citation list, not to decide what is interesting."""

DEFAULT_LIMIT: Final = 5

_TOKEN = re.compile(r"[a-z0-9]+")
# ponytail: case-folded alphanumeric tokens, no stemming and no alias normalisation,
# so "fibroblasts" misses "fibroblast" and ERBB2/HER2 are two different queries.
# §10.4 wants alias normalisation off the reference graph's Node.aliases; wire that
# in when the graph is loaded alongside the corpus, which it is not here.


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN.findall(text.casefold()))


class Support(StrEnum):
    FOUND = "found"              # records cleared the floor
    NONE_FOUND = "none_found"    # searched, nothing cleared it -> thin, and a finding
    NOT_BUILT = "not_built"      # nothing to search -> says nothing at all


@dataclass(frozen=True, slots=True)
class Hit:
    """One record and why it ranked where it did. The score is reconstructable.

    ``in_title`` and ``elsewhere`` are kept apart rather than summed into ``score``
    alone, so a reviewer can see that a hit scored 4 on one title term plus two body
    terms rather than on four body terms -- different kinds of relevance that would
    otherwise be the same integer.
    """

    record: SourceRecord
    score: int
    in_title: tuple[str, ...] = ()
    elsewhere: tuple[str, ...] = ()

    @property
    def matched(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.in_title) | set(self.elsewhere)))


@dataclass(frozen=True, slots=True)
class Retrieval:
    """What the corpus had to say about a query, and whether it was asked at all."""

    terms: tuple[str, ...]
    state: Support
    hits: tuple[Hit, ...] = ()
    searched: int = 0            # records actually scanned
    floor: int = DEFAULT_FLOOR

    def __post_init__(self) -> None:
        if (self.state is Support.FOUND) is not bool(self.hits):
            raise ValueError(
                f"hits is non-empty iff state is FOUND (state={self.state!r}, "
                f"{len(self.hits)} hit(s))"
            )
        if self.state is Support.NOT_BUILT and self.searched:
            raise ValueError(
                f"NOT_BUILT means nothing was searched, but searched={self.searched}"
            )

    @property
    def reads_as(self) -> str:
        """The sentence a reviewer sees. Three different sentences, on purpose."""
        q = " ".join(self.terms) or "(no terms)"
        if self.state is Support.FOUND:
            return (f"{len(self.hits)} of {self.searched} record(s) in the corpus "
                    f"mention {q} -- topically relevant, not confirmation that it "
                    "holds in this assay")
        if self.state is Support.NONE_FOUND:
            return (f"{self.searched} record(s) in the corpus were searched and none "
                    f"mentions {q} (floor {self.floor}) -- the literature held here "
                    "is thin on this")
        return (f"no corpus has been built, so nothing was searched for {q} -- this "
                "says nothing about what the literature holds")


def search(corpus: Corpus, terms: Iterable[str], *,
           floor: int = DEFAULT_FLOOR,
           limit: int = DEFAULT_LIMIT) -> Traced[Retrieval]:
    """Rank the corpus against ``terms``. Deterministic, offline, no LLM call.

    ``NOT_BUILT`` means the corpus has neither records nor misses; a corpus whose
    every lookup came back empty *was* built, and only that one is thin literature.
    """
    t = Trace()
    # ponytail: linear scan, one pass, no index; BM25 + RRF is the upgrade path
    wanted = tuple(dict.fromkeys(
        w for term in terms for w in _TOKEN.findall(str(term).casefold())
    ))

    if not corpus.records and not corpus.misses:
        t.note("corpus:not_built", Support.NOT_BUILT.value,
               because=("the corpus holds no records and no completed lookups",
                        "nothing was searched, so nothing may be concluded"),
               refs=tuple(wanted))
        return t.done(Retrieval(terms=wanted, state=Support.NOT_BUILT, floor=floor))

    scored: list[Hit] = []
    for rec in corpus.records:
        title = _tokens(rec.title)
        body = _tokens(" ".join((rec.abstract, rec.journal, rec.authors)))
        in_title = tuple(w for w in wanted if w in title)
        elsewhere = tuple(w for w in wanted if w not in title and w in body)
        score = TITLE_WEIGHT * len(in_title) + len(elsewhere)
        if score >= floor and score > 0:
            scored.append(Hit(record=rec, score=score,
                              in_title=in_title, elsewhere=elsewhere))

    scored.sort(key=lambda h: (-h.score, h.record.uri))
    hits = tuple(scored[:limit])

    if not hits:
        t.note("corpus:no_support", Support.NONE_FOUND.value,
               because=(f"{len(corpus.records)} record(s) searched, "
                        f"{len(corpus.misses)} completed lookup(s) found nothing",
                        f"no record reached the floor of {floor}"),
               refs=tuple(wanted))
        return t.done(Retrieval(terms=wanted, state=Support.NONE_FOUND,
                                searched=len(corpus.records), floor=floor))

    t.note("corpus:support_found", Support.FOUND.value,
           because=(f"{len(scored)} of {len(corpus.records)} record(s) cleared the "
                    f"floor of {floor}; top {len(hits)} returned",
                    f"ranking: {TITLE_WEIGHT} per distinct term in the title, 1 "
                    "elsewhere, ties on uri",
                    "topical relevance only -- never corroboration of a measurement"),
           refs=tuple(h.record.uri for h in hits))
    return t.done(Retrieval(terms=wanted, state=Support.FOUND, hits=hits,
                            searched=len(corpus.records), floor=floor))


@dataclass(frozen=True, slots=True)
class KeyState:
    """FOUND / NONE_FOUND / NOT_BUILT for one (gene, cell type) key of the study's tissue.

    ``gaps`` names every query of the key (or its gene's vocabulary and census) that
    did not run to completion; NONE_FOUND reads "thin" only when there are none.
    """

    gene: str
    cell_type: str
    state: Support
    statements: tuple[Statement, ...] = ()
    gaps: tuple[str, ...] = ()

    @property
    def reads_as(self) -> str:
        what = f"{self.gene} in {self.cell_type or 'this tissue'}"
        if self.state is Support.FOUND:
            n = len({s.uri for s in self.statements})
            return (f"{len(self.statements)} statement(s) about {what} from {n} "
                    "paper(s); topically relevant, never confirmation in this assay")
        if self.state is Support.NOT_BUILT:
            return f"the corpus was not built for {what}, so it cannot settle this"
        if self.gaps:
            return (f"incomplete search for {what} ({'; '.join(self.gaps)}): nothing "
                    "found so far, which says nothing either way")
        return f"searched completely: the literature is thin on {what}"


def key_state(corpus: Corpus, gene: str, cell_type: str) -> KeyState:
    """Resolve one key from the query log and statements; no queries is NOT_BUILT."""
    g, c = gene.casefold(), cell_type.casefold()
    runs = [q for q in corpus.queries if q.gene.casefold() == g
            and q.cell_type.casefold() in (c, "")]
    if not any(q.cell_type.casefold() == c for q in runs):
        return KeyState(gene, cell_type, Support.NOT_BUILT)
    found = tuple(s for s in corpus.statements
                  if s.gene.casefold() == g and s.cell_type.casefold() == c)
    gaps = tuple(f"{q.channel} {q.status.value}" for q in runs
                 if q.status is not QueryStatus.OK)
    return KeyState(gene, cell_type, Support.FOUND if found else Support.NONE_FOUND,
                    found, gaps)
