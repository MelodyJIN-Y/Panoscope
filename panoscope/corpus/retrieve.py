"""Retrieval over the corpus: a stated ranking, a floor, and three outcomes.

ARCHITECTURE §10.4 sketches *"hybrid score (BM25 + dense, weighted RRF) -> relevance
floor"*. This is the floor and the honest half of the score, and not the rest. A
linear scan over a per-study corpus of hundreds of records is what the data volume
warrants; a vector index here would be a dependency, an embedding model to version in
the corpus stamp, and a ranking nobody can read off the record -- bought with nothing
measured to justify it. §10.4's own reason for putting BM25 first survives intact:
**gene symbols break embedding search**, so lexical matching is the anchor, not the
fallback.

**The ranking is stated, not inferred.** A term in the title is worth
:data:`TITLE_WEIGHT`, a term anywhere else is worth 1, each query term counts once,
and ties break on URI. Every one of those is a number a reviewer can see on the
:class:`Hit`, which is the property that matters: a score nobody can reconstruct is a
relevance judgement smuggled in as arithmetic.

**Three outcomes, the same three as everywhere else.**

* :attr:`Support.FOUND` -- records cleared the floor.
* :attr:`Support.NONE_FOUND` -- the corpus was searched and nothing cleared it. This
  is thin literature, concretely: *n records searched, none matched*.
* :attr:`Support.NOT_BUILT` -- there was nothing to search. Not a finding at all.

The middle and the last are one ``if`` apart and read identically in a sentence unless
something keeps them apart, which is ``5b7ab08`` again at the retrieval layer.

**A hit is topical, never corroboration** (§8.2's Caution). A paper retrieved for a
triple is relevant to it; it is not evidence the marker works in your assay, and
:attr:`Retrieval.reads_as` says so rather than leaving the inference to the reader.
Nothing in this module produces a number that could reach a confidence band: a score
here orders evidence, which is exactly what reference and literature data are allowed
to do, and supplies no value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Iterable

from panoscope.corpus.store import Corpus, SourceRecord
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

    ``NOT_BUILT`` is decided on the corpus having neither records nor misses -- a
    corpus whose every lookup came back empty *was* built, and its silence is the
    productive kind. That single condition is the difference between "we looked and
    the literature is thin" and "nobody has looked", and §10.4's *no support found*
    may only ever be said about the first.
    """
    t = Trace()
    # ponytail: linear scan, one pass, no index. See the module docstring for why
    # BM25 + RRF is the upgrade path and what has to be true before it is worth it.
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
