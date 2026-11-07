"""The literature corpus: what a citation URI resolves to, and how it is found.

``types/claim.py`` makes ``Claim.source`` a URI and nothing more -- "claims do NOT
carry abstracts". This package is the other end of that pointer:

* :mod:`~panoscope.corpus.store` -- :class:`~panoscope.corpus.store.SourceRecord`,
  the live fetch that is the only way one is created, and the three-state
  :func:`~panoscope.corpus.store.resolve`.
* :mod:`~panoscope.corpus.retrieve` -- a stated linear ranking with a relevance
  floor, and the same three states at the query layer.
* :mod:`~panoscope.corpus.window` -- the build's input contract, gene vocabulary and
  the W0..W5 rung, modality and species facets, as pure code.
* :mod:`~panoscope.corpus.vocab` -- HGNC, MeSH and census lookups from real sources.

Three states everywhere, never two: resolved / looked-and-found-nothing /
not-looked-up. Only the middle one is thin literature, and only the middle one is a
claim about the world.

**Nothing here is bundled.** :func:`~panoscope.corpus.store.corpus_path` goes through
the reference licence gate (``redistributable=False``) and cannot land inside the
repository; the corpus is fetched, cached outside the checkout, and rebuilt wholesale
(ARCHITECTURE §8.3, §10.6). Importing this package touches no file and no network.
"""

from panoscope.corpus.retrieve import (
    DEFAULT_FLOOR, DEFAULT_LIMIT, TITLE_WEIGHT, Hit, KeyState, Retrieval, Support,
    key_state, search,
)
from panoscope.corpus.store import (
    CITATION_SCHEMES, RESOLVABLE, SPEC_VERSION, STORE_FILENAME, Connector, Corpus,
    CorpusUnavailable, CorpusVersion, Disposition, HollowRecord, Miss, QueryRun,
    QueryStatus, Resolution, Resolved, SourceRecord, Statement, UnboundSpan,
    corpus_path, fetch, load, log_hash, resolve, save,
)

__all__ = [
    "CITATION_SCHEMES", "DEFAULT_FLOOR", "DEFAULT_LIMIT", "RESOLVABLE",
    "SPEC_VERSION", "STORE_FILENAME", "TITLE_WEIGHT", "Connector", "Corpus",
    "CorpusUnavailable", "CorpusVersion", "Disposition", "Hit", "HollowRecord",
    "KeyState", "Miss", "QueryRun", "QueryStatus", "Resolution", "Resolved",
    "Retrieval", "SourceRecord", "Statement", "Support", "UnboundSpan",
    "corpus_path", "fetch", "key_state", "load", "log_hash", "resolve", "save",
    "search",
]
