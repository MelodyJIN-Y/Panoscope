"""Claims: the lab's accumulating knowledge, and the two gates that admit it.

One object type (``types/claim.py``); this package is the store around it and
ARCHITECTURE §7.5's split:

* :mod:`~panoscope.claims.evaluability` -- the capability gate, every claim class.
* :mod:`~panoscope.claims.transfer` -- condition deltas, manifest-sourced only.
* :mod:`~panoscope.claims.store` -- read, order, write, and :func:`~panoscope.claims.store.assess`.
* :mod:`~panoscope.claims.notes` -- the reader from ``agent/memory.py``'s note schema
  (IMPLEMENTATION §10).

Nothing here merges, dedupes, consolidates or resolves by recency.
"""
