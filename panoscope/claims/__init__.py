"""Claims: the lab's accumulating knowledge, and the two gates that admit it.

One object type across measurement, lab, curated and literature (``types/claim.py``,
frozen); this package is the store around it and ARCHITECTURE §7.5's split:

* :mod:`~panoscope.claims.evaluability` -- the capability gate, every claim class.
* :mod:`~panoscope.claims.transfer` -- condition deltas, manifest-sourced only.
* :mod:`~panoscope.claims.store` -- read, order, write, and the ordered
  :func:`~panoscope.claims.store.assess` that keeps §5's sequence in one place.
* :mod:`~panoscope.claims.notes` -- the bridge from ``agent/memory.py``'s note
  schema (IMPLEMENTATION §10). A reader: it never rewrites a note file.

Nothing here merges, dedupes, consolidates or resolves by recency. The disagreement
is the product.
"""
