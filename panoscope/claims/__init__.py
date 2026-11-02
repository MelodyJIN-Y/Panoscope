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

from panoscope.claims.evaluability import (
    DETECTABILITY, Evaluability, Evaluable, evaluability, rests_on,
)
from panoscope.claims.notes import (
    UnanchoredNote, load_notes_as_claims, lookup_state, note_to_claims,
)
from panoscope.claims.store import (
    SEED_RELPATH, Assessment, BadSeed, ClaimStore, NotCommittable, OutOfScope,
    Retracted, assess, from_dict, load_canonical_markers, load_dir,
    thin_literature, to_dict, write_claim,
)
from panoscope.claims.transfer import (
    BLOCKED, CANNOT_BE_SETTLED, MISMATCH, RETURN, WITH_CONDITIONS, Delta,
    TransferResult, manifest_sourced, transfer_check,
)

__all__ = [
    "BLOCKED", "CANNOT_BE_SETTLED", "DETECTABILITY", "MISMATCH", "RETURN",
    "SEED_RELPATH", "WITH_CONDITIONS", "Assessment", "BadSeed", "ClaimStore",
    "Delta", "Evaluability", "Evaluable", "NotCommittable", "OutOfScope",
    "Retracted", "TransferResult", "UnanchoredNote", "assess", "evaluability",
    "from_dict", "load_canonical_markers", "load_dir", "load_notes_as_claims",
    "lookup_state", "manifest_sourced", "note_to_claims", "rests_on",
    "thin_literature", "to_dict", "transfer_check", "write_claim",
]
