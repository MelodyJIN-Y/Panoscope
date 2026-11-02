"""The two things every platform adapter needs and must not reimplement: the file
digest, and §6.3's fourth blocker.

There is no base *class* here, and that is the point. ``PlatformAdapter`` is a
``Protocol`` (``types/contracts.py``) so an adapter declares its contract by having
the attributes, not by inheriting; a base class would put an import edge back between
the registry and its entries and make an out-of-tree adapter impossible. What is
shared is *behaviour nobody should write twice*, and that is exactly two functions.

**Why ``declared_clustering`` lives here rather than in each adapter.** §6.3's fourth
blocker is the only conditional one: an undeclared clustering mode blocks the study
when, and only when, there is more than one sample. ``Conditions`` deliberately
carries no sample count (see ``conditions.py``), so the check cannot live there, and
the adapter is the one object that holds the count first-hand. But "cannot live in
``Conditions``" is not "must be copied into every adapter": a second platform that
got the ``> 1`` wrong -- or that defaulted to ``"joint"`` because its vendor usually
does -- would silently make cluster 3 in sample A the same population as cluster 3 in
sample B, which is wrong under independent clustering and *invisible under either*,
because the labels match whichever answer is true. One implementation, one test.

**A single sample is not a guess.** With one sample there is nothing to cluster
jointly, nothing to integrate and nothing to compare across, so the three modes are
the same statement and ``"independent"`` is a tautology rather than a default. The
mode is only unknowable -- and therefore blocking -- once a second sample exists.

**An unrecognised mode string raises too.** ``"batch-corrected"`` in a manifest is not
a declaration of one of the three things the contract names; it is a word the writer
hoped would be understood. Accepting it would let an undeclared study pass the
blocker by typo, which is the failure mode the blocker exists to catch.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal, Sequence

from panoscope.types.contracts import UndeclaredClustering

ClusteringMode = Literal["joint", "independent", "integrated"]

CLUSTERING_MODES: Final[tuple[str, ...]] = ("joint", "independent", "integrated")
"""The closed vocabulary ``PlatformAdapter.clustering_mode`` may return."""

_CHUNK: Final[int] = 1 << 20


def digest(path: str | Path) -> str:
    """sha256 over the file's BYTES, for ``ReadingRecord.sha256``.

    Over the bytes rather than over the parsed table on purpose: "the same inputs"
    has to be a statement about the file that was actually read, not about what a
    parser made of it. A pandas upgrade that changed a dtype inference would move a
    digest taken over the frame, and the golden record would then report an input
    change that never happened.

    Chunked because a store's tables are not guaranteed small, and reading one whole
    file into memory to hash it is how a reader that works on the 280-gene demo
    falls over on a real transcriptome.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def declared_clustering(samples: Sequence[str], declared: str | None,
                        where: str) -> ClusteringMode:
    """§6.3's fourth blocker, enforced once. Never guesses; never defaults.

    ``where`` names the store in the error, because the biologist's next action is
    to go and declare the mode in that specific manifest, and "undeclared clustering
    mode" with nothing to open is a traceback rather than a task.
    """
    if declared is not None:
        mode = str(declared).strip().lower()
        if mode not in CLUSTERING_MODES:
            raise UndeclaredClustering(
                f"{where} declares clustering mode {declared!r}, which is not one of "
                f"{', '.join(CLUSTERING_MODES)}. An unrecognised word is not a "
                "declaration; say which of the three it was."
            )
        return mode  # type: ignore[return-value]  # checked against the closed list

    if len(samples) > 1:
        raise UndeclaredClustering(
            f"{where} holds {len(samples)} samples ({', '.join(map(str, samples))}) "
            "and does not declare how they were clustered (ARCHITECTURE §6.3). Joint "
            "and independent clustering give the same cluster labels two different "
            "meanings, so this cannot be defaulted: declare "
            f"{' | '.join(CLUSTERING_MODES)}."
        )
    return "independent"
