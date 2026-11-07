"""The two things every platform adapter needs and must not reimplement: the file
digest, and §6.3's fourth blocker.

No base *class*: ``PlatformAdapter`` is a ``Protocol`` (``types/contracts.py``).

**``declared_clustering``** lives here so no adapter can get it wrong on its own.
An undeclared clustering mode blocks the study when, and only when, there is more
than one sample (``Conditions`` carries no sample count). With one sample
``"independent"`` is a tautology, not a default. An unrecognised mode string raises.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, Literal, Sequence

from panoscope.types.contracts import UndeclaredClustering

ClusteringMode = Literal["joint", "independent", "integrated"]

CLUSTERING_MODES: Final[tuple[str, ...]] = ("joint", "independent", "integrated")
"""The closed vocabulary ``PlatformAdapter.clustering_mode`` may return."""

class UndeclaredSamples(UndeclaredClustering):
    """The same blocker, hit one level earlier: the sample LIST is undeclared.

    Per-sample inputs on disk with no samples named must not become "one sample,
    therefore independent". A subclass of ``UndeclaredClustering`` so every caller
    that renders the fourth blocker as a checklist line renders this one too.
    """


def digest(path: str | Path) -> str:
    """sha256 over the file's BYTES, for ``ReadingRecord.sha256``.

    Over the bytes, not the parsed table, so a parser upgrade cannot move it.
    Streamed, since a store's tables are not guaranteed small.
    """
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


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
