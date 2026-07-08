#!/usr/bin/env python
"""Print the Panoscope calibration table as Markdown, for the README.

One row per cluster (c1..c9), straight from ``verdict_for_cluster`` — no
invented values. Columns: Cluster, Cell type, Confidence, Verify, Driving
markers (the canonical drivers with their real jazzPanda glm_coef).

Usage:
    .venv/bin/python scripts/calibration_table.py

Deterministic and side-effect-free: it reads the precomputed jazzPanda output
via ``agent`` and writes only to stdout, so its output can be pasted directly
into the README calibration section.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running the script directly (python scripts/calibration_table.py) by
# putting the project root on the path before importing the agent package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import config as cfg  # noqa: E402
from agent import verdict as V  # noqa: E402
from agent.types import ClusterVerdict  # noqa: E402

# How many driving canonical markers to name per row (keeps the table readable).
MAX_DRIVERS: int = 3

COLUMNS: tuple[str, ...] = (
    "Cluster",
    "Cell type",
    "Confidence",
    "Verify",
    "Driving markers",
)


def _driving_markers_cell(verdict: ClusterVerdict) -> str:
    """Render the driving canonical markers as ``GENE (glm N.NN)`` list.

    Uses the same driving-marker list the verdict already computed, so the
    numbers trace to jazzPanda's glm_coef and nothing is re-derived here.
    """
    drivers = verdict.opening.driving_markers[:MAX_DRIVERS]
    if not drivers:
        return "—"
    return ", ".join(f"{d.gene} (glm {d.glm_coef:.2f})" for d in drivers)


def _row(verdict: ClusterVerdict) -> tuple[str, ...]:
    return (
        verdict.cluster,
        verdict.cell_type,
        verdict.confidence,
        "TRUE" if verdict.verify else "FALSE",
        _driving_markers_cell(verdict),
    )


def build_markdown() -> str:
    """Return the full Markdown calibration table for c1..c9."""
    rows = [_row(V.verdict_for_cluster(c)) for c in cfg.CLUSTER_ORDER]

    lines: list[str] = []
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    print(build_markdown())


if __name__ == "__main__":
    main()
