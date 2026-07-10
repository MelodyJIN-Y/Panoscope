"""Build the calibration table (Markdown) from a verdict list — no recompute.

The calibration table is the honest proof that the confidence rubric both COMMITS
on clean calls and FLAGS shaky ones (see ``tests/test_calibration.py`` and the
README). It is a pure projection of the already-computed ``ClusterVerdict`` set:
one row per cluster, every number tracing to jazzPanda's ``glm_coef``. The
pipeline writes it as a per-dataset tree artifact (``interp/calibration.md``) and
``scripts/calibration_table.py`` reuses this same builder for the README — one
implementation, so the two can never drift.
"""

from __future__ import annotations

from agent.types import ClusterVerdict

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

    Uses the driving-marker list the verdict already computed, so the numbers
    trace to jazzPanda's ``glm_coef`` and nothing is re-derived here.
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


def calibration_markdown(verdicts: list[ClusterVerdict]) -> str:
    """Return the Markdown calibration table for the given verdicts (no recompute)."""
    lines: list[str] = []
    lines.append("| " + " | ".join(COLUMNS) + " |")
    lines.append("| " + " | ".join("---" for _ in COLUMNS) + " |")
    for v in verdicts:
        lines.append("| " + " | ".join(_row(v)) + " |")
    return "\n".join(lines)
