#!/usr/bin/env python
"""Print the Panoscope calibration table as Markdown, for the README.

One row per cluster (c1..c9), straight from ``verdict_for_cluster`` — no
invented values. The table itself is built by ``pipeline.calibration`` (the same
builder the pipeline writes into each dataset tree as ``interp/calibration.md``),
so the README table and the per-dataset artifact can never drift.

Usage:
    .venv/bin/python scripts/calibration_table.py

Deterministic and side-effect-free: it reads the precomputed jazzPanda output
via ``agent`` and writes only to stdout, so its output can be pasted directly
into the README calibration section.
"""

from __future__ import annotations

from panoscope.agent import config as cfg
from panoscope.agent import verdict as V
from panoscope.pipeline.calibration import calibration_markdown


def build_markdown() -> str:
    """Return the full Markdown calibration table for c1..c9."""
    verdicts = [V.verdict_for_cluster(c) for c in cfg.CLUSTER_ORDER]
    return calibration_markdown(verdicts)


def main() -> None:
    print(build_markdown())


if __name__ == "__main__":
    main()
