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

import sys
from pathlib import Path

# Allow running the script directly (python scripts/calibration_table.py) by
# putting the project root on the path before importing the agent package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import config as cfg  # noqa: E402
from agent import verdict as V  # noqa: E402
from pipeline.calibration import calibration_markdown  # noqa: E402


def build_markdown() -> str:
    """Return the full Markdown calibration table for c1..c9."""
    verdicts = [V.verdict_for_cluster(c) for c in cfg.CLUSTER_ORDER]
    return calibration_markdown(verdicts)


def main() -> None:
    print(build_markdown())


if __name__ == "__main__":
    main()
