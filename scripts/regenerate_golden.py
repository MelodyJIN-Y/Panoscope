#!/usr/bin/env python
"""Regenerate the golden-output fixture — a deliberate, reviewable act.

The golden test (``tests/test_golden.py``) pins a normalised projection of the
deterministic interpretation artifacts for the demo study: the 9 per-cluster
verdicts, the 11-column verdict CSV, the holistic review, and the calibration
table. If behaviour changes, that test fails; this script is the ONLY way the
expectation is allowed to move.

Usage:
    .venv/bin/python scripts/regenerate_golden.py            # dry run: show the diff
    .venv/bin/python scripts/regenerate_golden.py --write    # rewrite the fixture

The default is a dry run on purpose: nothing regenerates by accident, and the
``--write`` run produces a diff in ``tests/golden/`` that a reviewer reads as
"here is exactly how the engine's output changed, and why".

Writes only ``tests/golden/demo_interpretation.json`` and its ``.sha256``. It
never runs ``pipeline.run.run()`` (which writes a whole study tree and reaches
for the live annotate stage) and never touches the network.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

# The same isolation tests/conftest.py gives the golden test: a reference source or
# a context note cached on this machine must not move the fixture.
for _var in ("PANOSCOPE_REFERENCE_DIR", "PANOSCOPE_CONTEXT_DIR"):
    os.environ[_var] = tempfile.mkdtemp(prefix="panoscope-golden-")

# Allow running the script directly (python scripts/regenerate_golden.py) by
# putting the project root on the path before importing the project packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests import golden_projection as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--write",
        action="store_true",
        help="rewrite tests/golden/demo_interpretation.json + .sha256 (default: dry run)",
    )
    args = ap.parse_args()

    projection = G.build_projection()
    new_hash = G.projection_hash(projection)

    try:
        old = G.load_fixture()
        old_hash = G.projection_hash(old)
    except FileNotFoundError:
        old, old_hash = None, None

    if old is None:
        print(f"[golden] no fixture yet at {G.FIXTURE_PATH.relative_to(_PROJECT_ROOT)}")
    elif old_hash == new_hash:
        print(f"[golden] fixture is already up to date ({new_hash[:16]}...). Nothing to do.")
        return 0
    else:
        print(f"[golden] fixture would change: {old_hash[:16]}... -> {new_hash[:16]}...\n")
        print(G.diff_report(old, projection))
        print()

    if not args.write:
        print("[golden] dry run. Re-run with --write to accept these values.")
        return 0

    digest = G.write_fixture(projection)
    print(f"[golden] wrote {G.FIXTURE_PATH.relative_to(_PROJECT_ROOT)}")
    print(f"[golden] wrote {G.HASH_PATH.relative_to(_PROJECT_ROOT)} ({digest})")
    print("[golden] review the diff before committing — it IS the record of the behaviour change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
