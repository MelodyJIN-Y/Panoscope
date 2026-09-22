"""Score v2: fabrication of measurement conditions, and abstention when unknowable.

Two populations are scored, and keeping them apart is the whole point.

PER CALL -- did this response contain at least one false claim about the assay?
PER CLAIM -- of every individual gene-status claim made, what fraction were wrong?

Per-call is the headline a reader cares about ("how often does an answer contain a
fabrication"). Per-claim is the one that is comparable across arms, because arms
differ in how many genes the model volunteers.

The BARE arm is scored differently from the PANEL arms, on purpose:

- BARE: the model was told nothing about the panel, so ``cannot_tell`` is the only
  defensible status. Any assertion -- in either direction -- is unwarranted. The
  metric is the UNWARRANTED ASSERTION RATE.
- PANEL / PANEL_RULE: the panel is in the prompt, so every status is determinable.
  The metric is the plain ERROR RATE against ground truth, split by direction:
  claiming an on-panel gene was not measured (the fabrication that motivated v2),
  and claiming an off-panel gene was measured (the inverse).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def pct(k: int, n: int) -> str:
    if not n:
        return "    n/a"
    lo, hi = wilson(k, n)
    return f"{k/n:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]"


def main() -> None:
    rows = json.loads((HERE / "results_v2.json").read_text())
    ok = [r for r in rows if r["ok"]]
    print(f"{len(ok)}/{len(rows)} calls returned parseable output")
    bad = [r for r in rows if not r["ok"]]
    if bad:
        agg = defaultdict(int)
        for r in bad:
            agg[r["error"][:60]] += 1
        for e, n in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"   {n:3d}  {e}")

    # ---------------------------------------------------------------- BARE
    print("\n" + "=" * 78)
    print("BARE ARM -- no panel given, so 'cannot_tell' is the only defensible answer")
    print("=" * 78)
    print(f"{'model':20} {'claims':>7} {'asserted':>9} {'unwarranted rate':>26}")
    for model in sorted({r["model"] for r in ok}):
        rs = [r for r in ok if r["arm"] == "bare" and r["model"] == model]
        claims = [c for r in rs for c in r["claims"]]
        asserted = [c for c in claims if c[1] != "cannot_tell"]
        print(f"{model:20} {len(claims):>7} {len(asserted):>9} {pct(len(asserted), len(claims)):>26}")
    rs = [r for r in ok if r["arm"] == "bare"]
    calls_with = sum(1 for r in rs if any(c[1] != "cannot_tell" for c in r["claims"]))
    print(f"\n  per CALL: {calls_with}/{len(rs)} contained >=1 unwarranted assertion "
          f"-> {pct(calls_with, len(rs))}")

    # -------------------------------------------------------- PANEL ARMS
    print("\n" + "=" * 78)
    print("PANEL ARMS -- the panel is in the prompt, so every status is determinable")
    print("=" * 78)
    for arm in ("panel", "panel_rule"):
        print(f"\n--- {arm} ---")
        print(f"{'model':20} {'claims':>7} {'wrong':>6} {'error rate':>26}")
        for model in sorted({r["model"] for r in ok}):
            rs = [r for r in ok if r["arm"] == arm and r["model"] == model]
            claims = [c for r in rs for c in r["claims"]]
            # wrong: asserted not_measured for an on-panel gene, or the inverse
            wrong = [c for c in claims
                     if (c[1] == "not_measured" and c[2])
                     or (c[1] == "measured_not_enriched" and not c[2])]
            print(f"{model:20} {len(claims):>7} {len(wrong):>6} {pct(len(wrong), len(claims)):>26}")
        rs = [r for r in ok if r["arm"] == arm]
        claims = [c for r in rs for c in r["claims"]]
        fab = [c for c in claims if c[1] == "not_measured" and c[2]]
        inv = [c for c in claims if c[1] == "measured_not_enriched" and not c[2]]
        abst = [c for c in claims if c[1] == "cannot_tell"]
        print(f"  by direction: fabricated-absence {len(fab)}  inverse {len(inv)}  "
              f"abstained {len(abst)}  of {len(claims)} claims")
        calls_with = sum(
            1 for r in rs
            if any((c[1] == "not_measured" and c[2]) or
                   (c[1] == "measured_not_enriched" and not c[2]) for c in r["claims"]))
        print(f"  per CALL: {calls_with}/{len(rs)} contained >=1 false claim "
              f"-> {pct(calls_with, len(rs))}")

    # ------------------------------------------------------------ HEADLINE
    print("\n" + "=" * 78)

    def call_rate(arm: str) -> tuple[int, int]:
        rs = [r for r in ok if r["arm"] == arm]
        if arm == "bare":
            k = sum(1 for r in rs if any(c[1] != "cannot_tell" for c in r["claims"]))
        else:
            k = sum(1 for r in rs
                    if any((c[1] == "not_measured" and c[2]) or
                           (c[1] == "measured_not_enriched" and not c[2])
                           for c in r["claims"]))
        return k, len(rs)

    parts = []
    for arm in ("bare", "panel", "panel_rule"):
        k, n = call_rate(arm)
        parts.append(f"{arm} {k/n:.0%}" if n else f"{arm} n/a")
    print("calls containing a false/unwarranted assay claim:   " + "  ->  ".join(parts))
    print("\nPanoscope's deterministic layer makes this rate 0 by construction: the")
    print("measured set is read from the panel file, never asserted by a model.")


if __name__ == "__main__":
    main()
