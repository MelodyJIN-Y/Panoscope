"""Score the panel-absence ablation.

The primary metric is the ERROR RATE: the fraction of calls in which an off-panel
canonical marker for the true cell type appears in the model's ``evidence_against``
list. That is the model using a gene that was never measured as a reason to doubt.

Three things are reported alongside it, because the error rate alone is easy to
misread:

- CORRECT HANDLING -- the model names the off-panel gene AND labels it unmeasured.
  This is the behaviour we want. A model that never mentions the gene is not doing
  well, it is merely silent, and the benchmark must not reward that.
- SILENT -- the gene is never mentioned at all.
- The confidence distribution, since the interesting failure is not only "listed it
  as evidence against" but "lowered the band because of it".
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Normal approximation is wrong at these n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def main() -> None:
    rows = json.loads((HERE / "results.json").read_text())
    ok = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]

    print(f"{len(ok)}/{len(rows)} calls returned parseable output")
    if failed:
        by_err = defaultdict(int)
        for r in failed:
            by_err[r["error"][:60]] += 1
        for e, n in sorted(by_err.items(), key=lambda x: -x[1]):
            print(f"  {n:3d}  {e}")

    # ---- primary table ---------------------------------------------------- #
    print("\n" + "=" * 86)
    print("ERROR RATE: off-panel canonical marker cited as evidence AGAINST the call")
    print("=" * 86)
    print(f"{'arm':12} {'model':20} {'n':>4} {'errors':>7} {'rate':>8}  {'95% CI':>16}")
    print("-" * 86)
    agg = defaultdict(list)
    for r in ok:
        agg[(r["arm"], r["model"])].append(r)
    for arm in ("bare", "panel", "panel_rule", "panoscope"):
        for (a, m), rs in sorted(agg.items()):
            if a != arm:
                continue
            n = len(rs)
            k = sum(1 for r in rs if r["offpanel_in_against"])
            lo, hi = wilson(k, n)
            print(f"{a:12} {m:20} {n:>4} {k:>7} {k/n:>7.1%}  [{lo:>5.1%},{hi:>5.1%}]")
        print("-" * 86)

    # ---- per-arm behaviour breakdown -------------------------------------- #
    print("\nHOW THE OFF-PANEL GENE WAS HANDLED (model arms only)")
    print(f"{'arm':12} {'n':>4} {'cited against':>14} {'named unmeasured':>17} {'silent':>8}")
    for arm in ("bare", "panel", "panel_rule"):
        rs = [r for r in ok if r["arm"] == arm]
        n = len(rs) or 1
        against = sum(1 for r in rs if r["offpanel_in_against"])
        unmeas = sum(1 for r in rs if r["offpanel_called_unmeasured"] and not r["offpanel_in_against"])
        silent = sum(1 for r in rs if not r["offpanel_mentioned"])
        print(f"{arm:12} {len(rs):>4} {against/n:>13.1%} {unmeas/n:>16.1%} {silent/n:>7.1%}")

    # ---- the hot spot ----------------------------------------------------- #
    print("\nPER-CLUSTER ERROR RATE (model arms pooled)")
    per = defaultdict(lambda: [0, 0])
    for r in ok:
        if r["arm"] == "panoscope":
            continue
        per[r["cluster"]][1] += 1
        if r["offpanel_in_against"]:
            per[r["cluster"]][0] += 1
    for c, (k, n) in sorted(per.items()):
        bar = "#" * round(20 * k / n) if n else ""
        print(f"  {c:4} {k:>3}/{n:<3} {k/n:>6.1%}  {bar}")

    # ---- confidence ------------------------------------------------------- #
    print("\nCONFIDENCE DISTRIBUTION")
    order = ["Very High", "High", "Medium-High", "Medium", "Low"]
    for arm in ("bare", "panel", "panel_rule", "panoscope"):
        rs = [r for r in ok if r["arm"] == arm]
        c = defaultdict(int)
        for r in rs:
            c[r["confidence"] or "(none)"] += 1
        parts = [f"{k}={c[k]}" for k in order if c.get(k)]
        extra = [f"{k}={v}" for k, v in c.items() if k not in order]
        print(f"  {arm:12} " + "  ".join(parts + extra))

    # ---- the verdict ------------------------------------------------------ #
    def rate(arm: str) -> float:
        rs = [r for r in ok if r["arm"] == arm]
        return sum(1 for r in rs if r["offpanel_in_against"]) / (len(rs) or 1)

    print("\n" + "=" * 86)
    b, p, pr = rate("bare"), rate("panel"), rate("panel_rule")
    print(f"bare {b:.1%}  ->  +panel {p:.1%}  ->  +panel&rule {pr:.1%}  ->  panoscope {rate('panoscope'):.1%}")
    print("\nReading it:")
    print("  If +panel collapses the rate, the finding is a prompting insight and a")
    print("  deterministic gate is hard to justify.")
    print("  If the rate survives being told the panel, that is a result about model")
    print("  behaviour and it is what the architecture is for.")


if __name__ == "__main__":
    main()
