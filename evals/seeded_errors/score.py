"""Score seeded-error detection. Misses first, then catches, then false positives.

The order of the report is the argument. A harness that opens with its recall is
marketing; what a reviewer needs first is the list of seeded errors that walked
straight through, because that list is the honest boundary of the architecture's
claim.

Three populations, and they answer different questions:

POISONED, expected caught   -- recall. Did the gate catch what it says it catches?
POISONED, expected missed   -- the boundary. Pre-registered as out of reach; if one
                               of these IS caught, that is news and gets printed.
CLEAN                       -- the false-positive rate. Recall alone is meaningless
                               for a gate that fails closed (ARCHITECTURE §16.1).

``summarise`` returns the numbers; ``main`` prints them. Tests score committed
fixtures offline, which is what makes the README's figures checkable rather than
quotable.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))


def pct(k: int, n: int) -> str:
    if not n:
        return "     n/a"
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {k/n:>6.1%}  [{lo:>5.1%},{hi:>5.1%}]"


def summarise(doc: dict[str, Any]) -> dict[str, Any]:
    """Every number the README quotes, from the rows alone."""
    rows = [r for r in doc["rows"] if r.get("ok", True)]
    out: dict[str, Any] = {
        "rows_total": len(doc["rows"]),
        "rows_parsed": len(rows),
        "arms": sorted({r["arm"] for r in rows}),
        "by_arm": {},
    }
    for arm in out["arms"]:
        a = [r for r in rows if r["arm"] == arm]
        poisoned = [r for r in a if not r["clean"]]
        clean = [r for r in a if r["clean"]]
        in_scope = [r for r in poisoned if r["expected_caught"]]
        boundary = [r for r in poisoned if not r["expected_caught"]]

        per_class: dict[str, dict[str, Any]] = {}
        for r in poisoned:
            c = per_class.setdefault(r["cls"], {"k": 0, "n": 0,
                                                "expected": r["expected_caught"]})
            c["n"] += 1
            c["k"] += int(r["detected"])

        out["by_arm"][arm] = {
            "recall_k": sum(int(r["detected"]) for r in in_scope),
            "recall_n": len(in_scope),
            "boundary_caught_k": sum(int(r["detected"]) for r in boundary),
            "boundary_n": len(boundary),
            "false_positive_k": sum(int(r["detected"]) for r in clean),
            "false_positive_n": len(clean),
            "per_class": per_class,
            "misses": sorted(r["id"] for r in poisoned if not r["detected"]),
            "surprises": sorted(r["id"] for r in boundary if r["detected"]),
        }
    return out


def main(path: str | Path = HERE / "results.json") -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    s = summarise(doc)
    print(f"{Path(path).name}: {s['rows_parsed']}/{s['rows_total']} rows usable "
          f"| unit bands {doc.get('unit_bands')} | panel {doc.get('panel_size')}")

    for arm in s["arms"]:
        a = s["by_arm"][arm]
        print("\n" + "=" * 78)
        print(f"ARM: {arm}")
        print("=" * 78)

        print("\nWHAT IT MISSED -- seeded errors that walked straight through")
        print("-" * 78)
        if not a["misses"]:
            print("  (none)")
        missed_by_class: dict[str, list[str]] = defaultdict(list)
        for mid in a["misses"]:
            missed_by_class[mid.split(":")[0]].append(mid)
        for cls in sorted(missed_by_class):
            ids = missed_by_class[cls]
            c = a["per_class"].get(cls, {})
            tag = ("pre-registered as out of reach" if c.get("expected") is False
                   else "EXPECTED TO BE CAUGHT -- this is a regression")
            shown = ", ".join(ids[:3]) + (" ..." if len(ids) > 3 else "")
            print(f"  {cls:34} {len(ids):>3}/{c.get('n', len(ids)):<3} {tag}")
            print(f"  {'':34} {shown}")

        if a["surprises"]:
            print("\n  caught although pre-registered as out of reach:")
            for sid in a["surprises"]:
                print(f"    {sid}")

        print("\nDETECTION BY CLASS")
        print("-" * 78)
        print(f"{'class':34} {'expected':>9}  detection")
        for cls in sorted(a["per_class"]):
            c = a["per_class"][cls]
            print(f"{cls:34} {str(c['expected']):>9}  {pct(c['k'], c['n'])}")

        print("\nHEADLINE")
        print("-" * 78)
        print(f"  recall, classes claimed catchable : "
              f"{pct(a['recall_k'], a['recall_n'])}")
        print(f"  classes pre-registered as missed  : "
              f"{pct(a['boundary_caught_k'], a['boundary_n'])} caught")
        print(f"  FALSE POSITIVES on clean controls : "
              f"{pct(a['false_positive_k'], a['false_positive_n'])}")
    return s


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json")
