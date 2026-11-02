"""Score generator swap / deterministic invariance (ARCHITECTURE §16.3).

The report opens with what MOVED, for the same reason ``seeded_errors/score.py``
opens with what it missed: the invariance is the advertised property, so the list of
fields that failed it is the only part a reviewer cannot get from the architecture
document.

Three populations, answering different questions:

PURE fields       -- pre-registered invariant. A move here is a REGRESSION: a seam
                     opened between generated text and a deterministic field.
SEAM fields       -- pre-registered movable. The rate is the finding, not the
                     failure, and it is reported with the generator that moved it.
per generator     -- whether one generator is responsible. A seam that only one
                     style trips is a different problem from one every style trips.

A property this scorer deliberately does NOT compute: "byte-identical across model
families" as a single headline. §16.3 words the claim that way, and worded that way
it is true of the pure fields by construction and false of the seam, so one number
over both would be meaningless. The two are reported apart.

``summarise`` returns the numbers; ``main`` prints them. Tests score committed
fixtures offline, which is what makes the README's figures checkable.
"""

from __future__ import annotations

import json
import math
import sys
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
        pure = [r for r in a if r["expected_invariant"]]
        seam = [r for r in a if not r["expected_invariant"]]

        per_field: dict[str, dict[str, Any]] = {}
        for r in a:
            f = per_field.setdefault(r["field"], {
                "k": 0, "n": 0, "expected_invariant": r["expected_invariant"]})
            f["n"] += 1
            f["k"] += int(r["moved"])

        per_generator: dict[str, dict[str, int]] = {}
        for r in seam:
            g = per_generator.setdefault(r["generator"], {"k": 0, "n": 0})
            g["n"] += 1
            g["k"] += int(r["moved"])

        out["by_arm"][arm] = {
            "n": len(a),
            "generators": sorted({r["generator"] for r in a}),
            "pure_moved_k": sum(int(r["moved"]) for r in pure),
            "pure_moved_n": len(pure),
            "seam_moved_k": sum(int(r["moved"]) for r in seam),
            "seam_moved_n": len(seam),
            "per_field": per_field,
            "per_generator": per_generator,
            # A pure field that moved is a regression and is named in full.
            "regressions": sorted(r["id"] for r in pure if r["moved"]),
            "seam_hits": sorted(r["id"] for r in seam if r["moved"]),
        }
    return out


def main(path: str | Path = HERE / "results.json") -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    s = summarise(doc)
    print(f"{Path(path).name}: {s['rows_parsed']}/{s['rows_total']} rows usable "
          f"| clusters {len(doc.get('clusters', []))} "
          f"| panel {doc.get('panel_size')}")

    for arm in s["arms"]:
        a = s["by_arm"][arm]
        print("\n" + "=" * 78)
        print(f"ARM: {arm}   ({a['n']} field observations, "
              f"{len(a['generators'])} generators)")
        print("=" * 78)

        print("\nREGRESSIONS -- a field pre-registered invariant that MOVED")
        print("-" * 78)
        if not a["regressions"]:
            print("  (none)")
        for rid in a["regressions"]:
            print(f"  {rid}")

        print("\nTHE SEAM -- fields a generator is able to move")
        print("-" * 78)
        print(f"  moved: {pct(a['seam_moved_k'], a['seam_moved_n'])}")
        for gen in sorted(a["per_generator"]):
            g = a["per_generator"][gen]
            print(f"    {gen:22} {pct(g['k'], g['n'])}")

        print("\nBY FIELD")
        print("-" * 78)
        print(f"{'field':20} {'invariant?':>11}  move rate")
        for name in sorted(a["per_field"]):
            f = a["per_field"][name]
            print(f"{name:20} {str(f['expected_invariant']):>11}  "
                  f"{pct(f['k'], f['n'])}")

        print("\nHEADLINE")
        print("-" * 78)
        print(f"  pure fields that moved (0 is the claim) : "
              f"{pct(a['pure_moved_k'], a['pure_moved_n'])}")
        print(f"  seam fields that moved (the finding)    : "
              f"{pct(a['seam_moved_k'], a['seam_moved_n'])}")
    return s


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json")
