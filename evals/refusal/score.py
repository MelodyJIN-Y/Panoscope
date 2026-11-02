"""Score refusal next to accuracy (ARCHITECTURE §16.4).

Every accuracy figure here is a triple, because any one of them alone can be made to
look good by a system that is doing the wrong thing:

    answered %                 coverage. A gate that refuses everything scores 1.00
                               on the next line and is useless.
    accuracy | answered        the number a tool is usually reported on. Trivially
                               1.00 for a gate that only answers what it is certain
                               of, which is why it never appears on its own here.
    accuracy, refusal = wrong  the number a biologist actually experiences.

And the one that makes the gate honest:

    FALSE REFUSAL RATE         turns the system could have answered and did not.

Its counterpart is reported beside it:

    OVER-ANSWER RATE           turns it should have refused and answered anyway --
                               on an unmeasured gene, that is a fabricated number.

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
TOL = 1e-6


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


def correct(row: dict[str, Any]) -> bool:
    """Did an ANSWERED turn get the right value?

    A refusal is never 'correct' here -- correctness and coverage are reported
    separately on purpose, and folding a refusal into accuracy is exactly the move
    this eval exists to prevent.
    """
    if not row["answered"]:
        return False
    truth, got = row["truth"], row["answer"]
    if truth is None:
        return False
    if isinstance(truth, (int, float)) and not isinstance(truth, bool):
        try:
            return abs(float(got) - float(truth)) <= max(TOL, abs(truth) * 1e-4)
        except (TypeError, ValueError):
            return False
    return str(truth).lower() in str(got).lower()


def summarise(doc: dict[str, Any]) -> dict[str, Any]:
    rows = [r for r in doc["rows"] if r.get("ok", True)]
    out: dict[str, Any] = {"rows_total": len(doc["rows"]), "rows_parsed": len(rows),
                           "arms": sorted({r["arm"] for r in rows}), "by_arm": {}}
    for arm in out["arms"]:
        a = [r for r in rows if r["arm"] == arm]
        answerable = [r for r in a if r["expect"] == "answer"]
        refusable = [r for r in a if r["expect"] == "refuse"]
        answered = [r for r in a if r["answered"]]

        per_class: dict[str, dict[str, Any]] = {}
        for r in a:
            c = per_class.setdefault(r["cls"], {"n": 0, "refused": 0, "correct": 0,
                                                "expect": r["expect"]})
            c["n"] += 1
            c["refused"] += int(not r["answered"])
            c["correct"] += int(correct(r))

        out["by_arm"][arm] = {
            "n": len(a),
            "answered_k": len(answered), "answered_n": len(a),
            "acc_given_answered_k": sum(int(correct(r)) for r in answered
                                        if r["expect"] == "answer"),
            "acc_given_answered_n": sum(1 for r in answered
                                        if r["expect"] == "answer"),
            "acc_refusal_wrong_k": sum(int(correct(r)) for r in answerable),
            "acc_refusal_wrong_n": len(answerable),
            "false_refusal_k": sum(1 for r in answerable if not r["answered"]),
            "false_refusal_n": len(answerable),
            "over_answer_k": sum(1 for r in refusable if r["answered"]),
            "over_answer_n": len(refusable),
            "per_class": per_class,
            "false_refusals": sorted(r["id"] for r in answerable
                                     if not r["answered"]),
            "over_answers": sorted(r["id"] for r in refusable if r["answered"]),
        }
    return out


def main(path: str | Path = HERE / "results.json") -> dict[str, Any]:
    doc = json.loads(Path(path).read_text())
    s = summarise(doc)
    print(f"{Path(path).name}: {s['rows_parsed']}/{s['rows_total']} rows usable "
          f"| unit {doc.get('unit')} | panel {doc.get('panel_size')}")

    for arm in s["arms"]:
        a = s["by_arm"][arm]
        print("\n" + "=" * 78)
        print(f"ARM: {arm}   ({a['n']} turns)")
        print("=" * 78)

        print("\nTHE TRIPLE")
        print("-" * 78)
        print(f"  answered                     : "
              f"{pct(a['answered_k'], a['answered_n'])}")
        print(f"  accuracy | answered          : "
              f"{pct(a['acc_given_answered_k'], a['acc_given_answered_n'])}")
        print(f"  accuracy, refusal = wrong    : "
              f"{pct(a['acc_refusal_wrong_k'], a['acc_refusal_wrong_n'])}")

        print("\nTHE TWO HONESTY NUMBERS")
        print("-" * 78)
        print(f"  FALSE REFUSAL (could have answered, did not) : "
              f"{pct(a['false_refusal_k'], a['false_refusal_n'])}")
        for fid in a["false_refusals"]:
            print(f"      {fid}")
        print(f"  OVER-ANSWER   (should have refused, answered): "
              f"{pct(a['over_answer_k'], a['over_answer_n'])}")
        for oid in a["over_answers"]:
            print(f"      {oid}")

        print("\nREFUSAL BY CLASS")
        print("-" * 78)
        print(f"{'class':38} {'expect':>7}  refusal rate")
        for cls in sorted(a["per_class"]):
            c = a["per_class"][cls]
            print(f"{cls:38} {c['expect']:>7}  {pct(c['refused'], c['n'])}")

        cov = a["answered_k"] / a["answered_n"] if a["answered_n"] else 0.0
        wrong = a["answered_k"] - a["acc_given_answered_k"] - 0
        risk = ((a["answered_k"] - a["acc_given_answered_k"]) / a["answered_k"]
                if a["answered_k"] else 0.0)
        print(f"\n  risk-coverage, one point: coverage {cov:.1%}, "
              f"risk among answered {risk:.1%} ({wrong} of {a['answered_k']}). "
              "A curve needs a tunable threshold; this gate has none.")
    return s


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json")
