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

THE THIRD POPULATION, and why the first two cannot see it
---------------------------------------------------------

Both metrics above take ground truth to be ``on_panel`` -- a binary. It is not.
CLAUDE.md makes detectability a capability frame of its own: 49 of this panel's 280
genes (17.5%) produced no usable spatial signal, so for those genes *both*
assertions are false. ``not_measured`` is wrong because the probe was on the panel;
``measured_not_enriched`` is wrong because it reports a failed measurement as a
negative result -- it is the ``UNDERPOWERED`` state being spent as ``TESTED_ABSENT``,
which is the exact collapse the four-state lattice exists to prevent. Only
``cannot_tell`` is defensible, in every arm.

This matters most for the arms that look best above. ``_PANEL_BLOCK`` pastes all 280
gene names as one flat list under "these are the N genes measured in this
experiment" -- so the prompt *itself* asserts the 49 were measured, and the panel
arms are scored against a ground truth their own prompt contradicted. The headline
94.8% -> 1.5% -> 0.0% reads as "supplying the panel fixes it"; this section is the
part of the assay that supplying the panel breaks.

Detectability is derived from the study at scoring time and never stored on a
record, so the committed ``results_v2.json`` scores under this metric with no new
API call.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[1]


def no_signal_genes() -> frozenset[str]:
    """Panel genes that produced no usable spatial signal, read off the study.

    Derived through the adapter's own predicate, so a study with full detectability
    answers the empty set and this section reports nothing -- the demo's 49 is read,
    never asserted. Mirrors ``ui/report._n_no_signal``.
    """
    sys.path.insert(0, str(REPO))
    from panoscope.agent import data as _data
    from panoscope.platforms.xenium import is_no_signal

    md = _data.load_markers()
    return frozenset(str(r.gene).upper() for r in md.itertuples()
                     if is_no_signal(str(r.top_cluster)))


def detectability(rows: list[dict], nosig: frozenset[str] | None = None) -> dict:
    """Score every claim about a gene that was on the panel but yielded no signal.

    ``cannot_tell`` is the only defensible status, so ``overclaimed`` counts the
    claims that spent an absence the measurement could not support -- in either
    direction, which is why the two are also reported apart.
    """
    nosig = no_signal_genes() if nosig is None else nosig
    ok = [r for r in rows if r["ok"]]
    by_arm: dict[str, dict] = {}
    for arm in sorted({r["arm"] for r in ok}):
        claims = [c for r in ok if r["arm"] == arm for c in r["claims"]
                  if c[0] in nosig]
        as_measured = [c for c in claims if c[1] == "measured_not_enriched"]
        as_absent = [c for c in claims if c[1] == "not_measured"]
        abstained = [c for c in claims if c[1] == "cannot_tell"]
        by_arm[arm] = {
            "n_claims": len(claims),
            "overclaimed_k": len(as_measured) + len(as_absent),
            "as_measured_not_enriched": len(as_measured),
            "as_not_measured": len(as_absent),
            "abstained": len(abstained),
        }
    return {"n_no_signal_genes": len(nosig), "by_arm": by_arm}


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

    # ------------------------------------------------- DETECTABILITY (frame 2)
    det = detectability(rows)
    if det["n_no_signal_genes"]:
        print("\n" + "=" * 78)
        print(f"DETECTABILITY -- the {det['n_no_signal_genes']} panel genes that produced "
              "no usable signal")
        print("only 'cannot_tell' is defensible: the probe was on the panel, so "
              "'not_measured'")
        print("is wrong; the measurement produced nothing, so 'measured_not_enriched' "
              "reports")
        print("a failed measurement as a negative result.")
        print("=" * 78)
        print(f"{'arm':12} {'claims':>7} {'overclaimed':>12} {'rate':>26}   "
              f"{'as-measured':>11} {'as-absent':>9}")
        for arm in ("bare", "panel", "panel_rule"):
            d = det["by_arm"].get(arm)
            if not d:
                continue
            print(f"{arm:12} {d['n_claims']:>7} {d['overclaimed_k']:>12} "
                  f"{pct(d['overclaimed_k'], d['n_claims']):>26}   "
                  f"{d['as_measured_not_enriched']:>11} {d['as_not_measured']:>9}")
        print("\n  The panel arms hand the model all 280 names under 'the genes measured")
        print("  in this experiment', which is the sentence these claims are agreeing with.")

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
