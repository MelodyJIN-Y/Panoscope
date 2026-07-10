#!/usr/bin/env python
"""Phase-0 spike: classical over-representation (ORA) of each cluster's jazzPanda
markers against MSigDB Hallmark, with the 280-gene panel as the background.

READ-ONLY, stdlib only (math.comb — no scipy/gseapy), prints a table; writes
nothing. Its job is to let us EYEBALL whether classical ORA on this targeted panel
is (a) biologically sensible and (b) honest about panel coverage, before we build
the full enrichment slice. The panel-coverage gate below is the confident floor:
a set is "enriched" only if q<0.05 AND overlap>=3 AND panel_hits>=3 AND the cluster
is not fragile (<=2 markers). Everything else is untestable and never surfaced.

Usage:
    .venv/bin/python scripts/enrichment_spike.py [path/to/hallmark.gmt]

The GMT defaults to data/genesets/hallmark.gmt, else the env var HALLMARK_GMT.
"""

from __future__ import annotations

import os
import sys
from math import comb
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agent import config as cfg  # noqa: E402
from agent import data  # noqa: E402

# The report gate (the confident floor) — mirrors the plan's non-negotiable bar.
MIN_PANEL_HITS = 3      # K: a set with <3 panel genes is untestable
MIN_OVERLAP = 3         # k: fewer than 3 overlapping genes is not defensible
Q_THRESHOLD = 0.05
FRAGILE_MARKER_COUNT = 2  # mirrors agent/verdict.py — c8/c9 are fragile


def _gmt_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    default = cfg.DATA_DIR_PATH / "genesets" / "hallmark.gmt"
    if default.exists():
        return default
    env = os.getenv("HALLMARK_GMT")
    if env:
        return Path(env)
    raise FileNotFoundError(
        "No Hallmark GMT found. Pass a path, set HALLMARK_GMT, or place it at "
        f"{default}."
    )


def _normalize_set_name(raw: str) -> str:
    """Enrichr names ('TNF-alpha Signaling via NF-kB') -> HALLMARK_ style token."""
    up = raw.strip().upper()
    for ch in " -/,()":
        up = up.replace(ch, "_")
    while "__" in up:
        up = up.replace("__", "_")
    up = up.strip("_")
    return up if up.startswith("HALLMARK_") else f"HALLMARK_{up}"


def _load_gmt(path: Path) -> dict[str, set[str]]:
    """Parse a GMT: 'set<tab>desc<tab>gene<tab>gene...'. Handles Enrichr's blank
    description field and 'gene,weight' tokens."""
    sets: dict[str, set[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        name = _normalize_set_name(parts[0])
        genes = {
            tok.split(",")[0].strip().upper()
            for tok in parts[1:]
            if tok.strip() and not tok.strip().replace(".", "").isdigit()
        }
        genes.discard("")
        if genes:
            sets[name] = genes
    return sets


def _hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k), X ~ Hypergeometric(N population, K successes, n drawn). Exact."""
    if k <= 0:
        return 1.0
    hi = min(K, n)
    if k > hi:
        return 0.0
    denom = comb(N, n)
    total = sum(comb(K, x) * comb(N - K, n - x) for x in range(k, hi + 1))
    return total / denom


def _bh_fdr(pvals: list[float]) -> list[float]:
    """Benjamini-Hochberg q-values for a family of p-values (order preserved)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        val = min(prev, pvals[i] * m / rank)
        q[i] = val
        prev = val
    return q


def main() -> None:
    gmt_path = _gmt_path()
    hallmark = _load_gmt(gmt_path)
    panel = frozenset(g.upper() for g in data.load_panel()["gene"].astype(str))
    N = len(panel)

    print(f"# Enrichment spike — ORA (hypergeometric), panel background N={N}")
    print(f"# Hallmark GMT: {gmt_path.name}  ({len(hallmark)} sets)")
    print(f"# Gate: q<{Q_THRESHOLD} AND overlap>={MIN_OVERLAP} AND panel_hits>={MIN_PANEL_HITS} AND not fragile\n")

    # Pre-scope every set to the panel once (K + panel members).
    scoped = {name: (genes & panel) for name, genes in hallmark.items()}

    total_pass = 0
    for cluster in cfg.CLUSTER_ORDER:
        markers = data.get_cluster_markers(cluster)
        marker_genes = {g.upper() for g in markers["gene"].astype(str)}
        n = len(marker_genes)
        cell_type = cfg.CLUSTER_KEY[cluster]["cell_type"]
        fragile = n <= FRAGILE_MARKER_COUNT

        rows = []
        for name, panel_set in scoped.items():
            K = len(panel_set)
            overlap = marker_genes & panel_set
            k = len(overlap)
            p = _hypergeom_sf(k, N, K, n)
            rows.append([name, len(hallmark[name]), K, n, k, sorted(overlap), p])
        qs = _bh_fdr([r[6] for r in rows])
        for r, q in zip(rows, qs):
            r.append(q)

        # Gate + sort enriched by q.
        enriched = [
            r for r in rows
            if (not fragile) and r[7] < Q_THRESHOLD and r[4] >= MIN_OVERLAP and r[2] >= MIN_PANEL_HITS
        ]
        enriched.sort(key=lambda r: r[7])
        testable = sum(1 for r in rows if r[2] >= MIN_PANEL_HITS)
        total_pass += len(enriched)

        flag = "  [FRAGILE -> all untestable]" if fragile else ""
        print(f"== {cluster}  {cell_type}  (n={n} markers, {testable}/{len(rows)} sets testable, "
              f"{len(enriched)} pass){flag}")
        if not enriched:
            print("   (no set passes the gate)\n")
            continue
        for name, full, K, n_, k, ov, p, q in enriched[:8]:
            cov = K / full if full else 0.0
            le = ",".join(ov[:6]) + ("…" if len(ov) > 6 else "")
            print(f"   {name:<34} q={q:.1e}  cov={K}/{full}({cov:.0%})  k={k}  LE:{le}")
        print()

    print(f"# TOTAL enriched sets across all clusters (post-gate): {total_pass}")


if __name__ == "__main__":
    main()
