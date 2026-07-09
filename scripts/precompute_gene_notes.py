"""Precompute grounded per-gene biology notes for the evidence table (ONE-TIME).

For each cluster's key marker genes, ask the Panoscope agent for a short note —
the gene's general role, its relevance to that cluster's cell type, and any
specificity caveat — WITH a real PubMed citation fetched live. The agent's
grounding gate guarantees the citation is real (never written from memory), so
these notes obey the confident-floor: every biological claim traces to a paper.

The notes are cached so the evidence-table column reads them instantly and the
demo never makes a live literature call on cluster open. Resumable: existing
(cluster, gene) notes are skipped, and the file is written after every gene.

Output: data/gene_notes/notes.json
  { "<cluster>": { "<GENE>": {gene, cluster, cell_type, summary, pmid,
                              citation:{pmid,title,authors,year}, verify} } }

Run (needs the PubMed MCP + the API key, like the app):
  .venv/bin/python scripts/precompute_gene_notes.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent import loop as agent_loop  # noqa: E402
from agent import verdict as agent_verdict  # noqa: E402
from agent.config import CLUSTER_ORDER  # noqa: E402
from agent.types import ClusterVerdict  # noqa: E402

_OUT = _ROOT / "data" / "gene_notes" / "notes.json"
_MAX_GENES_PER_CLUSTER = 8  # cover the markers the evidence table shows per cluster


def _shown_genes(verdict: ClusterVerdict) -> list[str]:
    """The marker genes the evidence table shows for a cluster: every canonical
    marker plus the strongest non-canonical supporters up to the cap (mirrors
    ui.evidence_table._rows_to_show), in the verdict's glm_coef order."""
    canonical = [e for e in verdict.evidence if e.is_canonical]
    non_canon = [e for e in verdict.evidence if not e.is_canonical]
    budget = max(0, _MAX_GENES_PER_CLUSTER - len(canonical))
    kept = {id(e) for e in canonical} | {id(e) for e in non_canon[:budget]}
    return [e.gene for e in verdict.evidence if id(e) in kept]


def _strip_dashes(text: str) -> str:
    """Never ship an em dash (project style): replace with a comma or a colon."""
    return (
        text.replace(" — ", ", ")
        .replace(" – ", ", ")
        .replace("—", ", ")
        .replace("–", ", ")
    )


def _load() -> dict:
    if _OUT.exists():
        try:
            return json.loads(_OUT.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(notes: dict) -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(notes, indent=2, ensure_ascii=False))


def _query(gene: str, cell_type: str, cluster: str) -> str:
    ct = cell_type.replace("_", " ")
    return (
        f"In ONE short sentence (max 18 words), state {gene}'s core biological role "
        f"and its relevance to {ct} identity; flag a specificity caveat only if "
        f"{gene} also clearly marks another lineage. Do NOT mention any numeric "
        f"statistics (glm_coef, pearson, correlations) since those are shown "
        f"separately. Do NOT use em dashes. Cite exactly one real PubMed paper. "
        f"Plain prose, no preamble."
    )


def _note_from_response(gene: str, cluster: str, cell_type: str, resp) -> dict:
    cite = resp.citations[0] if resp.citations else None
    return {
        "gene": gene,
        "cluster": cluster,
        "cell_type": cell_type,
        "summary": _strip_dashes(resp.text.strip()),
        "pmid": cite.pmid if cite else None,
        "citation": (
            {
                "pmid": cite.pmid,
                "title": cite.title,
                "authors": cite.authors,
                "year": cite.year,
            }
            if cite
            else None
        ),
        "verify": bool(resp.verify),
        "used_fallback": bool(getattr(resp, "used_fallback", False)),
    }


def main() -> None:
    notes = _load()
    t0 = time.time()
    total = 0
    for cluster in CLUSTER_ORDER:
        verdict = agent_verdict.verdict_for_cluster(cluster)
        cell_type = verdict.cell_type
        genes = _shown_genes(verdict)
        notes.setdefault(cluster, {})
        for gene in genes:
            total += 1
            if gene in notes[cluster] and notes[cluster][gene].get("summary"):
                print(f"  skip {cluster}/{gene} (cached)", flush=True)
                continue
            try:
                resp = agent_loop.chat(_query(gene, cell_type, cluster), cluster=cluster)
            except Exception as exc:  # noqa: BLE001 - keep going; save what we have
                print(f"  ERROR {cluster}/{gene}: {exc}", flush=True)
                continue
            note = _note_from_response(gene, cluster, cell_type, resp)
            notes[cluster][gene] = note
            _save(notes)  # incremental — resumable if interrupted
            pm = note["pmid"] or "no-cite"
            print(
                f"  {cluster}/{gene}: PMID {pm}"
                f"{' [verify]' if note['verify'] else ''}"
                f"{' [fallback]' if note['used_fallback'] else ''}",
                flush=True,
            )

    _save(notes)
    n_notes = sum(len(v) for v in notes.values())
    n_cited = sum(1 for v in notes.values() for x in v.values() if x.get("pmid"))
    print(
        f"\n[precompute_gene_notes] {n_notes} notes ({n_cited} cited) across "
        f"{len(notes)} clusters in {time.time()-t0:.0f}s -> {_OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
