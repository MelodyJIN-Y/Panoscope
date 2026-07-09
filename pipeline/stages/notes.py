"""Stage 6 (notes) — grounded per-cluster cell-type summaries (LIVE PubMed).

For each cluster's annotated cell type, ask the Panoscope agent for one crisp
plain-language summary of what that cell type is in this tissue and why its
drivers mark it, WITH a real PubMed citation fetched live. The agent's grounding
gate guarantees the citation is real, so these obey the confident floor: thin
literature yields ``pmid: null`` and says so, never a fabricated PMID.

Written to ``interp/celltype_notes.json`` so the Summary table's cell-type
summary column reads it instantly (no live call on page open). Resumable: an
existing cluster note is skipped and the file is rewritten after each cluster.
This is the only [LIVE] stage in slice 2; it degrades honestly and never blocks
the deterministic tree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from agent import config as cfg
from agent import loop as agent_loop
from agent import verdict as agent_verdict

from pipeline import paths

_MAX_WORDS = 16
_INLINE_PMID = re.compile(r"\s*[\(\[]?\s*PMID[:\s]*\d+\s*[\)\]]?", re.IGNORECASE)
# A trailing "..., Surname AB." author fragment the model sometimes appends.
_TRAIL_AUTHOR = re.compile(r",\s*[A-Z][a-z]+ [A-Z]{1,3}\.?\s*$")
# Phrases that mean the model wrote META-commentary (about the connector, the
# literature search, or the verdict) instead of a cell-type description.
_META_MARKERS = (
    "connector",
    "pubmed",
    "no hits",
    "returning no",
    "neither paper",
    "not a strong fit",
    "no clear",
    "is called",
    "glm_coef",
    "pearson",
    "confidence",
    "i could not",
    "i cannot",
    "i was unable",
)


def _strip_dashes(text: str) -> str:
    return (
        text.replace(" — ", ", ")
        .replace(" – ", ", ")
        .replace("—", ", ")
        .replace("–", ", ")
    )


def _shorten(text: str, max_words: int = _MAX_WORDS) -> str:
    """Crisp one-clause summary: drop inline PMID + em dashes + trailing author,
    keep the first sentence, cap length."""
    t = _INLINE_PMID.sub("", _strip_dashes(text.strip())).strip()
    m = re.search(r"[.;]\s", t)
    if m and m.start() < 220:
        t = t[: m.start()]
    t = _TRAIL_AUTHOR.sub("", t).strip()
    words = t.split()
    if len(words) > max_words:
        capped = " ".join(words[:max_words])
        ci = capped.rfind(",")
        t = capped[:ci] if ci > 20 else capped
    t = _TRAIL_AUTHOR.sub("", t.strip()).strip().rstrip(",;:. ")
    return (t + ".") if t else t


def _looks_meta(summary: str) -> bool:
    """True if the summary is model meta-commentary, not a cell-type description."""
    low = summary.lower()
    return any(m in low for m in _META_MARKERS)


def _fallback_summary(cell_type: str, markers: list[str]) -> str:
    """A clean deterministic description when the model's text is unusable.

    Grounded projection only: the cell type (from the key) and its driving markers
    (from jazzPanda). No invented biology — safe under the confident floor.
    """
    ct = cell_type.replace("_", " ")
    drv = ", ".join(markers[:3])
    return f"{ct} cells; driving markers {drv}." if drv else f"{ct} cells."


def _query(cell_type: str, markers: list[str]) -> str:
    ct = cell_type.replace("_", " ")
    drv = ", ".join(markers[:3])
    return (
        f"Describe, in 14 words or fewer, what a {ct} cell is in human breast "
        f"tissue and its role, and why markers like {drv} mark it. Write ONE plain "
        f"factual clause about the CELL TYPE itself. Do NOT mention the cluster id, "
        f"the confidence, any statistic, or whether literature exists; never write "
        f"about the search or the connector. If you find a supporting paper, cite "
        f"one real PubMed PMID after the clause; if not, give the clause anyway with "
        f"no citation."
    )


def _load(out_path: Path) -> dict[str, Any]:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save(out_path: Path, notes: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_celltype_notes(
    dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None
) -> dict[str, Any]:
    """Generate (resumable) one cited cell-type note per cluster into the tree."""
    out_path = paths.interp_dir(dataset_id, root) / "celltype_notes.json"
    notes = _load(out_path)

    for cluster in cfg.CLUSTER_ORDER:
        if cluster in notes and notes[cluster].get("summary"):
            continue
        verdict = agent_verdict.verdict_for_cluster(cluster)
        try:
            resp = agent_loop.chat(
                _query(verdict.cell_type, list(verdict.key_markers)), cluster=cluster
            )
        except Exception as exc:  # noqa: BLE001 - keep going; save what we have
            print(f"  ERROR {cluster}: {exc}", flush=True)
            continue
        cite = resp.citations[0] if resp.citations else None
        summary = _shorten(resp.text)
        if not summary or _looks_meta(summary):
            summary = _fallback_summary(verdict.cell_type, list(verdict.key_markers))
        notes[cluster] = {
            "cluster": cluster,
            "cell_type": verdict.cell_type,
            "summary": summary,
            "pmid": cite.pmid if cite else None,
            "citation": (
                {"pmid": cite.pmid, "title": cite.title, "authors": cite.authors, "year": cite.year}
                if cite
                else None
            ),
            "verify": bool(resp.verify),
        }
        _save(out_path, notes)
        print(f"  {cluster} {verdict.cell_type}: PMID {notes[cluster]['pmid'] or 'no-cite'}", flush=True)

    _save(out_path, notes)
    return notes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate per-cluster cell-type notes (live).")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    args = ap.parse_args()
    result = run_celltype_notes(args.dataset)
    n_cited = sum(1 for v in result.values() if v.get("pmid"))
    print(f"[notes] {len(result)} cell-type notes ({n_cited} cited)")
