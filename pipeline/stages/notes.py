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
    # citation-selection meta the model sometimes emits instead of the note
    "the citation",
    "this citation",
    "citation resolves",
    "strongest fit",
    "directly relevant",
    "directly addresses",
    "is the strongest",
    "this paper",
    "the paper",
    "the reference",
    "i chose",
    "i selected",
    "resolves cleanly",
)


def _strip_dashes(text: str) -> str:
    return (
        text.replace(" — ", ", ")
        .replace(" – ", ", ")
        .replace("—", ", ")
        .replace("–", ", ")
    )


def _balance_parens(t: str) -> str:
    """Drop a trailing UNMATCHED ``(`` fragment the word cap can leave behind.

    Capping at ``_MAX_WORDS`` can slice mid-parenthetical (e.g. ``…proteases
    (CPA3``), which reads as broken once the full summary is shown. When there is
    an unclosed ``(``, cut from the last one to the end. Never touches balanced
    parentheses.
    """
    if t.count("(") <= t.count(")"):
        return t
    return t[: t.rfind("(")].rstrip(" ,;:")


def _shorten(text: str, max_words: int = _MAX_WORDS) -> str:
    """Crisp one-clause summary: drop inline PMID + em dashes + trailing author,
    keep the first sentence, cap length, and never end on a dangling '('."""
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
    t = _balance_parens(t)
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


def _resolve_note(model_text: str, cite: Any, fallback_summary: str):
    """Turn the model's answer + its top citation into (summary, pmid, citation, used_fallback).

    Confident floor: if the model's text is unusable (empty or meta-commentary),
    use the deterministic fallback clause AND DROP the citation — a fallback clause
    is not supported by any paper, so stapling a real PMID to it would be a
    mismatched citation (worse than none). A citation is kept ONLY when the model's
    own prose survives as the summary.
    """
    summary = _shorten(model_text)
    used_fallback = (not summary) or _looks_meta(summary)
    if used_fallback:
        summary = fallback_summary
        cite = None
    citation = (
        {"pmid": cite.pmid, "title": cite.title, "authors": cite.authors, "year": cite.year}
        if cite
        else None
    )
    return summary, (cite.pmid if cite else None), citation, used_fallback


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
        fb = _fallback_summary(verdict.cell_type, list(verdict.key_markers))
        summary, pmid, citation, _ = _resolve_note(resp.text, cite, fb)
        notes[cluster] = {
            "cluster": cluster,
            "cell_type": verdict.cell_type,
            "summary": summary,
            "pmid": pmid,
            "citation": citation,
            "verify": bool(resp.verify),
        }
        _save(out_path, notes)
        print(f"  {cluster} {verdict.cell_type}: PMID {notes[cluster]['pmid'] or 'no-cite'}", flush=True)

    _save(out_path, notes)
    return notes


# --------------------------------------------------------------------------- #
# Per-marker biology notes (Output 4) — the skill reading each gene's evidence.
# --------------------------------------------------------------------------- #
def _gene_fallback(gene: str, cell_type: str, is_canonical: bool) -> str:
    """Clean deterministic note when the model's text is unusable (grounded role only)."""
    ct = cell_type.replace("_", " ")
    role = "a canonical marker" if is_canonical else "a supporting marker"
    return f"{gene} is {role} for {ct} in this cluster."


def _gene_query(gene: str, cell_type: str, cluster: str, ev: Any) -> str:
    """The skill's Output-4 note for THIS gene, grounded in its Tier-A evidence.

    The skill itself is in the agent's system prompt; here we keep the ASK concise
    (a chatty "apply Output 4" instruction makes the model narrate its citation
    choice instead of writing the note) and inject only the one evidence-driven
    caveat the skill's specificity rule needs, when the numbers warrant it.
    """
    ct = cell_type.replace("_", " ")
    caveat = ""
    if "localizes better with another cluster" in ev.caveats:
        caveat = (
            f" Because jazzPanda shows {gene}'s transcripts localize better with "
            f"another cluster here, add a brief specificity caveat that it also marks "
            f"another lineage."
        )
    elif "spatial pattern not unique" in ev.caveats:
        caveat = f" Add a brief caveat that {gene}'s spatial pattern is not unique."
    return (
        f"In 16 words or fewer, state {gene}'s core biological role and its relevance "
        f"to {ct} identity.{caveat} Reply with ONLY the biology clause: do not mention "
        f"the literature search, the citation, or your reasoning, no preamble, no "
        f"statistics, no em dash. Then cite exactly one real PubMed paper."
    )


def run_gene_notes(
    dataset_id: str = cfg.DATASET_ID, root: Optional[Path] = None
) -> dict[str, Any]:
    """Generate (resumable) a skill-grounded biology note for EVERY assigned marker.

    Iterates each cluster's Tier-A evidence (from ``agent.verdict``); for each gene
    asks the agent (which carries SKILL.md) to produce the Output-4 note grounded
    in that gene's evidence, then records the evaluation fields alongside the note.
    Confident floor: one real live PMID or none; deterministic role-only fallback
    when the model's text is unusable.
    """
    out_path = paths.interp_dir(dataset_id, root) / "gene_notes.json"
    notes = _load(out_path)

    for cluster in cfg.CLUSTER_ORDER:
        verdict = agent_verdict.verdict_for_cluster(cluster)
        notes.setdefault(cluster, {})
        for ev in verdict.evidence:
            gene = ev.gene
            if gene in notes[cluster] and notes[cluster][gene].get("summary"):
                continue
            try:
                resp = agent_loop.chat(
                    _gene_query(gene, verdict.cell_type, cluster, ev), cluster=cluster
                )
            except Exception as exc:  # noqa: BLE001 - keep going; save what we have
                print(f"  ERROR {cluster}/{gene}: {exc}", flush=True)
                continue
            cite = resp.citations[0] if resp.citations else None
            fb = _gene_fallback(gene, verdict.cell_type, ev.is_canonical)
            summary, pmid, citation, _ = _resolve_note(resp.text, cite, fb)
            notes[cluster][gene] = {
                "gene": gene,
                "cluster": cluster,
                "cell_type": verdict.cell_type,
                # Tier A — the skill's per-gene evaluation (jazzPanda + panel).
                "glm_coef": ev.glm_coef,
                "pearson": ev.pearson,
                "max_gc_corr": ev.max_gc_corr,
                "max_gg_corr": ev.max_gg_corr,
                "within_cluster_pctile": ev.within_cluster_pctile,
                "is_canonical": ev.is_canonical,
                "is_on_panel": ev.is_on_panel,
                "role": ev.role,
                "caveats": list(ev.caveats),
                # Tier B — the skill's Output-4 biology note (live-cited).
                "summary": summary,
                "pmid": pmid,
                "citation": citation,
                "caveat_flagged": "localizes better with another cluster" in ev.caveats,
                "verify": bool(resp.verify),
            }
            _save(out_path, notes)
            pm = notes[cluster][gene]["pmid"] or "no-cite"
            print(f"  {cluster}/{gene}: PMID {pm}", flush=True)

    _save(out_path, notes)
    return notes


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate skill-grounded notes (live).")
    ap.add_argument("--dataset", default=cfg.DATASET_ID)
    ap.add_argument("--celltype", action="store_true", help="cell-type notes")
    ap.add_argument("--genes", action="store_true", help="per-marker biology notes")
    args = ap.parse_args()
    if not args.celltype and not args.genes:
        args.celltype = args.genes = True  # default: both
    if args.celltype:
        ct = run_celltype_notes(args.dataset)
        print(f"[notes] {len(ct)} cell-type notes ({sum(1 for v in ct.values() if v.get('pmid'))} cited)")
    if args.genes:
        gn = run_gene_notes(args.dataset)
        n = sum(len(v) for v in gn.values())
        c = sum(1 for v in gn.values() for x in v.values() if x.get("pmid"))
        print(f"[notes] {n} gene notes ({c} cited)")
