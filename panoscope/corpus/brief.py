"""The brief a biologist runs in their own LLM client (ARCH §6.5): context, vocabulary, skill.

Every term is computed here from real sources (HGNC, MeSH) and the study; the client only
searches, screens and quotes. ``exchange.import_file`` checks what comes back.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Final

from panoscope.corpus import build as B
from panoscope.corpus import vocab
from panoscope.corpus.channels import M_CELL, cell_terms
from panoscope.corpus.exchange import SCHEMA
from panoscope.corpus.store import corpus_path
from panoscope.corpus.window import gene_terms
from panoscope.reference.sources import SourceUnavailable

SKILL: Final = Path(__file__).resolve().parents[1] / "skills" / "corpus-building" / "SKILL.md"
BLOCK: Final = 40           # terms per tissue-block chunk, for a refused long query
_CD = re.compile(r"\b(CD\d+[A-Za-z]?) (?=\w)")


def spellings(term: str) -> tuple[str, ...]:
    """A cell-type term plus its marker-plus ("CD8+ T cells", "CD8+T cells") and singular forms."""
    out = [term]
    if _CD.search(term):
        out += [_CD.sub(r"\1+ ", term), _CD.sub(r"\1+", term)]
    out += [t[:-1] for t in out if len(t) > 4 and t.endswith("s")]
    return tuple(dict.fromkeys(out))


def organ_disease_terms(label: str) -> tuple[str, ...]:
    """Natural-order entry terms of MeSH "<label> Diseases" and its descendants two levels down."""
    # ponytail: the whole subtree to depth 2 (kidney ~250 terms); an intake untick list
    # is the upgrade when a biologist wants it narrower.
    es, su = vocab.EUTILS + "esearch.fcgi", vocab.EUTILS + "esummary.fcgi"
    ids = vocab._get(es, {"db": "mesh", "term": f'"{label} Diseases"[MH]'}
                     )["esearchresult"]["idlist"]
    if not ids:
        return ()
    roots = [x["treenum"] for x in
             vocab._get(su, {"db": "mesh", "id": ids[0]})["result"][ids[0]]["ds_idxlinks"]]
    sub = vocab._get(es, {"db": "mesh", "retmax": "500",
                          "term": " OR ".join(f"{t}*[TN]" for t in roots)}
                     )["esearchresult"]["idlist"]
    out: dict[str, str] = {}
    for i in range(0, len(sub), 150):
        res = vocab._get(su, {"db": "mesh", "id": ",".join(sub[i:i + 150])})["result"]
        for uid in res.get("uids", []):
            r = res[uid]
            if any(t.count(".") - root.count(".") <= 2 for x in r.get("ds_idxlinks", [])
                   for root in roots if (t := x["treenum"]).startswith(root)):
                for t in r.get("ds_meshterms", []):
                    if "," not in t and "(" not in t:
                        out.setdefault(t.casefold().removesuffix("s"), t)
    return tuple(out.values())


def _gene(symbol: str) -> dict[str, Any]:
    """One gene's aliases; a failed HGNC or MeSH lookup is named, never silently empty."""
    got: list[Any] = []
    failed = []
    for look in (vocab.hgnc, vocab.mesh_terms):
        try:
            got.append(look(symbol))
        except SourceUnavailable as exc:
            got.append(None)
            failed.append(str(exc))
    terms = gene_terms(symbol, got[0], got[1] or ())
    status = f"FAILED: {'; '.join(failed)}" if failed else "ok"
    return {"symbol": symbol, "aliases_status": status,
            "aliases": [{"text": t.text, "origin": t.origin, "restricted": t.restricted}
                        for t in terms]}


def context(study: Any, genes: tuple[str, ...] = ()) -> dict[str, Any]:
    """study_context.json: the window contract the client searches within (D2 scope by default)."""
    ctx = B.context_for(study)
    tissue = list(B.tissue_terms(study, ctx))
    label = study.manifest.get("tissue_label") or ""
    try:
        organ, status = (list(organ_disease_terms(label)), "ok") if label else \
            ([], "no tissue label at intake")
    except (SourceUnavailable, KeyError) as exc:
        organ, status = [], f"FAILED: {exc}"
    seen = {x.casefold() for x in tissue}
    t = tissue + [o for o in organ if o.casefold() not in seen]
    bg = study.manifest.get("background") or {}
    return {
        "study_id": study.id, "brief_hash": B.brief_hash(study),
        "tissue": {"id": ctx.tissue_id, "label": label, "terms": tissue,
                   "blocks": [t[i:i + BLOCK] for i in range(0, len(t), BLOCK)]},
        "organ_disease_terms": {"status": status, "source": "MeSH", "terms": organ},
        "disease_state": {"value": ctx.disease_state, "window": "W0*" if ctx.w0_star else "W0",
                          "qualifiers": dict(ctx.disease_qualifiers)},
        "platform": ctx.platform, "modality_class": ctx.modality_class,
        "release_paper": {"pmid": ctx.release_pmid, "status": ctx.release_status},
        "papers_of_interest": [str(c.get("value", "")) for c in bg.get("papers_of_interest", ())],
        "cell_types": [{"label": e.label, "ontology_id": e.ontology_id,
                        "terms": list(dict.fromkeys(s for c in cell_terms(e)
                                                    for s in spellings(c)))}
                       for e in ctx.cell_types],
        "assay_terms": list(M_CELL),
        "genes": [_gene(g) for g in
                  genes or B.scope(study, tuple(e.label for e in ctx.cell_types))],
    }


def _md(c: dict[str, Any]) -> str:
    """The study section that heads BRIEF.md: everything the client may search with."""
    def q(xs: Any) -> str:
        return ", ".join(f'"{x}"' for x in xs) or "none"

    d, r = c["disease_state"], c["release_paper"]
    return "\n".join([
        f"# Literature brief for study {c['study_id']}", "",
        f"Copy into your output header: study_id `{c['study_id']}`, "
        f"brief_hash `{c['brief_hash']}`.",
        "", "## This study (precomputed by Panoscope; use exactly these terms)", "",
        f"- Tissue {c['tissue']['label'] or 'unlabelled'} ({c['tissue']['id']}): "
        f"{q(c['tissue']['terms'])}",
        f"- Organ-disease terms (MeSH, {c['organ_disease_terms']['status']}): "
        f"{q(c['organ_disease_terms']['terms'])}",
        f"- Disease state: {d['value'] or 'not declared'} (window {d['window']})"
        + "".join(f"; {k} {v}" for k, v in d["qualifiers"].items()),
        f"- Platform {c['platform']} ({c['modality_class']})",
        f"- Release paper: {('pmid ' + r['pmid']) if r['pmid'] else 'none'} ({r['status']})",
        f"- Papers of interest: {', '.join(c['papers_of_interest']) or 'none'}",
        "", "### Cell-type frame (a statement's `cell_type` is the label)", "",
        *(f"- **{e['label']}**: {q(e['terms'])}" for e in c["cell_types"]),
        "", "### Genes in scope and their aliases (restricted ones marked †)", "",
        *(f"- **{g['symbol']}** ({g['aliases_status']}): "
          + ", ".join(f'"{a["text"]}"' + ("†" if a["restricted"] else "") for a in g["aliases"])
          for g in c["genes"]),
        "", "The tissue block in chunks and the assay block are in study_context.json.",
        "", "---", "", ""])


def write(study: Any, out: Path | None = None, genes: tuple[str, ...] = ()) -> Path:
    """Write BRIEF.md, study_context.json and format.schema.json to ``out``; returns the folder."""
    out = Path(out) if out else corpus_path(study.id).parent / "brief"
    out.mkdir(parents=True, exist_ok=True)
    c = context(study, genes)
    skill = SKILL.read_text(encoding="utf-8").split("---\n", 2)[2].lstrip()
    (out / "BRIEF.md").write_text(_md(c) + skill, encoding="utf-8")
    (out / "study_context.json").write_text(json.dumps(c, indent=1), encoding="utf-8")
    shutil.copy2(SCHEMA, out / "format.schema.json")
    return out
