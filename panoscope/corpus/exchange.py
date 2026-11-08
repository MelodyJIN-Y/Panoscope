"""The corpus interchange format: a brief run in the biologist's own LLM client (ARCH §6.5).

The client's file is evidence of what it searched and read, nothing more. Code resolves
every identifier and fetches its text, checks every span verbatim with the in-app
build's verifier, recomputes rung, modality, species and stance, and writes a new
version (earlier versions kept). Format: ``format.schema.json`` beside this file.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Final

from panoscope.corpus import build as B
from panoscope.corpus import vocab
from panoscope.corpus.channels import EPMC, _record, _search, full_text, matcher
from panoscope.corpus.read import verified, verified_setting
from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, Disposition, QueryRun, QueryStatus, SourceRecord, corpus_path,
    read_version, save,
)
from panoscope.corpus.window import gene_terms
from panoscope.reference.sources import SourceUnavailable

SCHEMA: Final = Path(__file__).with_name("format.schema.json")
IN_CLIENT: Final = "built in user client"
_QUERY: Final = {"pmid": "EXT_ID:{} AND SRC:MED", "pmcid": "PMCID:{}", "doi": 'DOI:"{}"'}

Resolver = Callable[[str, str], "SourceRecord | None"]


def parse(path: Path | str) -> list[dict[str, Any]]:
    """The file's rows, each valid against its kind; any error refuses the whole file, by line."""
    from jsonschema import Draft202012Validator

    defs = json.loads(SCHEMA.read_text(encoding="utf-8"))["$defs"]
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[str] = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {n}: not JSON ({exc.msg})")
            continue
        kind = row.get("kind") if isinstance(row, dict) else None
        if kind not in defs:
            errors.append(f"line {n}: kind must be one of {sorted(defs)}")
            continue
        v = Draft202012Validator({"$ref": f"#/$defs/{kind}", "$defs": defs},
                                  format_checker=Draft202012Validator.FORMAT_CHECKER)
        errors += [f"line {n}: {'/'.join(map(str, e.path)) or kind}: {e.message}"
                   for e in v.iter_errors(row)]
        rows.append((n, row))
    if [r["kind"] for _, r in rows[:1]] != ["header"] or \
            sum(r["kind"] == "header" for _, r in rows) != 1:
        errors.append("line 1: the first line must be the file's only header")
    papers = {r.get("id") for _, r in rows if r["kind"] == "paper"}
    errors += [f"line {n}: paper {r.get('paper')!r} has no paper line" for n, r in rows
               if r["kind"] in ("statement", "setting") and r.get("paper") not in papers]
    if errors:
        raise CorpusUnavailable("import refused, nothing written:\n" + "\n".join(errors))
    return [r for _, r in rows]


def resolve_id(pid: str, retrieved: str) -> SourceRecord | None:
    """The Europe PMC record ``pid`` names; None when no record is that id."""
    scheme, _, v = pid.partition(":")
    _, hits = _search(_QUERY[scheme].format(v), pubmed=False, cap=5, probe=False)
    same = {"pmid": lambda r: r.uri == pid, "pmcid": lambda r: r.pmcid.upper() == v.upper(),
            "doi": lambda r: r.doi.casefold() == v.casefold()}[scheme]
    return next((r for h in hits if (r := _record(h, retrieved)) and same(r)), None)


def _archive(p: Path) -> None:
    """Copy the store at ``p`` to versions/<hash>.sqlite, so no version is overwritten (A12)."""
    try:
        v = read_version(p)
    except CorpusUnavailable:
        return
    if v is not None:
        (p.parent / "versions").mkdir(exist_ok=True)
        shutil.copy2(p, p.parent / "versions" / f"{v.content_hash}.sqlite")


def import_file(study: Any, path: Path | str, *, today: date,
                resolver: Resolver | None = None) -> tuple[Corpus, list[str]]:
    """Import a client-built file as the study's new corpus version; (corpus, receipt lines)."""
    resolver = resolver or resolve_id
    rows = parse(path)
    head = rows[0]
    if head["study_id"] != study.id:
        raise CorpusUnavailable(f"import refused: the file is for {head['study_id']!r}, "
                                f"not {study.id!r}")
    ctx = B.context_for(study)
    tissue = B.tissue_terms(study, ctx)
    current = B.brief_hash(study)
    stamp = today.isoformat()
    generator = (f"{head['model']} ({head['provider']}) in {head['client']}, {IN_CLIENT}, "
                 f"run {head['run_date']}"
                 + (f"; note: {head['operator_note']}" if head.get("operator_note") else ""))

    held: dict[str, SourceRecord] = {}
    unresolved: list[str] = []
    for r in (r for r in rows if r["kind"] == "paper"):
        try:
            rec = resolver(r["id"], stamp)
            why = "no record has this id"
        except (SourceUnavailable, KeyError, ValueError) as exc:
            rec, why = None, f"lookup failed ({exc})"
        if rec is None:
            unresolved.append(f"{r['id']}: {why}")
        else:
            held[r["id"]] = rec
    fetches = []                        # full text only for papers something cites
    for pid in dict.fromkeys(r["paper"] for r in rows if r["kind"] in ("statement", "setting")):
        rec = held.get(pid)
        if rec and rec.open_access and rec.pmcid and not rec.full_text:
            held[pid] = rec = replace(rec, full_text=full_text(rec))
            fetches.append(QueryRun(f"{EPMC}{rec.pmcid}/fullTextXML", "fulltext", "", total=1,
                                    retrieved=int(bool(rec.full_text)),
                                    status=QueryStatus.OK if rec.full_text else QueryStatus.FAILED))
    settings, set_rejected = [], []
    for r in (r for r in rows if r["kind"] == "setting"):
        rec = held.get(r["paper"])
        got = verified_setting(rec, r["field"], r["span"]) if rec and not rec.retracted else None
        if got is None:
            set_rejected.append(f"{r['paper']} {r['field']}: " + (
                "paper retracted or unresolved" if not rec or rec.retracted else
                "not verbatim" + ("" if rec.full_text else " in title or abstract; full text "
                                  "unavailable")))
        else:
            settings.append(got)
    corpus = Corpus(records=tuple({r.uri: r for r in held.values()}.values()),
                    settings=tuple(settings))

    aliases: dict[str, Any] = {}
    statements, rejected = [], []
    for s in (s for s in rows if s["kind"] == "statement"):
        g = s["gene"]
        rec = held.get(s["paper"])
        if rec is None or rec.retracted:                 # code's check, as in-app
            rejected.append(f"{s['paper']} {g}: paper {'retracted' if rec else 'unresolved'}")
            continue
        if s["section"] == "full_text" and not rec.full_text:
            rejected.append(f"{s['paper']} {g}: full text unavailable (not open access, or "
                            "its fetch failed; import again to retry)")
            continue
        if g not in aliases:
            try:
                h = vocab.hgnc(g)
            except SourceUnavailable:
                h = None                    # symbol only: fewer spans pass, never more
            aliases[g] = matcher(gene_terms(g, h))
        got = verified(ctx, corpus, rec, g, aliases[g], s["span"], s["cell_type"],
                       s["predicate"], s.get("comparator", ""), s["section"], 0, None,
                       tissue, generator)
        if isinstance(got, str):
            rejected.append(f"{s['paper']} {g}: {got}")
        else:
            statements.append(got)

    brief_ok = head["brief_hash"] == current
    queries = [QueryRun(f"brief {head['brief_hash']} against current {current}", "import:brief",
                        "", status=QueryStatus.OK if brief_ok else QueryStatus.DEGRADED),
               *fetches]
    for q in (q for q in rows if q["kind"] == "query"):
        status = QueryStatus(q["status"].lower())
        if status is QueryStatus.OK and q["retrieved"] < q["total"]:
            status = QueryStatus.TRUNCATED          # the client's OK is recomputed too
        queries.append(QueryRun(q["query"], f"client:{q['source']}", q["gene"],
                                q.get("cell_type", ""), q["total"], q["retrieved"], status))
    ledger = {held[r["id"]].uri: "RETRACTED" if held[r["id"]].retracted else r["disposition"]
              for r in rows if r["kind"] == "paper" and r["id"] in held}
    corpus = replace(corpus, statements=tuple(statements), queries=tuple(queries),
                     ledger=tuple(Disposition(i, d) for i, d in ledger.items()), built=stamp)
    corpus = replace(corpus, version=B.version_for(
        study, ctx, corpus, generator, f"{IN_CLIENT}: no Panoscope model calls",
        date.fromisoformat(head["run_date"])))

    p = corpus_path(study.id)
    _archive(p)
    save(corpus, p)
    _archive(p)
    rungs = Counter(s.rung or "beyond the ladder" for s in statements)
    return corpus, [
        f"accepted {len(statements)} statement(s), rejected {len(rejected)}",
        *(f"  rejected {x}" for x in rejected),
        f"settings: accepted {len(settings)}, rejected {len(set_rejected)}",
        *(f"  rejected {x}" for x in set_rejected),
        f"unresolved id(s): {len(unresolved)}", *(f"  {x}" for x in unresolved),
        "rungs: " + (", ".join(f"{k} {n}" for k, n in sorted(rungs.items())) or "none"),
        "brief matches the study's current brief" if brief_ok else
        f"WARNING: stale brief {head['brief_hash']} (current {current}); recorded in the log",
        f"version {corpus.version.content_hash}, {IN_CLIENT} ({generator})",
    ]
