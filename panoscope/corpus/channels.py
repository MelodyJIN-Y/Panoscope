"""Retrieval channels C0-C5 as code (harness v2 §5 phases 2-3).

Queries, paging, narrowing, dedupe, the prefilter gate, full-text fetch and the
co-occurrence sentence index. Europe PMC and E-utilities through ``vocab._get``
(tool=panoscope, rate-limited, backoff). Checkpointed after every query and fetch.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Final, Iterable

from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, Disposition, HollowRecord, QueryRun, QueryStatus,
    SourceRecord, corpus_path, load, save,
)
from panoscope.corpus.vocab import EUTILS, _get
from panoscope.corpus.window import StudyContext, Term
from panoscope.reference.sources import SourceUnavailable

EPMC: Final = "https://www.ebi.ac.uk/europepmc/webservices/rest/"
CAP: Final = 200
M_CELL: Final = (
    "spatial transcriptomics", "Xenium", "CosMx", "MERFISH", "MERSCOPE", "single-cell",
    "single cell", "scRNA-seq", "snRNA-seq", "in situ hybridization", "RNAscope", "smFISH",
    "immunohistochemistry", "multiplex immunofluorescence", "imaging mass cytometry",
)
_BODY: Final = frozenset({"C0", "C3", "C4:C0", "C4:C3"})   # pass the gate without an alias


# --------------------------------------------------------------------------- #
# Query building
# --------------------------------------------------------------------------- #
def _or(terms: Iterable[str], fields: tuple[str, ...] = ("",), *, pubmed: bool = False) -> str:
    ts = tuple(dict.fromkeys(t.replace('"', "").strip() for t in terms if t.strip()))
    parts = [f'"{t}"[{f}]' if pubmed else (f'{f}:"{t}"' if f else f'"{t}"')
             for t in ts for f in fields]
    return f"({' OR '.join(parts)})" if parts else ""


def compose(gene: str, *, tissue: str = "", cells: str = "", cites: str = "",
            extra: tuple[str, ...] = ()) -> str:
    """AND the blocks; a query with no tissue, cell-type or CITES anchor cannot be built."""
    if not (tissue or cells or cites):
        raise ValueError(f"refused: {gene!r} has no tissue, cell-type or CITES anchor (§5)")
    return " AND ".join(b for b in (cites, gene, tissue, cells, *extra) if b)


def _since(today: date, months: int, pubmed: bool) -> str:
    d = date(today.year - months // 12, today.month, min(today.day, 28))
    return (f'("{d:%Y/%m/%d}"[edat] : "3000"[edat])' if pubmed
            else f"FIRST_PDATE:[{d:%Y-%m-%d} TO 3000-12-31]")


def cell_terms(entry: Any) -> tuple[str, ...]:
    """A frame type's search terms: its label (underscores as spaces) and its synonyms."""
    return tuple(dict.fromkeys((entry.label.replace("_", " "), *(s for s, _ in entry.synonyms))))


def _ladder(gene: str, *, tissue: str = "", cells: tuple[str, ...] = (),
            cell_fields: tuple[str, ...] = ("",), cites: str = "",
            extra: tuple[str, ...] = (), pubmed: bool, today: date) -> list[str]:
    """Base query, then narrowed: + M_cell, then cell terms in the title, then 36 months."""
    c = _or(cells, ("tiab",) if pubmed else cell_fields, pubmed=pubmed)
    m = _or(M_CELL, ("tiab",) if pubmed else ("",), pubmed=pubmed)
    steps = [compose(gene, tissue=tissue, cells=c, cites=cites, extra=extra),
             compose(gene, tissue=tissue, cells=c, cites=cites, extra=(*extra, m))]
    if cells:
        c = _or(cells, ("ti",) if pubmed else ("TITLE",), pubmed=pubmed)
        steps.append(compose(gene, tissue=tissue, cells=c, cites=cites, extra=(*extra, m)))
    steps.append(compose(gene, tissue=tissue, cells=c, cites=cites,
                         extra=(*extra, m, _since(today, 36, pubmed))))
    return steps


# --------------------------------------------------------------------------- #
# Running a query: page to exhaustion or the cap, narrowing when over it
# --------------------------------------------------------------------------- #
def _search(query: str, *, pubmed: bool, cap: int, probe: bool) -> tuple[int, list[dict]]:
    """(total, newest-first results up to ``cap``); ``probe`` stops after one page when over cap."""
    if pubmed:
        r = _get(EUTILS + "esearch.fcgi", {"db": "pubmed", "term": query, "retmax": str(cap),
                                           "sort": "pub_date"})["esearchresult"]
        return int(r["count"]), [{"pmid": i} for i in r.get("idlist", [])]
    out: list[dict] = []
    cursor = "*"
    while True:
        d = _get(EPMC + "searchPOST", {"query": query, "format": "json", "resultType": "core",
                                   "pageSize": "100", "sort": "P_PDATE_D desc",
                                   "cursorMark": cursor})
        total = int(d.get("hitCount", 0))
        page = d.get("resultList", {}).get("result", [])
        out += page
        nxt = d.get("nextCursorMark")
        if (probe and total > cap) or len(out) >= min(total, cap) or not page \
                or not nxt or nxt == cursor:
            return total, out[:cap]
        cursor = nxt


def _uri(hit: dict) -> str:
    if hit.get("pmid"):
        return f"pmid:{hit['pmid']}"
    if hit.get("doi"):
        return f"doi:{hit['doi']}"
    return f"epmc:{hit.get('source', '')}/{hit.get('id', '')}"


_TAG = re.compile(r"<[^>]+>")


def _record(hit: dict, retrieved: str) -> SourceRecord | None:
    """A SourceRecord from one Europe PMC core result; None for a hollow one."""
    if "title" not in hit:
        return None
    pre = [c for c in (hit.get("commentCorrectionList") or {}).get("commentCorrection", [])
           if c.get("type") == "Preprint of" and c.get("source") == "MED" and c.get("id")]
    try:
        return SourceRecord(
            uri=_uri(hit), title=_TAG.sub("", str(hit.get("title") or "")).strip(),
            authors=str(hit.get("authorString") or ""), year=int(hit.get("pubYear") or 0),
            journal=str(((hit.get("journalInfo") or {}).get("journal") or {}).get("title") or ""),
            abstract=_TAG.sub("", str(hit.get("abstractText") or "")).strip(),
            via="Europe PMC", retrieved=retrieved, pmcid=str(hit.get("pmcid") or ""),
            doi=str(hit.get("doi") or ""),
            publication_type=tuple((hit.get("pubTypeList") or {}).get("pubType") or ()),
            licence=str(hit.get("license") or ""), open_access=hit.get("isOpenAccess") == "Y",
            epub=str(hit.get("firstPublicationDate") or ""),
            preprint_of=f"pmid:{pre[0]['id']}" if pre else "")
    except (HollowRecord, ValueError):
        return None


@dataclass
class _Run:
    """The corpus being built and where it is checkpointed."""

    corpus: Corpus
    path: Path
    cap: int
    retrieved: str

    def commit(self, **changes: Any) -> None:
        self.corpus = replace(self.corpus, **changes)
        save(self.corpus, self.path)

    def done(self, query: str, channel: str, gene: str, cell: str) -> QueryRun | None:
        h = QueryRun(query, channel, gene).request_hash
        return next((q for q in self.corpus.queries if q.request_hash == h and q.gene == gene
                     and q.cell_type == cell and q.status is not QueryStatus.FAILED), None)

    def log(self, run: QueryRun, records: Iterable[SourceRecord] = ()) -> QueryRun:
        """Append a row (replacing a FAILED one for the same request) and checkpoint."""
        keep = tuple(q for q in self.corpus.queries
                     if (q.request_hash, q.gene, q.cell_type)
                     != (run.request_hash, run.gene, run.cell_type))
        held = {r.uri for r in self.corpus.records}
        new = tuple({r.uri: r for r in records if r.uri not in held}.values())
        self.commit(queries=(*keep, run), records=(*self.corpus.records, *new))
        return run

    def ladder(self, steps: list[str], channel: str, gene: str, cell: str = "", *,
               pubmed: bool, degraded: bool = False) -> QueryRun:
        """Run the base query, narrowing while over the cap; TRUNCATED past the last step."""
        for i, q in enumerate(steps):
            last = i == len(steps) - 1
            if (prior := self.done(q, channel, gene, cell)) is not None:
                if prior.status is QueryStatus.NARROWED and not last:
                    continue
                return prior
            try:
                total, hits = _search(q, pubmed=pubmed, cap=self.cap, probe=not last)
            except (SourceUnavailable, KeyError, ValueError):
                return self.log(QueryRun(q, channel, gene, cell, status=QueryStatus.FAILED))
            if total > self.cap and not last:
                self.log(QueryRun(q, channel, gene, cell, total=total,
                                  status=QueryStatus.NARROWED))
                continue
            status = (QueryStatus.TRUNCATED if len(hits) < total else
                      QueryStatus.DEGRADED if degraded else QueryStatus.OK)
            recs = [r for h in hits if (r := _record(h, self.retrieved)) is not None]
            return self.log(QueryRun(q, channel, gene, cell, total=total, retrieved=len(hits),
                                     status=status, ids=tuple(_uri(h) for h in hits)), recs)
        raise AssertionError("unreachable: the last step always returns")

    def lookup(self, uris: Iterable[str], channel: str, gene: str) -> None:
        """Metadata for PMIDs that came back without a record (Europe PMC, 50 per call)."""
        held = {r.uri for r in self.corpus.records}
        want = sorted({u for u in uris if u.startswith("pmid:") and u not in held})
        for i in range(0, len(want), 50):
            batch = want[i:i + 50]
            q = "SRC:MED AND (" + " OR ".join(f"EXT_ID:{u[5:]}" for u in batch) + ")"
            try:
                _, hits = _search(q, pubmed=False, cap=len(batch), probe=False)
            except (SourceUnavailable, KeyError, ValueError):
                self.log(QueryRun(q, channel, gene, total=len(batch),
                                  status=QueryStatus.FAILED))
                continue
            recs = [r for h in hits if (r := _record(h, self.retrieved)) and r.uri in batch]
            n = len({r.uri for r in recs})
            self.log(QueryRun(q, channel, gene, total=len(batch), retrieved=n,
                              status=QueryStatus.OK if n == len(batch) else QueryStatus.DEGRADED,
                              ids=tuple(batch) if channel == "C5" else ()), recs)


# --------------------------------------------------------------------------- #
# Matching, sentences, full text
# --------------------------------------------------------------------------- #
def _alt(terms: Iterable[str], loose: bool = True) -> str:
    """Longest first. Loose: a space also matches "+" ("CD8 T cells" ~ "CD8+T cells") and a
    plural matches its singular (the matcher's suffix adds the s back)."""
    def pat(t: str) -> str:
        if loose and len(t) > 4 and t.endswith("s"):
            t = t[:-1]
        return re.escape(t).replace(r"\ ", r"\+?[\s-]?") if loose else re.escape(t)
    return "|".join(pat(t) for t in sorted(set(terms), key=len, reverse=True))


def matcher(aliases: Iterable[Term], cells: Iterable[str] = ()) -> re.Pattern[str]:
    """Word-bounded alias match: restricted aliases case-sensitive, the rest case-folded."""
    al = tuple(aliases)
    loose = _alt([*(a.text for a in al if not a.restricted), *cells])
    strict = _alt((a.text for a in al if a.restricted), loose=False)
    parts = [p for p in (f"(?i:{loose})" if loose else "", strict) if p]
    return re.compile(rf"(?<!\w)(?:{'|'.join(parts) or '(?!)'})(?:e?s)?(?!\w)")


_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])|\n+")


def sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) offsets of each sentence; paragraphs and sentence ends break."""
    # ponytail: "Fig. 2" and "et al." split a sentence; spans stay verbatim either way.
    out, start = [], 0
    for m in _BREAK.finditer(text):
        if text[start:m.start()].strip():
            out.append((start, m.start()))
        start = m.end()
    if text[start:].strip():
        out.append((start, len(text)))
    return out


@dataclass(frozen=True, slots=True)
class Sentence:
    uri: str
    section: str
    start: int
    end: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.uri}#{self.section}:{self.start}"


def cooccurrence(record: SourceRecord, aliases: Iterable[Term],
                 cells: Iterable[str]) -> tuple[Sentence, ...]:
    """Sentences naming an alias AND any frame synonym (every frame type, §5 phase 3)."""
    a, c = matcher(aliases), matcher((), cells)
    return tuple(Sentence(record.uri, sec, s, e, record.text(sec)[s:e])
                 for sec in ("abstract", "full_text")
                 for s, e in sentences(record.text(sec))
                 if a.search(record.text(sec)[s:e]) and c.search(record.text(sec)[s:e]))


def _jats(xml: str) -> str:
    """Body text of a JATS article: section headings as "## ", paragraphs, table rows."""
    body = ET.fromstring(xml).find("body")
    out: list[str] = []

    def txt(el: ET.Element) -> str:
        return " ".join("".join(el.itertext()).split())

    def walk(el: ET.Element) -> None:
        for ch in el:
            if ch.tag == "title":
                out.append(("## " if el.tag == "sec" else "") + txt(ch))
            elif ch.tag == "p":
                out.append(txt(ch))
            elif ch.tag == "tr":
                out.append(" | ".join(txt(c) for c in ch))
            else:
                walk(ch)

    if body is not None:
        walk(body)
    return "\n".join(x for x in out if x.strip("# "))


# --------------------------------------------------------------------------- #
# Dedupe, gate, full text
# --------------------------------------------------------------------------- #
def _dedupe(run: _Run) -> None:
    """DEDUP_OF by PMID, then DOI, then PMCID; a preprint collapses into its published version."""
    ledger = {d.id: d.disposition for d in run.corpus.ledger}
    held = {r.uri for r in run.corpus.records}
    first: dict[str, str] = {}
    for r in sorted(run.corpus.records, key=lambda r: not r.uri.startswith("pmid:")):
        keys = [k for k in (r.uri, r.doi and f"doi:{r.doi.casefold()}",
                            r.pmcid and f"pmc:{r.pmcid}") if k]
        target = r.preprint_of if r.preprint_of in held else next(
            (first[k] for k in keys if k in first), "")
        if target and target != r.uri:
            ledger.setdefault(r.uri, f"DEDUP_OF:{target}")
        else:
            for k in keys:
                first.setdefault(k, r.uri)
    run.commit(ledger=tuple(Disposition(i, d) for i, d in ledger.items()))


def _screen(run: _Run, gene: str, aliases: tuple[Term, ...]) -> list[SourceRecord]:
    """The prefilter gate for one gene's ids; returns the records that pass it."""
    ledger = {d.id: d.disposition for d in run.corpus.ledger}
    recs = {r.uri: r for r in run.corpus.records}
    rx = matcher(aliases)
    body: set[str] = set()
    ids: list[str] = []
    passed: set[str] = set()                # another gene's gate holds these open
    for q in run.corpus.queries:
        if q.channel == "gate":
            passed |= set(q.ids) if q.gene != gene else set()
        elif q.gene in (gene, ""):
            ids += q.ids
            body |= set(q.ids) if q.channel in _BODY else set()
    gated: list[SourceRecord] = []
    for i in dict.fromkeys(ids):
        uri = ledger[i][9:] if ledger.get(i, "").startswith("DEDUP_OF:") else i
        rec = recs.get(uri)
        if rec is None:
            continue                       # no metadata: unscreened, stays unclosed
        if rec.retracted:
            ledger.setdefault(uri, "RETRACTED")
        elif i in body or uri in body or rx.search(f"{rec.title}\n{rec.abstract}"):
            if ledger.get(uri) == "PREFILTER_NO_ALIAS":     # another gene's gate passes it
                del ledger[uri]
            if any("review" in t.casefold() for t in rec.publication_type):
                ledger.setdefault(uri, "REVIEW_CHASE_ONLY")
            gated.append(rec)
        elif uri not in passed:
            ledger.setdefault(uri, "PREFILTER_NO_ALIAS")
    run.commit(ledger=tuple(Disposition(i, d) for i, d in ledger.items()))
    ids_ = tuple(dict.fromkeys(r.uri for r in gated))
    run.log(QueryRun(f"prefilter gate for {gene}", "gate", gene, total=len(ids_),
                     retrieved=len(ids_), ids=ids_))
    return gated


def full_text(rec: SourceRecord) -> str:
    """The record's Europe PMC full text as plain text; "" when the fetch fails."""
    try:
        return _jats(_get(f"{EPMC}{rec.pmcid}/fullTextXML", {}, raw=True))
    except (SourceUnavailable, ET.ParseError):
        return ""


def _full_text(run: _Run, rec: SourceRecord) -> None:
    """Fetch an OA candidate's full text once; a failure is logged, the abstract stays."""
    url = f"{EPMC}{rec.pmcid}/fullTextXML"
    text = full_text(rec)
    if not text:
        run.log(QueryRun(url, "fulltext", "", total=1, status=QueryStatus.FAILED))
        return
    recs = tuple(replace(r, full_text=text) if r.uri == rec.uri else r
                 for r in run.corpus.records)
    run.corpus = replace(run.corpus, records=recs)
    run.log(QueryRun(url, "fulltext", "", total=1, retrieved=1))


# --------------------------------------------------------------------------- #
# One gene
# --------------------------------------------------------------------------- #
def retrieve(ctx: StudyContext, gene: str, aliases: tuple[Term, ...], *, today: date,
             tissue_terms: tuple[str, ...] = (), path: Path | None = None,
             papers_of_interest: tuple[str, ...] = (), cap: int = CAP) -> Corpus:
    """Run C0, C3, C1, C4, C2 and C5 for one gene, then dedupe, gate and fetch full text.

    Resumable: a query already logged (not FAILED) and a full text already held are not
    re-run, and the corpus is saved after every query and fetch.
    """
    p = path or corpus_path(ctx.study)
    try:
        corpus = load(p)
    except CorpusUnavailable:
        corpus = Corpus()
    run = _Run(corpus, p, cap, today.isoformat())
    open_ = tuple(a.text for a in aliases if not a.restricted)
    degraded = not open_                  # only restricted aliases: noisy in EPMC fields
    g_epmc = _or(open_ or (a.text for a in aliases))
    g_body = _or(open_ or (a.text for a in aliases), ("RESULTS", "FIG", "TABLE"))
    tissue = tissue_terms or (ctx.tissue,)
    t_plain, t_tiab = _or(tissue), _or(tissue, ("tiab",), pubmed=True)
    disease = (ctx.disease_state, *(v for _, v in ctx.disease_qualifiers))
    d_plain = (_or(disease),) if ctx.disease_state else ()
    d_tiab = (_or(disease, ("tiab",), pubmed=True),) if ctx.disease_state else ()

    # C0 dataset citers (+ C4 recency when truncated)
    if ctx.release_status == "confirmed":
        cites = f"CITES:{ctx.release_pmid}_MED"
        c0 = run.ladder(_ladder(g_epmc, cites=cites, pubmed=False, today=today), "C0", gene,
                        pubmed=False, degraded=degraded)
        if c0.status is QueryStatus.TRUNCATED:
            run.ladder([compose(g_epmc, cites=cites, extra=(_since(today, 24, False),))],
                       "C4:C0", gene, pubmed=False, degraded=degraded)
    elif ctx.release_status == "unresolved":
        note = f"CITES: release paper of {ctx.study} unresolved"
        if run.done(note, "C0", gene, "") is None:
            run.log(QueryRun(note, "C0", gene, status=QueryStatus.DEGRADED))

    for entry in ctx.cell_types:
        cell, cells = entry.label, cell_terms(entry)
        # C3 body sections, then its recency variant (always: body hits are no tiab superset)
        run.ladder(_ladder(g_body, tissue=t_plain, cells=cells, cell_fields=("RESULTS", "FIG"),
                           pubmed=False, today=today), "C3", gene, cell, pubmed=False,
                   degraded=degraded)
        run.ladder([compose(g_body, tissue=t_plain, cells=_or(cells, ("RESULTS", "FIG")),
                            extra=(_since(today, 24, False),))], "C4:C3", gene, cell,
                   pubmed=False, degraded=degraded)
        # C1 tiab facet: unrestricted aliases, restricted ones only in proximity to the cell
        prox = [f'"{a.text} {c}"[tiab:~6]' for a in aliases if a.restricted for c in cells]
        g_tiab = "(" + " OR ".join([*(f'"{a}"[tiab]' for a in open_), *prox]) + ")"
        c1 = run.ladder(_ladder(g_tiab, tissue=t_tiab, cells=cells, extra=d_tiab,
                                pubmed=True, today=today), "C1", gene, cell, pubmed=True)
        # C4 preprints with the C1 terms
        run.ladder([compose(g_epmc, tissue=t_plain, cells=_or(cells),
                            extra=(*d_plain, "SRC:PPR"))],
                   "C4:PPR", gene, cell, pubmed=False, degraded=degraded)
        if c1.status is QueryStatus.TRUNCATED:
            # C4 recency and C2 proximity: only for a truncated C1 (subsets otherwise)
            run.ladder([compose(g_tiab, tissue=t_tiab, cells=_or(cells, ("tiab",), pubmed=True),
                                extra=(*d_tiab, _since(today, 24, True)))],
                       "C4:C1", gene, cell, pubmed=True)
            near = "(" + " OR ".join(f'"{a.text} {c}"[tiab:~6]' for a in aliases
                                     for c in cells) + ")"
            run.ladder([compose(near, tissue=t_tiab)], "C2", gene, cell, pubmed=True)

    # C5 papers of interest: looked up, then screened like any other (never auto-accepted)
    if papers_of_interest:
        run.lookup((f"pmid:{p}" for p in papers_of_interest), "C5", "")
    mine = {i for q in run.corpus.queries if q.gene == gene for i in q.ids}
    preprints = [r.preprint_of for r in run.corpus.records if r.preprint_of and r.uri in mine]
    run.lookup([*mine, *preprints], "meta", gene)
    _dedupe(run)
    for rec in _screen(run, gene, aliases):
        if rec.open_access and rec.pmcid and not rec.full_text:
            _full_text(run, rec)
    return run.corpus
