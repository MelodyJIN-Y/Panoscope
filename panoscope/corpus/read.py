"""Model reading (harness v2 §5 phases 3-4): P1 records a paper's setting, P2 its statements.

The model only reads. It never searches, never writes an identifier, never sees the
study's assignment or numbers, and never computes a rung, flag or grade. Code checks
every quote verbatim against the stored text (a mismatch is dropped and logged), then
computes rung, modality, species, stance and flags (``corpus.window``).
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final

from panoscope.corpus.channels import _Run, cell_terms, cooccurrence, matcher
from panoscope.corpus.store import (
    Corpus, CorpusUnavailable, Disposition, QueryRun, QueryStatus, Setting, SourceRecord,
    Statement, corpus_path, load,
)
from panoscope.corpus.window import (
    Facets, Species, StudyContext, Term, modality, modality_match, rung, species_state,
)

log = logging.getLogger(__name__)

Reader = Callable[[str, str], tuple[str, int, int]]
"""(system, user) -> (text, input tokens, output tokens); raises on any failure."""

READER_MODEL: Final = os.getenv("PANOSCOPE_READER_MODEL", "claude-opus-5-5")
PRICE: Final = {"claude-opus-5-5": (4.0, 20.0)}   # USD per MTok, list price (2026-10)
SETTING_FIELDS: Final = ("species", "specimen", "tissue", "disease", "subtype", "treatment",
                         "stage", "assay", "system", "data_source")
RELATIONS: Final = {
    "MARKS": "supports", "ENRICHED_VS": "supports", "SUBSET_MARKER": "qualifies",
    "EXPRESSED_IN": "qualifies", "NOT_EXPRESSED_IN": "argues_against",
    "COLOCALISED": "neutral", "STATE_ASSOCIATED": "neutral", "FUNCTIONAL": "neutral",
}
_CLOSED: Final = ("DEDUP_OF:", "RETRACTED", "REVIEW_CHASE_ONLY", "P1_EXCLUDED:", "NON_TISSUE",
                  "PREFILTER_NO_ALIAS")

PREAMBLE: Final = (
    "You read one paper's text for Panoscope. Use only the text given. Never write a PMID, "
    "DOI or any identifier. Never use what you remember. Every quote and span is copied "
    "character for character from the text given: no ellipses, no stitching, no corrections. "
    "Reply with one JSON object and nothing else.")
P1: Final = PREAMBLE + (
    "\nRecord the paper's SETTING. For each field give a verbatim quote of at most 300 "
    "characters that states it, or null: " + ", ".join(SETTING_FIELDS) + ". specimen is "
    "tissue or biofluid; system is primary tissue, xenograft, organoid or cell line; "
    "data_source says whether the authors collected the samples or reanalysed public data. "
    'Also "paper_kind": primary_biology | methods_tool | review, and "exclude": '
    '{"reason", "quote"} ONLY when a quote states the work is non-human only or another '
    "tissue only; a missing species word is not a reason. Otherwise exclude is null.")
P2: Final = PREAMBLE + (
    "\nExtract statements about where the GENE is, and is not, expressed, from the PASSAGES. "
    "Each statement: passage id, span (one verbatim stretch of that passage, at most 300 "
    "characters, naming the gene and the cell type), cell_type (a FRAME label, "
    '"other:<as written>", or "none"), relation, comparator (as written, or null). The cell '
    "type is the cell that EXPRESSES the gene; a nearby cell, a cell in the same region or "
    "one carrying the gene's receptor is not it (use COLOCALISED or omit). Relations: MARKS, "
    "ENRICHED_VS, SUBSET_MARKER, EXPRESSED_IN, NOT_EXPRESSED_IN, COLOCALISED, "
    "STATE_ASSOCIATED, FUNCTIONAL. Extract claims about every frame cell type, not only one; "
    '"also in X" and "not in X" matter as much as "marks X". Then account for EVERY listed '
    'sentence id: "accounted": [{"sentence_id", "statements": [indexes] or "no_claim"}]. '
    'Reply {"statements": [...], "accounted": [...]}.')


@dataclass
class Receipt:
    """The build's cost receipt: model calls, tokens, and spans code rejected."""

    model: str = READER_MODEL
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rejected: int = 0

    @property
    def text(self) -> str:
        cost = ""
        if self.model in PRICE:
            i, o = PRICE[self.model]
            usd = (self.input_tokens * i + self.output_tokens * o) / 1e6
            cost = f", about ${usd:.2f} at list price"
        return (f"{self.calls} call(s) to {self.model}, {self.input_tokens} input and "
                f"{self.output_tokens} output tokens{cost}; {self.rejected} span(s) rejected")


def model_reader(receipt: Receipt) -> Reader:
    """The strongest model through the agent's client on the user's own key; refuses without one."""
    from panoscope.agent import loop

    client = loop._default_agent()._get_client() if loop.has_model_key() else None
    if client is None:
        raise CorpusUnavailable("corpus build refused: no model key (set ANTHROPIC_API_KEY "
                                "to your own API key; see docs/llm-access.md)")

    def read(system: str, user: str) -> tuple[str, int, int]:
        r = client.messages.create(model=receipt.model, max_tokens=16000, system=system,
                                   output_config={"effort": "high"},
                                   messages=[{"role": "user", "content": user}])
        if r.stop_reason == "refusal":
            raise RuntimeError("the model declined")
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, r.usage.input_tokens, r.usage.output_tokens
    return read


def _ask(reader: Reader, receipt: Receipt, system: str, user: str) -> dict:
    text, i, o = reader(system, user)
    receipt.calls += 1
    receipt.input_tokens += i
    receipt.output_tokens += o
    return json.loads(text[text.index("{"):text.rindex("}") + 1])


# --------------------------------------------------------------------------- #
# Text the model reads
# --------------------------------------------------------------------------- #
def paragraphs(text: str) -> list[tuple[int, int]]:
    """(start, end) of each non-empty line; the full text is one paragraph per line."""
    out, start = [], 0
    for line in text.split("\n"):
        if line.strip():
            out.append((start, start + len(line)))
        start += len(line) + 1
    return out


_METHODS = re.compile(r"^## .*(method|material|data availab|patients|samples)", re.I)


def methods(rec: SourceRecord) -> str:
    """The full text's Methods-like sections (heading match), "" without full text."""
    out, on = [], False
    for s, e in paragraphs(rec.full_text):
        line = rec.full_text[s:e]
        if line.startswith("## "):
            on = bool(_METHODS.match(line))
        elif on:
            out.append(line)
    return "\n".join(out)


_SPACES: Final = str.maketrans(dict.fromkeys("\xa0\u2002\u2003\u2004\u2005\u2006\u2007"
                                             "\u2008\u2009\u200a\u202f", " "))


def _find(text: str, quote: str, start: int = 0, end: int | None = None) -> int:
    """``str.find``, but a no-break or thin space in the source matches a plain one; one
    character for one, so the offset still slices the source's own characters."""
    return text.translate(_SPACES).find(quote.translate(_SPACES), start,
                                        len(text) if end is None else end)


def _locate(rec: SourceRecord, quote: str) -> tuple[str, int] | None:
    for sec in ("title", "abstract", "full_text"):
        if quote and (i := _find(rec.text(sec), quote)) >= 0:
            return sec, i
    return None


def _setting(corpus: Corpus, uri: str, field: str) -> str:
    return "\n".join(s.quote for s in corpus.settings if s.uri == uri and s.field == field)


# --------------------------------------------------------------------------- #
# P1: the setting
# --------------------------------------------------------------------------- #
_BIOFLUID = re.compile(r"\b(serum|plasma|urine|CSF|cerebrospinal|saliva|biofluid)\b", re.I)


def _biofluid_only(corpus: Corpus, uri: str) -> bool:
    """P1's specimen names a biofluid and P1 found no tissue: tumour plus plasma is tissue too."""
    return bool(_BIOFLUID.search(_setting(corpus, uri, "specimen"))
                and not _setting(corpus, uri, "tissue"))


def _p1(run: _Run, rec: SourceRecord, reader: Reader, receipt: Receipt) -> str:
    """Record the paper's setting once: "ok", "closed" (excluded or biofluid) or "failed"."""
    q = f"P1 setting of {rec.uri}"
    if run.done(q, "read:P1", "", "") is not None:
        return "closed" if any(d.id == rec.uri and d.disposition.startswith(
            ("P1_EXCLUDED:", "NON_TISSUE")) for d in run.corpus.ledger) else "ok"
    try:
        got = _ask(reader, receipt, P1,
                   f"ABSTRACT\n{rec.title}\n{rec.abstract}\n\nMETHODS\n{methods(rec)}")
    except Exception as exc:  # noqa: BLE001 - a failed read is a gap, never an empty read
        log.warning("P1 failed for %s: %s", rec.uri, exc)
        run.log(QueryRun(q, "read:P1", "", total=1, status=QueryStatus.FAILED))
        return "failed"
    settings = list(run.corpus.settings)
    for f in SETTING_FIELDS:
        quote = got.get(f)
        if not isinstance(quote, str) or not quote:
            continue
        if (at := _locate(rec, quote)) is None:
            receipt.rejected += 1
            log.warning("P1 %s quote for %s not verbatim, dropped: %r", f, rec.uri, quote)
            continue
        src = rec.text(at[0])[at[1]:at[1] + len(quote)]
        settings.append(Setting(rec.uri, f, src, at[1], at[1] + len(quote), at[0]))
    run.commit(settings=tuple(settings))
    ex = got.get("exclude")
    closed = ""
    if isinstance(ex, dict) and _locate(rec, str(ex.get("quote") or "")) is not None:
        closed = f"P1_EXCLUDED:{ex.get('reason') or 'excluded'}: {ex['quote']}"
    elif _biofluid_only(run.corpus, rec.uri):
        closed = "NON_TISSUE"
    if closed:
        run.commit(ledger=(*(d for d in run.corpus.ledger if d.id != rec.uri),
                           Disposition(rec.uri, closed)))
    run.log(QueryRun(q, "read:P1", "", total=1, retrieved=1))
    return "closed" if closed else "ok"


# --------------------------------------------------------------------------- #
# P2: statements, then everything code computes about them
# --------------------------------------------------------------------------- #
_STATE = re.compile(r"^[\s-]*(status|signal(?:l)?ing|pathway|related|signature|score|"
                    r"program(?:me)?|axis|amplification|copy number|mutations?)\b", re.I)
_MARKER = re.compile(r"\bmark(s|ed|ers?)?\b|specific|identif|defin|restricted|exclusive|"
                     r"distinguish", re.I)
_NEG = re.compile(r"\b(not|no|absent|negative|lack\w*|undetect\w*|low)\b", re.I)
_IN_VITRO = re.compile(r"\b(cell lines?|organoids?|xenografts?|in vitro|cultured)\b", re.I)
_QUANTITY = re.compile(r"\d\s*(%|-?fold|µm|um|mm)|\b\d+\s*/\s*\d+\b|\bn\s*=\s*\d+", re.I)
_REANALYSIS = re.compile(r"publicly available|downloaded from|obtained from (GEO|TCGA|the)",
                         re.I)


def relation(span: str, said: str, comparator: str, gene_rx: re.Pattern[str]) -> str | None:
    """The relation code lets the span carry; None rejects the statement."""
    if any(_STATE.match(span[m.end():]) for m in gene_rx.finditer(span)):
        return "STATE_ASSOCIATED"
    if said not in RELATIONS:
        return None
    if said == "MARKS" and not _MARKER.search(span):
        return "EXPRESSED_IN"
    if said == "ENRICHED_VS" and not (comparator and comparator in span):
        return "EXPRESSED_IN"
    if said == "NOT_EXPRESSED_IN" and not _NEG.search(span):
        return None
    return said


def _words(terms: tuple[str, ...], text: str) -> bool:
    return bool(terms) and bool(text) and bool(matcher((), terms).search(text))


def _statement(ctx: StudyContext, corpus: Corpus, rec: SourceRecord, gene: str, cell: str,
               rel: str, span: str, sec: str, start: int, tissue_terms: tuple[str, ...],
               generator: str) -> Statement:
    """Facets, rung, modality, species and flags of one verified span, all by code."""
    def s(field: str) -> str:
        return _setting(corpus, rec.uri, field)

    meth = "\n".join((s("assay"), s("species"), s("system"), methods(rec)))
    sp = species_state(span, meth, f"{rec.abstract}\n{rec.full_text}")
    disease_terms = tuple(t for t in (ctx.disease_state, *(v for _, v in ctx.disease_qualifiers))
                          if t)
    disease = (None if not ctx.disease_state
               else True if _words(disease_terms, f"{span}\n{s('disease')}")
               else False if s("disease") else None)
    tissue = _words(tissue_terms, f"{span}\n{s('tissue')}") or None
    in_vivo = (False if _IN_VITRO.search(f"{span}\n{s('system')}") else
               True if (tissue or s("specimen") or sp in (Species.HUMAN_STATED,
                                                          Species.HUMAN_INFERRED)) else None)
    analyte, res = modality(f"{span}\n{meth}")
    r = rung(Facets(tissue=tissue, cell_type=bool(cell) or None, disease=disease,
                    in_vivo=in_vivo, species=sp,
                    biofluid=bool(_BIOFLUID.search(span)) or _biofluid_only(corpus, rec.uri),
                    resolution=res), disease_declared=bool(ctx.disease_state))
    flags = [f for f, on in (
        ("SAME_DATASET", bool(ctx.release_pmid) and rec.uri == f"pmid:{ctx.release_pmid}"),
        ("PREPRINT", rec.uri.startswith("epmc:PPR")
         or any("preprint" in t.casefold() for t in rec.publication_type)),
        ("ABSTRACT_ONLY", not rec.full_text),
        ("SPECIES_UNSTATED", sp is Species.UNSTATED),
        ("CONTAINS_QUANTITY", bool(_QUANTITY.search(span))),
        ("REANALYSIS", bool(_REANALYSIS.search(s("data_source")))),
    ) if on]
    return Statement(uri=rec.uri, gene=gene, cell_type=cell, tissue=ctx.tissue,
                     predicate=rel, stance=RELATIONS[rel], span=span, start=start,
                     end=start + len(span), section=sec, rung=r.value if r else "",
                     analyte=analyte, resolution=res,
                     modality_match=modality_match(f"{span}\n{meth}", ctx.platform),
                     species=sp.value, flags=tuple(flags), written_by=generator)


Passage = tuple[str, str, int, int]          # (id, section, start, end)


def _p2(ctx: StudyContext, corpus: Corpus, rec: SourceRecord, gene: str,
        aliases: tuple[Term, ...], passages: list[Passage], sentence_ids: list[str],
        reader: Reader, receipt: Receipt, tissue_terms: tuple[str, ...], generator: str
        ) -> tuple[list[Statement], set[str]]:
    """One P2 call; the verified statements and the sentence ids it accounted for."""
    frame = {e.label.casefold(): e for e in ctx.cell_types}
    body = "\n\n".join(f"[{pid}] {rec.text(sec)[a:b]}" for pid, sec, a, b in passages)
    user = (f"GENE {gene}; aliases: {', '.join(a.text for a in aliases)}\n"
            "FRAME: " + "; ".join(f"{e.label} ({', '.join(cell_terms(e))})"
                                  for e in ctx.cell_types)
            + f"\n\nPASSAGES\n{body}\n\nSENTENCES TO ACCOUNT FOR: {', '.join(sentence_ids)}")
    got = _ask(reader, receipt, P2, user)
    where = {pid: (sec, a, b) for pid, sec, a, b in passages}
    gene_rx = matcher(aliases)
    kept: dict[int, Statement] = {}
    for i, st in enumerate(got.get("statements") or []):
        span = str(st.get("span") or "")
        sec, a, b = where.get(str(st.get("passage")), ("", 0, 0))
        at = _find(rec.text(sec), span, a, b) if sec and span else -1
        span = rec.text(sec)[at:at + len(span)] if at >= 0 else span
        entry = frame.get(str(st.get("cell_type") or "").casefold())
        rel = relation(span, str(st.get("relation") or ""), str(st.get("comparator") or ""),
                       gene_rx) if at >= 0 else None
        if (rel is None or not gene_rx.search(span)
                or (entry is not None and not _words(cell_terms(entry), span))):
            receipt.rejected += 1
            log.warning("P2 statement rejected for %s/%s: %r", rec.uri, gene, span[:120])
            continue
        kept[i] = _statement(ctx, corpus, rec, gene, entry.label if entry else "", rel,
                             span, sec, at, tissue_terms, generator)
    done = {str(a.get("sentence_id")) for a in got.get("accounted") or []
            if a.get("statements") == "no_claim"
            or any(isinstance(j, int) and j in kept for j in a.get("statements") or [])}
    return list(kept.values()), done


def passages(rec: SourceRecord, aliases: tuple[Term, ...], cells: tuple[str, ...]
             ) -> tuple[list[Passage], dict[str, tuple[str, int, int]]]:
    """Abstract, co-occurrence paragraphs +-1 and alias table rows; the sentence ids in them."""
    gene_rx = matcher(aliases)
    sents = cooccurrence(rec, aliases, cells)
    paras = paragraphs(rec.full_text)
    want = {k for j, (a, b) in enumerate(paras) for s in sents
            if s.section == "full_text" and a <= s.start < b for k in (j - 1, j, j + 1)}
    want |= {j for j, (a, b) in enumerate(paras)
             if " | " in rec.full_text[a:b] and gene_rx.search(rec.full_text[a:b])}
    out: list[Passage] = [("A", "abstract", 0, len(rec.abstract))] if rec.abstract else []
    out += [(f"P{j}", "full_text", *paras[j]) for j in sorted(want) if 0 <= j < len(paras)]
    ids: dict[str, tuple[str, int, int]] = {}
    for pid, sec, a, b in out:
        inside = [x for x in sents if x.section == sec and a <= x.start < b]
        ids.update({f"{pid}.s{k}": (sec, x.start, x.end) for k, x in enumerate(inside)})
    return out, ids


def read(ctx: StudyContext, gene: str, aliases: tuple[Term, ...], *, reader: Reader,
         receipt: Receipt, tissue_terms: tuple[str, ...] = (), path: Path | None = None,
         retrieved: str = "") -> Corpus:
    """P1 then P2 for every gated paper of ``gene``; resumable, saved after each paper.

    Every co-occurrence sentence is accounted for or re-asked once; still unaccounted
    is counted as unextracted (the read row is DEGRADED, so the key cannot read thin).
    """
    p = path or corpus_path(ctx.study)
    run = _Run(load(p), p, 0, retrieved)
    generator = f"{receipt.model} via the Anthropic API (user key)"
    tissue = tissue_terms or (ctx.tissue,)
    cells = tuple(t for e in ctx.cell_types for t in cell_terms(e))
    gated = [i for q in run.corpus.queries if q.channel == "gate" and q.gene == gene
             for i in q.ids]
    for uri in dict.fromkeys(gated):
        ledger = {d.id: d.disposition for d in run.corpus.ledger}
        rec = next(iter(run.corpus.find(uri)), None)
        q = f"P2 {uri} for {gene}"
        if (rec is None or ledger.get(uri, "").startswith(_CLOSED)
                or run.done(q, "read", gene, "") is not None):
            continue
        if (setting := _p1(run, rec, reader, receipt)) != "ok":
            if setting == "failed":     # this gene's key must not read thin past it
                run.log(QueryRun(q, "read", gene, total=1, status=QueryStatus.FAILED))
            continue
        ps, ids = passages(rec, aliases, cells)
        try:
            found, done = _p2(ctx, run.corpus, rec, gene, aliases, ps, list(ids), reader,
                              receipt, tissue, generator)
            left = [i for i in ids if i not in done]
            if left:    # one re-ask with only the unaccounted sentences
                again, done = _p2(ctx, run.corpus, rec, gene, aliases,
                                  [(i, *ids[i]) for i in left], left, reader, receipt,
                                  tissue, generator)
                found += again
                left = [i for i in left if i not in done]
        except Exception as exc:  # noqa: BLE001 - a failed read is a gap, never an empty read
            log.warning("P2 failed for %s/%s: %s", uri, gene, exc)
            run.log(QueryRun(q, "read", gene, total=len(ids), status=QueryStatus.FAILED))
            continue
        statements = (*run.corpus.statements, *found)
        n = sum(1 for s in statements if s.uri == uri)
        same = bool(ctx.release_pmid) and uri == f"pmid:{ctx.release_pmid}"
        run.commit(statements=statements, ledger=(
            *(d for d in run.corpus.ledger if d.id != uri),
            Disposition(uri, "SAME_DATASET_READ" if same else f"READ:{n}:{len(left)}")))
        run.log(QueryRun(q, "read", gene, total=len(ids), retrieved=len(ids) - len(left),
                         status=QueryStatus.DEGRADED if left else QueryStatus.OK))
    return run.corpus
