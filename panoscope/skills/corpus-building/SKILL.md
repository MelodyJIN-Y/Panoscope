---
name: corpus-building
description: Build Panoscope's narrow-window literature corpus for one study in any LLM client that has literature tools (a PubMed or Europe PMC MCP server, or web access to them). Use when a biologist hands you a Panoscope brief folder (BRIEF.md, study_context.json, format.schema.json) and asks you to find what the literature says about each gene in their tissue, disease state and cell types. You search, screen and quote; Panoscope's code then re-fetches every paper, checks every quote verbatim, and computes every rung, label and grade itself.
---

# Building a Panoscope literature corpus in your own client

You are searching the literature for ONE study: each gene in scope, in THIS tissue, THIS
disease state, EVERY cell type in the study's frame, in humans. Panoscope imports the file
you write and trusts nothing you computed. Its code resolves every identifier, re-fetches
each paper's text from Europe PMC, checks every quoted span character for character, and
recomputes the rung (W0..W5), modality, species, stance and flags. Your job is recall and
exact quotation, not judgment.

## Inputs (all precomputed by Panoscope's code; never guess or add to them)

`study_context.json` (also rendered at the top of BRIEF.md) gives:

- `study_id`, `brief_hash`: copy both into your output header.
- `genes`: each gene in scope with its `aliases` (from HGNC and MeSH). An alias marked
  `restricted` (3 characters or fewer, an English word, a common acronym, a bare CD number)
  is used ONLY in proximity to a cell-type term, never on its own. If a gene's
  `aliases_status` is not `ok`, search with the aliases it does list and log one `FAILED`
  query line for its vocabulary (source `vocab`).
- `tissue.terms` and `organ_disease_terms.terms`: together they are the tissue block T.
  Organ-disease terms matter: many in-window papers say "lupus nephritis" or "renal cell
  carcinoma" and never "kidney". `tissue.blocks` is T already split into chunks.
- `disease_state`: the study's declared disease state and qualifiers, as the disease terms
  D. When it is empty the window is W0*: tissue and cell type, any disease state.
- `cell_types`: the full frame. Each entry has its `label` (use it verbatim as a
  statement's `cell_type`) and `terms` (its synonyms, including marker-plus spellings such
  as "CD8+ T cells" and singular forms).
- `assay_terms`: the cell-resolved assay block M, used only to narrow an oversized query.
- `release_paper`: this dataset's own paper, with its status. `papers_of_interest`: ids
  the biologist named.

## Absolute rules

1. Never write a PMID, PMCID or DOI that did not come from a tool result in this run. No
   identifier from memory, ever. Panoscope resolves each one and rejects any that does not
   resolve.
2. Never use what you remember about a gene, a cell type or a paper. Only what the tools
   returned.
3. Every span is copied character for character from the paper's title, abstract or Europe
   PMC / PubMed Central full text: at most 300 characters, one passage, no ellipses, no
   stitching, no corrections. A span from a publisher PDF or HTML page that is not in
   Europe PMC will fail Panoscope's check and be dropped. Panoscope checks against Europe
   PMC's own text: where your tool renders a character differently (PubMed's superscript
   "⁺" where Europe PMC has "+"), copy Europe PMC's rendering.
4. Never compute a rung, a grade, a species state or a modality label. Fields with those
   names are ignored on import.
5. Never rank or filter by citation count. Never call a citation-metrics tool.
6. No novelty phrasing ("first", "novel", "new finding", "previously unknown") anywhere you
   write. Report what a paper says, not whether it is new.
7. Literature never supplies a value for this study and never confirms it. A paper's number
   stays inside its span.
8. Every query has an anchor: a tissue term, a cell-type term, or `CITES:<release pmid>_MED`.
   Never a gene alone. Never a species or MeSH filter.
9. Log every call with the EXACT query string sent. A call that errors is `FAILED`, which is
   not "no results".
10. Every identifier any query returns ends with exactly one paper line and disposition.
    Nothing disappears silently.
11. "Thin" is never yours to declare. If you found little, say what you ran.

## The window: narrow first, widen only when thin

Priority is the narrow window: this gene, this tissue, this disease state, a frame cell
type, human. Read those papers first and most deeply. Widen one step at a time, and only for
a cell type where the narrow window gave fewer than 2 papers with a statement: same tissue
in another disease state; then the same cell type in another tissue; then other species;
then in vitro. Never let wider papers crowd out narrow ones. Papers that re-analyse this
study's own dataset (their methods or data availability name the release paper's data;
citing the release paper alone is not enough) are this dataset's own papers: read them,
record their statements, and give them the disposition `SAME_DATASET_READ`. Panoscope
decides from the paper itself which statements it shows apart and does not count.

## Searching, channel by channel (in this order, for each gene)

G is the gene block: `"alias1" OR "alias2" ...` over the unrestricted aliases.
C_x is cell type x's terms ORed. T is the tissue block.

- **C0 dataset citers** (only when `release_paper.status` is `confirmed`). First fetch the
  release paper itself and read it in full for the gene: a paper line with channel
  `C0:release` and disposition `SAME_DATASET_READ`. Then Europe PMC: `CITES:<pmid>_MED AND (G)`, newest first, paged with the
  cursor to the end. If the status is not `confirmed`, log one `FAILED` query line saying
  the release paper is unresolved.
- **C3 body sections**, for each cell type x: Europe PMC
  `(RESULTS:"a1" OR FIG:"a1" OR TABLE:"a1" ...) AND (RESULTS:"s1" OR FIG:"s1" ...) AND (T)`.
- **C1 title/abstract facet**, for each cell type x: PubMed
  `(G in [tiab]) AND (T in [tiab]) AND (C_x in [tiab])`. Do NOT add the disease terms: many
  in-window papers never write the disease phrase in the title or abstract; the disease
  state is read from the paper, and Panoscope sets the rung from it.
- **C4 recency**: C0, C3 and C1 again restricted to the last 24 months, newest first; plus
  Europe PMC preprints (`SRC:PPR`) with the C1 terms.
- **C2 proximity**, only for a cell type whose C1 was truncated:
  `"<alias> <term>"[tiab:~6] AND (T)`, including restricted aliases.
- **C5 papers of interest**: look each one up and screen it like any other hit. Never
  accept one automatically.

Refused queries, in every channel: if a source refuses or errors (HTTP 400, 414, 500 or
503), split instead of giving up: run one alias at a time, and if T is still too long, run
one `tissue.blocks` chunk at a time. Log every part.

Paging: page every query to the end or to 200 results. Over 200: narrow once by adding
`AND (M)`, then by putting C_x in the title only, then to the last 36 months, each step
keeping the ones before it, logging each step. Still over 200: keep the newest 200 and mark the query `TRUNCATED`. A query line's
`total` is the hit count the tool reported, `retrieved` the number you actually screened.

## Screening and dispositions

Fetch metadata for every hit. Keep a hit if an alias appears whole-word in its title or
abstract, or it came from C0 or C3; otherwise `PREFILTER_NO_ALIAS`. For every kept
open-access paper, fetch the full text (fetching is not reading; do it broadly). Every
returned id gets exactly one disposition:

`DEDUP_OF:<id>` | `PREFILTER_NO_ALIAS` | `RETRACTED` | `REVIEW_CHASE_ONLY` |
`P1_EXCLUDED:<reason with a verbatim quote>` | `NON_TISSUE` (serum, plasma, urine, CSF only) |
`OUT_OF_LADDER` | `NOT_SCREENED_TRUNCATED` | `NOT_READ_BUDGET` | `READ:<n statements>` |
`SAME_DATASET_READ`.

Exclude only on a quote saying non-human only or another tissue only. A missing species
word is not a reason. Reviews give no statements: chase the references in their sentences
that name the gene and a cell type, and screen those papers.

## Reading and statements

In each paper you read, find every sentence that names an alias AND any frame cell type
(not only the one you expect). Each gives one statement or none. The cell type in the span
must be the cell that EXPRESSES the gene, not a neighbour, a region, or a cell carrying the
gene's receptor. When a paper names a cell state that is a subset of a frame type and is not
in its terms (for example "Tph" or "CXCL13-producing"), record the statement against the
frame label if the span also names that label's term; do not add search terms, and list the
state label in your closing reply so the biologist can add it as a synonym at intake.

Predicates: `MARKS` (marker, specific, identifies, defines, restricted, exclusive,
distinguishes) | `ENRICHED_VS` (a comparator in the span; copy its words from the span,
verbatim, into `comparator`, or it reads as `EXPRESSED_IN`) | `SUBSET_MARKER` | `EXPRESSED_IN` |
`NOT_EXPRESSED_IN` (not, no, absent, negative, lacks, undetectable, low) | `COLOCALISED`
(signal in a region containing the cell, not attributed to it) | `STATE_ASSOCIATED` (the
alias is followed by status, signaling, pathway, signature, score, program, axis,
amplification, copy number or mutation) | `FUNCTIONAL`.

## Output: one JSONL file in Panoscope's format

One JSON object per line, valid against `format.schema.json`. The first line is the only
header. Then one `paper` line per screened id, one `statement` line per statement, one
`query` line per call. `client`, `provider` and `model` are what the biologist ran you in
(ask if you do not know; never guess), `run_date` is today. An illustrative file (the id is
deliberately fake; never copy it):

```
{"kind": "header", "format_version": 1, "study_id": "my_study", "brief_hash": "0123456789abcdef", "client": "Claude Desktop", "provider": "Anthropic", "model": "the model you are", "run_date": "2026-11-08"}
{"kind": "query", "query": "(\"POSTN\"[tiab]) AND (\"breast\"[tiab]) AND (\"fibroblasts\"[tiab])", "source": "PubMed", "gene": "POSTN", "cell_type": "Fibroblasts", "total": 1, "retrieved": 1, "status": "OK"}
{"kind": "paper", "id": "pmid:00000000", "title": "As the tool returned it", "channel": "C1", "disposition": "READ:1"}
{"kind": "statement", "paper": "pmid:00000000", "gene": "POSTN", "cell_type_as_written": "cancer-associated fibroblasts", "cell_type": "Fibroblasts", "predicate": "MARKS", "span": "Copied character for character from the abstract.", "section": "abstract"}
```

Before you finish: re-check every span against the saved tool output and drop any that is
not found (never edit a span to make it match), and check that every id has a paper line.
Then tell the biologist where the file is, which queries `FAILED` or were `TRUNCATED`, and
any cell-state labels worth adding as synonyms. They import it with
`panoscope corpus import --study <study_id> <file>` or the app's "Import corpus file".
