---
name: geneset-enrichment
description: Interpret per-cluster gene-set enrichment results (MSigDB Hallmark) for spatial transcriptomics clusters that have already been cell-typed. Use when a user has an enrichment result — a per-cluster, per-gene-set table with a score/statistic, a p/q value, the set size, and the overlapping/leading-edge genes — from the jazzPanda competitive gene-set test or a classical over-representation (ORA) run on a targeted panel, and wants a grounded per-cluster interpretation of which biological PROGRAMS are active, a confidence, and a cited biological read. Enforces a panel-coverage rule (the enrichment analog of the panel-absence rule), a two-tier confidence report grounded on the method's score, a cross-cluster themes review, and a no-fabrication / real-citation discipline. Complements — never replaces — the cell-type call from the marker workflow.
---

# Interpreting Gene-Set Enrichment for Spatial Clusters

This skill turns a per-cluster gene-set enrichment result into a grounded read of which biological **programs** are active in each already-annotated cluster: a confidence, a `verify` flag, and a one-clause cited rationale per pathway. It is the interpretation layer; the enrichment method (the jazzPanda competitive gene-set test, or a classical ORA) is the engine. It is a **separate workflow** from marker-gene cell typing: enrichment says *what program is running*, markers say *what cell it is*. Use enrichment to enrich the cell-type story, never to re-derive cell identity.

It is built for a wet-lab biologist reading a **targeted spatial panel** (e.g. a ~280-gene Xenium breast panel). On a targeted panel, enrichment is biased and low-powered, so the caveats below are not optional: follow them strictly.

## Core principles (never violate)

- **Confident floor.** Every pathway, score, q-value, and leading-edge gene must trace to the enrichment result or the panel. Never invent a pathway, a statistic, or a gene. If it is not in the source, do not state it.
- **Open ceiling.** The biological meaning of an enriched program is a literature-grounded direction, not a fact. Label it as a direction.
- **Panel-coverage rule (the headline catch).** A Hallmark set has ~200 genes, but only the genes on the panel were ever measured. Enrichment is computed over that panel-scoped subset only. ALWAYS state how many of the set's genes were on the panel (`K` of `N`), and say plainly: **panel-scoped, not genome-wide**. A set with too few panel genes, or driven by too few genes, is untestable — do not call it enriched.
- **Cite everything, with real references.** Every interpretive (biological) claim about a program must carry a citation to a real paper, looked up **live** through a literature connector (PubMed / bioRxiv). Never write a PMID from memory: a fabricated citation is the worst possible failure, worse than no citation. If a lookup returns nothing, say the literature is thin; do not invent a reference.
- **Defer to the biologist, always.** The tool informs and preserves the decision; it never overrides the cell-type call. Enrichment that disagrees with the cell type is reported as a tension, not a correction.
- **When evidence is thin, set `verify = TRUE` and say "re-check this."** A program driven by two genes on a curated panel is a hypothesis, not a finding.

## What the enrichment score means (read this before interpreting any number)

Two methods can feed this skill; both are re-scoped to the panel by the pipeline.

- **jazzPanda competitive gene-set test** (the primary method): for a set and a target cluster, it lasso-selects the set's genes whose spatial vectors track the cluster, then a one-sided z-test compares their inverse-SE-weighted coefficient to the mean of all **other panel genes** (the competitive background). The `test_statistic` (bigger = more enriched) is the score; `p_value`/`p_adj_bh` accompany it; `genes_selected` are the honest driving genes; `gc_corr` is their mean spatial correlation with the cluster.
- **Classical ORA** (comparison): a hypergeometric test of the cluster's marker genes against the set, with the panel as the background universe. The score is `-log10(q)`.

The two scores are on different scales and must never be compared numerically — only their agreement/dissent on *which* sets are enriched is meaningful.

## The interpretation, per cluster

### Step 1: Read the enriched programs
For each cluster, the pipeline has already gated the sets into three tiers:
- **enriched** — clears the bar (`q < 0.05`, `>= 3` driving genes on panel, `>= 3` set genes on panel).
- **suggestive** — same gates but `q` in `[0.05, 0.25]`: real-but-weak, carries `verify = TRUE`.
- **untestable** — too few panel genes or a leading edge of 1–2 genes: never surfaced as a program.

Read the enriched programs top-down by score. A high score with many driving genes and reasonable panel coverage is the strongest evidence.

### Step 2: Apply the panel-coverage rule (CRITICAL: before claiming any program)
For each program, state `K` of `N` genes on the panel. If coverage is low (a few percent — common on a targeted panel), the program is a **direction within the panel's coverage**, not a genome-wide enrichment. Never phrase it as "the cluster is enriched for X" without the panel-scope caveat. A large score on a 3-gene panel footprint is still only 3 genes.

### Step 3: Relate the program to the cell-type call
The cluster already has a cell-type call from the marker workflow. Read each enriched program against it:
- **Concordant** (proliferation in a tumor, cytotoxic/interferon programs in T cells) → the program reinforces the identity; say so.
- **Cross-lineage** (an immune program enriched in a stromal or epithelial cluster) → likely reflects infiltration, shared panel genes, or spatial admixture, not a mis-call. Flag it as a tension to check, never as a re-typing.
- **Shared across many clusters** (an interferon or inflammatory program enriched in most clusters) → partly reflects the panel's design and shared genes; down-weight it as cluster-specific evidence.

### Step 4: Confidence
Confidence is anchored on the top enriched program's score (per method), then demoted one band for very low panel coverage or a tiny leading edge, and set to `verify = TRUE` when no program clears the enriched bar or the top program is shaky. Bands: **Very High, High, Medium-High, Medium, Low**. (Exact edges live in `config.py`; do not restate numbers you were not given.)

## Step 5: Holistic pathway themes (across clusters)
After each cluster, read the whole set together: which programs recur across clusters, and do the enriched programs cohere with the compartment structure (epithelial / stromal / immune / endothelial). A program enriched in one cluster is cluster-specific; one enriched across many is a shared/atmospheric signal (or panel bias). State recurrence as a count, never as a stronger per-cluster claim.

## Step 6: Literature grounding and citation
When you state what a program *is* or *does* biologically, look it up **live** and cite a real, resolvable PMID the biologist can open. Prefer tissue-specific evidence. When the literature is thin, say so; never invent a reference.

## Outputs

### Output 1: Per-pathway biology note (cited)
For each enriched (and suggestive) program in a cluster, one crisp clause: what the program is and why it is relevant to THIS cluster's cell-type identity (concordant, or a flagged cross-lineage tension). Cite exactly **one real, live-looked-up PubMed paper** per note (never from memory). Do **not** restate the score/q/coverage (they are already shown). Keep it to one sentence; never use an em dash. If the program is cross-lineage for this cell type, say so plainly. A note whose literature is thin says so and carries no citation.

### Output 2: Per-cluster enrichment summary (cited)
One or two sentences naming the cluster's dominant program(s) and what they say about its state, anchored to the cell-type call, always carrying the panel-scoped caveat. One real PMID if a supporting paper exists; none if thin.

## Quality checklist
- [ ] Every program, score, q, and leading-edge gene traces to the enrichment result: nothing invented.
- [ ] Panel-coverage rule applied: every claim states `K` of `N` on panel and is scoped "not genome-wide".
- [ ] Only gated (enriched/suggestive) programs are interpreted; untestable ones are not surfaced.
- [ ] Cross-lineage programs are flagged as tensions, never as a re-typing of the cluster.
- [ ] Every biological claim carries a real, connector-fetched citation; no PMID from memory.
- [ ] Weak/low-coverage clusters flagged `verify = TRUE`.
