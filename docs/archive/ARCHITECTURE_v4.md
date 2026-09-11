# Panoscope architecture

**A human-in-the-loop interpretation layer for bioinformatics results.**

Version 4, September 2026. Supersedes `BLUEPRINT.md` as the architecture of record.
Sections marked *built* exist today; *new* is proposed work.

**Changes from v3.** The single "ontology layer" splits into **reference data** (§5) and
the **literature corpus** (§6), which differ on storage, refresh, licensing, and failure
modes. Interpretation is no longer one task: modules declare a **task kind** (§7.1), and
the unit is read from the result rather than declared by the method. The **rival test
generalises** to all task kinds (§7.5). Reconciliation splits into same-question and
different-question paths (§7.7). UI gets a proper section (§11). Operational requirements
are added (§16). Two scope cuts: reference data ships two source kinds, not five, and one
deployment mode, not three.

**Changes in v4.1.** The **interpretation record** is promoted to its own section (§15): a
per-unit artifact written by the deterministic core at run time, recording how each result
file was read and why each interpretation came out as it did. Biologist-facing, not a
debugging tool.

---

## 1. What Panoscope is

A biologist has a result produced by someone else. Panoscope reads it faithfully, says what
it supports and what it does not, and lets the biologist argue with it.

Two boundaries define the tool.

**Panoscope does not run the analysis.** Method output is a consumed input, never
recomputed. This makes the confidence assessment an audit rather than a self-assessment: a
tool that produces a result and then scores its own confidence is grading its own work.

**The biologist makes the call.** Enforced in the write path (§8.5): only the biologist
commits a lab claim, and no claim of any kind can move a number.

---

## 2. Core principle: every claim carries its measurement conditions

The panel-absence rule says an off-panel gene was never measured, so its absence is not
evidence. The architecture is that rule generalised.

A claim is never simply true. It was established under conditions: an assay, a modality, a
tissue, a panel, a segmentation, a statistic. Using it elsewhere is an inference.

| Claim | Established under |
| --- | --- |
| jazzPanda: transcripts localize to cluster C | this panel, platform, segmentation, spatial GLM |
| Cell Ontology: gene X marks cell type Y | flow cytometry, dissociated cells, a tissue |
| Literature: this marker is unreliable in FFPE | a specific tissue and modality |
| Lab: POSTN here is tumour bleed | a 280-gene Xenium panel, breast, this dataset |

They are one object type (§8). Two questions follow, and only the second is novel:

- Is this claim still **true**? Temporal validity. Every memory system handles this.
- Is this claim still **applicable** here? Conditional validity. None of them do.

A claim can be perfectly true and completely inapplicable. Checking that is deterministic
(§8.3).

---

## 3. Three layers

```
  EVIDENCE            immutable, external, versioned
    method results · SpatialData store · reference data · literature corpus · lab claims
                              |
  DETERMINISTIC CORE  owns every number and every gate, zero LLM calls
    platform adapters + method modules -> normalised evidence
    task-specific interpretation · conditionality checks · rival tests
    reconciliation · propagation · claim store · retrieval
                              |
  GENERATION          fenced and swappable
    reads the two layers below, phrases what it is handed
    grounding gate · answer policy · model routing · deterministic fallback
```

The boundaries exist so the generator can be swapped while everything else is held
constant. That is both the local-model story and the ablation.

---

## 4. Evidence

### 4.1 Spatial data: SpatialData (scverse)

The on-disk format for spatial evidence is a SpatialData `.zarr` store, replacing the
hard-coded tidy layer.

| Was | Is | What it fixes |
| --- | --- | --- |
| `cells.csv` | Shapes (GeoDataFrame) | real geometry |
| `transcripts.csv.gz` | Points | lazy, no full read |
| `umap.csv` | `table.obsm["X_umap"]` | standard location |
| `cluster_key.json` | `table.obs[<col>]` | derived, not hardcoded |
| panel TSV | `table.var_names` | the panel *is* what was measured |
| `density/*.parquet` | computed from Points | bin size in real µm |

The panel-absence rule becomes structural: the measured set is `var_names`, true by
construction on every platform. Coordinate systems give physical units, so hex bin size and
the adjacency check (§7.8) work in micrometres rather than invented thresholds.

Constraints: write to Zarr then read back (required for performance); read with
`selection=[shapes, points, table]` and never images; behind a `panoscope[spatial]` extra
with the legacy reader retained; a `.zarr` is a directory so the demo store ships
image-free or comes from Zenodo; reader drift is real, so the prep stage validates what
Panoscope depends on and fails loudly.

### 4.2 Other evidence

**Method results.** As given. Never recomputed, kept **separate from the SpatialData
store**: the store is what was measured, the result is what someone computed from it, and
that boundary is what the audit claim rests on.

**Reference data** (§5), **literature corpus** (§6), **lab claims** (§8).

**Rule: evidence precedes belief.** A derived object can be revised or discarded. The
evidence it came from is never rewritten.

**Export exception.** Confirmed calls may be written back as an `obs` column on export for
napari-spatialdata and squidpy interop. An output artifact, never a write into the store.

### 4.3 Study, dataset, sample

| Level | Is | Owns |
| --- | --- | --- |
| **Study** | the unit of work | tissue, research background, reference view, literature corpus, study-scoped claims |
| **Dataset** | one SpatialData store | one panel, one platform; chat history and review state |
| **Sample** | one replicate / section | its own coordinate system and shapes |

"Session" is avoided: it implies transient, and this persists. Resuming a dataset tomorrow
is a persistence concern, not a new concept.

**Scope hierarchy = conditions hierarchy.** This is why the levels fit rather than bolt on:

| Claim scope | Crosses | Transfer check |
| --- | --- | --- |
| sample | nothing | not invoked |
| dataset | samples, same panel | neutral; section differences noted |
| study | datasets, panels may differ | fires |
| lab | studies | fires hard |

With one dataset the transfer check is hypothetical. With two on different panels it has an
obvious job. Multi-dataset support therefore strengthens the novel contribution.

**Clustering mode must be declared.** Under joint clustering, c2 is the same population in
every sample and evidence pools. Under independent clustering, c2 in two samples are
unrelated objects sharing a label and pooling is wrong. An undeclared mode is a
**validation failure**, not a default.

---

## 5. Reference data

Structured, finite, versioned, shared. This is the atlas-level lookup layer: HGNC, Cell
Ontology, Uberon, MSigDB, and a pluggable marker source.

**It resolves, it does not retrieve.** No ranking, no relevance, no fetch at question time.
A query either has an answer or it does not.

### 5.1 Three tiers

| Tier | Scope | Lifecycle |
| --- | --- | --- |
| **Sources** | global, shared across studies | fetched once per upstream release, cached at `~/.panoscope/reference/<source>/<release>/` |
| **Reference view** | per study | the traversed subgraph for this tissue; derived, rebuildable |
| **Lab claims** | per study, accumulating | authored, owned, irreplaceable (§8) |

Only the third needs backing up, and only the third compounds. A study does not hold a
private copy of the ontology; it holds a view onto the shared cache.

*(Naming: "reference view" over "slice" or "scope". Scope is already load-bearing for
claims and the collision would be confusing.)*

### 5.2 Two source kinds, deliberately

**Cell-type structure.** CL hierarchy, Uberon tissue, HGNC symbols and aliases, marker
edges from a pluggable source. Serves: candidate cell types for a tissue, canonical marker
lists including off-panel genes, sibling types as marker rivals.

**Gene-set structure.** MSigDB membership, and set-to-set overlap computed from shared
genes. Serves: enrichment rivals, since "is this really EMT or a collagen set sharing
genes" is an overlap query once membership is imported.

Three further kinds were considered and cut. Gene function (GO) has no current consumer.
Assay knowledge has no curated source worth importing and arrives as lab claims and
literature, which the claim store already handles. Experimental-design knowledge is
computed from the result table rather than looked up, so it belongs in the core (§7.5).

Adding a source kind later is one importer plus node and edge types, not a new subsystem.
Committing to a taxonomy now would mean writing contracts against untested categories.

### 5.3 What it answers

Which cell types occur in this tissue. What are the canonical markers for fibroblast. Is
FAP on this panel. Which gene sets share genes with this one. Is this symbol an alias.

**Most of what makes Panoscope work is a structural lookup, not a retrieval.** Panel
absence, rival candidates, off-panel canonical lists, set overlap. Which means
literature-off is a degraded mode, not a broken one, and the deterministic guarantees
survive it intact.

### 5.4 Status: candidate source, never evidence

**The hard line.** Reference data says gene X *marks* cell type Y, mostly established in
dissociated scRNA-seq, flow, or IHC. jazzPanda says gene X's transcripts *localize* where
cluster C's cells are. Different claims. Treating a marker edge as evidence that a spatial
cluster is that type swaps one for the other, the same error as weighting fold-change over
the spatial coefficient.

Reference edges may generate rivals, supply canonical marker lists, scope the corpus, and
carry citations. They may **never** move a confidence band, count as agreement, or
corroborate a measurement claim.

**Coverage is uneven**, especially for tissue-specific and disease-associated types. A
missing node reads as *not in reference data*, never as *does not exist*. The
panel-absence rule applied to the graph.

**Assay knowledge is where the differentiator lives and where curated sources are weakest.**
"POSTN bleeds in dense stroma on this panel" is what Claire knows and no database holds. So
the sparsest knowledge kind is the one the feedback loop matters most for, which is worth
stating in the paper rather than treating as a gap.

---

## 6. Literature corpus

Unstructured, unbounded, per-study, never complete. Ranked rather than resolved, goes
stale, and its licence situation differs entirely from reference data because it stores
retrieved spans rather than redistributing a database.

**Scoped by reference data.** The traversal decides which triples to fetch, so reference
data is a prerequisite and the corpus is a layer on top. Build in that order.

### 6.1 The join

The key is the **(gene, cell type, tissue)** triple, not the gene. POSTN in breast is a
different question from POSTN in kidney.

```
tissue -> candidate cell types -> marker genes (incl. off-panel)
       -> literature per triple -> attached to the reference edge
```

This fixes corpus scoping without a heuristic: off-panel canonical markers arrive because
they are edges from the cell type, not because they appeared in the results. Cost scales
with the candidate cell-type set (a dozen or two), not with panel size, which is what keeps
5,000-gene panels tractable.

### 6.2 What the disagreements buy

- **Edge with strong tissue-specific literature** — a well-supported prior
- **Edge with thin or absent literature here** — a weak prior, and it should read that way
- **Edge whose literature qualifies it** (unreliable in FFPE, shared with another lineage
  in tumours) — the qualification travels with the edge every time it is used
- **Edge the tissue literature contradicts** — surfaced as tension, not resolved

A flat corpus cannot distinguish a well-supported edge from an unsupported one.

**Caution.** Attaching literature to an edge can look like validation. A paper retrieved
for a triple is topically relevant, not confirmation the marker works in your assay. The
paper's own context is displayed and a tissue or modality mismatch is visible, or this
makes weak edges look strong.

### 6.3 Storage and refresh

Records and spans in the study's SQLite store, versioned as a **corpus version** recording
build date, tissue, embedding model, query log, and reference releases used. Rebuilt
wholesale; no per-record reconciliation. Staleness is a policy, not an accident: a corpus
older than a threshold is flagged in the report header.

---

## 7. Deterministic core

Zero LLM calls. Fully testable. This is where the guarantees live.

### 7.1 Interpretation is not one task

Panoscope is an interpretation agent for bioinformatics results generally, and different
analyses pose different interpretive questions. Treating them all as "a call per unit with
a band" is the marker task over-generalised.

| | Marker | Gene-set enrichment | Differential expression |
| --- | --- | --- | --- |
| Question | what *is* this? | what's running here? | what changed? |
| Task kind | **assignment** | **characterisation** | **comparison** |
| Answer | one label from a label space | ranked annotated set | ranked annotated set, with direction |
| Confidence means | confidence in the call | per-item support and testability | per-item support and effect |
| Verify flag | per unit | per item | per item |

For markers there is a label space (the Cell Ontology) and the job is choosing from it,
which is why a band, a rival test, and a per-unit verify flag all make sense.

For characterisation and comparison there is no label space and no single answer, so a
unit-level band is meaningless. What the biologist needs is per-item support plus an honest
account of what could not be tested. The header for an enrichment unit is not "CAF, High".
It is closer to *four programs enriched, three untestable on this panel, top is EMT at
z 4.2 on 18 of 200 panel genes*.

Three output types, not one: `Assignment`, `Characterisation`, `Comparison`. The CSV export
schema differs per task, since `cell_type` and `confidence` columns make no sense for a DE
result.

### 7.2 The unit is read from the result

Enrichment might be keyed on cluster, on cell type, or on region. DE might be condition
versus condition across all cells, within each cell type, or cluster versus rest. **The
module reads the unit from the result schema.** It cannot be declared in advance by the
method or the platform.

An unrecognised result shape **fails loudly** rather than being guessed at, the same
discipline as undeclared clustering.

### 7.3 Platform adapters

One per technology. Declares how to read the store, **how "measured" is derived** (Xenium
puts negative controls in `var`, so the rule differs per platform), which views are
available given the elements present, the coordinate system and units, the clustering mode,
the samples, and **standing assay conditions** attached to every measurement claim from
this platform.

That last item is how §2's conditionality reaches the data automatically rather than by
hand.

### 7.4 Method modules

Declares: input schema and unit detection, task kind, what its statistic measures, **what
it does not measure** (mandatory), how absence reads, its own confidence rubric, standing
limitations, its rival space (§7.5), which reference traversals are meaningful, and which
disagreements with other families are expected.

Everything downstream reads a **normalised evidence object**, never a raw method table.

**Invariant.** The jazzPanda and DE rubrics must produce different bands on a constructed
case where fold-change is high and spatial localisation is low. If they agree, the framework
is method-blind and the value proposition inverts. This is the single most important test in
the DE work.

### 7.5 The rival test, generalised

The rival test is not "name a competing cell type". It is the biologist asking **what else
could produce this pattern**, which exists for every analysis type. Only the space of
alternatives changes.

| Task | The rival is | Discriminating evidence | Rival space from |
| --- | --- | --- | --- |
| Assignment | a competing cell type | the marker that separates them | reference data (siblings) |
| Characterisation | a competing explanation for the signal | which genes drive it, overlap with another set, cross-unit profile | reference data (set overlap) |
| Comparison | a confound | per-sample consistency, effect within cell type, driving cells concentrated in one replicate | **the data itself** |

For enrichment the alternative is not a different label but a different mechanism producing
the same signal: a few genes shared with another set, a set enriched everywhere and
therefore atmospheric, or a leading edge that is just the unit's own top markers
reappearing.

For DE the alternatives are experimental-design questions, answered from the result rather
than from curated knowledge. **A module declaring "my rivals come from the data, not from
reference data" is a valid and different answer**, and the contract must allow it.

The common shape: given the current reading, name a specific alternative account, name the
evidence that would separate them, then say honestly what the data cannot settle. The
honest failure differs usefully — assignment: *the deciding marker is off-panel*;
characterisation: *too few panel genes to separate these sets*; comparison: *one replicate
drives this, so condition and batch cannot be separated*.

### 7.6 Named failure modes

Before asking *what is this*, ask *is this a coherent unit*: spatially incoherent cells, a
tissue region rather than a cell type, transcript bleed from a neighbour, a mixed profile
suggesting over-segmentation. Named, computed, more actionable than a low band.

### 7.7 Reconciliation: same question or different question

Two results only agree or disagree if they answer the same question. This was wrong in v3
and would have produced nonsense.

| Relationship | Outcomes |
| --- | --- |
| Same question (two marker methods) | agree · disagree |
| Different question (enrichment vs marker call) | **coherent · tension · not comparable**, never *agree* |

An enrichment result does not corroborate a cell-type call because it is not answering the
identity question. It can show coherence (a proliferation program in a tumour cluster) or
tension (an immune program in a stromal cluster). A DE result about condition differences
within a cluster says nothing about that cluster's identity.

**Dependency structure is mandatory.** Enrichment is computed from the marker result, so
enrichment agreeing with its parent is not corroboration.

1. Independent: lab validation (IHC, orthogonal assay)
2. Partly independent: methods asking different questions of the same data
3. Dependent: a result derived from another result

**Hard rule: reconciliation can lower confidence or raise a flag. It can never raise
confidence.** Output is four buckets, not a score: checkable, agreed, disagreed
unexpectedly, and **not checkable with the results provided**.

### 7.8 Spatial adjacency check

A gene supports unit C but also localizes in unit D. If C and D are spatially
interdigitated that is a **segmentation bleed candidate**; if D sits elsewhere it is a
**genuine specificity problem**. Same statistic, different readings, and only spatial data
separates them. No scRNA-seq tool can produce this. Display flag, not a band input, until
its firing rate is measured.

### 7.9 Replicate concordance

Under joint clustering, compare a unit's profile across replicates. Divergence is a flag:
batch effect, section quality, or real biology, and the check does not claim which.
**Agreement never raises confidence** — replicates are not independent witnesses. Raises
rather than returning a number under independent clustering.

### 7.10 Downstream propagation

A flagged call carries its flag into everything depending on it. A biologist does not act
on "medium confidence"; they act on "four downstream results depend on a call I flagged".

---

## 8. The claim store

Reference edges, literature spans, lab notes, and measurement results are one object type,
differing only in who may change them and how much weight they carry.

### 8.1 The claim

```
subject       node or edge id (gene · cell type · program · tissue · unit)
assertion     what is claimed
authority     measurement | lab | curated | literature
source        citation · note id · edge id · method + statistic
conditions    assay · modality · tissue · panel hash · method · segmentation · fixation
depends_on    genes or nodes the claim rests on
scope         sample:<id> | dataset:<id> | study | lab
valid         active | superseded | retracted
created, author, tension, licence, spans
```

### 8.2 Authority, never merged

```
measurement  >  lab  >  curated  >  literature
```

Conflicts are grouped and displayed, never resolved. **The authority label must never
blur.** If a lab claim renders as a curated fact, or a curated edge as evidence, the design
collapses. Visible in storage, prompt, UI, and report. This inverts the field's default,
which resolves by recency.

### 8.3 The transfer check *(the novel contribution)*

One deterministic function over every claim class. Scope gate first, then: required genes
missing from the panel → **blocked**; identical panel hash → **transferable**; otherwise
compare modality, assay family, tissue through the Uberon hierarchy, fixation, and method
family, each mismatch downgrading to **conditional** and naming itself.

A blocked claim is still surfaced with its reason. Silent dropping is the failure mode, not
a safe default.

Runs at retrieval time, so switching dataset mid-session stays correct. This is also what
answers the reference-data trust problem: a CL marker edge is a belief formed under one set
of conditions being used under another, which is exactly what the check is for.

### 8.4 Retrieval

Deterministic, no LLM call before the answer model receives context.

```
seeds from context -> traverse (module's mask, depth <= 2)
  -> collect claims -> transfer check -> hybrid score (BM25 + dense, weighted RRF)
  -> relevance floor -> group by authority -> pack under budget
```

Hybrid, not dense-only: gene symbols break embedding search, so BM25 with HGNC alias
normalisation is the anchor. Below the relevance floor the answer is *no support found*,
stated concretely (the traversal reached these nodes and no literature was retrieved).
Scope filters before retrieval, never after. Per-turn log records the traversal path,
transfer verdicts, and what was retrieved and ignored.

### 8.5 Write path and lifecycle

Two writers only: the deterministic engine writes measurement claims; the biologist writes
lab claims. Reference and literature claims are imported, not written. The agent may
**propose**; only the biologist commits.

No LLM-assisted dedupe, no background consolidation, no auto-editing. Those mechanisms
manage noisy conversational streams and are what destroyed the audit trails of consumer
memory systems. None of Panoscope's claim classes are noisy.

Supersession only by the owning authority. Contradicted lab claims are **flagged for
review**, never auto-invalidated. No decay, no scoring, no merging.

### 8.6 Storage

| Class | Store |
| --- | --- |
| Lab | git, typed JSON, one file per claim |
| Reference view + literature | study SQLite, versioned as a set |
| Measurement | dataset tree, regenerated by the pipeline; claims hold references |

Two append-only logs: usage, and disagreement events. The second is where you find out
whether lab knowledge and the literature are diverging.

---

## 9. Generation

**Grounding gate.** Every number, marker, PMID, note id, and edge id traces to source or
the answer is discarded for the deterministic fallback. Fails closed.

A cited claim must have been **in the packed context for that turn**, not merely exist.
That is the difference between citation existence and citation grounding.

Span-level entailment ships as a **measured flag, not a hard gate**, until its
false-rejection rate is known.

**Answer policy travels separately from context** (strict / balanced / permissive), so the
same retrieved context can be replayed under a different policy. That replay is the
ablation.

### 9.1 Two phases, one mode

| | Corpus build | Chat |
| --- | --- | --- |
| Frequency | once per study | every turn |
| Data touched | public reference + literature | lab claims, results, unpublished context |
| Job | read papers, write span-bound cards | select, assemble, phrase, refuse |
| Model | strongest available | local where privacy requires |

Once the corpus is frozen the chat model reads rather than derives, and the deterministic
core owns every number, so the chat model's job is smaller than it looks.

**Empirical support.** Eywa's diagnostic: Qwen3 32B scored 69.1% over its own memory store
and 79.7% over a Sonnet-written store. Write-path quality lifts a weaker reader ~10 points.

**Caution.** Small models are weakest at refusing when the data cannot settle a question.
MEMIR's backbone comparison shows frontier-class around 48% on BEAM against ~30% for a 14B
open model, widest on contradiction resolution and multi-evidence synthesis — the rival test
and the holistic review.

**Fails safe.** A weaker generator does not produce ungrounded output; the gate fails
closed. It produces more fallbacks and flatter prose. A usability cost, not a correctness
cost, and the measurable consequence is fallback rate.

**One supported mode: hybrid** (cloud build, local chat with escalation for rival test,
holistic review, and reconciliation). Full-cloud is hybrid with one setting changed.
Air-gap is a **property**, not a mode: model selection is config and the corpus is a
versioned artifact, so air-gap is supportable when asked without maintaining a code path
nobody exercises. Unexercised paths rot and then the docs lie.

Generator model, version, and mode go in the report header.

### 9.2 Local model candidates

Version numbering across sources is inconsistent; treat these as families to evaluate, not
pins. Selection criteria in priority order: instruction adherence under a long contract;
citation and refusal behaviour; structured output; context handling at 16–32K rather than
the advertised maximum; licence.

| Candidate | Fit |
| --- | --- |
| Qwen3 instruct (14B dense, or ~30B MoE with ~3B active) | Best default. Apache 2.0, strong adherence, MoE variant reportedly fits ~16GB at Q4. |
| Command R 35B | Built-in citation grounding, returning inline source references. Aligned with click-back. ~24GB. Benchmark despite the size. |
| Gemma (12B / 27B) | Strong adherence, runs on one well-specced Mac. Licence is not Apache 2.0. |
| Phi / Mistral Small (~4–8B) | Constrained hardware fallback. Higher fallback rate; not for rival testing. |

Embeddings pinned and recorded in the corpus version, since they are part of the
reproducibility surface.

---

## 10. Licensing

Ship the code, not the data.

| Resource | Licence | Use |
| --- | --- | --- |
| Cell Ontology, Uberon, GO | CC BY 4.0 | safe; attribution only |
| HGNC | freely available | alias normalisation |
| MSigDB | CC BY 4.0 overall | Hallmark fine; `KEGG_MEDICUS_` C2 sets are CC BY-SA, never bundled in an MIT repo |
| PanglaoDB | paper is CC BY 4.0 | confirm the tables before redistributing |
| CellMarker | no explicit redistribution licence found | fetch, do not bundle |
| CL-KG extension | expected CC BY, unconfirmed | swap in when it lands |

Fetch at setup, cache locally, record **source, version, and licence per claim**. Falls out
of the architecture for free since claims already carry provenance. `--official-sources-only`
filters on the source column.

---

## 11. Interface

### 11.1 Fixed shell, module-supplied panels

Fixed: the unit rail, the call or summary header, the chat pane, the claims drawer, report
export. Module-supplied: evidence table columns, the primary plot, detail plots, what a
pin selects and which views respond, and the cross-unit overview.

| | Assignment (markers) | Characterisation (enrichment) | Comparison (DE) |
| --- | --- | --- | --- |
| Rail | clusters | units from the result | contrasts |
| Header | suggested cell type + band + verify | per-item counts: enriched, suggestive, untestable | per-item counts: up, down, untestable |
| Table | gene, statistic, measured, role | program, z/q, K of N, leading edge | gene, log2FC, p, direction |
| Primary | evidence table + spatial | panel coverage per program | volcano |
| Detail | density, cell map, UMAP | leading-edge genes with their own marker stats | per-group distributions |
| Overview | unit composition | program × unit heatmap | contrast summary |

The program-by-unit heatmap earns its place specifically: it is how a program enriched
everywhere is seen as atmospheric rather than unit-specific.

**Pin generalises.** Pin means a selected subject — a gene, a program, or a contrast — and
the module declares which views respond. Same state field, different semantics per module.

### 11.2 Every plot carries its conditionality

The panel-absence rule applied to figures, and the requirement most easily missed.

A volcano plot on a targeted panel looks exactly like one on whole transcriptome, and a
reader infers genome-wide coverage unless the figure says *280 of ~20,000 genes tested,
panel-scoped*. An enrichment bar needs its K of N. A density hexbin needs its bin size in
µm and that it is area-normalised. A UMAP needs which embedding produced it.

A misleading figure travels further than a misleading sentence, because it goes in a talk.

**Companion rule: no plot computes a statistic that is not already in the normalised
evidence.** Plots render, they do not derive. The visual analogue of the grounding gate.

### 11.3 Click-back

Every sentence resolves to the claim it came from, with that claim's authority and
conditions visible. A sentence with nothing to click should not have been rendered.

### 11.4 One cross-method plot

Where a unit has both DE and spatial markers, plot log2FC against the spatial coefficient.
Genes high on one axis and low on the other are exactly the distinction jazzPanda exists to
make. Belongs to reconciliation, not to either module.

---

## 12. Data flow

```
STUDY    tissue + research background + one or more datasets
         -> reference view built from the shared cache (per tissue)
         -> literature corpus fetched per triple, versioned

PREP     vendor output (Xenium | MERSCOPE | CosMx)
         -> spatialdata_io.<platform>() -> write zarr -> read back
         -> validate: table, var_names, cluster column, coordinate system,
            clustering mode declared

RUN      per dataset: results -> unit detection -> modules -> normalised evidence
         -> task-specific interpretation -> conditionality checks
         -> adjacency + concordance -> reconciliation -> propagation
         -> interpretation record written per unit (§15)
         (deterministic, offline, reproducible)

REVIEW   chat + task-specific panels + spatial views
         -> traversal retrieval -> transfer check -> grouped by authority
         -> biologist confirms, overrides, or commits a lab claim

REPORT   assembled from durable artifacts, no live network
         header: study, dataset, platform, panel hash, zarr hash,
                 reference releases, corpus version + staleness,
                 generator model + version, spec version
```

---

## 13. Explicitly refused

| Refused | Why |
| --- | --- |
| Running the analysis | Makes confidence self-referential |
| LLM-extracted reference edges | Fabrication surface; edges are imported |
| Reference edges as evidence | Marks ≠ localizes |
| "Agree" between results answering different questions | Category error |
| LLM confidence blended into a score | The model must not move a number |
| Ensemble voting across methods | Obscures which evidence did the work |
| Weighting evidence by citation count | Popularity is not validity |
| Background consolidation or memory synthesis | Destroys the audit trail |
| LLM rewriting or deduping claims | The agent does not author the lab's knowledge |
| Recency-based conflict resolution | Wrong authority order for this domain |
| Treating replicate agreement as corroboration | Not independent witnesses |
| Pooling clusters across independently-clustered samples | Same label, different objects |
| Carrying a call across datasets by label match | Panels differ; the transfer check decides |
| A unit-level confidence band on characterisation or comparison | No label space, so the band is meaningless |
| A plot that computes its own statistic | Plots render, they do not derive |
| Bundling restrictively-licensed data | Fetch, don't ship |
| Guessing a result's unit or a dataset's clustering mode | Fail loudly instead |
| A model narrating the interpretation record after the fact | It would drift from what the code did |

---

## 14. Evaluation

**Seeded-error detection.** Swap labels, assign unsupported calls; detection rate at a
fixed false-flag rate on clean data. Ground truth is free to generate, which makes this the
highest-value eval in the system.

**Transfer-check accuracy** against expert judgment, on lab claims and reference edges.

**Sycophancy rate.** Lab claim wrong, data disagrees: surfaces tension or capitulates?
Nobody in this space measures this.

**Generator ablation.** Gate constant, model swapped. Gate rejection rate, citation support
rate, refusal accuracy on *not measured* and *no support found*, fallback rate.

**Refusal rate next to accuracy.** A gate that fails closed can look safe while refusing.

**Failure taxonomy** for diagnosis rather than one score: coverage, grounding, revision,
scope, temporal, retrieval, synthesis, measurement.

**Expert concordance**, and a human-annotator comparison if time allows.

---

## 15. The interpretation record *(new)*

A per-unit artifact written **by the deterministic core at run time**, recording how the
result file was read and why the interpretation came out as it did.

Distinct from three things that already exist: the retrieval log is per turn and
machine-facing; the report shows conclusions; `panoscope explain` (§16) reconstructs on
demand for a developer. The record is the layer between the data and the conclusion, it is
written once when the run happens, and its reader is the biologist.

### 15.1 Two parts

**How the file was read.** Every interpretation rests on parsing decisions that are
invisible today and are where silent errors live:

- which column was taken as the statistic, and which as the unit
- what the unit was detected as, and on what evidence (§7.2)
- rows dropped, and why
- genes in the result but not on the panel
- genes in the result but not in reference data
- the platform adapter's derived measured set and its size
- clustering mode as declared

If Panoscope reads `avg_log2FC` where the biologist expected `avg_logFC` from a different
Seurat version, this is the only thing that catches it.

**Why the interpretation came out that way.**

- the rubric applied and the threshold crossed
- drivers with their numbers
- which conditionality checks fired, and their outcomes
- rivals considered and what separated them
- what could not be settled, and why
- claims used, with their authority and transfer verdicts

### 15.2 Two layers, never blurred

The **factual layer** is recorded, not generated: it is a byproduct of the deterministic
core running, available with no LLM involvement, and identical in a literature-off or fully
local run.

The **interpretive layer** is generated prose sitting on top, marked as generated, and
subject to the grounding gate like any other output.

**The constraint that makes this work: the record is written by the core, not narrated by
the model afterwards.** A model summarising what it thinks happened is a weaker artifact
that can drift from what the code did, which defeats the purpose. Core functions return
their decisions alongside their results; no separate reconstruction pass.

That is a small refactor done early in the type contracts and an annoying one done late,
which is why it belongs in the contracts work, not in hardening.

### 15.3 What it buys

**A biologist can check the tool understood their file.** This is the first thing anyone
should verify and currently nothing supports it.

**A reviewer asking why unit 4 got that call** is answered by pointing at an artifact
rather than a rerun.

**Reproducibility with teeth.** Two runs on the same input produce byte-identical factual
layers, so a diff localises any change to the input file, a reference release, or the code.
This makes the reproducibility claim checkable rather than asserted, and it costs almost
nothing because the engine is already deterministic.

**Invariant:** the factual layer of the record is deterministic. Same inputs, same bytes.

---

## 16. Operational requirements *(new)*

The design has been ahead of the engineering. These are the parts that make it runnable.

**Failure and recovery.** Corpus build is resumable and checkpointed; PubMed rate limits
back off rather than abort; a corrupt zarr fails at validate with a named cause; a claim
referencing a superseded corpus version resolves against the recorded version or reports
that it cannot; concurrent claim edits are a git merge, so claim files are one-per-file and
never rewritten.

**Artifact versioning.** Every artifact carries a `spec_version`. A study built under 0.1
must open under 0.3 or the lab's accumulated knowledge dies at the first upgrade, which
destroys the compounding claim. Migrations are code, tested, one-way.

**Observability.** `panoscope explain <unit>` renders the interpretation record (§15) for
a developer, plus live state the record does not carry. The record is the durable artifact;
`explain` is the on-demand view. Structured logs with a correlation id per turn.

**Performance budgets**, declared and tested rather than discovered:

| Operation | Budget |
| --- | --- |
| Reference view build (one tissue) | < 2 min, cached thereafter |
| Corpus build (~20 cell types) | < 30 min, resumable |
| Dataset load (5k panel, 500k cells, no images) | < 10 s |
| Retrieval per turn | < 300 ms |
| Full deterministic run per dataset | < 5 min |

**Contract tests** run against every module and adapter, so adding one proves it conforms.
**Golden-output test**: full pipeline on the demo, snapshot the report, diff on every
change. **Property tests** for stated universals, such as reconciliation never raising
confidence.

---

## 17. Build order

| # | Work | Size |
| --- | --- | --- |
| 1 | Packaging, `CITATION.cff`, tagged release, Zenodo DOI | 1 day |
| 2 | `types/` contracts frozen (claim, conditions, evidence, task outputs, **decision records**) | 1–2 days |
| 3 | Xenium adapter + SpatialData reader + prep stage | 3–4 days |
| 4 | Method module refactor: jazzPanda + enrichment, task kinds, unit detection | 3 days |
| 5 | Claim schema + transfer check + store | 3 days |
| 6 | Reference data: sources cache + reference view + traversal | 4 days |
| 7 | **Thin-slice validation on a second dataset with a different panel** | 2 days |
| 8 | DE module + rubric divergence test + data-derived rivals | 3 days |
| 9 | Literature corpus per triple + span binding | 3 days |
| 10 | Adjacency, concordance, reconciliation, propagation | 3–4 days |
| 11 | Task-specific UI panels + record viewer | 5 days |
| 12 | Operational hardening (§15) | 1 week |
| 13 | Evaluation | ongoing |

**Item 7 is the decision point, not a milestone.** It exercises the novel contribution end
to end and will say more in two days than another design round. Specifically: does the
transfer check fire usefully, or does it fire constantly? If constantly, the central claim
needs rethinking before it goes in a paper.

**Conference vs paper.** November needs the story and a working build, both of which exist.
Items 1 to 4 are the ones worth doing soon.

---

## 18. Open decisions

1. **DE as assignment.** People do use DE output for cell typing. If supported, task kind
   is a property of *how a result is used*, not of the method, and the contract changes
   shape. Settle before the DE module.
2. **DE rival checks.** Are per-sample consistency, composition shift, and batch
   confounding computable from a typical DE result table, or do they need the expression
   matrix? Test against a real result before specifying.
3. **Report granularity** — per dataset, or study-level aggregation.
4. **Sample views** — one at a time with a switcher, or side-by-side.
5. **Demo scope** — restore sample2 (cheaper, exercises multi-sample) or add a second
   dataset (exercises the transfer check, worth more).
6. **Marker edge source** — CL + CellMarker now with CL-KG as a swap-in, or wait. Worth
   emailing the CL team; a downstream tool adding panel-conditional checking is a use case
   they would want to know about.
7. **Reference cache on HPC** — a per-user home directory cache may be wrong for WEHI; a
   shared read-only path needs a different resolution strategy.
8. **Local chat model** — Qwen3 as the Apache default, or Command R for native citation
   grounding at a memory cost. Benchmark on the eval turn set.
9. **Capitalisation** — `PanoScope` vs `Panoscope`, still unresolved in the repo.

---

## 19. The claim

An interpretation layer in which every claim carries the measurement conditions under which
it was formed, and applicability is checked deterministically at retrieval rather than
assumed. Task-aware across assignment, characterisation, and comparison, with a claim store
that treats method output, reference data, literature, and lab knowledge as one object type
ordered by authority, and a swappable generator that makes the guarantees measurable.

Not another autonomous spatial agent. A human-in-the-loop interpretation layer that is
independent of the method it audits, and that hands the decision back.
