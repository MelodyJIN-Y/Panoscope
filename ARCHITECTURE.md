# Panoscope architecture

**A human-in-the-loop interpretation layer for bioinformatics results.**

Version 5, September 2026. Supersedes `docs/archive/ARCHITECTURE_v4.md`, which superseded
`docs/archive/BLUEPRINT.md`. Sections marked *built* exist today; *new* is proposed work.

---

## Changes from v4

Four are structural, and **two are retractions**. The retractions are the important part: both
were found by measuring, before review rather than during it.

**The transfer check does not work as specified, and we can now prove it.** v4 called conditional
validity "the novel contribution". Measured against the data on disk (§7.4), the check returns a
**constant** on curated claims — 0.000 bits of entropy across 280 vendor panel edges, with rules 3, 4,
6 and 7 firing on 100% of them. It carries real information only on lab and cross-dataset measurement
claims. v5 splits the function, restricts the verdict-bearing half to claims whose conditions come
from a manifest, and adds the rule that actually discriminates: **detectability**. The claim that
survives is stronger than the one it replaces, because it is measured rather than asserted.

**The gate is not justified by out-reasoning the model, and we can now prove that too.** The panel
ablation (§16.2) was placed before the engineering so a negative answer would arrive cheaply, and it
did: telling a model the 280-gene panel drops false claims about the assay from **94.8% of calls to
1.5%**, and adding an explicit rule takes it to zero. So the warrant for the deterministic layer is
not reasoning quality — it is the **systems guarantee that the measurement conditions are present and
correct**, which a prompt has only when someone remembers and the architecture has structurally. The
bare-arm 94.8% survives as a measured fabrication rate about experimental conditions, pending one
open check: whether published tools put the gene universe in the prompt at all.

**Everything extensible becomes a registry with a declared contract** (§4). Workflows, methods,
platform adapters and reference sources all have the same shape: declare, register, and the core does
not change. This replaces v4's fixed `methods/` list and its two-source reference layer.

**Workflow and method separate** (§4.1). The *interpretation workflow* — marker interpretation, DE
interpretation, enrichment interpretation — owns the interpretive question and the review surface. A
*method* registers into a workflow and owns its own assumptions, statistic semantics and caveats. One
workflow, many methods. v4 conflated the two, which is why `skills/jazzpanda-markers/` is named after a
method rather than a question.

**Study intake is a first-class stage** (§6). v4 stipulated `Conditions` in the type system and never
said who populates it. Nothing did. Intake is that step.

**The interface moves into the interpretation contract** (§13). v4 treated `UISpec` as a UI concern. In
v5 a workflow *declares* its review surface, because the review surface is how the biologist audits the
interpretation — it is part of the contract, not a rendering of it.

Two scope decisions are recorded in §15 (Explicitly refused): the novelty axis, and SVG.

---

## 1. What Panoscope is

A biologist has a result produced by someone else. Panoscope reads it faithfully, says what it supports
and what it does not, and lets the biologist argue with it.

Two boundaries define the tool.

**Panoscope does not run the analysis.** Method output is a consumed input, never recomputed. This
makes the confidence assessment an audit rather than a self-assessment: a tool that produces a result
and then scores its own confidence is grading its own work.

**The biologist makes the call.** Enforced in the write path (§10.5): only the biologist commits a lab
claim, and no claim of any kind can move a number.

---

## 2. Core principle: every claim carries its measurement conditions

The panel-absence rule says an off-panel gene was never measured, so its absence is not evidence. The
architecture is that rule generalised.

A claim is never simply true. It was established under conditions: an assay, a modality, a tissue, a
panel, a segmentation, a statistic. Using it elsewhere is an inference.

| Claim | Established under |
| --- | --- |
| jazzPanda: transcripts localize to cluster C | this panel, platform, segmentation, spatial GLM |
| Cell Ontology: gene X marks cell type Y | flow cytometry, dissociated cells, a tissue |
| Literature: this marker is unreliable in FFPE | a specific tissue and modality |
| Lab: POSTN here is tumour bleed | a 280-gene Xenium panel, breast, this dataset |

They are one object type (§10). Two questions follow:

- Is this claim still **true**? Temporal validity. Every memory system handles this.
- Is this claim still **applicable** here? Conditional validity.

v4 claimed the second question as novel. It is not — GRADE's *indirectness* domain formalised it in
evidence-based medicine two decades ago, and Pearl & Bareinboim's transportability is its causal
formalism. **What is defensible is enforcement**: conditions carried on every claim and checked
deterministically at retrieval, with refusal surfaced rather than silent, plus an empirical
characterisation of when that enforcement discriminates and when it degenerates. Both halves matter,
and the second half is in §7.4.

### 2.1 The rule v4 stated wrongly

v4 §5.4 said reference edges "may never move a confidence band". The code cannot honour that and
neither could any implementation. `agent/verdict.py` selects the driving marker by canonicity, and the
driver sets the band — swap the canonical marker list and the band changes. A weak canonical marker is
additionally demoted out of the driver pool entirely.

This is unavoidable, because you need curated knowledge to know which genes are candidate markers at
all. The rule that is both true and enforceable:

> **Reference data determines which question is asked of the measurement. The measurement alone
> determines the answer.**

Reference data may select and order evidence, generate rivals, supply canonical lists, scope the
corpus, and carry citations. It may never supply a value, count as agreement, or corroborate a
measurement.

### 2.2 What the knowledge apparatus buys

Stack the hard rules — reference never corroborates, literature never corroborates a measurement,
replicate agreement never raises confidence, reconciliation can never raise confidence, no claim moves
a number — and one consequence follows that v4 never stated: **the band is a function of a single
method's output, and every other layer can only subtract or flag.** The only promote in the engine is a
second statistic from the same measurement (high spatial specificity on the driver).

That is a deliberate design, and it must be stated rather than discovered. What the apparatus buys is
the **flags**, the **named rivals**, the **undecidables**, and the **blocked-with-reason** — not the
number. A reviewer will ask; the answer should be in the abstract.

---

## 3. Three layers

```
  EVIDENCE            immutable, external, versioned
    method results · SpatialData store · reference data · literature corpus · lab claims
                              |
  DETERMINISTIC CORE  owns every number and every gate, zero LLM calls
    platform adapters + method modules -> normalised evidence
    measurement status · task-specific interpretation · evaluability · transfer check
    rival tests · reconciliation · propagation · claim store · retrieval
                              |
  GENERATION          fenced and swappable
    reads the two layers below, phrases what it is handed
    grounding gate · answer policy · model routing · deterministic fallback
```

The boundaries exist so the generator can be swapped while everything else is held constant. That is
both the local-model story and the ablation — and it produces the result in §16.3.

---

## 4. Everything extensible is a registry *(new)*

Four registries, one shape. A member declares its contract; the core does not change when one is added.

| Registry | A member declares | The core stays fixed because |
| --- | --- | --- |
| **Workflows** | the interpretive question, the rubric shape, the review surface | the shell renders from a declared spec |
| **Methods** | assumptions, input schema, unit detection, what the statistic measures and **what it does not**, rival space | everything downstream reads normalised evidence |
| **Platform adapters** | how to read the store, how "measured" is derived, available views, coordinate system, clustering mode, standing conditions | conditionality reaches the data automatically |
| **Reference sources** | node and edge types, conditions, provenance, licence | claims already carry provenance |

**Contract tests are parameterised over every registered member**, so adding one proves conformance.
That is what makes extensibility a claim rather than an aspiration.

### 4.1 Workflow and method

A **workflow** is an interpretive question. A **method** is one way of producing evidence that the
question can be asked of.

```
skills/marker-interpretation/SKILL.md        <- the question: what IS this unit?
    methods/jazzpanda.py                     <- spatial GLM
    methods/seurat_de.py                     <- cluster-vs-rest mean expression

skills/enrichment-interpretation/SKILL.md    <- the question: what is RUNNING here?
    methods/jazzpanda_cgst.py
    methods/ora.py

skills/de-interpretation/SKILL.md            <- the question: does this result MAKE SENSE?
    methods/limma_voom.py
    methods/edger.py
```

A method module is **a machine-readable declaration of what a method's paper says it measures and does
not measure.** Adding a method means reading the paper and the implementation, stating the contract,
and registering — not writing interpretation logic. That is the extensibility claim, and it is why the
registry table above lists "what it does not measure" as mandatory.

Two methods inside one workflow is what turns the abstraction from design into demonstration. It also
exercises the rubric-divergence invariant (§9.4) and produces the cross-method figure (§13.6), so it is
one piece of work that pays three times.

### 4.2 Reference sources are pluggable, not a fixed set

v4 shipped "two source kinds, deliberately". v5 ships a **source interface** and some standard sources.
The reason is that spatial work frequently needs a tissue-specific atlas that no general ontology
covers, and that should be a registered source carrying its own conditions and licence — not a special
case in the core. Start standard, update often, let a lab add what its tissue requires.

---

## 5. Evidence

### 5.1 Spatial data: SpatialData (scverse)

The on-disk format for spatial evidence is a SpatialData `.zarr` store, replacing the hard-coded tidy
layer.

| Was | Is | What it fixes |
| --- | --- | --- |
| `cells.csv` | Shapes (GeoDataFrame) | real geometry |
| `transcripts.csv.gz` | Points | lazy, no full read |
| `umap.csv` | `table.obsm["X_umap"]` | standard location |
| `cluster_key.json` | `table.obs[<col>]` | derived, not hardcoded |
| panel TSV | `table.var_names` | the panel *is* what was measured |
| `density/*.parquet` | computed from Points | bin size in real µm |

The panel-absence rule becomes structural: the measured set is `var_names`, true by construction on
every platform. Coordinate systems give physical units, so hex bin size and the adjacency check (§9.9)
work in micrometres rather than invented thresholds.

Constraints: write to Zarr then read back; read with `selection=[shapes, points, table]` and never
images; behind a `panoscope[spatial]` extra with the legacy tidy reader retained; a `.zarr` is a
directory, so the demo store ships image-free or comes from Zenodo; reader drift is real, so the prep
stage validates what Panoscope depends on and fails loudly.

**Sequencing note.** The existing tidy CSV/parquet contract already handles the second dataset. The
SpatialData reader is real capability but it is not on the critical path for the multi-dataset claim —
build `platforms/base.py` plus a tidy adapter first, and validate the zarr reader against the tidy
adapter's output on the same study.

### 5.2 Other evidence

**Method results.** As given. Never recomputed, kept **separate from the SpatialData store**: the store
is what was measured, the result is what someone computed from it, and that boundary is what the audit
claim rests on.

**Reference data** (§7), **literature corpus** (§8), **lab claims** (§10).

**Rule: evidence precedes belief.** A derived object can be revised or discarded. The evidence it came
from is never rewritten.

**Export exception.** Confirmed calls may be written back as an `obs` column on export for
napari-spatialdata and squidpy interop. An output artifact, never a write into the store.

### 5.3 Study, dataset, sample

| Level | Is | Owns |
| --- | --- | --- |
| **Study** | the unit of work | tissue, research background, reference view, literature corpus, study-scoped claims |
| **Dataset** | one SpatialData store | one panel, one platform; chat history and review state |
| **Sample** | one replicate / section | its own coordinate system and shapes |

"Session" is avoided: it implies transient, and this persists.

**Scope hierarchy = conditions hierarchy.** This is why the levels fit rather than bolt on:

| Claim scope | Crosses | Transfer check |
| --- | --- | --- |
| sample | nothing | not invoked |
| dataset | samples, same panel | neutral; section differences noted |
| study | datasets, panels may differ | fires |
| lab | studies | fires hard |

**Clustering mode must be declared.** Under joint clustering, c2 is the same population in every sample
and evidence pools. Under independent clustering, c2 in two samples are unrelated objects sharing a
label and pooling is wrong. An undeclared mode is a **validation failure**, not a default.

---

## 6. Study intake *(new)*

The step that populates `Conditions`. Every applicability check in the architecture reads that object;
before v5, nothing wrote it.

### 6.1 What it produces

**Structured artifacts, not a prompt.** `Conditions`, the reference view, the literature corpus, the
study background — with the prompt *assembled* from them at call time (§11). A stored prompt string is
not inspectable, versionable, diffable or citable; the artifacts are, and they can be corrected later
without re-running the expensive step.

### 6.2 Provenance on every field

Intake is where fabrication could enter the *foundation*. If a model infers "probably breast" from gene
names and that becomes a condition, every downstream check is built on a guess. So each field records
how it was obtained:

| Provenance | Meaning |
| --- | --- |
| `stated` | the biologist said it |
| `read` | parsed from a file — the panel, the manifest, the result header |
| `inferred` | the model proposed it and the biologist confirmed |
| `unknown` | explicitly not known |

`unknown` is first-class and propagates; it is never a silent default. This is the panel-absence
discipline applied to the study's own metadata — *not stated* ≠ *not true* — and it is deliberately the
same four-state shape as the measurement lattice in §9.5.

### 6.3 What blocks

| Blocks the study | Declared-unknown allowed |
| --- | --- |
| the measured set (structural — it *is* the panel file) | fixation |
| clustering mode, when more than one sample | segmentation |
| tissue (the corpus scoping key) | |
| platform / assay | |

Blocking is consistent with the fail-loud discipline elsewhere: an undeclared clustering mode is
already a validation failure. The line exists so intake does not teach people to guess — a biologist
working from a collaborator's section may genuinely not know the segmentation version, and forcing an
answer is worse than a declared unknown. Downstream, a check reading an unknown field returns *cannot
be settled* and names what is missing.

### 6.4 Interaction: checklist, then a short targeted chat

The failure mode is a biologist losing patience in a twenty-question interview.

1. **Read everything derivable from the files first.** Zero questions for anything already on disk.
2. **Show a checklist** — what is known, what is missing and blocking, what is optional. The biologist
   ticks what he can supply. This inverts control: he declares what he has rather than being driven
   through a script.
3. **Chat only about ticked items and blocking gaps**, propose-and-confirm rather than interrogate.
4. **Stop early.** Everything optional is skippable and addable later; the study artifact is editable.

Optional context, gathered when offered: research background; papers of particular interest (cheap for
the biologist, high value for the corpus); lab conventions and known caveats, stored as lab-scoped
claims at setup rather than surfacing mid-session; and expected cell types — **stored as a claim to
test, never as context feeding the call.** A stated prior the agent can agree or disagree with is
valuable; the same text as input would anchor the interpretation and defeat the independent audit. It
also provides a sycophancy probe on real data (§16.5).

### 6.5 Where it runs

**Two paths, one contract.** Intake ships as a standalone skill the biologist runs in their own agent
client, *and* as an in-app path with a bring-your-own key. The study artifact is the interface between
them, so the app's API key is genuinely optional rather than nominally optional.

The skill path is the cleanest compliance position available (see `docs/llm-access.md`): the biologist
authenticates themselves, to their provider, in a first-party client, with their own credential;
Panoscope never collects, stores or intermediates it, and resells nothing.

**One caveat changes the design.** The skill path does not carry the grounding guarantee — in the
biologist's own client there is no deterministic core running, only a prompt asking a model to behave.
So the grounding checker ships as an executable inside the skill folder, and **the study artifact
carries its verdicts** along with provider, route, model id and query date. Without that, the two paths
emit indistinguishable outputs of very different trustworthiness.

---

## 7. Reference data

Structured, finite, versioned, shared. **It resolves, it does not retrieve.** No ranking, no relevance,
no fetch at question time. A query either has an answer or it does not.

### 7.1 Three tiers

| Tier | Scope | Lifecycle |
| --- | --- | --- |
| **Sources** | global, shared across studies | fetched once per upstream release, cached under a configurable root |
| **Reference view** | per study | the traversed subgraph for this tissue; derived, rebuildable |
| **Lab claims** | per study, accumulating | authored, owned, irreplaceable (§10) |

Only the third needs backing up, and only the third compounds.

### 7.2 What it answers

Which cell types occur in this tissue. What are the canonical markers for fibroblast. Is FAP on this
panel. Which gene sets share genes with this one. Is this symbol an alias.

**Most of what makes Panoscope work is a structural lookup, not a retrieval.** Panel absence, rival
candidates, off-panel canonical lists, set overlap. Which means literature-off is a degraded mode, not
a broken one, and the deterministic guarantees survive it intact.

### 7.3 Status: candidate source, never evidence

Reference data says gene X *marks* cell type Y, mostly established in dissociated scRNA-seq, flow or
IHC. A spatial method says gene X's transcripts *localize* where cluster C's cells are. Different
claims. See §2.1 for the rule that governs their use.

**Coverage is uneven**, especially for tissue-specific and disease-associated types. A missing node
reads as *not in reference data*, never as *does not exist* — the panel-absence rule applied to the
graph.

**Assay knowledge is where the differentiator lives and where curated sources are weakest.** "POSTN
bleeds in dense stroma on this panel" is what a biologist knows and no database holds. So the sparsest
knowledge kind is the one the feedback loop matters most for.

### 7.4 The measured behaviour of conditional validity *(new — this section is a result)*

Run against the two files on disk, with the seven v4 rules implemented as specified:

| Claim set | n | H(verdict) | What drives it |
| --- | --- | --- | --- |
| Vendor panel edges (curated) | 280 | **0.000 bits** | nothing — every claim returns `conditional` |
| Hand-picked canonical markers (curated) | 60 | 0.754 bits | **entirely rule 1** — the panel-absence rule already shipped |
| Lab notes (synthetic, realistic) | 24 | **1.561 bits** | rule 1 dependencies, rule 5 tissue, rule 7 method |

Rules 3, 4, 6 and 7 fire on **100.0%** of the vendor set; rule 5 on **0.0%**. `H(mismatch-tuple)` is
also **0.000**, so returning named deltas instead of a label does not rescue it — it is the same four
strings every time.

A counterfactual sweep confirms this is structural rather than a sampling artifact: holding claims
fixed and varying context across Xenium/breast/FFPE, fresh-frozen, CosMx/lung, Visium, IHC and matched
scRNA-seq, five of six return `conditional`, and the only `transferable` is the degenerate case where
the context *is* the claim's formation conditions.

**And the verdict predicts nothing, while the measurement's own statistics do:**

| Predictor of cross-lineage discordance | AUROC |
| --- | --- |
| transfer verdict | **undefined** (constant) |
| `1 − pearson` | **0.667** |
| `1 − glm_coef` | 0.632 |
| `max_gc_corr > pearson` | 0.584 |

### 7.5 What replaces it

**Split the function.**

- **`evaluability(claim, context)`** — a *capability* gate. Are the entities this claim depends on
  measured here? Applies to every claim in every class. This is the panel-absence rule generalised
  (§9.5), and it is a fact about the instrument, not a judgment about the claim.
- **`transfer_check(claim, context)`** — applies **only** to claims whose conditions come from a
  manifest: lab, measurement, and dataset-scoped curated sets such as a vendor panel annotation.
  Database-curated and literature claims receive a **provenance stamp**, not a verdict.

This also answers the objection that a deterministic check over model-extracted conditions is not
deterministic: the verdict-bearing half never runs on extracted conditions.

**Never report "% conditional" as a result.** It will be constant. Report the named condition deltas,
and render the class-level constants once as a banner — *"curated edges in this corpus were established
in dissociated scRNA-seq on fresh tissue; you are in FFPE in-situ"* — rather than repeating them on
every claim.

**Add detectability.** 49 of 280 panel genes (17.5%) yield no usable spatial signal. A curated edge on
such a gene passes the dependency gate and is still unusable. As a capability frame this carries
**0.652 bits** on curated claims — unlike the seven original rules, it actually varies. It is
deterministic, IO-free, dataset-specific, and invisible to every rule v4 specified.

**Check the demotion floor.** With an unbounded stream of conditional claims, "confidence can only go
down" collapses everything to Low. Either demonstrate band-distribution stability as the claim store
grows (10 / 100 / 1000 claims) or adopt a documented one-demotion-per-condition-class rule. v4 did not
address this and a reviewer will.

### 7.6 The claim that survives

> Curated marker→cell-type edges cannot be adjudicated by their formation conditions — that check is a
> constant. They can be adjudicated by the measurement itself. So the interpretation layer must let the
> measurement overrule curated knowledge, never the reverse.

That is §2.1's rule, derived instead of stipulated.

---

## 8. Literature corpus

Unstructured, unbounded, per-study, never complete. Ranked rather than resolved, goes stale, and its
licence situation differs entirely from reference data because it stores retrieved spans rather than
redistributing a database.

**Scoped by reference data.** The traversal decides which triples to fetch, so reference data is a
prerequisite. Build in that order.

### 8.1 The join

The key is the **(gene, cell type, tissue)** triple, not the gene. POSTN in breast is a different
question from POSTN in kidney.

```
tissue -> candidate cell types -> marker genes (incl. off-panel)
       -> literature per triple -> attached to the reference edge
```

Off-panel canonical markers arrive because they are edges from the cell type, not because they appeared
in the results. Cost scales with the candidate cell-type set, not with panel size.

### 8.2 What the disagreements buy

- **Edge with strong tissue-specific literature** — a well-supported prior
- **Edge with thin or absent literature here** — a weak prior, and it should read that way
- **Edge whose literature qualifies it** — the qualification travels with the edge every time it is used
- **Edge the tissue literature contradicts** — surfaced as tension, not resolved

**Caution.** Attaching literature to an edge can look like validation. A paper retrieved for a triple is
topically relevant, not confirmation the marker works in your assay. The paper's own context is
displayed and a tissue or modality mismatch is visible, or this makes weak edges look strong.

### 8.3 Storage and refresh

Records and spans in the study's SQLite store, versioned as a **corpus version** recording build date,
tissue, embedding model, query log and reference releases used. Rebuilt wholesale. Staleness is a
policy, not an accident: a corpus older than a threshold is flagged in the report header.

---

## 9. Deterministic core

Zero LLM calls. Fully testable. This is where the guarantees live.

### 9.1 Interpretation is not one task

| | Marker | Gene-set enrichment | Differential expression |
| --- | --- | --- | --- |
| Question | what *is* this? | what's running here? | does this result make sense? |
| Task kind | **assignment** | **characterisation** | **comparison** |
| Answer | one label from a label space | ranked annotated set | ranked annotated set, with direction |
| Confidence means | confidence in the call | per-item support and testability | per-item support and effect |
| Verify flag | per unit | per item | per item |

For markers there is a label space and the job is choosing from it, which is why a band, a rival test
and a per-unit verify flag all make sense. For the other two there is no label space and no single
answer, so a unit-level band is not merely unhelpful — it is undefined. What the biologist needs is
per-item support plus an honest account of what could not be tested.

**Task kind belongs to the method, not to the use.** A DE module is always `comparison`. If someone
wants cell typing from DE output, that is a second method registered into the marker workflow, not a
second reading of the same result. This keeps the registry flat and `interpret()` single-signature.

**Three output types collapse to two.** `Assignment` has a label space and therefore a band;
`Finding` covers characterisation and comparison, which were structurally identical in v4 and differed
only by an optional direction field. The boundary is *does this task produce a label from a label
space?* — which makes v4's `test_no_band_on_non_assignment` a guard on the type rather than a
behavioural test.

**No fourth task kind.** The three partition the interpretive question, and the partition is complete
for the whole analysis roadmap: differential abundance is a comparison over cell-type subjects;
neighbourhood and co-localisation are characterisation with pair-shaped subjects; ligand-receptor is
either, depending on whether the question is "who is talking" or "what changed". Niche *decomposes*
rather than breaking the partition — characterising a niche by composition is characterisation, naming
it ("tumour–immune boundary") is an assignment that correctly gets a band. That the deciding axis cuts
*through* an analysis family and lands correctly both times is the evidence the partition is real.

Only `TaskKind` is frozen. `unit_kind`, subject shape and method stay open registries. Do not reserve an
unused fourth member: dead on-disk vocabulary that every future reader must handle.

### 9.2 The unit is read from the result

Enrichment might be keyed on cluster, on cell type, or on region. DE might be condition versus condition
across all cells, within each cell type, or cluster versus rest. **The module reads the unit from the
result schema.** It cannot be declared in advance by the method or the platform.

An unrecognised result shape **fails loudly** rather than being guessed at.

### 9.3 Platform adapters

One per technology. Declares how to read the store, **how "measured" is derived** (Xenium puts negative
controls in `var`, so the rule differs per platform), which views are available given the elements
present, the coordinate system and units, the clustering mode, the samples, and **standing assay
conditions** attached to every measurement claim from this platform.

That last item is how §2's conditionality reaches the data automatically rather than by hand.

### 9.4 Method modules

Declares: input schema and unit detection, task kind, what its statistic measures, **what it does not
measure** (mandatory), how absence reads, its own confidence rubric, standing limitations, its rival
space, which reference traversals are meaningful, and which disagreements with other families are
expected.

Everything downstream reads a **normalised evidence object**, never a raw method table.

**Invariant.** A spatial-localisation rubric and a mean-expression rubric must produce different bands
on a constructed case where fold-change is high and spatial localisation is low. If they agree, the
framework is method-blind and the value proposition inverts. This is the single most important test in
the DE work, and it should be written *before* the DE module.

### 9.5 Measurement status: the panel-absence rule generalised *(new)*

The codebase currently contains **three independent implementations of this one idea** — a marker role
enum, an enrichment `untestable` tier with K-of-N coverage, and a discriminator that nulls statistics
for off-panel genes. Three vocabularies, three test suites, one rule. Unifying them is the highest-value
item in the rewrite, because that rule is the project's headline claim.

> A negative observation is evidence only if the measurement was **capable** of producing a positive
> one. Capability is a property of the apparatus, declared *before* the result is read. Where capability
> is absent, the negative is not weak evidence — it is **no evidence**, and it must be reported as such
> rather than dropped.

Four states, not two:

| State | Meaning | Absence is evidence? |
| --- | --- | --- |
| `NEVER_MEASURED` | not in the capability frame | **no** |
| `UNDERPOWERED` | in the frame, below the floor | **no** |
| `TESTED_ABSENT` | in the frame, tested, not reported | **yes** |
| `OBSERVED` | reported with a statistic | — |

A boolean cannot express two things the code already needs. **`TESTED_ABSENT` is the productive half**
— the discriminator argues *against* a rival precisely because its markers were measured and peaked
elsewhere. Collapsing to a boolean turns the panel rule from a discriminator into paralysis. And
**composite subjects need a composition rule**: state is the *meet* over required roles, with the
blocking constituents named, demoted by coverage (K of N against a named floor) over member roles.

**One capability frame per entity kind.** `panel_contains` is the gene frame; the annotation label set
is the cell-type frame; the design's replicate counts are the condition frame. *There is one absence
primitive per entity kind* — which is what lets the rule cover cases nobody has built yet:

| Case | Result |
| --- | --- |
| gene on panel, peaks in this unit | `OBSERVED` |
| gene on panel, peaks elsewhere | `TESTED_ABSENT` → argues against the rival |
| gene off panel | `NEVER_MEASURED`, gene named |
| gene set with 2 panel hits against a floor of 3 | `UNDERPOWERED` |
| ligand-receptor pair, one partner off panel | `NEVER_MEASURED`, **partner named** |
| co-localisation pair, one cell type not in the annotation | `NEVER_MEASURED` — the composition analogue |
| differential abundance with n=2 | `UNDERPOWERED` **unit**, not item |
| niche defined by three cell types, one absent | `NEVER_MEASURED`, component named |

Niche analysis has never been written in this codebase and needs no new concept. That is the evidence
the design is right.

**Detectability (§7.5) enters here** as a frame, not a special case: a gene that produced no usable
signal is in the panel frame but fails the detectability frame.

### 9.6 The rival test

The rival test is not "name a competing cell type". It is the biologist asking **what else could produce
this pattern**, which exists for every analysis type. Only the space of alternatives changes.

| Task | The rival is | Discriminating evidence | Rival space from |
| --- | --- | --- | --- |
| Assignment | a competing cell type | the marker that separates them | reference data (siblings) |
| Characterisation | a competing explanation | driving genes, set overlap, cross-unit profile | reference data (set overlap) |
| Comparison | a confound | per-sample consistency, effect within cell type, driving-cell concentration | **the data itself** |

**A module declaring "my rivals come from the data, not from reference data" is a valid and different
answer**, and the contract must allow it.

The common shape: given the current reading, name a specific alternative account, name the evidence that
would separate them, then say honestly what the data cannot settle. The honest failure differs usefully
— assignment: *the deciding marker is off-panel*; characterisation: *too few panel genes to separate
these sets*; comparison: *one replicate drives this, so condition and batch cannot be separated*.

### 9.7 The DE workflow's four checks *(new)*

"Does this result make sense?" is not a rhetorical question — it is four computable ones:

1. **Panel scoping.** A volcano on 280 genes is not a genome-wide volcano. State K of N tested, and
   which canonical genes for the affected process were never measured. The panel-absence rule applied
   to a comparison.
2. **Replicate / sample-driven confound.** Is the effect driven by one replicate or section? With a
   single biological replicate this returns *cannot be settled* — which is the honest output and the
   clearest demonstration of the refusal behaviour.
3. **Composition versus expression.** Did the gene change, or did the cell-type mixture change? The
   classic misreading in spatial and single-cell data, answerable from cluster abundances.
4. **Cross-method coherence.** Fold-change against the spatial coefficient. Genes high on one axis and
   low on the other are exactly the distinction a spatial method exists to make — and this is the
   figure that proves the framework is not method-blind.

### 9.8 Named failure modes

Before asking *what is this*, ask *is this a coherent unit*: spatially incoherent cells, a tissue region
rather than a cell type, transcript bleed from a neighbour, a mixed profile suggesting
over-segmentation. Named, computed, more actionable than a low band.

### 9.9 Spatial adjacency check

A gene supports unit C but also localizes in unit D. If C and D are spatially interdigitated that is a
**segmentation bleed candidate**; if D sits elsewhere it is a **genuine specificity problem**. Same
statistic, different readings, and only spatial data separates them. No scRNA-seq tool can produce this.
Display flag, not a band input, until its firing rate is measured.

### 9.10 Replicate concordance

Under joint clustering, compare a unit's profile across replicates. Divergence is a flag: batch effect,
section quality, or real biology, and the check does not claim which. **Agreement never raises
confidence** — replicates are not independent witnesses. Raises rather than returning a number under
independent clustering.

### 9.11 Reconciliation: same question or different question

Two results only agree or disagree if they answer the same question.

| Relationship | Outcomes |
| --- | --- |
| Same question (two marker methods) | agree · disagree |
| Different question (enrichment vs marker call) | **coherent · tension · not comparable**, never *agree* |

An enrichment result does not corroborate a cell-type call because it is not answering the identity
question. It can show coherence (a proliferation program in a tumour cluster) or tension (an immune
program in a stromal cluster).

**Dependency structure is mandatory.** Enrichment computed from the marker result agreeing with its
parent is not corroboration.

1. Independent: lab validation (IHC, orthogonal assay)
2. Partly independent: methods asking different questions of the same data
3. Dependent: a result derived from another result

**Hard rule: reconciliation can lower confidence or raise a flag. It can never raise confidence.**
Output is four buckets, not a score: checkable, agreed, disagreed unexpectedly, and **not checkable with
the results provided**.

Reconciliation is a **join on the subject key across task kinds**, which is why every workflow must key
its items in one shared space (`IMPLEMENTATION.md` §2).

### 9.12 Downstream propagation

A flagged call carries its flag into everything depending on it. A biologist does not act on "medium
confidence"; they act on "four downstream results depend on a call I flagged".

---

## 10. The claim store

Reference edges, literature spans, lab notes and measurement results are one object type, differing only
in who may change them and how much weight they carry.

### 10.1 The claim

```
subject       a structured subject (gene · cell type · program · pair · tuple · unit)
assertion     what is claimed
authority     measurement | lab | curated | literature
source        citation URI · note id · edge id · method + statistic
conditions    assay · modality · tissue · frames · method · segmentation · fixation
depends_on    entities the claim rests on
scope         sample:<id> | dataset:<id> | study | lab
valid         active | superseded | retracted
created, author, tension, licence, spans, spec_version
```

### 10.2 Authority, never merged

```
measurement  >  lab  >  curated  >  literature
```

Conflicts are grouped and displayed, never resolved. **The authority label must never blur.** If a lab
claim renders as a curated fact, or a curated edge as evidence, the design collapses. Visible in
storage, prompt, UI and report. This inverts the field's default, which resolves by recency.

### 10.3 Evaluability and transfer

See §7.5. Two functions, not one. Both are pure, IO-free and clock-free. A blocked claim is still
surfaced with its reason — **silent dropping is the failure mode, not a safe default.** Both run at
retrieval time, so switching dataset mid-session stays correct.

### 10.4 Retrieval

Deterministic, no LLM call before the answer model receives context.

```
seeds from context -> traverse (workflow's mask, depth <= 2)
  -> collect claims -> evaluability -> transfer check (where applicable)
  -> hybrid score (BM25 + dense, weighted RRF) -> relevance floor
  -> group by authority -> pack under budget
```

Hybrid, not dense-only: gene symbols break embedding search, so BM25 with alias normalisation is the
anchor. Below the relevance floor the answer is *no support found*, stated concretely. Scope filters
before retrieval, never after. A per-turn log records the traversal path, verdicts, and what was
retrieved and ignored.

### 10.5 Write path and lifecycle

Two writers only: the deterministic engine writes measurement claims; the biologist writes lab claims.
Reference and literature claims are imported. The agent may **propose**; only the biologist commits.

No LLM-assisted dedupe, no background consolidation, no auto-editing. Those mechanisms manage noisy
conversational streams and are what destroyed the audit trails of consumer memory systems. None of
Panoscope's claim classes are noisy.

Supersession only by the owning authority. Contradicted lab claims are **flagged for review**, never
auto-invalidated. No decay, no scoring, no merging.

### 10.6 Storage

| Class | Store | Tracked? |
| --- | --- | --- |
| Lab | git, typed JSON, one file per claim | **yes** — this is what the lab owns |
| Reference view + literature | study SQLite, versioned as a set | **no** — binary, rebuildable; gitignored with a documented rebuild path |
| Measurement | dataset tree, regenerated by the pipeline; claims hold references | yes (the tree) |

v4 placed binary SQLite inside a directory labs were told to version-control. Say which parts are
tracked, and give the rebuild command for the parts that are not.

Two append-only logs: usage, and disagreement events. The second is where you find out whether lab
knowledge and the literature are diverging.

---

## 11. Generation

**Grounding gate.** Every number, marker, PMID, note id and edge id traces to source or the answer is
discarded for the deterministic fallback. Fails closed.

A cited claim must have been **in the packed context for that turn**, not merely exist. That is the
difference between citation existence and citation grounding.

Span-level entailment ships as a **measured flag, not a hard gate**, until its false-rejection rate is
known.

**Answer policy travels separately from context** (strict / balanced / permissive), so the same
retrieved context can be replayed under a different policy. That replay is the ablation.

### 11.1 Two phases

| | Corpus build | Chat |
| --- | --- | --- |
| Frequency | once per study | every turn |
| Data touched | public reference + literature | lab claims, results, unpublished context |
| Job | read papers, write span-bound cards | select, assemble, phrase, refuse |
| Model | strongest available | local where privacy requires |

Once the corpus is frozen the chat model reads rather than derives, and the deterministic core owns
every number, so the chat model's job is smaller than it looks.

**Fails safe.** A weaker generator does not produce ungrounded output; the gate fails closed. It
produces more fallbacks and flatter prose. A usability cost, not a correctness cost, and the measurable
consequence is fallback rate.

**Caution.** Small models are weakest at refusing when the data cannot settle a question — which is
exactly this tool's core behaviour. The rival test, holistic review and reconciliation are the turns
that escalate.

**Never let the model decide `verify`.** Offering a model an "insufficient information" option has been
measured to make frontier models perform *worse than random* on analytical questions. `verify` comes
from the deterministic rubric, always.

Generator model, version, access date and mode go in the report header.

### 11.2 Model access

The access architecture is specified in `docs/llm-access.md`. In summary: one configurable client plus
one OpenAI-compatible adapter covers an API key, an institutional endpoint (Azure / Vertex / Bedrock),
and a local server (Ollama / vLLM) — two code paths, not twelve. Prompt caching is mandatory, not an
optimisation: it is worth roughly four times more than any provider switch on this workload.

**And Tier A runs with no model at all.** The deterministic core plus the fallback produce a complete,
grounded, citable answer for every unit with no network. *Panoscope runs with no LLM; the literature
layer degrades, the numbers do not.* Nobody else in this field makes that claim, and it is true.

---

## 12. Licensing

Ship the code, not the data. Fetch at setup, cache locally, record **source, version and licence per
claim** — which falls out of the architecture for free, since claims already carry provenance. A
`--official-sources-only` flag filters on the source column.

Restrictively-licensed sources are fetched, never bundled. CI greps `data/` and fails on known
marker-database filenames.

---

## 13. The interface *(rewritten)*

**The review surface is part of the interpretation contract, not a rendering of it.**

The governing principle: the visualisation is **uniquely designed per analysis unit, interactive, and
exists so the biologist can audit the AI's interpretation.** You cannot review a cell-type call with the
same figure you would use to review an enrichment result — so the workflow that owns the interpretive
question also declares the surface on which its claims are reviewed.

This is why `UISpec` moved out of the UI layer. A hand-built screen per method does not generalise and
cannot be tested; a declared surface can be both.

The full design — layouts, headers, evidence tables, the measurement-status vocabulary made visual, the
`UISpec` dataclasses, multi-sample, and the argue-then-commit flow — is in **`docs/interface-design.md`**.
What follows are the architectural commitments.

### 13.1 Fixed shell, workflow-supplied panels

Fixed: the unit rail, the header, the chat pane, the claims drawer, report export. Supplied by the
workflow: evidence table columns, the primary plot, detail plots, what a pin selects and which views
respond, and the cross-unit overview.

An unregistered analysis type **degrades gracefully** to a generic surface rather than crashing. That
is a requirement, not a nicety: it is what makes the registry safe to extend.

### 13.2 Every plot carries its conditionality

The requirement most easily missed. A volcano plot on a targeted panel looks exactly like one on whole
transcriptome, and a reader infers genome-wide coverage unless the figure says *280 of ~20,000 genes
tested, panel-scoped*. An enrichment bar needs its K of N. A density hexbin needs its bin size in µm and
that it is area-normalised. A UMAP needs which embedding produced it.

**A misleading figure travels further than a misleading sentence, because it goes in a talk.**

Enforced at construction: a panel with no conditionality record refuses to draw.

### 13.3 Plots render, they do not derive

No plot computes a statistic that is not already in the normalised evidence. The visual analogue of the
grounding gate.

### 13.4 Measurement status is visible, not implied

The four states of §9.5 must be distinguishable in the evidence table. A never-measured item is not a
blank row and not a zero — it is rendered as structurally absent, with the reason, and it carries no
number. This is the project's headline claim, and it currently has no pixel that demonstrates it.

### 13.5 Click-back

Every sentence resolves to the claim it came from, with that claim's authority and conditions visible.
**A sentence with nothing to click should not have been rendered.**

### 13.6 One cross-method plot

Where a unit has both a comparison and a spatial marker result, plot fold-change against the spatial
coefficient. Genes high on one axis and low on the other are exactly the distinction a spatial method
exists to make. Belongs to reconciliation, not to either workflow.

### 13.7 Multi-sample: one primary plus a context strip

Not equal-weight small multiples. The constraint is rhetorical, not perceptual: *divergence is a flag,
agreement is not corroboration* is an **asymmetric** claim, and an equal-weight grid is the best design
ever invented for making a viewer conclude "they all match, so this is solid" — exactly the inference
the architecture forbids. The layout must be as asymmetric as the claim.

### 13.8 Argue, then commit

The AI posts an interpretation; the biologist explores the figures, argues over several turns, and then
commits a decision. **Figures are for exploration and checking; the decision is explicit and separate.**
The commit records the call *with the disagreement visible* — never a bare acknowledgement, never a
silent overrule — and the argument is preserved in a form a report reader can follow without reading a
chat transcript.

---

## 14. Data flow

```
STUDY    intake -> conditions (with provenance) + background + papers of interest
         -> reference view built from the shared cache (per tissue)
         -> literature corpus fetched per triple, versioned

PREP     vendor output (Xenium | MERSCOPE | CosMx)
         -> platform adapter -> validate: table, measured set, unit column,
            coordinate system, clustering mode declared

RUN      per dataset: results -> unit detection -> methods -> normalised evidence
         -> measurement status -> task-specific interpretation
         -> evaluability + transfer -> rival tests
         -> adjacency + concordance -> reconciliation -> propagation
         -> interpretation record written per unit (§17)
         (deterministic, offline, reproducible)

REVIEW   chat + workflow-declared review surface + spatial views
         -> traversal retrieval -> grouped by authority
         -> biologist confirms, overrides, or commits a lab claim

REPORT   assembled from durable artifacts, no live network
         header: study, dataset, platform, frames + hashes, reference releases,
                 corpus version + staleness, generator model + version + access date,
                 spec version
```

---

## 15. Explicitly refused

| Refused | Why |
| --- | --- |
| Running the analysis | Makes confidence self-referential |
| LLM-extracted reference edges | Fabrication surface; edges are imported |
| Reference edges as evidence | Marks ≠ localizes |
| "Agree" between results answering different questions | Category error |
| LLM confidence blended into a score | The model must not move a number |
| Letting the model choose `verify` | Measured to make models worse than random |
| Ensemble voting across methods | Obscures which evidence did the work |
| Weighting evidence by citation count | Popularity is not validity |
| Background consolidation or memory synthesis | Destroys the audit trail |
| LLM rewriting or deduping claims | The agent does not author the lab's knowledge |
| Recency-based conflict resolution | Wrong authority order for this domain |
| Treating replicate agreement as corroboration | Not independent witnesses |
| Pooling clusters across independently-clustered samples | Same label, different objects |
| Carrying a call across datasets by label match | Panels differ; the checks decide |
| A unit-level band on characterisation or comparison | No label space, so the band is meaningless |
| A plot that computes its own statistic | Plots render, they do not derive |
| A plot with no conditionality | A misleading figure travels further than a sentence |
| Bundling restrictively-licensed data | Fetch, don't ship |
| Guessing a result's unit or a dataset's clustering mode | Fail loudly instead |
| A model narrating the interpretation record after the fact | It would drift from what the code did |
| **Reporting "% conditional" as a result** | It is a constant (§7.4) |
| **A novelty verdict** — "is this finding new, or already published?" | A claim about the whole literature that no retrieval layer can honestly support. Absence of evidence in a corpus is not evidence of novelty |
| **Spatially variable gene (SVG) interpretation** | Out of scope by decision. The interpretive question is not settled enough to declare a workflow contract for it |
| **Consumer-subscription authentication** | Prohibited by every major provider's terms (`docs/llm-access.md`). Foreclosed by contract, not by cost |

---

## 16. Evaluation

### 16.1 Seeded-error detection

Programmatic injections across pre-registered classes — wrong number, fabricated gene, real gene wrong
unit, fabricated PMID, real PMID that does not support the claim, off-panel absence used as evidence, a
curated edge raising a band, replicate agreement raising confidence, a lab claim applied out of scope, a
claim returned outside its scope. Ground truth is free by construction.

**Report the false-positive rate on clean data alongside recall.** Recall alone is meaningless for a
gate that fails closed. The baseline that isolates the new work is the existing grounding checker alone.

### 16.2 The panel ablation — **run 2026-09-22, and it came back negative**

This experiment was placed before the engineering precisely so a negative answer would
arrive cheaply. It did. The result is recorded here rather than quietly dropped.

**Design.** Three model tiers, 9 clusters, 3 arms, 5 replicates, 405/405 calls parsed.
Each gene a model raised had to be classified `measured_not_enriched` / `not_measured` /
**`cannot_tell`** — an explicit abstention option, so the result cannot be dismissed as
"you asked it to guess".

| Arm | Calls containing a false or unwarranted claim about the assay |
| --- | --- |
| bare — top markers only, no panel | **94.8%** [89.7, 97.5] |
| panel — the full 280-gene list in the prompt | **1.5%** [0.4, 5.2] |
| panel + an explicit statement of the rule | **0.0%** [0.0, 2.8] |

Per individual claim the bare arm asserts measurement status it cannot know ~60% of the
time, consistently across tiers (Haiku 56.5%, Sonnet 59.1%, Opus 61.1%). Given the panel:
2 errors in 1,502 claims, both the *inverse* direction, zero fabricated absences.

**What this does not support.** Supplying the panel essentially solves this failure mode.
**The deterministic layer is therefore not justified by out-reasoning the model**, and any
framing that claims otherwise is not supported by our own data.

**What it does support**, and this is the narrower claim to make:

> The contribution is not reasoning about the measurement conditions better than a model.
> It is **guaranteeing they are present and correct**. That is a systems property. A prompt
> has it when someone remembers to include the panel; the architecture has it structurally,
> because the measured set is read from the panel file and can never be asserted by a model.

**The bare-arm figure stands on its own.** 94.8% is a measured fabrication rate about
experimental conditions, in a domain where §16.1's survey found fabrication is essentially
never measured. Its weight depends on one open question: *do published tools put the gene
universe in the prompt?* If they do not, the bare arm is their operating condition and the
number describes the field rather than the models. That check is cheap and should be done
before the number is used in a write-up.

**Caveats to carry.** Three model tiers but one family. Nine clusters — wide enough for a
60-point effect, not a subtle one. Models were not pinned to dated snapshots; pin before
any published run. Full method, raw results and the v1 design whose metric failed are in
`evals/panel_absence/`.

**The panel-masking generalisation still stands as the way to scale this**: take a
whole-transcriptome dataset with known cell types, mask genes to simulate a realistic
targeted panel, and every masked canonical marker becomes a ground-truth item — unlimited
items, no annotation cost. It is worth building for the bare-arm question, not for the
gate question, which is now answered.

### 16.3 Generator ablation and Tier-A invariance

Gate constant, model swapped, across three or four families including one open-weight. Report gate
rejection rate, citation support rate, refusal accuracy and fallback rate.

**The result that makes the architecture claim measurable:** Tier A outputs are byte-identical across
model families while ungrounded output varies. This answers a documented, unsolved problem — a large
fraction of published biomedical LLM work depends on models that retire within a couple of years — and
no competitor has published it.

### 16.4 Refusal reported next to accuracy

Every accuracy figure becomes a triple: answered %, accuracy given answered, and accuracy with refusal
counted as wrong. Plus a risk–coverage curve. A gate that fails closed can look safe while refusing.

### 16.5 Sycophancy

Lab claim wrong, data disagrees: does the system surface tension or capitulate? **Narrow the claim** —
sycophancy is benchmarked elsewhere. What is unoccupied is sycophancy *against a ground-truth
measurement the system independently holds*, where correct behaviour is **tension-surfacing, not
correction**. Score over-correction as a failure mode too, or a stubborn system games the benchmark.

### 16.6 Expert concordance and failure taxonomy

Claim-level rating by multiple experts with a pre-registered rubric and reported inter-rater agreement.
A failure taxonomy for diagnosis rather than one score: coverage, grounding, revision, scope, temporal,
retrieval, synthesis, measurement — with a column for "can this architecture catch it in principle?".
The *no* rows are the honest part.

---

## 17. The interpretation record

A per-unit artifact written **by the deterministic core at run time**, recording how the result file was
read and why the interpretation came out as it did. Biologist-facing, not a debugging tool.

### 17.1 Two parts

**How the file was read.** Every interpretation rests on parsing decisions that are invisible today and
are where silent errors live: which column was taken as the statistic and which as the unit; what the
unit was detected as and on what evidence; rows dropped and why; entities in the result but not in the
capability frame; entities not in reference data; the derived measured set and its size; clustering mode
as declared.

If Panoscope reads the wrong fold-change column because a different software version named it
differently, this is the only thing that catches it.

**Why the interpretation came out that way.** The rubric applied and the threshold crossed; drivers with
their numbers; which checks fired and their outcomes; rivals considered and what separated them; what
could not be settled and why; claims used, with authority and verdicts.

### 17.2 Two layers, never blurred

The **factual layer** is recorded, not generated — a byproduct of the core running, available with no
LLM involvement, identical in a literature-off or fully local run.

The **interpretive layer** is generated prose sitting on top, marked as generated, and subject to the
grounding gate.

**The constraint that makes this work: the record is written by the core, not narrated by the model
afterwards.** Core functions return their decisions alongside their results; there is no separate
reconstruction pass. A model summarising what it thinks happened is a weaker artifact that can drift
from what the code did.

### 17.3 What it buys

**A biologist can check the tool understood their file.** The first thing anyone should verify, and
nothing currently supports it.

**A reviewer asking why unit 4 got that call** is answered by pointing at an artifact rather than a
rerun.

**Reproducibility with teeth.** Two runs on the same input produce byte-identical factual layers, so a
diff localises any change to the input file, a reference release, or the code. This is also the
mechanism behind §16.3.

**Invariant:** the factual layer is deterministic. Same inputs, same bytes.

---

## 18. Operational requirements

**Failure and recovery.** Corpus build is resumable and checkpointed; literature rate limits back off
rather than abort; a corrupt store fails at validate with a named cause; a claim referencing a
superseded corpus version resolves against the recorded version or reports that it cannot; claim files
are one-per-file and never rewritten, so concurrent edits are a git merge.

**Artifact versioning.** Every artifact carries `spec_version` **from day one** — the field is cheap and
cannot be added retroactively. Migration *code* waits until a real version break has a real study behind
it; writing tested migrations for schemas nobody holds is speculative work.

**Observability.** `panoscope explain <unit>` renders the interpretation record plus live state the
record does not carry. `panoscope diff <run_a> <run_b>` compares factual layers and localises any
change. `panoscope doctor` prints the resolved data root, studies found, artifact versions, installed
extras and credential status. Structured logs with a correlation id per turn.

**Performance budgets**, declared and tested rather than discovered:

| Operation | Budget |
| --- | --- |
| Reference view build (one tissue) | < 2 min, cached thereafter |
| Corpus build (~20 cell types) | < 30 min, resumable |
| Dataset load (5k panel, 500k cells, no images) | < 10 s |
| Retrieval per turn | < 300 ms |
| Full deterministic run per dataset | < 5 min |

**Cost instrumentation.** Emit a per-intake cost receipt from the provider's usage figures, so the
paper's cost claim is measured rather than modelled.

**Contract tests** run against every registered workflow, method, adapter and source. **Golden-output
test**: full pipeline on the demo, snapshot, diff on every change. **Property tests** for stated
universals, such as reconciliation never raising confidence.

---

## 19. Build order

See `IMPLEMENTATION.md` §7 for the phased sequence with acceptance criteria. In outline:

| # | Work | Note |
| --- | --- | --- |
| 0 | Safety net: golden-output test, cache isolation, import surface | before touching any source |
| 0.5 | The panel-list ablation (§16.2) | sizes the paper; run before committing to its shape |
| 1 | Package rename, packaging, editable install | two commits, never one |
| 2 | `types/` frozen — subject, measured, traced, outputs | blocks everything |
| 3 | Config de-singleton; `Study` object | acceptance: the frozen-path workaround can be deleted |
| 4 | Hardcode purge, then the two-studies-in-one-process gate | the cheap half of the decision point |
| 5 | Platform adapter + tidy reader; method modules with task kinds | |
| 6 | **DE workflow + the method-agnostic demonstration** | decouples the paper from an unpublished dependency |
| 7 | Claims: schema, evaluability, transfer check, store | |
| 8 | Reference sources, view, traversal | |
| 9 | Literature corpus per triple + span binding | |
| 10 | Adjacency, concordance, reconciliation, propagation | |
| 11 | Review surfaces + record viewer | the interface contribution |
| 12 | Operational hardening | |
| 13 | Evaluation | |

---

## 20. Open decisions

1. **Report granularity** — per dataset, or study-level aggregation. Proposed default: per-dataset,
   aggregation deferred.
2. **Reference cache location on shared infrastructure** — a per-user home cache may be wrong for an
   institute. Proposed default: `PANOSCOPE_REFERENCE_DIR` overriding a `~/.panoscope` default.
3. **Local model family** — benchmark several on the eval turn set and choose empirically; the
   comparison is itself a figure. Do not choose by leaderboard rank: the most capable open model is not
   reliably the most faithful.
4. **Whether to restore the second sample** of the demo dataset for replicate concordance. Both samples
   are present in the source objects.
5. **The demotion floor** (§7.5) — demonstrate stability, or adopt a documented rule.

---

## 21. The claim

An interpretation layer in which every claim carries the measurement conditions under which it was
formed, applicability is enforced deterministically at retrieval, and the review surface is part of the
interpretation contract rather than a rendering of it. Task-aware across assignment, characterisation
and comparison, with a claim store that treats method output, reference data, literature and lab
knowledge as one object type ordered by authority, and a swappable generator that makes the guarantees
measurable.

Not another autonomous spatial agent. A human-in-the-loop interpretation layer that is independent of
the method it audits, that says what it could not test, and that hands the decision back.
