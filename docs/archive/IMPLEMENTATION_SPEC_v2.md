# Panoscope implementation spec

Developer companion to `PANOSCOPE_ARCHITECTURE_v4.md`. Version 2.

**Changes from v1.** `claims/ontology.py` and `claims/literature.py` split into two
packages, `reference/` and `corpus/`, because they resolve and retrieve respectively and
share almost nothing. Method modules gain a **task kind** and **unit detection**. Three
output types replace `ClusterVerdict`. The rival test becomes a contract method with a
declared rival space. Reconciliation splits by question identity. Operational requirements
get a section (§14).

**v2.1.** The **interpretation record** (§3.4, §10.1) is added: core functions return their
decisions alongside their results, and the record is assembled from those returns rather
than reconstructed. This changes the return signature of every core function, which is why
it lands in the type contracts at item 2 and not in hardening.

Python 3.11+. Every contract is a `Protocol` or a frozen dataclass so it can be tested
against a stub before the implementation exists.

Anything called an **invariant** has a test that fails loudly. Those tests are the
deliverable, not the feature.

---

## 1. Target repo layout

```
panoscope/
  pyproject.toml  CITATION.cff  README.md  CLAUDE.md  ARCHITECTURE.md

  panoscope/
    config.py                    # thresholds, model ids, cache paths; no dataset constants
    cli.py                       # panoscope demo | run | explain | reference | corpus

    types/                       # frozen contracts, no logic
      claim.py                   #   Claim, Conditions, Authority, TransferResult
      evidence.py                #   GeneEvidence, NormalisedEvidence
      outputs.py                 #   Assignment, Characterisation, Comparison
      record.py                  #   ReadingRecord, Decision, InterpretationRecord
      contracts.py               #   PlatformAdapter, MethodModule protocols

    platforms/                   # one adapter per technology
      base.py  xenium.py  cosmx.py  merscope.py  __init__.py (registry)

    methods/                     # one module per method
      jazzpanda.py  enrichment.py  de.py  __init__.py (registry)

    reference/                   # RESOLVES. structured, shared, versioned
      sources.py                 #   HGNC | CL | Uberon | MSigDB | marker source
      cache.py                   #   ~/.panoscope/reference/<source>/<release>/
      graph.py                   #   node/edge tables, traversal, masks
      view.py                    #   per-study reference view (the traversed subgraph)

    corpus/                      # RETRIEVES. unstructured, per-study, ranked
      fetch.py                   #   per-(gene, cell_type, tissue) triple
      spans.py                   #   span extraction + binding
      retrieve.py                #   query shape, hybrid, RRF, relevance floor
      version.py                 #   corpus version + staleness

    claims/
      store.py  schema.py  transfer.py

    core/                        # deterministic, zero LLM
      record.py                  #   assembles records from Decision returns
      interpret.py               #   task-kind dispatch -> output type
      verdict.py                 #   band computation (assignment only)
      rival.py                   #   rival test dispatch
      reconcile.py  adjacency.py  concordance.py
      failure_modes.py  propagate.py

    agent/
      loop.py  tools.py  mcp_client.py  fallback.py
      grounding_check.py  policy.py

    data/
      spatial.py  legacy.py  study.py
    ui/  pipeline/
  skills/  tests/  evals/  scripts/  studies/
```

The package rename (`agent/`, `ui/`, `pipeline/` move under `panoscope/`) is mechanical and
should be its own commit before anything else, or every later diff is polluted by paths.

---

## 2. Packaging

```toml
[project]
name = "panoscope"
requires-python = ">=3.11"
dependencies = ["anthropic","pandas","pyarrow","plotly","streamlit",
                "python-dotenv","mcp","pyyaml","numpy"]

[project.optional-dependencies]
spatial   = ["spatialdata>=0.2","spatialdata-io>=0.1","geopandas","shapely"]
reference = ["rdflib","igraph","requests"]
local     = ["litellm"]
dev       = ["pytest","pytest-cov","ruff","mypy","hypothesis"]

[project.scripts]
panoscope = "panoscope.cli:main"
```

**Invariant:** `pytest -m "not live and not spatial"` passes with no extras, no network, no
API key. `pip install panoscope && panoscope demo` works on a fresh clone.

---

## 3. Core types

### 3.1 Claim and conditions

```python
class Authority(IntEnum):
    LITERATURE = 1; CURATED = 2; LAB = 3; MEASUREMENT = 4

@dataclass(frozen=True)
class Conditions:
    assay: str | None = None          # "Xenium" | "flow" | "scRNA-seq"
    modality: str | None = None       # "in-situ" | "dissociated" | "protein"
    tissue: str | None = None         # Uberon id
    panel_hash: str | None = None
    panel_genes: frozenset[str] = frozenset()
    method: str | None = None         # "spatial_glm" | "wilcoxon" | "cgst"
    segmentation: str | None = None
    fixation: str | None = None       # "FFPE" | "fresh-frozen"

@dataclass(frozen=True)
class Claim:
    id: str; subject: str; assertion: str
    authority: Authority; source: str
    conditions: Conditions
    depends_on: frozenset[str] = frozenset()
    scope: str = "study"              # sample:<id> | dataset:<id> | study | lab
    valid: str = "active"; superseded_by: str | None = None
    created: str = ""; author: str | None = None
    tension: tuple[str, ...] = (); licence: str | None = None
    spans: tuple[str, ...] = ()
    spec_version: str = "0.1"
```

**Invariant:** a `LAB` claim with no `conditions.panel_hash` fails validation. Without
formation conditions it cannot be transfer-checked and must not be storable.

### 3.2 Normalised evidence

```python
@dataclass(frozen=True)
class GeneEvidence:
    gene: str
    statistic: float
    statistic_name: str               # "glm_coef" | "avg_log2FC" | "z"
    corroborating: dict[str, float]
    measured: bool                    # from the panel, not from a missing row
    also_localises: tuple[str, ...] = ()

@dataclass(frozen=True)
class NormalisedEvidence:
    unit: str                         # cluster id | cell type | contrast | region
    unit_kind: str                    # "cluster" | "cell_type" | "contrast" | "region"
    module: str
    items: tuple[GeneEvidence, ...]   # genes or programs, depending on the module
    measured_set: frozenset[str]
    conditions: Conditions            # inherited from the platform adapter
```

**Invariant:** no module in `core/` or `ui/` imports a method-specific column name.

### 3.3 Three output types

```python
class TaskKind(StrEnum):
    ASSIGNMENT = "assignment"          # markers -> cell type
    CHARACTERISATION = "characterisation"   # enrichment -> what's running
    COMPARISON = "comparison"          # DE -> what changed

@dataclass(frozen=True)
class Assignment:                      # one label from a label space
    unit: str; label: str; label_id: str | None    # CL id when resolvable
    band: str; verify: bool
    drivers: tuple[GeneEvidence, ...]
    offpanel: tuple[str, ...]
    rationale: str

@dataclass(frozen=True)
class Item:                            # a program or a gene, with per-item support
    name: str; statistic: float; statistic_name: str
    tier: str                          # "supported" | "suggestive" | "untestable"
    coverage: tuple[int, int] | None   # K of N on panel
    leading: tuple[str, ...]
    verify: bool
    direction: str | None = None       # comparison only: "up" | "down"

@dataclass(frozen=True)
class Characterisation:
    unit: str; items: tuple[Item, ...]
    untestable_count: int
    summary: str                       # "4 enriched, 3 untestable, top EMT z 4.2 (18/200)"

@dataclass(frozen=True)
class Comparison:
    unit: str                          # the contrast
    items: tuple[Item, ...]
    untestable_count: int
    summary: str
```

**Invariant (`test_no_band_on_non_assignment`):** `Characterisation` and `Comparison` have
no unit-level confidence field. There is no label space, so a unit band is meaningless.
Confidence is per item.

CSV export schema differs per task. `cell_type` and `confidence` columns do not appear on a
DE export.

### 3.4 Decisions and the interpretation record

Every core function returns its decisions alongside its result. The record is **assembled
from those returns**, never reconstructed by a later pass and never narrated by a model.

```python
@dataclass(frozen=True)
class Decision:
    step: str                      # "unit_detection" | "band" | "panel_absence" | "rival" ...
    outcome: str                   # short, machine-comparable
    because: tuple[str, ...]       # the specific values that drove it
    refs: tuple[str, ...] = ()     # claim ids, gene symbols, edge ids

@dataclass(frozen=True)
class ReadingRecord:               # how the file was read
    path: str; sha256: str
    statistic_column: str; unit_column: str
    unit_kind: str; unit_detection_because: tuple[str, ...]
    rows_in: int; rows_dropped: int; drop_reasons: dict[str, int]
    genes_off_panel: tuple[str, ...]
    genes_not_in_reference: tuple[str, ...]
    measured_set_size: int
    clustering_mode: str

@dataclass(frozen=True)
class InterpretationRecord:
    unit: str; task_kind: TaskKind
    reading: ReadingRecord
    decisions: tuple[Decision, ...]
    claims_used: tuple[tuple[str, str], ...]     # (claim_id, transfer_verdict)
    undecidable: tuple[str, ...]
    spec_version: str
    prose: str | None = None                      # generated layer, marked as such
```

The signature change this implies:

```python
# before
def confidence(ev, canonical) -> Band
# after
def confidence(ev, canonical) -> tuple[Band, tuple[Decision, ...]]
```

Applies to `confidence`, `detect_unit`, `interpret`, `rival_test`, `transfer_check`,
`reconcile`, `classify_shared_marker`, `replicate_concordance`. Mechanical, but it must
happen at item 2 or every one of them gets rewritten later.

**Invariants:**
- `test_record_deterministic` — same inputs produce byte-identical `ReadingRecord` and
  `decisions`. Serialise and hash.
- `test_record_not_generated` — `core/` never imports the anthropic client. Asserted by
  import inspection, so the record cannot become model-narrated by accident.
- `test_every_band_has_decisions` — no output carries a band with an empty `decisions`
  tuple. A conclusion with no recorded reason is a bug.
- `test_prose_is_gated` — the `prose` field passes the grounding gate like any other
  generated output.

---

## 4. Platform adapter

```python
class PlatformAdapter(Protocol):
    name: str

    def measured_genes(self, sdata) -> frozenset[str]:
        """Xenium: var_names minus negative controls and custom add-ons. Never a constant."""
    def cluster_labels(self, sdata) -> dict[str, str]: ...
    def samples(self, sdata) -> list[str]: ...
    def clustering_mode(self, sdata) -> Literal["joint","independent"]:
        """MUST be explicit. Raises UndeclaredClustering. Never guesses."""
    def sample_of(self, sdata, cell_ids) -> dict[str, str]: ...
    def available_views(self, sdata) -> frozenset[View]:
        """Derived from elements present. No points -> no density. Never fake."""
    def coordinate_system(self, sdata) -> tuple[str, str]:   # (name, unit -> µm)
    def standing_conditions(self, sdata) -> Conditions:
        """Assay caveats attached to every measurement claim from this platform."""
    def validate(self, sdata) -> list[str]:
        """Human-readable problems. Empty means usable."""
```

**Invariant (`test_undeclared_clustering_raises`):** more than one sample and no declared
mode raises at load. Guessing produces a wrong verdict from pooling unrelated clusters that
share a label.

`detect(sdata)` is best-effort from store attrs and **raises on ambiguity**.

---

## 5. Method module

```python
class MethodModule(Protocol):
    name: str
    task_kind: TaskKind
    statistic_name: str
    measures: str
    does_not_measure: str                 # MANDATORY, rendered in the UI
    rival_space: Literal["reference", "data", "none"]

    def detect_unit(self, table) -> UnitSpec:
        """Read unit kind and ids FROM THE RESULT SCHEMA.
        Raises UnrecognisedResult on an unknown shape. Never guesses."""

    def load(self, path, adapter, sdata) -> dict[str, NormalisedEvidence]: ...

    def absence_reads_as(self, gene: str, measured: bool) -> str:
        """'not measured' | 'not detected' | 'not significant'"""

    def interpret(self, ev, reference) -> Assignment | Characterisation | Comparison:
        """Task-specific. Assignment computes a band; the others do not."""

    def rival_test(self, ev, alternative, reference) -> RivalResult: ...

    def traversals(self) -> tuple[tuple[str,str], ...]: ...
    def expected_disagreements(self) -> tuple[Disagreement, ...]: ...
    def panels(self) -> UISpec:           # §11
```

**Invariant (`test_module_rubrics_differ`):** jazzPanda and DE produce different bands on a
constructed case with high fold-change and low spatial localisation. If they agree the
framework is method-blind. Write this test *before* the DE module.

**Invariant (`test_unrecognised_result_raises`):** a table whose unit cannot be determined
raises rather than defaulting to "cluster".

### 5.1 Rival test dispatch

```python
@dataclass(frozen=True)
class RivalResult:
    alternative: str
    supporting: tuple[str, ...]        # evidence for the current reading
    against: tuple[str, ...]           # evidence against the alternative
    undecidable: tuple[str, ...]       # what the data cannot settle, verbatim to the UI
```

| Module | `rival_space` | Alternative is | Discriminating evidence | Honest failure |
| --- | --- | --- | --- | --- |
| jazzpanda | `reference` | competing cell type (CL siblings) | the separating marker | *the deciding marker is off-panel* |
| enrichment | `reference` | competing explanation: overlapping set, atmospheric signal, leading edge = own markers | driving genes, set overlap, cross-unit profile | *too few panel genes to separate these sets* |
| de | `data` | a confound: composition, batch, one replicate | per-sample consistency, effect within cell type, driving-cell concentration | *one replicate drives this; condition and batch cannot be separated* |

`rival_space="data"` is a first-class answer. `core/rival.py` dispatches on it and must not
assume a reference lookup.

**Open (arch §17.2):** confirm the DE checks are computable from a typical DE result table
rather than needing the expression matrix. Test against a real result before implementing.

---

## 6. Reference data

Resolves. No ranking, no relevance, no fetch at question time.

```python
# reference/cache.py
def ensure(source: str, release: str | None = None) -> Path:
    """Fetch-once into ~/.panoscope/reference/<source>/<release>/.
    PANOSCOPE_REFERENCE_DIR overrides for shared HPC installs."""

# reference/graph.py
def resolve_alias(symbol: str) -> str
def cell_types_in(tissue_uberon: str, depth: int = 1) -> list[str]
def canonical_markers(cell_type: str) -> list[tuple[str, Conditions]]
def siblings(cell_type: str) -> list[str]
def set_overlap(program_a: str, program_b: str) -> float
def traverse(seeds, mask, depth) -> set[str]

# reference/view.py
def build_view(study, sources: list[str], max_cell_types: int = 40) -> ReferenceView
```

Sources shipped: HGNC, Cell Ontology, Uberon (via Ubergraph for inferred relations),
MSigDB, one pluggable marker source. GO, assay knowledge, and design knowledge are **not**
source kinds (arch §5.2).

Schema (SQLite, in the shared cache for `node`/`edge`, in the study store for the view):

```sql
CREATE TABLE node (
  id TEXT PRIMARY KEY,            -- "gene:POSTN" | "ct:CL_0000057" | "tissue:UBERON_0000310"
  type TEXT NOT NULL, label TEXT NOT NULL,
  source TEXT NOT NULL, release TEXT NOT NULL);

CREATE TABLE edge (
  id TEXT PRIMARY KEY,
  src TEXT NOT NULL, dst TEXT NOT NULL,
  rel TEXT NOT NULL,              -- marks | is_a | part_of | member_of | alias_of
  source TEXT NOT NULL, release TEXT NOT NULL, licence TEXT NOT NULL,
  conditions_json TEXT);          -- Conditions for `marks` edges
CREATE INDEX idx_edge_src ON edge(src, rel);
CREATE INDEX idx_edge_dst ON edge(dst, rel);
```

**Invariants:**
- `test_no_bundled_restricted_data` — CI greps `data/` and fails on known marker-DB
  filenames.
- `test_missing_node_reads_as_unknown` — a cell type absent from CL returns *not in
  reference data*, never *does not exist*.
- `test_resolve_is_deterministic` — no network at query time; a cold cache raises with a
  fetch instruction rather than silently degrading.

---

## 7. Literature corpus

Retrieves. Per study, versioned, ranked.

```python
# corpus/fetch.py
def build(view: ReferenceView, tissue: str, *, background: str | None) -> CorpusVersion:
    """Fetch per (gene, cell_type, tissue) triple. Resumable and checkpointed."""

# corpus/retrieve.py
@dataclass(frozen=True)
class Query:
    shape: Literal["gene","cell_type","program","method","tissue","claim","open"]
    seeds: frozenset[str]
    unit: str | None

def plan(text, context) -> Query          # hand-coded rules, NO LLM call
def retrieve(q, module, store, context, budget) -> RetrievalResult
```

| Shape | graph | keyword | dense |
| --- | --- | --- | --- |
| gene | 1.5 | 2.0 | 1.5 |
| cell_type | 3.0 | 1.5 | 1.5 |
| program | 2.0 | 1.5 | 2.0 |
| method | 0.5 | 1.5 | 2.0 |
| open | 1.0 | 1.5 | 2.0 |

Weighted RRF, k=20. BM25 over `span_fts` with HGNC-normalised query terms.

**Invariants:**
- `test_relevance_floor` — no supporting spans returns `no_support`, and the rendered
  answer says so, naming the nodes the traversal reached.
- `test_symbol_retrieval` — a deprecated alias retrieves the current symbol's claims.
- `test_authority_not_merged` — packed context keeps authority groups separate and labelled.
- `test_corpus_staleness_surfaced` — a corpus past the threshold flags in the report header.

---

## 8. Claim store and transfer check

```python
class ClaimStore:
    def add(self, claim: Claim) -> None:
        """LAB -> git file; CURATED/LITERATURE -> study sqlite;
        MEASUREMENT -> reference into the dataset tree."""
    def for_subjects(self, subjects, context) -> list[tuple[Claim, TransferResult]]:
        """Only read path. Scope gate and transfer applied BEFORE return."""
    def supersede(self, claim_id, by, actor: Authority) -> None:
        """Raises unless actor owns that claim's authority class."""
```

```python
# claims/transfer.py  — pure, no IO, no clock
def transfer_check(claim: Claim, context: Conditions) -> TransferResult
```

0. Scope gate: `sample:X` not returned outside X, `dataset:Y` not outside Y. Filter, not a
   downgrade, runs first.
1. `claim.depends_on - context.panel_genes` non-empty → **blocked**, naming the genes.
2. Identical `panel_hash` → **transferable**, return early.
3. Modality mismatch (dissociated or protein used on in-situ) → **conditional**.
4. Assay family mismatch → **conditional**.
5. Tissue mismatch via the Uberon hierarchy (same or ancestor is neutral) → **conditional**.
6. Fixation mismatch → **conditional**.
7. Method family mismatch → **conditional**.

**Invariants:** `test_transfer_pure`; `test_blocked_is_surfaced` (a blocked claim still
appears with its reason — silent dropping is the failure mode); `test_offpanel_never_
penalises` (a `marks` edge whose gene is off-panel is blocked and never contributes to a
band); `test_scope_gate_before_return`.

~150 lines including the Uberon ancestor walk.

Lab claim on disk at `studies/<study>/context/claims/<id>.json`, one file, git-tracked,
never mutated on read. Logs are append-only JSONL beside it: `usage`, `disagreement`,
`retrieval`.

---

## 9. Study, dataset, sample

```python
@dataclass(frozen=True)
class Sample:   id: str; coordinate_system: str; n_cells: int
@dataclass(frozen=True)
class Dataset:
    id: str; zarr_path: Path; platform: str
    panel_hash: str; panel_genes: frozenset[str]
    clustering_mode: Literal["joint","independent"]
    samples: tuple[Sample, ...]
@dataclass(frozen=True)
class Study:
    id: str; tissue_uberon: str; background: str | None
    datasets: tuple[Dataset, ...]
    reference_view: int; corpus_version: int

def resolve(study, dataset_id=None, sample_id=None) -> Conditions:
    """Sample conditions inherit dataset, which inherit study."""
```

```
studies/<study_id>/
  study.json                 # tissue, background, dataset ids, view + corpus versions
  reference_view.sqlite      # the traversed subgraph
  corpus.sqlite              # records, spans, corpus_version
  context/claims/*.json  context/logs/*.jsonl
  datasets/<dataset_id>/
    <dataset>.zarr/  inputs/  chat/<id>.jsonl  manifest.json
    interp/
      verdicts.csv  clusters/<unit>.json
      records/<unit>.json        # interpretation record, §3.4
```

Chat history and review state scope to **dataset**, which is what makes resuming feel like
a session without a session object.

---

## 10. Core

```python
# core/record.py
def assemble(unit, reading: ReadingRecord,
             decisions: list[Decision], claims, undecidable) -> InterpretationRecord
def write(record, dataset_tree: Path) -> Path
    """studies/<s>/datasets/<d>/interp/records/<unit>.json"""

# core/interpret.py
def interpret(ev, module, reference) -> tuple[Output, tuple[Decision, ...]]
    """Dispatch on module.task_kind. Assignment computes a band via core/verdict.py;
    the others compute per-item tiers only."""

# core/reconcile.py
SAME_QUESTION = {("jazzpanda","other_marker_method")}
DEPENDENCY = {("enrichment","jazzpanda"): "derived",
              ("de","jazzpanda"): "partly_independent",
              ("ihc","*"): "independent"}

def reconcile(outputs: dict[str, Any]) -> ReconcileResult:
    """Same question -> agree | disagree.
       Different question -> coherent | tension | not_comparable. NEVER agree."""

# core/adjacency.py
def cluster_adjacency(sdata, adapter, k: int = 6) -> dict[tuple[str,str], float]
def classify_shared_marker(gene, home, other, adj) -> Literal["bleed_candidate",
                                                              "specificity_problem"]

# core/concordance.py
def replicate_concordance(evs_by_sample) -> ConcordanceResult
    """Joint clustering only; raises otherwise."""

# core/propagate.py
def propagate(outputs, artifacts) -> dict[str, list[Flag]]
```

**Invariants:**
- `test_never_raises_confidence` — property test (hypothesis) over random inputs:
  reconciled band ≤ single-method band, always.
- `test_different_question_never_agrees` — enrichment vs a marker call never returns
  `agree`.
- `test_derived_is_not_corroboration` — enrichment agreeing with its parent returns
  `not_comparable`.
- `test_concordance_never_raises` and `test_concordance_requires_joint`.

---

## 11. UI spec from the module

```python
@dataclass(frozen=True)
class UISpec:
    rail_label: str                      # "Clusters" | "Units" | "Contrasts"
    header: Literal["call","counts"]     # assignment vs the other two
    table_columns: tuple[Column, ...]
    primary_plot: PlotSpec
    detail_plots: tuple[PlotSpec, ...]
    overview_plot: PlotSpec | None
    pin_kind: Literal["gene","program","contrast"]
    pin_drives: tuple[View, ...]
```

**Record viewer.** Every unit header links to its interpretation record, rendered in two
labelled sections: *how this file was read* and *why this call*. The first is the one
biologists should check first and currently nothing supports. The report embeds a condensed
form per unit and links to the full record.

```python
@dataclass(frozen=True)
class PlotSpec:
    kind: str                            # "volcano" | "coverage_bar" | "heatmap" | "hexbin"
    fields: tuple[str, ...]              # MUST exist in NormalisedEvidence
    conditionality: str                  # rendered on the figure, mandatory
```

**Invariant (`test_plot_fields_exist`):** every `PlotSpec.fields` entry resolves to a field
already present in `NormalisedEvidence`. Plots render; they do not derive. The visual
analogue of the grounding gate.

**Invariant (`test_conditionality_present`):** every `PlotSpec` has a non-empty
`conditionality` string. A volcano on a targeted panel is indistinguishable from a
whole-transcriptome volcano unless the figure says *280 of ~20,000 genes tested,
panel-scoped*. A misleading figure travels further than a misleading sentence.

Cross-method plot (log2FC against spatial coefficient) belongs to `core/reconcile.py`, not
to either module.

---

## 12. Generation

```python
# agent/policy.py
def route(turn_kind: str, mode: str) -> tuple[str, AnswerPolicy]:
    """turn_kind: opening | phrasing | open_chat | rival_test | holistic | reconciliation.
    Context and instructions are assembled at the call site, never concatenated in the
    store, so the same context can be replayed under a different policy."""
```

Grounding gate extension: a cited claim id must have been **in the packed context for that
turn**, not merely exist.

**Invariant (`test_claim_cited_must_be_retrieved`).**

One supported mode: hybrid. Air-gap is a property (model id is config, corpus is a
versioned artifact), not a maintained code path.

---

## 13. Migration map

| Existing | Action |
| --- | --- |
| `agent/data.py` | split → `data/spatial.py` (new) + `data/legacy.py` (kept) |
| `agent/config.py` | strip dataset constants |
| `agent/verdict.py` | → `core/verdict.py`; rubric → `methods/jazzpanda.py`; wraps in `core/interpret.py` |
| `agent/memory.py` + `agent/user_memory.py` | → `claims/store.py`; note schema → `Claim` |
| `agent/discriminate.py` | → `core/rival.py`; candidates from `reference.siblings` |
| `agent/enrichment.py` | → `methods/enrichment.py`, `task_kind=CHARACTERISATION` |
| `agent/skeptic.py`, `agent/holistic.py` | read `NormalisedEvidence`; holistic gains reconcile output |
| `agent/grounding_check.py` | extend resolver for claim and edge ids; gate `record.prose` |
| `agent/loop.py` | reads the record instead of re-deriving rationale for the opening turn |
| `scripts/prep_data.R` | deprecated → `pipeline/stages/prep.py` |
| `scripts/precompute_density.py` | optional; density can bin lazily from Points |
| `data/cluster_key.json` | delete; derived from `table.obs` |
| `ui/state.py` | study/dataset/sample in session state; chat keyed on dataset |
| `ui/cluster_rail.py` | label and contents from `UISpec.rail_label` |
| `ui/evidence_table.py` | columns from `UISpec.table_columns` |
| `ui/spatial_stage.py` | one sample's coordinate system; sample switcher |
| `ui/report.py` | section shape per task kind; header gains versions (§14) |

Migration is behind a reader interface; both paths coexist. Cut over per view.

---

## 14. Operational

**Failure and recovery.** Corpus build checkpoints per triple and resumes. PubMed rate
limits back off, never abort. A corrupt zarr fails at `validate()` with a named cause. A
claim referencing a superseded corpus version resolves against the recorded version or
reports that it cannot. Claim files are one-per-file and never rewritten, so concurrent
edits are a git merge.

**Artifact versioning.** Every artifact carries `spec_version`. Migrations live in
`panoscope/migrations/`, are tested, and are one-way. A study built under 0.1 must open
under 0.3 or the lab's accumulated knowledge dies at the first upgrade.

**Observability.** The interpretation record (§3.4) is the durable artifact, written at run
time. `panoscope explain <unit>` renders it plus live state the record does not carry.
Structured logs with a correlation id per turn.

`panoscope diff <run_a> <run_b>` compares factual layers across runs and localises any
change to the input file, a reference release, or the code. This is what makes the
reproducibility claim checkable rather than asserted.

**Performance budgets**, asserted in `tests/test_budgets.py` with generous margins:

| Operation | Budget |
| --- | --- |
| Reference view build (one tissue) | < 2 min, cached |
| Corpus build (~20 cell types) | < 30 min, resumable |
| Dataset load (5k panel, 500k cells, no images) | < 10 s |
| Retrieval per turn | < 300 ms |
| Full deterministic run per dataset | < 5 min |

---

## 15. Tests

| Group | Defends |
| --- | --- |
| `test_platform_*` | panel derivation, view availability, undeclared clustering raises |
| `test_module_*` | rubric divergence, unit detection, does-not-measure present |
| `test_transfer_*` | purity, blocked-is-surfaced, off-panel never penalises, scope gate |
| `test_claims_*` | authority not merged, supersession ownership |
| `test_reference_*` | determinism, no bundled data, missing node reads as unknown |
| `test_corpus_*` | relevance floor, alias retrieval, staleness surfaced |
| `test_reconcile_*` | never raises confidence (property), different question never agrees |
| `test_ui_*` | plot fields exist, conditionality present |
| `test_record_*` | deterministic, not generated, every band has decisions, prose gated |
| `test_grounding_*` | cited must be retrieved |
| `test_budgets` | §14 |

Plus **contract tests** parameterised over every registered adapter and module, so adding
one proves conformance, and a **golden-output test** that snapshots the demo report and
diffs on every change.

`evals/` is separate from `tests/` because it needs data and produces numbers:
`seeded_errors.py`, `transfer_expert.py`, `sycophancy.py`, `generator_swap.py`,
`turns.yaml` (30–50 turns with known-correct behaviour including refusals).

CI jobs: `core` (no extras, no network), `spatial`, `reference`, `licence` (greps for
bundled data), `lint` (`ruff` + `mypy` scoped to `types/` and `claims/`, where a type error
is a design error).

---

## 16. Order and size

| # | Work | Size | Note |
| --- | --- | --- | --- |
| 1 | Package rename + packaging + release | 1 day | own commit |
| 2 | `types/` frozen incl. `Decision`/record; core signatures return decisions | 1–2 days | blocks everything; late = full rewrite |
| 3 | Xenium adapter + `data/spatial.py` + prep | 3–4 days | |
| 4 | Module refactor: task kinds, unit detection, jazzPanda + enrichment | 3 days | |
| 5 | Claim schema + transfer check + store | 3 days | |
| 6 | `reference/` sources + view + traversal | 4 days | |
| 7 | **Thin slice on a second dataset, different panel** | 2 days | **decision point** |
| 8 | DE module + rubric divergence + data-derived rivals | 3 days | after arch §17.1–2 |
| 9 | `corpus/` fetch + spans + retrieve | 3 days | |
| 10 | Adjacency, concordance, reconcile, propagate | 3–4 days | |
| 11 | Task-specific UI panels + record viewer | 5 days | |
| 12 | Operational hardening | 1 week | |
| 13 | Evaluation | ongoing | |

Item 7 is a decision point, not a milestone. It answers whether the transfer check fires
usefully or fires constantly. If constantly, the central claim needs rethinking before it
goes in a paper, and everything after item 7 should wait on that answer.
