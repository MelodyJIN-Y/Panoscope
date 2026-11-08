# Panoscope implementation spec

Developer companion to `ARCHITECTURE.md`. Version 3. Supersedes
`docs/archive/IMPLEMENTATION_SPEC_v2.md`.

Python 3.11+. Every contract is a `Protocol` or a frozen dataclass so it can be tested against a stub
before the implementation exists.

Anything called an **invariant** has a test that fails loudly. **Those tests are the deliverable, not
the feature.**

---

## Changes from v2

| | v2 | v3 | Why |
| --- | --- | --- | --- |
| Evidence item | `GeneEvidence`, gene-keyed | `EvidenceItem` with a structured `Subject` | Reconciliation is a join across task kinds; gene-keyed and name-keyed items cannot join |
| Measured | `measured: bool` | four-state `Measured` lattice | A boolean cannot express *tested and absent* (the productive half) or composite subjects |
| Outputs | `Assignment`, `Characterisation`, `Comparison` | `Assignment`, `Finding` | The latter two were structurally identical |
| Decisions | `-> tuple[R, tuple[Decision, ...]]` on 8 signatures | `Traced[T]` with a local-only builder | Signature inspection becomes the invariant; an ambient recorder would be an injection seam |
| Registries | `methods/` list, two reference source kinds | four registries with declared contracts | Extensibility as a testable claim |
| Transfer | one 7-rule function | `evaluability` + `transfer_check`, split by conditions provenance | The 7-rule form is measurably constant on curated claims |
| UI | `UISpec` in the UI layer | `UISpec` declared by the workflow | The review surface is part of the interpretation contract |
| `Conditions` | inlines `panel_genes: frozenset[str]` | `frames: tuple[FrameRef, ...]` | Claims are git-tracked and re-read per retrieval; inlining a panel is ~8 KB per claim |
| Python floor | 3.11 | **3.11** (v3 briefly lowered to 3.10, then reverted) | The 3.10 floor rested on "no 3.11-only syntax is used". `types/` uses `enum.StrEnum`, which is 3.11+, so that justification no longer holds. Reverted rather than shimmed: a `(str, Enum)` backport renders `str(member)` as `Class.MEMBER` instead of the value, which would make record bytes depend on the interpreter — the one thing ARCHITECTURE §17.3 forbids. 3.10 also reaches end-of-life in October 2026 |
| `requires-python` | — | upper-bounded deps + lockfile | Unbounded pins are a reproducibility hazard |

Two v2 bugs made a stated invariant unsatisfiable: `corroborating: dict[str, float]` and
`drop_reasons: dict[str, int]` are `dict` fields on frozen dataclasses — unhashable, insertion-ordered,
and that order flows through `asdict()` into the JSON bytes, so `test_record_deterministic`
("byte-identical") could never pass. Both are now sorted tuples of pairs, guarded by
`test_no_dict_fields_in_contracts`.

---

## 1. Target repo layout

```
panoscope/
  pyproject.toml  CITATION.cff  README.md  CLAUDE.md  ARCHITECTURE.md  IMPLEMENTATION.md

  panoscope/
    config.py                    # thresholds, model ids, cache paths; NO dataset constants
    cli.py                       # demo | run | explain | diff | doctor | studies | fetch-demo

    types/                       # frozen contracts, no logic
      entity.py                  #   EntityKind, Entity
      subject.py                 #   Subject, Shape, Role, Gate, SHAPES, subject()
      measured.py                #   MeasuredState, Coverage, Frame, FrameRef, MeasuredWorld, Measured
      evidence.py                #   EvidenceItem, NormalisedEvidence
      outputs.py                 #   TaskKind, Tier, FindingItem, Assignment, Finding
      trace.py                   #   Decision, Traced, Trace, RULES
      record.py                  #   ReadingRecord, InterpretationRecord
      claim.py                   #   Authority, Conditions, Claim, Evaluability, TransferResult
      uispec.py                  #   UISpec, Column, PlotSpec, Conditionality, RenderProfile
      contracts.py               #   Workflow, MethodModule, PlatformAdapter, ReferenceSource

    workflows/                   # one per interpretive question
      marker.py  enrichment.py  de.py  __init__.py (registry)
    methods/                     # register into a workflow
      jazzpanda.py  seurat_de.py  jazzpanda_cgst.py  limma_voom.py  __init__.py (registry)
    platforms/
      base.py  tidy.py  xenium.py  __init__.py (registry)
    reference/
      sources/  cache.py  graph.py  view.py  __init__.py (registry)
    corpus/
      fetch.py  spans.py  retrieve.py  version.py
    claims/
      store.py  schema.py  evaluability.py  transfer.py
    core/                        # deterministic, zero LLM
      measure.py  interpret.py  verdict.py  rival.py
      reconcile.py  adjacency.py  concordance.py  failure_modes.py  propagate.py
      record.py  export.py
    agent/
      loop.py  tools.py  client.py  fallback.py  grounding_check.py  policy.py  intake.py
    data/
      study.py  layout.py  spatial.py  legacy.py
    ui/  pipeline/  skills/  assets/

  studies/                       # the study store (was data/datasets/)
  tests/  evals/  scripts/  docs/
```

Three things called "data" is one too many: the repo-root store becomes `studies/`, and
`panoscope/data/` holds readers. Rename in the move commit or spend six months disambiguating.

`panoscope/types/` shadowing stdlib `types` is legal under absolute imports, but note it.

Capitalisation: **`Panoscope`**, settled.

---

## 2. Core types

### 2.1 Entity and Subject

```python
class EntityKind(StrEnum):
    GENE = "gene"; GENE_SET = "gene_set"; CELL_TYPE = "cell_type"
    UNIT = "unit"; CONDITION = "condition"; CONTRAST = "contrast"; SAMPLE = "sample"

@dataclass(frozen=True, slots=True, order=True)
class Entity:
    """``id`` is the identifier the capability frame is keyed on (HGNC symbol, annotation
    label, set name). ``label`` is display only and is NEVER load-bearing."""
    kind: EntityKind
    id: str
    label: str = ""
    def __str__(self) -> str: return f"{self.kind.value}:{self.id}"

class Gate(StrEnum):
    REQUIRED = "required"   # participates in the MEET (absent -> subject unmeasurable)
    MEMBER   = "member"     # participates in COVERAGE (K of N)
    CONTEXT  = "context"    # provenance only

@dataclass(frozen=True, slots=True)
class Role:
    name: str; kind: EntityKind; gate: Gate = Gate.REQUIRED; many: bool = False

@dataclass(frozen=True, slots=True)
class Shape:
    name: str
    roles: tuple[Role, ...]
    symmetric: tuple[str, ...] = ()      # interchangeable role names
    display: str = "{}"

@dataclass(frozen=True, slots=True)
class Subject:
    """WHAT an evidence item is about. The single key space every task kind shares."""
    shape: str
    parts: tuple[tuple[str, Entity], ...]   # (role, entity), CANONICAL order

    @property
    def key(self) -> str:
        return self.shape + "|" + "|".join(f"{r}={e}" for r, e in self.parts)
    def one(self, role: str) -> Entity | None: ...
    def all(self, role: str) -> tuple[Entity, ...]: ...
    def entities(self) -> tuple[Entity, ...]: ...
```

The shape registry is **data**, and it is where a new analysis family lands:

```python
SHAPES: dict[str, Shape] = {
    # built
    "gene":      Shape("gene", (Role("g", EntityKind.GENE),), display="{g}"),
    "gene_set":  Shape("gene_set", (Role("set", EntityKind.GENE_SET),
                                    Role("member", EntityKind.GENE, Gate.MEMBER, many=True)),
                       display="{set}"),
    "cell_type": Shape("cell_type", (Role("ct", EntityKind.CELL_TYPE),), display="{ct}"),
    # declared, not built
    "cell_type_pair": Shape("cell_type_pair",
        (Role("a", EntityKind.CELL_TYPE), Role("b", EntityKind.CELL_TYPE)),
        symmetric=("a", "b"), display="{a} <-> {b}"),
    "adjacency": Shape("adjacency",
        (Role("centre", EntityKind.CELL_TYPE), Role("neighbour", EntityKind.CELL_TYPE)),
        display="{centre} -> {neighbour}"),
    "interaction": Shape("interaction",
        (Role("ligand", EntityKind.GENE), Role("receptor", EntityKind.GENE),
         Role("source", EntityKind.CELL_TYPE), Role("target", EntityKind.CELL_TYPE)),
        display="{ligand}->{receptor} ({source}->{target})"),
    "gene_pair": Shape("gene_pair", (Role("a", EntityKind.GENE), Role("b", EntityKind.GENE)),
                       symmetric=("a", "b"), display="{a} - {b}"),
    "module":    Shape("module", (Role("hub", EntityKind.GENE),
                                  Role("member", EntityKind.GENE, Gate.MEMBER, many=True)),
                       display="{hub} module"),
    "niche":     Shape("niche", (Role("niche", EntityKind.UNIT),
                                 Role("component", EntityKind.CELL_TYPE, many=True)),
                       display="{niche}"),
}

def subject(shape: str, **roles) -> Subject:
    """The ONE constructor. Validates against the registry and canonicalises symmetric
    role groups. Constructing a Subject directly is a bug."""
```

Adding niche, ligand-receptor or differential abundance later is **a dict row**: no union member
reopened, no `match` statement touched, no serializer edited.

### 2.2 Measured — the generalised panel-absence rule

```python
class MeasuredState(StrEnum):
    NEVER_MEASURED = "never_measured"   # not in the frame       -> absence NOT evidence
    UNDERPOWERED   = "underpowered"     # in frame, below floor   -> absence NOT evidence
    TESTED_ABSENT  = "tested_absent"    # tested, not reported    -> absence IS evidence
    OBSERVED       = "observed"

_RANK = {MeasuredState.NEVER_MEASURED: 0, MeasuredState.UNDERPOWERED: 1,
         MeasuredState.TESTED_ABSENT: 2, MeasuredState.OBSERVED: 3}

@dataclass(frozen=True, slots=True)
class Coverage:
    """K of N against a named floor. ONE home for every K-of-N in the system:
    panel hits, leading edge, replicates, cells in a niche, permutations."""
    what: str; k: int; n: int; floor: int
    @property
    def ok(self) -> bool: return self.k >= self.floor
    @property
    def fraction(self) -> float: return (self.k / self.n) if self.n else 0.0

@dataclass(frozen=True, slots=True)
class Frame:
    """What the apparatus could observe, for ONE entity kind."""
    kind: EntityKind
    name: str                 # "panel:xenium_hbc_v1" | "annotation:leiden_r0.8" | "detect:nosig"
    members: frozenset[str]
    digest: str               # sha256 over sorted members; DERIVED, asserted

@dataclass(frozen=True, slots=True)
class FrameRef:
    """What a Claim carries. Membership lives once, in the manifest."""
    kind: EntityKind; name: str; digest: str; size: int

@dataclass(frozen=True, slots=True)
class Measured:
    state: MeasuredState
    frame: str
    because: tuple[str, ...] = ()
    blocked_by: tuple[Entity, ...] = ()      # constituents that pulled the meet down
    coverage: tuple[Coverage, ...] = ()
    @property
    def testable(self) -> bool:
        return self.state in (MeasuredState.OBSERVED, MeasuredState.TESTED_ABSENT)
    @property
    def absence_is_evidence(self) -> bool:
        return self.state is MeasuredState.TESTED_ABSENT
```

```python
# core/measure.py
def resolve(subject, world, reported, statistic) -> Traced[Measured]:
    """MEET over required roles, DEMOTED by coverage over member roles.

    per constituent e of a REQUIRED role:
        e not in world.frame(e.kind).members  -> NEVER_MEASURED
        e in frame, not in reported(e.kind)   -> TESTED_ABSENT
        e in frame and reported               -> OBSERVED
    composite: state = min(constituents by _RANK); blocked_by = all at the minimum
    then: any failing Coverage over MEMBER roles demotes to UNDERPOWERED
    finally: statistic is None IFF state is not OBSERVED
             (read off Measured.carries_number -- UNDERPOWERED withholds it too)
    """
```

The last line makes a grounding violation **structurally unrepresentable**: the number is unavailable in
exactly the states where quoting it would be wrong.

One frame per entity kind: `GENE` → the panel; `CELL_TYPE` → the annotation label set; `CONDITION` → the
design levels meeting the replicate floor; `GENE_SET` → the collection. **Detectability is a second
`GENE` frame**, not a special case.

A missing annotation label set is the starting state, not a blocker: build and validate pass, review
opens on annotation, and the `CELL_TYPE` frame is empty or partial until the biologist calls clusters.
The four states apply to it as to any frame.

### 2.3 Evidence

```python
class Direction(StrEnum):
    UP = "up"; DOWN = "down"

@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """INPUT side. No tier, no verify — the interpreter adds those."""
    subject: Subject
    statistic: float | None
    statistic_name: str
    measured: Measured
    corroborating: tuple[tuple[str, float], ...] = ()   # SORTED pairs, never a dict
    direction: Direction | None = None
    flags: tuple[str, ...] = ()          # REGISTERED ids, never prose
    shared_with: tuple[str, ...] = ()    # other units carrying this subject

@dataclass(frozen=True, slots=True)
class NormalisedEvidence:
    unit: str
    unit_kind: str                       # open vocabulary
    unit_subject: Subject
    unit_measured: Measured              # "too few replicates" lives HERE, not on items
    module: str
    items: tuple[EvidenceItem, ...]
    reported: tuple[Frame, ...]
    conditions: Conditions
```

`flags` are registered ids, not English. The current UI string-matches the exact phrase *"localizes
better with another cluster"* to decide whether to draw a badge; `flags=("specificity:localises_elsewhere",)`
plus a copy table in the UI fixes that and makes the test robust to a copy edit.

### 2.4 Outputs

```python
class TaskKind(StrEnum):
    ASSIGNMENT = "assignment"; CHARACTERISATION = "characterisation"; COMPARISON = "comparison"

class Tier(StrEnum):
    SUPPORTED = "supported"; SUGGESTIVE = "suggestive"; UNTESTABLE = "untestable"

@dataclass(frozen=True, slots=True)
class FindingItem:
    """OUTPUT side. Wraps the evidence rather than copying its fields — one home per number."""
    evidence: EvidenceItem
    tier: Tier
    verify: bool
    because: tuple[str, ...] = ()
    @property
    def subject(self) -> Subject: return self.evidence.subject
    @property
    def direction(self): return self.evidence.direction

@dataclass(frozen=True, slots=True)
class Assignment:
    """Has a label space, therefore has a band."""
    unit: str; unit_kind: str
    label: str; label_id: str | None
    band: str; verify: bool
    drivers: tuple[EvidenceItem, ...]
    unmeasured: tuple[EvidenceItem, ...]      # NOT bare names — must say which partner blocked
    rationale: str                            # deterministic prose from core, NOT the gated layer

@dataclass(frozen=True, slots=True)
class Finding:
    """CHARACTERISATION and COMPARISON. No band. No confidence. Ever."""
    unit: str; unit_kind: str
    task_kind: TaskKind                       # never ASSIGNMENT
    items: tuple[FindingItem, ...]
    summary: str
    @property
    def untestable_count(self) -> int:
        return sum(1 for i in self.items if i.tier is Tier.UNTESTABLE)
```

`Finding` declares no field named `band`, `confidence` or `confidence_score`. The v2 invariant becomes a
guard on the type rather than a behavioural test.

### 2.5 Decisions: `Traced[T]`

```python
@dataclass(frozen=True, slots=True)
class Decision:
    step: str                        # "confidence" | "detect_unit" | "measure"
    rule: str                        # REGISTERED id: "panel_absence" | "coverage_floor" | ...
    outcome: str
    because: tuple[str, ...] = ()    # deterministic, source-quoting, clock-free
    refs: tuple[str, ...] = ()       # "gene:ERBB2#glm_coef=21.4378", "claim:abc"

@dataclass(frozen=True, slots=True)
class Traced(Generic[T]):
    value: T
    decisions: tuple[Decision, ...] = ()

class Trace:
    """A LOCAL, mutable builder. Constructed INSIDE a function, returned as frozen data.
    NEVER a parameter: decisions flow OUT, never in. That rule is what keeps core/
    un-injectable and every function pure in its declared arguments."""
    def note(self, *, rule, outcome, because=(), refs=()) -> None: ...
    def absorb(self, t: Traced[X]) -> X:
        """Splice a callee's decisions in call order and unwrap its value."""
    def done(self, value: T) -> Traced[T]: ...
```

Why not the alternatives: a bare tuple return is untyped and ugly under nesting; a recorder passed *in*
invites a caller-injected recorder; a **contextvar is an ambient seam through which a narrator could be
injected into `core/`**, and `test_record_not_generated` must not depend on nobody exploiting it.

Determinism requirements: decision order is **code order** (append, splice on `absorb`, never sort); one
shared `fmt_num()` with fixed precision; no timestamps inside `because`.

### 2.6 The record

```python
@dataclass(frozen=True, slots=True)
class ReadingRecord:
    path: str; sha256: str
    statistic_column: str; unit_column: str
    unit_kind: str; unit_detection_because: tuple[str, ...]
    rows_in: int; rows_dropped: int
    drop_reasons: tuple[tuple[str, int], ...]        # sorted pairs, never a dict
    entities_outside_frame: tuple[str, ...]
    entities_not_in_reference: tuple[str, ...]
    frames: tuple[FrameRef, ...]
    clustering_mode: str

@dataclass(frozen=True, slots=True)
class InterpretationRecord:
    unit: str; task_kind: TaskKind
    reading: ReadingRecord
    decisions: tuple[Decision, ...]
    claims_used: tuple[tuple[str, str], ...]         # (claim_id, verdict)
    undecidable: tuple[str, ...]
    spec_version: str
    prose: str | None = None                          # generated layer, marked as such
```

### 2.7 Claim and Conditions

```python
class Authority(IntEnum):
    LITERATURE = 1; CURATED = 2; LAB = 3; MEASUREMENT = 4

class Provenance(StrEnum):
    STATED = "stated"; READ = "read"; INFERRED = "inferred"; UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class Field:
    """A condition value plus HOW it was obtained. Intake writes these (ARCHITECTURE §6.2)."""
    value: str | None
    provenance: Provenance

@dataclass(frozen=True, slots=True)
class Conditions:
    assay: Field; modality: Field; tissue: Field        # tissue as an ontology id
    method: Field; segmentation: Field; fixation: Field
    frames: tuple[FrameRef, ...]                        # replaces panel_hash + panel_genes
    # disease state and its qualifiers: declared-unknown OK, never guessed, default UNKNOWN
    disease_state: Field; disease_subtype: Field; disease_stage: Field
    disease_treatment: Field                            # naive | treated, else UNKNOWN

@dataclass(frozen=True, slots=True)
class Claim:
    id: str
    subject: Subject
    predicate: str                   # REGISTERED: "is_cell_type" | "marks" | "means_here" | ...
    object: str | None
    authority: Authority
    source: str                      # URI: "pmid:…" | "doi:…" | "note:…" | "panel:…" | "file:sha256:…"
    conditions: Conditions
    depends_on: tuple[Entity, ...] = ()
    scope: str = "study"             # sample:<id> | dataset:<id> | study | lab
    status: str = "firm"             # firm | tentative — orthogonal to authority, keep it
    valid: str = "active"
    superseded_by: str | None = None
    created: str = ""; author: str | None = None
    tension: tuple[str, ...] = ()
    licence: str | None = None
    spans: tuple[str, ...] = ()
    spec_version: str = "0.1"
```

`Claim.source` is a URI resolved to a `SourceRecord` in a reference store — claims do not carry
abstracts. That is also what makes `licence` meaningful.

**Invariant:** a `LAB` claim with no frame reference fails validation. Without formation conditions it
cannot be checked and must not be storable.

---

## 3. Contracts

```python
class PlatformAdapter(Protocol):
    name: str
    def frames(self, store) -> tuple[Frame, ...]:
        """Xenium: var_names minus negative controls and custom add-ons. NEVER a constant."""
    def unit_labels(self, store) -> dict[str, str]: ...
    def samples(self, store) -> list[str]: ...
    def clustering_mode(self, store) -> Literal["joint", "independent", "integrated"]:
        """MUST be explicit. Raises UndeclaredClustering. Never guesses."""
    def available_views(self, store) -> frozenset[str]:
        """Derived from elements present. No points -> no density. Never fake."""
    def coordinate_system(self, store) -> tuple[str, float]:   # (name, unit -> µm)
    def standing_conditions(self, store) -> Conditions: ...
    def validate(self, store) -> list[str]: ...


class MethodModule(Protocol):
    """A machine-readable declaration of what a method's paper says it measures
    and does not measure."""
    name: str
    workflow: str                         # which interpretive question it answers
    statistic_name: str
    measures: str
    does_not_measure: str                 # MANDATORY, rendered in the UI
    assumptions: tuple[str, ...]          # MANDATORY, from the paper
    rival_space: Literal["reference", "data", "none"]
    citation: str                         # the method's own paper

    def detect_unit(self, table) -> UnitSpec:
        """Read unit kind and ids FROM THE RESULT SCHEMA.
        Raises UnrecognisedResult on an unknown shape. Never guesses."""
    def load(self, path, adapter, store) -> dict[str, NormalisedEvidence]: ...
    def absence_reads_as(self, subject, measured) -> str: ...
    def rubric(self, ev) -> Traced[str]: ...      # band, assignment only
    def traversals(self) -> tuple[tuple[str, str], ...]: ...


class Workflow(Protocol):
    """Owns the interpretive question AND the review surface."""
    name: str                             # "marker-interpretation"
    task_kind: TaskKind
    question: str                         # the biologist-facing question, one sentence
    skill_path: str                       # skills/<name>/SKILL.md
    def interpret(self, ev, reference) -> Traced[Assignment | Finding]: ...
    def rival_test(self, ev, alternative, reference) -> Traced[RivalResult]: ...
    def checks(self) -> tuple[Check, ...]: ...     # e.g. the DE workflow's four
    def uispec(self) -> UISpec: ...                # ARCHITECTURE §13


class ReferenceSource(Protocol):
    name: str; release: str; licence: str; redistributable: bool
    def conditions(self) -> Conditions:
        """The conditions under which THIS SOURCE's edges were established."""
    def nodes(self) -> Iterator[Node]: ...
    def edges(self) -> Iterator[Edge]: ...
```

**Invariant (`test_module_rubrics_differ`):** a spatial-localisation rubric and a mean-expression rubric
produce different bands on a constructed case with high fold-change and low spatial localisation. Write
this test *before* the DE method.

**Invariant (`test_every_method_declares_limits`):** `does_not_measure` and `assumptions` are non-empty
for every registered method.

---

## 4. The UISpec contract

```python
@dataclass(frozen=True, slots=True)
class Conditionality:
    """R4: no conditionality record, no plot. Enforced at construction."""
    text: str                       # "280 of ~20,000 genes tested, panel-scoped"
    frames: tuple[FrameRef, ...]

@dataclass(frozen=True, slots=True)
class Column:
    key: str; label: str
    source: str                     # MUST resolve to a field on EvidenceItem / FindingItem
    hatch_when_unmeasured: bool = True

@dataclass(frozen=True, slots=True)
class PlotSpec:
    kind: str                       # "hexbin" | "cell_map" | "umap" | "volcano" | "coverage_bar" | ...
    fields: tuple[str, ...]         # MUST exist in NormalisedEvidence
    conditionality: Conditionality  # mandatory, non-empty
    pin_roles: tuple[str, ...] = () # which subject roles drive this view

@dataclass(frozen=True, slots=True)
class UISpec:
    rail_label: str                 # "Clusters" | "Units" | "Contrasts"
    header: Literal["call", "counts"]     # assignment vs the other two — structural
    table_columns: tuple[Column, ...]
    primary_plot: PlotSpec
    detail_plots: tuple[PlotSpec, ...]
    overview_plot: PlotSpec | None
    pin_kind: str                   # a shape name; "pin" generalises to a pinned SUBJECT
```

**Invariants:** every `PlotSpec.fields` entry resolves to a field already in `NormalisedEvidence` (plots
render, they do not derive); every `PlotSpec` has non-empty conditionality; `header == "counts"` for
every non-assignment workflow; an unregistered shape falls back to a default render profile rather than
crashing.

---

## 5. Claims: evaluability and transfer

```python
# claims/evaluability.py — applies to EVERY claim class
def evaluability(claim: Claim, world: MeasuredWorld) -> Traced[Evaluability]:
    """Are the entities this claim depends on measured here?
    Pure. No IO, no clock. This is the panel-absence rule generalised."""

# claims/transfer.py — applies ONLY to manifest-sourced conditions
def transfer_check(claim: Claim, context: Conditions) -> Traced[TransferResult]:
    """LAB, MEASUREMENT, and dataset-scoped CURATED claims only.
    Database-curated and literature claims get a provenance stamp, not a verdict."""
```

```python
@dataclass(frozen=True, slots=True)
class Delta:
    field: str; claim_value: str | None; context_value: str | None; severity: str

@dataclass(frozen=True, slots=True)
class TransferResult:
    action: Literal["return", "return_with_conditions", "return_blocked_with_reason"]
    deltas: tuple[Delta, ...]
```

Order: **scope gate first** (a filter, not a downgrade), then evaluability, then — for eligible classes
only — the condition deltas. `UNKNOWN`-provenance fields never silently apply; they produce a
*cannot be settled* delta naming the missing condition.

**Report deltas, never a verdict distribution.** §7.4 of the architecture measured why.

**Invariants:** `test_transfer_pure`; `test_blocked_is_surfaced` (a blocked claim still appears with its
reason); `test_offpanel_never_penalises`; `test_scope_gate_before_return`;
`test_unknown_condition_never_silently_applies`; `test_no_verdict_for_database_curated`.

---

## 6. Core functions

```python
# core/interpret.py
def interpret(ev, workflow, reference) -> Traced[Assignment | Finding]:
    """Dispatch on workflow.task_kind. ASSIGNMENT computes a band; the others compute
    per-item tiers only."""

# core/reconcile.py
DEPENDENCY = {("enrichment", "marker"): "derived",
              ("de", "marker"): "partly_independent",
              ("ihc", "*"): "independent"}

def reconcile(outputs: dict[str, Any]) -> Traced[ReconcileResult]:
    """Same question -> agree | disagree.
       Different question -> coherent | tension | not_comparable. NEVER agree.
       Joins on subject.key ACROSS task kinds."""

# core/export.py
PROFILES: dict[TaskKind, tuple[Column, ...]] = {...}   # a table, not a second dataclass
```

**Invariants:** `test_never_raises_confidence` (property test over random inputs: reconciled band ≤
single-method band, always); `test_different_question_never_agrees`; `test_derived_is_not_corroboration`;
`test_concordance_never_raises`; `test_concordance_requires_joint`;
`test_reconcile_joins_on_subject_key` — the test that proves the `Subject` abstraction paid for itself.

### 6.1 Corpus interchange: the build in the biologist's own client (ARCHITECTURE §6.5)

The primary corpus path. Panoscope exports a brief, the biologist runs it in any LLM client with
whatever literature tools it has, and the client writes one JSONL file in the format
`panoscope/corpus/format.schema.json` defines. Each line is one `kind`:

| kind | fields |
| --- | --- |
| `header` (first line, once) | `format_version`, `study_id`, `brief_hash` (from the brief), `client`, `provider`, `model`, `run_date`, `operator_note` |
| `paper` | `id` (`pmid:` / `pmcid:` / `doi:`), `title`, `channel` (why screened), `disposition` (the ledger vocabulary) |
| `statement` | `paper`, `gene`, `cell_type_as_written`, `cell_type` (frame label or `""`), `tissue_as_written`, `disease_as_written`, `predicate` (the P2 relations), `comparator`, `span` (verbatim), `section` |
| `query` | `query` (exact string), `source`, `gene`, `cell_type`, `total`, `retrieved`, `status` (`OK` / `TRUNCATED` / `FAILED`) |

`panoscope corpus import --study <id> <file>` (`corpus/exchange.py`) **trusts nothing the client
computed**. A schema error anywhere refuses the file with line-numbered errors, and nothing is
written. Every id is resolved by code against Europe PMC, and an id that does not resolve is
rejected, never kept, and one code finds retracted is closed `RETRACTED` with its statements
rejected, whatever the client's disposition. Text is fetched by code. Each span is checked verbatim with the in-app
build's verifier (`read.verified`), and a mismatch is rejected and listed. Rung, modality,
species and stance are recomputed, so any client `rung`, `stance`, `species` or `grade` is
ignored. A client `OK` with `retrieved < total` reads `TRUNCATED`, and a key with an incomplete
log reads "incomplete search".

A `brief_hash` that is not the study's current one is warned and recorded in the query log. The
new `CorpusVersion` is stamped `<model> (<provider>) in <client>, built in user client, run
<date>` as the user stated it. Every version is also copied to `versions/<hash>.sqlite` beside
the store, so earlier versions are kept and never overwritten (A12). The import prints a receipt
listing accepted and rejected statements with reasons, unresolved ids and the rung distribution.
The in-app API build (`panoscope corpus build`) remains the secondary path.

---

## 7. Migration

`main` stays frozen and demo-able. All work on `v2`.

### Phase A — safety net, before touching any source

1. CI: add `v2` to triggers; add a non-blocking `out-of-tree` job that installs the package and runs
   pytest from `/tmp` with the repo moved aside. It fails today — that is the point. `pytest.ini`'s
   `pythonpath = .` currently hides every "works because cwd is the repo root" bug.
2. **Track the small second study.** It is currently untracked and it is the CI fixture, the second
   study, and what makes the refactor testable.
3. `tests/__init__.py` and a `conftest.py` with an autouse fixture clearing **all eleven process-global
   `lru_cache`s** (all `maxsize=1`). Invisible today because every test uses one study; the instant a
   test opens the second, every later test silently gets its data and the failure looks like a refactor
   bug.
4. **The golden-output test.** Hash a normalised projection of the verdict CSV, the per-unit JSONs, the
   holistic review and the calibration table from a full pipeline run into a temp root. **The single
   most valuable artifact in the plan** — it is what makes refactoring 23k lines safe, and it doubles as
   the reproducibility claim and the basis of the model-invariance result.
5. Import-surface test (`pkgutil.walk_packages`) and an `xfail(strict=True)` no-import-time-IO test.

### Phase B — the move, as two commits, never one

`git mv` only (git records 100% renames, `--follow` survives), then an anchored import rewrite of ~120
lines. Keep the **pushed tip** green rather than every commit.

**Do not use `sys.modules` compat shims.** In a codebase whose central problem is module-level
singletons, a shim can produce two module objects with two divergent cluster-key dicts. There are no
external consumers — no release, no tags — so rewrite outright.

`skills/` and `assets/*.png` move **into** the package: they are read at runtime via a project-root
constant and would vanish under a site-packages install.

Then: `pyproject.toml`, editable install, delete `pytest.ini`; delete ~515 lines of dead UI (two modules
with zero call sites); delete the duplicated canonical-marker maps and the import-time assertion that
validates the unused copy.

### Phase C — dissolve the config singleton

A frozen, hashable `Study` plus a `contextvars.ContextVar`, and — the trick that makes this incremental
— **PEP 562 module `__getattr__` on `config`**, so the dataset-derived constants become lazy at ~215
call sites *without editing them*. Only ~35 need hand-editing: six `from config import X` forms, ~25
`dataset: str = cfg.DATASET_ID` default args, and three duplicate path-resolution blocks.

Keying `@lru_cache` on the frozen `Study` fixes cross-study cache staleness for free.

> **Acceptance: the frozen-path workaround in the pipeline can be deleted.** It exists only to route
> around import-frozen paths, and both confirmed manifest bugs vanish once the readers resolve at call
> time. If it cannot be deleted, Phase C is not done.

### Phase D — hardcode purge, then the plurality gate

Grounding-integrity leaks first, because they are the reviewer-facing ones:

| Leak | Effect |
| --- | --- |
| `canonical_markers()` / `offpanel_canonical()` fall back to the **breast** map with no dataset guard | any study naming a cluster "Tumor" inherits breast markers, which feed driver selection and the band |
| the demo's pDC refinement is gated only on "unit 8 exists and is called Dendritic" | a different tissue with such a unit emits a breast-specific refinement |

Then: the hardcoded 9-unit check; direct cluster-key subscripts; hardcoded tissue/platform constants;
the nine-colour palette dict; the hardcoded UI context chip; and the project-root and state paths that
break under site-packages.

> **Acceptance gate** (a CI assertion, not a milestone):
> ```
> open two studies in one process
>   assert unit counts differ and both are correct
>   assert the readers return distinct objects per study     # cache is per-study
>   assert the second study emits no refinement from the first
>   assert canonical markers for a shared label are empty in the second study
>   assert the manifest tissue differs
> ```

### Phase E onward

Contracts (§2) → platform adapter + tidy reader → methods with task kinds → **DE workflow and the
method-agnostic demonstration** → claims → reference → corpus → reconciliation → review surfaces →
hardening → evaluation.

---

## 8. Packaging

```toml
[project]
name = "panoscope"
requires-python = ">=3.11"
dependencies = ["pandas>=2,<3", "pyarrow>=14", "numpy>=1.26,<3", "pyyaml>=6", "platformdirs>=4"]

[project.optional-dependencies]
app     = ["streamlit>=1.59,<2", "plotly>=5.20,<7"]
agent   = ["anthropic>=0.40", "mcp>=1.2", "python-dotenv>=1"]
export  = ["python-docx>=1.1", "fpdf2>=2.7"]
spatial = ["spatialdata>=0.2", "zarr>=2.17"]
dev     = ["pytest>=8", "pytest-cov", "ruff", "mypy"]

[project.scripts]
panoscope = "panoscope.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["panoscope"]
[tool.hatch.build]
exclude = ["studies", "Raw_data_*", "jazzPanda output", "handoff", "context", "tests"]
```

**Upper-bound every dependency and ship a lockfile.** An unbounded UI-framework pin is a reproducibility
hazard when the UI keys off undocumented DOM internals.

**The study store never ships in the wheel** — hatchling exclude plus a CI assert that the wheel is under
5 MB. The demo study becomes a **Zenodo artifact with its own DOI**, fetched and checksum-verified by
`panoscope fetch-demo`: code DOI and data DOI separately citable. Only the small fixture study is
bundled, so `pip install panoscope && pytest` works with no download.

Two free wins that belong here, not in hardening: the two large per-cell frames convert **CSV → parquet**
(the reader already prefers parquet, so it is a data-only change), and the ~840 per-gene density files
collapse into **three partitioned parquets** with a pushed-down filter — one function changes and the
index file disappears.

---

## 9. Tests

| Group | Defends |
| --- | --- |
| `test_subject_*` | canonicality, registered shapes, key injectivity, label consistency |
| `test_measured_*` | absence of unmeasured never moves a band; unmeasured carries no number; composite meet; off-panel partner blocks an interaction; missing cell type blocks a pair; too-few-replicates underpowers the *unit*; **measured absence is usable evidence** |
| `test_platform_*` | frame derivation, view availability, undeclared clustering raises |
| `test_method_*` | rubric divergence, unit detection, `does_not_measure` present |
| `test_transfer_*` | purity, blocked-is-surfaced, off-panel never penalises, scope gate first, unknown never applies |
| `test_claims_*` | authority not merged, supersession ownership |
| `test_reference_*` | determinism, no bundled restricted data, missing node reads as unknown |
| `test_corpus_*` | relevance floor, alias retrieval, staleness surfaced |
| `test_reconcile_*` | never raises confidence (property), different question never agrees, joins on subject key |
| `test_ui_*` | plot fields exist, conditionality present, counts-header on non-assignment, unknown shape degrades |
| `test_record_*` | deterministic bytes, not generated, every band has decisions, prose gated |
| `test_contracts_*` | no dict fields; every core function returns `Traced`; no core function accepts a `Trace`; every decision rule registered |
| `test_grounding_*` | cited must be retrieved |
| `test_budgets` | the performance table |

Plus **contract tests parameterised over every registered workflow, method, adapter and source**, and a
**golden-output test** snapshotting the demo report.

`test_record_not_generated` needs two forms: an AST walk asserting no `core/` module imports a
generation client, **and** a subprocess check that the client is absent from `sys.modules` after
importing `core` — the AST walk cannot see an indirect import.

`evals/` is separate from `tests/` because it needs data and produces numbers: seeded errors, the
panel-masking benchmark, transfer accuracy, sycophancy, generator swap, and a turn set with known-correct
behaviour including refusals.

CI jobs: `core` (no extras, no network), `spatial`, `reference`, `licence` (greps for bundled data),
`lint`, `wheel-size`, `out-of-tree`.

---

## 10. Migration map

| Existing | Action |
| --- | --- |
| `agent/config.py` | strip dataset constants; PEP 562 `__getattr__`; keep the pure rubric constants |
| `agent/data.py` | **Row superseded — do not do this split.** The unit-count assertion is dropped and per-study caching is done, via a `_per_study` decorator rather than a `Study` argument. The `data/spatial.py` this row names already shipped as `platforms/xenium.py`; the archive spec's wording (`data/spatial.py` **new** + `data/legacy.py` **kept**) shows it was always a NEW reader beside a dying one, not a cut through 291 lines. And `pipeline/run.py` calls `load_panel()` and `load_cells()` thirteen lines apart, so `agent/data.py` would have to survive as a pure re-export shim — the split would add an import hop at ~30 call sites and remove nothing. The real duplication is `_resolved`, copied into `ui/data_access.py`, and the `viz/*` path literals known independently in four places. |
| `agent/types.py` | → `types/*`; `MarkerRow` deleted (dead); `OffPanelNote` deleted — it becomes an `EvidenceItem` with `NEVER_MEASURED`, which removes the type split the new design exists to avoid |
| `agent/verdict.py` | rubric → `methods/jazzpanda.py`; band → `core/verdict.py`; `is_canonical` → a **`CURATED` claim** (~60 seed claims, the largest single chunk) |
| `agent/enrichment.py` | → `methods/jazzpanda_cgst.py` + the enrichment workflow; **`ClusterEnrichment.confidence` is deleted** (see below) |
| `agent/discriminate.py` | → `core/rival.py`; the four buckets become filters on `Measured.state` |
| `agent/skeptic.py`, `agent/holistic.py` | read `NormalisedEvidence`; holistic's unit becomes `study:<id>` |
| `agent/memory.py`, `agent/user_memory.py` | → `claims/store.py`; the note schema → `Claim`; **keep `status`** and **keep the thin-literature signal** |
| `agent/grounding_check.py` | extend the resolver for claim and edge ids; gate `record.prose`; **wire the note-scope pass, which is inert in production today** |
| `agent/loop.py` | reads the record instead of re-deriving rationale; skill path via `importlib.resources` |
| `skills/jazzpanda-markers/` | → `skills/marker-interpretation/`, jazzPanda one registered method |
| `skills/geneset-enrichment/` | → `skills/enrichment-interpretation/` |
| `ui/data_access.py` | becomes the `Study`-aware reader; it is already the seam — ~45 functions, imported by 16 of 22 UI modules |
| `ui/holistic.py`, `ui/lab_knowledge.py` | **delete** — zero call sites |
| `data/datasets/` | → `studies/` |

### The one behaviour change with a visible surface

`ClusterEnrichment` currently carries a unit-level confidence, persisted as two columns of a committed
CSV and rendered as a pill in the Pathways header. The no-band invariant deletes all of it. **This is a
breaking change to a committed artifact and a visible surface — sequence it deliberately, do not
discover it at method-refactor time.** The replacement is strictly more informative:

> `7 supported · 3 suggestive · 12 untestable (panel coverage)`

`cluster_verify` **can** stay: "a human should look" is well-defined without a label space.

### On-disk compatibility

Never mutate the committed per-unit JSONs. v3 records write to `interp/records/<task_kind>/<unit>.json`;
the v1 files become a **generated compatibility view** through one adapter, so the UI and the store are
untouched during migration. The gate is unusually strong because inputs are hashed: rebuild each record
from the same hashed inputs, project it back, and compare field by field.

Byte-identity is not achievable for deliberately deleted fields — a vestigial percentile, `is_canonical`
(now a claim), and prose caveats (now flags). Those live in an explicit reviewed `V1_DROPPED` allowlist;
**adding to it must be a deliberate diff, not a green test.** The verdict CSV stays frozen at 11 columns
in exact order.

---

## 11. Order and parallelism

The implementation target is roughly one month with parallel agent teams. Phases A–D are mostly
**serial** — each changes the ground the next stands on — but everything after the plurality gate
parallelises cleanly along the registry boundaries, which is the practical payoff of §4.

| Stream | Depends on | Can run parallel with |
| --- | --- | --- |
| Contracts (§2) | Phase C | — (blocks everything) |
| Platform adapter + tidy reader | contracts | methods, claims |
| Methods: jazzPanda, DE, enrichment | contracts | adapter, claims, reference |
| Claims: schema, evaluability, transfer | contracts | methods, reference |
| Reference sources + view | contracts | methods, claims |
| Corpus | reference | reconciliation |
| Reconciliation, adjacency, concordance | methods | corpus |
| Review surfaces | contracts + UISpec | everything above |
| Benchmark + eval harness | contracts | everything above |

Two rules for parallel work: every stream lands behind the **golden-output test**, and no stream may
edit `types/` after it freezes — a contract change is a separate, reviewed commit that re-runs every
stream's tests.
